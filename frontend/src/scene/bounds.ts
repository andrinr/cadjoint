/**
 * Bounding boxes for culling: `cadjoint.sdf._lowering.node_bounds`, in the
 * float32 arithmetic JAX does it in. Every operation rounds to float32
 * (`Math.fround`), so the boxes agree with Python's to the rounding of the
 * few sums and transcendental functions XLA may order or approximate
 * differently. The boxes are buffer values, not module text, and a skip test
 * is exact with a margin, so that is the agreement they need.
 */
import type { SceneNode, Vec } from "./graph.ts";

const f = Math.fround;

export type Bounds = { center: Vec; half: Vec; scale: number };

type V3 = [number, number, number];

const v3 = (x: number, y: number, z: number): V3 => [f(x), f(y), f(z)];
const asV3 = (value: unknown): V3 => {
  const a = value as number[];
  return v3(a[0], a[1], a[2]);
};
const add = (a: Vec, b: Vec): V3 => v3(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
const sub = (a: Vec, b: Vec): V3 => v3(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const mul = (a: Vec, b: Vec): V3 => v3(a[0] * b[0], a[1] * b[1], a[2] * b[2]);
const muls = (a: Vec, s: number): V3 => v3(a[0] * s, a[1] * s, a[2] * s);
const min3 = (a: Vec, b: Vec): V3 => v3(Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.min(a[2], b[2]));
const max3 = (a: Vec, b: Vec): V3 => v3(Math.max(a[0], b[0]), Math.max(a[1], b[1]), Math.max(a[2], b[2]));
const abs3 = (a: Vec): V3 => v3(Math.abs(a[0]), Math.abs(a[1]), Math.abs(a[2]));
const sum3 = (a: Vec): number => f(f(a[0] + a[1]) + a[2]);
const dot3 = (a: Vec, b: Vec): number => sum3(mul(a, b));
const sqrt = (x: number): number => f(Math.sqrt(x));
const scalar = (value: unknown): number => f(value as number);

const ZERO: V3 = [0, 0, 0];

const low = (b: Bounds): V3 => sub(b.center, b.half);
const high = (b: Bounds): V3 => add(b.center, b.half);

function hull(boxes: (Bounds | null)[]): Bounds | null {
  if (!boxes.length || boxes.some((b) => b === null)) return null;
  const all = boxes as Bounds[];
  let lo = low(all[0]);
  let hi = high(all[0]);
  let scale = all[0].scale;
  for (const box of all.slice(1)) {
    lo = min3(lo, low(box));
    hi = max3(hi, high(box));
    scale = Math.min(scale, box.scale);
  }
  return { center: muls(add(lo, hi), 0.5), half: muls(sub(hi, lo), 0.5), scale };
}

function grown(bounds: Bounds | null, amount: number): Bounds | null {
  if (bounds === null) return null;
  const grow = bounds.scale === 1 ? amount : f(amount / bounds.scale);
  return { center: bounds.center, half: add(bounds.half, [grow, grow, grow]), scale: bounds.scale };
}

function cylinderHull(bounds: Bounds, origin: Vec, axisIn: Vec): Bounds {
  const norm = sqrt(Math.max(dot3(axisIn, axisIn), f(1e-12)));
  const unit = v3(axisIn[0] / norm, axisIn[1] / norm, axisIn[2] / norm);
  const radiusSphere = sqrt(dot3(bounds.half, bounds.half));
  const relative = sub(bounds.center, origin);
  const height = dot3(relative, unit);
  const radial = sub(relative, muls(unit, height));
  const radius = f(sqrt(dot3(radial, radial)) + radiusSphere);
  const lowH = f(height - radiusSphere);
  const highH = f(height + radiusSphere);
  const across = muls(
    v3(...(unit.map((a) => sqrt(Math.max(f(1 - f(a * a)), 0))) as V3)),
    radius,
  );
  const alongLow = min3(muls(unit, lowH), muls(unit, highH));
  const alongHigh = max3(muls(unit, lowH), muls(unit, highH));
  const lo = sub(add(origin, alongLow), across);
  const hi = add(add(origin, alongHigh), across);
  return { center: muls(add(lo, hi), 0.5), half: muls(sub(hi, lo), 0.5), scale: bounds.scale };
}

const applyRows = (rows: V3[], vector: Vec): V3 => v3(dot3(rows[0], vector), dot3(rows[1], vector), dot3(rows[2], vector));

/** `_rotation_rows`: Rodrigues about `axis` by `angle`, normalised with a 1e-24 floor. */
export function rotationRows(axisIn: Vec, angleIn: number): V3[] {
  const a = asV3(axisIn);
  const norm = sqrt(Math.max(dot3(a, a), f(1e-24)));
  return rodrigues(v3(a[0] / norm, a[1] / norm, a[2] / norm), angleIn);
}

/** `Rotate._rotation_matrix`: the same rotation, normalised by `jnp.linalg.norm`. */
export function rotationMatrix(axisIn: Vec, angleIn: number): V3[] {
  const a = asV3(axisIn);
  const norm = sqrt(dot3(a, a));
  return rodrigues(v3(a[0] / norm, a[1] / norm, a[2] / norm), angleIn);
}

function rodrigues(axis: V3, angleIn: number): V3[] {
  const angle = f(angleIn);
  const c = f(Math.cos(angle));
  const s = f(Math.sin(angle));
  const t = f(1 - c);
  const [x, y, z] = axis;
  const m = (a: number, b: number) => f(a * b);
  return [
    v3(f(m(m(t, x), x) + c), f(m(m(t, x), y) - m(s, z)), f(m(m(t, x), z) + m(s, y))),
    v3(f(m(m(t, x), y) + m(s, z)), f(m(m(t, y), y) + c), f(m(m(t, y), z) - m(s, x))),
    v3(f(m(m(t, x), z) - m(s, y)), f(m(m(t, y), z) + m(s, x)), f(m(m(t, z), z) + c)),
  ];
}

function reflectionRows(normalIn: Vec): V3[] {
  const n0 = asV3(normalIn);
  const norm = sqrt(Math.max(dot3(n0, n0), f(1e-24)));
  const n = v3(n0[0] / norm, n0[1] / norm, n0[2] / norm);
  const basis: V3[] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  return basis.map((row, i) => sub(row, muls(n, f(2 * n[i]))));
}

function transformed(bounds: Bounds | null, rows: V3[], offset: Vec): Bounds | null {
  if (bounds === null) return null;
  return {
    center: add(applyRows(rows, bounds.center), offset),
    half: applyRows(rows.map(abs3), bounds.half),
    scale: bounds.scale,
  };
}

const vertices = (node: SceneNode, key: string): V3[] =>
  ((node.params[key] ?? []) as Vec[]).map((v) => v3(v[0], v[1], 0));

const declared = (node: SceneNode, name: string): boolean =>
  name in node.params &&
  !(node.implicit ?? (node.params[name] === 0 ? [name] : [])).includes(name);

/**
 * The box `node`'s solid lies in, given its children's, or null for no
 * promise. A version 1 graph does not say whether a zero draft or twist was
 * declared; a zero one is read as not declared, which is the usual case.
 */
export function nodeBounds(node: SceneNode, children: (Bounds | null)[]): Bounds | null {
  const p = node.params;
  const leaf = (half: V3): Bounds => ({ center: ZERO, half, scale: 1 });
  switch (node.type) {
    case "sphere": {
      const r = Math.abs(scalar(p.radius));
      return leaf(v3(r, r, r));
    }
    case "box":
      return leaf(abs3(asV3(p.size)));
    case "round_box": {
      const r = Math.abs(scalar(p.radius));
      return leaf(add(abs3(asV3(p.size)), [r, r, r]));
    }
    case "cylinder": {
      const r = Math.abs(scalar(p.radius));
      const h = Math.abs(scalar(p.height));
      return leaf(v3(r, r, h));
    }
    case "capsule": {
      const r = Math.abs(scalar(p.radius));
      const h = Math.abs(scalar(p.height));
      return leaf(v3(r, r, h + r));
    }
    case "torus": {
      const big = Math.abs(scalar(p.major_radius));
      const small = Math.abs(scalar(p.minor_radius));
      return leaf(v3(big + small, big + small, small));
    }
    case "extruded_polygon": {
      if (declared(node, "draft")) return null;
      const verts = vertices(node, "vertices");
      const halfDepth = f(Math.abs(scalar(p.depth)) * 0.5);
      if (declared(node, "twist")) {
        const radius = sqrt(verts.map((v) => f(f(v[0] * v[0]) + f(v[1] * v[1]))).reduce((a, b) => Math.max(a, b)));
        return leaf(v3(radius, radius, halfDepth));
      }
      const lo = verts.reduce(min3);
      const hi = verts.reduce(max3);
      const centre = muls(add(lo, hi), 0.5);
      const half = muls(sub(hi, lo), 0.5);
      return { center: v3(centre[0], centre[1], 0), half: v3(half[0], half[1], halfDepth), scale: 1 };
    }
    case "revolved_polygon": {
      const verts = vertices(node, "vertices");
      const radius = Math.max(f(Math.max(...verts.map((v) => v[0])) + scalar(p.offset)), 0);
      const lowY = Math.min(...verts.map((v) => v[1]));
      const highY = Math.max(...verts.map((v) => v[1]));
      return {
        center: v3(0, f(f(lowY + highY) * 0.5), 0),
        half: v3(radius, f(f(highY - lowY) * 0.5), radius),
        scale: 1,
      };
    }
    case "lofted_polygon": {
      const verts = [...vertices(node, "vertices"), ...vertices(node, "vertices_b")];
      const lo = verts.reduce(min3);
      const hi = verts.reduce(max3);
      const centre = muls(add(lo, hi), 0.5);
      const half = muls(sub(hi, lo), 0.5);
      return {
        center: v3(centre[0], centre[1], 0),
        half: v3(half[0], half[1], f(Math.abs(scalar(p.height)) * 0.5)),
        scale: 1,
      };
    }
    case "union":
      return grown(hull(children), Math.max(f(scalar(p.smoothness ?? 0.1) * 4), f(1e-10)));
    case "difference":
    case "intersection":
      return children.length ? children[0] : null;
    case "xor":
      return hull(children);
  }
  if (!children.length || children[0] === null) return null;
  const child = children[0];
  switch (node.type) {
    case "translate":
      return { center: add(child.center, asV3(p.offset)), half: child.half, scale: child.scale };
    case "rotate":
      return transformed(child, rotationRows(p.axis as Vec, p.angle as number), ZERO);
    case "scale": {
      const scale = asV3(p.scale);
      const magnitude = abs3(scale);
      const largest = Math.max(magnitude[0], magnitude[1], magnitude[2]);
      const smallest = Math.min(magnitude[0], magnitude[1], magnitude[2]);
      return {
        center: mul(child.center, scale),
        half: mul(child.half, magnitude),
        scale: f(child.scale * f(smallest / largest)),
      };
    }
    case "mirror": {
      const origin = asV3(p.origin);
      const rows = reflectionRows(p.normal as Vec);
      return transformed(child, rows, sub(origin, applyRows(rows, origin)));
    }
    case "shell":
      return grown(child, f(Math.abs(scalar(p.thickness)) * 0.5));
    case "offset":
      return grown(child, Math.max(scalar(p.distance), 0));
    case "twist":
      return cylinderHull(child, ZERO, asV3(p.axis));
    case "linear_pattern": {
      const count = Math.trunc(p.count as number);
      if (count <= 1) return child;
      const direction = asV3(p.direction);
      const norm = sqrt(Math.max(dot3(direction, direction), f(1e-24)));
      const axis = v3(direction[0] / norm, direction[1] / norm, direction[2] / norm);
      const shift = muls(axis, f(scalar(p.spacing) * (count - 1)));
      const lo = min3(low(child), add(low(child), shift));
      const hi = max3(high(child), add(high(child), shift));
      return { center: muls(add(lo, hi), 0.5), half: muls(sub(hi, lo), 0.5), scale: child.scale };
    }
    case "polar_pattern":
      if (Math.trunc(p.count as number) <= 1) return child;
      return cylinderHull(child, asV3(p.origin), asV3(p.direction));
  }
  return null;
}

/** The root's box, every child's computed first; null when the scene makes no promise. */
export function graphBounds(node: SceneNode): Bounds | null {
  try {
    return nodeBounds(node, node.children.map(graphBounds));
  } catch {
    return null;
  }
}
