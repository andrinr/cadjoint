/**
 * WGSL for a `cadjoint-scene` graph: `cadjoint.backends.wgsl.direct`, in
 * TypeScript, so the browser can build its own shader from the graph.
 *
 * The Python module stays the reference. For the same scene this one has to
 * write the same module text, the same parameter slots and the same culling
 * dependencies; `tests/backends/test_wgsl_frontend.py` holds the two to that
 * and runs both on a device. Kernel bodies, slot order, naming and number
 * formatting therefore follow the Python line for line, and a change to one
 * is a change to both.
 *
 * Node types the Python module has no kernel for (round_box, plane, twist,
 * xor and scale) raise `UnsupportedNode` here too. The Python worker then
 * falls back to its traced backend, which has no browser counterpart.
 */
import { nodeBounds, rotationMatrix, type Bounds } from "./bounds.ts";
import { DEFAULT_MATERIAL, sceneRoot, type SceneGraph, type SceneNode, type Vec } from "./graph.ts";
import { formatG9 } from "./pyformat.ts";

/** `codegen.DEFAULT_PARAMETER_GROUP` / `_BINDING`. */
export const DEFAULT_PARAMETER_GROUP = 3;
export const DEFAULT_PARAMETER_BINDING = 0;
/** `codegen.PARAMETER_ADDRESS_SPACE`. */
export const PARAMETER_ADDRESS_SPACE = "storage, read";
/** `codegen.PARAMETER_SLOT_BYTES`. */
export const PARAMETER_SLOT_BYTES = 16;
/** `codegen.RESERVED_PARAMETER_SLOTS`: the NaN slot and the cull margin. */
export const RESERVED_PARAMETER_SLOTS = 2;
/** `_culling.CULL_MARGIN`. */
export const CULL_MARGIN = 1e-4;

export class UnsupportedNode extends Error {}

/** One uniform slot: `(name, components, value)` as `DirectProgram.parameters` has it. */
export type Slot = [name: string, components: number, value: number[]];

export type DirectProgram = {
  wgsl: string;
  /** Profile vertices of the literal form, `[x, y]` each; empty in the uniform form. */
  vertices: [number, number][];
  parameters: Slot[];
  /** Slots holding culling boxes rather than design parameters. */
  derived: string[];
  /** Per derived slot, the parameter slots it was computed from. */
  depends: Record<string, string[]>;
};

type Kernel = { params: string[]; arity: Record<string, number>; body: string };

const PRIMITIVES: Record<string, [className: string, kernel: Kernel]> = {
  sphere: ["Sphere", { params: ["radius"], arity: { radius: 1 }, body: "    return length(p) - radius;" }],
  box: [
    "Box",
    {
      params: ["size"],
      arity: { size: 3 },
      body:
        "    let q = abs(p) - size;\n" +
        "    let max_q = max(q.x, max(q.y, q.z));\n" +
        "    let clamped = max(q, vec3<f32>(0.0));\n" +
        "    let squared = dot(clamped, clamped);\n" +
        "    if (max_q <= 0.0) { return max_q; }\n" +
        "    return sqrt(squared);",
    },
  ],
  torus: [
    "Torus",
    {
      params: ["major_radius", "minor_radius"],
      arity: { major_radius: 1, minor_radius: 1 },
      body:
        "    let q_xy = length(p.xy) - major_radius;\n" +
        "    return length(vec2<f32>(q_xy, p.z)) - minor_radius;",
    },
  ],
  cylinder: [
    "Cylinder",
    {
      params: ["radius", "height"],
      arity: { radius: 1, height: 1 },
      body:
        "    let d_xy = sqrt(p.x * p.x + p.y * p.y + 1e-20) - radius;\n" +
        "    let d_z = abs(p.z) - height;\n" +
        "    let max_d = max(d_xy, d_z);\n" +
        "    if (max_d <= 0.0) { return max_d; }\n" +
        "    let a = max(d_xy, 0.0);\n" +
        "    let b = max(d_z, 0.0);\n" +
        "    return sqrt(a * a + b * b);",
    },
  ],
  capsule: [
    "Capsule",
    {
      params: ["radius", "height"],
      arity: { radius: 1, height: 1 },
      body:
        "    let z = clamp(p.z, -height, height);\n" +
        "    return length(vec3<f32>(p.x, p.y, p.z - z)) - radius;",
    },
  ],
};

const PROFILE_BUFFER = "@group(1) @binding(0) var<storage, read> profile_vertices: array<vec2<f32>>;";

const PROFILE_FROM_UNIFORM = `fn profile_vertex(i: u32) -> vec2<f32> {
    return sdf_parameters.values[i].xy;
}`;

const PROFILE_FROM_BUFFER = `fn profile_vertex(i: u32) -> vec2<f32> {
    return profile_vertices[i];
}`;

const PROFILE_DISTANCE = `fn profile_distance(p: vec2<f32>, offset: u32, count: u32) -> f32 {
    var d = dot(p - profile_vertex(offset), p - profile_vertex(offset));
    var s = 1.0;
    // The previous vertex is carried rather than re-read: one buffer load per
    // edge instead of two, and no modulo. Edge for edge the same arithmetic.
    var vj = profile_vertex(offset + count - 1u);
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let vi = profile_vertex(offset + i);
        let e = vj - vi;
        let w = p - vi;
        let t = clamp(dot(w, e) / dot(e, e), 0.0, 1.0);
        let b = w - e * t;
        d = min(d, dot(b, b));
        let c1 = p.y >= vi.y;
        let c2 = p.y < vj.y;
        let c3 = (e.x * w.y) > (e.y * w.x);
        if ((c1 == c2) && (c2 == c3)) { s = -s; }
        vj = vi;
    }
    return s * sqrt(d);
}`;

const EXTRUDED = `fn prim_extrudedpolygon(
    p: vec3<f32>, offset: u32, count: u32, depth: f32, slope: f32, twist: f32
) -> f32 {
    // \`slope\` is tan(draft) and \`twist\` is radians over the full depth, both
    // folded by the caller; zero is the plain extrusion and both terms
    // vanish, matching the Python's static skips.
    let theta = twist * p.z / depth;
    let c = cos(theta);
    let s = sin(theta);
    let xy = vec2<f32>(p.x * c + p.y * s, p.y * c - p.x * s);
    let d2 = profile_distance(xy, offset, count) + slope * (p.z + depth * 0.5);
    let dz = abs(p.z) - depth * 0.5;
    let max_d = max(d2, dz);
    if (max_d <= 0.0) { return max_d; }
    let a = max(d2, 0.0);
    let b = max(dz, 0.0);
    return sqrt(a * a + b * b);
}`;

const PROFILE_DISTANCE_REVOLVED = `fn on_axis(x: f32, axis: f32) -> bool {
    return abs(x - axis) <= 1e-6 * (1.0 + abs(x) + abs(axis));
}

fn profile_distance_revolved(p: vec2<f32>, offset: u32, count: u32, axis: f32) -> f32 {
    var d = 1e30;
    var s = 1.0;
    // The previous vertex is carried rather than re-read: one buffer load per
    // edge instead of two, and no modulo. Edge for edge the same arithmetic.
    var vj = profile_vertex(offset + count - 1u);
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let vi = profile_vertex(offset + i);
        let e = vj - vi;
        let w = p - vi;
        if (!(on_axis(vi.x, axis) && on_axis(vj.x, axis))) {
            let t = clamp(dot(w, e) / dot(e, e), 0.0, 1.0);
            let b = w - e * t;
            d = min(d, dot(b, b));
        }
        let c1 = p.y >= vi.y;
        let c2 = p.y < vj.y;
        let c3 = (e.x * w.y) > (e.y * w.x);
        if ((c1 == c2) && (c2 == c3)) { s = -s; }
        vj = vi;
    }
    return s * sqrt(d);
}`;

const REVOLVED = `fn prim_revolvedpolygon(p: vec3<f32>, offset: u32, count: u32, radial: f32) -> f32 {
    let r = sqrt(p.x * p.x + p.z * p.z + 1e-20) - radial;
    return profile_distance_revolved(vec2<f32>(r, p.y), offset, count, -radial);
}`;

const ROTATE = `fn rotate_about(p: vec3<f32>, origin: vec3<f32>, axis: vec3<f32>, angle: f32) -> vec3<f32> {
    let v = p - origin;
    let c = cos(angle);
    let s = sin(angle);
    let parallel = axis * dot(v, axis);
    return origin + v * c + cross(axis, v) * s + parallel * (1.0 - c);
}`;

const PROFILE_LERPED = `fn profile_distance_lerped(
    p: vec2<f32>, a: u32, b: u32, count: u32, t: f32
) -> f32 {
    let first = mix(profile_vertex(a), profile_vertex(b), t);
    var d = dot(p - first, p - first);
    var s = 1.0;
    var vj = mix(profile_vertex(a + count - 1u), profile_vertex(b + count - 1u), t);
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let vi = mix(profile_vertex(a + i), profile_vertex(b + i), t);
        let e = vj - vi;
        let w = p - vi;
        let u = clamp(dot(w, e) / dot(e, e), 0.0, 1.0);
        let q = w - e * u;
        d = min(d, dot(q, q));
        let c1 = p.y >= vi.y;
        let c2 = p.y < vj.y;
        let c3 = (e.x * w.y) > (e.y * w.x);
        if ((c1 == c2) && (c2 == c3)) { s = -s; }
        vj = vi;
    }
    return s * sqrt(d);
}`;

const LOFTED = `fn prim_loftedpolygon(
    p: vec3<f32>, a: u32, b: u32, count: u32, height: f32
) -> f32 {
    let t = clamp(p.z / height + 0.5, 0.0, 1.0);
    let d2 = profile_distance_lerped(p.xy, a, b, count, t);
    let dz = abs(p.z) - height * 0.5;
    let max_d = max(d2, dz);
    if (max_d <= 0.0) { return max_d; }
    let ea = max(d2, 0.0);
    let eb = max(dz, 0.0);
    return sqrt(ea * ea + eb * eb);
}`;

const MATERIAL_ENTRY_POINTS = `fn material_base(p: vec3<f32>) -> vec4<f32> {
    let m = material_block_impl(p);
    return vec4<f32>(m[0].x, m[1].x, m[2].x, m[3].x);
}

fn material_optics(p: vec3<f32>) -> vec4<f32> {
    let m = material_block_impl(p);
    return vec4<f32>(m[0].y, m[1].y, m[2].y, m[3].y);
}`;

const CULL_BOX = `fn cull_box(p: vec3<f32>, center: vec3<f32>, half_size: vec3<f32>) -> f32 {
    let q = max(abs(p - center) - half_size, vec3<f32>(0.0));
    return sqrt(dot(q, q));
}`;

const MARGIN_TOKEN = "__CADJOINT_CULL_MARGIN__";

const SHARED: Record<string, string> = {
  profile_distance: PROFILE_DISTANCE,
  profile_distance_revolved: PROFILE_DISTANCE_REVOLVED,
  profile_vertex: PROFILE_FROM_BUFFER,
  profile_distance_lerped: PROFILE_LERPED,
  prim_loftedpolygon: LOFTED,
  prim_extrudedpolygon: EXTRUDED,
  prim_revolvedpolygon: REVOLVED,
  rotate_about: ROTATE,
  cull_box: CULL_BOX,
};
for (const [className, kernel] of Object.values(PRIMITIVES)) {
  const sym = className.toLowerCase();
  const args = kernel.params
    .map((parameter) => `${parameter}: ${kernel.arity[parameter] === 1 ? "f32" : "vec3<f32>"}`)
    .join(", ");
  SHARED[`prim_${sym}`] = `fn prim_${sym}(p: vec3<f32>, ${args}) -> f32 {\n${kernel.body}\n}`;
}

/** Translate's point map; Scale's has no kernel (see the module comment). */
const TRANSFORMS: Record<string, [string[], Record<string, number>, (p: string, b: Record<string, string>) => string]> = {
  translate: [["offset"], { offset: 3 }, (p, b) => `${p} - ${b.offset}`],
};

const BOOLEANS = new Set(["union", "intersection", "difference"]);
const PROFILES = new Set(["extruded_polygon", "revolved_polygon", "lofted_polygon"]);

/** The graph types this emitter covers. */
export const SUPPORTED_TYPES = [
  ...Object.keys(PRIMITIVES),
  ...PROFILES,
  "translate",
  "rotate",
  "mirror",
  "shell",
  "offset",
  "linear_pattern",
  "polar_pattern",
  ...BOOLEANS,
];

/** A parameter's value as a flat list of floats (`direct._value`). */
function value(node: SceneNode, name: string): number[] {
  const v = node.params[name];
  if (v === undefined) throw new UnsupportedNode(`${node.type} has no parameter ${name}`);
  return (Array.isArray(v) ? (v as (number | number[])[]).flat() : [v]).map((x) => (x === null ? NaN : Number(x)));
}

function literal(values: number[], arity: number): string {
  if (arity === 1) return formatG9(values[0]);
  return `vec${arity}<f32>(${values.slice(0, arity).map(formatG9).join(", ")})`;
}

function freeName(node: SceneNode, name: string): string | null {
  const declared = node.free?.[name];
  return typeof declared === "string" ? declared : null;
}

/** Python's `name in node.params`: draft and twist are in only when declared. */
function hasParam(node: SceneNode, name: string): boolean {
  if (!(name in node.params)) return false;
  if (node.type !== "extruded_polygon" || (name !== "draft" && name !== "twist")) return true;
  if (node.implicit) return !node.implicit.includes(name);
  return node.params[name] !== 0;
}

/** The node's graph children in Python's `node.children()` order. */
const childrenOf = (node: SceneNode): SceneNode[] => node.children;

function norm3(v: number[]): number {
  return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

class Emitter {
  readonly derived: string[] = [];
  readonly depends: Record<string, string[]> = {};
  readonly slots: Slot[] = [];
  readonly functions: string[] = [];
  readonly vertices: [number, number][] = [];
  private readonly bounds = new Map<string, Bounds | null>();
  private readonly boxes = new Map<string, [string, string, number]>();
  private readonly reads = new Map<string, Set<string>>();
  private readonly slotOf = new Map<string, number>();
  private readonly kernels = new Set<string>();
  private readonly nodes = new Map<string, string>();
  private readonly materials = new Map<string, string>();
  private readonly sketches = new Map<string, number>();
  private readonly anonymous = new WeakMap<SceneNode, string>();
  private anonymousCount = 0;

  readonly uniforms: boolean;
  readonly culling: boolean;

  constructor(uniforms: boolean, culling: boolean) {
    this.uniforms = uniforms;
    this.culling = culling;
  }

  /** Node identity: the v2 id, or object identity for a v1 graph. */
  private key(node: SceneNode): string {
    if (node.id !== undefined) return node.id;
    let key = this.anonymous.get(node);
    if (key === undefined) {
      key = `#${this.anonymousCount++}`;
      this.anonymous.set(node, key);
    }
    return key;
  }

  private vertexSlice(node: SceneNode, prefix: "v" | "w"): [number, number] {
    const graphKey = prefix === "v" ? "vertices" : "vertices_b";
    const verts = (node.params[graphKey] ?? []) as Vec[];
    const free = (node.free?.[graphKey] ?? []) as (string | null)[];
    if (this.uniforms) {
      let first: number | null = null;
      verts.forEach((v, i) => {
        const declared = free[i] ?? null;
        const nodeKey = this.key(node);
        if (!this.sketches.has(nodeKey)) this.sketches.set(nodeKey, this.sketches.size);
        const ordinal = this.sketches.get(nodeKey)!;
        const key = declared ?? `sketch${ordinal}.${prefix}${i}`;
        if (!this.slotOf.has(key)) {
          this.slotOf.set(key, this.slots.length);
          this.slots.push([key, 2, [Number(v[0]), Number(v[1])]]);
        }
        const index = this.slotOf.get(key)!;
        if (first === null) first = index;
      });
      return [first ?? 0, verts.length];
    }
    const offset = this.vertices.length;
    for (const v of verts) this.vertices.push([Number(v[0]), Number(v[1])]);
    return [offset, verts.length];
  }

  read(node: SceneNode, name: string, arity: number): string {
    const values = value(node, name);
    if (!this.uniforms) return literal(values, arity);
    const declared = freeName(node, name);
    if (declared === null) return literal(values, arity);
    if (!this.slotOf.has(declared)) {
      this.slotOf.set(declared, this.slots.length);
      this.slots.push([declared, arity, values.slice(0, arity)]);
    }
    return `sdf_parameters.values[${this.slotOf.get(declared)}].${"xyzw".slice(0, arity)}`;
  }

  private need(...symbols: string[]): void {
    for (const symbol of symbols) {
      if (this.kernels.has(symbol)) continue;
      this.kernels.add(symbol);
      this.functions.push(SHARED[symbol]);
    }
  }

  functionFor(node: SceneNode): string {
    const key = this.key(node);
    const known = this.nodes.get(key);
    if (known !== undefined) return known;
    const symbol = `node${this.nodes.size}`;
    this.nodes.set(key, symbol);
    const body = this.body(node);
    this.functions.push(`fn ${symbol}(p: vec3<f32>) -> f32 {\n${body}\n}`);
    return symbol;
  }

  // ── materials ──────────────────────────────────────────────────────────

  materialFor(node: SceneNode): string {
    const key = this.key(node);
    const known = this.materials.get(key);
    if (known !== undefined) return known;
    const symbol = `mat${this.materials.size}`;
    this.materials.set(key, symbol);
    const body = this.materialBody(node);
    this.functions.push(`fn ${symbol}(p: vec3<f32>) -> mat4x2<f32> {\n${body}\n}`);
    return symbol;
  }

  private materialBody(node: SceneNode): string {
    const type = node.type;
    if (BOOLEANS.has(type)) {
      const children = childrenOf(node);
      const smoothness = hasParam(node, "smoothness") ? value(node, "smoothness")[0] : 0;
      const k = formatG9(Math.max(smoothness * 4, 1e-10));
      const lines = [
        `    var acc = ${this.functionFor(children[0])}(p);`,
        `    var m = ${this.materialFor(children[0])}(p);`,
      ];
      children.slice(1).forEach((child, index) => {
        lines.push(
          `    let d${index} = ${this.functionFor(child)}(p);`,
          `    let n${index} = ${this.materialFor(child)}(p);`,
          `    let t${index} = clamp(0.5 + 0.5 * (d${index} - acc) / ${k}, 0.0, 1.0);`,
          `    m = m * t${index} + n${index} * (1.0 - t${index});`,
          `    let h${index} = max(${k} - abs(acc - d${index}), 0.0);`,
          `    acc = min(acc, d${index}) - h${index} * h${index} * 0.25 / ${k};`,
        );
      });
      lines.push("    return m;");
      return lines.join("\n");
    }
    if (type in TRANSFORMS || ["rotate", "mirror", "shell", "offset"].includes(type)) {
      const child = this.materialFor(childrenOf(node)[0]);
      return `    return ${child}(${this.movedPoint(node)});`;
    }
    if (type === "linear_pattern" || type === "polar_pattern") {
      return `    return ${this.materialFor(childrenOf(node)[0])}(p);`;
    }
    const material = node.material ?? DEFAULT_MATERIAL;
    const num = (x: number | null) => (x === null ? NaN : x);
    const colour = material.color.map(num);
    const row0 = [colour[0], colour[1], colour[2], num(material.roughness)];
    const row1 = [material.metallic, material.opacity, material.ior, material.reflectivity].map(num);
    const columns = [0, 1, 2, 3].map((i) => `vec2<f32>(${formatG9(row0[i])}, ${formatG9(row1[i])})`).join(", ");
    return `    return mat4x2<f32>(${columns});`;
  }

  private rotationLiteral(node: SceneNode): string {
    // `Rotate._rotation_matrix(...).T`, spelled by columns: column c of the
    // transpose is row c of the matrix.
    const rows = rotationMatrix(value(node, "axis"), value(node, "angle")[0]);
    return rows.map((row) => `vec3<f32>(${row.map(formatG9).join(", ")})`).join(", ");
  }

  private movedPoint(node: SceneNode): string {
    const type = node.type;
    if (type === "shell" || type === "offset") return "p";
    if (type === "rotate") return `mat3x3<f32>(${this.rotationLiteral(node)}) * p`;
    if (type === "mirror") {
      const origin = literal(value(node, "origin"), 3);
      const normal = literal(value(node, "normal"), 3);
      return `p - 2.0 * dot(p - ${origin}, ${normal}) * ${normal}`;
    }
    const [names, arity, template] = TRANSFORMS[type];
    const bindings = Object.fromEntries(names.map((name) => [name, this.read(node, name, arity[name])]));
    return template("p", bindings);
  }

  // ── culling ────────────────────────────────────────────────────────────

  private nodeBoundsOf(node: SceneNode): Bounds | null {
    const key = this.key(node);
    if (this.bounds.has(key)) return this.bounds.get(key)!;
    this.bounds.set(key, null);
    const children = childrenOf(node).map((child) => this.nodeBoundsOf(child));
    let bounds: Bounds | null;
    try {
      bounds = nodeBounds(node, children);
    } catch {
      bounds = null;
    }
    if (bounds !== null) {
      const finite = [...bounds.center, ...bounds.half].every(Number.isFinite);
      if (!(finite && bounds.scale > 0)) bounds = null;
    }
    this.bounds.set(key, bounds);
    return bounds;
  }

  private boxFor(node: SceneNode): [string, string, number] | null {
    if (!this.culling) return null;
    const key = this.key(node);
    const known = this.boxes.get(key);
    if (known !== undefined) return known;
    const bounds = this.nodeBoundsOf(node);
    if (bounds === null) return null;
    this.need("cull_box");
    let box: [string, string, number];
    if (this.uniforms) {
      const index = this.boxes.size;
      const reads = [...this.slotsRead(node)].sort();
      const expressions: string[] = [];
      for (const [part, values] of [["center", bounds.center], ["half", bounds.half]] as const) {
        const name = `__cull${index}.${part}`;
        this.slotOf.set(name, this.slots.length);
        this.slots.push([name, 3, [...values]]);
        this.derived.push(name);
        this.depends[name] = reads;
        expressions.push(`sdf_parameters.values[${this.slotOf.get(name)}].xyz`);
      }
      box = [expressions[0], expressions[1], bounds.scale];
    } else {
      box = [literal(bounds.center, 3), literal(bounds.half, 3), bounds.scale];
    }
    this.boxes.set(key, box);
    return box;
  }

  private slotsRead(node: SceneNode): Set<string> {
    const key = this.key(node);
    const known = this.reads.get(key);
    if (known !== undefined) return known;
    const names = new Set<string>();
    const ordinal = this.sketches.get(key);
    const check = (declared: string | null, own: string) => {
      if (declared !== null && this.slotOf.has(declared)) names.add(declared);
      else if (ordinal !== undefined && this.slotOf.has(`sketch${ordinal}.${own}`)) names.add(`sketch${ordinal}.${own}`);
    };
    for (const [graphKey, prefix] of [["vertices", "v"], ["vertices_b", "w"]] as const) {
      const verts = (node.params[graphKey] ?? []) as Vec[];
      const free = (node.free?.[graphKey] ?? []) as (string | null)[];
      verts.forEach((_, i) => check(free[i] ?? null, `${prefix}${i}`));
    }
    for (const name of Object.keys(node.params)) {
      if (name === "vertices" || name === "vertices_b" || !hasParam(node, name)) continue;
      check(freeName(node, name), name);
    }
    for (const child of childrenOf(node)) for (const name of this.slotsRead(child)) names.add(name);
    this.reads.set(key, names);
    return names;
  }

  private static boxTest(point: string, box: [string, string, number], threshold: string): string {
    const [center, half, scale] = box;
    let distance = `cull_box(${point}, ${center}, ${half})`;
    if (scale !== 1) distance = `${formatG9(scale)} * ${distance}`;
    return `${distance} >= ${threshold} + ${MARGIN_TOKEN}`;
  }

  // ── distance ───────────────────────────────────────────────────────────

  private body(node: SceneNode): string {
    const type = node.type;
    const primitive = PRIMITIVES[type];
    if (primitive) {
      const [className, kernel] = primitive;
      const sym = className.toLowerCase();
      this.need(`prim_${sym}`);
      const args = kernel.params.map((parameter) => this.read(node, parameter, kernel.arity[parameter])).join(", ");
      return `    return prim_${sym}(p, ${args});`;
    }
    if (PROFILES.has(type)) return this.profileBody(node);
    if (type === "shell" || type === "offset") {
      const child = this.functionFor(childrenOf(node)[0]);
      if (type === "shell") return `    return abs(${child}(p)) - ${formatG9(value(node, "thickness")[0] / 2)};`;
      return `    return ${child}(p) - ${formatG9(value(node, "distance")[0])};`;
    }
    if (type === "mirror") {
      const child = this.functionFor(childrenOf(node)[0]);
      const origin = literal(value(node, "origin"), 3);
      const normal = literal(value(node, "normal"), 3);
      return `    let n = ${normal};\n    return ${child}(p - 2.0 * dot(p - ${origin}, n) * n);`;
    }
    if (type === "rotate") {
      const columns = this.rotationLiteral(node);
      const child = this.functionFor(childrenOf(node)[0]);
      return `    return ${child}(mat3x3<f32>(${columns}) * p);`;
    }
    if (type in TRANSFORMS) {
      const [names, arity, template] = TRANSFORMS[type];
      const bindings = Object.fromEntries(names.map((name) => [name, this.read(node, name, arity[name])]));
      const child = this.functionFor(childrenOf(node)[0]);
      return `    return ${child}(${template("p", bindings)});`;
    }
    if (type === "linear_pattern" || type === "polar_pattern") return this.patternBody(node);
    if (BOOLEANS.has(type)) return this.booleanBody(node);
    throw new UnsupportedNode(`${type} has no direct WGSL kernel. Supported: ${SUPPORTED_TYPES.join(", ")}`);
  }

  private culledBooleanBody(type: string, nodes: SceneNode[], children: string[], k: string): string {
    const lines = [`    var d = ${children[0]}(p);`];
    for (let index = 1; index < nodes.length; index++) {
      const child = children[index];
      let operand: string, threshold: string, update: string;
      if (type === "union") {
        operand = `${child}(p)`;
        threshold = `max(d + ${k}, 0.0)`;
        update = `d = min(d, e${index}) - h${index} * h${index} * 0.25 / ${k};`;
      } else {
        operand = `(-${child}(p))`;
        threshold = `max(${k} - d, 0.0)`;
        update = `d = max(d, e${index}) + h${index} * h${index} * 0.25 / ${k};`;
      }
      const body = [`let e${index} = ${operand};`, `let h${index} = max(${k} - abs(d - e${index}), 0.0);`, update];
      const box = this.boxFor(nodes[index]);
      if (box === null) {
        lines.push(...body.map((line) => `    ${line}`));
      } else {
        lines.push(`    if (!(${Emitter.boxTest("p", box, threshold)})) {`);
        lines.push(...body.map((line) => `        ${line}`));
        lines.push("    }");
      }
    }
    lines.push("    return d;");
    return lines.join("\n");
  }

  private profileBody(node: SceneNode): string {
    if (node.type === "extruded_polygon") {
      this.need("profile_vertex", "profile_distance", "prim_extrudedpolygon");
      const [offset, count] = this.vertexSlice(node, "v");
      const depth = this.read(node, "depth", 1);
      let slope = 0;
      if (hasParam(node, "draft")) {
        const draft = value(node, "draft")[0] * (Math.PI / 180);
        slope = Math.sin(draft) / Math.cos(draft);
      }
      let twist = 0;
      if (hasParam(node, "twist")) twist = value(node, "twist")[0] * (Math.PI / 180);
      return `    return prim_extrudedpolygon(p, ${offset}u, ${count}u, ${depth}, ${formatG9(slope)}, ${formatG9(twist)});`;
    }
    if (node.type === "lofted_polygon") {
      this.need("profile_vertex", "profile_distance_lerped", "prim_loftedpolygon");
      const [a, countA] = this.vertexSlice(node, "v");
      const [b, countB] = this.vertexSlice(node, "w");
      if (countA !== countB) throw new UnsupportedNode("A loft needs both profiles to have the same vertex count.");
      const height = this.read(node, "height", 1);
      return `    return prim_loftedpolygon(p, ${a}u, ${b}u, ${countA}u, ${height});`;
    }
    this.need("profile_vertex", "profile_distance_revolved", "prim_revolvedpolygon");
    const [offset, count] = this.vertexSlice(node, "v");
    const radial = this.read(node, "offset", 1);
    return `    return prim_revolvedpolygon(p, ${offset}u, ${count}u, ${radial});`;
  }

  private patternBody(node: SceneNode): string {
    const childNode = childrenOf(node)[0];
    const child = this.functionFor(childNode);
    const count = Math.trunc(value(node, "count")[0]);
    const mask = hasParam(node, "skip_mask") ? Math.trunc(value(node, "skip_mask")[0]) : 0;
    const kept: number[] = [];
    for (let i = 0; i < count; i++) if (!(Number((BigInt(mask) >> BigInt(i)) & 1n))) kept.push(i);
    if (kept.length <= 1) return `    return ${child}(p);`;
    const lines = [`    var d = ${child}(p);`];
    lines.push(`    let instances = array<i32, ${kept.length - 1}>(${kept.slice(1).join(", ")});`);
    lines.push(`    for (var k = 0; k < ${kept.length - 1}; k = k + 1) {`);
    lines.push("        let i = f32(instances[k]);");
    if (node.type === "linear_pattern") {
      const direction = value(node, "direction");
      const length = norm3(direction) || 1;
      const axis = direction.map((x) => x / length);
      const spacing = value(node, "spacing")[0];
      lines.push(`        let q = p - ${literal(axis, 3)} * (${formatG9(spacing)} * i);`);
    } else {
      this.need("rotate_about");
      const origin = literal(value(node, "origin"), 3);
      const direction = value(node, "direction");
      const length = norm3(direction) || 1;
      const axis = direction.map((x) => x / length);
      lines.push(`        let theta = -6.283185307179586 * i / ${formatG9(count)};`);
      lines.push(`        let q = rotate_about(p, ${origin}, ${literal(axis, 3)}, theta);`);
    }
    const box = this.boxFor(childNode);
    if (box === null) {
      lines.push(`        d = min(d, ${child}(q));`);
    } else {
      lines.push(`        if (!(${Emitter.boxTest("q", box, "max(d, 0.0)")})) {`);
      lines.push(`            d = min(d, ${child}(q));`);
      lines.push("        }");
    }
    lines.push("    }");
    lines.push("    return d;");
    return lines.join("\n");
  }

  private booleanBody(node: SceneNode): string {
    const type = node.type;
    const smoothness = hasParam(node, "smoothness") ? value(node, "smoothness")[0] : 0;
    const k = formatG9(Math.max(smoothness * 4, 1e-10));
    const nodes = childrenOf(node);
    const children = nodes.map((child) => this.functionFor(child));
    if ((type === "union" || type === "difference") && this.culling)
      return this.culledBooleanBody(type, nodes, children, k);
    const lines = [`    var d = ${children[0]}(p);`];
    for (const child of children.slice(1)) {
      const operand = type === "difference" ? `(-${child}(p))` : `${child}(p)`;
      const combine = type === "union" ? "min" : "max";
      const sign = type === "union" ? "-" : "+";
      lines.push(`    let e = ${operand};`);
      lines.push(`    let h = max(${k} - abs(d - e), 0.0);`);
      lines.push(`    d = ${combine}(d, e) ${sign} h * h * 0.25 / ${k};`);
    }
    lines.push("    return d;");
    return uniqueLets(lines).join("\n");
  }
}

/** `direct._unique_lets`: every `let e`/`let h` in a body gets its own name. */
function uniqueLets(lines: string[]): string[] {
  const out: string[] = [];
  let index = 0;
  for (const line of lines) {
    if (line.trim().startsWith("let e =")) index += 1;
    out.push(
      line
        .replaceAll(" e ", ` e${index} `)
        .replaceAll(" h ", ` h${index} `)
        .replaceAll("(d - e)", `(d - e${index})`)
        .replaceAll("d, e)", `d, e${index})`),
    );
  }
  return out;
}

function uniformBlock(slots: Slot[]): string {
  const pad = (n: number) => String(n).padStart(6, " ");
  const lines = [`// ${slots.length} free parameters, one vec4<f32> slot each.`];
  slots.forEach(([name, components], index) => {
    lines.push(`// ${pad(index * PARAMETER_SLOT_BYTES)}  ${components}f  free   ${name}`);
  });
  lines.push(`// ${pad(slots.length * PARAMETER_SLOT_BYTES)}  1f  reserved  NaN (unread)`);
  lines.push(`// ${pad((slots.length + 1) * PARAMETER_SLOT_BYTES)}  1f  reserved  cull margin`);
  lines.push("struct SdfParameters {");
  lines.push(`    values: array<vec4<f32>, ${slots.length + RESERVED_PARAMETER_SLOTS}>,`);
  lines.push("};");
  lines.push(
    `@group(${DEFAULT_PARAMETER_GROUP}) @binding(${DEFAULT_PARAMETER_BINDING}) ` +
      `var<${PARAMETER_ADDRESS_SPACE}> sdf_parameters: SdfParameters;`,
  );
  return lines.join("\n");
}

export type DirectOptions = { entryPoint?: string; uniforms?: boolean; culling?: boolean };

/** `compile_sdf_direct` for a scene graph. */
export function compileGraphDirect(graph: SceneGraph, options: DirectOptions = {}): DirectProgram {
  const { entryPoint = "sdf", uniforms = false, culling = false } = options;
  const geometry = sceneRoot(graph);
  const emitter = new Emitter(uniforms, culling);
  const root = emitter.functionFor(geometry);
  const materialRoot = emitter.materialFor(geometry);
  const entry = [
    `fn ${entryPoint}(p: vec3<f32>) -> f32 {\n    return ${root}(p);\n}`,
    `fn material_block_impl(p: vec3<f32>) -> mat4x2<f32> {\n    return ${materialRoot}(p);\n}`,
    MATERIAL_ENTRY_POINTS,
  ].join("\n\n");
  let sections = [...emitter.functions];
  if (uniforms) {
    sections = sections.map((part) => (part === PROFILE_FROM_BUFFER ? PROFILE_FROM_UNIFORM : part));
    sections.unshift(uniformBlock(emitter.slots));
  } else if (emitter.vertices.length) {
    sections.unshift(PROFILE_BUFFER);
  }
  const margin = uniforms ? `sdf_parameters.values[${emitter.slots.length + 1}].x` : formatG9(CULL_MARGIN);
  return {
    wgsl: [...sections, entry].join("\n\n").replaceAll(MARGIN_TOKEN, margin),
    vertices: emitter.vertices,
    parameters: emitter.slots,
    derived: emitter.derived,
    depends: emitter.depends,
  };
}
