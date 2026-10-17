/**
 * A port of the cadjoint operations the space-frame node uses, on jax-js.
 *
 * Points travel as three column arrays, and every transform constant is a
 * plain JS number, so only the coordinates need jax-js's reference
 * counting: every function *consumes* the arrays it is given, and a body
 * that uses one twice takes `.ref` on all but the last use.
 *
 * Each formula mirrors the cadjoint source it names; reference.json (from
 * make_reference.py) is what holds them to it.
 */
import { numpy as np } from "@jax-js/jax";

type A = np.Array;
export type Pts = { x: A; y: A; z: A };
/** A field consumes its points and returns one distance per point. */
export type Field = (p: Pts) => A;
/** A scalar parameter: a JS number, or an Array when it is differentiated. */
type S = number | A;

// A field runs once per evaluation, so an Array parameter is consumed by
// the one primitive that uses it, like any other argument.

/** Hand the same points to two consumers. */
function fork(p: Pts): [Pts, Pts] {
  return [{ x: p.x.ref, y: p.y.ref, z: p.z.ref }, p];
}

/** cadjoint/sdf/primitives/sphere.py: ||p|| - r. */
export const sphere = (r: S): Field => ({ x, y, z }) =>
  np.sqrt(np.square(x).add(np.square(y)).add(np.square(z))).sub(r);

/** cadjoint/sdf/primitives/cylinder.py: Z axis, half-height h, exact inside/outside. */
export const cylinder = (r: S, h: number): Field => ({ x, y, z }) => {
  const dxy = np.sqrt(np.square(x).add(np.square(y)).add(1e-20)).sub(r);
  const dz = np.abs(z).sub(h);
  const maxd = np.maximum(dxy.ref, dz.ref);
  const squared = np.square(np.maximum(dxy, 0)).add(np.square(np.maximum(dz, 0)));
  const inside = maxd.ref.lessEqual(0);
  const safe = np.where(inside.ref, 1, squared);
  return np.where(inside, maxd, np.sqrt(safe));
};

/** Translate: evaluate the child at p - offset. */
export const translate = (f: Field, [ox, oy, oz]: number[]): Field => ({ x, y, z }) =>
  f({ x: x.sub(ox), y: y.sub(oy), z: z.sub(oz) });

/**
 * cadjoint/sdf/transforms/affine/rotate.py: Rodrigues' matrix R about a
 * unit axis; the child is evaluated at R^T p.
 */
export const rotate = (f: Field, axis: number[], angle: number): Field => {
  const n = Math.hypot(...axis);
  const [ax, ay, az] = axis.map((v) => v / n);
  const c = Math.cos(angle), s = Math.sin(angle), t = 1 - c;
  const R = [
    [t * ax * ax + c, t * ax * ay - s * az, t * ax * az + s * ay],
    [t * ax * ay + s * az, t * ay * ay + c, t * ay * az - s * ax],
    [t * ax * az - s * ay, t * ay * az + s * ax, t * az * az + c],
  ];
  // (R^T p)_i = sum_j R[j][i] p_j
  return ({ x, y, z }) => {
    const row = (i: number, last: boolean) => {
      const xs = last ? x : x.ref, ys = last ? y : y.ref, zs = last ? z : z.ref;
      return xs.mul(R[0][i]).add(ys.mul(R[1][i])).add(zs.mul(R[2][i]));
    };
    const px = row(0, false), py = row(1, false), pz = row(2, true);
    return f({ x: px, y: py, z: pz });
  };
};

/** cadjoint/sdf/boolean/smooth.py: Quilez's polynomial smooth minimum. */
function smoothMin(a: A, b: A, k: number): A {
  const k4 = Math.max(k * 4, 1e-10);
  const h = np.maximum(np.abs(a.ref.sub(b.ref)).neg().add(k4), 0);
  return np.minimum(a, b).sub(np.square(h).mul(0.25 / k4));
}

/** cadjoint Union: pairwise (smooth) minimum, left to right. */
export const union = (fields: Field[], k: number): Field => (p) => {
  let rest = p;
  let result: A | null = null;
  fields.forEach((f, i) => {
    const last = i === fields.length - 1;
    const [mine, others] = last ? [rest, rest] : fork(rest);
    const d = f(mine);
    rest = others;
    result = result === null ? d : k > 0 ? smoothMin(result, d, k) : np.minimum(result, d);
  });
  return result!;
};

/** cadjoint Intersection, sharp: pointwise maximum. */
export const intersection = (a: Field, b: Field): Field => (p) => {
  const [pa, pb] = fork(p);
  return np.maximum(a(pa), b(pb));
};

/** cadjoint/sdf/transforms/fields.py Shell: |f| - thickness / 2. */
export const shell = (f: Field, thickness: number): Field => (p) =>
  np.abs(f(p)).sub(thickness / 2);

const DIRECTIONS: [number, number][] = [[0, 0], [115, 0], [115, 120], [115, 240], [70, 60], [70, 300]];
const rad = (deg: number) => (deg * Math.PI) / 180;

/** skills/cadjoint/references/examples/space_frame_node.py, in jax-js. */
export function spaceFrameNode(hubRadius: S = 15, blend = 4, wall = 2): Field {
  const tubes = DIRECTIONS.map(([tilt, turn]) =>
    rotate(rotate(translate(cylinder(10, 22), [0, 0, 22]), [0, 1, 0], rad(tilt)), [0, 0, 1], rad(turn)),
  );
  const solid = union([sphere(hubRadius), ...tubes], blend);
  return intersection(shell(solid, wall), sphere(40));
}
