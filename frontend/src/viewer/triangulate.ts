/**
 * Ear clipping: a simple polygon into triangles.
 *
 * The face highlight needs a *fill*, and a face boundary is not convex — the
 * starter's fin comb is a sixteen-point comb with three notches in it, so a
 * triangle fan from the centroid would spill outside the part and, under
 * alpha blending, print the overlap darker than the rest. Ear clipping is the
 * smallest correct answer for a simple (non-self-intersecting) loop, which is
 * what a face boundary always is.
 *
 * Pure indices in, indices out; unit tested in `test/triangulate.test.ts`.
 */

export type Point2 = readonly [number, number];

/** Twice the signed area of a polygon: positive when it winds anticlockwise. */
export function signedArea(polygon: readonly Point2[]): number {
  let total = 0;
  for (let index = 0, previous = polygon.length - 1; index < polygon.length; previous = index++) {
    total += (polygon[previous][0] - polygon[index][0]) * (polygon[previous][1] + polygon[index][1]);
  }
  return total / 2;
}

/** Twice the signed area of the triangle abc; positive when anticlockwise. */
const turn = (a: Point2, b: Point2, c: Point2): number =>
  (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);

/** Whether p lies inside (or on) the triangle abc. */
function inTriangle(p: Point2, a: Point2, b: Point2, c: Point2): boolean {
  const d1 = turn(a, b, p);
  const d2 = turn(b, c, p);
  const d3 = turn(c, a, p);
  const negative = d1 < 0 || d2 < 0 || d3 < 0;
  const positive = d1 > 0 || d2 > 0 || d3 > 0;
  return !(negative && positive);
}

/**
 * Triangulate a simple polygon, returning flat index triples into it.
 *
 * Winding is normalized first, so a clockwise boundary triangulates exactly
 * like an anticlockwise one; the result is always in the input's own index
 * space, which is what lets the caller map straight back to world points.
 *
 * A polygon that cannot be reduced any further (a duplicated vertex, a
 * degenerate spur) ends the loop rather than spinning: the fill is decoration
 * for a highlight, and a partial fan is better than a hang.
 */
export function triangulate(polygon: readonly Point2[]): number[] {
  const count = polygon.length;
  if (count < 3) return [];
  const anticlockwise = signedArea(polygon) > 0;
  const remaining: number[] = [];
  for (let index = 0; index < count; index++) {
    remaining.push(anticlockwise ? index : count - 1 - index);
  }

  const triangles: number[] = [];
  let guard = remaining.length * remaining.length;
  while (remaining.length > 3 && guard-- > 0) {
    let clipped = false;
    for (let index = 0; index < remaining.length; index++) {
      const previous = remaining[(index + remaining.length - 1) % remaining.length];
      const current = remaining[index];
      const next = remaining[(index + 1) % remaining.length];
      const a = polygon[previous];
      const b = polygon[current];
      const c = polygon[next];
      // Convex corner only; a reflex one cuts across the polygon.
      if (turn(a, b, c) <= 0) continue;
      // …and no other vertex may sit inside the ear.
      const blocked = remaining.some(
        (other) =>
          other !== previous &&
          other !== current &&
          other !== next &&
          inTriangle(polygon[other], a, b, c),
      );
      if (blocked) continue;
      triangles.push(previous, current, next);
      remaining.splice(index, 1);
      clipped = true;
      break;
    }
    if (!clipped) break;
  }
  if (remaining.length === 3) triangles.push(remaining[0], remaining[1], remaining[2]);
  return triangles;
}
