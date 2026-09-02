/**
 * Ear clipping, held to the case that made it necessary.
 *
 * A triangle fan would pass every convex test and still be wrong for the one
 * polygon this feature actually has to fill — the starter's fin comb, whose
 * cap is a sixteen-point loop with three notches cut into it. So the notch is
 * the test: no triangle may cover a point outside the polygon.
 */

import { describe, expect, it } from "vitest";
import { signedArea, triangulate, type Point2 } from "../src/viewer/triangulate";

/** A comb: a base bar with two square notches cut down into its top edge. */
const COMB: Point2[] = [
  [0, 0],
  [6, 0],
  [6, 3],
  [5, 3],
  [5, 1],
  [4, 1],
  [4, 3],
  [2, 3],
  [2, 1],
  [1, 1],
  [1, 3],
  [0, 3],
];

const SQUARE: Point2[] = [
  [0, 0],
  [2, 0],
  [2, 2],
  [0, 2],
];

/** Area of the triangles a triangulation produces, ignoring winding. */
function triangleArea(polygon: readonly Point2[], indices: readonly number[]): number {
  let total = 0;
  for (let index = 0; index < indices.length; index += 3) {
    const [a, b, c] = [polygon[indices[index]], polygon[indices[index + 1]], polygon[indices[index + 2]]];
    total += Math.abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2;
  }
  return total;
}

describe("triangulate", () => {
  it("refuses anything that is not a polygon", () => {
    expect(triangulate([])).toEqual([]);
    expect(triangulate([[0, 0]])).toEqual([]);
    expect(triangulate([[0, 0], [1, 0]])).toEqual([]);
  });

  it("produces n − 2 triangles for a simple loop", () => {
    expect(triangulate(SQUARE)).toHaveLength(2 * 3);
    expect(triangulate(COMB)).toHaveLength((COMB.length - 2) * 3);
  });

  it("covers exactly the polygon's area, notches included", () => {
    // 6×3 bar minus two 1×2 notches = 18 − 4 = 14.
    expect(Math.abs(signedArea(COMB))).toBeCloseTo(14, 6);
    expect(triangleArea(COMB, triangulate(COMB))).toBeCloseTo(14, 6);
    expect(triangleArea(SQUARE, triangulate(SQUARE))).toBeCloseTo(4, 6);
  });

  it("gives the same triangles whichever way the boundary winds", () => {
    const reversed = [...COMB].reverse();
    expect(triangleArea(reversed, triangulate(reversed))).toBeCloseTo(14, 6);
    expect(signedArea(COMB) * signedArea(reversed)).toBeLessThan(0);
  });

  it("never spans a notch, which a centroid fan would", () => {
    // (1.5, 2) sits in the left notch: inside the bounding box, outside the
    // polygon. No triangle may contain it.
    const point: Point2 = [1.5, 2];
    const indices = triangulate(COMB);
    const covers = (a: Point2, b: Point2, c: Point2) => {
      const side = (p: Point2, q: Point2) =>
        (q[0] - p[0]) * (point[1] - p[1]) - (q[1] - p[1]) * (point[0] - p[0]);
      const d = [side(a, b), side(b, c), side(c, a)];
      return !(d.some((v) => v < -1e-9) && d.some((v) => v > 1e-9));
    };
    for (let index = 0; index < indices.length; index += 3) {
      expect(
        covers(COMB[indices[index]], COMB[indices[index + 1]], COMB[indices[index + 2]]),
        `triangle ${index / 3}`,
      ).toBe(false);
    }
  });

  it("terminates on a degenerate loop instead of spinning", () => {
    const degenerate: Point2[] = [
      [0, 0],
      [1, 0],
      [1, 0],
      [0, 0],
    ];
    expect(() => triangulate(degenerate)).not.toThrow();
  });
});
