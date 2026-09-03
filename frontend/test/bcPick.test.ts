import { describe, expect, it } from "vitest";
import { nearestVertex, rectAabbProposal, round3, sphereProposal } from "../src/bcPick";
import { projectPoint, type View } from "../src/viewer/math";

// A three-quarter view in the Z-up world the camera math uses: the elevation
// is the Z component, so the camera stands above and in front of the cube and
// sees the +x face obliquely — which is what keeps a rect drawn around that
// face from also catching the far one.
const VIEW: View = {
  position: [4, -6, 3],
  target: [0, 0, 0],
  width: 800,
  height: 500,
};

/** A small vertex cloud: cube corners plus a point behind the camera. */
const WORLDS: [number, number, number][] = [
  [-1, -1, -1],
  [1, -1, -1],
  [1, 1, -1],
  [-1, 1, -1],
  [-1, -1, 1],
  [1, -1, 1],
  [1, 1, 1],
  [-1, 1, 1],
];
const positions = WORLDS.flat();

describe("nearestVertex", () => {
  it("finds the vertex whose projection is closest to the pixel", () => {
    for (const [index, world] of WORLDS.entries()) {
      const screen = projectPoint(world, VIEW);
      const hit = nearestVertex(positions, screen.x + 2, screen.y - 1, VIEW);
      expect(hit).not.toBeNull();
      expect(hit!.index).toBe(index);
      expect(hit!.world).toEqual(world);
    }
  });

  it("misses when nothing is inside the pixel radius", () => {
    expect(nearestVertex(positions, 5, 5, VIEW, 8)).toBeNull();
  });

  it("ignores vertices behind the camera", () => {
    const behind = [8, 6, 12];
    const merged = [...behind, ...positions];
    const front = projectPoint(WORLDS[6], VIEW);
    const hit = nearestVertex(merged, front.x, front.y, VIEW);
    expect(hit).not.toBeNull();
    expect(hit!.world).toEqual(WORLDS[6]);
  });
});

describe("sphereProposal", () => {
  it("rounds the centre and sizes the radius from the mean cell spacing", () => {
    const proposal = sphereProposal([0.12345, -0.5, 1.00049], {
      spacing: [0.1, 0.2, 0.3],
    });
    expect(proposal).toEqual({
      kind: "sphere",
      center: [0.123, -0.5, 1],
      radius: 0.4,
    });
  });

  it("falls back to a small fixed radius without grid info", () => {
    expect(sphereProposal([0, 0, 0], null).radius).toBeCloseTo(0.1);
  });
});

describe("rectAabbProposal", () => {
  it("bounds exactly the vertices projecting inside the rectangle", () => {
    // A rect tightly around the +x face corners of the cube.
    const face = [1, 2, 5, 6].map((index) => projectPoint(WORLDS[index], VIEW));
    const rect = {
      x0: Math.min(...face.map((point) => point.x)) - 2,
      y0: Math.min(...face.map((point) => point.y)) - 2,
      x1: Math.max(...face.map((point) => point.x)) + 2,
      y1: Math.max(...face.map((point) => point.y)) + 2,
    };
    const proposal = rectAabbProposal(positions, rect, VIEW);
    expect(proposal).not.toBeNull();
    expect(proposal!.kind).toBe("box");
    // The +x face spans x = 1, y and z = ±1, padded a hair beyond rounding.
    expect(proposal!.min[0]).toBeCloseTo(0.999, 3);
    expect(proposal!.max[0]).toBeCloseTo(1.001, 3);
    expect(proposal!.min[1]).toBeCloseTo(-1.001, 3);
    expect(proposal!.max[1]).toBeCloseTo(1.001, 3);
    expect(proposal!.min[2]).toBeCloseTo(-1.001, 3);
    expect(proposal!.max[2]).toBeCloseTo(1.001, 3);
  });

  it("accepts corners given in any drag direction", () => {
    const all = WORLDS.map((world) => projectPoint(world, VIEW));
    const rect = {
      x0: Math.max(...all.map((point) => point.x)) + 4,
      y0: Math.max(...all.map((point) => point.y)) + 4,
      x1: Math.min(...all.map((point) => point.x)) - 4,
      y1: Math.min(...all.map((point) => point.y)) - 4,
    };
    const proposal = rectAabbProposal(positions, rect, VIEW)!;
    expect(proposal.min).toEqual([-1.001, -1.001, -1.001]);
    expect(proposal.max).toEqual([1.001, 1.001, 1.001]);
  });

  it("returns null for an empty rectangle", () => {
    expect(rectAabbProposal(positions, { x0: 0, y0: 0, x1: 4, y1: 4 }, VIEW)).toBeNull();
  });
});

describe("round3", () => {
  it("rounds to three decimals", () => {
    expect(round3(1.23456)).toBe(1.235);
    expect(round3(-0.0004)).toBe(-0);
  });
});
