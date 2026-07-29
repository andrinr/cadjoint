import { describe, expect, it } from "vitest";
import type { ConstructionProfile } from "../src/types";
import { nearestInsertIndex, pickEdge, pickVertex, type PickView } from "../src/viewer/hittest";
import { projectPoint, type Vec3 } from "../src/viewer/math";

const VIEW: PickView = {
  position: [4, 3, 6],
  target: [0, 0, 0],
  width: 800,
  height: 500,
};

/** A unit square sketch on the world XY plane. */
function square(id: string, editable = true): ConstructionProfile {
  const corners: [number, number][] = [
    [-1, -1],
    [1, -1],
    [1, 1],
    [-1, 1],
  ];
  return {
    id,
    name: id,
    line: 3,
    editable,
    plane: { origin: [0, 0, 0], u: [1, 0, 0], v: [0, 1, 0], normal: [0, 0, 1] },
    vertices: corners.map(([x, y]) => ({
      name: `${id}_v`,
      free: true,
      uv: [x, y],
      world: [x, y, 0],
      span: [0, 8],
    })),
  };
}

/** Where a profile vertex lands on screen. */
function screenOf(profile: ConstructionProfile, index: number) {
  return projectPoint(profile.vertices[index].world, VIEW);
}

describe("pickVertex", () => {
  const profile = square("p0");

  it("picks the vertex under the cursor", () => {
    for (let index = 0; index < 4; index++) {
      const point = screenOf(profile, index);
      const hit = pickVertex([profile], point.x, point.y, VIEW);
      expect(hit).not.toBeNull();
      expect(hit!.vertexIndex).toBe(index);
      expect(hit!.profileId).toBe("p0");
    }
  });

  it("misses when the cursor is outside the pick radius", () => {
    const point = screenOf(profile, 0);
    expect(pickVertex([profile], point.x + 60, point.y + 60, VIEW, 12)).toBeNull();
  });

  it("prefers the nearer of two candidates", () => {
    const first = screenOf(profile, 0);
    const second = screenOf(profile, 1);
    const midpoint = { x: (first.x + second.x) / 2, y: (first.y + second.y) / 2 };
    const biased = { x: midpoint.x + (first.x - midpoint.x) * 0.9, y: midpoint.y + (first.y - midpoint.y) * 0.9 };
    const radius = Math.hypot(first.x - second.x, first.y - second.y);
    const hit = pickVertex([profile], biased.x, biased.y, VIEW, radius);
    expect(hit!.vertexIndex).toBe(0);
  });

  it("skips locked profiles when only editable ones are wanted", () => {
    const locked = square("locked", false);
    const point = screenOf(locked, 2);
    expect(pickVertex([locked], point.x, point.y, VIEW)).not.toBeNull();
    expect(pickVertex([locked], point.x, point.y, VIEW, 12, true)).toBeNull();
  });

  it("ignores vertices behind the camera", () => {
    const behind = square("behind");
    behind.vertices = behind.vertices.map((vertex) => ({
      ...vertex,
      world: [10, 8, 15] as [number, number, number],
    }));
    expect(pickVertex([behind], 400, 250, VIEW)).toBeNull();
  });
});

describe("pickEdge", () => {
  const profile = square("p0");

  it("returns the index just after the edge's start vertex", () => {
    for (let index = 0; index < 4; index++) {
      const start = screenOf(profile, index);
      const end = screenOf(profile, (index + 1) % 4);
      const hit = pickEdge([profile], (start.x + end.x) / 2, (start.y + end.y) / 2, VIEW);
      expect(hit).not.toBeNull();
      expect(hit!.insertIndex).toBe(index + 1);
    }
  });

  it("treats the closing edge as an append", () => {
    const start = screenOf(profile, 3);
    const end = screenOf(profile, 0);
    const hit = pickEdge([profile], (start.x + end.x) / 2, (start.y + end.y) / 2, VIEW);
    expect(hit!.insertIndex).toBe(4);
  });

  it("misses when the cursor is far from every edge", () => {
    expect(pickEdge([profile], 5, 5, VIEW, 10)).toBeNull();
  });

  it("never offers to edit a locked profile", () => {
    const locked = square("locked", false);
    const start = screenOf(locked, 0);
    const end = screenOf(locked, 1);
    expect(pickEdge([locked], (start.x + end.x) / 2, (start.y + end.y) / 2, VIEW)).toBeNull();
  });
});

describe("nearestInsertIndex", () => {
  const profile = square("p0");

  it("inserts right after the closest vertex", () => {
    const point = screenOf(profile, 2);
    expect(nearestInsertIndex(profile, point.x, point.y, VIEW)).toBe(3);
  });

  it("appends when no vertex is visible", () => {
    const hidden: ConstructionProfile = {
      ...profile,
      vertices: profile.vertices.map((vertex) => ({
        ...vertex,
        world: [20, 15, 30] as Vec3 as [number, number, number],
      })),
    };
    expect(nearestInsertIndex(hidden, 400, 250, VIEW)).toBe(4);
  });
});
