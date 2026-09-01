import { describe, expect, it } from "vitest";
import type { ConstructionTransform } from "../src/types";
import {
  AXES,
  angleAroundAxis,
  angleDelta,
  closestPointOnAxis,
  gizmoEdges,
  gizmoScale,
  pickGizmoAxis,
  placeEdges,
  rotationMatrix,
  scaleDimensions,
} from "../src/viewer/gizmo";
import { projectPoint, type Vec3, type View } from "../src/viewer/math";

const VIEW: View = {
  position: [4, 3, 6],
  target: [0, 0, 0],
  width: 800,
  height: 500,
};

describe("rotationMatrix", () => {
  // Cross-language check: these values come from `_rotation_matrix` in
  // cadjoint/construction/solid.py, which computes in float32 — hence the 1e-6
  // tolerance. If the two implementations drift further than that, the
  // wireframe stops sitting on the solid it is supposed to outline.
  it("matches the Python composition for intrinsic XYZ angles", () => {
    const expected = [
      [0.749596297741, -0.66034913063, 0.045215312392],
      [0.631376206875, 0.692859172821, -0.348296314478],
      [0.198669329286, 0.289629489183, 0.936293423176],
    ];
    const matrix = rotationMatrix([0.3, -0.2, 0.7]);
    for (let row = 0; row < 3; row++) {
      for (let column = 0; column < 3; column++) {
        expect(matrix[row][column]).toBeCloseTo(expected[row][column], 6);
      }
    }
  });

  it("is the identity for zero angles", () => {
    const matrix = rotationMatrix([0, 0, 0]);
    for (let row = 0; row < 3; row++) {
      for (let column = 0; column < 3; column++) {
        expect(matrix[row][column]).toBeCloseTo(row === column ? 1 : 0, 9);
      }
    }
  });

  it("turns +X onto +Y for a quarter turn about Z", () => {
    const matrix = rotationMatrix([0, 0, Math.PI / 2]);
    const turned = [matrix[0][0], matrix[1][0], matrix[2][0]];
    expect(turned[0]).toBeCloseTo(0, 6);
    expect(turned[1]).toBeCloseTo(1, 6);
  });
});

describe("placeEdges", () => {
  const transform: ConstructionTransform = {
    position: [1, 0, 0],
    rotation: [0, 0, 0],
    dimensions: { size: [1, 1, 1] },
    line: 3,
    call: "box",
    positionArgument: "position",
    canRotate: true,
  };
  const edges = [
    [
      [2, 0, 0],
      [2, 1, 0],
    ],
  ];

  it("leaves geometry untouched when the placement is unchanged", () => {
    const placed = placeEdges(edges, transform, [1, 0, 0], [0, 0, 0]);
    expect(placed[0][0][0]).toBeCloseTo(2, 6);
    expect(placed[0][1][1]).toBeCloseTo(1, 6);
  });

  it("translates with the position", () => {
    const placed = placeEdges(edges, transform, [3, 0, 0], [0, 0, 0]);
    expect(placed[0][0][0]).toBeCloseTo(4, 6);
  });

  it("rotates about the primitive's own origin, not the world origin", () => {
    // The edge point sits +1 along X from the primitive at (1, 0, 0); a quarter
    // turn about Z must swing it to +1 along Y from that same origin.
    const placed = placeEdges(edges, transform, [1, 0, 0], [0, 0, Math.PI / 2]);
    expect(placed[0][0][0]).toBeCloseTo(1, 6);
    expect(placed[0][0][1]).toBeCloseTo(1, 6);
  });

  it("resizes local geometry while preserving its placement", () => {
    const placed = placeEdges(
      edges,
      transform,
      [1, 0, 0],
      [0, 0, 0],
      { size: [2, 1, 1] },
    );
    expect(placed[0][0][0]).toBeCloseTo(3, 6);
    expect(placed[0][1][1]).toBeCloseTo(1, 6);
  });
});

describe("scaleDimensions", () => {
  it("scales one box axis", () => {
    expect(scaleDimensions("box", { size: [1, 2, 3] }, 1, 1.5)).toEqual({
      size: [1, 3, 3],
    });
  });

  it("scales spheres uniformly from any handle", () => {
    expect(scaleDimensions("sphere", { radius: 2 }, 0, 0.5)).toEqual({ radius: 1 });
  });

  it("uses radial and height dimensions for cylinders", () => {
    expect(scaleDimensions("cylinder", { radius: 2, height: 3 }, 0, 1.5)).toEqual({
      radius: 3,
      height: 3,
    });
    expect(scaleDimensions("cylinder", { radius: 2, height: 3 }, 2, 2)).toEqual({
      radius: 2,
      height: 6,
    });
  });

  it("never mirrors through zero", () => {
    expect(scaleDimensions("sphere", { radius: 2 }, 0, -4)).toEqual({ radius: 0.1 });
  });
});

describe("closestPointOnAxis", () => {
  it("finds the axis parameter nearest a perpendicular ray", () => {
    const ray = { origin: [2, 5, 0] as Vec3, direction: [0, -1, 0] as Vec3 };
    // The ray passes over x = 2, so the nearest point on the X axis is t = 2.
    expect(closestPointOnAxis(ray, [0, 0, 0], AXES[0])).toBeCloseTo(2, 6);
  });

  it("is stable for a ray parallel to the axis", () => {
    const ray = { origin: [0, 1, 0] as Vec3, direction: [1, 0, 0] as Vec3 };
    expect(Number.isFinite(closestPointOnAxis(ray, [0, 0, 0], AXES[0]))).toBe(true);
  });

  it("tracks the pointer along the axis", () => {
    const first = closestPointOnAxis(
      { origin: [1, 4, 0], direction: [0, -1, 0] },
      [0, 0, 0],
      AXES[0],
    );
    const second = closestPointOnAxis(
      { origin: [3, 4, 0], direction: [0, -1, 0] },
      [0, 0, 0],
      AXES[0],
    );
    expect(second - first).toBeCloseTo(2, 6);
  });
});

describe("angleAroundAxis", () => {
  it("measures rotation in the plane normal to the axis", () => {
    const origin: Vec3 = [0, 0, 0];
    const first = angleAroundAxis({ origin: [1, 0, 5], direction: [0, 0, -1] }, origin, AXES[2]);
    const second = angleAroundAxis({ origin: [0, 1, 5], direction: [0, 0, -1] }, origin, AXES[2]);
    expect(first).not.toBeNull();
    expect(second).not.toBeNull();
    expect(Math.abs(angleDelta(first!, second!))).toBeCloseTo(Math.PI / 2, 5);
  });

  it("returns null when the ray grazes the plane", () => {
    const ray = { origin: [1, 0, 0] as Vec3, direction: [0, 1, 0] as Vec3 };
    expect(angleAroundAxis(ray, [0, 0, 0], AXES[2])).toBeNull();
  });
});

describe("angleDelta", () => {
  it("takes the short way round the circle", () => {
    expect(angleDelta(3.0, -3.0)).toBeCloseTo(2 * Math.PI - 6, 6);
    expect(angleDelta(0.1, 0.4)).toBeCloseTo(0.3, 6);
    expect(Math.abs(angleDelta(0, Math.PI))).toBeCloseTo(Math.PI, 6);
  });
});

describe("pickGizmoAxis", () => {
  const origin: Vec3 = [0, 0, 0];
  const size = gizmoScale(VIEW, origin);

  it("picks the axis whose arrow is under the cursor", () => {
    for (const index of [0, 1, 2] as const) {
      const tip: Vec3 = [
        AXES[index][0] * size * 0.6,
        AXES[index][1] * size * 0.6,
        AXES[index][2] * size * 0.6,
      ];
      const point = projectPoint(tip, VIEW);
      expect(pickGizmoAxis(origin, size, "translate", point.x, point.y, VIEW)).toBe(index);
    }
  });

  it("misses when the cursor is away from every handle", () => {
    expect(pickGizmoAxis(origin, size, "translate", 5, 5, VIEW)).toBeNull();
  });

  it("picks rotate rings too", () => {
    const groups = gizmoEdges(origin, size, "rotate");
    // Sample partway round the ring: the three rings genuinely cross on the
    // axes, so a point there is ambiguous by construction.
    const [start] = groups[2].edges[5];
    const point = projectPoint(start, VIEW);
    expect(pickGizmoAxis(origin, size, "rotate", point.x, point.y, VIEW)).toBe(2);
  });

  it("scales with distance so it stays usable when zoomed out", () => {
    const near = gizmoScale({ ...VIEW, position: [1, 1, 1] }, origin);
    const far = gizmoScale({ ...VIEW, position: [40, 30, 60] }, origin);
    expect(far).toBeGreaterThan(near);
  });
});
