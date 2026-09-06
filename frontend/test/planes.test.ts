import { describe, expect, it } from "vitest";
import {
  labelAnchor,
  normalTip,
  PLANE_MARGIN,
  PLANE_MIN_HALF,
  planeAxes,
  planeFrame,
  planeFrames,
  previewFrame,
} from "../src/planes";
import { pickPlane } from "../src/viewer/hittest";
import { packPlaneOverlay } from "../src/viewer/overlayGeometry";
import { projectPoint, type Vec3 } from "../src/viewer/math";
import type { ConstructionNode } from "../src/types";

const VIEW = { position: [4, 3, 6] as Vec3, target: [0, 0, 0] as Vec3, width: 800, height: 500 };

const close = (a: readonly number[], b: readonly number[]) =>
  a.every((value, index) => Math.abs(value - b[index]) < 1e-9);

/** A sketch on a stated plane, with the given points. */
function sketch(
  id: string,
  uv: [number, number][],
  overrides: Partial<ConstructionNode> = {},
  planeOverrides: Partial<NonNullable<ConstructionNode["plane"]>> = {},
): ConstructionNode {
  const origin: Vec3 = [0.5, 0, 0.2];
  const u: Vec3 = [1, 0, 0];
  const v: Vec3 = [0, 1, 0];
  const vertices = uv.map(([x, y]) => ({
    stableId: null,
    name: null,
    free: true,
    uv: [x, y] as [number, number],
    world: [origin[0] + x, origin[1] + y, origin[2]] as [number, number, number],
    span: null,
  }));
  return {
    id,
    stableId: `assign:${id}`,
    kind: "profile",
    name: id,
    line: 3,
    editable: true,
    edges: [],
    plane: {
      origin: [...origin],
      u: [...u],
      v: [...v],
      normal: [0, 0, 1],
      stableId: `plane:${id}`,
      reference: null,
      ...planeOverrides,
    },
    faces: [],
    vertices,
    transform: {
      position: [...origin],
      rotation: [0, 0, 0],
      dimensions: {},
      line: 3,
      call: "SketchPlane",
      positionArgument: "origin",
      canRotate: false,
      bindings: {},
    },
    spans: {},
    constraints: [],
    operators: [],
    material: null,
    ...overrides,
  };
}

describe("planeAxes", () => {
  it("gives the identity frame for the default +Z normal", () => {
    const { u, v } = planeAxes([0, 0, 1]);
    expect(close(u, [1, 0, 0])).toBe(true);
    expect(close(v, [0, 1, 0])).toBe(true);
  });

  it("matches the construction module's frame for a +Y normal", () => {
    // Sketch plane normal +Y gives u = -X, v = +Z (see scenes/starter.py).
    const { u, v } = planeAxes([0, 1, 0]);
    expect(close(u, [-1, 0, 0])).toBe(true);
    expect(close(v, [0, 0, 1])).toBe(true);
  });

  it("is right-handed for any normal", () => {
    for (const normal of [
      [1, 0, 0],
      [0, -1, 0],
      [0.3, 0.4, 0.5],
    ] as Vec3[]) {
      const { u, v } = planeAxes(normal);
      const n = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
      ];
      const length = Math.hypot(...normal);
      expect(close(n, normal.map((c) => c / length))).toBe(true);
    }
  });
});

describe("planeFrame", () => {
  it("frames the sketch's points with a margin, centred on them", () => {
    const frame = planeFrame(
      sketch("s", [
        [1, 1],
        [3, 1],
        [3, 2],
        [1, 2],
      ]),
    )!;
    expect(frame.centerU).toBe(2);
    expect(frame.centerV).toBe(1.5);
    expect(frame.halfU).toBe(1 + PLANE_MARGIN);
    expect(frame.halfV).toBe(0.5 + PLANE_MARGIN);
    // Corners are origin + u * (centre ± half) + v * (centre ± half).
    expect(close(frame.corners[0], [0.5 + 2 - 1.25, 1.5 - 0.75, 0.2])).toBe(true);
    expect(close(frame.corners[2], [0.5 + 2 + 1.25, 1.5 + 0.75, 0.2])).toBe(true);
  });

  it("never shrinks below the minimum, and centres an empty plane on its origin", () => {
    const frame = planeFrame(sketch("s", []))!;
    expect(frame.halfU).toBe(PLANE_MIN_HALF);
    expect(frame.centerU).toBe(0);
    expect(close(labelAnchor(frame), [0.5 + PLANE_MIN_HALF, PLANE_MIN_HALF, 0.2])).toBe(true);
  });

  it("marks a face-derived plane as derived and not editable", () => {
    const frame = planeFrame(
      sketch("s", [[0, 0]], { transform: null }, {
        reference: { constructor: "on", owner: "body", accessor: "cap", argument: '"+"' },
      }),
    )!;
    expect(frame.derived).toBe(true);
    expect(frame.editable).toBe(false);
  });

  it("returns null for a primitive, which has no plane", () => {
    expect(planeFrame(sketch("b", [], { kind: "box", plane: null }))).toBeNull();
    expect(planeFrames([sketch("b", [], { kind: "box", plane: null }), sketch("s", [])])).toHaveLength(1);
  });

  it("points the normal glyph along the plane normal", () => {
    const frame = planeFrame(sketch("s", []))!;
    const tip = normalTip(frame);
    expect(tip[0]).toBe(frame.origin[0]);
    expect(tip[2]).toBeGreaterThan(frame.origin[2]);
  });
});

describe("previewFrame", () => {
  it("is the square add_sketch writes, plus the margin, facing the normal", () => {
    const frame = previewFrame([1, 2, 3], [0, 1, 0]);
    expect(frame.nodeId).toBeNull();
    expect(frame.halfU).toBe(0.6 + PLANE_MARGIN);
    expect(close(frame.normal, [0, 1, 0])).toBe(true);
    for (const corner of frame.corners) expect(corner[1]).toBe(2);
  });
});

describe("packPlaneOverlay", () => {
  const FILL_FLOATS = 7;
  const EDGE_FLOATS = 10;

  it("emits two triangles and eleven segments per plane", () => {
    const { fill, outline } = packPlaneOverlay([planeFrame(sketch("s", []))!], null, null, null);
    expect(fill.length / FILL_FLOATS).toBe(6);
    // 4 frame edges + 2 origin cross + 1 shaft + 2 barbs.
    expect(outline.length / EDGE_FLOATS).toBe(9);
  });

  it("draws the selected plane in a different ink from a resting one", () => {
    const frame = planeFrame(sketch("s", []))!;
    const resting = packPlaneOverlay([frame], null, null, null);
    const selected = packPlaneOverlay(
      [frame],
      { nodeId: "s", vertexIndex: null, part: "plane" },
      null,
      null,
    );
    expect(resting.outline.slice(6, 10)).not.toEqual(selected.outline.slice(6, 10));
  });

  it("adds the preview ghost after the real planes", () => {
    const frame = planeFrame(sketch("s", []))!;
    const { fill } = packPlaneOverlay([frame], null, null, previewFrame([0, 0, 0], [0, 0, 1]));
    expect(fill.length / FILL_FLOATS).toBe(12);
  });
});

describe("pickPlane", () => {
  it("picks a plane by its frame edge and by its origin mark", () => {
    const node = sketch("s", [
      [-1, -1],
      [1, -1],
      [1, 1],
      [-1, 1],
    ]);
    const frames = planeFrames([node]);
    const frame = frames[0];
    const edgeMid: Vec3 = [
      (frame.corners[0][0] + frame.corners[1][0]) / 2,
      (frame.corners[0][1] + frame.corners[1][1]) / 2,
      (frame.corners[0][2] + frame.corners[1][2]) / 2,
    ];
    const onEdge = projectPoint(edgeMid, VIEW);
    expect(pickPlane(frames, onEdge.x, onEdge.y, VIEW)?.nodeId).toBe("s");
    const atOrigin = projectPoint(frame.origin, VIEW);
    expect(pickPlane(frames, atOrigin.x, atOrigin.y, VIEW)?.nodeId).toBe("s");
    // Far from the frame is nothing: the fill is not a handle.
    const away = projectPoint([0.5, 0, 5], VIEW);
    expect(pickPlane(frames, away.x, away.y, VIEW)).toBeNull();
  });

  it("never picks the preview, which belongs to no sketch", () => {
    const preview = previewFrame([0, 0, 0], [0, 0, 1]);
    const at = projectPoint(preview.origin, VIEW);
    expect(pickPlane([preview], at.x, at.y, VIEW)).toBeNull();
  });
});
