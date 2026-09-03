/**
 * Face picking, held to the four rejections it is made of.
 *
 * The numbers here come from the starter scene's own payload: the fin comb's
 * two caps sit at y = ∓0.6 with a 0.00199 tolerance, which is *below* twice
 * the raymarcher's own surface epsilon — so the floor in `faceTolerance` is
 * not a nicety, it is what stops a hit the rendered image says is on the cap
 * from being rejected as off-plane.
 */

import { describe, expect, it } from "vitest";
import {
  RAYMARCH_SURFACE_EPSILON,
  faceLabel,
  faceReference,
  faceTolerance,
  pickFace,
  pickFaceSurface,
  polygonDistance,
  referenceFor,
  resolveSurfaceHit,
  tangentReference,
  toFacePlane,
} from "../src/faces";
import type { ConstructionFace, ConstructionNode } from "../src/types";

/** A square face, `size` across, centred on `origin` with the given normal. */
function face(overrides: Partial<ConstructionFace> = {}): ConstructionFace {
  const base: ConstructionFace = {
    id: "profile_0:cap-",
    stableId: null,
    ownerStableId: null,
    key: "cap-",
    kind: "cap",
    origin: [0, -0.6, 0],
    normal: [0, -1, 0],
    xAxis: [-1, 0, 0],
    yAxis: [0, 0, -1],
    polygon: [
      [0.5, -0.6, 0.5],
      [-0.5, -0.6, 0.5],
      [-0.5, -0.6, -0.5],
      [0.5, -0.6, -0.5],
    ],
    tolerance: 0.0019906,
    reference: { call: "cap", args: ["-"] },
    owner: { kind: "extrude", line: 113, variable: "sink" },
    usable: true,
    ...overrides,
  };
  return base;
}

const nodeWith = (...faces: ConstructionFace[]): ConstructionNode =>
  ({
    id: "profile_0",
    stableId: null,
    kind: "profile",
    name: "comb",
    line: 91,
    editable: true,
    edges: [],
    plane: null,
    faces,
    vertices: [],
    transform: null,
    spans: {},
    constraints: [],
    operators: [],
    material: null,
  }) as ConstructionNode;

describe("tolerance", () => {
  it("never runs tighter than twice the raymarcher's surface epsilon", () => {
    // The comb's own tolerance is 0.00199; twice the epsilon is 0.002.
    expect(faceTolerance(face())).toBe(2 * RAYMARCH_SURFACE_EPSILON);
    expect(faceTolerance(face({ tolerance: 0.05 }))).toBe(0.05);
  });
});

describe("polygon containment", () => {
  const square: [number, number][] = [
    [0, 0],
    [2, 0],
    [2, 2],
    [0, 2],
  ];

  it("is negative inside, positive outside, zero on the boundary", () => {
    expect(polygonDistance([1, 1], square)).toBeCloseTo(-1, 6);
    expect(polygonDistance([3, 1], square)).toBeCloseTo(1, 6);
    expect(polygonDistance([2, 1], square)).toBeCloseTo(0, 6);
  });

  it("projects a world point into the face's own axes", () => {
    // xAxis is −X and yAxis is −Z, so world (−0.25, −0.6, −0.25) is (0.25, 0.25).
    expect(toFacePlane(face(), [-0.25, -0.6, -0.25])).toEqual([0.25, 0.25]);
  });
});

describe("pickFace", () => {
  const nodes = [nodeWith(face(), face({
    id: "profile_0:cap+",
    key: "cap+",
    origin: [0, 0.6, 0],
    normal: [0, 1, 0],
    yAxis: [0, 0, 1],
    polygon: [
      [0.5, 0.6, 0.5],
      [-0.5, 0.6, 0.5],
      [-0.5, 0.6, -0.5],
      [0.5, 0.6, -0.5],
    ],
    reference: { call: "cap", args: ["+"] },
  }))];

  it("accepts a hit on the plane, inside the boundary, facing the same way", () => {
    const pick = pickFace(nodes, [0.2, -0.6, 0.1], [0, -1, 0]);
    expect(pick?.face.key).toBe("cap-");
    expect(pick?.nodeId).toBe("profile_0");
    expect(pick?.alignment).toBeCloseTo(1, 6);
  });

  it("rejects a hit further off the plane than the tolerance", () => {
    expect(pickFace(nodes, [0.2, -0.6015, 0.1], [0, -1, 0])).not.toBeNull();
    expect(pickFace(nodes, [0.2, -0.61, 0.1], [0, -1, 0])).toBeNull();
  });

  it("rejects a hit outside the boundary by more than the tolerance", () => {
    expect(pickFace(nodes, [0.5005, -0.6, 0], [0, -1, 0])).not.toBeNull();
    expect(pickFace(nodes, [0.7, -0.6, 0], [0, -1, 0])).toBeNull();
  });

  it("rejects a surface facing away from the face it landed on", () => {
    // The far cap of a thin part: on the plane, inside the boundary, and
    // pointing the other way. Without the alignment floor it would win.
    expect(pickFace(nodes, [0, -0.6, 0], [0, 1, 0])).toBeNull();
    expect(pickFace(nodes, [0, -0.6, 0], [0.6, -0.8, 0])?.face.key).toBe("cap-");
    expect(pickFace(nodes, [0, -0.6, 0], [0.9, -0.44, 0])).toBeNull();
  });

  it("takes the nearest plane when two faces both accept the hit", () => {
    const thin = [
      nodeWith(
        face({ id: "a", key: "a", origin: [0, -0.6, 0] }),
        face({ id: "b", key: "b", origin: [0, -0.6005, 0] }),
      ),
    ];
    expect(pickFace(thin, [0, -0.6, 0], [0, -1, 0])?.face.key).toBe("a");
  });

  it("breaks a tie on alignment", () => {
    const coplanar = [
      nodeWith(
        face({ id: "a", key: "a", normal: [0, -1, 0] }),
        face({ id: "b", key: "b", normal: [0.2, -0.9798, 0] }),
      ),
    ];
    expect(pickFace(coplanar, [0, -0.6, 0], [0, -1, 0])?.face.key).toBe("a");
  });

  it("answers null when the pointer is over no declared face at all", () => {
    expect(pickFace([], [0, 0, 0], [0, 0, 1])).toBeNull();
  });
});

describe("ray casting against declared faces", () => {
  const nodes = [nodeWith(face())];

  it("hits the face and orients its normal toward the viewer", () => {
    const hit = pickFaceSurface(nodes, { origin: [0, -3, 0], direction: [0, 1, 0] });
    expect(hit?.point[1]).toBeCloseTo(-0.6, 6);
    expect(hit?.normal).toEqual([0, -1, 0]);
  });

  it("misses when the ray passes outside the boundary", () => {
    expect(pickFaceSurface(nodes, { origin: [3, -3, 0], direction: [0, 1, 0] })).toBeNull();
  });

  it("prefers whichever of the two casts is nearer", () => {
    // A cylinder standing in front of the comb shadows it.
    const cylinder = {
      ...nodeWith(),
      id: "cylinder_1",
      kind: "cylinder" as const,
      line: 171,
      transform: {
        position: [0, -1.5, 0] as [number, number, number],
        rotation: [0, 0, 0] as [number, number, number],
        dimensions: { radius: 0.3, height: 0.3 },
        line: 171,
        call: "cylinder",
        positionArgument: "position",
        canRotate: true,
      },
    } as ConstructionNode;
    const hit = resolveSurfaceHit([...nodes, cylinder], {
      origin: [0, -3, 0],
      direction: [0, 1, 0],
    });
    expect(hit?.nodeId).toBe("cylinder_1");
  });
});

describe("writing a pick back into the source", () => {
  it("names a cap by its sign", () => {
    expect(faceReference(face())).toEqual({ kind: "cap", owner: 113, sign: "-" });
    expect(faceReference(face({ reference: { call: "cap", args: ["+"] } }))).toEqual({
      kind: "cap",
      owner: 113,
      sign: "+",
    });
  });

  it("names a side wall by its edge index and a primitive face by its key", () => {
    expect(faceReference(face({ kind: "side", reference: { call: "side", args: [3] } }))).toEqual({
      kind: "side",
      owner: 113,
      edge: 3,
    });
    expect(
      faceReference(face({ kind: "planar", reference: { call: "face", args: ["+x"] } })),
    ).toEqual({ kind: "face", owner: 113, key: "+x" });
  });

  it("refuses a face whose feature has no name in the source", () => {
    expect(faceReference(face({ usable: false }))).toBeNull();
    expect(faceReference(face({ owner: { kind: "extrude", line: 113, variable: null } }))).toBeNull();
  });

  it("falls back to a tangent plane at the picked point", () => {
    const nodes = [nodeWith(face())];
    expect(
      tangentReference(nodes, {
        nodeId: "profile_0",
        point: [0.1, -0.6, 0.2],
        normal: [0, -1, 0],
        t: 1,
      }),
    ).toEqual({ kind: "tangent", owner: 113, near: [0.1, -0.6, 0.2] });
  });

  it("re-resolves a target against the tree it is about to patch", () => {
    // The owner line moved (a sketch was inserted above it); the face id did
    // not, which is the whole reason the click carries an id rather than a
    // reference.
    const moved = [nodeWith(face({ owner: { kind: "extrude", line: 117, variable: "sink" } }))];
    expect(
      referenceFor(moved, { faceId: "profile_0:cap-", nodeId: "profile_0", near: [0, 0, 0] }),
    ).toEqual({ kind: "cap", owner: 117, sign: "-" });
    expect(
      referenceFor(moved, { faceId: "profile_0:gone", nodeId: "profile_0", near: [0, 0, 0] }),
    ).toBeNull();
    expect(
      referenceFor(moved, { faceId: null, nodeId: "profile_0", near: [1, 2, 3] }),
    ).toEqual({ kind: "tangent", owner: 117, near: [1, 2, 3] });
  });

  it("names a face the way the source would", () => {
    expect(faceLabel(face())).toBe("sink.cap('-')");
    expect(faceLabel(face({ reference: { call: "side", args: [2] } }))).toBe("sink.side(2)");
  });
});
