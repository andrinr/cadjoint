import { describe, expect, it } from "vitest";
import {
  DEFAULT_SKETCH_PLANE,
  pickSurfacePoint,
  quickPlaneEmission,
  quickPlaneNormal,
} from "../src/sketchPlanes";
import type { ConstructionNode, ConstructionTransform } from "../src/types";
import type { Ray } from "../src/viewer/math";

function solid(
  id: string,
  kind: "box" | "sphere" | "cylinder",
  position: [number, number, number],
  dimensions: Record<string, number | number[]>,
  rotation: [number, number, number] = [0, 0, 0],
): ConstructionNode {
  const transform: ConstructionTransform = {
    position,
    rotation,
    dimensions,
    line: 2,
    call: kind,
    positionArgument: "position",
    canRotate: true,
  };
  return {
    id,
    kind,
    name: id,
    line: 2,
    editable: true,
    edges: [],
    plane: null,
    vertices: [],
    transform,
    spans: {},
    constraints: [],
    operators: [],
    material: null,
  };
}

const ray = (origin: [number, number, number], direction: [number, number, number]): Ray => ({
  origin,
  direction,
});

describe("quick-pick plane emission", () => {
  it("XY is the default and emits no explicit normal", () => {
    expect(DEFAULT_SKETCH_PLANE).toBe("xy");
    const emission = quickPlaneEmission("xy", ray([0, 0, 5], [0, 0, -1]), 5);
    expect(emission.origin).toEqual([0, 0, 0]);
    expect(emission.normal).toBeNull();
  });

  it("YZ and ZX emit their world normals with the plane hit", () => {
    const yz = quickPlaneEmission("yz", ray([5, 1, 2], [-1, 0, 0]), 5);
    expect(yz.origin).toEqual([0, 1, 2]);
    expect(yz.normal).toEqual([1, 0, 0]);

    const zx = quickPlaneEmission("zx", ray([1, 5, 2], [0, -1, 0]), 5);
    expect(zx.origin).toEqual([1, 0, 2]);
    expect(zx.normal).toEqual([0, 1, 0]);
  });

  it("falls back to a point in front of the camera when edge-on", () => {
    const emission = quickPlaneEmission("xy", ray([0, 0, 1], [1, 0, 0]), 3);
    expect(emission.origin).toEqual([3, 0, 1]);
  });

  it("face has no quick normal", () => {
    expect(quickPlaneNormal("face")).toBeNull();
  });
});

describe("surface picking for on-face sketches", () => {
  it("hits a sphere and reports the outward normal at the hit point", () => {
    const sphere = solid("s", "sphere", [0, 0, 0], { radius: 0.5 });
    const hit = pickSurfacePoint([sphere], ray([0, 0, 5], [0, 0, -1]));
    expect(hit).not.toBeNull();
    expect(hit!.nodeId).toBe("s");
    expect(hit!.point[2]).toBeCloseTo(0.5, 5);
    expect(hit!.normal[2]).toBeCloseTo(1, 5);
  });

  it("hits the correct face of a box (half-extent semantics)", () => {
    const box = solid("b", "box", [1, 0, 0], { size: [0.5, 0.5, 0.5] });
    const hit = pickSurfacePoint([box], ray([1, 0, 5], [0, 0, -1]));
    expect(hit).not.toBeNull();
    expect(hit!.point).toEqual([1, 0, 0.5]);
    expect(hit!.normal).toEqual([0, 0, 1]);

    const side = pickSurfacePoint([box], ray([5, 0.2, 0.1], [-1, 0, 0]));
    expect(side!.point[0]).toBeCloseTo(1.5, 5);
    expect(side!.normal).toEqual([1, 0, 0]);
  });

  it("respects the primitive's rotation when computing the normal", () => {
    // Box yawed 45° about Z: the +X face normal turns with it.
    const box = solid("b", "box", [0, 0, 0], { size: [0.5, 0.5, 0.5] }, [0, 0, Math.PI / 4]);
    const hit = pickSurfacePoint([box], ray([5, 0, 0], [-1, 0, 0]));
    expect(hit).not.toBeNull();
    const [nx, ny, nz] = hit!.normal;
    expect(Math.hypot(nx, ny, nz)).toBeCloseTo(1, 5);
    expect(nz).toBeCloseTo(0, 5);
    // A face normal of the rotated box points along ±45° in XY.
    expect(Math.abs(Math.abs(nx) - Math.SQRT1_2)).toBeLessThan(1e-5);
    expect(Math.abs(Math.abs(ny) - Math.SQRT1_2)).toBeLessThan(1e-5);
  });

  it("hits a cylinder wall and its cap with the right normals", () => {
    const cylinder = solid("c", "cylinder", [0, 0, 0], { radius: 0.4, height: 0.5 });
    const wall = pickSurfacePoint([cylinder], ray([5, 0, 0.1], [-1, 0, 0]));
    expect(wall).not.toBeNull();
    expect(wall!.point[0]).toBeCloseTo(0.4, 5);
    expect(wall!.normal[0]).toBeCloseTo(1, 5);

    const cap = pickSurfacePoint([cylinder], ray([0.1, 0.1, 5], [0, 0, -1]));
    expect(cap).not.toBeNull();
    expect(cap!.point[2]).toBeCloseTo(0.5, 5);
    expect(cap!.normal).toEqual([0, 0, 1]);
  });

  it("chooses the nearest of several solids along the ray", () => {
    const near = solid("near", "sphere", [0, 0, 2], { radius: 0.5 });
    const far = solid("far", "sphere", [0, 0, -2], { radius: 0.5 });
    const hit = pickSurfacePoint([far, near], ray([0, 0, 5], [0, 0, -1]));
    expect(hit!.nodeId).toBe("near");
  });

  it("orients the emitted normal toward the viewer", () => {
    const sphere = solid("s", "sphere", [0, 0, 0], { radius: 0.5 });
    const hit = pickSurfacePoint([sphere], ray([0, 0, 5], [0, 0, -1]));
    // dot(normal, ray.direction) < 0 means the sketch faces the camera.
    expect(hit!.normal[2] * -1).toBeLessThan(0 + 1e-9);
    expect(hit!.normal[2]).toBeGreaterThan(0);
  });

  it("ignores profiles and misses cleanly", () => {
    const sphere = solid("s", "sphere", [0, 0, 0], { radius: 0.5 });
    expect(pickSurfacePoint([sphere], ray([5, 5, 5], [0, 0, -1]))).toBeNull();
    const profile = { ...sphere, kind: "profile" as const };
    expect(pickSurfacePoint([profile], ray([0, 0, 5], [0, 0, -1]))).toBeNull();
  });
});
