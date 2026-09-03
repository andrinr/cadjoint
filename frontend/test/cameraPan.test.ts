/**
 * Panning moves the orbit target in the camera's own screen plane.
 *
 * The property that matters is geometric, not numeric: whatever the yaw and
 * pitch, a pan must slide the target across the screen and never towards or
 * away from the eye, and a horizontal drag must stay level in a Z-up world.
 * A second, hand-rolled basis in `panCamera` once broke exactly this — it was
 * written Y-up while the camera is Z-up, so a sideways drag climbed out of the
 * ground plane.  These tests fail on that basis and pass on the shared one.
 */
import { describe, expect, it } from "vitest";
import { panCamera } from "../src/components/viewer/camera";
import { cameraPosition, subtract, type CameraState, type Vec3 } from "../src/viewer/math";

const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (v: Vec3) => Math.sqrt(dot(v, v));

const cameraAt = (yaw: number, pitch: number): CameraState => ({
  yaw,
  pitch,
  distance: 6,
  target: [0.3, -0.2, 0.1],
});

/** Yaw/pitch pairs either side of the poles and the axes. */
const VIEWS: ReadonlyArray<readonly [string, number, number]> = [
  ["front", 0, 0],
  ["right", Math.PI / 2, 0],
  ["back", Math.PI, 0],
  ["iso", Math.PI / 4, 0.6],
  ["steep", -1.1, 1.3],
  ["below", 2.2, -0.9],
];

describe("panCamera", () => {
  it.each(VIEWS)("slides across the view, never along it (%s)", (_name, yaw, pitch) => {
    const camera = cameraAt(yaw, pitch);
    const view = subtract(camera.target, cameraPosition(camera));
    for (const [dx, dy] of [
      [120, 0],
      [0, 90],
      [-70, 40],
    ]) {
      const moved = subtract(panCamera(camera, dx, dy).target, camera.target);
      expect(norm(moved)).toBeGreaterThan(0);
      // Perpendicular to the view direction: a pan changes what is on screen,
      // never how far away it is.
      expect(Math.abs(dot(moved, view)) / (norm(moved) * norm(view))).toBeLessThan(1e-9);
    }
  });

  it.each(VIEWS)("keeps a horizontal drag level in Z (%s)", (_name, yaw, pitch) => {
    const camera = cameraAt(yaw, pitch);
    const moved = subtract(panCamera(camera, 100, 0).target, camera.target);
    // Screen-right of a Z-up orbit camera is horizontal at every pitch, so a
    // sideways drag must not lift the target off its height.
    expect(Math.abs(moved[2])).toBeLessThan(1e-9);
    expect(Math.hypot(moved[0], moved[1])).toBeGreaterThan(0);
  });

  it("lifts the target in +Z when dragged up from a level view", () => {
    const camera = cameraAt(0, 0);
    // Screen y grows downward, so a negative delta is an upward drag; from a
    // level view the screen's up *is* world +Z.
    const moved = subtract(panCamera(camera, 0, -50).target, camera.target);
    expect(moved[2]).toBeLessThan(0);
    expect(Math.hypot(moved[0], moved[1])).toBeLessThan(1e-9);
  });

  it("moves the target opposite the drag, so the scene follows the pointer", () => {
    const camera = cameraAt(0, 0);
    // Front view: screen-right is world +X. Dragging right must carry the
    // scene right, which means the target goes the other way.
    const moved = subtract(panCamera(camera, 100, 0).target, camera.target);
    expect(moved[0]).toBeLessThan(0);
  });

  it("scales with distance, so the pan feels the same at any zoom", () => {
    const near = { ...cameraAt(0.4, 0.3), distance: 2 };
    const far = { ...cameraAt(0.4, 0.3), distance: 20 };
    const nearMove = norm(subtract(panCamera(near, 60, 30).target, near.target));
    const farMove = norm(subtract(panCamera(far, 60, 30).target, far.target));
    expect(farMove / nearMove).toBeCloseTo(10, 6);
  });
});
