/**
 * The viewer's uniform blocks against the buffers they are written into.
 *
 * Each block's length is fixed by a WGSL struct; a pack function that grows or
 * shrinks by one float misaligns every field after it without any error.
 */

import { describe, expect, it } from "vitest";
import { DEFAULT_SLICE } from "../src/simulation";
import { DEFAULT_DISPLAY, DEFAULT_QUALITY, QUALITY_PRESETS } from "../src/viewer/display";
import { cameraPosition, type CameraState } from "../src/viewer/math";
import { GRATICULE_UNIFORM_SIZE } from "../src/viewer/pipelines";
import {
  packGraticuleUniforms,
  packOverlayUniforms,
  packSceneUniforms,
  packSimulationUniforms,
} from "../src/viewer/uniforms";

const camera: CameraState = { yaw: 0.6, pitch: 0.4, distance: 5, target: [1, 2, 3] };
const position = cameraPosition(camera);
const size = { width: 640, height: 480 };
const matrix = new Float32Array(16).map((_, index) => index + 1);

describe("uniform blocks", () => {
  it("packs the scene block as 7 × vec4", () => {
    const scene = packSceneUniforms({
      size,
      camera,
      position,
      display: DEFAULT_DISPLAY,
      quality: QUALITY_PRESETS[DEFAULT_QUALITY],
      sampleCount: 3,
      flags: 0,
    });
    expect(scene.byteLength).toBe(112);
    expect(Array.from(scene.slice(0, 2))).toEqual([640, 480]);
    expect(Array.from(scene.slice(8, 11))).toEqual([1, 2, 3]);
    expect(scene[20]).toBe(3);
  });

  it("packs the overlay block with its nudge and casing last", () => {
    const overlay = packOverlayUniforms(matrix, position, size, 0.5, 0.25, {
      forward: [0, 0, -1],
      orthographic: true,
    });
    expect(overlay.byteLength).toBe(128);
    expect(Array.from(overlay.slice(0, 16))).toEqual(Array.from(matrix));
    expect(Array.from(overlay.slice(24, 28))).toEqual([0.5, 0.25, 0, 0]);
    // The view direction last, flagged orthographic: a flat face shades flat.
    expect(Array.from(overlay.slice(28))).toEqual([0, 0, -1, 1]);
    const perspective = packOverlayUniforms(matrix, position, size, 0.5, 0.25, {
      forward: [0, 0, -1],
      orthographic: false,
    });
    expect(perspective[31]).toBe(0);
  });

  it("packs the simulation block, differing between passes only in params.z", () => {
    const frame = {
      matrix,
      position,
      clip: DEFAULT_SLICE,
      bounds: { min: [0, 0, 0], max: [1, 1, 1] },
      range: [0, 2] as [number, number],
      ramp: "quality" as const,
    };
    const base = packSimulationUniforms({ ...frame, pixel: 0 });
    const hover = packSimulationUniforms({ ...frame, pixel: 0.125 });
    expect(base.byteLength).toBe(112);
    expect(base[21]).toBe(0.5);
    expect(base[24]).toBe(1);
    const differing = Array.from(base.keys()).filter((index) => base[index] !== hover[index]);
    expect(differing).toEqual([22]);
  });

  it("packs the graticule block to its buffer's size", () => {
    const graticule = packGraticuleUniforms({
      size,
      cssWidth: 320,
      camera,
      position,
      projection: "perspective",
      emphasis: 1,
    });
    expect(graticule.byteLength).toBe(GRATICULE_UNIFORM_SIZE);
    // The line width is in CSS pixels, scaled into framebuffer pixels.
    expect(graticule[2]).toBe(2);
  });
});
