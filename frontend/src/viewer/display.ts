/**
 * What the viewer is asked to show, separate from how it draws it.
 *
 * These are the knobs the render panel and the presets write and the renderer
 * reads: projection, shading and shadow mode, which overlays are on, and the
 * quality tier the path tracer budgets against. They are plain data with no
 * WebGPU in them, which is why they live apart from the renderer — the panel,
 * the preset store and its tests can import them without touching a device.
 *
 * `DISPLAY` and `displayFlags` are the one place the flag bits are spelled
 * out; they must stay in step with the DISPLAY_* constants in `_webgpu.py`,
 * because the shader reads the same packed integer.
 */

import type { Projection } from "./math";

export interface Shaders {
  preview: string;
  path: string;
  present: string;
}

export interface QualityPreset {
  label: string;
  pixelBudget: number;
  maxRatio: number;
  bounces: number;
  shadowSamples: number;
  samples: number;
}

export const QUALITY_PRESETS: Record<string, QualityPreset> = {
  draft: {
    label: "Draft",
    pixelBudget: 320_000,
    maxRatio: 1,
    bounces: 3,
    shadowSamples: 1,
    samples: 128,
  },
  high: {
    label: "High",
    pixelBudget: 900_000,
    maxRatio: 1.25,
    bounces: 6,
    shadowSamples: 2,
    samples: 512,
  },
  ultra: {
    label: "Ultra",
    pixelBudget: 1_600_000,
    maxRatio: 2,
    bounces: 8,
    shadowSamples: 4,
    samples: 1024,
  },
};

/** Display flag bits, matching the DISPLAY_* constants in _webgpu.py. */
export const DISPLAY = {
  shadows: 1,
  reflections: 2,
  flat: 4,
  hideSolid: 8,
  hardShadows: 16,
} as const;

/** Off, one crisp occlusion ray, or a penumbra. */
export type ShadowMode = "off" | "hard" | "soft";

export interface DisplaySettings {
  projection: Projection;
  shadows: ShadowMode;
  reflections: boolean;
  flatShading: boolean;
  hideSolid: boolean;
  /** 0 disables x-ray; 1 is fully translucent. */
  xray: number;
  /** The instrument faceplate behind the scene, its gain readout, title block. */
  showGraticule: boolean;
  showSketches: boolean;
  showMeshEdges: boolean;
  showMeshWireframe: boolean;
  showConstraints: boolean;
  showFixedConstraints: boolean;
  showDistanceConstraints: boolean;
  showConstraintValues: boolean;
}

export const DEFAULT_DISPLAY: DisplaySettings = {
  projection: "orthographic",
  // Flat shading with crisp shadows reads like a working drawing, which is
  // what you want while modelling; full shading is a click away.
  shadows: "hard",
  reflections: true,
  flatShading: true,
  hideSolid: false,
  xray: 1,
  // On by default: the viewport had no spatial scale on it at all, and a
  // drafting ground is what tells you how big the part you are looking at is.
  // The Studio preset turns it off — a presentation render is not a drawing.
  showGraticule: true,
  showSketches: true,
  showMeshEdges: false,
  showMeshWireframe: false,
  showConstraints: true,
  showFixedConstraints: true,
  showDistanceConstraints: true,
  showConstraintValues: true,
};

/** Orbit angles for the standard views, in radians. */
export const VIEW_PRESETS: Record<string, { yaw: number; pitch: number }> = {
  iso: { yaw: 0.75, pitch: 0.32 },
  front: { yaw: 0, pitch: 0 },
  back: { yaw: Math.PI, pitch: 0 },
  right: { yaw: Math.PI / 2, pitch: 0 },
  left: { yaw: -Math.PI / 2, pitch: 0 },
  top: { yaw: 0, pitch: Math.PI / 2 },
  bottom: { yaw: 0, pitch: -Math.PI / 2 },
};

/** Pack the display settings into the integer the scene shader reads. */
export function displayFlags(
  display: DisplaySettings,
  simulationActive: boolean,
): number {
  const { shadows, reflections, flatShading, hideSolid } = display;
  return (
    (shadows === "off" ? 0 : DISPLAY.shadows) |
    (shadows === "hard" ? DISPLAY.hardShadows : 0) |
    (reflections ? DISPLAY.reflections : 0) |
    (flatShading ? DISPLAY.flat : 0) |
    // While the simulation surface is shown the raymarched solid is hidden:
    // the preview pass still supplies the environment background and clears
    // depth to 1, and the FEM mesh depth-tests into that frame.
    (hideSolid || simulationActive ? DISPLAY.hideSolid : 0)
  );
}
