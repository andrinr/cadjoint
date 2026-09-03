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
import type { ShaderProgramPayload } from "./shaderProgram";

export interface Shaders {
  preview: string;
  path: string;
  present: string;
  /**
   * The parameter buffer these two scene shaders read, when they were built
   * in the uniform form.
   *
   * `null` (or absent) is the literal form, where every design parameter is
   * a constant in the source and any edit is a different module. With a
   * program the source is byte-identical across parameter values, which is
   * what lets `setShaders` recognise a values-only edit and answer it with a
   * `writeBuffer` instead of a recompile. A type-only import: this stays
   * plain data with no WebGPU in it.
   */
  program?: ShaderProgramPayload | null;
}

export interface QualityPreset {
  label: string;
  pixelBudget: number;
  maxRatio: number;
  bounces: number;
  shadowSamples: number;
  samples: number;
  /**
   * Sphere-tracing steps a primary ray may take.
   *
   * The tier used to stop at the JavaScript boundary: pixel budget, bounces,
   * shadow samples and the sample target were all read here, but the march
   * itself was a constant baked into the WGSL — 96 steps in the preview and
   * 160 in the path tracer — so "Ultra" meant two different marches on the
   * two sides of the same picture. It is a uniform now (`path_settings.w`),
   * and this is the one place the ladder is written.
   */
  marchSteps: number;
}

export const QUALITY_PRESETS: Record<string, QualityPreset> = {
  draft: {
    label: "Draft",
    pixelBudget: 320_000,
    maxRatio: 1,
    bounces: 3,
    shadowSamples: 1,
    samples: 128,
    marchSteps: 64,
  },
  high: {
    label: "High",
    pixelBudget: 900_000,
    maxRatio: 1.25,
    bounces: 6,
    shadowSamples: 2,
    samples: 512,
    marchSteps: 96,
  },
  ultra: {
    label: "Ultra",
    pixelBudget: 1_600_000,
    maxRatio: 2,
    bounces: 8,
    shadowSamples: 4,
    samples: 1024,
    marchSteps: 192,
  },
};

/**
 * The tier everything starts on.
 *
 * Ultra is the default everywhere — the renderer's own field, the three
 * shipped render presets, and the fallback a damaged persisted setting lands
 * on — so the first frame a user sees is the best one the machine can draw
 * rather than a tier they have to go and find.
 */
export const DEFAULT_QUALITY = "ultra";

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

/**
 * Which of the three ways of looking at the distance field is on.
 *
 * `solid` is the raymarched surface everything else in the app assumes.
 * `slice` and `gradient` cut the *field* open on a plane instead: one shows
 * the signed value, the other shows |∇f|, which is the diagnostic that says
 * where the field has stopped being a metric distance. They are views of the
 * same scene SDF the solid is traced from — the branch is inside the same
 * generated shader — so there is no second scene to fall out of step.
 */
export type SdfView = "solid" | "slice" | "gradient" | "normal" | "depth";

/**
 * The integer each view is written to the shader as.
 *
 * One table, in one place, because the string is the app's vocabulary and the
 * number is the shader's; anywhere else and the two drift.
 */
export const SDF_VIEW_CODE: Record<SdfView, number> = {
  solid: 0,
  slice: 1,
  gradient: 2,
  normal: 3,
  depth: 4,
};

/** The two views that cut a plane, as opposed to reshading the surface. */
export const isSliceView = (view: SdfView): boolean =>
  view === "slice" || view === "gradient";

/**
 * Half-width of the slab the slice fraction sweeps, in world units.
 *
 * The frontend never sees the scene's bounds — the SDF arrives as compiled
 * WGSL, not as geometry — so the plane sweeps a stated slab about the origin
 * rather than a bounding box it would have to guess. 2 world units is 2 m by
 * the repository's declared unit (see `graticule.ts`), which covers every
 * starter scene with room to spare, and the legend prints the plane's actual
 * coordinate so the number is never implied.
 */
export const SDF_SLICE_RANGE = 2;

/** World coordinate of the slice plane for a 0…1 fraction. */
export function slicePosition(fraction: number): number {
  return (Math.min(1, Math.max(0, fraction)) * 2 - 1) * SDF_SLICE_RANGE;
}

export interface DisplaySettings {
  projection: Projection;
  shadows: ShadowMode;
  reflections: boolean;
  flatShading: boolean;
  hideSolid: boolean;
  /** 0 disables x-ray; 1 is fully translucent. */
  xray: number;
  /** The ground grid on z = 0, its spacing readout, and the title block. */
  showGraticule: boolean;
  /**
   * Master switch over everything the app draws *about* the model rather than
   * of it: construction edges, sketch handles, the gizmo, constraint marks and
   * their dimension labels, and the boundary-condition preview. Off is the
   * presentation state — the solid, the field and the floor, nothing else.
   * The finer `showSketches` / `showConstraints` switches live under it.
   */
  showOverlays: boolean;
  showSketches: boolean;
  showMeshEdges: boolean;
  showMeshWireframe: boolean;
  showConstraints: boolean;
  showFixedConstraints: boolean;
  showDistanceConstraints: boolean;
  showConstraintValues: boolean;
  /**
   * Solid, a signed-distance slice, |∇f| on that slice, world normals, or
   * linear camera depth.
   */
  sdfView: SdfView;
  /** Slice plane normal: 0 = X, 1 = Y, 2 = Z. Matches the Results tab. */
  sdfAxis: 0 | 1 | 2;
  /** Where the plane sits in the slab, 0…1. */
  sdfFraction: number;
  /**
   * The level set the solid is traced at.
   *
   * `f = 0` is the surface; a positive offset is the field's outward offset
   * (a dilation), a negative one an erosion. It is applied inside the shader
   * to every `sdf()` read — primary rays, shadows and normals alike — so the
   * offset surface shades and casts like a real surface rather than a shell
   * drawn over the old one.
   */
  isoOffset: number;
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
  // On by default: without a floor there is nothing in the viewport that says
  // where anything is, and a drafting ground is what tells you how big the
  // part is as well as which way up it stands. The Studio preset turns it off
  // — a presentation render is not a drawing.
  showGraticule: true,
  showOverlays: true,
  showSketches: true,
  showMeshEdges: false,
  showMeshWireframe: false,
  showConstraints: true,
  showFixedConstraints: true,
  showDistanceConstraints: true,
  showConstraintValues: true,
  sdfView: "solid",
  sdfAxis: 0,
  sdfFraction: 0.5,
  isoOffset: 0,
};

/**
 * A standard view: where the camera stands, and what the readout calls it.
 *
 * `label` exists for one reason. The eight corner views are *directions* —
 * one for each octant — but the name for any of them is the same word, ISO,
 * because that is what a 1:1:1 direction is called whatever octant it points
 * into. The readout would otherwise invent eight names nobody uses.
 */
export interface ViewPreset {
  yaw: number;
  pitch: number;
  /** Overrides the key when the readout names this view. */
  label?: string;
}

/**
 * Camera direction from a world-space offset, in the app's yaw/pitch.
 *
 * The world is Z-up and `cameraPosition` builds its offset as azimuth about
 * +Z measured from −Y with elevation toward +Z (see `octant` in
 * `graticule.ts`), so this is that construction read backwards.
 */
export function anglesForDirection(
  x: number,
  y: number,
  z: number,
): { yaw: number; pitch: number } {
  const length = Math.hypot(x, y, z) || 1;
  // Straight up and straight down have no azimuth, and asking for one gets a
  // wrong answer rather than no answer: `atan2(0, -0)` is π, not 0, so the
  // two polar views would come back yawed half a turn and the whole scene
  // (and the cube's own TOP label) would arrive upside down.
  const polar = x === 0 && y === 0;
  return {
    yaw: polar ? 0 : Math.atan2(x, -y),
    pitch: Math.asin(z / length),
  };
}

/** Axis names in the order a view is spoken: front/back, left/right, top/bottom. */
const DIRECTION_WORDS: [number, string, string][] = [
  [1, "front", "back"],
  [0, "left", "right"],
  [2, "bottom", "top"],
];

/** `[1, -1, 1]` → `"front-right-top"`; a single axis → `"front"`. */
export function viewPresetName(direction: readonly [number, number, number]): string {
  const parts: string[] = [];
  for (const [axis, negative, positive] of DIRECTION_WORDS) {
    const sign = direction[axis];
    if (sign < 0) parts.push(negative);
    else if (sign > 0) parts.push(positive);
  }
  return parts.join("-");
}

/**
 * Orbit angles for every standard view, in radians.
 *
 * All twenty-six of them: the six axis views, the twelve edge views (45° about
 * one axis) and the eight corner views. They are generated from the sign
 * triples rather than written out, because the ViewCube derives the same
 * triples from its own geometry and the two must agree exactly — a hand-typed
 * table is where a corner button and its camera drift apart.
 *
 * `iso` is kept as an alias for the +X−Y+Z corner: it is the name the session
 * starts on and the one the render state persists.
 */
export const VIEW_PRESETS: Record<string, ViewPreset> = (() => {
  const presets: Record<string, ViewPreset> = {};
  const isoCorner = anglesForDirection(1, -1, 1);
  presets.iso = { ...isoCorner, label: "ISO" };
  for (const x of [-1, 0, 1]) {
    for (const y of [-1, 0, 1]) {
      for (const z of [-1, 0, 1]) {
        if (x === 0 && y === 0 && z === 0) continue;
        const rank = Math.abs(x) + Math.abs(y) + Math.abs(z);
        const angles = anglesForDirection(x, y, z);
        presets[viewPresetName([x, y, z])] = {
          ...angles,
          // A corner is an isometric direction whichever octant it is in;
          // an edge and an axis are named by the faces they sit between.
          ...(rank === 3 ? { label: "ISO" } : {}),
        };
      }
    }
  }
  return presets;
})();

/** True when two orbit angles name the same view (yaw wraps; poles spin). */
export function sameView(
  yaw: number,
  pitch: number,
  preset: { yaw: number; pitch: number },
  tolerance = 0.01,
): boolean {
  if (Math.abs(pitch - preset.pitch) >= tolerance) return false;
  // Looking straight up or down, yaw is a spin about the view axis and does
  // not change which view it is.
  if (Math.abs(Math.abs(preset.pitch) - Math.PI / 2) < 1e-6) return true;
  const delta = yaw - preset.yaw;
  return Math.abs(Math.atan2(Math.sin(delta), Math.cos(delta))) < tolerance;
}

/** The preset key the camera is currently sitting on, or null. */
export function matchViewPreset(yaw: number, pitch: number): string | null {
  for (const [name, preset] of Object.entries(VIEW_PRESETS)) {
    if (sameView(yaw, pitch, preset)) return name;
  }
  return null;
}

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
