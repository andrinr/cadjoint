/**
 * The viewer's uniform blocks, packed as the shaders read them.
 *
 * Pure: each function returns the bytes for one `writeBuffer`, so the layouts
 * can be checked without a device. The scene block mirrors `Uniforms` in
 * `cadjoint/viewer/_webgpu.py`; the overlay, simulation and graticule blocks
 * mirror the structs in `overlay.wgsl`, `simulation.wgsl` and
 * `graticule.wgsl`.
 */

import { slicePlane, type SliceState } from "../simulation";
import { CHROME, hexToRgb } from "../tokens";
import {
  BACKGROUND_RADIANCE,
  GRATICULE,
  HANDLE_RADIUS_PX,
  KEY_LIGHT_INTENSITY,
  LINE_WIDTH_PX,
} from "./frameConstants";
import {
  SDF_VIEW_CODE,
  effectiveMarchSteps,
  slicePosition,
  type DisplaySettings,
  type QualityPreset,
} from "./display";
import { GRID_ALPHA, GRID_FADE, GRID_MAJOR_EVERY, gridSpacing } from "./graticule";
import { FOV_SCALE, cameraBasis, orthoHeightFor, type CameraState, type Vec3 } from "./math";

/** The framebuffer a frame is drawn into, in device pixels. */
export interface FrameSize {
  width: number;
  height: number;
}

/** The scene block: 7 × vec4, `Uniforms` in `_webgpu.py`. */
export function packSceneUniforms(frame: {
  size: FrameSize;
  camera: CameraState;
  position: Vec3;
  display: DisplaySettings;
  quality: QualityPreset;
  sampleCount: number;
  flags: number;
}): Float32Array<ArrayBuffer> {
  const { size, camera, position, display, quality } = frame;
  // The six trailing scalars of the first five vec4s are the SDF views and
  // the march budget; `Uniforms` in `cadjoint/viewer/_webgpu.py` documents
  // the packing and why they ride here rather than in an eighth vec4.
  const { sdfView, sdfAxis, sdfFraction, isoOffset } = display;
  const sdfMode = SDF_VIEW_CODE[sdfView] ?? 0;
  return new Float32Array([
    size.width, size.height, sdfMode, sdfAxis,
    position[0], position[1], position[2], slicePosition(sdfFraction),
    camera.target[0], camera.target[1], camera.target[2], isoOffset,
    0.55, 0.35, 0.8, KEY_LIGHT_INTENSITY,
    BACKGROUND_RADIANCE[0],
    BACKGROUND_RADIANCE[1],
    BACKGROUND_RADIANCE[2],
    // The slice's contours are ruled at the floor grid's own spacing, so a
    // contour interval is the same stated number the GRID readout is
    // showing and the two annotations cannot disagree.
    gridSpacing(camera.distance),
    frame.sampleCount,
    quality.bounces,
    quality.shadowSamples,
    // The tier's budget unless the display settings override it; both are
    // rendering choices that ride in the uniform, so changing either is a
    // buffer write and a redraw rather than a recompile.
    effectiveMarchSteps(display, quality),
    display.projection === "orthographic" ? 1 : 0,
    orthoHeightFor(camera.distance),
    frame.flags,
    display.xray,
  ]);
}

/**
 * The overlay block: the view-projection, the eye, the stroke metrics, the
 * depth nudge and casing that tell construction strokes from mesh edges, and
 * the view direction.
 *
 * The direction is the camera's forward axis, with `w` 1 under an
 * orthographic projection: there every ray is parallel to it, and a surface
 * shaded by the vector from the eye *point* instead got a radial gradient
 * across a flat plane, brightest where the plane passes nearest the eye.
 */
export function packOverlayUniforms(
  matrix: Float32Array,
  position: Vec3,
  size: FrameSize,
  depthNudge: number,
  casing: number,
  view: { forward: Vec3; orthographic: boolean },
): Float32Array<ArrayBuffer> {
  const overlay = new Float32Array(OVERLAY_UNIFORM_FLOATS);
  overlay.set(matrix, 0);
  overlay.set([position[0], position[1], position[2], 0], 16);
  overlay.set([size.width, size.height, LINE_WIDTH_PX, HANDLE_RADIUS_PX], 20);
  overlay.set([depthNudge, casing, 0, 0], 24);
  overlay.set([view.forward[0], view.forward[1], view.forward[2], view.orthographic ? 1 : 0], 28);
  return overlay;
}

/** The overlay block's size: view-projection, then four vec4s. */
export const OVERLAY_UNIFORM_FLOATS = 32;

/**
 * The FEM surface block: the view-projection, the clip plane, the field
 * range, and the ramp choice with the eye.
 *
 * `pixel` is params.z: 0 for the surface itself, one framebuffer pixel in
 * NDC for the hover-outline pass, the step `vs_sim_edge` shifts each copy of
 * the outline by.
 */
export function packSimulationUniforms(frame: {
  matrix: Float32Array;
  position: Vec3;
  clip: SliceState;
  bounds: { min: readonly number[]; max: readonly number[] };
  range: readonly [number, number];
  ramp: "field" | "quality";
  pixel: number;
}): Float32Array<ArrayBuffer> {
  const { position, clip } = frame;
  const { normal, offset } = slicePlane(clip, frame.bounds);
  const [low, high] = frame.range;
  const span = high - low;
  const inverseRange = span > 1e-12 ? 1 / span : 0;
  const sim = new Float32Array(28);
  sim.set(frame.matrix, 0);
  sim.set([normal[0], normal[1], normal[2], offset], 16);
  sim.set([low, inverseRange, frame.pixel, clip.enabled ? 1 : 0], 20);
  // extra.yzw is the camera position: the FEM surface builds its facet
  // normal from screen-space derivatives, and on the paper ground it needs
  // the eye vector both to orient that normal and to darken the
  // silhouette, which is what separates a hot field from the background.
  sim.set([frame.ramp === "quality" ? 1 : 0, position[0], position[1], position[2]], 24);
  return sim;
}

/**
 * The camera the ground grid raycasts with, and the grid's own metrics.
 *
 * The basis is recomputed here rather than passed through the view matrix
 * because the shader needs the same three vectors `primary_ray` uses, not a
 * matrix: it reconstructs the ray for its fragment and intersects the floor
 * with it. The line width is the one length in CSS pixels, scaled by the
 * ratio the framebuffer was, so a hairline is a hairline at every quality
 * tier.
 */
export function packGraticuleUniforms(frame: {
  size: FrameSize;
  /** The canvas's CSS width, for the CSS-to-framebuffer pixel ratio. */
  cssWidth: number;
  camera: CameraState;
  position: Vec3;
  projection: DisplaySettings["projection"];
  /** One dimmer for the whole plane, so the three weights keep their order. */
  emphasis: number;
}): Float32Array<ArrayBuffer> {
  const { size, camera, position, emphasis } = frame;
  const scale = size.width / Math.max(frame.cssWidth, 1);
  const px = (css: number) => css * scale;
  const tone = (name: "graticule-line" | "graticule-axis") => hexToRgb(CHROME[name]);
  const [lineR, lineG, lineB] = tone("graticule-line");
  const [axisR, axisG, axisB] = tone("graticule-axis");
  const { forward, right, up } = cameraBasis(position, camera.target);
  const distance = camera.distance;
  const target = camera.target;
  return new Float32Array([
    size.width,
    size.height,
    Math.max(1, px(GRATICULE.lineWidth)),
    frame.projection === "orthographic" ? 1 : 0,
    position[0], position[1], position[2], orthoHeightFor(distance),
    right[0], right[1], right[2],
    size.width / Math.max(size.height, 1),
    up[0], up[1], up[2], gridSpacing(distance),
    forward[0], forward[1], forward[2], FOV_SCALE,
    lineR, lineG, lineB, GRID_ALPHA.minor * emphasis,
    lineR, lineG, lineB, GRID_ALPHA.major * emphasis,
    axisR, axisG, axisB, GRID_ALPHA.axis * emphasis,
    distance * GRID_FADE.start,
    distance * GRID_FADE.end,
    GRATICULE.axisWidth,
    GRID_MAJOR_EVERY,
    target[0], target[1], target[2], 0,
  ]);
}
