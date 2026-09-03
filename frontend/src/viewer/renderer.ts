/**
 * WebGPU renderer for the playground viewer.
 *
 * Draws the compiled SDF scene (raymarched preview or progressive path trace)
 * and the construction-tree overlay on top of it. The preview pass writes
 * `frag_depth` from the ray hit position, so construction edges and handles
 * are depth-tested against the solid. Transform controls use a separate
 * always-visible pass so the selected object cannot hide its own gizmo.
 *
 * Ported from the previous inline playground script, with the depth attachment,
 * the overlay pipelines, and pan support added.
 *
 * What is left here is the device and the frame: adapter setup, the GPU
 * buffers and textures, and the passes that draw them. The declarative parts
 * moved out — the settings the panels write live in `display.ts`, the vertex
 * packing in `overlayGeometry.ts`, and the pipeline descriptors in
 * `pipelines.ts` — and are re-exported below so `viewer/renderer` stays the
 * one import path callers know.
 */

import {
  DEFAULT_SLICE,
  meshBounds,
  slicePlane,
  type SliceState,
} from "../simulation";
import type {
  ConstructionFace,
  ConstructionNode,
  GizmoMode,
  MeshEdgePayload,
  Selection,
  SimulationMeshPayload,
} from "../types";
import { gizmoScale, type AxisIndex } from "./gizmo";
import {
  cameraBasis,
  cameraPosition,
  FOV_SCALE,
  orthoHeightFor,
  viewProjection,
  type CameraState,
  type Vec3,
  type View,
} from "./math";
import {
  DEFAULT_DISPLAY,
  DEFAULT_QUALITY,
  QUALITY_PRESETS,
  SDF_VIEW_CODE,
  VIEW_PRESETS,
  displayFlags,
  slicePosition,
  type DisplaySettings,
  type QualityPreset,
  type Shaders,
} from "./display";
import {
  EDGE_STRIDE,
  FACE_STRIDE,
  GIZMO_STRIDE,
  HANDLE_STRIDE,
  packConstructionOverlay,
  packFaceHighlight,
  packGizmoInstances,
  packMeshEdgeInstances,
} from "./overlayGeometry";
import {
  DEPTH_FORMAT,
  GRATICULE_UNIFORM_SIZE,
  createGraticulePipeline,
  createOverlayPipelines,
  type DepthPair,
  createSimulationPipelines,
  sharedLayout,
} from "./pipelines";
import {
  PARAMETER_SLOT_BYTES,
  ShaderModuleCache,
  packParameters,
  sameLayout,
  type ShaderProgramPayload,
} from "./shaderProgram";
import { GRID_ALPHA, GRID_FADE, GRID_MAJOR_EVERY, gridSpacing } from "./graticule";

/** What the shader path has done this session. See `Renderer.shaderStats`. */
export interface ShaderStats {
  /** Render pipelines created since construction. */
  pipelineBuilds: number;
  /** Parameter uploads that stood in for a pipeline rebuild. */
  parameterUploads: number;
  /** Whether the installed scene reads its parameters from a buffer. */
  hasParameterBuffer: boolean;
  /** Shader modules served from cache. */
  hits: number;
  /** Shader modules that had to be compiled. */
  misses: number;
}
import { CHROME, hexToRgb } from "../tokens";

export {
  DEFAULT_DISPLAY,
  DEFAULT_QUALITY,
  DISPLAY,
  QUALITY_PRESETS,
  SDF_SLICE_RANGE,
  SDF_VIEW_CODE,
  VIEW_PRESETS,
  displayFlags,
  isSliceView,
  matchViewPreset,
  sameView,
  slicePosition,
  type DisplaySettings,
  type QualityPreset,
  type SdfView,
  type ShadowMode,
  type Shaders,
  type ViewPreset,
} from "./display";

/**
 * The viewport's paper ground, `#e6e6e9`, as the swapchain sees it.
 *
 * The colour target is `bgra8unorm` and the SDF shaders gamma-encode
 * themselves, so this is a *display-encoded* value: it is what a `loadOp:
 * "clear"` writes, and it is what the fullscreen preview pass has to land on
 * for a ray that hits nothing. Mirrored as `VIEWPORT_BACKGROUND` in
 * `src/simColors.ts`, which is where every overlay measures itself against it.
 */
const BACKGROUND: GPUColorDict = { r: 0.902, g: 0.902, b: 0.914, a: 1 };

/**
 * The same paper, as *linear radiance* for `u.bg_color`.
 *
 * `environment_radiance` in `cadjoint/viewer/_webgpu.py` returns this
 * unchanged and the result goes through ACES then gamma 2.2, so handing it the
 * display value would land the background near `#c4c4c8` — visibly grey
 * against the chrome. These are the pre-images of the three channels above
 * under `pow(aces(x), 1/2.2)`, solved once here rather than per fragment.
 */
const BACKGROUND_RADIANCE: readonly [number, number, number] = [0.9684, 0.9684, 1.0819];

/**
 * Key-light intensity for the PBR and path-traced modes.
 *
 * The part is lit against paper, so the key cannot run hot: at 3.0 every lit
 * face landed in the tone map's shoulder and converged with the background.
 * 1.5 keeps the lit faces below the ground and lets the shading, not the
 * exposure, carry the form.
 */
const KEY_LIGHT_INTENSITY = 1.5;

/** Fraction of the distance to the camera that construction overlays are pulled forward. */
const DEPTH_NUDGE = 0.004;
const LINE_WIDTH_PX = 2.4;
const HANDLE_RADIUS_PX = 6.5;

/**
 * The ground grid's metrics.
 *
 * The line width is in CSS pixels (the renderer scales it into framebuffer
 * pixels, which are fewer than CSS pixels under the quality budget's
 * resolution cap): 1px, because this is furniture and a 2px grid on paper
 * reads as a table. The fades are multiples of the orbit distance, so the
 * plane dissolves at the same *apparent* place however far out you are.
 */
const GRATICULE = {
  lineWidth: 1,
  /** The two ground axes carry more weight than a grid line, and no more. */
  axisWidth: 1.6,
} as const;

export interface RendererCallbacks {
  onStatus?: (kind: string, text: string) => void;
  onError?: (message: string) => void;
  onReady?: () => void;
}

function timeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  return Promise.race([
    promise,
    new Promise<T>((_, reject) =>
      setTimeout(() => reject(new Error(`${label} timed out after ${ms} ms.`)), ms),
    ),
  ]);
}

export class Renderer {
  private canvas!: HTMLCanvasElement;
  private device: GPUDevice | null = null;
  private context: GPUCanvasContext | null = null;
  private format: GPUTextureFormat = "bgra8unorm";
  private adapterLabel = "WebGPU";

  private uniformBuffer!: GPUBuffer;
  private viewBuffer!: GPUBuffer;
  private overlayBuffer!: GPUBuffer;
  private meshOverlayBuffer!: GPUBuffer;
  private graticuleBuffer!: GPUBuffer;

  private previewPipeline: GPURenderPipeline | null = null;
  private previewDepthPipeline: GPURenderPipeline | null = null;
  private previewBindGroup: GPUBindGroup | null = null;
  private pathPipeline: GPURenderPipeline | null = null;

  // ── The scene's parameters, and what they save ────────────────────────
  // A scene shader built in the uniform form reads its design parameters
  // out of a `@group(3)` buffer instead of carrying them as constants, so
  // the source does not change when a value does. Everything here exists
  // to notice that and answer a parameter edit with a `writeBuffer`
  // rather than a shader module and four pipelines.
  /** Compiled modules held by source, so an undo lands on a hit. */
  private shaderModules = new ShaderModuleCache();
  /** The sources currently installed, or null before the first compile. */
  private installedShaders: Shaders | null = null;
  /** The scene's uniform contract; null in the literal form. */
  private program: ShaderProgramPayload | null = null;
  private parameterBuffer: GPUBuffer | null = null;
  private parameterBufferBytes = 0;
  private parameterLayout: GPUBindGroupLayout | null = null;
  private previewParameterGroup: GPUBindGroup | null = null;
  private pathParameterGroup: GPUBindGroup | null = null;
  /**
   * Values layered over the program's own, keyed by parameter name.
   *
   * A handle or gizmo drag writes these at frame rate while the server
   * patch that makes them permanent is still in flight; the compile that
   * lands afterwards clears them by installing the same numbers.
   */
  private parameterOverrides: Record<string, readonly number[]> | null = null;
  /**
   * Render pipelines created since the renderer was constructed.
   *
   * Public because it is the thing the drag test asserts about: a drag
   * that only moves parameter values must not advance this at all.
   */
  pipelineBuilds = 0;
  /** Parameter uploads that stood in for a pipeline rebuild. */
  parameterUploads = 0;
  private presentPipeline: GPURenderPipeline | null = null;
  private edgePipeline: DepthPair | null = null;
  private facePipeline: DepthPair | null = null;
  private handlePipeline: DepthPair | null = null;
  private gizmoEdgePipeline: GPURenderPipeline | null = null;
  private gizmoArrowPipeline: GPURenderPipeline | null = null;
  private gizmoScalePipeline: GPURenderPipeline | null = null;
  private overlayBindGroup: GPUBindGroup | null = null;
  private meshOverlayBindGroup: GPUBindGroup | null = null;
  /**
   * How firmly the floor is printed, 0…1.
   *
   * Set by the viewer pane: sketching on a plane that is not the floor steps
   * the grid back so the sketch's own plane is the reference. Everything else
   * about the grid is fixed by the token layer.
   */
  groundEmphasis = 1;

  private graticulePipeline: GPURenderPipeline | null = null;
  private graticuleBindGroup: GPUBindGroup | null = null;

  private depthTexture: GPUTexture | null = null;
  private accumulation: GPUTexture[] = [];
  private pathBindGroups: GPUBindGroup[] = [];
  private presentBindGroups: GPUBindGroup[] = [];
  private accumulationWidth = 0;
  private accumulationHeight = 0;
  private readIndex = 0;
  private sampleCount = 0;

  private edgeBuffer: GPUBuffer | null = null;
  private handleBuffer: GPUBuffer | null = null;
  private edgeCapacity = 0;
  private handleCapacity = 0;
  private edgeCount = 0;
  private handleCount = 0;
  private gizmoBuffer: GPUBuffer | null = null;
  private gizmoCapacity = 0;
  private gizmoCount = 0;
  private visibleGizmoMode: GizmoMode = "translate";
  // The face under the pointer while face picking is armed: a filled wash
  // (its own triangle pipeline, since nothing else in the overlay is a
  // surface) plus a hairline outline drawn through the edge pipeline.
  private faceHighlight: ConstructionFace | null = null;
  private faceFillBuffer: GPUBuffer | null = null;
  private faceFillCapacity = 0;
  private faceFillVertices = 0;
  private faceOutlineBuffer: GPUBuffer | null = null;
  private faceOutlineCapacity = 0;
  private faceOutlineCount = 0;
  private meshEdgeBuffer: GPUBuffer | null = null;
  private meshEdgeCapacity = 0;
  private meshWireCount = 0;
  private meshSharpCount = 0;
  private meshEdges: MeshEdgePayload | null = null;

  // FEM simulation surface: an indexed triangle mesh with a scalar per
  // vertex, drawn instead of the raymarched solid while simulation display
  // is on. Two bind groups share one pipeline: the base pass and the
  // hover-highlight pass differ only in the tint uniform.
  private simPipeline: GPURenderPipeline | null = null;
  private simBindGroup: GPUBindGroup | null = null;
  private simHighlightBindGroup: GPUBindGroup | null = null;
  private simUniformBuffer: GPUBuffer | null = null;
  private simHighlightUniformBuffer: GPUBuffer | null = null;
  private simVertexBuffer: GPUBuffer | null = null;
  private simIndexBuffer: GPUBuffer | null = null;
  private simIndexCount = 0;
  private simRange: [number, number] = [0, 0];
  private simBounds = { min: [0, 0, 0], max: [0, 0, 0] };
  private simHighlight: { start: number; count: number } | null = null;
  private simClip: SliceState = { ...DEFAULT_SLICE };
  private _simulationActive = false;
  // Current surface payload plus the display overrides layered over it:
  // an alternative nodal field, warped (deformed) positions, and a per-
  // vertex highlight tint for BC previews. Any change re-interleaves the
  // vertex buffer — tens of thousands of vertices, cheap enough per edit.
  private simPayload: SimulationMeshPayload | null = null;
  private simScalarOverride: readonly number[] | null = null;
  private simPositionOverride: readonly number[] | null = null;
  private simOverlay: Float32Array | null = null;
  // Element-edge hairlines over the surface (payload.edges index pairs).
  private simEdgePipeline: GPURenderPipeline | null = null;
  private simEdgeIndexBuffer: GPUBuffer | null = null;
  private simEdgeIndexCount = 0;
  private _simulationEdgesVisible = false;
  /** Which ramp fs_sim applies: solved fields vs mesh quality. */
  simulationRamp: "field" | "quality" = "field";

  private shaderRevision = 0;
  private framePending = false;
  private initError = "";

  // The session opens on the +X−Y+Z corner, at the exact 1:1:1 direction
  // rather than near it: the readout says ISO because the camera is on an
  // isometric direction, not because someone once typed angles that looked
  // about right. `VIEW_PRESETS.iso` is the same pair of numbers.
  camera: CameraState = {
    ...VIEW_PRESETS.iso,
    distance: 4.6,
    target: [0, 0, 0],
  };
  display: DisplaySettings = { ...DEFAULT_DISPLAY };
  quality: QualityPreset = QUALITY_PRESETS[DEFAULT_QUALITY];
  pathTracing = false;
  pathReady = false;
  interacting = false;

  private profiles: readonly ConstructionNode[] = [];
  private _gizmoMode: GizmoMode = "translate";
  private _gizmoAxis: AxisIndex | null = null;
  private selection: Selection | null = null;
  private hover: Selection | null = null;

  constructor(private callbacks: RendererCallbacks = {}) {}

  get rendererLabel(): string {
    return this.adapterLabel;
  }

  get unavailableReason(): string {
    return this.initError;
  }

  get currentSampleCount(): number {
    return this.sampleCount;
  }

  get gizmoMode(): GizmoMode {
    return this._gizmoMode;
  }

  set gizmoMode(mode: GizmoMode) {
    if (mode === this._gizmoMode) return;
    this._gizmoMode = mode;
    this.uploadOverlay();
    this.invalidate();
  }

  get gizmoAxis(): AxisIndex | null {
    return this._gizmoAxis;
  }

  set gizmoAxis(axis: AxisIndex | null) {
    if (axis === this._gizmoAxis) return;
    this._gizmoAxis = axis;
    this.uploadOverlay();
    this.scheduleRender();
  }

  /** Requested mode, reduced to the transforms a selected node supports. */
  gizmoModeFor(node: ConstructionNode): GizmoMode {
    if (this.gizmoMode === "rotate" && !node.transform?.canRotate) return "translate";
    if (this.gizmoMode === "scale" && node.kind === "profile") return "translate";
    return this.gizmoMode;
  }

  /** Framebuffer size, used by hit testing to match projected pixels. */
  get viewport(): { width: number; height: number } {
    return { width: this.canvas?.width ?? 1, height: this.canvas?.height ?? 1 };
  }

  get cameraPosition(): Vec3 {
    return cameraPosition(this.camera);
  }

  /** The view descriptor picking and overlay projection must agree on. */
  get view(): View {
    return {
      position: cameraPosition(this.camera),
      target: this.camera.target,
      width: this.viewport.width,
      height: this.viewport.height,
      projection: this.display.projection,
      orthoHeight: orthoHeightFor(this.camera.distance),
    };
  }

  /**
   * Point the camera at a standard view.
   *
   * Direction only. A preset used to also force the projection — orthographic
   * for the axis views, perspective for "iso" — which quietly conflated two
   * different things: *isometric* names a direction (a 1:1:1 line through the
   * scene), *orthographic* names a projection (parallel rays). You can look
   * down an isometric direction in either projection, and which one you are in
   * is the toggle beside the cube, not a side effect of clicking a corner.
   *
   * And it does not take the zoom with it either. A version of this snapped
   * the distance to the nearest rung of the floor grid's 1-2-5 ladder on every
   * preset click, on the argument that a view should be framed in a round
   * number of divisions. The argument is fine and the place is wrong: the
   * session opens at 4.6 units, which is not on the ladder, so the very first
   * press of FRONT zoomed out 16% before it turned — one press, two changes,
   * and the second one unasked for. Detents belong to the control that sets
   * the zoom (the wheel, in `zoomCamera`), not to the one that sets the
   * direction.
   */
  applyViewPreset(name: string): void {
    const preset = VIEW_PRESETS[name];
    if (!preset) return;
    this.camera = { ...this.camera, yaw: preset.yaw, pitch: preset.pitch };
    this.invalidate();
  }

  private displayFlags(): number {
    return displayFlags(this.display, this._simulationActive);
  }

  async init(canvas: HTMLCanvasElement): Promise<void> {
    this.canvas = canvas;
    if (!window.isSecureContext) {
      this.fail("WebGPU requires a secure context. Open this viewer on localhost or over HTTPS.");
      return;
    }
    if (!navigator.gpu) {
      this.fail("WebGPU is not available in this browser. Use a current Chrome, Edge, or Safari.");
      return;
    }
    try {
      const adapter = await timeout(
        navigator.gpu.requestAdapter({ powerPreference: "high-performance" }),
        8_000,
        "WebGPU adapter request",
      );
      if (!adapter) {
        this.fail("No WebGPU adapter was found on this system.");
        return;
      }
      this.adapterLabel = await this.identify(adapter);
      this.device = await timeout(adapter.requestDevice(), 8_000, "WebGPU device request");
      const context = canvas.getContext("webgpu");
      if (!context) {
        this.fail("The browser exposed WebGPU but could not create a canvas context.");
        return;
      }
      this.context = context;
      this.format = navigator.gpu.getPreferredCanvasFormat();
      this.context.configure({ device: this.device, format: this.format, alphaMode: "opaque" });

      this.uniformBuffer = this.createUniform(112);  // 7 x vec4, see Uniforms in _webgpu.py
      this.viewBuffer = this.createUniform(64);
      this.overlayBuffer = this.createUniform(112);
      this.meshOverlayBuffer = this.createUniform(112);
      this.graticuleBuffer = this.createUniform(GRATICULE_UNIFORM_SIZE);
      this.buildPipelines();

      this.device.addEventListener("uncapturederror", (event) => {
        const message = (event as GPUUncapturedErrorEvent).error?.message ?? "WebGPU error";
        this.callbacks.onError?.(`WebGPU validation error: ${message}`);
      });
      this.device.lost.then((info) => {
        this.initError = `WebGPU device lost: ${info.message}`;
        this.callbacks.onError?.(this.initError);
      });
      this.callbacks.onReady?.();
    } catch (error) {
      this.fail(error instanceof Error ? error.message : String(error));
    }
  }

  private fail(message: string): void {
    this.initError = message;
    this.callbacks.onError?.(message);
  }

  private async identify(adapter: GPUAdapter): Promise<string> {
    try {
      const info = adapter.info ?? (await (adapter as { requestAdapterInfo?: () => Promise<GPUAdapterInfo> }).requestAdapterInfo?.());
      const parts = [info?.vendor, info?.architecture].filter(Boolean);
      return parts.length ? `WebGPU · ${parts.join(" ")}` : "WebGPU";
    } catch {
      return "WebGPU";
    }
  }

  private createUniform(size: number): GPUBuffer {
    return this.device!.createBuffer({
      size,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  /** Install the overlay and simulation pipelines on a fresh device. */
  private buildPipelines(): void {
    const device = this.device!;
    const overlay = createOverlayPipelines(
      device,
      this.format,
      this.overlayBuffer,
      this.meshOverlayBuffer,
    );
    this.edgePipeline = overlay.edgePipeline;
    this.facePipeline = overlay.facePipeline;
    this.handlePipeline = overlay.handlePipeline;
    this.gizmoEdgePipeline = overlay.gizmoEdgePipeline;
    this.gizmoArrowPipeline = overlay.gizmoArrowPipeline;
    this.gizmoScalePipeline = overlay.gizmoScalePipeline;
    this.overlayBindGroup = overlay.overlayBindGroup;
    this.meshOverlayBindGroup = overlay.meshOverlayBindGroup;

    const graticule = createGraticulePipeline(device, this.format, this.graticuleBuffer);
    this.graticulePipeline = graticule.graticulePipeline;
    this.graticuleBindGroup = graticule.graticuleBindGroup;

    const simulation = createSimulationPipelines(device, this.format, (size) =>
      this.createUniform(size),
    );
    this.simPipeline = simulation.simPipeline;
    this.simEdgePipeline = simulation.simEdgePipeline;
    this.simUniformBuffer = simulation.simUniformBuffer;
    this.simHighlightUniformBuffer = simulation.simHighlightUniformBuffer;
    this.simBindGroup = simulation.simBindGroup;
    this.simHighlightBindGroup = simulation.simHighlightBindGroup;

    // Draw the empty viewport as soon as there is a device to draw it with.
    // Before the first compile there is no scene shader and nothing else would
    // ask for a frame, and an unpainted swap chain is the black the user saw.
    this.invalidate();
  }

  /** Whether the FEM surface replaces the raymarched solid. */
  get simulationActive(): boolean {
    return this._simulationActive;
  }

  set simulationActive(active: boolean) {
    if (active === this._simulationActive) return;
    this._simulationActive = active;
    this.invalidate();
  }

  /** Replace the FEM surface mesh (null clears it); resets view overrides. */
  setSimulationMesh(payload: SimulationMeshPayload | null): void {
    this.simIndexCount = 0;
    this.simHighlight = null;
    this.simPayload = payload;
    this.simScalarOverride = null;
    this.simPositionOverride = null;
    this.simOverlay = null;
    if (!payload || !this.device || payload.indices.length === 0) {
      this.invalidate();
      return;
    }
    const indices = new Uint32Array(payload.indices);
    this.simIndexBuffer?.destroy();
    this.simIndexBuffer = this.device.createBuffer({
      label: "simulation indices",
      size: indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.simIndexBuffer, 0, indices);
    this.simIndexCount = indices.length;
    // Element-edge hairlines, when the payload carries edge index pairs.
    this.simEdgeIndexBuffer?.destroy();
    this.simEdgeIndexBuffer = null;
    this.simEdgeIndexCount = 0;
    if (payload.edges && payload.edges.length >= 2) {
      const edges = new Uint32Array(payload.edges);
      this.simEdgeIndexBuffer = this.device.createBuffer({
        label: "simulation element edges",
        size: edges.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(this.simEdgeIndexBuffer, 0, edges);
      this.simEdgeIndexCount = edges.length;
    }
    this.simRange = payload.range;
    this.uploadSimulationVertices();
  }

  get simulationEdgesVisible(): boolean {
    return this._simulationEdgesVisible;
  }

  /** Show or hide the element-edge overlay on the simulation surface. */
  set simulationEdgesVisible(visible: boolean) {
    if (visible === this._simulationEdgesVisible) return;
    this._simulationEdgesVisible = visible;
    this.invalidate();
  }

  /** Swap the displayed nodal field without re-solving or re-meshing. */
  setSimulationScalars(scalars: readonly number[] | null, range?: [number, number]): void {
    this.simScalarOverride = scalars;
    if (range) this.simRange = range;
    else if (!scalars && this.simPayload) this.simRange = this.simPayload.range;
    this.uploadSimulationVertices();
  }

  /** Deformed view: draw the surface at offset positions (null = undeformed). */
  setSimulationPositions(positions: readonly number[] | null): void {
    this.simPositionOverride = positions;
    this.uploadSimulationVertices();
  }

  /** Per-vertex RGBA highlight tint (BC previews); null clears it. */
  setSimulationOverlay(colors: Float32Array | null): void {
    this.simOverlay = colors;
    this.uploadSimulationVertices();
  }

  /** Re-interleave and upload the surface vertices from the current view. */
  private uploadSimulationVertices(): void {
    const payload = this.simPayload;
    if (!payload || !this.device || this.simIndexCount === 0) {
      this.invalidate();
      return;
    }
    const positions = this.simPositionOverride ?? payload.positions;
    const scalars = this.simScalarOverride ?? payload.scalars;
    const overlay = this.simOverlay;
    const vertexCount = payload.vertex_count;
    const interleaved = new Float32Array(vertexCount * 8);
    for (let index = 0; index < vertexCount; index++) {
      const base = index * 8;
      interleaved[base] = positions[index * 3];
      interleaved[base + 1] = positions[index * 3 + 1];
      interleaved[base + 2] = positions[index * 3 + 2];
      interleaved[base + 3] = scalars[index] ?? 0;
      if (overlay) {
        interleaved[base + 4] = overlay[index * 4];
        interleaved[base + 5] = overlay[index * 4 + 1];
        interleaved[base + 6] = overlay[index * 4 + 2];
        interleaved[base + 7] = overlay[index * 4 + 3];
      }
    }
    this.simVertexBuffer?.destroy();
    this.simVertexBuffer = this.device.createBuffer({
      label: "simulation vertices",
      size: interleaved.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.simVertexBuffer, 0, interleaved);
    this.simBounds = meshBounds(positions);
    this.invalidate();
  }

  /** Tint one face group's triangle range on hover (null clears it). */
  setSimulationHighlight(range: { start: number; count: number } | null): void {
    this.simHighlight = range;
    this.scheduleRender();
  }

  /** Move the ParaView-style clip plane of the simulation surface. */
  setSimulationClip(clip: SliceState): void {
    this.simClip = { ...clip };
    this.scheduleRender();
  }

  /**
   * Compile and install a freshly generated scene.
   *
   * The preview pipeline lands first so the viewer updates immediately; the
   * path-trace pipelines follow. A revision guard drops results from a compile
   * that has since been superseded.
   *
   * The cheap case comes first. In the uniform form the two scene shaders are
   * byte-identical for every value of every design parameter, so a compile
   * whose sources match the installed ones can only differ in its numbers:
   * it is answered by writing the parameter buffer and redrawing, with no
   * `createShaderModule` and no pipeline at all. That is the whole point of
   * the form — a slider drag on `scenes/motor_shield.py` was 440 ms of module
   * compilation and 520 ms of pipeline creation per edit before it.
   */
  async setShaders(shaders: Shaders): Promise<void> {
    if (!this.device) {
      this.callbacks.onError?.(this.initError || "WGSL compiled, but WebGPU is unavailable.");
      return;
    }
    const program = shaders.program ?? null;
    if (this.isValuesOnlyEdit(shaders, program)) {
      // Newest values win: an older rebuild still in flight reads `program`
      // rather than the payload it started from, so it cannot resurrect them.
      this.program = program;
      this.parameterOverrides = null;
      this.uploadParameters();
      this.parameterUploads += 1;
      this.destroyAccumulation();
      this.invalidate();
      return;
    }

    const revision = ++this.shaderRevision;
    this.pathReady = false;
    this.program = program;
    this.parameterOverrides = null;
    this.ensureParameterBuffer();

    const previewModule = await this.shaderModules.get(
      this.device,
      shaders.preview,
      "Preview WGSL",
    );
    const preview = sharedLayout(this.device, "Preview bindings", [0, 2]);
    const previewLayout = this.device.createPipelineLayout({
      bindGroupLayouts: this.sceneBindGroupLayouts(preview.bindGroupLayout),
    });
    const previewDescriptor = (writeMask: number): GPURenderPipelineDescriptor => ({
      layout: previewLayout,
      vertex: { module: previewModule, entryPoint: "vs_main" },
      fragment: {
        module: previewModule,
        entryPoint: "fs_main_depth",
        targets: [{ format: this.format, writeMask }],
      },
      primitive: { topology: "triangle-list" },
      // "always": this fullscreen pass establishes both colour and depth, and a
      // ray miss writes depth 1.0 — which "less" would reject against the 1.0
      // clear, discarding every background fragment.
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: "always" },
    });

    const [colorPipeline, depthOnlyPipeline] = await Promise.all([
      this.device.createRenderPipelineAsync(previewDescriptor(GPUColorWrite.ALL)),
      this.device.createRenderPipelineAsync(previewDescriptor(0)),
    ]);
    this.pipelineBuilds += 2;
    if (revision !== this.shaderRevision) return;

    this.previewPipeline = colorPipeline;
    this.previewDepthPipeline = depthOnlyPipeline;
    this.previewBindGroup = this.device.createBindGroup({
      layout: preview.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 2, resource: { buffer: this.viewBuffer } },
      ],
    });
    this.previewParameterGroup = this.parameterBindGroup(this.parameterLayout);
    this.uploadParameters();
    this.destroyAccumulation();
    this.invalidate();

    const [pathModule, presentModule] = await Promise.all([
      this.shaderModules.get(this.device, shaders.path, "Path tracer WGSL"),
      this.shaderModules.get(this.device, shaders.present, "Present WGSL"),
    ]);
    const [pathPipeline, presentPipeline] = await Promise.all([
      this.device.createRenderPipelineAsync({
        layout: "auto",
        vertex: { module: pathModule, entryPoint: "vs_main" },
        fragment: {
          module: pathModule,
          entryPoint: "fs_path_trace",
          targets: [{ format: "rgba16float" }],
        },
        primitive: { topology: "triangle-list" },
      }),
      this.device.createRenderPipelineAsync({
        layout: "auto",
        vertex: { module: presentModule, entryPoint: "vs_present" },
        fragment: {
          module: presentModule,
          entryPoint: "fs_present",
          targets: [{ format: this.format }],
        },
        primitive: { topology: "triangle-list" },
        // Shares the pass with the depth prepass and overlay, so it needs a
        // matching depth state even though it neither tests nor writes depth.
        depthStencil: {
          format: DEPTH_FORMAT,
          depthWriteEnabled: false,
          depthCompare: "always",
        },
      }),
    ]);
    this.pipelineBuilds += 2;
    if (revision !== this.shaderRevision) return;
    this.pathPipeline = pathPipeline;
    this.presentPipeline = presentPipeline;
    // The path pipeline derives its own layout, so its parameter bind group
    // has to come from that layout rather than the preview's: an automatic
    // layout is exclusive to the pipeline that produced it.
    this.pathParameterGroup = this.program
      ? this.parameterBindGroup(pathPipeline.getBindGroupLayout(this.program.group))
      : null;
    this.installedShaders = { ...shaders };
    this.pathReady = true;
    this.invalidate();
  }

  /**
   * Whether this compile changed only parameter values.
   *
   * True when all three sources are the ones already installed *and* the
   * uniform contract has the same slots in the same places — the layout is
   * part of the source's meaning, so a program that repacked its buffer is
   * a different shader even if the text somehow matched. In the literal
   * form both programs are null, `sameLayout` agrees, and identical sources
   * mean nothing changed at all: the upload is a no-op and correct.
   */
  private isValuesOnlyEdit(
    shaders: Shaders,
    program: ShaderProgramPayload | null,
  ): boolean {
    const installed = this.installedShaders;
    return Boolean(
      installed &&
        this.previewPipeline &&
        this.pathPipeline &&
        shaders.preview === installed.preview &&
        shaders.path === installed.path &&
        shaders.present === installed.present &&
        sameLayout(program, this.program),
    );
  }

  /**
   * The bind group layouts of a scene pipeline, group by group.
   *
   * Group 0 is the camera and view uniforms every scene shader has always
   * had. A uniform-form shader adds its parameter block at the group the
   * payload names — 3 today — and WebGPU needs the groups in between
   * spelled out, so they are declared empty. An empty layout binds nothing
   * and needs no bind group set against it.
   */
  private sceneBindGroupLayouts(scene: GPUBindGroupLayout): GPUBindGroupLayout[] {
    const layouts = [scene];
    this.parameterLayout = null;
    if (!this.device || !this.program) return layouts;
    this.parameterLayout = this.device.createBindGroupLayout({
      label: "SDF parameter bindings",
      entries: [
        {
          binding: this.program.binding,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        },
      ],
    });
    while (layouts.length < this.program.group) {
      layouts.push(this.device.createBindGroupLayout({ entries: [] }));
    }
    layouts.push(this.parameterLayout);
    return layouts;
  }

  /** The parameter buffer bound against one layout, or null without a program. */
  private parameterBindGroup(layout: GPUBindGroupLayout | null): GPUBindGroup | null {
    if (!this.device || !this.program || !this.parameterBuffer || !layout) return null;
    return this.device.createBindGroup({
      label: "SDF parameters",
      layout,
      entries: [{ binding: this.program.binding, resource: { buffer: this.parameterBuffer } }],
    });
  }

  /**
   * Size the parameter buffer to the current program.
   *
   * Only ever called on a full rebuild: destroying the buffer invalidates
   * every bind group holding it, and the rebuild makes new ones straight
   * after. A values-only edit is guarded by `sameLayout`, so it never
   * reaches here and its bind groups stay valid.
   */
  private ensureParameterBuffer(): void {
    if (!this.device) return;
    const size = this.program
      ? Math.max(this.program.buffer_bytes, PARAMETER_SLOT_BYTES)
      : 0;
    if (size === this.parameterBufferBytes) return;
    this.parameterBuffer?.destroy();
    this.parameterBufferBytes = size;
    this.parameterBuffer = size
      ? this.device.createBuffer({
          label: "SDF parameters",
          size,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })
      : null;
  }

  /** Write the current parameter values, drag overrides included. */
  private uploadParameters(): void {
    if (!this.device || !this.program || !this.parameterBuffer) return;
    const packed = packParameters(this.program, this.parameterOverrides ?? undefined);
    this.device.queue.writeBuffer(this.parameterBuffer, 0, packed.buffer as ArrayBuffer);
  }

  /**
   * Show these parameter values now, without a compile.
   *
   * What a handle or gizmo drag calls on every pointer move: the shader
   * already reads its parameters from a buffer, so moving one is a few
   * hundred bytes and a redraw. The server patch follows at its own pace
   * and lands as an ordinary compile, which clears the overrides by
   * installing the same numbers.
   *
   * @param overrides Values by parameter name; null drops back to the
   *   program's own.
   * @returns False in the literal form, where there is no buffer to write
   *   and the caller must wait for the round trip.
   */
  setParameterOverrides(
    overrides: Readonly<Record<string, readonly number[]>> | null,
  ): boolean {
    if (!this.program || !this.parameterBuffer) return false;
    this.parameterOverrides = overrides ? { ...overrides } : null;
    this.uploadParameters();
    // Counted with the compile-path uploads, because they are the same
    // event seen from two sides: a value reached the GPU without a shader
    // being built. A drag's worth of these is what the e2e test measures.
    this.parameterUploads += 1;
    this.destroyAccumulation();
    this.invalidate();
    return true;
  }

  /** Whether the installed scene reads its parameters from a buffer. */
  get hasParameterBuffer(): boolean {
    return this.program !== null && this.parameterBuffer !== null;
  }

  /** Shader module cache counters, for tests and the performance study. */
  get shaderCacheStats(): { hits: number; misses: number; size: number } {
    return {
      hits: this.shaderModules.hits,
      misses: this.shaderModules.misses,
      size: this.shaderModules.size,
    };
  }

  /**
   * What the shader path has done this session, for the e2e tests.
   *
   * Published on `window` by the app (`__cadjointShaders`) because the
   * claim worth testing is a negative one — a handle drag rebuilds *no*
   * pipelines — and a negative is only checkable against a counter.
   */
  get shaderStats(): ShaderStats {
    return {
      pipelineBuilds: this.pipelineBuilds,
      parameterUploads: this.parameterUploads,
      hasParameterBuffer: this.hasParameterBuffer,
      hits: this.shaderModules.hits,
      misses: this.shaderModules.misses,
    };
  }

  /** Replace the construction geometry drawn on top of the scene. */
  setConstruction(
    profiles: readonly ConstructionNode[],
    selection: Selection | null,
    hover: Selection | null,
  ): void {
    this.profiles = profiles;
    this.selection = selection;
    this.hover = hover;
    this.uploadOverlay();
    this.scheduleRender();
  }

  /** Replace the dual-contour mesh edges gated by the mesh-edges display flag. */
  setMeshEdges(payload: MeshEdgePayload | null): void {
    this.meshEdges = payload;
    this.uploadOverlay();
    this.scheduleRender();
  }

  /**
   * Highlight one analytic face, or clear the highlight.
   *
   * Takes the face itself rather than a polygon: the fill has to be
   * triangulated in the face's own frame, and `usable` decides the weight —
   * a face the source cannot name is still shown, at half strength.
   */
  setFaceHighlight(face: ConstructionFace | null): void {
    if (this.faceHighlight === face) return;
    this.faceHighlight = face;
    this.uploadOverlay();
    this.scheduleRender();
  }

  private uploadOverlay(): void {
    if (!this.device) return;
    const { edges, handles } = packConstructionOverlay(
      this.profiles,
      this.selection,
      this.hover,
    );

    // Transform controls have their own buffer and pass.
    const target = this.gizmoTarget();
    let gizmo: number[] = [];
    if (target) {
      this.visibleGizmoMode = this.gizmoModeFor(target.node);
      gizmo = packGizmoInstances(
        target.origin,
        gizmoScale(this.view, target.origin),
        this.visibleGizmoMode,
        this.gizmoAxis,
      );
    }

    const meshSegments = packMeshEdgeInstances(this.meshEdges);
    const highlight = packFaceHighlight(this.faceHighlight);

    this.faceFillVertices = highlight.fill.length / (FACE_STRIDE / 4);
    this.faceOutlineCount = highlight.outline.length / (EDGE_STRIDE / 4);
    this.faceFillBuffer = this.writeInstances(
      this.faceFillBuffer,
      new Float32Array(highlight.fill),
      FACE_STRIDE,
      (capacity) => (this.faceFillCapacity = capacity),
      this.faceFillCapacity,
      "face highlight fill",
    );
    this.faceOutlineBuffer = this.writeInstances(
      this.faceOutlineBuffer,
      new Float32Array(highlight.outline),
      EDGE_STRIDE,
      (capacity) => (this.faceOutlineCapacity = capacity),
      this.faceOutlineCapacity,
      "face highlight outline",
    );
    this.edgeCount = edges.length / (EDGE_STRIDE / 4);
    this.handleCount = handles.length / (HANDLE_STRIDE / 4);
    this.gizmoCount = gizmo.length / (GIZMO_STRIDE / 4);
    this.meshWireCount = this.meshEdges?.wire.length ?? 0;
    this.meshSharpCount = this.meshEdges?.sharp.length ?? 0;
    this.meshEdgeBuffer = this.writeInstances(
      this.meshEdgeBuffer,
      new Float32Array(meshSegments),
      EDGE_STRIDE,
      (capacity) => (this.meshEdgeCapacity = capacity),
      this.meshEdgeCapacity,
      "mesh edges",
    );
    this.edgeBuffer = this.writeInstances(
      this.edgeBuffer,
      new Float32Array(edges),
      EDGE_STRIDE,
      (capacity) => (this.edgeCapacity = capacity),
      this.edgeCapacity,
      "overlay edges",
    );
    this.handleBuffer = this.writeInstances(
      this.handleBuffer,
      new Float32Array(handles),
      HANDLE_STRIDE,
      (capacity) => (this.handleCapacity = capacity),
      this.handleCapacity,
      "overlay handles",
    );
    this.gizmoBuffer = this.writeInstances(
      this.gizmoBuffer,
      new Float32Array(gizmo),
      GIZMO_STRIDE,
      (capacity) => (this.gizmoCapacity = capacity),
      this.gizmoCapacity,
      "transform gizmo",
    );
  }

  /** The selected primitive the gizmo should attach to, if any. */
  gizmoTarget(): { node: ConstructionNode; origin: Vec3 } | null {
    const active = this.selection;
    if (!active || active.vertexIndex !== null) return null;
    const node = this.profiles.find((candidate) => candidate.id === active.nodeId);
    if (!node?.transform || !node.editable) return null;
    if (node.kind === "profile" && node.vertices.length > 0) {
      const sum = node.vertices.reduce<Vec3>(
        (center, vertex) => [
          center[0] + vertex.world[0],
          center[1] + vertex.world[1],
          center[2] + vertex.world[2],
        ] as Vec3,
        [0, 0, 0],
      );
      const count = node.vertices.length;
      return {
        node,
        // The plane origin can sit far outside an offset polygon. Keep the
        // translation arrows on the selected geometry where they are visible.
        origin: [sum[0] / count, sum[1] / count, sum[2] / count],
      };
    }
    return { node, origin: node.transform.position as Vec3 };
  }

  private writeInstances(
    buffer: GPUBuffer | null,
    data: Float32Array<ArrayBuffer>,
    stride: number,
    setCapacity: (capacity: number) => void,
    capacity: number,
    label: string,
  ): GPUBuffer | null {
    if (data.length === 0) return buffer;
    const device = this.device!;
    const required = data.byteLength;
    if (!buffer || capacity < required) {
      buffer?.destroy();
      const size = Math.max(required, stride * 64);
      buffer = device.createBuffer({
        label,
        size,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
      setCapacity(size);
    }
    device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  }

  private destroyAccumulation(): void {
    for (const texture of this.accumulation) texture.destroy();
    this.accumulation = [];
    this.pathBindGroups = [];
    this.presentBindGroups = [];
    this.accumulationWidth = 0;
    this.accumulationHeight = 0;
  }

  private ensureAccumulation(): void {
    if (
      this.accumulation.length === 2 &&
      this.accumulationWidth === this.canvas.width &&
      this.accumulationHeight === this.canvas.height
    ) {
      return;
    }
    this.destroyAccumulation();
    const device = this.device!;
    this.accumulationWidth = this.canvas.width;
    this.accumulationHeight = this.canvas.height;
    const descriptor: GPUTextureDescriptor = {
      size: [this.canvas.width, this.canvas.height],
      format: "rgba16float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    };
    this.accumulation = [device.createTexture(descriptor), device.createTexture(descriptor)];
    this.pathBindGroups = this.accumulation.map((texture) =>
      device.createBindGroup({
        layout: this.pathPipeline!.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuffer } },
          { binding: 1, resource: texture.createView() },
        ],
      }),
    );
    this.presentBindGroups = this.accumulation.map((texture) =>
      device.createBindGroup({
        layout: this.presentPipeline!.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: texture.createView() }],
      }),
    );
    this.readIndex = 0;
    this.sampleCount = 0;
  }

  private ensureDepthTexture(): GPUTextureView {
    if (
      !this.depthTexture ||
      this.depthTexture.width !== this.canvas.width ||
      this.depthTexture.height !== this.canvas.height
    ) {
      this.depthTexture?.destroy();
      this.depthTexture = this.device!.createTexture({
        label: "scene depth",
        size: [this.canvas.width, this.canvas.height],
        format: DEPTH_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }
    return this.depthTexture.createView();
  }

  /**
   * Match the backing store to the displayed size within the quality budget.
   *
   * Runs with or without a GPU: hit testing projects into framebuffer pixels,
   * so the viewport has to be right even when WebGPU is unavailable.
   */
  resize(): void {
    if (!this.canvas) return;
    const cssWidth = Math.max(1, this.canvas.clientWidth);
    const cssHeight = Math.max(1, this.canvas.clientHeight);
    const requested = Math.min(window.devicePixelRatio || 1, this.quality.maxRatio);
    const budget = Math.sqrt(this.quality.pixelBudget / (cssWidth * cssHeight));
    const ratio = Math.min(requested, budget);
    const width = Math.max(1, Math.floor(cssWidth * ratio));
    const height = Math.max(1, Math.floor(cssHeight * ratio));
    if (this.canvas.width === width && this.canvas.height === height) return;

    this.canvas.width = width;
    this.canvas.height = height;
    if (this.device && this.context) {
      this.context.configure({ device: this.device, format: this.format, alphaMode: "opaque" });
      this.destroyAccumulation();
    }
  }

  /**
   * Re-attach the swap chain to the canvas.
   *
   * `resize()` reconfigures whenever the backing store changes, which covers
   * almost everything. The exception is a canvas that is moved in the DOM
   * without changing size — which is what a dock rebuild does when the new
   * layout happens to give the viewport the same rectangle. The element and
   * its context survive that move, but configuring again costs nothing and
   * removes the question.
   */
  reconfigure(): void {
    if (!this.device || !this.context) return;
    this.context.configure({ device: this.device, format: this.format, alphaMode: "opaque" });
  }

  invalidate(): void {
    this.sampleCount = 0;
    this.scheduleRender();
  }

  scheduleRender(): void {
    if (this.framePending) return;
    this.framePending = true;
    requestAnimationFrame(() => this.render());
  }

  private writeUniforms(): void {
    const device = this.device!;
    const position = cameraPosition(this.camera);
    // The six trailing scalars of the first five vec4s are the SDF views and
    // the march budget; `Uniforms` in `cadjoint/viewer/_webgpu.py` documents
    // the packing and why they ride here rather than in an eighth vec4.
    const { sdfView, sdfAxis, sdfFraction, isoOffset } = this.display;
    const sdfMode = SDF_VIEW_CODE[sdfView] ?? 0;
    const scene = new Float32Array([
      this.canvas.width, this.canvas.height, sdfMode, sdfAxis,
      position[0], position[1], position[2], slicePosition(sdfFraction),
      this.camera.target[0], this.camera.target[1], this.camera.target[2], isoOffset,
      0.55, 0.35, 0.8, KEY_LIGHT_INTENSITY,
      BACKGROUND_RADIANCE[0],
      BACKGROUND_RADIANCE[1],
      BACKGROUND_RADIANCE[2],
      // The slice's contours are ruled at the floor grid's own spacing, so a
      // contour interval is the same stated number the GRID readout is
      // showing and the two annotations cannot disagree.
      gridSpacing(this.camera.distance),
      this.sampleCount,
      this.quality.bounces,
      this.quality.shadowSamples,
      this.quality.marchSteps,
      this.display.projection === "orthographic" ? 1 : 0,
      orthoHeightFor(this.camera.distance),
      this.displayFlags(),
      this.display.xray,
    ]);
    device.queue.writeBuffer(this.uniformBuffer, 0, scene);

    const matrix = viewProjection(this.view);
    device.queue.writeBuffer(this.viewBuffer, 0, matrix);

    const overlay = new Float32Array(28);
    overlay.set(matrix, 0);
    overlay.set([position[0], position[1], position[2], 0], 16);
    overlay.set(
      [this.canvas.width, this.canvas.height, LINE_WIDTH_PX, HANDLE_RADIUS_PX],
      20,
    );
    overlay.set([DEPTH_NUDGE, 0, 0, 0], 24);
    device.queue.writeBuffer(this.overlayBuffer, 0, overlay);

    // Mesh edges sit exactly on the surface creases the construction
    // wireframe also traces; a smaller nudge keeps them consistently behind
    // coincident construction lines instead of z-fighting into dashes.
    overlay.set([DEPTH_NUDGE * 0.4, 0, 0, 0], 24);
    device.queue.writeBuffer(this.meshOverlayBuffer, 0, overlay);

    if (this.display.showGraticule) this.writeGraticuleUniforms();

    if (this._simulationActive && this.simUniformBuffer && this.simHighlightUniformBuffer) {
      const { normal, offset } = slicePlane(this.simClip, this.simBounds);
      const [low, high] = this.simRange;
      const span = high - low;
      const inverseRange = span > 1e-12 ? 1 / span : 0;
      const ramp = this.simulationRamp === "quality" ? 1 : 0;
      const sim = new Float32Array(28);
      sim.set(matrix, 0);
      sim.set([normal[0], normal[1], normal[2], offset], 16);
      sim.set([low, inverseRange, 0, this.simClip.enabled ? 1 : 0], 20);
      // extra.yzw is the camera position: the FEM surface builds its facet
      // normal from screen-space derivatives, and on the paper ground it needs
      // the eye vector both to orient that normal and to darken the
      // silhouette, which is what separates a hot field from the background.
      sim.set([ramp, position[0], position[1], position[2]], 24);
      device.queue.writeBuffer(this.simUniformBuffer, 0, sim);
      // The highlight pass re-draws a face group's range with a warm tint.
      sim.set([low, inverseRange, 0.55, this.simClip.enabled ? 1 : 0], 20);
      device.queue.writeBuffer(this.simHighlightUniformBuffer, 0, sim);
    }
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
  private writeGraticuleUniforms(): void {
    const scale = this.canvas.width / Math.max(this.canvas.clientWidth, 1);
    const px = (css: number) => css * scale;
    const tone = (name: "graticule-line" | "graticule-axis") => hexToRgb(CHROME[name]);
    const [lineR, lineG, lineB] = tone("graticule-line");
    const [axisR, axisG, axisB] = tone("graticule-axis");
    const position = cameraPosition(this.camera);
    const { forward, right, up } = cameraBasis(position, this.camera.target);
    const distance = this.camera.distance;
    const target = this.camera.target;
    // One dimmer for the whole plane, so the three weights keep their order.
    const emphasis = this.groundEmphasis;
    this.device!.queue.writeBuffer(
      this.graticuleBuffer,
      0,
      new Float32Array([
        this.canvas.width,
        this.canvas.height,
        Math.max(1, px(GRATICULE.lineWidth)),
        this.display.projection === "orthographic" ? 1 : 0,
        position[0], position[1], position[2], orthoHeightFor(distance),
        right[0], right[1], right[2],
        this.canvas.width / Math.max(this.canvas.height, 1),
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
      ]),
    );
  }

  /** Draw the graticule between the scene's depth and the overlays. */
  private drawGraticule(pass: GPURenderPassEncoder): void {
    if (!this.display.showGraticule || !this.graticulePipeline || !this.graticuleBindGroup) {
      return;
    }
    pass.setPipeline(this.graticulePipeline);
    pass.setBindGroup(0, this.graticuleBindGroup);
    pass.draw(3);
  }

  /** Draw the FEM surface (and its hovered face group) into the pass. */
  private drawSimulation(pass: GPURenderPassEncoder): void {
    if (
      !this._simulationActive ||
      !this.simPipeline ||
      !this.simBindGroup ||
      !this.simVertexBuffer ||
      !this.simIndexBuffer ||
      this.simIndexCount === 0
    ) {
      return;
    }
    pass.setPipeline(this.simPipeline);
    pass.setBindGroup(0, this.simBindGroup);
    pass.setVertexBuffer(0, this.simVertexBuffer);
    pass.setIndexBuffer(this.simIndexBuffer, "uint32");
    pass.drawIndexed(this.simIndexCount);
    // The BC preview is a mark on the surface, not part of the field, so it
    // goes with the rest of the construction overlay.
    const highlight = this.display.showOverlays ? this.simHighlight : null;
    if (highlight && this.simHighlightBindGroup) {
      pass.setBindGroup(0, this.simHighlightBindGroup);
      pass.drawIndexed(highlight.count, 1, highlight.start);
    }
    if (
      this._simulationEdgesVisible &&
      this.simEdgePipeline &&
      this.simEdgeIndexBuffer &&
      this.simEdgeIndexCount > 0
    ) {
      pass.setPipeline(this.simEdgePipeline);
      pass.setBindGroup(0, this.simBindGroup);
      pass.setVertexBuffer(0, this.simVertexBuffer);
      pass.setIndexBuffer(this.simEdgeIndexBuffer, "uint32");
      pass.drawIndexed(this.simEdgeIndexCount);
    }
  }

  /**
   * Whether construction geometry should read through the solid.
   *
   * The x-ray strength is the same number the shader fades the surface with,
   * so the overlay's rule and the solid's translucency turn on together: while
   * you can see into the part, you can see the sketch inside it. The floor
   * grid is not in this population — it is the ground, not construction — so
   * it always tests depth and is always occluded by the part standing on it.
   */
  private get seeThroughOverlays(): boolean {
    return this.display.xray > 0;
  }

  private drawOverlay(pass: GPURenderPassEncoder): void {
    if (!this.overlayBindGroup) return;
    const depth = this.seeThroughOverlays
      ? (pair: DepthPair) => pair.seen
      : (pair: DepthPair) => pair.tested;
    const wantMeshWire = this.display.showMeshWireframe && this.meshWireCount > 0;
    const wantMeshSharp = this.display.showMeshEdges && this.meshSharpCount > 0;
    if (
      (wantMeshWire || wantMeshSharp) &&
      this.meshEdgeBuffer &&
      this.edgePipeline &&
      this.meshOverlayBindGroup
    ) {
      pass.setPipeline(depth(this.edgePipeline!));
      pass.setBindGroup(0, this.meshOverlayBindGroup);
      pass.setVertexBuffer(0, this.meshEdgeBuffer);
      if (wantMeshWire) pass.draw(6, this.meshWireCount, 0, 0);
      if (wantMeshSharp) pass.draw(6, this.meshSharpCount, 0, this.meshWireCount);
      pass.setBindGroup(0, this.overlayBindGroup);
    }
    // Everything below is the construction overlay: what the app draws *about*
    // the model. One switch turns the lot off for a presentation frame; the
    // mesh edges above belong to the mesh and keep their own.
    if (!this.display.showOverlays) return;
    // Fill first, then its own outline: the wash does not write depth, so the
    // hairline lands on top of it rather than fighting it.
    if (this.faceFillVertices && this.faceFillBuffer && this.facePipeline) {
      pass.setPipeline(depth(this.facePipeline!));
      pass.setBindGroup(0, this.overlayBindGroup);
      pass.setVertexBuffer(0, this.faceFillBuffer);
      pass.draw(this.faceFillVertices);
    }
    if (this.faceOutlineCount && this.faceOutlineBuffer && this.edgePipeline) {
      pass.setPipeline(depth(this.edgePipeline!));
      pass.setBindGroup(0, this.overlayBindGroup);
      pass.setVertexBuffer(0, this.faceOutlineBuffer);
      pass.draw(6, this.faceOutlineCount);
    }
    if (this.display.showSketches && this.edgeCount && this.edgeBuffer && this.edgePipeline) {
      pass.setPipeline(depth(this.edgePipeline!));
      pass.setBindGroup(0, this.overlayBindGroup);
      pass.setVertexBuffer(0, this.edgeBuffer);
      pass.draw(6, this.edgeCount);
    }
    if (
      this.display.showSketches &&
      this.handleCount &&
      this.handleBuffer &&
      this.handlePipeline
    ) {
      pass.setPipeline(depth(this.handlePipeline!));
      pass.setBindGroup(0, this.overlayBindGroup);
      pass.setVertexBuffer(0, this.handleBuffer);
      pass.draw(6, this.handleCount);
    }
    if (this.gizmoCount && this.gizmoBuffer) {
      const pipeline =
        this.visibleGizmoMode === "translate"
          ? this.gizmoArrowPipeline
          : this.visibleGizmoMode === "scale"
            ? this.gizmoScalePipeline
            : this.gizmoEdgePipeline;
      if (pipeline) {
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, this.overlayBindGroup);
        pass.setVertexBuffer(0, this.gizmoBuffer);
        pass.draw(
          this.visibleGizmoMode === "translate"
            ? 9
            : this.visibleGizmoMode === "scale"
              ? 12
              : 6,
          this.gizmoCount,
        );
      }
    }
  }

  /**
   * The viewport with nothing in it: paper, and the floor ruled on it.
   *
   * Before the first compile, and whenever a shader rebuild or a compile error
   * has left no preview pipeline, there is still a viewport to draw. Clearing
   * to black is what an uninitialised swap chain does, and it looks like a
   * fault; clearing to the same paper the chrome is on, with the ground grid
   * over it, says "nothing here yet" instead.
   */
  private renderEmpty(): void {
    const device = this.device!;
    this.writeUniforms();
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.context!.getCurrentTexture().createView(),
          clearValue: BACKGROUND,
          loadOp: "clear",
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: this.ensureDepthTexture(),
        depthClearValue: 1,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });
    this.drawGraticule(pass);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  private render(): void {
    this.framePending = false;
    if (!this.device || !this.context) return;
    this.resize();
    if (!this.previewPipeline || !this.previewBindGroup) {
      this.renderEmpty();
      return;
    }
    this.writeUniforms();

    const device = this.device;
    const encoder = device.createCommandEncoder();
    const swapchain = this.context.getCurrentTexture().createView();
    const depthView = this.ensureDepthTexture();
    const depthAttachment: GPURenderPassDepthStencilAttachment = {
      view: depthView,
      depthClearValue: 1,
      depthLoadOp: "clear",
      depthStoreOp: "store",
    };
    // The path tracer draws the SDF scene, which simulation display replaces.
    const tracing =
      this.pathTracing && this.pathReady && !this.interacting && !this._simulationActive;

    if (tracing) {
      this.ensureAccumulation();
      const writeIndex = 1 - this.readIndex;
      const tracePass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: this.accumulation[writeIndex].createView(),
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: "clear",
            storeOp: "store",
          },
        ],
      });
      tracePass.setPipeline(this.pathPipeline!);
      tracePass.setBindGroup(0, this.pathBindGroups[this.readIndex]);
      if (this.pathParameterGroup) {
        tracePass.setBindGroup(this.program!.group, this.pathParameterGroup);
      }
      tracePass.draw(3);
      tracePass.end();

      const presentPass = encoder.beginRenderPass({
        colorAttachments: [
          { view: swapchain, clearValue: BACKGROUND, loadOp: "clear", storeOp: "store" },
        ],
        depthStencilAttachment: depthAttachment,
      });
      presentPass.setPipeline(this.presentPipeline!);
      presentPass.setBindGroup(0, this.presentBindGroups[writeIndex]);
      presentPass.draw(3);
      // Re-run the cheap preview purely for depth so overlays interleave with
      // the path-traced image, which carries no depth of its own. The
      // graticule needs it for the same reason — without a depth buffer it
      // would paint over the part instead of behind it.
      const wantDepth =
        (this.edgeCount && this.display.showSketches) || this.display.showGraticule;
      if (wantDepth && this.previewDepthPipeline) {
        presentPass.setPipeline(this.previewDepthPipeline);
        presentPass.setBindGroup(0, this.previewBindGroup);
        if (this.previewParameterGroup) {
          presentPass.setBindGroup(this.program!.group, this.previewParameterGroup);
        }
        presentPass.draw(3);
      }
      this.drawGraticule(presentPass);
      this.drawOverlay(presentPass);
      presentPass.end();
      this.readIndex = writeIndex;
    } else {
      const previewPass = encoder.beginRenderPass({
        colorAttachments: [
          { view: swapchain, clearValue: BACKGROUND, loadOp: "clear", storeOp: "store" },
        ],
        depthStencilAttachment: depthAttachment,
      });
      previewPass.setPipeline(this.previewPipeline);
      previewPass.setBindGroup(0, this.previewBindGroup);
      if (this.previewParameterGroup) {
        previewPass.setBindGroup(this.program!.group, this.previewParameterGroup);
      }
      previewPass.draw(3);
      this.drawSimulation(previewPass);
      // After every pass that writes depth, before every pass that does not:
      // the faceplate is occluded by the part and the FEM surface, and the
      // construction overlays are drawn on top of it.
      this.drawGraticule(previewPass);
      this.drawOverlay(previewPass);
      previewPass.end();
    }

    device.queue.submit([encoder.finish()]);

    if (tracing) {
      this.sampleCount += 1;
      this.callbacks.onStatus?.(
        "ready",
        `${this.quality.label} path trace · ${this.adapterLabel} · ${this.sampleCount} spp`,
      );
      if (this.sampleCount < this.quality.samples) this.scheduleRender();
    } else {
      const suffix = this.pathTracing
        ? this.pathReady
          ? " · moving"
          : " · preparing path tracer…"
        : "";
      this.callbacks.onStatus?.(
        "ready",
        `${this.quality.label} preview · ${this.adapterLabel}${suffix}`,
      );
    }
  }

  destroy(): void {
    this.destroyAccumulation();
    this.depthTexture?.destroy();
    this.edgeBuffer?.destroy();
    this.faceFillBuffer?.destroy();
    this.faceOutlineBuffer?.destroy();
    this.handleBuffer?.destroy();
    this.gizmoBuffer?.destroy();
    this.simVertexBuffer?.destroy();
    this.simIndexBuffer?.destroy();
    this.device?.destroy();
  }
}
