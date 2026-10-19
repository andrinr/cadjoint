/**
 * WebGPU renderer for the playground viewer.
 *
 * Draws the compiled SDF scene (raymarched preview or progressive path trace)
 * and the construction-tree overlay on top of it. The preview pass writes
 * `frag_depth` from the ray hit position, so construction edges and handles
 * are depth-tested against the solid. Transform controls use a separate
 * always-visible pass so the selected object cannot hide its own gizmo.
 *
 * This module holds the device and the frame: adapter setup, the uniform
 * buffers and render targets, and the passes that draw into them. What each
 * pass draws lives with its owner: the compiled scene in `scenePrograms.ts`,
 * the construction overlay in `overlayLayer.ts`, the B-rep inspector in
 * `brepLayer.ts` and the FEM surface in `simulationLayer.ts`. The settings the
 * panels write live in `display.ts`, the uniform packing in `uniforms.ts`, the
 * vertex packing in `overlayGeometry.ts`, and the pipeline descriptors in
 * `pipelines.ts`.
 */

import { planeFrames, type PlaneFrame } from "../planes";
import type { SliceState } from "../simulation";
import type {
  ConstructionFace,
  ConstructionNode,
  GizmoMode,
  MeshEdgePayload,
  Selection,
  SurfacePayload,
} from "../types";
import { BrepLayer, type BrepGeometry } from "./brepLayer";
import {
  DEFAULT_DISPLAY,
  DEFAULT_QUALITY,
  QUALITY_PRESETS,
  VIEW_PRESETS,
  displayFlags,
  type DisplaySettings,
  type QualityPreset,
  type Shaders,
} from "./display";
import {
  BACKGROUND,
  CASING_PX,
  DEPTH_NUDGE,
  GESTURE_FRAME_BUDGET_MS,
  MIN_GESTURE_SCALE,
  timeout,
} from "./frameConstants";
import type { AxisIndex } from "./gizmo";
import {
  cameraPosition,
  orthoHeightFor,
  viewProjection,
  type CameraState,
  type Vec3,
  type View,
} from "./math";
import { OverlayLayer } from "./overlayLayer";
import {
  DEPTH_FORMAT,
  GRATICULE_UNIFORM_SIZE,
  createGraticulePipeline,
  createOverlayPipelines,
  createSimulationPipelines,
  createUpscalePipeline,
} from "./pipelines";
import type { ShaderProgramPayload } from "./shaderProgram";
import { ScenePrograms } from "./scenePrograms";
import { SimulationLayer, type IndexRange } from "./simulationLayer";
import { packGraticuleUniforms, packOverlayUniforms, packSceneUniforms } from "./uniforms";

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

/** For `shell/compileCycle.ts`, which imports the settings type from here. */
export type { DisplaySettings } from "./display";

export interface RendererCallbacks {
  onStatus?: (kind: string, text: string) => void;
  onError?: (message: string) => void;
  onReady?: () => void;
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

  /** The compiled scene: shader modules, pipelines and the parameter buffer. */
  private scene = new ScenePrograms(
    {
      device: () => this.device,
      format: () => this.format,
      uniformBuffers: () => ({ scene: this.uniformBuffer, view: this.viewBuffer }),
      cullBounds: () => this.display.cullBounds,
      previewReplaced: () => {
        this.previewMsPerPixel = null;
      },
      sceneChanged: () => {
        this.destroyAccumulation();
        this.invalidate();
      },
      invalidate: () => this.invalidate(),
    },
    DEFAULT_DISPLAY.cullBounds,
  );
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

  /** Construction skeleton, sketch planes, face highlight, mesh edges, gizmo. */
  private overlay = new OverlayLayer();
  /** The B-rep inspector, drawn instead of the raymarched solid while shown. */
  private brep = new BrepLayer();
  /** The FEM surface, drawn instead of the raymarched solid while it is active. */
  private simulation = new SimulationLayer();

  private framePending = false;
  private initError = "";

  // The session opens on the +X−Y+Z corner, at the exact 1:1:1 direction
  // rather than near it, so the readout's ISO is literally true.
  // `VIEW_PRESETS.iso` is the same pair of numbers.
  camera: CameraState = {
    ...VIEW_PRESETS.iso,
    distance: 4.6,
    target: [0, 0, 0],
  };
  display: DisplaySettings = { ...DEFAULT_DISPLAY };
  quality: QualityPreset = QUALITY_PRESETS[DEFAULT_QUALITY];
  pathTracing = false;
  private _interacting = false;

  // ── The gesture preview ───────────────────────────────────────────────
  /** Measured GPU cost of a preview frame per pixel, smoothed; null until seen. */
  private previewMsPerPixel: number | null = null;
  /** Linear scale of the preview while the current gesture lasts; 1 at rest. */
  private gestureScale = 1;
  private upscalePipeline: GPURenderPipeline | null = null;
  private gestureColor: GPUTexture | null = null;
  private gestureDepth: GPUTexture | null = null;
  private gestureBindGroup: GPUBindGroup | null = null;

  constructor(private callbacks: RendererCallbacks = {}) {}

  /**
   * Whether a gesture — an orbit, a pan, a gizmo or vertex drag — is in
   * flight. Starting one fixes the preview scale for its length from what
   * frames have been measured to cost; ending one restores full resolution
   * and redraws, so the frame left on screen is always the full one.
   */
  get interacting(): boolean {
    return this._interacting;
  }

  set interacting(active: boolean) {
    if (active === this._interacting) return;
    this._interacting = active;
    this.gestureScale = active ? this.chooseGestureScale() : 1;
    if (!active) this.invalidate();
  }

  /** The preview scale a gesture can afford, from the measured frame cost. */
  private chooseGestureScale(): number {
    if (!this.canvas || this.previewMsPerPixel === null) return 1;
    const predicted = this.previewMsPerPixel * this.canvas.width * this.canvas.height;
    if (predicted <= GESTURE_FRAME_BUDGET_MS) return 1;
    // Cost goes with pixel count, so with the square of the linear scale;
    // eighths, so a size is reused rather than nudged from gesture to gesture.
    const scale = Math.sqrt(GESTURE_FRAME_BUDGET_MS / predicted);
    return Math.max(MIN_GESTURE_SCALE, Math.floor(scale * 8) / 8);
  }

  /** Fold one preview frame's measured GPU time into the per-pixel cost. */
  private recordPreviewCost(milliseconds: number, pixels: number): void {
    const sample = milliseconds / Math.max(pixels, 1);
    this.previewMsPerPixel =
      this.previewMsPerPixel === null ? sample : this.previewMsPerPixel * 0.7 + sample * 0.3;
  }

  /** The small targets a gesture frame is raymarched into, sized to `scale`. */
  private ensureGestureTargets(scale: number): { width: number; height: number } {
    const device = this.device!;
    const width = Math.max(1, Math.ceil(this.canvas.width * scale));
    const height = Math.max(1, Math.ceil(this.canvas.height * scale));
    if (this.gestureColor?.width !== width || this.gestureColor?.height !== height) {
      this.gestureColor?.destroy();
      this.gestureDepth?.destroy();
      this.gestureColor = device.createTexture({
        label: "gesture preview colour",
        size: [width, height],
        format: this.format,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      this.gestureDepth = device.createTexture({
        label: "gesture preview depth",
        size: [width, height],
        format: DEPTH_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      this.upscalePipeline ??= createUpscalePipeline(device, this.format);
      this.gestureBindGroup = device.createBindGroup({
        layout: this.upscalePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.gestureColor.createView() },
          { binding: 1, resource: this.gestureDepth.createView() },
          {
            binding: 2,
            resource: device.createSampler({ magFilter: "linear", minFilter: "linear" }),
          },
        ],
      });
    }
    return { width, height };
  }

  get unavailableReason(): string {
    return this.initError;
  }

  get gizmoMode(): GizmoMode {
    return this.overlay.gizmoMode;
  }

  set gizmoMode(mode: GizmoMode) {
    if (mode === this.overlay.gizmoMode) return;
    this.overlay.gizmoMode = mode;
    this.uploadOverlay();
    this.invalidate();
  }

  get gizmoAxis(): AxisIndex | null {
    return this.overlay.gizmoAxis;
  }

  set gizmoAxis(axis: AxisIndex | null) {
    if (axis === this.overlay.gizmoAxis) return;
    this.overlay.gizmoAxis = axis;
    this.uploadOverlay();
    this.scheduleRender();
  }

  /** Requested mode, reduced to the transforms a selected node supports. */
  gizmoModeFor(node: ConstructionNode): GizmoMode {
    return this.overlay.gizmoModeFor(node);
  }

  /** The selected primitive the gizmo should attach to, if any. */
  gizmoTarget(): { node: ConstructionNode; origin: Vec3 } | null {
    return this.overlay.gizmoTarget();
  }

  /** Framebuffer size, used by hit testing to match projected pixels. */
  get viewport(): { width: number; height: number } {
    return { width: this.canvas?.width ?? 1, height: this.canvas?.height ?? 1 };
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
   * Direction only. *Isometric* names a direction (a 1:1:1 line through the
   * scene), *orthographic* names a projection (parallel rays): you can look
   * down an isometric direction in either projection, and which one you are
   * in is the toggle beside the cube, not a side effect of clicking a corner.
   *
   * The zoom is left alone too. Detents belong to the control that sets the
   * zoom (the wheel, in `zoomCamera`), not to the one that sets the
   * direction; snapping the distance here would make one press change two
   * things.
   */
  applyViewPreset(name: string): void {
    const preset = VIEW_PRESETS[name];
    if (!preset) return;
    this.camera = { ...this.camera, yaw: preset.yaw, pitch: preset.pitch };
    this.invalidate();
  }

  private displayFlags(): number {
    // The inspector's faces replace the solid exactly as the FEM surface does.
    return displayFlags(this.display, this.simulation.active || this.brepActive);
  }

  /** Whether the B-rep inspector's faces stand in for the raymarched solid. */
  private get brepActive(): boolean {
    return this.display.showBrep === true && this.brep.hasFaces && !this.simulation.active;
  }

  /**
   * Replace the B-rep inspector's geometry (null clears it).
   *
   * Takes the packed arrays rather than the payload: colouring, the legend's
   * hidden kinds and the hover are the inspector's business (src/brep.ts),
   * and a hover is a re-pack of one buffer, not a new B-rep.
   */
  setBrep(geometry: BrepGeometry | null): void {
    if (!this.device) return;
    this.brep.set(this.device, geometry);
    // The solid's visibility rides in the scene uniform's flags.
    this.invalidate();
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
    this.overlay.attach(overlay);
    this.brep.attach(overlay);

    const graticule = createGraticulePipeline(device, this.format, this.graticuleBuffer);
    this.graticulePipeline = graticule.graticulePipeline;
    this.graticuleBindGroup = graticule.graticuleBindGroup;

    this.simulation.attach(
      createSimulationPipelines(device, this.format, (size) => this.createUniform(size)),
    );

    // Draw the empty viewport as soon as there is a device to draw it with.
    // Before the first compile there is no scene shader and nothing else would
    // ask for a frame, and an unpainted swap chain shows black.
    this.invalidate();
  }

  /** Whether the FEM surface replaces the raymarched solid. */
  get simulationActive(): boolean {
    return this.simulation.active;
  }

  set simulationActive(active: boolean) {
    if (active === this.simulation.active) return;
    this.simulation.active = active;
    // The skeleton changes tone over a field (`packConstructionOverlay`).
    this.uploadOverlay();
    this.invalidate();
  }

  /** Replace the FEM surface mesh (null clears it); resets view overrides. */
  setSimulationMesh(payload: SurfacePayload | null): void {
    this.simulation.setMesh(this.device, payload);
    this.invalidate();
  }

  get simulationEdgesVisible(): boolean {
    return this.simulation.edgesVisible;
  }

  /** Show or hide the element-edge overlay on the simulation surface. */
  set simulationEdgesVisible(visible: boolean) {
    if (visible === this.simulation.edgesVisible) return;
    this.simulation.edgesVisible = visible;
    this.invalidate();
  }

  /** Which ramp fs_sim applies: solved fields vs mesh quality. */
  get simulationRamp(): "field" | "quality" {
    return this.simulation.ramp;
  }

  set simulationRamp(ramp: "field" | "quality") {
    this.simulation.ramp = ramp;
  }

  /** Swap the displayed nodal field without re-solving or re-meshing. */
  setSimulationScalars(scalars: readonly number[] | null, range?: [number, number]): void {
    this.simulation.setScalars(this.device, scalars, range);
    this.invalidate();
  }

  /** Deformed view: draw the surface at offset positions (null = undeformed). */
  setSimulationPositions(positions: readonly number[] | null): void {
    this.simulation.setPositions(this.device, positions);
    this.invalidate();
  }

  /** Per-vertex RGBA highlight tint (BC previews); null clears it. */
  setSimulationOverlay(colors: Float32Array | null): void {
    this.simulation.setOverlay(this.device, colors);
    this.invalidate();
  }

  /** Outline one face group's triangle range on hover (null clears it). */
  setSimulationHighlight(range: IndexRange | null): void {
    this.simulation.setHighlight(this.device, range);
    this.scheduleRender();
  }

  /** Move the ParaView-style clip plane of the simulation surface. */
  setSimulationClip(clip: SliceState): void {
    this.simulation.setClip(clip);
    this.scheduleRender();
  }

  /**
   * Compile and install a freshly generated scene; see
   * `ScenePrograms.setShaders`.
   */
  setShaders(shaders: Shaders): Promise<void> {
    if (!this.device) {
      this.callbacks.onError?.(this.initError || "WGSL compiled, but WebGPU is unavailable.");
      return Promise.resolve();
    }
    return this.scene.setShaders(this.device, shaders);
  }

  /** Whether the path tracer is built for the installed scene. */
  get pathReady(): boolean {
    return this.scene.pathReady;
  }

  /** Render pipelines created since construction; see `shaderStats`. */
  get pipelineBuilds(): number {
    return this.scene.pipelineBuilds;
  }

  /** Parameter uploads that stood in for a pipeline rebuild. */
  get parameterUploads(): number {
    return this.scene.parameterUploads;
  }

  /**
   * Show these parameter values now, without a compile; see
   * `ScenePrograms.setParameterOverrides`.
   *
   * @returns False in the literal form, where there is no buffer to write
   *   and the caller must wait for the round trip.
   */
  setParameterOverrides(
    overrides: Readonly<Record<string, readonly number[]>> | null,
  ): boolean {
    return this.scene.setParameterOverrides(overrides);
  }

  /** The installed scene's uniform contract, or null in the literal form. */
  get parameterProgram(): ShaderProgramPayload | null {
    return this.scene.parameterProgram;
  }

  /** Drop a live drag preview that no compile came along to replace. */
  dropStaleParameterOverrides(): void {
    this.scene.dropStaleParameterOverrides();
  }

  /** Whether the installed scene reads its parameters from a buffer. */
  get hasParameterBuffer(): boolean {
    return this.scene.hasParameterBuffer;
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
      pipelineBuilds: this.scene.pipelineBuilds,
      parameterUploads: this.scene.parameterUploads,
      hasParameterBuffer: this.scene.hasParameterBuffer,
      ...this.scene.moduleStats,
    };
  }

  /** Replace the construction geometry drawn on top of the scene. */
  setConstruction(
    profiles: readonly ConstructionNode[],
    selection: Selection | null,
    hover: Selection | null,
    planes: readonly PlaneFrame[] = planeFrames(profiles),
  ): void {
    // Which planes are drawn is the pane's decision (mode, selection,
    // hover, the display switches); the renderer draws what it is handed.
    this.overlay.setConstruction(profiles, selection, hover, planes);
    this.refreshOverlay();
  }

  /**
   * The construction nodes under the boolean the object tree picked: their
   * skeletons draw as selected, since the boolean has no outline of its own.
   */
  setFocusGroup(nodes: ReadonlySet<string>): void {
    if (this.overlay.setFocusGroup(nodes)) this.refreshOverlay();
  }

  /** Replace the dual-contour mesh edges gated by the mesh-edges display flag. */
  setMeshEdges(payload: MeshEdgePayload | null): void {
    this.overlay.setMeshEdges(payload);
    this.refreshOverlay();
  }

  /** Show (or drop) the ghost of the plane a sketch click would create. */
  setPlanePreview(frame: PlaneFrame | null): void {
    if (this.overlay.setPlanePreview(frame)) this.refreshOverlay();
  }

  /**
   * Highlight one analytic face, or clear the highlight.
   *
   * Takes the face itself rather than a polygon: the fill has to be
   * triangulated in the face's own frame, and `usable` decides the weight —
   * a face the source cannot name is still shown, at half strength.
   */
  setFaceHighlight(face: ConstructionFace | null): void {
    if (this.overlay.setFaceHighlight(face)) this.refreshOverlay();
  }

  private uploadOverlay(): void {
    if (!this.device) return;
    this.overlay.upload(this.device, this.simulation.active, this.view);
  }

  private refreshOverlay(): void {
    this.uploadOverlay();
    this.scheduleRender();
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
        layout: this.scene.pathPipeline!.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuffer } },
          { binding: 1, resource: texture.createView() },
        ],
      }),
    );
    this.presentBindGroups = this.accumulation.map((texture) =>
      device.createBindGroup({
        layout: this.scene.presentPipeline!.getBindGroupLayout(0),
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
    this.scene.syncCullMargin();
    const position = cameraPosition(this.camera);
    const size = { width: this.canvas.width, height: this.canvas.height };
    device.queue.writeBuffer(
      this.uniformBuffer,
      0,
      packSceneUniforms({
        size,
        camera: this.camera,
        position,
        display: this.display,
        quality: this.quality,
        sampleCount: this.sampleCount,
        flags: this.displayFlags(),
      }),
    );

    const matrix = viewProjection(this.view);
    device.queue.writeBuffer(this.viewBuffer, 0, matrix);
    device.queue.writeBuffer(
      this.overlayBuffer,
      0,
      packOverlayUniforms(matrix, position, size, DEPTH_NUDGE, CASING_PX),
    );
    // Mesh edges sit exactly on the surface creases the construction
    // wireframe also traces; a smaller nudge keeps them consistently behind
    // coincident construction lines instead of z-fighting into dashes.
    device.queue.writeBuffer(
      this.meshOverlayBuffer,
      0,
      packOverlayUniforms(matrix, position, size, DEPTH_NUDGE * 0.4, 0),
    );

    if (this.display.showGraticule) {
      device.queue.writeBuffer(
        this.graticuleBuffer,
        0,
        packGraticuleUniforms({
          size,
          cssWidth: this.canvas.clientWidth,
          camera: this.camera,
          position,
          projection: this.display.projection,
          emphasis: this.groundEmphasis,
        }),
      );
    }

    this.simulation.writeUniforms(device, matrix, position, this.canvas.width);
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
    // The inspector's faces are opaque: x-ray is the raymarched solid's, and
    // a skeleton drawn through the B-rep would read as one of its edges.
    return this.display.xray > 0 && !this.brepActive;
  }

  private drawOverlay(pass: GPURenderPassEncoder): void {
    this.overlay.draw(pass, this.display, this.seeThroughOverlays);
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

  /** Wall-clock times of recent frames, for the viewport's FPS readout. */
  private frameLog: number[] = [];
  /** CPU milliseconds spent encoding and submitting each recent frame. */
  private cpuLog: number[] = [];

  /**
   * What the last second of drawing cost, for the performance readout. `fps`
   * is null when nothing was drawn for a second (rendering is on demand);
   * `cpuMs` is CPU encode-and-submit time, not GPU time.
   */
  get frameStats(): { fps: number | null; cpuMs: number | null; width: number; height: number } {
    const now = performance.now();
    const recent = this.frameLog.filter((time) => now - time <= 1000);
    const cpu = this.cpuLog.length
      ? this.cpuLog.reduce((sum, value) => sum + value, 0) / this.cpuLog.length
      : null;
    return {
      fps: recent.length ? recent.length : null,
      cpuMs: recent.length ? cpu : null,
      // Before the viewport mounts there is no canvas to measure.
      width: this.canvas?.width ?? 0,
      height: this.canvas?.height ?? 0,
    };
  }

  private render(): void {
    const started = performance.now();
    this.renderFrame();
    const ended = performance.now();
    this.frameLog.push(ended);
    if (this.frameLog.length > 240) this.frameLog.splice(0, this.frameLog.length - 240);
    this.cpuLog.push(ended - started);
    if (this.cpuLog.length > 30) this.cpuLog.shift();
  }

  private renderFrame(): void {
    this.framePending = false;
    if (!this.device || !this.context) return;
    this.resize();
    const scene = this.scene;
    const { previewPipeline, previewBindGroup } = scene;
    if (!previewPipeline || !previewBindGroup) {
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
      this.pathTracing &&
      scene.pathReady &&
      !this.interacting &&
      !this.simulation.active &&
      !this.brepActive;
    /** Pixels the preview march ran at this frame; 0 when it did not run. */
    let previewPixels = 0;

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
      tracePass.setPipeline(scene.pathPipeline!);
      tracePass.setBindGroup(0, this.pathBindGroups[this.readIndex]);
      scene.bindParameters(tracePass, "path");
      tracePass.draw(3);
      tracePass.end();

      const presentPass = encoder.beginRenderPass({
        colorAttachments: [
          { view: swapchain, clearValue: BACKGROUND, loadOp: "clear", storeOp: "store" },
        ],
        depthStencilAttachment: depthAttachment,
      });
      presentPass.setPipeline(scene.presentPipeline!);
      presentPass.setBindGroup(0, this.presentBindGroups[writeIndex]);
      presentPass.draw(3);
      // Re-run the cheap preview purely for depth so overlays interleave with
      // the path-traced image, which carries no depth of its own. The
      // graticule needs it for the same reason — without a depth buffer it
      // would paint over the part instead of behind it.
      const wantDepth =
        (this.overlay.hasSkeletonEdges && this.display.showSketches) || this.display.showGraticule;
      if (wantDepth && scene.previewDepthPipeline) {
        presentPass.setPipeline(scene.previewDepthPipeline);
        presentPass.setBindGroup(0, previewBindGroup);
        scene.bindParameters(presentPass, "preview");
        presentPass.draw(3);
      }
      this.drawGraticule(presentPass);
      this.drawOverlay(presentPass);
      presentPass.end();
      this.readIndex = writeIndex;
    } else {
      // A gesture on a scene too slow for the frame budget: raymarch into the
      // small targets, then let the upscale pass stand in for the preview
      // draw. The FEM surface is a mesh and draws at full size regardless.
      const scale = this.simulation.active || this.brepActive ? 1 : this.gestureScale;
      if (scale < 1) {
        const size = this.ensureGestureTargets(scale);
        // The march derives its rays from `u.resolution`, so the small pass
        // has to be told its own size; the next frame's `writeUniforms`
        // restores the full one.
        device.queue.writeBuffer(this.uniformBuffer, 0, new Float32Array([size.width, size.height]));
        const smallPass = encoder.beginRenderPass({
          colorAttachments: [
            {
              view: this.gestureColor!.createView(),
              clearValue: BACKGROUND,
              loadOp: "clear",
              storeOp: "store",
            },
          ],
          depthStencilAttachment: {
            view: this.gestureDepth!.createView(),
            depthClearValue: 1,
            depthLoadOp: "clear",
            depthStoreOp: "store",
          },
        });
        smallPass.setPipeline(previewPipeline);
        smallPass.setBindGroup(0, previewBindGroup);
        scene.bindParameters(smallPass, "preview");
        smallPass.draw(3);
        smallPass.end();
      }
      previewPixels = scale < 1
        ? this.gestureColor!.width * this.gestureColor!.height
        : this.canvas.width * this.canvas.height;
      const previewPass = encoder.beginRenderPass({
        colorAttachments: [
          { view: swapchain, clearValue: BACKGROUND, loadOp: "clear", storeOp: "store" },
        ],
        depthStencilAttachment: depthAttachment,
      });
      if (scale < 1) {
        previewPass.setPipeline(this.upscalePipeline!);
        previewPass.setBindGroup(0, this.gestureBindGroup);
      } else {
        previewPass.setPipeline(previewPipeline);
        previewPass.setBindGroup(0, previewBindGroup);
        scene.bindParameters(previewPass, "preview");
      }
      previewPass.draw(3);
      this.simulation.draw(previewPass, this.display.showOverlays);
      if (this.brepActive) this.brep.draw(previewPass);
      // After every pass that writes depth, before every pass that does not:
      // the faceplate is occluded by the part and the FEM surface, and the
      // construction overlays are drawn on top of it.
      this.drawGraticule(previewPass);
      this.drawOverlay(previewPass);
      previewPass.end();
    }

    device.queue.submit([encoder.finish()]);
    if (previewPixels > 0) {
      // Submit to completion: the GPU's time on this frame, plus whatever was
      // still queued ahead of it — an overestimate when frames pile up, which
      // only ever errs toward a smaller gesture preview.
      const submitted = performance.now();
      void device.queue
        .onSubmittedWorkDone()
        .then(() => this.recordPreviewCost(performance.now() - submitted, previewPixels));
    }

    if (tracing) {
      this.sampleCount += 1;
      this.callbacks.onStatus?.(
        "ready",
        `${this.quality.label} path trace · ${this.adapterLabel} · ${this.sampleCount} spp`,
      );
      if (this.sampleCount < this.quality.samples) this.scheduleRender();
    } else {
      const suffix = this.pathTracing
        ? scene.pathReady
          ? " · moving"
          : " · preparing path tracer…"
        : "";
      this.callbacks.onStatus?.(
        "ready",
        `${this.quality.label} preview · ${this.adapterLabel}${suffix}`,
      );
    }
  }

  /**
   * Time the preview: `frames` redraws while the camera turns, each waited
   * on until the GPU has finished it.
   *
   * Serialising on `onSubmittedWorkDone` costs the pipelining a live orbit
   * would get, so the number is an upper bound on frame time — but it is the
   * same bound before and after a change, which is what it is for. Published
   * as `__cadjointBench` for the performance study and the e2e tests.
   *
   * @param frames Frames to time after two warm-up frames.
   * @param interacting Draw as during a drag rather than at rest.
   * @returns Per-frame milliseconds and the framebuffer they were drawn at.
   */
  async benchmark(
    frames = 30,
    interacting = false,
  ): Promise<{ medianMs: number; meanMs: number; samples: number[]; width: number; height: number }> {
    const device = this.device;
    if (!device || !this.scene.previewPipeline) {
      return { medianMs: NaN, meanMs: NaN, samples: [], width: 0, height: 0 };
    }
    const wasInteracting = this.interacting;
    const startYaw = this.camera.yaw;
    this.interacting = interacting;
    const samples: number[] = [];
    let width = 0;
    let height = 0;
    for (let index = -2; index < frames; index++) {
      this.camera = { ...this.camera, yaw: startYaw + (index + 2) * 0.01 };
      this.sampleCount = 0;
      const started = performance.now();
      this.renderFrame();
      await device.queue.onSubmittedWorkDone();
      if (index >= 0) samples.push(performance.now() - started);
      width = this.canvas.width;
      height = this.canvas.height;
    }
    this.interacting = wasInteracting;
    this.camera = { ...this.camera, yaw: startYaw };
    this.invalidate();
    const sorted = [...samples].sort((a, b) => a - b);
    return {
      medianMs: sorted[Math.floor(sorted.length / 2)],
      meanMs: samples.reduce((sum, value) => sum + value, 0) / samples.length,
      samples,
      width,
      height,
    };
  }

  /**
   * Time a cold compile of the installed scene: to the first preview frame
   * finished on the GPU, and to the path tracer being ready.
   *
   * A comment unique to this call is appended to every source so neither
   * this cache nor the browser's own shader cache can serve it.
   */
  async benchmarkCompile(): Promise<{ firstFrameMs: number; pathReadyMs: number }> {
    const shaders = this.scene.installedShaders;
    const device = this.device;
    if (!shaders || !device) return { firstFrameMs: NaN, pathReadyMs: NaN };
    const salt = `\n// bench ${performance.now()} ${Math.random()}\n`;
    const started = performance.now();
    const done = this.setShaders({
      ...shaders,
      preview: shaders.preview + salt,
      path: shaders.path + salt,
      present: shaders.present + salt,
    });
    let firstFrameMs = NaN;
    const firstFrame = (async () => {
      while (this.scene.installedPreview !== shaders.preview + salt) {
        await new Promise((resolve) => requestAnimationFrame(resolve));
      }
      this.renderFrame();
      await device.queue.onSubmittedWorkDone();
      firstFrameMs = performance.now() - started;
    })();
    await done;
    const pathReadyMs = performance.now() - started;
    await firstFrame;
    return { firstFrameMs, pathReadyMs };
  }

  destroy(): void {
    this.destroyAccumulation();
    this.gestureColor?.destroy();
    this.gestureDepth?.destroy();
    this.depthTexture?.destroy();
    this.overlay.destroy();
    this.brep.destroy();
    this.simulation.destroy();
    this.device?.destroy();
  }
}
