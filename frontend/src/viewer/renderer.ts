/**
 * WebGPU renderer for the playground viewer.
 *
 * Draws the compiled SDF scene (raymarched preview or progressive path trace)
 * and the construction-tree overlay on top of it. The preview pass writes
 * `frag_depth` from the ray hit position, so overlay edges and handles are
 * depth-tested against the solid rather than floating over it.
 *
 * Ported from the previous inline playground script, with the depth attachment,
 * the overlay pipelines, and pan support added.
 */

import type { ConstructionProfile, Selection } from "../types";
import {
  cameraPosition,
  viewProjection,
  type CameraState,
  type Vec3,
} from "./math";
// Kept as a standalone .wgsl file so the shader has one source of truth that
// both the bundler and the Python shader-validation test read.
import OVERLAY_WGSL from "./overlay.wgsl?raw";

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

const DEPTH_FORMAT: GPUTextureFormat = "depth32float";
const BACKGROUND: GPUColorDict = { r: 0.035, g: 0.045, b: 0.035, a: 1 };

/** Fraction of the distance to the camera that overlays are pulled forward. */
const DEPTH_NUDGE = 0.004;
const LINE_WIDTH_PX = 2.4;
const HANDLE_RADIUS_PX = 6.5;

const EDGE_STRIDE = 40;
const HANDLE_STRIDE = 32;

type Rgba = readonly [number, number, number, number];

const COLORS: Record<string, Rgba> = {
  edge: [0.851, 1.0, 0.341, 0.95],
  edgeLocked: [0.58, 0.6, 0.56, 0.7],
  handle: [1.0, 0.506, 0.404, 1.0],
  handleSelected: [0.98, 0.99, 0.94, 1.0],
  handleHover: [1.0, 0.72, 0.4, 1.0],
  handleLocked: [0.62, 0.64, 0.6, 0.9],
};

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

  private previewPipeline: GPURenderPipeline | null = null;
  private previewDepthPipeline: GPURenderPipeline | null = null;
  private previewBindGroup: GPUBindGroup | null = null;
  private pathPipeline: GPURenderPipeline | null = null;
  private presentPipeline: GPURenderPipeline | null = null;
  private edgePipeline: GPURenderPipeline | null = null;
  private handlePipeline: GPURenderPipeline | null = null;
  private overlayBindGroup: GPUBindGroup | null = null;

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

  private shaderRevision = 0;
  private framePending = false;
  private initError = "";

  camera: CameraState = { yaw: 0.75, pitch: 0.32, distance: 4.6, target: [0, 0, 0] };
  quality: QualityPreset = QUALITY_PRESETS.high;
  pathTracing = false;
  pathReady = false;
  interacting = false;

  private profiles: readonly ConstructionProfile[] = [];
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

  /** Framebuffer size, used by hit testing to match projected pixels. */
  get viewport(): { width: number; height: number } {
    return { width: this.canvas?.width ?? 1, height: this.canvas?.height ?? 1 };
  }

  get cameraPosition(): Vec3 {
    return cameraPosition(this.camera);
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

      this.uniformBuffer = this.createUniform(96);
      this.viewBuffer = this.createUniform(64);
      this.overlayBuffer = this.createUniform(112);
      this.buildOverlayPipelines();

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

  private async checkedModule(code: string, label: string): Promise<GPUShaderModule> {
    const module = this.device!.createShaderModule({ code, label });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    if (errors.length) {
      throw new Error(
        errors.map((m) => `${label} ${m.lineNum}:${m.linePos} ${m.message}`).join("\n"),
      );
    }
    return module;
  }

  /**
   * A bind group layout shared by several pipelines.
   *
   * `layout: "auto"` derives a layout that is *exclusive* to one pipeline, so a
   * bind group built from one pipeline cannot be bound to another even when the
   * bindings are identical. Declaring the layout explicitly lets the edge and
   * handle passes — and the two preview variants — share a single bind group.
   */
  private sharedLayout(label: string, bindings: number[]): {
    bindGroupLayout: GPUBindGroupLayout;
    pipelineLayout: GPUPipelineLayout;
  } {
    const device = this.device!;
    const bindGroupLayout = device.createBindGroupLayout({
      label,
      entries: bindings.map((binding) => ({
        binding,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: "uniform" as const },
      })),
    });
    return {
      bindGroupLayout,
      pipelineLayout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    };
  }

  private buildOverlayPipelines(): void {
    const device = this.device!;
    const module = device.createShaderModule({ code: OVERLAY_WGSL, label: "Overlay WGSL" });
    const { bindGroupLayout, pipelineLayout } = this.sharedLayout("Overlay bindings", [0]);
    const blend: GPUBlendState = {
      color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
      alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
    };
    const depthStencil: GPUDepthStencilState = {
      format: DEPTH_FORMAT,
      depthWriteEnabled: true,
      depthCompare: "less-equal",
    };

    this.edgePipeline = device.createRenderPipeline({
      label: "Overlay edges",
      layout: pipelineLayout,
      vertex: {
        module,
        entryPoint: "vs_edge",
        buffers: [
          {
            arrayStride: EDGE_STRIDE,
            stepMode: "instance",
            attributes: [
              { shaderLocation: 0, offset: 0, format: "float32x3" },
              { shaderLocation: 1, offset: 12, format: "float32x3" },
              { shaderLocation: 2, offset: 24, format: "float32x4" },
            ],
          },
        ],
      },
      fragment: { module, entryPoint: "fs_edge", targets: [{ format: this.format, blend }] },
      primitive: { topology: "triangle-list" },
      depthStencil,
    });

    this.handlePipeline = device.createRenderPipeline({
      label: "Overlay handles",
      layout: pipelineLayout,
      vertex: {
        module,
        entryPoint: "vs_handle",
        buffers: [
          {
            arrayStride: HANDLE_STRIDE,
            stepMode: "instance",
            attributes: [
              { shaderLocation: 0, offset: 0, format: "float32x3" },
              { shaderLocation: 1, offset: 12, format: "float32x4" },
              { shaderLocation: 2, offset: 28, format: "float32" },
            ],
          },
        ],
      },
      fragment: { module, entryPoint: "fs_handle", targets: [{ format: this.format, blend }] },
      primitive: { topology: "triangle-list" },
      depthStencil,
    });

    this.overlayBindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: this.overlayBuffer } }],
    });
  }

  /**
   * Compile and install a freshly generated scene.
   *
   * The preview pipeline lands first so the viewer updates immediately; the
   * path-trace pipelines follow. A revision guard drops results from a compile
   * that has since been superseded.
   */
  async setShaders(shaders: Shaders): Promise<void> {
    if (!this.device) {
      this.callbacks.onError?.(this.initError || "WGSL compiled, but WebGPU is unavailable.");
      return;
    }
    const revision = ++this.shaderRevision;
    this.pathReady = false;

    const previewModule = await this.checkedModule(shaders.preview, "Preview WGSL");
    const preview = this.sharedLayout("Preview bindings", [0, 2]);
    const previewDescriptor = (writeMask: number): GPURenderPipelineDescriptor => ({
      layout: preview.pipelineLayout,
      vertex: { module: previewModule, entryPoint: "vs_main" },
      fragment: {
        module: previewModule,
        entryPoint: "fs_main_depth",
        targets: [{ format: this.format, writeMask }],
      },
      primitive: { topology: "triangle-list" },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: "less" },
    });

    const [colorPipeline, depthOnlyPipeline] = await Promise.all([
      this.device.createRenderPipelineAsync(previewDescriptor(GPUColorWrite.ALL)),
      this.device.createRenderPipelineAsync(previewDescriptor(0)),
    ]);
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
    this.destroyAccumulation();
    this.invalidate();

    const [pathModule, presentModule] = await Promise.all([
      this.checkedModule(shaders.path, "Path tracer WGSL"),
      this.checkedModule(shaders.present, "Present WGSL"),
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
    if (revision !== this.shaderRevision) return;
    this.pathPipeline = pathPipeline;
    this.presentPipeline = presentPipeline;
    this.pathReady = true;
    this.invalidate();
  }

  /** Replace the construction geometry drawn on top of the scene. */
  setConstruction(
    profiles: readonly ConstructionProfile[],
    selection: Selection | null,
    hover: Selection | null,
  ): void {
    this.profiles = profiles;
    this.selection = selection;
    this.hover = hover;
    this.uploadOverlay();
    this.scheduleRender();
  }

  private uploadOverlay(): void {
    if (!this.device) return;
    const edges: number[] = [];
    const handles: number[] = [];

    for (const profile of this.profiles) {
      const count = profile.vertices.length;
      const edgeColor = profile.editable ? COLORS.edge : COLORS.edgeLocked;
      for (let index = 0; index < count; index++) {
        const start = profile.vertices[index].world;
        const end = profile.vertices[(index + 1) % count].world;
        edges.push(start[0], start[1], start[2], end[0], end[1], end[2], ...edgeColor);
      }
      for (let index = 0; index < count; index++) {
        const selected =
          this.selection?.profileId === profile.id && this.selection.vertexIndex === index;
        const hovered = this.hover?.profileId === profile.id && this.hover.vertexIndex === index;
        const color = !profile.editable
          ? COLORS.handleLocked
          : selected
            ? COLORS.handleSelected
            : hovered
              ? COLORS.handleHover
              : COLORS.handle;
        const position = profile.vertices[index].world;
        handles.push(position[0], position[1], position[2], ...color, selected || hovered ? 1 : 0);
      }
    }

    this.edgeCount = edges.length / (EDGE_STRIDE / 4);
    this.handleCount = handles.length / (HANDLE_STRIDE / 4);
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
    const scene = new Float32Array([
      this.canvas.width, this.canvas.height, 0, 0,
      position[0], position[1], position[2], 0,
      this.camera.target[0], this.camera.target[1], this.camera.target[2], 0,
      0.55, 0.8, 0.35, 3,
      BACKGROUND.r, BACKGROUND.g, BACKGROUND.b, 1,
      this.sampleCount, this.quality.bounces, this.quality.shadowSamples, 0,
    ]);
    device.queue.writeBuffer(this.uniformBuffer, 0, scene);

    const aspect = this.canvas.width / this.canvas.height;
    const matrix = viewProjection(position, this.camera.target, aspect);
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
  }

  private drawOverlay(pass: GPURenderPassEncoder): void {
    if (!this.overlayBindGroup) return;
    if (this.edgeCount && this.edgeBuffer && this.edgePipeline) {
      pass.setPipeline(this.edgePipeline);
      pass.setBindGroup(0, this.overlayBindGroup);
      pass.setVertexBuffer(0, this.edgeBuffer);
      pass.draw(6, this.edgeCount);
    }
    if (this.handleCount && this.handleBuffer && this.handlePipeline) {
      pass.setPipeline(this.handlePipeline);
      pass.setBindGroup(0, this.overlayBindGroup);
      pass.setVertexBuffer(0, this.handleBuffer);
      pass.draw(6, this.handleCount);
    }
  }

  private render(): void {
    this.framePending = false;
    if (!this.device || !this.context || !this.previewPipeline || !this.previewBindGroup) return;
    this.resize();
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
    const tracing = this.pathTracing && this.pathReady && !this.interacting;

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
      // the path-traced image, which carries no depth of its own.
      if (this.edgeCount && this.previewDepthPipeline) {
        presentPass.setPipeline(this.previewDepthPipeline);
        presentPass.setBindGroup(0, this.previewBindGroup);
        presentPass.draw(3);
      }
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
      previewPass.draw(3);
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
    this.handleBuffer?.destroy();
    this.device?.destroy();
  }
}
