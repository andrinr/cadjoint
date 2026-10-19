/**
 * The compiled SDF scene: its shader modules, its four pipelines, and the
 * parameter buffer a uniform-form scene reads its design values from.
 *
 * A scene shader built in the uniform form reads its design parameters out
 * of a `@group(3)` buffer instead of carrying them as constants, so the
 * source does not change when a value does. Most of this class exists to
 * notice that and answer a parameter edit with a `writeBuffer` rather than a
 * shader module and four pipelines.
 */

import type { Shaders } from "./display";
import { DEPTH_FORMAT, sharedLayout } from "./pipelines";
import {
  CULL_MARGIN_OFF,
  CULL_MARGIN_ON,
  PARAMETER_SLOT_BYTES,
  ShaderModuleCache,
  packParameters,
  sameLayout,
  type ShaderProgramPayload,
} from "./shaderProgram";

/** What the scene needs from the renderer that owns it. */
export interface SceneHost {
  device(): GPUDevice | null;
  /** The swapchain format the preview and present passes target. */
  format(): GPUTextureFormat;
  /** The camera uniforms (binding 0) and the view matrix (binding 2) of group 0. */
  uniformBuffers(): { scene: GPUBuffer; view: GPUBuffer };
  /** `display.cullBounds`: whether the scene's skip tests get their margin. */
  cullBounds(): boolean;
  /** A different preview pipeline was installed; its frame cost is unmeasured. */
  previewReplaced(): void;
  /** What the scene draws changed: drop the accumulated samples and redraw. */
  sceneChanged(): void;
  /** Redraw. */
  invalidate(): void;
}

/** Which scene pipeline a parameter bind group belongs to. */
export type ScenePass = "preview" | "path";

export class ScenePrograms {
  /** Compiled modules held by source, so an undo lands on a hit. */
  private shaderModules = new ShaderModuleCache();
  private _installedShaders: Shaders | null = null;
  private _installedPreview: string | null = null;
  /** The scene's uniform contract; null in the literal form. */
  private program: ShaderProgramPayload | null = null;
  private parameterBuffer: GPUBuffer | null = null;
  private parameterBufferBytes = 0;
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
  /** `cullBounds` as of the last parameter upload. */
  private uploadedCullBounds: boolean;
  private revision = 0;

  private _previewPipeline: GPURenderPipeline | null = null;
  private _previewDepthPipeline: GPURenderPipeline | null = null;
  private _previewBindGroup: GPUBindGroup | null = null;
  private _pathPipeline: GPURenderPipeline | null = null;
  private _presentPipeline: GPURenderPipeline | null = null;
  private _pathReady = false;
  private _pipelineBuilds = 0;
  private _parameterUploads = 0;

  constructor(
    private readonly host: SceneHost,
    /** The cull setting the renderer starts with. */
    initialCullBounds: boolean,
  ) {
    this.uploadedCullBounds = initialCullBounds;
  }

  /** The sources currently installed, or null before the first compile. */
  get installedShaders(): Shaders | null {
    return this._installedShaders;
  }

  /** The preview source the installed preview pipeline was built from. */
  get installedPreview(): string | null {
    return this._installedPreview;
  }

  get previewPipeline(): GPURenderPipeline | null {
    return this._previewPipeline;
  }

  /** The preview with colour writes off: depth for overlays over a path trace. */
  get previewDepthPipeline(): GPURenderPipeline | null {
    return this._previewDepthPipeline;
  }

  get previewBindGroup(): GPUBindGroup | null {
    return this._previewBindGroup;
  }

  get pathPipeline(): GPURenderPipeline | null {
    return this._pathPipeline;
  }

  get presentPipeline(): GPURenderPipeline | null {
    return this._presentPipeline;
  }

  /** Whether the path tracer and present pipelines match the installed scene. */
  get pathReady(): boolean {
    return this._pathReady;
  }

  /**
   * Render pipelines created since construction: the thing the drag test
   * asserts about, since a drag that only moves parameter values must not
   * advance it at all.
   */
  get pipelineBuilds(): number {
    return this._pipelineBuilds;
  }

  /** Parameter uploads that stood in for a pipeline rebuild. */
  get parameterUploads(): number {
    return this._parameterUploads;
  }

  /** Shader modules served from cache, and compiled. */
  get moduleStats(): { hits: number; misses: number } {
    return { hits: this.shaderModules.hits, misses: this.shaderModules.misses };
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
   * `createShaderModule` and no pipeline at all.
   */
  async setShaders(device: GPUDevice, shaders: Shaders): Promise<void> {
    const program = shaders.program ?? null;
    if (this.isValuesOnlyEdit(shaders, program)) {
      // Newest values win: an older rebuild still in flight reads `program`
      // rather than the payload it started from, so it cannot resurrect them.
      this.program = program;
      this.parameterOverrides = null;
      this.uploadParameters();
      this._parameterUploads += 1;
      this.host.sceneChanged();
      return;
    }

    const revision = ++this.revision;
    // Nothing about the installed scene changes yet. Everything below the
    // first `await` takes real time — seconds of module and pipeline building
    // for a large scene — and every frame drawn in that window still draws
    // the *previous* scene, which it can only do if the previous scene is
    // whole: its bind groups hold its own buffer, and the draw reads its own
    // program for the group index to bind them at. The replacement is one
    // synchronous block, after the pipelines exist and after the revision
    // guard.
    this._pathReady = false;

    const format = this.host.format();
    const previewModule = await this.shaderModules.get(device, shaders.preview, "Preview WGSL");
    const preview = sharedLayout(device, "Preview bindings", [0, 2]);
    const parameterLayout = this.parameterLayoutFor(program);
    const previewLayout = device.createPipelineLayout({
      bindGroupLayouts: this.sceneBindGroupLayouts(
        preview.bindGroupLayout,
        program,
        parameterLayout,
      ),
    });
    const previewDescriptor = (writeMask: number): GPURenderPipelineDescriptor => ({
      layout: previewLayout,
      vertex: { module: previewModule, entryPoint: "vs_main" },
      fragment: {
        module: previewModule,
        entryPoint: "fs_main_depth",
        targets: [{ format, writeMask }],
      },
      primitive: { topology: "triangle-list" },
      // "always": this fullscreen pass establishes both colour and depth, and a
      // ray miss writes depth 1.0 — which "less" would reject against the 1.0
      // clear, discarding every background fragment.
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: "always" },
    });

    const [colorPipeline, depthOnlyPipeline] = await Promise.all([
      device.createRenderPipelineAsync(previewDescriptor(GPUColorWrite.ALL)),
      device.createRenderPipelineAsync(previewDescriptor(0)),
    ]);
    this._pipelineBuilds += 2;
    if (revision !== this.revision) return;

    // ── the swap ─────────────────────────────────────────────────────────
    // Synchronous, and in this order: the program and its buffer, then every
    // bind group that holds them. No `await` may appear between here and the
    // last assignment, because a frame drawn halfway through would mix the
    // two scenes.
    this.program = program;
    this.parameterOverrides = null;
    this.adoptParameterBuffer();
    this._previewPipeline = colorPipeline;
    this._installedPreview = shaders.preview;
    // A different scene costs differently; measure it afresh.
    this.host.previewReplaced();
    this._previewDepthPipeline = depthOnlyPipeline;
    const uniforms = this.host.uniformBuffers();
    this._previewBindGroup = device.createBindGroup({
      layout: preview.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: uniforms.scene } },
        { binding: 2, resource: { buffer: uniforms.view } },
      ],
    });
    this.previewParameterGroup = this.parameterBindGroup(parameterLayout);
    // The path pipeline is still the outgoing one and is not rebuilt until
    // below; its group belongs to the buffer that just went out of use.
    // `pathReady` is false, so nothing draws it — dropping it says so.
    this.pathParameterGroup = null;
    this.uploadParameters();
    this.host.sceneChanged();

    const [pathModule, presentModule] = await Promise.all([
      this.shaderModules.get(device, shaders.path, "Path tracer WGSL"),
      this.shaderModules.get(device, shaders.present, "Present WGSL"),
    ]);
    const [pathPipeline, presentPipeline] = await Promise.all([
      device.createRenderPipelineAsync({
        layout: "auto",
        vertex: { module: pathModule, entryPoint: "vs_main" },
        fragment: {
          module: pathModule,
          entryPoint: "fs_path_trace",
          targets: [{ format: "rgba16float" }],
        },
        primitive: { topology: "triangle-list" },
      }),
      device.createRenderPipelineAsync({
        layout: "auto",
        vertex: { module: presentModule, entryPoint: "vs_present" },
        fragment: {
          module: presentModule,
          entryPoint: "fs_present",
          targets: [{ format }],
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
    this._pipelineBuilds += 2;
    if (revision !== this.revision) return;
    this._pathPipeline = pathPipeline;
    this._presentPipeline = presentPipeline;
    // The path pipeline derives its own layout, so its parameter bind group
    // has to come from that layout rather than the preview's: an automatic
    // layout is exclusive to the pipeline that produced it.
    this.pathParameterGroup = this.program
      ? this.parameterBindGroup(pathPipeline.getBindGroupLayout(this.program.group))
      : null;
    this._installedShaders = { ...shaders };
    this._pathReady = true;
    this.host.invalidate();
  }

  /** Bind the parameter block for one scene pipeline, if the scene has one. */
  bindParameters(pass: GPURenderPassEncoder, kind: ScenePass): void {
    const group = kind === "preview" ? this.previewParameterGroup : this.pathParameterGroup;
    if (group) pass.setBindGroup(this.program!.group, group);
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
  private isValuesOnlyEdit(shaders: Shaders, program: ShaderProgramPayload | null): boolean {
    const installed = this._installedShaders;
    return Boolean(
      installed &&
        this._previewPipeline &&
        this._pathPipeline &&
        shaders.preview === installed.preview &&
        shaders.path === installed.path &&
        shaders.present === installed.present &&
        sameLayout(program, this.program),
    );
  }

  /**
   * The parameter block's layout for a program that is not installed yet.
   *
   * Takes the program rather than reading `this.program`, because it is
   * called while the *previous* program is still the installed one: nothing
   * about a scene is swapped in until its pipelines exist.
   */
  private parameterLayoutFor(program: ShaderProgramPayload | null): GPUBindGroupLayout | null {
    const device = this.host.device();
    if (!device || !program) return null;
    return device.createBindGroupLayout({
      label: "SDF parameter bindings",
      entries: [
        {
          binding: program.binding,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          // Storage, not uniform: `PARAMETER_ADDRESS_SPACE` in
          // `cadjoint/backends/wgsl/codegen.py` has the measurements.
          buffer: { type: "read-only-storage" },
        },
      ],
    });
  }

  /**
   * The bind group layouts of a scene pipeline, group by group.
   *
   * Group 0 is the camera and view uniforms every scene shader has. A
   * uniform-form shader adds its parameter block at the group the payload
   * names, and WebGPU needs the groups in between spelled out, so they are
   * declared empty. An empty layout binds nothing and needs no bind group
   * set against it.
   */
  private sceneBindGroupLayouts(
    scene: GPUBindGroupLayout,
    program: ShaderProgramPayload | null,
    parameterLayout: GPUBindGroupLayout | null,
  ): GPUBindGroupLayout[] {
    const device = this.host.device();
    const layouts = [scene];
    if (!device || !program || !parameterLayout) return layouts;
    while (layouts.length < program.group) {
      layouts.push(device.createBindGroupLayout({ entries: [] }));
    }
    layouts.push(parameterLayout);
    return layouts;
  }

  /** The parameter buffer bound against one layout, or null without a program. */
  private parameterBindGroup(layout: GPUBindGroupLayout | null): GPUBindGroup | null {
    const device = this.host.device();
    if (!device || !this.program || !this.parameterBuffer || !layout) return null;
    return device.createBindGroup({
      label: "SDF parameters",
      layout,
      entries: [{ binding: this.program.binding, resource: { buffer: this.parameterBuffer } }],
    });
  }

  /**
   * Give the freshly installed program a buffer of its own size.
   *
   * Called from inside the swap, so every bind group that held the outgoing
   * buffer is rebuilt in the same synchronous step. A buffer whose size has
   * not changed is kept and simply rewritten, which is why a values-only
   * edit never needs this at all.
   *
   * The outgoing buffer is deliberately **not** destroyed. It is tiny — a
   * few hundred bytes for the largest shipped scene — so `destroy()`
   * reclaims nothing worth having, and it is live ammunition: one reference
   * left anywhere, in a cached bind group or a frame already queued, and the
   * next draw fails validation with no stack and no way back. Letting the
   * collector take it cannot fail.
   */
  private adoptParameterBuffer(): void {
    const device = this.host.device();
    if (!device) return;
    const size = this.program
      ? Math.max(this.program.buffer_bytes, PARAMETER_SLOT_BYTES)
      : 0;
    if (size === this.parameterBufferBytes) return;
    this.parameterBufferBytes = size;
    this.parameterBuffer = size
      ? device.createBuffer({
          label: "SDF parameters",
          size,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        })
      : null;
  }

  /**
   * Write the current parameter values, drag overrides and cull margin
   * included.
   */
  private uploadParameters(): void {
    const device = this.host.device();
    if (!device || !this.program || !this.parameterBuffer) return;
    const cullBounds = this.host.cullBounds();
    this.uploadedCullBounds = cullBounds;
    const packed = packParameters(
      this.program,
      this.parameterOverrides ?? undefined,
      cullBounds ? CULL_MARGIN_ON : CULL_MARGIN_OFF,
    );
    device.queue.writeBuffer(this.parameterBuffer, 0, packed.buffer as ArrayBuffer);
  }

  /**
   * Re-upload when the cull toggle has moved since the last write.
   *
   * The margin lives in the scene's parameter buffer rather than the viewer's
   * own uniform block — the skip tests are inside the generated shader, which
   * reads nothing else — so it is the one render setting the scene uniforms
   * do not carry. Checked per frame and written only on a change, because
   * the buffer is the drag's hot path and re-packing it every frame for a
   * switch nobody touched would be a cost with no reason.
   */
  syncCullMargin(): void {
    if (!this.program || this.uploadedCullBounds === this.host.cullBounds()) return;
    this.uploadParameters();
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
  setParameterOverrides(overrides: Readonly<Record<string, readonly number[]>> | null): boolean {
    if (!this.program || !this.parameterBuffer) return false;
    this.parameterOverrides = overrides ? { ...overrides } : null;
    this.uploadParameters();
    // Counted with the compile-path uploads, because they are the same
    // event seen from two sides: a value reached the GPU without a shader
    // being built. A drag's worth of these is what the e2e test measures.
    this._parameterUploads += 1;
    this.host.sceneChanged();
    return true;
  }

  /**
   * The installed scene's uniform contract, or null in the literal form.
   *
   * Read by the drag path, which has to know whether the parameter a handle
   * names actually has a slot before it promises a live drag.
   */
  get parameterProgram(): ShaderProgramPayload | null {
    return this.program;
  }

  /**
   * Drop a live drag preview that no compile came along to replace.
   *
   * The ordinary end of a drag is a patch, a recompile, and the same numbers
   * arriving through the payload — which clears the overrides on its way in.
   * A patch the server refuses ends without any of that, and the overrides
   * would keep drawing a solid the program does not describe. Called once
   * the commit has settled, so it is a no-op in every case but that one.
   */
  dropStaleParameterOverrides(): void {
    if (this.parameterOverrides) this.setParameterOverrides(null);
  }

  /** Whether the installed scene reads its parameters from a buffer. */
  get hasParameterBuffer(): boolean {
    return this.program !== null && this.parameterBuffer !== null;
  }
}
