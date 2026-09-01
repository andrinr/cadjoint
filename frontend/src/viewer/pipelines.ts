/**
 * Building the viewer's render pipelines.
 *
 * Pipeline creation is long, declarative, and written once at device setup:
 * vertex attribute offsets that must agree with `overlayGeometry.ts`, blend
 * and depth states, and the bind groups the passes swap between. Keeping it
 * here leaves the renderer holding the state and the per-frame passes rather
 * than two hundred lines of descriptors.
 *
 * The one rule worth knowing: never `layout: "auto"` for anything whose bind
 * group is shared. An automatic layout is *exclusive* to the pipeline that
 * derived it, so a bind group built from one pipeline cannot be bound to
 * another even when the bindings are identical — which is exactly what the
 * edge/handle passes and the two preview variants need to do.
 */

import {
  EDGE_STRIDE,
  GIZMO_STRIDE,
  HANDLE_STRIDE,
} from "./overlayGeometry";
// Kept as standalone .wgsl files so the shaders have one source of truth that
// both the bundler and the Python shader-validation test read.
import OVERLAY_WGSL from "./overlay.wgsl?raw";
import SIMULATION_WGSL from "./simulation.wgsl?raw";

export const DEPTH_FORMAT: GPUTextureFormat = "depth32float";

export interface SharedLayout {
  bindGroupLayout: GPUBindGroupLayout;
  pipelineLayout: GPUPipelineLayout;
}

/** A bind group layout shared by several pipelines. */
export function sharedLayout(
  device: GPUDevice,
  label: string,
  bindings: number[],
): SharedLayout {
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

/** Compile a WGSL module and surface its compilation errors as an exception. */
export async function compileModule(
  device: GPUDevice,
  code: string,
  label: string,
): Promise<GPUShaderModule> {
  const module = device.createShaderModule({ code, label });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === "error");
  if (errors.length) {
    throw new Error(
      errors.map((m) => `${label} ${m.lineNum}:${m.linePos} ${m.message}`).join("\n"),
    );
  }
  return module;
}

export interface OverlayPipelines {
  edgePipeline: GPURenderPipeline;
  handlePipeline: GPURenderPipeline;
  gizmoEdgePipeline: GPURenderPipeline;
  gizmoArrowPipeline: GPURenderPipeline;
  gizmoScalePipeline: GPURenderPipeline;
  overlayBindGroup: GPUBindGroup;
  meshOverlayBindGroup: GPUBindGroup;
}

/** The construction-overlay and transform-gizmo pipelines. */
export function createOverlayPipelines(
  device: GPUDevice,
  format: GPUTextureFormat,
  overlayBuffer: GPUBuffer,
  meshOverlayBuffer: GPUBuffer,
): OverlayPipelines {
  const module = device.createShaderModule({ code: OVERLAY_WGSL, label: "Overlay WGSL" });
  const { bindGroupLayout, pipelineLayout } = sharedLayout(device, "Overlay bindings", [0]);
  const blend: GPUBlendState = {
    color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
    alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
  };
  const depthStencil: GPUDepthStencilState = {
    format: DEPTH_FORMAT,
    depthWriteEnabled: true,
    depthCompare: "less-equal",
  };

  const edgePipeline = device.createRenderPipeline({
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
    fragment: { module, entryPoint: "fs_edge", targets: [{ format, blend }] },
    primitive: { topology: "triangle-list" },
    depthStencil,
  });

  const handlePipeline = device.createRenderPipeline({
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
    fragment: { module, entryPoint: "fs_handle", targets: [{ format, blend }] },
    primitive: { topology: "triangle-list" },
    depthStencil,
  });

  const alwaysVisibleDepth: GPUDepthStencilState = {
    format: DEPTH_FORMAT,
    depthWriteEnabled: false,
    depthCompare: "always",
  };

  const gizmoEdgePipeline = device.createRenderPipeline({
    label: "Always-visible rotation gizmo",
    layout: pipelineLayout,
    vertex: {
      module,
      entryPoint: "vs_edge",
      buffers: [
        {
          arrayStride: GIZMO_STRIDE,
          stepMode: "instance",
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x3" },
            { shaderLocation: 1, offset: 12, format: "float32x3" },
            { shaderLocation: 2, offset: 24, format: "float32x4" },
          ],
        },
      ],
    },
    fragment: { module, entryPoint: "fs_edge", targets: [{ format, blend }] },
    primitive: { topology: "triangle-list" },
    depthStencil: alwaysVisibleDepth,
  });

  const gizmoArrowPipeline = device.createRenderPipeline({
    label: "Always-visible translation gizmo",
    layout: pipelineLayout,
    vertex: {
      module,
      entryPoint: "vs_gizmo_arrow",
      buffers: [
        {
          arrayStride: GIZMO_STRIDE,
          stepMode: "instance",
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x3" },
            { shaderLocation: 1, offset: 12, format: "float32x3" },
            { shaderLocation: 2, offset: 24, format: "float32x4" },
            { shaderLocation: 3, offset: 40, format: "float32" },
          ],
        },
      ],
    },
    fragment: { module, entryPoint: "fs_gizmo_arrow", targets: [{ format, blend }] },
    primitive: { topology: "triangle-list" },
    depthStencil: alwaysVisibleDepth,
  });

  const gizmoScalePipeline = device.createRenderPipeline({
    label: "Always-visible scale gizmo",
    layout: pipelineLayout,
    vertex: {
      module,
      entryPoint: "vs_gizmo_scale",
      buffers: [
        {
          arrayStride: GIZMO_STRIDE,
          stepMode: "instance",
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x3" },
            { shaderLocation: 1, offset: 12, format: "float32x3" },
            { shaderLocation: 2, offset: 24, format: "float32x4" },
            { shaderLocation: 3, offset: 40, format: "float32" },
          ],
        },
      ],
    },
    fragment: { module, entryPoint: "fs_gizmo_arrow", targets: [{ format, blend }] },
    primitive: { topology: "triangle-list" },
    depthStencil: alwaysVisibleDepth,
  });

  return {
    edgePipeline,
    handlePipeline,
    gizmoEdgePipeline,
    gizmoArrowPipeline,
    gizmoScalePipeline,
    overlayBindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: overlayBuffer } }],
    }),
    meshOverlayBindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: meshOverlayBuffer } }],
    }),
  };
}

export interface SimulationPipelines {
  simPipeline: GPURenderPipeline;
  simEdgePipeline: GPURenderPipeline;
  simUniformBuffer: GPUBuffer;
  simHighlightUniformBuffer: GPUBuffer;
  simBindGroup: GPUBindGroup;
  simHighlightBindGroup: GPUBindGroup;
}

/**
 * The indexed triangle-mesh pipeline for FEM results.
 *
 * Follows the overlay pattern: an explicit layout (never `layout: "auto"`)
 * so the base and highlight passes can bind different uniform buffers to
 * one pipeline. Vertices interleave a position with the nodal scalar; the
 * fragment stage ramps the scalar and applies the clip plane.
 */
export function createSimulationPipelines(
  device: GPUDevice,
  format: GPUTextureFormat,
  createUniform: (size: number) => GPUBuffer,
): SimulationPipelines {
  const module = device.createShaderModule({
    code: SIMULATION_WGSL,
    label: "Simulation WGSL",
  });
  const { bindGroupLayout, pipelineLayout } = sharedLayout(
    device,
    "Simulation bindings",
    [0],
  );

  const simPipeline = device.createRenderPipeline({
    label: "Simulation surface",
    layout: pipelineLayout,
    vertex: {
      module,
      entryPoint: "vs_sim",
      buffers: [
        {
          arrayStride: 32,
          stepMode: "vertex",
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x3" },
            { shaderLocation: 1, offset: 12, format: "float32" },
            // BC-preview tint: rgb hue + blend strength per vertex.
            { shaderLocation: 2, offset: 16, format: "float32x4" },
          ],
        },
      ],
    },
    fragment: { module, entryPoint: "fs_sim", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
    depthStencil: {
      format: DEPTH_FORMAT,
      depthWriteEnabled: true,
      depthCompare: "less-equal",
    },
  });

  // Element-edge hairlines: same vertex buffer (position only), a line
  // list over the payload's edge index pairs, nudged toward the camera in
  // the vertex stage so they sit on top of their own faces.
  const simEdgePipeline = device.createRenderPipeline({
    label: "Simulation element edges",
    layout: pipelineLayout,
    vertex: {
      module,
      entryPoint: "vs_sim_edge",
      buffers: [
        {
          arrayStride: 32,
          stepMode: "vertex",
          // The scalar rides along so the hairline can pick a colour that
          // contrasts with the ramp value underneath it.
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x3" },
            { shaderLocation: 1, offset: 12, format: "float32" },
          ],
        },
      ],
    },
    fragment: { module, entryPoint: "fs_sim_edge", targets: [{ format }] },
    primitive: { topology: "line-list" },
    depthStencil: {
      format: DEPTH_FORMAT,
      depthWriteEnabled: false,
      depthCompare: "less-equal",
    },
  });

  const simUniformBuffer = createUniform(112);
  const simHighlightUniformBuffer = createUniform(112);
  return {
    simPipeline,
    simEdgePipeline,
    simUniformBuffer,
    simHighlightUniformBuffer,
    simBindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: simUniformBuffer } }],
    }),
    simHighlightBindGroup: device.createBindGroup({
      layout: bindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: simHighlightUniformBuffer } }],
    }),
  };
}
