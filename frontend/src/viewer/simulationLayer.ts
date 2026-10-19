/**
 * The FEM simulation surface the viewer draws instead of the raymarched solid.
 *
 * An indexed triangle mesh with a scalar per vertex. Two bind groups share
 * one pipeline: the base pass and the hover-highlight pass differ only in
 * their uniform. Every method that takes a device treats null as "no GPU yet"
 * and keeps only the CPU state; the owner schedules the redraw.
 */

import {
  DEFAULT_SLICE,
  groupOutline,
  meshBounds,
  type SliceState,
} from "../simulation";
import type { SurfacePayload } from "../types";
import type { Vec3 } from "./math";
import type { SimulationPipelines } from "./pipelines";
import { packSimulationUniforms } from "./uniforms";

/** A run of the surface's triangle list: one face group. */
export interface IndexRange {
  start: number;
  count: number;
}

export class SimulationLayer {
  /** Whether the surface replaces the raymarched solid. */
  active = false;
  /** Whether the element-edge hairlines are drawn over the surface. */
  edgesVisible = false;
  /** Which ramp `fs_sim` applies: solved fields vs mesh quality. */
  ramp: "field" | "quality" = "field";

  private pipelines: SimulationPipelines | null = null;
  private vertexBuffer: GPUBuffer | null = null;
  private indexBuffer: GPUBuffer | null = null;
  private indexCount = 0;
  private range: [number, number] = [0, 0];
  private bounds = { min: [0, 0, 0], max: [0, 0, 0] };
  private highlight: IndexRange | null = null;
  /** The hovered face group's outline: index pairs over the surface vertices. */
  private outlineBuffer: GPUBuffer | null = null;
  private outlineCount = 0;
  private clip: SliceState = { ...DEFAULT_SLICE };
  // The surface payload plus the display overrides layered over it: an
  // alternative nodal field, warped (deformed) positions, and a per-vertex
  // highlight tint for BC previews. Any change re-interleaves the vertex
  // buffer — tens of thousands of vertices, cheap enough per edit.
  private payload: SurfacePayload | null = null;
  private scalarOverride: readonly number[] | null = null;
  private positionOverride: readonly number[] | null = null;
  private overlay: Float32Array | null = null;
  /** Element-edge hairlines over the surface (`payload.edges` index pairs). */
  private edgeIndexBuffer: GPUBuffer | null = null;
  private edgeIndexCount = 0;

  /** Install the pipelines, uniforms and bind groups built on a fresh device. */
  attach(pipelines: SimulationPipelines): void {
    this.pipelines = pipelines;
  }

  /** Replace the surface mesh (null clears it); resets the view overrides. */
  setMesh(device: GPUDevice | null, payload: SurfacePayload | null): void {
    this.indexCount = 0;
    this.highlight = null;
    this.outlineBuffer?.destroy();
    this.outlineBuffer = null;
    this.outlineCount = 0;
    this.payload = payload;
    this.scalarOverride = null;
    this.positionOverride = null;
    this.overlay = null;
    if (!payload || !device || payload.indices.length === 0) return;
    const indices = new Uint32Array(payload.indices);
    this.indexBuffer?.destroy();
    this.indexBuffer = device.createBuffer({
      label: "simulation indices",
      size: indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.indexBuffer, 0, indices);
    this.indexCount = indices.length;
    this.edgeIndexBuffer?.destroy();
    this.edgeIndexBuffer = null;
    this.edgeIndexCount = 0;
    if (payload.edges && payload.edges.length >= 2) {
      const edges = new Uint32Array(payload.edges);
      this.edgeIndexBuffer = device.createBuffer({
        label: "simulation element edges",
        size: edges.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(this.edgeIndexBuffer, 0, edges);
      this.edgeIndexCount = edges.length;
    }
    this.range = payload.range;
    this.uploadVertices(device);
  }

  /** Swap the displayed nodal field without re-solving or re-meshing. */
  setScalars(
    device: GPUDevice | null,
    scalars: readonly number[] | null,
    range?: [number, number],
  ): void {
    this.scalarOverride = scalars;
    if (range) this.range = range;
    else if (!scalars && this.payload) this.range = this.payload.range;
    this.uploadVertices(device);
  }

  /** Deformed view: draw the surface at offset positions (null = undeformed). */
  setPositions(device: GPUDevice | null, positions: readonly number[] | null): void {
    this.positionOverride = positions;
    this.uploadVertices(device);
  }

  /** Per-vertex RGBA highlight tint (BC previews); null clears it. */
  setOverlay(device: GPUDevice | null, colors: Float32Array | null): void {
    this.overlay = colors;
    this.uploadVertices(device);
  }

  /**
   * Outline one face group's triangle range (null clears it). Hover over a
   * solved field is drawn, not tinted: the boundary edges go to an index
   * buffer the element-edge pipeline draws three pixels wide.
   */
  setHighlight(device: GPUDevice | null, range: IndexRange | null): void {
    this.highlight = range;
    this.outlineBuffer?.destroy();
    this.outlineBuffer = null;
    this.outlineCount = 0;
    const indices = this.payload?.indices;
    if (range && indices && device) {
      const outline = groupOutline(indices, range.start, range.count);
      if (outline.length >= 2) {
        this.outlineBuffer = device.createBuffer({
          label: "simulation hover outline",
          size: outline.byteLength,
          usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.outlineBuffer, 0, outline);
        this.outlineCount = outline.length;
      }
    }
  }

  /** Move the ParaView-style clip plane. */
  setClip(clip: SliceState): void {
    this.clip = { ...clip };
  }

  /** Re-interleave and upload the surface vertices from the current view. */
  private uploadVertices(device: GPUDevice | null): void {
    const payload = this.payload;
    if (!payload || !device || this.indexCount === 0) return;
    const positions = this.positionOverride ?? payload.positions;
    const scalars = this.scalarOverride ?? payload.scalars;
    const overlay = this.overlay;
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
    this.vertexBuffer?.destroy();
    this.vertexBuffer = device.createBuffer({
      label: "simulation vertices",
      size: interleaved.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.vertexBuffer, 0, interleaved);
    this.bounds = meshBounds(positions);
  }

  /** Write the surface and hover-outline uniforms; nothing while inactive. */
  writeUniforms(device: GPUDevice, matrix: Float32Array, position: Vec3, width: number): void {
    if (!this.active || !this.pipelines) return;
    const frame = {
      matrix,
      position,
      clip: this.clip,
      bounds: this.bounds,
      range: this.range,
      ramp: this.ramp,
    };
    device.queue.writeBuffer(
      this.pipelines.simUniformBuffer,
      0,
      packSimulationUniforms({ ...frame, pixel: 0 }),
    );
    device.queue.writeBuffer(
      this.pipelines.simHighlightUniformBuffer,
      0,
      packSimulationUniforms({ ...frame, pixel: 2 / Math.max(width, 1) }),
    );
  }

  /**
   * Draw the surface, its element edges, and the hovered face group's
   * outline.
   *
   * @param showOverlays The display's overlay switch: the hover outline is
   *   construction overlay, not field.
   */
  draw(pass: GPURenderPassEncoder, showOverlays: boolean): void {
    const pipelines = this.pipelines;
    if (!this.active || !pipelines || !this.vertexBuffer || !this.indexBuffer || this.indexCount === 0) {
      return;
    }
    pass.setPipeline(pipelines.simPipeline);
    pass.setBindGroup(0, pipelines.simBindGroup);
    pass.setVertexBuffer(0, this.vertexBuffer);
    pass.setIndexBuffer(this.indexBuffer, "uint32");
    pass.drawIndexed(this.indexCount);
    if (this.edgesVisible && this.edgeIndexBuffer && this.edgeIndexCount > 0) {
      pass.setPipeline(pipelines.simEdgePipeline);
      pass.setBindGroup(0, pipelines.simBindGroup);
      pass.setVertexBuffer(0, this.vertexBuffer);
      pass.setIndexBuffer(this.edgeIndexBuffer, "uint32");
      pass.drawIndexed(this.edgeIndexCount);
    }
    // Drawn after the element edges so it lands on top.
    if (showOverlays && this.highlight && this.outlineBuffer && this.outlineCount > 0) {
      pass.setPipeline(pipelines.simEdgePipeline);
      pass.setBindGroup(0, pipelines.simHighlightBindGroup);
      pass.setVertexBuffer(0, this.vertexBuffer);
      pass.setIndexBuffer(this.outlineBuffer, "uint32");
      pass.drawIndexed(this.outlineCount, 3);
    }
  }

  destroy(): void {
    this.vertexBuffer?.destroy();
    this.outlineBuffer?.destroy();
    this.indexBuffer?.destroy();
    this.edgeIndexBuffer?.destroy();
  }
}
