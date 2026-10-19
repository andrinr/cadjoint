/**
 * The B-rep inspector's geometry: the derived B-rep's own faces, drawn
 * instead of the raymarched solid while it is shown, with its edges and
 * vertices on top. `src/brep.ts` packs the arrays.
 */

import { GrowableVertexBuffer } from "./gpuBuffers";
import { EDGE_STRIDE, FACE_STRIDE, HANDLE_STRIDE } from "./overlayGeometry";
import type { OverlayPipelines } from "./pipelines";

/** The packed arrays the inspector hands the renderer. */
export interface BrepGeometry {
  faces: Float32Array;
  edges: Float32Array;
  vertices: Float32Array;
}

export class BrepLayer {
  private pipelines: OverlayPipelines | null = null;
  private faces = new GrowableVertexBuffer("B-rep faces", FACE_STRIDE);
  private edges = new GrowableVertexBuffer("B-rep edges", EDGE_STRIDE);
  private vertices = new GrowableVertexBuffer("B-rep vertices", HANDLE_STRIDE);

  /** Install the overlay pipelines built on a fresh device (the face pipeline is among them). */
  attach(pipelines: OverlayPipelines): void {
    this.pipelines = pipelines;
  }

  /** Whether there are faces to stand in for the solid. */
  get hasFaces(): boolean {
    return this.faces.count > 0;
  }

  /** Replace the geometry; null clears it. */
  set(device: GPUDevice, geometry: BrepGeometry | null): void {
    const empty = new Float32Array(0);
    this.faces.write(device, (geometry?.faces ?? empty) as Float32Array<ArrayBuffer>);
    this.edges.write(device, (geometry?.edges ?? empty) as Float32Array<ArrayBuffer>);
    this.vertices.write(device, (geometry?.vertices ?? empty) as Float32Array<ArrayBuffer>);
  }

  /**
   * Faces first (they write the depth the solid would have), then edges and
   * vertices depth-tested against them. Always the tested variants — x-ray
   * translucency belongs to the raymarched solid, which is not on screen.
   */
  draw(pass: GPURenderPassEncoder): void {
    const pipelines = this.pipelines;
    if (!pipelines || !this.faces.buffer) return;
    pass.setPipeline(pipelines.brepFacePipeline);
    pass.setBindGroup(0, pipelines.overlayBindGroup);
    pass.setVertexBuffer(0, this.faces.buffer);
    pass.draw(this.faces.count);
    if (this.edges.count && this.edges.buffer) {
      // The mesh overlay's uniform: no casing, and the smaller nudge.
      pass.setPipeline(pipelines.edgePipeline.tested);
      pass.setBindGroup(0, pipelines.meshOverlayBindGroup);
      pass.setVertexBuffer(0, this.edges.buffer);
      pass.draw(6, this.edges.count);
    }
    if (this.vertices.count && this.vertices.buffer) {
      pass.setPipeline(pipelines.handlePipeline.tested);
      pass.setBindGroup(0, pipelines.overlayBindGroup);
      pass.setVertexBuffer(0, this.vertices.buffer);
      pass.draw(6, this.vertices.count);
    }
  }

  destroy(): void {
    this.faces.destroy();
    this.edges.destroy();
    this.vertices.destroy();
  }
}
