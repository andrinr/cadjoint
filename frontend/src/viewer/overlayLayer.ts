/**
 * What the viewer draws *about* the model: the construction skeleton and its
 * handles, the sketch planes, the picked-face highlight, the mesher's edges
 * and the transform gizmo.
 *
 * Holds the state the panes hand over, packs it into instance buffers
 * (`overlayGeometry.ts` has the layouts) and draws them into the frame's
 * pass. The setters report whether anything changed; the owner re-uploads
 * and schedules the redraw.
 */

import type {
  ConstructionFace,
  ConstructionNode,
  GizmoMode,
  MeshEdgePayload,
  Selection,
} from "../types";
import type { PlaneFrame } from "../planes";
import type { DisplaySettings } from "./display";
import { gizmoScale, gizmoTarget, type AxisIndex } from "./gizmo";
import { GrowableVertexBuffer } from "./gpuBuffers";
import type { Vec3, View } from "./math";
import {
  EDGE_STRIDE,
  FACE_STRIDE,
  GIZMO_STRIDE,
  HANDLE_STRIDE,
  packConstructionOverlay,
  packFaceHighlight,
  packGizmoInstances,
  packMeshEdgeInstances,
  packPlaneOverlay,
} from "./overlayGeometry";
import type { DepthPair, OverlayPipelines } from "./pipelines";

export class OverlayLayer {
  /** The transform the gizmo offers, before a node's own limits. */
  gizmoMode: GizmoMode = "translate";
  /** The gizmo axis under the pointer or in a drag. */
  gizmoAxis: AxisIndex | null = null;

  private pipelines: OverlayPipelines | null = null;
  private profiles: readonly ConstructionNode[] = [];
  private planes: readonly PlaneFrame[] = [];
  private selection: Selection | null = null;
  private hover: Selection | null = null;
  private focusGroup: ReadonlySet<string> = new Set();
  private meshEdges: MeshEdgePayload | null = null;
  /** The plane the sketch tool would create under the pointer. */
  private planePreview: PlaneFrame | null = null;
  // The face under the pointer while face picking is armed: a filled wash
  // (its own triangle pipeline, since nothing else in the overlay is a
  // surface) plus a hairline outline drawn through the edge pipeline.
  private faceHighlight: ConstructionFace | null = null;
  /** `gizmoMode` reduced to what the gizmo's node supports, as last packed. */
  private visibleGizmoMode: GizmoMode = "translate";

  private edges = new GrowableVertexBuffer("overlay edges", EDGE_STRIDE);
  private handles = new GrowableVertexBuffer("overlay handles", HANDLE_STRIDE);
  private gizmo = new GrowableVertexBuffer("transform gizmo", GIZMO_STRIDE);
  private faceFill = new GrowableVertexBuffer("face highlight fill", FACE_STRIDE);
  private faceOutline = new GrowableVertexBuffer("face highlight outline", EDGE_STRIDE);
  private meshEdgeSegments = new GrowableVertexBuffer("mesh edges", EDGE_STRIDE);
  private meshWireCount = 0;
  private meshSharpCount = 0;

  /** Install the pipelines built on a fresh device. */
  attach(pipelines: OverlayPipelines): void {
    this.pipelines = pipelines;
  }

  /** Whether any construction edges are packed. */
  get hasSkeletonEdges(): boolean {
    return this.edges.count > 0;
  }

  setConstruction(
    profiles: readonly ConstructionNode[],
    selection: Selection | null,
    hover: Selection | null,
    planes: readonly PlaneFrame[],
  ): void {
    this.profiles = profiles;
    this.selection = selection;
    this.hover = hover;
    this.planes = planes;
  }

  setFocusGroup(nodes: ReadonlySet<string>): boolean {
    if (nodes.size === 0 && this.focusGroup.size === 0) return false;
    this.focusGroup = nodes;
    return true;
  }

  setMeshEdges(payload: MeshEdgePayload | null): void {
    this.meshEdges = payload;
  }

  setPlanePreview(frame: PlaneFrame | null): boolean {
    if (this.planePreview === frame) return false;
    this.planePreview = frame;
    return true;
  }

  setFaceHighlight(face: ConstructionFace | null): boolean {
    if (this.faceHighlight === face) return false;
    this.faceHighlight = face;
    return true;
  }

  /** `gizmoMode`, reduced to the transforms a selected node supports. */
  gizmoModeFor(node: ConstructionNode): GizmoMode {
    if (this.gizmoMode === "rotate" && !node.transform?.canRotate) return "translate";
    if (this.gizmoMode === "scale" && node.kind === "profile") return "translate";
    return this.gizmoMode;
  }

  /** The selected primitive the gizmo attaches to, if any. */
  gizmoTarget(): { node: ConstructionNode; origin: Vec3 } | null {
    return gizmoTarget(this.selection, this.profiles);
  }

  /**
   * Pack and upload every overlay buffer.
   *
   * @param overField Whether the FEM surface is shown: the skeleton changes
   *   tone over a field.
   * @param view The view the gizmo is sized for.
   */
  upload(device: GPUDevice, overField: boolean, view: View): void {
    const { edges, handles } = packConstructionOverlay(
      this.profiles,
      this.selection,
      this.hover,
      overField,
      this.focusGroup,
    );

    const target = this.gizmoTarget();
    let gizmo: number[] = [];
    if (target) {
      this.visibleGizmoMode = this.gizmoModeFor(target.node);
      gizmo = packGizmoInstances(
        target.origin,
        gizmoScale(view, target.origin),
        this.visibleGizmoMode,
        this.gizmoAxis,
      );
    }

    const meshSegments = packMeshEdgeInstances(this.meshEdges);
    const highlight = packFaceHighlight(this.faceHighlight);
    // The sketch planes draw through the face-highlight buffers: the same
    // wash and the same hairline, in the same pass, so a plane costs nothing
    // the overlay did not already pay for.
    const planes = packPlaneOverlay(
      this.planes,
      this.selection,
      this.hover,
      this.planePreview,
    );
    const fills = planes.fill.concat(highlight.fill);
    const outlines = planes.outline.concat(highlight.outline);

    this.faceFill.write(device, new Float32Array(fills));
    this.faceOutline.write(device, new Float32Array(outlines));
    this.meshWireCount = this.meshEdges?.wire.length ?? 0;
    this.meshSharpCount = this.meshEdges?.sharp.length ?? 0;
    this.meshEdgeSegments.write(device, new Float32Array(meshSegments));
    this.edges.write(device, new Float32Array(edges));
    this.handles.write(device, new Float32Array(handles));
    this.gizmo.write(device, new Float32Array(gizmo));
  }

  /**
   * Draw the overlay into the frame's pass.
   *
   * @param seeThrough Whether construction geometry reads through the solid
   *   (an x-rayed raymarched solid).
   */
  draw(pass: GPURenderPassEncoder, display: DisplaySettings, seeThrough: boolean): void {
    const pipelines = this.pipelines;
    if (!pipelines) return;
    const { edgePipeline, handlePipeline, facePipeline, overlayBindGroup } = pipelines;
    const depth = seeThrough
      ? (pair: DepthPair) => pair.seen
      : (pair: DepthPair) => pair.tested;
    // The skeleton is the exception to `seen`: through an x-rayed solid it is
    // drawn twice, behind the part as a dashed, dimmed hidden line (`hidden`,
    // below) and in front of it at full weight (`tested`), so the two never
    // read as one weight.
    const wantSkeletonEdges = Boolean(
      display.showOverlays && display.showSketches && this.edges.count && this.edges.buffer,
    );
    const wantHandles = Boolean(
      display.showOverlays && display.showSketches && this.handles.count && this.handles.buffer,
    );
    const wantMeshWire = display.showMeshWireframe && this.meshWireCount > 0;
    const wantMeshSharp = display.showMeshEdges && this.meshSharpCount > 0;
    if ((wantMeshWire || wantMeshSharp) && this.meshEdgeSegments.buffer) {
      pass.setPipeline(depth(edgePipeline));
      pass.setBindGroup(0, pipelines.meshOverlayBindGroup);
      pass.setVertexBuffer(0, this.meshEdgeSegments.buffer);
      if (wantMeshWire) pass.draw(6, this.meshWireCount, 0, 0);
      if (wantMeshSharp) pass.draw(6, this.meshSharpCount, 0, this.meshWireCount);
      pass.setBindGroup(0, overlayBindGroup);
    }
    // Everything below is the construction overlay: what the app draws *about*
    // the model. One switch turns the lot off for a presentation frame; the
    // mesh edges above belong to the mesh and keep their own.
    if (!display.showOverlays) return;
    // Hidden lines first: they test against the scene's depth alone, before
    // any overlay has written its own.
    if (seeThrough) {
      if (wantSkeletonEdges && edgePipeline.hidden) {
        pass.setPipeline(edgePipeline.hidden);
        pass.setBindGroup(0, overlayBindGroup);
        pass.setVertexBuffer(0, this.edges.buffer);
        pass.draw(6, this.edges.count);
      }
      if (wantHandles && handlePipeline.hidden) {
        pass.setPipeline(handlePipeline.hidden);
        pass.setBindGroup(0, overlayBindGroup);
        pass.setVertexBuffer(0, this.handles.buffer);
        pass.draw(6, this.handles.count);
      }
    }
    // Fill first, then its own outline: the wash does not write depth, so the
    // hairline lands on top of it rather than fighting it.
    if (this.faceFill.count && this.faceFill.buffer) {
      pass.setPipeline(depth(facePipeline));
      pass.setBindGroup(0, overlayBindGroup);
      pass.setVertexBuffer(0, this.faceFill.buffer);
      pass.draw(this.faceFill.count);
    }
    if (this.faceOutline.count && this.faceOutline.buffer) {
      pass.setPipeline(depth(edgePipeline));
      pass.setBindGroup(0, overlayBindGroup);
      pass.setVertexBuffer(0, this.faceOutline.buffer);
      pass.draw(6, this.faceOutline.count);
    }
    if (wantSkeletonEdges) {
      pass.setPipeline(edgePipeline.tested);
      pass.setBindGroup(0, overlayBindGroup);
      pass.setVertexBuffer(0, this.edges.buffer);
      pass.draw(6, this.edges.count);
    }
    if (wantHandles) {
      pass.setPipeline(handlePipeline.tested);
      pass.setBindGroup(0, overlayBindGroup);
      pass.setVertexBuffer(0, this.handles.buffer);
      pass.draw(6, this.handles.count);
    }
    if (this.gizmo.count && this.gizmo.buffer) {
      const [pipeline, vertices] =
        this.visibleGizmoMode === "translate"
          ? [pipelines.gizmoArrowPipeline, 9]
          : this.visibleGizmoMode === "scale"
            ? [pipelines.gizmoScalePipeline, 12]
            : [pipelines.gizmoEdgePipeline, 6];
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, overlayBindGroup);
      pass.setVertexBuffer(0, this.gizmo.buffer);
      pass.draw(vertices, this.gizmo.count);
    }
  }

  destroy(): void {
    for (const instances of [
      this.edges,
      this.handles,
      this.gizmo,
      this.faceFill,
      this.faceOutline,
      this.meshEdgeSegments,
    ]) {
      instances.destroy();
    }
  }
}
