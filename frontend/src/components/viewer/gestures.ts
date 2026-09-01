/**
 * The vocabulary of an in-flight pointer gesture.
 *
 * A press decides once what the drag means — orbit, pan, a pending tap on the
 * FEM surface, a BC rubber band, a sketch-handle drag, or a gizmo drag — and
 * the move and release handlers switch on that decision instead of re-picking.
 * The variants carry everything the release needs to write the edit back to
 * the source, which is why the gizmo case snapshots the transform it started
 * from.
 *
 * Two-click constraint flows are the other piece of half-finished pointer
 * state, so their pending-first-pick type lives here too.
 */

import type { GizmoMode } from "../../types";
import type { AxisIndex } from "../../viewer/gizmo";
import type { EdgeConstraintTool, VertexConstraintTool } from "../../constraints";

export type Gesture =
  | { kind: "none" }
  | { kind: "orbit"; x: number; y: number }
  | { kind: "pan"; x: number; y: number }
  /** Pending click on the FEM surface: becomes an orbit once it moves. */
  | { kind: "simtap"; x: number; y: number; clientX: number; clientY: number }
  /** Shift-drag rectangle proposing a Nodes.box BC selection. */
  | { kind: "bcrect"; x0: number; y0: number; x1: number; y1: number }
  | { kind: "drag"; nodeId: string; vertexIndex: number; moved: boolean }
  | {
      kind: "gizmo";
      nodeId: string;
      axis: AxisIndex;
      mode: GizmoMode;
      /** Drag origin: axis parameter for translate, angle for rotate. */
      start: number;
      position: [number, number, number];
      rotation: [number, number, number];
      dimensions: Record<string, number | number[]>;
      /** World length represented by the on-screen scale handle. */
      scaleLength: number;
      /** World position where the visible gizmo was placed. */
      gizmoOrigin: [number, number, number];
      moved: boolean;
    };

/** First pick of a two-click constraint flow, keyed by the active tool. */
export type PendingConstraint =
  | { kind: VertexConstraintTool; first: { nodeId: string; vertexIndex: number } }
  | {
      kind: EdgeConstraintTool;
      first: { nodeId: string; start: number; end: number };
    };
