/**
 * The one-line status bar under the viewport.
 *
 * It answers "what will a click do right now", so it is a direct readout of
 * the same state the pointer handlers branch on — in the same order they
 * branch. When a new mode or tool changes what a click means, its sentence
 * belongs in this chain; nothing else here decides anything.
 */

import { bcPickArmed, editingMode, pendingLoft, simView, sketchPlane, tool } from "../../state";
import {
  CONSTRAINT_TOOL_NAMES,
  isEdgeConstraintTool,
  isVertexConstraintTool,
  type EdgeConstraintTool,
  type VertexConstraintTool,
} from "../../constraints";
import type { PendingConstraint } from "./gestures";

export interface ViewerHintProps {
  /** The half-finished two-click constraint pick, if there is one. */
  pendingConstraint: PendingConstraint | null;
}

export function ViewerHint(props: ViewerHintProps) {
  return (
    <p class="viewer-hint" data-testid="viewer-hint">
      <b class="hint-mode" data-testid="hint-mode">
        {editingMode()}
      </b>
      {" · "}
      {editingMode() === "simulate"
        ? bcPickArmed()
          ? "Pick BC: click the mesh → sphere · Shift-drag → box · confirm in the builder"
          : simView()
            ? "Click the mesh to probe values · pick BC regions from the study builder · Esc returns to model"
            : "Simulation setup · M cycles modes · Esc returns to model"
        : pendingLoft()
        ? "Loft: click the second sketch in the viewport · Esc to cancel"
        : tool() === "sketch"
        ? sketchPlane() === "face"
          ? "Sketch: click a solid's face to place it there · Esc to cancel"
          : `Sketch: click to place on the ${sketchPlane().toUpperCase()} plane · Esc to cancel`
        : tool() === "polygon"
        ? "Point: click sketch edges to add vertices · Esc to finish"
        : isVertexConstraintTool(tool())
          ? props.pendingConstraint
            ? `${CONSTRAINT_TOOL_NAMES[tool() as VertexConstraintTool]}: choose a second point in the same sketch`
            : `${CONSTRAINT_TOOL_NAMES[tool() as VertexConstraintTool]}: choose the first sketch point`
        : isEdgeConstraintTool(tool())
          ? props.pendingConstraint
            ? `${CONSTRAINT_TOOL_NAMES[tool() as EdgeConstraintTool]}: choose a second edge in the same sketch`
            : `${CONSTRAINT_TOOL_NAMES[tool() as EdgeConstraintTool]}: choose the first sketch edge`
          : tool() !== "select"
            ? `Click to place a ${tool()} · Esc to cancel`
            : "Drag handles or the gizmo · Drag to orbit · Space, Shift or right-drag to pan · Del removes"}
    </p>
  );
}
