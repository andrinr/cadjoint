/**
 * The one-line status bar under the viewport.
 *
 * It answers "what will a click do right now", so it is a direct readout of
 * the same state the pointer handlers branch on — in the same order they
 * branch. When a new mode or tool changes what a click means, its sentence
 * belongs in this chain; nothing else here decides anything.
 */

import {
  bcPickArmed,
  editingMode,
  faceHover,
  nodeById,
  pendingLoft,
  selection,
  simView,
  sketchPlane,
  tool,
} from "../../state";
import { faceLabel } from "../../faces";
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

/**
 * What clicking right now would plant, and where.
 *
 * Three states, and they are the three answers the face picker can give: a
 * face it can write, a face it can see but not name, and no face at all —
 * which is a curved surface, and becomes a tangent plane rather than a
 * refusal. The sentence also says *which* sketch, because that is the
 * question a CAD user asks before clicking.
 */
function faceSentence(): string {
  const active = selection();
  const node = active ? nodeById(active.nodeId) : null;
  const onto =
    editingMode() === "sketch" && node?.kind === "profile" && node.line !== null
      ? `re-plants ${node.name ?? "the selected sketch"}`
      : "starts a new sketch";
  const pick = faceHover();
  if (!pick) {
    return `Sketch on face: hover a flat face · a curved surface ${onto} on a tangent plane · Esc to cancel`;
  }
  if (!pick.face.usable) {
    return `Sketch on face: ${faceLabel(pick.face)} has no name in the source — assign its feature to a variable first`;
  }
  return `Sketch on face: click ${faceLabel(pick.face)} — ${onto} there · Esc to cancel`;
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
        : tool() === "face"
        ? faceSentence()
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
