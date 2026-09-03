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
import type { BindingState } from "../../viewer/dragBinding";
import type { PendingConstraint } from "./gestures";

export interface ViewerHintProps {
  /** The half-finished two-click constraint pick, if there is one. */
  pendingConstraint: PendingConstraint | null;
  /**
   * The sketch handle under the pointer, and whether dragging it is live.
   *
   * Null whenever nothing is hovered or the construction overlay is off —
   * the sentence describes a mark, so it goes away with the mark.
   */
  handle: { name: string | null; state: BindingState } | null;
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

/**
 * What dragging the hovered handle will cost.
 *
 * The one line that says out loud what the filled and hollow handles mean, at
 * the moment it matters — pointing at one — rather than as a standing legend.
 * A free parameter has a slot in the shader's uniform buffer, so the solid
 * follows the pointer; anything else is a literal in the program text, and
 * the solid only catches up when the rewritten source recompiles.
 */
function handleSentence(handle: { name: string | null; state: BindingState }): string {
  const label = handle.name ?? "This point";
  return handle.state === "free"
    ? `${label} · free parameter: the solid follows the drag`
    : `${label} · fixed value: the solid follows on release, after a recompile`;
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
            ? "Click the mesh to probe values · pick BC regions from the study builder · Esc steps back to model"
            : "Simulation setup · M cycles modes · Esc steps back to model"
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
            : props.handle
              ? handleSentence(props.handle)
              : "Drag handles or the gizmo · Drag to orbit · Right-drag to pan"}
    </p>
  );
}
