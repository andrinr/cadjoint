/**
 * What the rail's tools do, shared by the rail, the menu and the keys. Each
 * comes as a pair: the act, and why it cannot act now (null when it can), the
 * sentence a disabled item puts in its title.
 */

import {
  busy,
  editingMode,
  nodeById,
  pendingLoft,
  selection,
  setEditingMode,
  setGizmoMode,
  setPendingLoft,
  setSelection,
  setSelectionMode,
  setTool,
  tool,
} from "../state";
import { isEdgeConstraintTool } from "../constraints";
import type { ConstraintKind, ConstructionNode, GizmoMode, SelectionMode, ToolMode } from "../types";

/** The selected sketch profile, when it can carry an operator. */
export function modifyProfile(): ConstructionNode | null {
  const active = selection();
  const node = active && nodeById(active.nodeId);
  return node?.kind === "profile" && node.line !== null ? node : null;
}

export type ShapeKey = "extrude" | "revolve" | "loft";

/** Why Extrude / Revolve / Loft cannot run on the selection, or null. */
export function shapeBlocked(key: ShapeKey): string | null {
  const profile = modifyProfile();
  if (!profile) return "Select a sketch first";
  if (busy()) return "Wait for the compile to finish";
  if (key === "extrude") {
    return profile.operators.some((item) => item.kind === "extrude")
      ? "This sketch is already extruded"
      : null;
  }
  return profile.operators.length > 0 ? "This sketch already drives an operation" : null;
}

/** Start (or, a second time, abandon) a loft from the selected sketch. */
export function toggleLoft(): void {
  const profile = modifyProfile();
  if (!profile) return;
  if (pendingLoft()?.nodeId === profile.id) {
    setPendingLoft(null);
    return;
  }
  setSelectionMode("object");
  setPendingLoft({ nodeId: profile.id, line: profile.line! });
}

/** A whole object is selected, so a transform gizmo can apply. */
export function transformable(): boolean {
  const active = selection();
  return active !== null && active.vertexIndex === null;
}

/** Whether the selection supports the gizmo `mode`. */
export function supportsGizmo(mode: GizmoMode): boolean {
  if (!transformable()) return false;
  const node = nodeById(selection()!.nodeId);
  if (!node?.transform) return false;
  if (mode === "rotate") return node.transform.canRotate;
  if (mode === "scale") return node.kind !== "profile";
  return true;
}

export function gizmoBlocked(mode: GizmoMode): string | null {
  if (supportsGizmo(mode)) return null;
  if (mode === "scale" && transformable()) return "Scale needs a solid primitive";
  return "Select a compatible object first";
}

/** Arm a create tool; the second press disarms it. */
export function toggleCreateTool(key: ToolMode): void {
  setTool(tool() === key ? "select" : key);
}

/**
 * Arm a create tool from a key or the menu. Solids are placed in Model mode,
 * and a sketch tool leaves Simulate for Model, where the rail shows it.
 */
export function armCreateTool(key: ToolMode): void {
  const solid = key === "box" || key === "sphere" || key === "cylinder";
  if (solid || editingMode() === "simulate") setEditingMode("model");
  setTool(key);
}

/** The constraint tools the rail and the Constrain menu arm. */
export const CONSTRAINT_TOOLS: ToolMode[] = [
  "distance",
  "horizontal",
  "vertical",
  "coincident",
  "parallel",
  "perpendicular",
];

/** Arm a two-pick constraint tool (point tools pick vertices). */
export function toggleConstraintTool(key: ToolMode): void {
  if (!isEdgeConstraintTool(key)) setSelectionMode("vertex");
  setTool(tool() === key ? "select" : key);
}

export function constraintBlocked(): string | null {
  return busy() ? "Wait for the compile to finish" : null;
}

/** Why the selected point cannot be fixed, or null. */
export function fixBlocked(): string | null {
  const active = selection();
  const node = active && nodeById(active.nodeId);
  if (!node || node.kind !== "profile" || active!.vertexIndex === null) {
    return "Select a sketch point first";
  }
  if (node.line === null) return "This sketch is defined dynamically in code";
  if (busy()) return "Wait for the compile to finish";
  return null;
}

/** Anchor the selected sketch point where it is. */
export function fixSelectedPoint(
  addConstraint: (
    line: number,
    kind: ConstraintKind,
    indices: number[],
    value?: number | number[],
  ) => Promise<void>,
): void {
  if (fixBlocked() !== null) return;
  const active = selection()!;
  const node = nodeById(active.nodeId)!;
  const vertex = node.vertices[active.vertexIndex!];
  void addConstraint(node.line!, "fixed", [active.vertexIndex!], vertex.uv);
}

/** Switch what a click selects; leaving vertices lifts a vertex pick to its object. */
export function setSelectKind(kind: SelectionMode): void {
  setTool("select");
  setSelectionMode(kind);
  const active = selection();
  if (kind === "object" && active && active.vertexIndex !== null) {
    setSelection({ nodeId: active.nodeId, vertexIndex: null });
    if (nodeById(active.nodeId)?.kind === "profile") setGizmoMode("translate");
  }
}
