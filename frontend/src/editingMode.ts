/**
 * Top-level editing modes, in the way Blender scopes its tool set to a mode.
 *
 * Pure logic only — which modes exist, how the keyboard cycles them, which
 * tool clusters each mode shows, and when a selection auto-enters sketch
 * mode. The signals live in `state.ts`; the DOM lives in `ToolRail.tsx`.
 */

import { browserStorage, type KeyValueStorage } from "./storage";
import type { ConstructionNode } from "./types";

export type EditingMode = "model" | "sketch" | "simulate";

/** Switcher order, also the keyboard cycling order. */
export const EDITING_MODES: EditingMode[] = ["model", "sketch", "simulate"];

/** The next mode when cycling with the keyboard (wraps around). */
export function cycleEditingMode(mode: EditingMode, step: 1 | -1 = 1): EditingMode {
  const index = EDITING_MODES.indexOf(mode);
  const count = EDITING_MODES.length;
  return EDITING_MODES[(index + step + count) % count];
}

/** Tool clusters the rail can show; each mode picks a subset. */
export type RailGroupId =
  | "select"
  | "create"
  | "modify"
  | "transform"
  | "annotate"
  | "delete";

const GROUPS_BY_MODE: Record<EditingMode, RailGroupId[]> = {
  // Shape is in Model as well: a sketch is extruded from Model as often as from
  // Sketch, and its tools disable with the reason until a sketch is selected.
  model: ["select", "create", "modify", "transform", "delete"],
  sketch: ["select", "create", "modify", "transform", "annotate", "delete"],
  simulate: [],
};

/** The clusters the rail shows in `mode`, in display order. */
export function railGroupsForMode(mode: EditingMode): RailGroupId[] {
  return GROUPS_BY_MODE[mode];
}

/** Whether one cluster is available in `mode`. */
export function groupVisibleInMode(group: RailGroupId, mode: EditingMode): boolean {
  return GROUPS_BY_MODE[mode].includes(group);
}

/** Create-cluster children are themselves mode-filtered. */
export const CREATE_TOOL_MODES: Record<string, EditingMode[]> = {
  sketch: ["model", "sketch"],
  // Available in both, and it means something different in each: in Model it
  // starts a new sketch on the face, in Sketch it re-plants the open one.
  face: ["model", "sketch"],
  polygon: ["model", "sketch"],
  box: ["model"],
  sphere: ["model"],
  cylinder: ["model"],
};

/** Whether a create tool is offered in `mode`. */
export function createToolVisibleInMode(tool: string, mode: EditingMode): boolean {
  return (CREATE_TOOL_MODES[tool] ?? ["model"]).includes(mode);
}

/**
 * Decide whether a selection change should auto-enter sketch mode.
 *
 * Predictable and cancelable: only a *newly* selected sketch profile enters
 * sketch mode (`lastAutoNodeId` remembers the previous auto-entry, so leaving
 * sketch mode manually while the same sketch stays selected is respected),
 * and simulate mode is never hijacked by a selection.
 */
export function autoEnterSketchMode(
  mode: EditingMode,
  node: ConstructionNode | null | undefined,
  lastAutoNodeId: string | null,
): { mode: EditingMode; lastAutoNodeId: string | null } {
  if (mode === "simulate") return { mode, lastAutoNodeId };
  if (!node || node.kind !== "profile") {
    // Leaving the sketch (or selecting a solid) re-arms auto-entry.
    return { mode, lastAutoNodeId: null };
  }
  if (node.id === lastAutoNodeId) return { mode, lastAutoNodeId };
  return { mode: "sketch", lastAutoNodeId: node.id };
}

export const EDITING_MODE_STORAGE_KEY = "cadjoint.editingMode.v1";

/** The persisted editing mode, defaulting to model on anything unexpected. */
export function loadEditingMode(
  storage: KeyValueStorage | null = browserStorage(),
): EditingMode {
  try {
    const raw = storage?.getItem(EDITING_MODE_STORAGE_KEY);
    // A persisted "render" from before the mode was retired falls back to
    // model — render settings live in the eye-icon popover now.
    return raw === "sketch" || raw === "simulate" ? raw : "model";
  } catch {
    return "model";
  }
}

export function persistEditingMode(
  mode: EditingMode,
  storage: KeyValueStorage | null = browserStorage(),
): void {
  try {
    storage?.setItem(EDITING_MODE_STORAGE_KEY, mode);
  } catch {
    // Persistence is a convenience; private-mode storage failures are fine.
  }
}
