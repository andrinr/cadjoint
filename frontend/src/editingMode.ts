/**
 * Top-level editing modes, in the way Blender scopes its tool set to a mode.
 *
 * Pure logic only — which modes exist, how the keyboard cycles them, which
 * tool clusters each mode shows, and when a selection auto-enters sketch
 * mode. The signals live in `state.ts`; the DOM lives in `ToolRail.tsx`.
 */

import type { ConstructionNode } from "./types";

export type EditingMode = "model" | "sketch" | "simulate" | "render";

/** Switcher order, also the keyboard cycling order. */
export const EDITING_MODES: EditingMode[] = ["model", "sketch", "simulate", "render"];

/**
 * Per-mode identity: label plus the accent color threaded through the mode
 * switcher, tool rail, dock headers, hint bar, and viewport border. The CSS
 * mirrors these values via `.app[data-mode=…]`; keep the two in sync.
 */
export const MODE_ACCENTS: Record<EditingMode, string> = {
  model: "#d9ff57",
  sketch: "#7fd6f5",
  simulate: "#ffb25c",
  render: "#d8a6ff",
};

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
  model: ["select", "create", "transform", "delete"],
  sketch: ["select", "create", "modify", "transform", "annotate", "delete"],
  simulate: [],
  render: [],
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
 * and simulate/render mode is never hijacked by a selection.
 */
export function autoEnterSketchMode(
  mode: EditingMode,
  node: ConstructionNode | null | undefined,
  lastAutoNodeId: string | null,
): { mode: EditingMode; lastAutoNodeId: string | null } {
  if (mode === "simulate" || mode === "render") return { mode, lastAutoNodeId };
  if (!node || node.kind !== "profile") {
    // Leaving the sketch (or selecting a solid) re-arms auto-entry.
    return { mode, lastAutoNodeId: null };
  }
  if (node.id === lastAutoNodeId) return { mode, lastAutoNodeId };
  return { mode: "sketch", lastAutoNodeId: node.id };
}

export const EDITING_MODE_STORAGE_KEY = "cadjoint.editingMode.v1";

/** Minimal storage surface so tests can inject a memory store. */
export interface StringStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

function defaultStorage(): StringStorage | undefined {
  try {
    return typeof localStorage === "undefined" ? undefined : localStorage;
  } catch {
    return undefined;
  }
}

/** The persisted editing mode, defaulting to model on anything unexpected. */
export function loadEditingMode(storage: StringStorage | undefined = defaultStorage()): EditingMode {
  try {
    const raw = storage?.getItem(EDITING_MODE_STORAGE_KEY);
    return raw === "sketch" || raw === "simulate" || raw === "render" ? raw : "model";
  } catch {
    return "model";
  }
}

export function persistEditingMode(
  mode: EditingMode,
  storage: StringStorage | undefined = defaultStorage(),
): void {
  try {
    storage?.setItem(EDITING_MODE_STORAGE_KEY, mode);
  } catch {
    // Persistence is a convenience; private-mode storage failures are fine.
  }
}
