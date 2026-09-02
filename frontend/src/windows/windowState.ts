/**
 * The window state machine: open, closed, minimised.
 *
 * The dock library owns geometry — which group a window sits in, whether it is
 * tabbed, floating, or split. What it has no opinion about is the two states a
 * window can be in while *not* in the grid: closed (gone; the Window menu can
 * bring it back) and minimised (parked in the tray strip, one click from where
 * it was). That bookkeeping is ours, and it is pure, so it is tested without a
 * DOM.
 *
 * The rules are small and worth stating:
 *   - the viewport is permanent: close and minimise are no-ops on it;
 *   - minimising a closed window does nothing (there is nothing to park);
 *   - restoring anything — closed or minimised — opens it;
 *   - each mode carries its own state, so leaving Simulate with the Objects
 *     window minimised and coming back finds it still minimised there.
 */

import type { EditingMode } from "../editingMode";
import { EDITING_MODES } from "../editingMode";
import {
  defaultWindowsForMode,
  isPermanent,
  isWindowId,
  WINDOW_IDS,
  type WindowId,
} from "./panels";

export type WindowStatus = "open" | "minimised" | "closed";

/** Every window's status in one mode. */
export type ModeWindows = Record<WindowId, WindowStatus>;

/** Every mode's window statuses. */
export type WindowStates = Record<EditingMode, ModeWindows>;

export type WindowAction =
  | { kind: "open"; id: WindowId }
  | { kind: "close"; id: WindowId }
  | { kind: "minimise"; id: WindowId }
  | { kind: "restore"; id: WindowId }
  | { kind: "toggle"; id: WindowId }
  | { kind: "reset"; mode: EditingMode };

/** A mode's default statuses: its own windows open, everything else closed. */
export function defaultModeWindows(mode: EditingMode): ModeWindows {
  const open = new Set<string>(defaultWindowsForMode(mode));
  const state = {} as ModeWindows;
  for (const id of WINDOW_IDS) state[id] = open.has(id) ? "open" : "closed";
  return state;
}

export function defaultWindowStates(): WindowStates {
  const states = {} as WindowStates;
  for (const mode of EDITING_MODES) states[mode] = defaultModeWindows(mode);
  return states;
}

/**
 * Apply an action to one mode's windows.
 *
 * Returns the same object when nothing changed, so callers can skip a write.
 */
export function reduceModeWindows(state: ModeWindows, action: WindowAction): ModeWindows {
  if (action.kind === "reset") return defaultModeWindows(action.mode);
  const { id } = action;
  if (!isWindowId(id)) return state;
  const current = state[id];
  let next: WindowStatus = current;

  switch (action.kind) {
    case "open":
    case "restore":
      next = "open";
      break;
    case "close":
      next = isPermanent(id) ? current : "closed";
      break;
    case "minimise":
      // Nothing to park: a closed window has no place to come back to.
      next = isPermanent(id) || current === "closed" ? current : "minimised";
      break;
    case "toggle":
      next = current === "open" ? (isPermanent(id) ? "open" : "closed") : "open";
      break;
  }

  if (next === current) return state;
  return { ...state, [id]: next };
}

export function isOpen(state: ModeWindows, id: WindowId): boolean {
  return state[id] === "open";
}

/** Windows parked in the tray, in the canonical window order. */
export function minimisedWindows(state: ModeWindows): WindowId[] {
  return WINDOW_IDS.filter((id) => state[id] === "minimised");
}

/** Windows the Window menu can reopen: closed, or parked. */
export function reopenableWindows(state: ModeWindows): WindowId[] {
  return WINDOW_IDS.filter((id) => state[id] !== "open");
}
