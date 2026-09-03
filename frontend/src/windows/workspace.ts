/**
 * Layout persistence: what the dock looked like, per mode, across reloads.
 *
 * One browser-local record holds three things per editing mode — the dock
 * library's serialised grid, our own open/minimised/closed statuses, and the
 * geometry of any floating windows (the library keeps that inside its own
 * blob, so we only carry what it does not).
 *
 * Everything here is pure: `parseWorkspace` never throws on junk, unknown
 * window ids are dropped rather than trusted, and a layout from a future
 * version is discarded in favour of the defaults. A stored layout is a
 * convenience, never a correctness requirement — a user who ends up with an
 * unreadable record must get a working dock, not a blank page.
 */

import { EDITING_MODES, type EditingMode } from "../editingMode";
import { isWindowId, WINDOW_IDS, type WindowId } from "./panels";
import {
  defaultModeWindows,
  defaultWindowStates,
  type ModeWindows,
  type WindowStates,
  type WindowStatus,
} from "./windowState";

export const WORKSPACE_STORAGE_KEY = "cadjoint.windows.v1";
export const WORKSPACE_VERSION = 1;

/**
 * The dock library's serialised grid. Opaque on purpose: its shape is the
 * library's business, and pinning it here would make every upgrade a
 * migration. We only ever read `panels` to check the ids are ones we know.
 */
export interface DockLayout {
  panels?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface Workspace {
  version: number;
  /** Per-mode dock grid, absent when that mode has never been arranged. */
  layouts: Partial<Record<EditingMode, DockLayout>>;
  /** Per-mode window statuses. */
  windows: WindowStates;
}

export function emptyWorkspace(): Workspace {
  return { version: WORKSPACE_VERSION, layouts: {}, windows: defaultWindowStates() };
}

/** The window ids a serialised dock grid refers to. */
export function layoutPanelIds(layout: DockLayout | undefined): string[] {
  if (!layout || typeof layout !== "object") return [];
  const panels = layout.panels;
  if (!panels || typeof panels !== "object") return [];
  return Object.keys(panels);
}

/**
 * Whether a stored grid can still be restored.
 *
 * A layout naming a window this build no longer has would leave the library
 * asking for a component we cannot create, so the whole layout is dropped and
 * the mode falls back to its default arrangement. That is a rare, cheap loss;
 * a half-restored dock is not.
 */
export function isRestorableLayout(layout: DockLayout | undefined): boolean {
  const ids = layoutPanelIds(layout);
  if (ids.length === 0) return false;
  return ids.every((id) => isWindowId(id));
}

function parseModeWindows(value: unknown, mode: EditingMode): ModeWindows {
  const fallback = defaultModeWindows(mode);
  if (!value || typeof value !== "object") return fallback;
  const record = value as Record<string, unknown>;
  const next = { ...fallback };
  for (const id of WINDOW_IDS) {
    const status = record[id];
    if (status === "open" || status === "minimised" || status === "closed") {
      next[id] = status as WindowStatus;
    }
  }
  return next;
}

/** Read a stored record, tolerating every shape of corruption. */
export function parseWorkspace(raw: string | null | undefined): Workspace {
  if (!raw) return emptyWorkspace();
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return emptyWorkspace();
  }
  if (!parsed || typeof parsed !== "object") return emptyWorkspace();
  const record = parsed as Record<string, unknown>;
  // A record written by a newer build may mean anything; start clean.
  if (record.version !== WORKSPACE_VERSION) return emptyWorkspace();

  const workspace = emptyWorkspace();
  const layouts = record.layouts;
  const windows = record.windows;
  for (const mode of EDITING_MODES) {
    if (layouts && typeof layouts === "object") {
      const layout = (layouts as Record<string, unknown>)[mode] as DockLayout | undefined;
      if (isRestorableLayout(layout)) workspace.layouts[mode] = layout;
    }
    workspace.windows[mode] = parseModeWindows(
      windows && typeof windows === "object"
        ? (windows as Record<string, unknown>)[mode]
        : undefined,
      mode,
    );
  }
  return workspace;
}

export function serializeWorkspace(workspace: Workspace): string {
  return JSON.stringify({
    version: WORKSPACE_VERSION,
    layouts: workspace.layouts,
    windows: workspace.windows,
  });
}

/** A round trip through storage keeps only what `parseWorkspace` accepts. */
export function normaliseWorkspace(workspace: Workspace): Workspace {
  return parseWorkspace(serializeWorkspace(workspace));
}

/** The minimal storage surface, so tests can pass a plain object. */
export interface WorkspaceStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

export function readWorkspace(storage: WorkspaceStorage | undefined): Workspace {
  if (!storage) return emptyWorkspace();
  try {
    return parseWorkspace(storage.getItem(WORKSPACE_STORAGE_KEY));
  } catch {
    // Private-mode storage throws on read as well as write.
    return emptyWorkspace();
  }
}

export function writeWorkspace(
  storage: WorkspaceStorage | undefined,
  workspace: Workspace,
): void {
  if (!storage) return;
  try {
    storage.setItem(WORKSPACE_STORAGE_KEY, serializeWorkspace(workspace));
  } catch {
    // Layout persistence is a convenience; a full quota must not break the dock.
  }
}

export function clearWorkspace(storage: WorkspaceStorage | undefined): void {
  if (!storage) return;
  try {
    storage.removeItem(WORKSPACE_STORAGE_KEY);
  } catch {
    // As above.
  }
}

/** Windows a mode's stored grid places in the dock, filtered to known ids. */
export function storedWindowsForMode(
  workspace: Workspace,
  mode: EditingMode,
): WindowId[] {
  return layoutPanelIds(workspace.layouts[mode]).filter(isWindowId);
}
