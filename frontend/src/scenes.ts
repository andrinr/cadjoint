/**
 * Scene-file name handling, and the guard that stands in front of a load.
 *
 * The naming rules mirror the server's (`sanitize_scene_name` in
 * `_scenes.py`) so a bad name is refused in the dialog instead of
 * round-tripping to a 4xx. The discard guard lives here because two places
 * now replace the whole program — the File menu and the Scenes window — and
 * asking the question in two different wordings would be two different
 * promises about the same unsaved work.
 */

import { dirty } from "./state";

const SCENE_STEM = /^[A-Za-z0-9_-][A-Za-z0-9._ -]*$/;
export const MAX_SCENE_NAME_LENGTH = 128;

/**
 * Normalize a typed scene name to `stem.py`, or return null when invalid.
 *
 * A missing `.py` suffix is added; path separators, traversal, hidden files,
 * and empty names are rejected.
 */
export function sanitizeSceneName(raw: string): string | null {
  const trimmed = raw.trim();
  if (trimmed === "") return null;
  const name = trimmed.endsWith(".py") ? trimmed : `${trimmed}.py`;
  if (name.length > MAX_SCENE_NAME_LENGTH) return null;
  if (name.includes("/") || name.includes("\\") || name.includes("\0")) return null;
  const stem = name.slice(0, -".py".length);
  if (!SCENE_STEM.test(stem)) return null;
  return name;
}

/**
 * Ask before replacing an edited program; true when it is safe to proceed.
 *
 * A clean buffer never asks: the confirmation exists to protect work, and a
 * dialog in front of an action that loses nothing only trains people to
 * dismiss dialogs.
 */
export function confirmDiscardChanges(): boolean {
  return !dirty() || window.confirm("Discard unsaved changes to the current scene?");
}
