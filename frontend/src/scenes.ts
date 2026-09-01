/**
 * Scene-file name handling for the File menu.
 *
 * Mirrors the server's rules (`sanitize_scene_name` in playground.py) so a
 * bad name is refused in the dialog instead of round-tripping to a 4xx.
 */

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
