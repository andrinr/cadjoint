/**
 * Keyboard shortcut vocabulary: the one table the handlers, the menu bar and
 * the Help dialog read.
 *
 * Every row binds a key to a *command id* (see `shell/commands.ts`), so the
 * key, the menu item and the Help sheet are three views of one record and
 * cannot drift apart. Two families:
 *
 * - **viewport keys** — a bare letter or symbol. They act only while nobody
 *   is typing (the editor, a text field, a select own their keys first) and
 *   are dispatched by `components/viewer/keyboard.ts`;
 * - **application keys** — ⌘ (Ctrl off the Mac) plus a key. They work from
 *   anywhere, always call `preventDefault` so the browser does not also save
 *   or open the page, and are dispatched by `shell/shellShortcuts.ts`.
 *
 * A few rows are *display-only* (`dispatch: false`): the view rose's digits,
 * which ViewCube matches by `KeyboardEvent.code` (see `VIEW_KEYS`), Escape's
 * ladder, Delete, and gestures. They are here so the menu and the sheet can
 * quote them; their handlers live with the thing they act on.
 */

/** One binding. */
export interface Shortcut {
  /** The command it runs (`shell/commands.ts`). */
  command: string;
  /** `KeyboardEvent.key`, lowercased; for display-only rows, informational. */
  key: string;
  /** Needs ⌘ on the Mac, Ctrl elsewhere. */
  mod?: boolean;
  /** Shift must be held (`true`), must not be (`false`, the default), or is
   *  part of the character itself (`"any"`, e.g. `?`). */
  shift?: boolean | "any";
  /** How the menu and the sheet print it. */
  label: string;
  /** What the sheet says it does. */
  action: string;
  /** Help sheet section. */
  group: ShortcutGroupTitle;
  /** False when another handler owns the key and this row only quotes it. */
  dispatch?: false;
}

export const SHORTCUT_GROUP_TITLES = [
  "Modes",
  "Tools",
  "Shape",
  "View",
  "Viewport",
  "File",
  "Program",
  "Simulate",
] as const;
export type ShortcutGroupTitle = (typeof SHORTCUT_GROUP_TITLES)[number];

export const SHORTCUTS: Shortcut[] = [
  // ── Modes ────────────────────────────────────────────────────────────
  { command: "cycle-mode", key: "m", label: "M", action: "Cycle Model → Sketch → Simulate", group: "Modes" },
  { command: "cycle-mode-back", key: "m", shift: true, label: "⇧M", action: "Cycle modes backwards", group: "Modes" },
  { command: "mode-model", key: "escape", label: "Esc", action: "Return to Model mode (after everything else is cancelled)", group: "Modes", dispatch: false },

  // ── Tools ────────────────────────────────────────────────────────────
  // O and V rather than 1 and 2: the digits belong to the view rose, on the
  // numpad convention every 3D application shares (see VIEW_KEYS).
  { command: "select-object", key: "o", label: "O", action: "Select whole objects", group: "Tools" },
  { command: "select-vertex", key: "v", label: "V", action: "Select sketch vertices", group: "Tools" },
  { command: "tool-sketch", key: "k", label: "K", action: "Start a sketch", group: "Tools" },
  { command: "tool-polygon", key: "p", label: "P", action: "Add a point to a sketch", group: "Tools" },
  { command: "tool-box", key: "b", label: "B", action: "Place a box", group: "Tools" },
  { command: "tool-sphere", key: "s", label: "S", action: "Place a sphere", group: "Tools" },
  { command: "tool-cylinder", key: "c", label: "C", action: "Place a cylinder", group: "Tools" },

  // ── Shape ────────────────────────────────────────────────────────────
  { command: "shape-extrude", key: "e", label: "E", action: "Extrude the selected sketch", group: "Shape" },
  { command: "gizmo-translate", key: "g", label: "G", action: "Move gizmo", group: "Shape" },
  { command: "gizmo-rotate", key: "r", label: "R", action: "Rotate gizmo", group: "Shape" },
  { command: "gizmo-scale", key: "t", label: "T", action: "Scale gizmo", group: "Shape" },

  // ── View ─────────────────────────────────────────────────────────────
  // Blender's numpad layout, on the main row as well, so a laptop gets it.
  // ViewCube matches these by code; Ctrl gives the opposite face.
  { command: "view-front", key: "1", label: "1", action: "Front view (Ctrl: back)", group: "View", dispatch: false },
  { command: "view-right", key: "3", label: "3", action: "Right view (Ctrl: left)", group: "View", dispatch: false },
  { command: "view-top", key: "7", label: "7", action: "Top view (Ctrl: bottom)", group: "View", dispatch: false },
  { command: "view-opposite", key: "9", label: "9", action: "Turn to the opposite side", group: "View", dispatch: false },
  { command: "view-ortho", key: "5", label: "5", action: "Toggle orthographic / perspective", group: "View", dispatch: false },
  { command: "grid-finer", key: "[", label: "[", action: "Finer grid", group: "View" },
  { command: "grid-coarser", key: "]", label: "]", action: "Coarser grid", group: "View" },
  { command: "show-construction", key: "c", shift: true, label: "⇧C", action: "Toggle the construction overlay", group: "View" },

  // ── Viewport ─────────────────────────────────────────────────────────
  { command: "pan-hold", key: " ", label: "Space (hold)", action: "Pan while dragging", group: "Viewport", dispatch: false },
  { command: "pan-drag", key: "shift", label: "⇧ drag", action: "Pan the camera", group: "Viewport", dispatch: false },
  { command: "grid-wheel", key: "alt", label: "⌥ wheel", action: "Step the grid", group: "Viewport", dispatch: false },
  { command: "select-none", key: "escape", label: "Esc", action: "Cancel the tool, then clear the selection", group: "Viewport", dispatch: false },
  { command: "edit-delete", key: "backspace", label: "⌫", action: "Delete the selection", group: "Viewport", dispatch: false },
  { command: "help-shortcuts", key: "?", shift: "any", label: "?", action: "This sheet", group: "Viewport" },

  // ── File ─────────────────────────────────────────────────────────────
  { command: "file-open", key: "o", mod: true, label: "⌘O", action: "Open a scene (the Scenes window)", group: "File" },
  { command: "file-save", key: "s", mod: true, label: "⌘S", action: "Save", group: "File" },
  { command: "file-save-as", key: "s", mod: true, shift: true, label: "⇧⌘S", action: "Save as", group: "File" },
  { command: "file-export", key: "e", mod: true, label: "⌘E", action: "Export a mesh or STEP", group: "File" },

  // ── Program ──────────────────────────────────────────────────────────
  { command: "program-run", key: "enter", mod: true, label: "⌘↵", action: "Run the program", group: "Program" },
  { command: "program-stop", key: ".", mod: true, label: "⌘.", action: "Stop the running compile or solve", group: "Program" },
  { command: "edit-undo", key: "z", mod: true, label: "⌘Z", action: "Undo the last source change", group: "Program" },
  { command: "edit-redo", key: "z", mod: true, shift: true, label: "⇧⌘Z", action: "Redo", group: "Program" },

  // ── Simulate ─────────────────────────────────────────────────────────
  { command: "sim-solve", key: "enter", mod: true, shift: true, label: "⇧⌘↵", action: "Solve the open (or first) study", group: "Simulate" },
];

/** The label of the key bound to `command`, if there is one. */
export function shortcutLabel(command: string): string | undefined {
  return SHORTCUTS.find((shortcut) => shortcut.command === command)?.label;
}

/** The subset of a `KeyboardEvent` the matcher reads. */
export interface KeyChord {
  key: string;
  shiftKey: boolean;
  metaKey: boolean;
  ctrlKey: boolean;
  altKey: boolean;
}

/**
 * The dispatchable shortcut a key event means, in one family.
 *
 * `family: "viewport"` matches bare keys (no ⌘/Ctrl/Alt); `"app"` matches
 * ⌘/Ctrl chords. Shift is exact unless the row says `"any"`.
 */
export function matchShortcut(
  event: KeyChord,
  family: "viewport" | "app",
): Shortcut | undefined {
  const mod = event.metaKey || event.ctrlKey;
  if (event.altKey) return undefined;
  if (family === "viewport" ? mod : !mod) return undefined;
  // Shift+/ is "?" on the layouts that have it there; some synthetic and
  // remapped key events report the unshifted character, so accept both.
  const raw = event.key.toLowerCase();
  const key = raw === "/" && event.shiftKey ? "?" : raw;
  return SHORTCUTS.find(
    (shortcut) =>
      shortcut.dispatch !== false &&
      Boolean(shortcut.mod) === (family === "app") &&
      shortcut.key === key &&
      (shortcut.shift === "any" || Boolean(shortcut.shift) === event.shiftKey),
  );
}

/**
 * The view rose's keys, on Blender's numpad convention.
 *
 * 1 front, 3 right, 7 top, with Ctrl for the opposite face; 5 toggles the
 * projection; 9 turns the camera to the other side of whatever it is looking
 * at. It is the one layout a CAD or DCC user already has in their hands, and
 * the geometry behind it is worth stating: 1/3/7 are the three corners of the
 * numpad's bottom-left triangle, which is the same +X/+Y/+Z frame the cube
 * shows.
 *
 * Keyed by `KeyboardEvent.code`, and both the main row and the numpad are
 * listed: a laptop has no numpad, and a keyboard that has one should not have
 * its dedicated view keys ignored.
 */
export interface ViewKeyBinding {
  /** VIEW_PRESETS key, or null when the binding is not a direction. */
  preset: string | null;
  /** The view reached by holding Ctrl. */
  opposite?: string;
  /** Set on the key that toggles orthographic against perspective. */
  projection?: true;
  /** Set on the key that swings the camera to the opposite side. */
  reverse?: true;
}

export const VIEW_KEYS: Record<string, ViewKeyBinding> = {
  Digit1: { preset: "front", opposite: "back" },
  Numpad1: { preset: "front", opposite: "back" },
  Digit3: { preset: "right", opposite: "left" },
  Numpad3: { preset: "right", opposite: "left" },
  Digit7: { preset: "top", opposite: "bottom" },
  Numpad7: { preset: "top", opposite: "bottom" },
  Digit5: { preset: null, projection: true },
  Numpad5: { preset: null, projection: true },
  Digit9: { preset: null, reverse: true },
  Numpad9: { preset: null, reverse: true },
};

export interface ShortcutItem {
  keys: string;
  action: string;
}

export interface ShortcutGroup {
  title: string;
  items: ShortcutItem[];
}

/** The Help sheet, grouped, straight from `SHORTCUTS`. */
export const SHORTCUT_GROUPS: ShortcutGroup[] = SHORTCUT_GROUP_TITLES.map((title) => ({
  title,
  items: SHORTCUTS.filter((shortcut) => shortcut.group === title).map((shortcut) => ({
    keys: shortcut.label,
    action: shortcut.action,
  })),
}));
