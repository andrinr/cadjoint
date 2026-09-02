/**
 * Keyboard shortcut vocabulary, shared by the handlers and the Help dialog.
 *
 * `VIEWER_TOOL_KEYS` is the authoritative key→action table consumed by
 * ViewerPane's key handler; the Help dialog renders `SHORTCUT_GROUPS`, which
 * is written against the same actions, so the two cannot drift silently.
 */

/** Single-key tool actions handled by the viewer's key handler. */
export type ViewerToolAction =
  | "select-object"
  | "select-vertex"
  | "tool-polygon"
  | "tool-box"
  | "tool-sphere"
  | "tool-cylinder"
  | "gizmo-translate"
  | "gizmo-rotate"
  | "cycle-mode";

export const VIEWER_TOOL_KEYS: Record<string, ViewerToolAction> = {
  // O and V rather than 1 and 2: the digits belong to the view rose now, on
  // the numpad convention every 3D application shares (see VIEW_KEYS). The
  // letters are the mnemonic the digits never were — O for object, V for
  // vertex — and the tool rail's tooltips quote them.
  o: "select-object",
  v: "select-vertex",
  p: "tool-polygon",
  b: "tool-box",
  s: "tool-sphere",
  c: "tool-cylinder",
  g: "gizmo-translate",
  r: "gizmo-rotate",
  m: "cycle-mode",
};

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

export const SHORTCUT_GROUPS: ShortcutGroup[] = [
  {
    title: "Modes",
    items: [
      { keys: "M", action: "Cycle Model → Sketch → Simulate" },
      { keys: "Shift + M", action: "Cycle modes backwards" },
      { keys: "Esc", action: "Return to Model mode" },
    ],
  },
  {
    title: "Tools",
    items: [
      { keys: "O", action: "Select whole objects" },
      { keys: "V", action: "Select sketch vertices" },
      { keys: "P", action: "Add a point to a sketch" },
      { keys: "B", action: "Place a box" },
      { keys: "S", action: "Place a sphere" },
      { keys: "C", action: "Place a cylinder" },
    ],
  },
  {
    title: "Transform",
    items: [
      { keys: "G", action: "Move gizmo" },
      { keys: "R", action: "Rotate gizmo" },
    ],
  },
  {
    // Blender's numpad layout, on the main row as well, so a laptop gets it.
    // The rose sets a *direction*; 5 is the only key here that touches the
    // projection, because that is a different question entirely.
    title: "View",
    items: [
      { keys: "1 / Ctrl + 1", action: "Front view / Back view" },
      { keys: "3 / Ctrl + 3", action: "Right view / Left view" },
      { keys: "7 / Ctrl + 7", action: "Top view / Bottom view" },
      { keys: "9", action: "Turn to the opposite side" },
      { keys: "5", action: "Toggle orthographic / perspective" },
      { keys: "Click the cube", action: "Face, edge, or corner view" },
    ],
  },
  {
    title: "Viewport",
    items: [
      { keys: "Space (hold)", action: "Pan while dragging" },
      { keys: "Shift + drag", action: "Pan the camera" },
      { keys: "Esc", action: "Cancel the tool and clear the selection" },
      { keys: "Del / ⌫", action: "Delete the selection" },
    ],
  },
  {
    title: "Application",
    items: [
      { keys: "Ctrl/⌘ + Enter", action: "Run the program" },
      { keys: "Ctrl/⌘ + Z", action: "Undo the last source change" },
      { keys: "Ctrl/⌘ + Shift + Z", action: "Redo" },
    ],
  },
];
