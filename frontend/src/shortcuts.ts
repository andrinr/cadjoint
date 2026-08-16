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
  | "gizmo-rotate";

export const VIEWER_TOOL_KEYS: Record<string, ViewerToolAction> = {
  "1": "select-object",
  "2": "select-vertex",
  p: "tool-polygon",
  b: "tool-box",
  s: "tool-sphere",
  c: "tool-cylinder",
  g: "gizmo-translate",
  r: "gizmo-rotate",
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
    title: "Tools",
    items: [
      { keys: "1", action: "Select whole objects" },
      { keys: "2", action: "Select sketch vertices" },
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
