/**
 * The menu bar, as data.
 *
 * The structure is not written here: it is `proposedMenus()` from the design
 * page's feature index (`design/features.ts`), the list that was agreed as the
 * menu bar. Every leaf there is bound below to a command id, so the menu the
 * app shows is exactly the menu the design page shows, and a feature added to
 * the index without a command fails `test/menuModel.test.ts` rather than
 * appearing as a dead item.
 *
 * Keys come from `shortcuts.ts` — the table the handlers dispatch from — not
 * from the index's prose, so the key a menu item quotes is the key that runs
 * it.
 */

import { MENU_ORDER, proposedMenus } from "../design/features";
import type { EditingMode } from "../editingMode";
import { shortcutLabel } from "../shortcuts";
import { WINDOW_DEFS, type WindowId } from "../windows/panels";

export interface MenuEntry {
  label: string;
  command: string;
  /** Printed right-aligned. */
  key?: string;
  /** Modes the item applies in; absent means all. */
  modes?: EditingMode[];
  /** One of a set (the modes): drawn with a dot, not a check. */
  radio?: boolean;
  testid: string;
}

export interface MenuGroup {
  title: string | null;
  entries: MenuEntry[];
  /** Opens to the side from one parent item instead of listing inline. */
  submenu: boolean;
}

export interface MenuModel {
  /** "File" */
  name: string;
  /** "file", for test ids. */
  id: string;
  groups: MenuGroup[];
}

/** Every menu leaf, by its path, to the command it runs. */
export const MENU_COMMANDS: Record<string, string> = {
  "File › New": "file-new",
  "File › Open…": "file-open",
  "File › Save": "file-save",
  "File › Save As…": "file-save-as",
  "File › Download scene.py": "file-download",
  "File › Export…": "file-export",

  "File › Program › Run": "program-run",
  "File › Program › Stop": "program-stop",
  "Help › Developer › Show generated WGSL": "program-wgsl",

  "Edit › Undo": "edit-undo",
  "Edit › Redo": "edit-redo",
  "Edit › Select › Objects": "select-object",
  "Edit › Select › Vertices": "select-vertex",
  "Edit › Select › None": "select-none",
  "Edit › Delete": "edit-delete",

  "View › Camera › Front": "view-front",
  "View › Camera › Right": "view-right",
  "View › Camera › Top": "view-top",
  "View › Camera › Opposite side": "view-opposite",
  "View › Camera › Orthographic": "view-ortho",
  "View › Zoom › In one grid step": "grid-finer",
  "View › Zoom › Out one grid step": "grid-coarser",
  "View › Render › Preset…": "render-preset",
  "View › Render › Settings…": "render-settings",
  "View › Show › Construction": "show-construction",
  "View › Show › Feature edges": "show-edges",
  "View › Show › Wireframe": "show-wireframe",
  "View › Show › Solid": "show-solid",
  "View › Inspect › Distance slice": "inspect-slice",
  "View › Inspect › Normals": "inspect-normals",
  "View › Inspect › Depth": "inspect-depth",

  "Model › Create › Sketch": "tool-sketch",
  "Model › Create › Sketch on face": "tool-face",
  "Model › Create › Point": "tool-polygon",
  "Model › Create › Box": "tool-box",
  "Model › Create › Sphere": "tool-sphere",
  "Model › Create › Cylinder": "tool-cylinder",
  "Model › Create › Material": "create-material",

  "Model › Shape › Extrude": "shape-extrude",
  "Model › Shape › Revolve": "shape-revolve",
  "Model › Shape › Loft": "shape-loft",
  "Model › Shape › Move": "gizmo-translate",
  "Model › Shape › Rotate": "gizmo-rotate",
  "Model › Shape › Scale": "gizmo-scale",

  "Model › Constrain › Distance": "constrain-distance",
  "Model › Constrain › Horizontal": "constrain-horizontal",
  "Model › Constrain › Vertical": "constrain-vertical",
  "Model › Constrain › Coincident": "constrain-coincident",
  "Model › Constrain › Parallel": "constrain-parallel",
  "Model › Constrain › Perpendicular": "constrain-perpendicular",
  "Model › Constrain › Fix": "constrain-fix",

  "Simulate › New mesh": "sim-new-mesh",
  "Simulate › New study › Thermal": "sim-study-thermal",
  "Simulate › New study › Elastic": "sim-study-elastic",
  "Simulate › New study › Flow": "sim-study-flow",
  "Simulate › Solve": "sim-solve",
  "Simulate › Optimize": "sim-optimize",

  "Window › Mode › Model": "mode-model",
  "Window › Mode › Sketch": "mode-sketch",
  "Window › Mode › Simulate": "mode-simulate",
  ...Object.fromEntries(
    WINDOW_DEFS.filter((def) => !def.permanent).map((def) => [
      `Window › Windows › ${def.title}`,
      `window-${def.id}`,
    ]),
  ),
  "Window › Float the focused window": "window-float",
  "Window › Reset layout": "window-reset",

  "Help › Keyboard shortcuts": "help-shortcuts",
  "Help › Design language": "help-design",
  "Help › README": "help-readme",
};

/**
 * Test ids the end-to-end specs already address, kept so the menu could be
 * regenerated without rewriting them. Everything else is `menu-<command>`.
 */
const LEGACY_TESTIDS: Record<string, string> = {
  "file-new": "menu-file-new",
  "file-open": "menu-file-open",
  "file-save": "menu-file-save",
  "file-save-as": "menu-file-save-as",
  "file-download": "menu-file-download",
  "file-export": "menu-file-export",
  "edit-undo": "menu-edit-undo",
  "edit-redo": "menu-edit-redo",
  "mode-model": "menu-window-mode-model",
  "mode-sketch": "menu-window-mode-sketch",
  "mode-simulate": "menu-window-mode-simulate",
  "window-reset": "menu-window-reset",
  "help-shortcuts": "menu-help-shortcuts",
  "help-readme": "menu-help-readme",
};

export function menuPath(menu: string, section: string | null, label: string): string {
  return section ? `${menu} › ${section} › ${label}` : `${menu} › ${label}`;
}

/** The window a `window-<id>` command toggles, or null. */
export function windowOfCommand(command: string): WindowId | null {
  if (!command.startsWith("window-")) return null;
  const id = command.slice("window-".length);
  return WINDOW_DEFS.some((def) => def.id === id) ? (id as WindowId) : null;
}

export function testidFor(command: string): string {
  if (LEGACY_TESTIDS[command]) return LEGACY_TESTIDS[command];
  // Windows keep the Window menu's `menu-window-<id>` the specs open them by.
  return `menu-${command}`;
}

/** The seven menus, in order, every leaf bound to its command. */
export function menuModel(): MenuModel[] {
  const proposed = proposedMenus();
  return MENU_ORDER.map((name) => ({
    name,
    id: name.toLowerCase(),
    groups: (proposed.get(name) ?? []).map((section) => ({
      title: section.title,
      submenu: section.submenu,
      entries: section.items.map((item) => {
        const path = menuPath(name, section.title, item.label);
        const command = MENU_COMMANDS[path] ?? "";
        return {
          label: item.label,
          command,
          key: shortcutLabel(command),
          modes: item.modes,
          radio: command.startsWith("mode-") || undefined,
          testid: testidFor(command),
        };
      }),
    })),
  }));
}

/** Leaves of `proposedMenus()` that no command is bound to (should be none). */
export function unboundMenuPaths(): string[] {
  const missing: string[] = [];
  for (const [menu, sections] of proposedMenus()) {
    for (const section of sections) {
      for (const item of section.items) {
        const path = menuPath(menu, section.title, item.label);
        if (!MENU_COMMANDS[path]) missing.push(path);
      }
    }
  }
  return missing;
}

/** Why an item is disabled by mode, or null when it applies in `mode`. */
export function modeReason(modes: EditingMode[] | undefined, mode: EditingMode): string | null {
  if (!modes || modes.includes(mode)) return null;
  const list =
    modes.length === 1 ? modes[0] : `${modes.slice(0, -1).join(", ")} or ${modes.at(-1)}`;
  return `Only in ${list} mode`;
}

/** Menu ids, in bar order, for arrow-key travel between menus. */
export const MENU_IDS = MENU_ORDER.map((name) => name.toLowerCase());
