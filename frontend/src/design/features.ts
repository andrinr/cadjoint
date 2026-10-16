/**
 * Every feature of the playground and where the rules put it.
 *
 * One record per feature. The design page renders both the feature index and
 * the menu bar from this list, so the menus it shows are exactly the commands
 * here.
 *
 * Seven menus carry `rule.menu`: Create / Shape / Constrain nest under Model
 * as submenus, Run and Stop are a section of File, the WGSL view is Help ›
 * Developer. Modes live under Window; a mode-only command is shown disabled
 * with the reason.
 *
 * The placement rules this list applies (N5, N9 on the page):
 *
 * - A command lives on what it acts on: a declaration's card, the toolbar
 *   for the whole program, the menu bar for the application, the rail and
 *   the keyboard for the picture.
 * - The menu bar is the complete index. Every command has a menu home, and
 *   the toolbar, rail, panels and keys are shortcuts to it. The menu shows a
 *   command that does not apply in the current mode disabled, with the
 *   reason; the rail and panels show only what applies.
 * - One name per command, the same in every home.
 */

import type { EditingMode } from "../editingMode";

/** Where a feature can be reached. Every field is optional. */
export interface Homes {
  /** "Menu › Section › Item", or "Menu › Item". */
  menu?: string;
  /** Toolbar control, described. */
  toolbar?: string;
  /** Tool-rail group › item. */
  rail?: string;
  /** Keyboard shortcut. */
  key?: string;
  /** Window (and the control in it). */
  panel?: string;
  /** Direct manipulation in the viewport. */
  viewport?: string;
}

export interface Feature {
  name: string;
  group: FeatureGroup;
  /** Modes it applies in; absent means every mode. */
  modes?: EditingMode[];
  /** Where the rules put it. */
  rule: Homes;
}

export const FEATURE_GROUPS = [
  "File",
  "Program",
  "Edit and selection",
  "View",
  "Create",
  "Shape",
  "Constrain",
  "Materials",
  "Simulate",
  "Optimize",
  "Desk",
  "Help",
] as const;
export type FeatureGroup = (typeof FEATURE_GROUPS)[number];

const MODEL_SKETCH: EditingMode[] = ["model", "sketch"];

export const FEATURES: Feature[] = [
  // ── File ────────────────────────────────────────────────────────────────
  {
    name: "New from the starter",
    group: "File",
    rule: { menu: "File › New" },
  },
  {
    name: "Open a scene",
    group: "File",
    rule: { menu: "File › Open…", key: "⌘O", panel: "Scenes (parked)" },
  },
  {
    name: "Save",
    group: "File",
    rule: { menu: "File › Save", key: "⌘S" },
  },
  {
    name: "Save as",
    group: "File",
    rule: { menu: "File › Save As…", key: "⇧⌘S" },
  },
  {
    name: "Download scene.py",
    group: "File",
    rule: { menu: "File › Download scene.py" },
  },
  {
    name: "Export mesh or STEP",
    group: "File",
    rule: { menu: "File › Export…", key: "⌘E" },
  },
  {
    name: "Refresh the scene list",
    group: "File",
    rule: { panel: "Scenes › Refresh" },
  },

  // ── Program ─────────────────────────────────────────────────────────────
  {
    name: "Run",
    group: "Program",
    rule: { menu: "File › Program › Run", toolbar: "Run", key: "⌘↵" },
  },
  {
    name: "Stop the compile",
    group: "Program",
    rule: { menu: "File › Program › Stop", toolbar: "job chip ×", key: "⌘." , panel: "Processes" },
  },
  {
    name: "Show generated WGSL",
    group: "Program",
    rule: { menu: "Help › Developer › Show generated WGSL" },
  },
  {
    name: "Undo",
    group: "Program",
    rule: { menu: "Edit › Undo", key: "⌘Z" },
  },
  {
    name: "Redo",
    group: "Program",
    rule: { menu: "Edit › Redo", key: "⇧⌘Z" },
  },

  // ── Edit and selection ──────────────────────────────────────────────────
  {
    name: "Select whole objects",
    group: "Edit and selection",
    modes: MODEL_SKETCH,
    rule: { menu: "Edit › Select › Objects", rail: "Select › Object", key: "O" },
  },
  {
    name: "Select sketch vertices",
    group: "Edit and selection",
    modes: MODEL_SKETCH,
    rule: { menu: "Edit › Select › Vertices", rail: "Select › Vertex", key: "V" },
  },
  {
    name: "Delete the selection",
    group: "Edit and selection",
    modes: MODEL_SKETCH,
    rule: { menu: "Edit › Delete", rail: "Delete", key: "⌫" },
  },
  {
    name: "Clear the selection",
    group: "Edit and selection",
    rule: { menu: "Edit › Select › None", key: "Esc", viewport: "click empty space" },
  },
  {
    name: "Edit an argument",
    group: "Edit and selection",
    rule: { panel: "Properties", viewport: "drag a handle" },
  },
  {
    name: "Follow the selection again",
    group: "Edit and selection",
    rule: { panel: "Properties › back" },
  },

  // ── View ────────────────────────────────────────────────────────────────
  {
    name: "Standard views",
    group: "View",
    rule: { menu: "View › Camera › Front, Right, Top", key: "1 3 7", viewport: "view cube faces, ⌃ for the opposite" },
  },
  {
    name: "Turn to the opposite side",
    group: "View",
    rule: { menu: "View › Camera › Opposite side", key: "9" },
  },
  {
    name: "Orthographic / perspective",
    group: "View",
    rule: { menu: "View › Camera › Orthographic", key: "5", viewport: "beside the cube › projection" },
  },
  {
    name: "Orbit, pan, zoom",
    group: "View",
    rule: { viewport: "drag, Shift/Space + drag, wheel" },
  },
  {
    name: "Zoom one grid step",
    group: "View",
    rule: { menu: "View › Zoom › In one grid step, Out one grid step", key: "[ ]", viewport: "Alt + wheel" },
  },
  {
    name: "Render preset",
    group: "View",
    rule: { menu: "View › Render › Preset…", viewport: "beside the cube › Render settings › presets" },
  },
  {
    name: "Construction overlay",
    group: "View",
    rule: { menu: "View › Show › Construction", viewport: "beside the cube › Render settings › Annotations", key: "⇧C" },
  },
  {
    name: "Layout grid",
    group: "View",
    rule: { menu: "View › Show › Layout grid", key: "⇧G" },
  },
  {
    name: "Feature edges, mesh wireframe",
    group: "View",
    rule: { menu: "View › Show › Feature edges, Wireframe", viewport: "beside the cube › Render settings › Annotations" },
  },
  {
    name: "Hide the solid",
    group: "View",
    rule: { menu: "View › Show › Solid", viewport: "beside the cube › Render settings › Annotations" },
  },
  {
    name: "Distance-field slice",
    group: "View",
    rule: { menu: "View › Inspect › Distance slice", viewport: "beside the cube › Render settings › Distance field" },
  },
  {
    name: "Normals, depth",
    group: "View",
    rule: { menu: "View › Inspect › Normals, Depth", viewport: "beside the cube › Render settings › Diagnostics" },
  },
  {
    name: "Shading, shadows, reflections, quality",
    group: "View",
    rule: { viewport: "beside the cube › Render settings › Shading, Effects, Marching", menu: "View › Render › Settings…" },
  },

  // ── Create ──────────────────────────────────────────────────────────────
  {
    name: "Sketch",
    group: "Create",
    modes: MODEL_SKETCH,
    rule: { menu: "Model › Create › Sketch", rail: "Create › Sketch", key: "K" },
  },
  {
    name: "Sketch on a face",
    group: "Create",
    modes: MODEL_SKETCH,
    rule: { menu: "Model › Create › Sketch on face", rail: "Create › Sketch on face" },
  },
  {
    name: "Point",
    group: "Create",
    modes: MODEL_SKETCH,
    rule: { menu: "Model › Create › Point", rail: "Create › Point", key: "P" },
  },
  {
    name: "Box, sphere, cylinder",
    group: "Create",
    modes: ["model"],
    rule: { menu: "Model › Create › Box, Sphere, Cylinder", rail: "Create › Box, Sphere, Cylinder", key: "B S C" },
  },

  // ── Shape ───────────────────────────────────────────────────────────────
  {
    name: "Extrude, revolve, loft",
    group: "Shape",
    modes: MODEL_SKETCH,
    rule: { menu: "Model › Shape › Extrude, Revolve, Loft", rail: "Shape › Extrude, Revolve, Loft", key: "E" },
  },
  {
    name: "Move, rotate, scale",
    group: "Shape",
    modes: MODEL_SKETCH,
    rule: { menu: "Model › Shape › Move, Rotate, Scale", rail: "Transform › Move, Rotate, Scale", key: "G R T" },
  },

  // ── Constrain ───────────────────────────────────────────────────────────
  {
    name: "Distance, horizontal, vertical, coincident",
    group: "Constrain",
    modes: ["sketch"],
    rule: { menu: "Model › Constrain › Distance, Horizontal, Vertical, Coincident", rail: "Constrain › Distance, Horizontal, Vertical, Coincident", panel: "Sketch (list only)" },
  },
  {
    name: "Parallel, perpendicular",
    group: "Constrain",
    modes: ["sketch"],
    rule: { menu: "Model › Constrain › Parallel, Perpendicular", rail: "Constrain › Parallel, Perpendicular", panel: "Sketch (list only)" },
  },
  {
    name: "Fix a point",
    group: "Constrain",
    modes: ["sketch"],
    rule: { menu: "Model › Constrain › Fix", rail: "Constrain › Fix", panel: "Sketch › anchor" },
  },
  {
    name: "Remove a constraint",
    group: "Constrain",
    modes: ["sketch"],
    rule: { panel: "Sketch › chip ×", key: "⌫ on a selected chip" },
  },
  {
    name: "Constraint solver settings",
    group: "Constrain",
    modes: ["sketch"],
    rule: { panel: "Sketch › solver" },
  },

  // ── Materials ───────────────────────────────────────────────────────────
  {
    name: "New material",
    group: "Materials",
    modes: ["model"],
    rule: { menu: "Model › Create › Material", panel: "Materials › +" },
  },
  {
    name: "Assign a material",
    group: "Materials",
    rule: { panel: "Properties › material · Materials (drag)", viewport: "drop" },
  },
  {
    name: "Edit a material",
    group: "Materials",
    rule: { panel: "Materials › inspector" },
  },

  // ── Simulate ────────────────────────────────────────────────────────────
  {
    name: "New mesh",
    group: "Simulate",
    modes: ["simulate"],
    rule: { menu: "Simulate › New mesh", panel: "Meshes › +" },
  },
  {
    name: "Build a mesh",
    group: "Simulate",
    modes: ["simulate"],
    rule: { panel: "Meshes › card" },
  },
  {
    name: "New study",
    group: "Simulate",
    modes: ["simulate"],
    rule: { menu: "Simulate › New study › Thermal, Elastic, Flow", panel: "Studies › + Study (kind menu)" },
  },
  {
    name: "Add a boundary condition",
    group: "Simulate",
    modes: ["simulate"],
    rule: { panel: "Studies › card › + Boundary condition", viewport: "pick faces" },
  },
  {
    name: "Solve",
    group: "Simulate",
    modes: ["simulate"],
    rule: { menu: "Simulate › Solve", panel: "Studies › card › Solve", key: "⇧⌘↵" },
  },
  {
    name: "Stop a solve",
    group: "Simulate",
    modes: ["simulate"],
    rule: { menu: "File › Program › Stop", panel: "Studies › card › Stop", toolbar: "job chip ×", key: "⌘." },
  },
  {
    name: "Probe the field",
    group: "Simulate",
    modes: ["simulate"],
    rule: { viewport: "hover the field", panel: "Results › probe" },
  },
  {
    name: "Deformation scale",
    group: "Simulate",
    modes: ["simulate"],
    rule: { panel: "Results › warp" },
  },

  // ── Optimize ────────────────────────────────────────────────────────────
  {
    name: "Run an optimization",
    group: "Optimize",
    rule: { menu: "Simulate › Optimize", panel: "Optimize › card › Run" },
  },
  {
    name: "Replay a trajectory",
    group: "Optimize",
    rule: { panel: "Optimize › trajectory", key: "← → on the tape" },
  },

  // ── Desk ────────────────────────────────────────────────────────────────
  {
    name: "Switch mode",
    group: "Desk",
    rule: { toolbar: "mode switcher", menu: "Window › Mode › Model, Sketch, Simulate", key: "M cycles", viewport: "selecting a sketch" },
  },
  {
    name: "Show or hide a window",
    group: "Desk",
    rule: { menu: "Window › Windows › scene.py, Objects, Properties, Sketch, Materials, Meshes, Studies, Results, Optimize, Scenes, Processes" },
  },
  {
    name: "Float a window",
    group: "Desk",
    rule: { menu: "Window › Float the focused window", panel: "tab › float" },
  },
  {
    name: "Reset layout",
    group: "Desk",
    rule: { menu: "Window › Reset layout" },
  },
  {
    name: "Running work",
    group: "Desk",
    rule: { toolbar: "job chip", panel: "Processes (parked)" },
  },

  // ── Help ────────────────────────────────────────────────────────────────
  {
    name: "Keyboard shortcuts",
    group: "Help",
    rule: { menu: "Help › Keyboard shortcuts", key: "?" },
  },
  {
    name: "Design language",
    group: "Help",
    rule: { menu: "Help › Design language" },
  },
  {
    name: "README",
    group: "Help",
    rule: { menu: "Help › README" },
  },
];

/** The menu bar, derived from the features' `rule.menu` homes. */
export interface MenuItem {
  label: string;
  key?: string;
  modes?: EditingMode[];
}
export interface MenuSection {
  title: string | null;
  items: MenuItem[];
  /** Drawn as a submenu (SUBMENUS) rather than an inline labelled section. */
  submenu: boolean;
}

/**
 * Seven menus. Every command has one menu home (N9), and the homes nest so the
 * bar spends no width on commands the rail, the keys and the panels already
 * reach: the rail's three groups are Model's submenus, Run and Stop are File's
 * program section, and the generated WGSL is a developer view under Help.
 */
export const MENU_ORDER = ["File", "Edit", "View", "Model", "Simulate", "Window", "Help"];

/** Sections drawn as a submenu that opens to the side, rather than inline. */
export const SUBMENUS = new Set([
  "Edit › Select",
  "View › Camera",
  "View › Zoom",
  "View › Show",
  "View › Inspect",
  "Model › Create",
  "Model › Shape",
  "Model › Constrain",
  "Simulate › New study",
  "Help › Developer",
]);

export function proposedMenus(): Map<string, MenuSection[]> {
  const menus = new Map<string, MenuSection[]>(MENU_ORDER.map((name) => [name, []]));
  for (const feature of FEATURES) {
    const path = feature.rule.menu;
    if (!path) continue;
    const parts = path.split(" › ");
    const menu = parts[0];
    const section = parts.length > 2 ? parts[1] : null;
    const leaf = parts[parts.length - 1];
    const sections = menus.get(menu);
    if (!sections) continue;
    let target = sections.find((s) => s.title === section);
    if (!target) {
      target = { title: section, items: [], submenu: section !== null && SUBMENUS.has(`${menu} › ${section}`) };
      sections.push(target);
    }
    // "Front, Right, Top…" and "Box, Sphere, Cylinder" are several items.
    const labels = leaf.endsWith("…") && !leaf.includes(",") ? [leaf] : leaf.split(/,\s*/);
    // "B S C" against "Box, Sphere, Cylinder" is one key per item.
    const keys = feature.rule.key?.split(" ") ?? [];
    const perItem = labels.length > 1 && keys.length === labels.length;
    labels.forEach((label, index) =>
      target!.items.some((item) => item.label === label) ||
      target!.items.push({
        label,
        key: perItem ? keys[index] : index === 0 && labels.length === 1 ? feature.rule.key : undefined,
        modes: feature.modes,
      }),
    );
  }
  return menus;
}
