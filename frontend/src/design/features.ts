/**
 * Every feature of the playground, where it is reachable today, and where
 * the guide puts it.
 *
 * One record per feature. `now` is read off the shipped app (MenuBar.tsx,
 * Toolbar.tsx, ToolRail.tsx, shortcuts.ts and the windows); `rule` is the
 * proposal. The design page renders both the index and the proposed menu
 * bar from this list, so the menus it shows are exactly the commands here.
 *
 * The menu structure `rule.menu` describes was agreed with Andrin on
 * 2026-10-16 as proposed: ten menus, Create / Shape / Constrain kept apart to
 * match the rail, Program as its own menu, modes under Window, mode-only
 * commands shown disabled with the reason.
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
  now: Homes;
  rule: Homes;
  /** What is wrong with `now`, when something is. */
  issue?: string;
  /** How the issue was settled, once it has been. */
  decision?: string;
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
    now: { menu: "File › New", toolbar: "Reset (icon only, beside Run)" },
    rule: { menu: "File › New" },
    issue: "Two homes, two names (New, Reset); the toolbar copy is a destructive icon-only button next to Run.",
    decision: "Remove the toolbar Reset; File › New is the only home.",
  },
  {
    name: "Open a scene",
    group: "File",
    now: { menu: "File › Open…", panel: "Scenes (parked)" },
    rule: { menu: "File › Open…", key: "⌘O", panel: "Scenes (parked)" },
  },
  {
    name: "Save",
    group: "File",
    now: { menu: "File › Save" },
    rule: { menu: "File › Save", key: "⌘S" },
    issue: "No shortcut for the most frequent file command.",
    decision: "⌘S, with preventDefault so the browser does not save the page.",
  },
  {
    name: "Save as",
    group: "File",
    now: { menu: "File › Save As…" },
    rule: { menu: "File › Save As…", key: "⇧⌘S" },
  },
  {
    name: "Download scene.py",
    group: "File",
    now: { menu: "File › Download scene.py" },
    rule: { menu: "File › Download scene.py" },
  },
  {
    name: "Export mesh or STEP",
    group: "File",
    now: { menu: "File › Export…" },
    rule: { menu: "File › Export…", key: "⌘E" },
  },
  {
    name: "Refresh the scene list",
    group: "File",
    now: { panel: "Scenes › Refresh" },
    rule: { panel: "Scenes › Refresh" },
  },

  // ── Program ─────────────────────────────────────────────────────────────
  {
    name: "Run",
    group: "Program",
    now: { toolbar: "Run", key: "⌘↵" },
    rule: { menu: "Program › Run", toolbar: "Run", key: "⌘↵" },
    issue: "No menu home.",
    decision: "Program › Run (menu structure, settled).",
  },
  {
    name: "Stop the compile",
    group: "Program",
    now: { toolbar: "job chip ×", panel: "Processes" },
    rule: { menu: "Program › Stop", toolbar: "job chip ×", key: "⌘." , panel: "Processes" },
  },
  {
    name: "Show generated WGSL",
    group: "Program",
    now: { toolbar: "<> (icon only)" },
    rule: { menu: "Program › Show generated WGSL" },
    issue: "A developer view holds a primary toolbar slot beside Run.",
    decision: "Moves to Program › Show generated WGSL; the toolbar slot goes.",
  },
  {
    name: "Undo",
    group: "Program",
    now: { menu: "Edit › Undo", key: "⌘Z" },
    rule: { menu: "Edit › Undo", key: "⌘Z" },
  },
  {
    name: "Redo",
    group: "Program",
    now: { menu: "Edit › Redo", key: "⇧⌘Z" },
    rule: { menu: "Edit › Redo", key: "⇧⌘Z" },
  },

  // ── Edit and selection ──────────────────────────────────────────────────
  {
    name: "Select whole objects",
    group: "Edit and selection",
    modes: MODEL_SKETCH,
    now: { rail: "Select › Object", key: "O" },
    rule: { menu: "Edit › Select › Objects", rail: "Select › Object", key: "O" },
  },
  {
    name: "Select sketch vertices",
    group: "Edit and selection",
    modes: MODEL_SKETCH,
    now: { rail: "Select › Vertex", key: "V" },
    rule: { menu: "Edit › Select › Vertices", rail: "Select › Vertex", key: "V" },
  },
  {
    name: "Delete the selection",
    group: "Edit and selection",
    modes: MODEL_SKETCH,
    now: { rail: "Delete (icon only)", key: "⌫" },
    rule: { menu: "Edit › Delete", rail: "Delete (worded ≥1280px)", key: "⌫" },
    issue: "The rail's delete is icon-only (I2).",
    decision: "The rail's Delete carries the word at 1280px and wider, like every rail icon (I1).",
  },
  {
    name: "Clear the selection",
    group: "Edit and selection",
    now: { key: "Esc", viewport: "click empty space" },
    rule: { menu: "Edit › Select › None", key: "Esc", viewport: "click empty space" },
  },
  {
    name: "Edit an argument",
    group: "Edit and selection",
    now: { panel: "Properties", viewport: "drag a handle" },
    rule: { panel: "Properties", viewport: "drag a handle" },
  },
  {
    name: "Follow the selection again",
    group: "Edit and selection",
    now: { panel: "Properties › back" },
    rule: { panel: "Properties › back" },
  },

  // ── View ────────────────────────────────────────────────────────────────
  {
    name: "Standard views",
    group: "View",
    now: { key: "1 3 7", viewport: "view cube faces, ⌃ for the opposite" },
    rule: { menu: "View › Camera › Front, Right, Top", key: "1 3 7", viewport: "view cube faces, ⌃ for the opposite" },
    issue: "No menu home: there is no View menu.",
    decision: "A View menu (menu structure, settled).",
  },
  {
    name: "Turn to the opposite side",
    group: "View",
    now: { key: "9" },
    rule: { menu: "View › Camera › Opposite side", key: "9" },
  },
  {
    name: "Orthographic / perspective",
    group: "View",
    now: { key: "5", viewport: "view cube toggle" },
    rule: { menu: "View › Camera › Orthographic", key: "5", viewport: "view cube toggle" },
  },
  {
    name: "Orbit, pan, zoom",
    group: "View",
    now: { viewport: "drag, Shift/Space + drag, wheel" },
    rule: { viewport: "drag, Shift/Space + drag, wheel" },
  },
  {
    name: "Grid spacing",
    group: "View",
    now: { viewport: "Alt + wheel" },
    rule: { menu: "View › Grid › Finer, Coarser", key: "[ ]", viewport: "Alt + wheel" },
    issue: "Reachable only by a gesture nobody can discover.",
    decision: "View › Grid › Finer, Coarser, on [ and ].",
  },
  {
    name: "Render preset",
    group: "View",
    now: { toolbar: "Render settings › presets" },
    rule: { menu: "View › Render › Preset…", toolbar: "Render settings › presets" },
  },
  {
    name: "Construction overlay",
    group: "View",
    now: { toolbar: "Render settings › Annotations" },
    rule: { menu: "View › Show › Construction", toolbar: "Render settings › Annotations", key: "⇧C" },
  },
  {
    name: "Feature edges, mesh wireframe",
    group: "View",
    now: { toolbar: "Render settings › Annotations" },
    rule: { menu: "View › Show › Feature edges, Wireframe", toolbar: "Render settings › Annotations" },
  },
  {
    name: "Hide the solid",
    group: "View",
    now: { toolbar: "Render settings › Annotations" },
    rule: { menu: "View › Show › Solid", toolbar: "Render settings › Annotations" },
  },
  {
    name: "Distance-field slice",
    group: "View",
    now: { toolbar: "Render settings › Distance field" },
    rule: { menu: "View › Inspect › Distance slice", toolbar: "Render settings › Distance field" },
  },
  {
    name: "Normals, depth",
    group: "View",
    now: { toolbar: "Render settings › Diagnostics" },
    rule: { menu: "View › Inspect › Normals, Depth", toolbar: "Render settings › Diagnostics" },
  },
  {
    name: "Shading, shadows, reflections, quality",
    group: "View",
    now: { toolbar: "Render settings › Shading, Effects, Marching" },
    rule: { toolbar: "Render settings › Shading, Effects, Marching", menu: "View › Render › Settings…" },
  },

  // ── Create ──────────────────────────────────────────────────────────────
  {
    name: "Sketch",
    group: "Create",
    modes: MODEL_SKETCH,
    now: { rail: "Create › Sketch" },
    rule: { menu: "Create › Sketch", rail: "Create › Sketch", key: "K" },
  },
  {
    name: "Sketch on a face",
    group: "Create",
    modes: MODEL_SKETCH,
    now: { rail: "Create › Sketch on face" },
    rule: { menu: "Create › Sketch on face", rail: "Create › Sketch on face" },
  },
  {
    name: "Point",
    group: "Create",
    modes: MODEL_SKETCH,
    now: { rail: "Create › Point", key: "P" },
    rule: { menu: "Create › Point", rail: "Create › Point", key: "P" },
  },
  {
    name: "Box, sphere, cylinder",
    group: "Create",
    modes: ["model"],
    now: { rail: "Create › Box, Sphere, Cylinder", key: "B S C" },
    rule: { menu: "Create › Box, Sphere, Cylinder", rail: "Create › Box, Sphere, Cylinder", key: "B S C" },
  },

  // ── Shape ───────────────────────────────────────────────────────────────
  {
    name: "Extrude, revolve, loft",
    group: "Shape",
    modes: ["sketch"],
    now: { rail: "Modify › Extrude, Revolve, Loft" },
    rule: { menu: "Shape › Extrude, Revolve, Loft", rail: "Shape › Extrude, Revolve, Loft", key: "E" },
    issue: "The rail calls the group Modify; what it does is turn a sketch into a solid.",
    decision: "Rename the rail group Shape.",
  },
  {
    name: "Move, rotate, scale",
    group: "Shape",
    modes: MODEL_SKETCH,
    now: { rail: "Transform › Move, Rotate, Scale", key: "G R" },
    rule: { menu: "Shape › Move, Rotate, Scale", rail: "Transform › Move, Rotate, Scale", key: "G R T" },
    issue: "Scale has no key.",
    decision: "Scale on T.",
  },

  // ── Constrain ───────────────────────────────────────────────────────────
  {
    name: "Distance, horizontal, vertical, coincident",
    group: "Constrain",
    modes: ["sketch"],
    now: { rail: "Annotate › Distance, Horizontal, Vertical, Coincident", panel: "Sketch › Distance, Horiz, Vert, Coinc" },
    rule: { menu: "Constrain › Distance, Horizontal, Vertical, Coincident", rail: "Constrain › Distance, Horizontal, Vertical, Coincident", panel: "Sketch (list only)" },
    issue: "Two homes with two sets of names (Coincident / Coinc); the rail group is called Annotate.",
    decision: "The rail (and Constrain menu) adds, the Sketch panel lists and removes; the rail group is renamed Constrain.",
  },
  {
    name: "Parallel, perpendicular",
    group: "Constrain",
    modes: ["sketch"],
    now: { rail: "Annotate › Parallel, Perpendicular", panel: "Sketch › ∥ Edges, ⊥ Edges" },
    rule: { menu: "Constrain › Parallel, Perpendicular", rail: "Constrain › Parallel, Perpendicular", panel: "Sketch (list only)" },
    issue: "Glyph labels in the panel, words on the rail.",
    decision: "Words everywhere; the panel no longer adds.",
  },
  {
    name: "Fix a point",
    group: "Constrain",
    modes: ["sketch"],
    now: { panel: "Sketch › anchor" },
    rule: { menu: "Constrain › Fix", rail: "Constrain › Fix", panel: "Sketch › anchor" },
    issue: "Only in the panel, unlike every other constraint.",
    decision: "Fix joins the rail's Constrain group. No letter key: F stays free for framing the selection.",
  },
  {
    name: "Remove a constraint",
    group: "Constrain",
    modes: ["sketch"],
    now: { panel: "Sketch › chip ×" },
    rule: { panel: "Sketch › chip ×", key: "⌫ on a selected chip" },
  },
  {
    name: "Constraint solver settings",
    group: "Constrain",
    modes: ["sketch"],
    now: { panel: "Sketch › solver" },
    rule: { panel: "Sketch › solver" },
  },

  // ── Materials ───────────────────────────────────────────────────────────
  {
    name: "New material",
    group: "Materials",
    modes: ["model"],
    now: { panel: "Materials › +" },
    rule: { menu: "Create › Material", panel: "Materials › +" },
  },
  {
    name: "Assign a material",
    group: "Materials",
    now: { panel: "Materials (drag a swatch onto an object)" },
    rule: { panel: "Properties › material · Materials (drag)", viewport: "drop" },
    issue: "Drag is the only way: no keyboard or selection-based path.",
    decision: "Properties gains a material field that rewrites the source; drag stays as the shortcut.",
  },
  {
    name: "Edit a material",
    group: "Materials",
    now: { panel: "Materials › inspector" },
    rule: { panel: "Materials › inspector" },
  },

  // ── Simulate ────────────────────────────────────────────────────────────
  {
    name: "New mesh",
    group: "Simulate",
    modes: ["simulate"],
    now: { panel: "Meshes › +" },
    rule: { menu: "Simulate › New mesh", panel: "Meshes › +" },
  },
  {
    name: "Build a mesh",
    group: "Simulate",
    modes: ["simulate"],
    now: { panel: "Meshes › card" },
    rule: { panel: "Meshes › card" },
  },
  {
    name: "New study",
    group: "Simulate",
    modes: ["simulate"],
    now: { panel: "Studies › + Thermal, + Elastic, + Flow" },
    rule: { menu: "Simulate › New study › Thermal, Elastic, Flow", panel: "Studies › + Study (kind menu)" },
    issue: "Three add buttons side by side truncate in a 320px column.",
    decision: "One dashed + Study button opening the three kinds, as in Simulate › New study.",
  },
  {
    name: "Add a boundary condition",
    group: "Simulate",
    modes: ["simulate"],
    now: { panel: "Studies › card › + Boundary condition", viewport: "pick faces" },
    rule: { panel: "Studies › card › + Boundary condition", viewport: "pick faces" },
  },
  {
    name: "Solve",
    group: "Simulate",
    modes: ["simulate"],
    now: { panel: "Studies › card › Solve" },
    rule: { menu: "Simulate › Solve", panel: "Studies › card › Solve", key: "⇧⌘↵" },
  },
  {
    name: "Stop a solve",
    group: "Simulate",
    modes: ["simulate"],
    now: { panel: "Studies › card › Stop", toolbar: "job chip ×" },
    rule: { menu: "Program › Stop", panel: "Studies › card › Stop", toolbar: "job chip ×", key: "⌘." },
  },
  {
    name: "Probe the field",
    group: "Simulate",
    modes: ["simulate"],
    now: { viewport: "hover the field", panel: "Results › probe" },
    rule: { viewport: "hover the field", panel: "Results › probe" },
  },
  {
    name: "Deformation scale",
    group: "Simulate",
    modes: ["simulate"],
    now: { panel: "Results › warp" },
    rule: { panel: "Results › warp" },
  },

  // ── Optimize ────────────────────────────────────────────────────────────
  {
    name: "Run an optimization",
    group: "Optimize",
    now: { panel: "Optimize › card › Run" },
    rule: { menu: "Simulate › Optimize", panel: "Optimize › card › Run" },
  },
  {
    name: "Replay a trajectory",
    group: "Optimize",
    now: { panel: "Optimize › trajectory" },
    rule: { panel: "Optimize › trajectory", key: "← → on the tape" },
  },

  // ── Desk ────────────────────────────────────────────────────────────────
  {
    name: "Switch mode",
    group: "Desk",
    now: { toolbar: "mode switcher", menu: "Window › Model mode…", key: "M / ⇧M", viewport: "selecting a sketch" },
    rule: { toolbar: "mode switcher", menu: "Window › Mode › Model, Sketch, Simulate", key: "M cycles", viewport: "selecting a sketch" },
  },
  {
    name: "Show or hide a window",
    group: "Desk",
    now: { menu: "Window › Editor, Object tree, Sketch panel…" },
    rule: { menu: "Window › Windows › scene.py, Objects, Properties, Sketch, Materials, Meshes, Studies, Results, Optimize, Scenes, Processes" },
    issue: "The menu names differ from the tab names (N4).",
    decision: "The menu uses the tab names (menu structure, settled).",
  },
  {
    name: "Float a window",
    group: "Desk",
    now: { menu: "Window › Float …", panel: "tab › float" },
    rule: { menu: "Window › Float the focused window", panel: "tab › float" },
  },
  {
    name: "Reset layout",
    group: "Desk",
    now: { menu: "Window › Reset layout" },
    rule: { menu: "Window › Reset layout" },
  },
  {
    name: "Running work",
    group: "Desk",
    now: { toolbar: "job chip", panel: "Processes (parked)" },
    rule: { toolbar: "job chip", panel: "Processes (parked)" },
  },

  // ── Help ────────────────────────────────────────────────────────────────
  {
    name: "Keyboard shortcuts",
    group: "Help",
    now: { menu: "Help › Keyboard shortcuts" },
    rule: { menu: "Help › Keyboard shortcuts", key: "?" },
  },
  {
    name: "Design language",
    group: "Help",
    now: {},
    rule: { menu: "Help › Design language" },
  },
  {
    name: "README",
    group: "Help",
    now: { menu: "Help › CADJOINT README" },
    rule: { menu: "Help › README" },
  },
];

/** The proposed menu bar, derived from the features' `rule.menu` homes. */
export interface MenuItem {
  label: string;
  key?: string;
  modes?: EditingMode[];
}
export interface MenuSection {
  title: string | null;
  items: MenuItem[];
}

export const MENU_ORDER = ["File", "Edit", "View", "Create", "Shape", "Constrain", "Simulate", "Program", "Window", "Help"];

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
      target = { title: section, items: [] };
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
