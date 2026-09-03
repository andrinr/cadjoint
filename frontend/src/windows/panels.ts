/**
 * What windows exist, and which editing mode each belongs to.
 *
 * The dock is free-form: any window can be moved, tabbed, floated or closed.
 * What a mode still decides is the *default arrangement* — which windows are
 * open and where they sit when you first enter that mode, and which layout is
 * restored when you come back to it. Modes never force visibility: once you
 * are in a mode you can open or close whatever you like, and that becomes the
 * layout that mode remembers.
 *
 * This module is deliberately data only, so the layout rules can be tested
 * without a DOM.
 */

import type { EditingMode } from "../editingMode";

export type WindowId =
  | "viewport"
  | "editor"
  | "objects"
  | "materials"
  | "sketch"
  | "meshes"
  | "studies"
  | "optimize"
  | "results"
  | "scenes"
  | "processes";

export interface WindowDef {
  id: WindowId;
  /** Tab label. Mono and tracked in the chrome, so keep it short. */
  title: string;
  /**
   * Modes whose default layout includes this window. A window can still be
   * opened from the Window menu in any mode; this only seeds the defaults.
   */
  modes: readonly EditingMode[];
  /**
   * The viewport is furniture, not a document: it cannot be closed or
   * minimised, because an empty dock with no viewport is not a state any
   * CAD user asked for.
   */
  permanent?: boolean;
  /**
   * Present in every desk, but parked in the tray rather than docked.
   *
   * The Processes window is the only one of these: it is worth having one
   * click away in every mode — a solve you started in Simulate is still
   * running while you edit in Model — but it is a monitor, and a monitor
   * that takes a column of the desk by default is a monitor in the way.
   * Parked is the honest default: the tray names it, so it is discoverable
   * without being imposed.
   */
  parked?: boolean;
  /**
   * A second test hook on the tab's own label.
   *
   * The four simulation windows used to be tabs of one Simulate panel, and
   * the audit tool and the end-to-end suite reach for them by the names that
   * tab strip had. They are ordinary windows now, so the dock's tab strip
   * *is* that strip — and the label carries the old id so the control the
   * tests click is still the control that raises the window.
   */
  tabTestId?: string;
}

export const WINDOW_DEFS: readonly WindowDef[] = [
  { id: "viewport", title: "Viewport", modes: ["model", "sketch", "simulate"], permanent: true },
  { id: "editor", title: "scene.py", modes: ["model", "sketch", "simulate"] },
  { id: "objects", title: "Objects", modes: ["model", "sketch"] },
  { id: "materials", title: "Materials", modes: ["model"] },
  { id: "sketch", title: "Sketch", modes: ["sketch"] },
  { id: "meshes", title: "Meshes", modes: ["simulate"], tabTestId: "sim-tab-meshes" },
  { id: "studies", title: "Studies", modes: ["simulate"], tabTestId: "sim-tab-studies" },
  { id: "optimize", title: "Optimize", modes: ["model", "simulate"], tabTestId: "sim-tab-optimize" },
  { id: "results", title: "Results", modes: ["simulate"], tabTestId: "sim-tab-results" },
  { id: "scenes", title: "Scenes", modes: [], parked: true },
  { id: "processes", title: "Processes", modes: [], parked: true },
];

const BY_ID = new Map<string, WindowDef>(WINDOW_DEFS.map((def) => [def.id, def]));

export const WINDOW_IDS: readonly WindowId[] = WINDOW_DEFS.map((def) => def.id);

export function windowDef(id: string): WindowDef | undefined {
  return BY_ID.get(id);
}

export function isWindowId(id: string): id is WindowId {
  return BY_ID.has(id);
}

export function windowTitle(id: string): string {
  return BY_ID.get(id)?.title ?? id;
}

export function isPermanent(id: string): boolean {
  return BY_ID.get(id)?.permanent === true;
}

/** The tab label's compatibility test hook, when the window has one. */
export function tabTestId(id: string): string | undefined {
  return BY_ID.get(id)?.tabTestId;
}

/** Whether a window's default state is parked in the tray, not docked. */
export function isParked(id: string): boolean {
  return BY_ID.get(id)?.parked === true;
}

/** The windows a mode's default layout opens, in dock order. */
export function defaultWindowsForMode(mode: EditingMode): WindowId[] {
  return DEFAULT_LAYOUTS[mode].map((step) => step.id);
}

/** Where a window lands relative to one already in the dock. */
export interface OpenPlacement {
  /** Existing window to attach to; absent means "wherever there is room". */
  reference?: WindowId;
  /** `within` tabs the new window into the reference's group. */
  direction?: "left" | "right" | "above" | "below" | "within";
  /**
   * Extent along the split axis, in CSS pixels, applied once the whole
   * default desk is built.
   */
  size?: number;
  /** Add the window without pulling focus onto it (tabbed behind). */
  inactive?: boolean;
}

export interface LayoutStep extends OpenPlacement {
  id: WindowId;
}

/**
 * The two column widths every default desk is built from.
 *
 * They are the same in all three modes on purpose. Selecting a sketch in the
 * viewport auto-enters Sketch mode, so a mode change can happen *under the
 * pointer* — and a desk that gave the viewport a different rectangle in each
 * mode would move the model out from under the click that caused it. Identical
 * columns mean the default mode switch changes what is in the side windows and
 * nothing else. Once a user resizes anything, their arrangement is what that
 * mode remembers, which is the point of the whole system.
 */
const EDITOR_WIDTH = 460;
const COLUMN_WIDTH = 320;

/**
 * Each mode's default arrangement, built by opening these in order.
 *
 * Model puts the code on the left, the viewport in the middle, and the
 * property windows in a right-hand column where Materials and Optimize share
 * a tab strip. Sketch swaps Materials for the sketch's own properties.
 * Simulate gives the right column to the
 * four simulation windows: Studies over Results, with Meshes tabbed behind
 * the first and Optimize behind the second — setup above, outcomes below.
 */
export const DEFAULT_LAYOUTS: Record<EditingMode, readonly LayoutStep[]> = {
  model: [
    { id: "viewport" },
    { id: "editor", reference: "viewport", direction: "left", size: EDITOR_WIDTH },
    { id: "objects", reference: "viewport", direction: "right", size: COLUMN_WIDTH },
    { id: "materials", reference: "objects", direction: "below" },
    { id: "optimize", reference: "materials", direction: "within", inactive: true },
  ],
  sketch: [
    { id: "viewport" },
    { id: "editor", reference: "viewport", direction: "left", size: EDITOR_WIDTH },
    { id: "objects", reference: "viewport", direction: "right", size: COLUMN_WIDTH },
    { id: "sketch", reference: "objects", direction: "below" },
  ],
  simulate: [
    { id: "viewport" },
    { id: "editor", reference: "viewport", direction: "left", size: EDITOR_WIDTH },
    { id: "studies", reference: "viewport", direction: "right", size: COLUMN_WIDTH },
    { id: "meshes", reference: "studies", direction: "within", inactive: true },
    // Setup is taller than outcomes on purpose: a study card carries its
    // whole boundary-condition list and its Solve button, and a desk that
    // split the column evenly parked Solve just under the seam.
    { id: "results", reference: "studies", direction: "below", size: 300 },
    { id: "optimize", reference: "results", direction: "within", inactive: true },
  ],
};

/** Fallback placement for a window opened into an already-arranged dock. */
export const FALLBACK_PLACEMENTS: Record<WindowId, OpenPlacement> = {
  viewport: {},
  editor: { reference: "viewport", direction: "left", size: 460 },
  objects: { reference: "viewport", direction: "right", size: 300 },
  materials: { reference: "objects", direction: "below" },
  sketch: { reference: "objects", direction: "below" },
  meshes: { reference: "viewport", direction: "right", size: 340 },
  studies: { reference: "viewport", direction: "right", size: 340 },
  optimize: { reference: "objects", direction: "below" },
  results: { reference: "viewport", direction: "right", size: 340 },
  // A browser of documents opens *where the document is*: tabbed into the
  // editor's group, not as a fourth column. A column would have to come out
  // of somebody's width, and at 1280 the one it took it from could no longer
  // show its own tab controls — for a window you open to pick a file and
  // then close again.
  scenes: { reference: "editor", direction: "within" },
  // A monitor reads as an instrument strip under the work, not as another
  // property column beside it.
  processes: { reference: "viewport", direction: "below", size: 260 },
};
