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
  | "optimize"
  | "simulate";

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
}

export const WINDOW_DEFS: readonly WindowDef[] = [
  { id: "viewport", title: "Viewport", modes: ["model", "sketch", "simulate"], permanent: true },
  { id: "editor", title: "scene.py", modes: ["model", "sketch", "simulate"] },
  { id: "objects", title: "Objects", modes: ["model", "sketch"] },
  { id: "materials", title: "Materials", modes: ["model"] },
  { id: "sketch", title: "Sketch", modes: ["sketch"] },
  { id: "optimize", title: "Optimize", modes: ["model"] },
  { id: "simulate", title: "Simulate", modes: ["simulate"] },
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
 * Simulate gives the FEM panel the whole right column, because its tabs
 * (meshes, studies, results) already fill it.
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
    { id: "simulate", reference: "viewport", direction: "right", size: COLUMN_WIDTH },
  ],
};

/** Fallback placement for a window opened into an already-arranged dock. */
export const FALLBACK_PLACEMENTS: Record<WindowId, OpenPlacement> = {
  viewport: {},
  editor: { reference: "viewport", direction: "left", size: 460 },
  objects: { reference: "viewport", direction: "right", size: 300 },
  materials: { reference: "objects", direction: "below" },
  sketch: { reference: "objects", direction: "below" },
  optimize: { reference: "objects", direction: "below" },
  simulate: { reference: "viewport", direction: "right", size: 340 },
};
