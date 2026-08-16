/**
 * Pure logic for the FEM simulate mode.
 *
 * Everything the SimulatePanel and the renderer share that is not WebGPU:
 * per-group boundary-condition state, the viridis ramp (kept in sync with the
 * constants baked into simulation.wgsl), and the clip-plane math behind the
 * ParaView-style slicing slider. No DOM, no signals — unit-testable as-is.
 */

import type { SimulationBc, SimulationKind } from "./types";

/** What the user assigned to one face group. */
export type BcAssignment =
  | { type: "dirichlet"; value: number }
  | { type: "traction"; value: [number, number, number] };

/** Face-group id -> assignment; groups without an entry carry no condition. */
export type BcMap = Record<string, BcAssignment>;

/** Return a new map with `groupId` assigned, or removed when null. */
export function setAssignment(
  bcs: BcMap,
  groupId: string,
  assignment: BcAssignment | null,
): BcMap {
  const next = { ...bcs };
  if (assignment === null) delete next[groupId];
  else next[groupId] = assignment;
  return next;
}

/**
 * Boundary conditions for a solve request.
 *
 * A thermal solve only understands fixed temperatures, so traction
 * assignments are dropped rather than sent to fail server validation;
 * an elastic solve keeps both (dirichlet means "fully clamped" there).
 */
export function requestBcs(bcs: BcMap, kind: SimulationKind): SimulationBc[] {
  return Object.entries(bcs)
    .filter(([, assignment]) => kind === "elastic" || assignment.type === "dirichlet")
    .sort(([a], [b]) => (a < b ? -1 : 1))
    .map(([group, assignment]) => ({ group, type: assignment.type, value: assignment.value }));
}

/** True when `bcs` can drive a solve of `kind` (needs an anchor patch). */
export function canSolve(bcs: BcMap, kind: SimulationKind): boolean {
  return requestBcs(bcs, kind).some((bc) => bc.type === "dirichlet");
}

// Polynomial fit of matplotlib's viridis (degree 6 per channel) — the same
// coefficients live in simulation.wgsl; keep the two lists in sync.
const VIRIDIS: readonly (readonly [number, number, number])[] = [
  [0.2744554245, 0.0057679624, 0.3326638811],
  [0.1077083262, 1.3964696839, 1.3867705979],
  [-0.3272410968, 0.2148135645, 0.0919768808],
  [-4.5999315182, -5.7582381893, -19.2918089503],
  [6.2037359013, 14.1539649474, 56.6562995652],
  [4.7517868889, -13.7494394044, -65.3209678276],
  [-5.432077171, 4.641571316, 26.2721076045],
] as const;

/** Map a normalized scalar through the viridis ramp; clamps to [0, 1]. */
export function viridis(t: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, t));
  const color = [0, 0, 0];
  for (let channel = 0; channel < 3; channel++) {
    // Horner evaluation from the highest-order coefficient down.
    let value = VIRIDIS[VIRIDIS.length - 1][channel];
    for (let order = VIRIDIS.length - 2; order >= 0; order--) {
      value = value * x + VIRIDIS[order][channel];
    }
    color[channel] = Math.min(1, Math.max(0, value));
  }
  return color as [number, number, number];
}

/** CSS linear-gradient for the result legend, sampled from the same ramp. */
export function rampCss(stops = 12): string {
  const colors: string[] = [];
  for (let index = 0; index <= stops; index++) {
    const [red, green, blue] = viridis(index / stops).map((value) =>
      Math.round(value * 255),
    );
    colors.push(`rgb(${red}, ${green}, ${blue})`);
  }
  return `linear-gradient(to right, ${colors.join(", ")})`;
}

/** Compact legend label: fixed digits for ordinary values, exponent outside. */
export function formatScalar(value: number): string {
  if (!Number.isFinite(value)) return "–";
  const magnitude = Math.abs(value);
  if (magnitude !== 0 && (magnitude >= 10_000 || magnitude < 0.001)) {
    return value.toExponential(2);
  }
  return value.toFixed(3).replace(/\.?0+$/, "") || "0";
}

/** Slicing control state: a plane perpendicular to one axis. */
export interface SliceState {
  axis: 0 | 1 | 2;
  /** 0 hides everything, 1 shows the full mesh. */
  fraction: number;
  enabled: boolean;
}

export const DEFAULT_SLICE: SliceState = { axis: 0, fraction: 1, enabled: false };

/** Axis-aligned bounds of a flat xyz position array. */
export function meshBounds(positions: readonly number[]): {
  min: [number, number, number];
  max: [number, number, number];
} {
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let index = 0; index + 2 < positions.length; index += 3) {
    for (let axis = 0; axis < 3; axis++) {
      const value = positions[index + axis];
      if (value < min[axis]) min[axis] = value;
      if (value > max[axis]) max[axis] = value;
    }
  }
  if (!Number.isFinite(min[0])) return { min: [0, 0, 0], max: [0, 0, 0] };
  return { min, max };
}

/**
 * The clip plane a slice state describes over the given bounds.
 *
 * Fragments with `dot(world, normal) > offset` are discarded: at fraction 1
 * the plane sits just past the far side (nothing clipped), at 0 just before
 * the near side (everything clipped), linear in between.
 */
export function slicePlane(
  slice: SliceState,
  bounds: { min: readonly number[]; max: readonly number[] },
): { normal: [number, number, number]; offset: number } {
  const normal: [number, number, number] = [0, 0, 0];
  normal[slice.axis] = 1;
  const low = bounds.min[slice.axis];
  const high = bounds.max[slice.axis];
  // A hair of margin keeps faces lying exactly on the bounds visible at 1.
  const margin = Math.max(1e-6, (high - low) * 1e-4);
  const fraction = Math.min(1, Math.max(0, slice.fraction));
  const offset = low - margin + fraction * (high - low + 2 * margin);
  return { normal, offset };
}
