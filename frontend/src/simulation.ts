/**
 * Pure logic for the FEM simulate mode.
 *
 * Everything the SimulatePanel and the renderer share that is not WebGPU:
 * the viridis ramp (kept in sync with the constants baked into
 * simulation.wgsl) and the clip-plane math behind the ParaView-style slicing
 * slider. Boundary-condition state lives in the scene program itself — the
 * panel edits it through /patch, so there is no client-side BC model here.
 * No DOM, no signals — unit-testable as-is.
 */

// Every color role (field ramp, quality ramp, overlays, edges) lives in
// simColors.ts; these re-exports keep the ramp's historical home working.
import { fieldRamp, fieldRampCss } from "./simColors";

/** Map a normalized scalar through the FIELD (viridis) ramp. */
export const viridis = fieldRamp;

/** CSS linear-gradient for the field legend, sampled from the same ramp. */
export const rampCss = fieldRampCss;

/** Compact legend label: fixed digits for ordinary values, exponent outside. */
export function formatScalar(value: number): string {
  if (!Number.isFinite(value)) return "–";
  const magnitude = Math.abs(value);
  if (magnitude !== 0 && (magnitude >= 10_000 || magnitude < 0.001)) {
    return value.toExponential(2);
  }
  return value.toFixed(3).replace(/\.?0+$/, "") || "0";
}

/** What the results browser should feed the heatmap right now. */
export interface ResultViewInput {
  /** The field the solve response's mesh scalars carry. */
  defaultField: string;
  /** Field chosen in the picker, or null before any switch. */
  activeField: string | null;
  /** Whether the mesh-quality view overrides the field view. */
  qualityView: boolean;
  fields: Record<string, number[]>;
  ranges: Record<string, [number, number]>;
  payloadScalars: readonly number[];
  payloadRange: [number, number];
  /** Per-vertex quality, fetched lazily; null until available. */
  qualityScalars: readonly number[] | null;
  qualityRange: [number, number] | null;
}

export interface ResultView {
  scalars: readonly number[];
  range: [number, number];
  label: string;
}

/**
 * Resolve the field picker + quality toggle into concrete display scalars.
 *
 * The quality view only takes over once its per-vertex scalars exist (they
 * are fetched on demand); otherwise the active field falls back through the
 * per-vertex field table to the payload's own scalars.
 */
export function resolveResultView(input: ResultViewInput): ResultView {
  if (input.qualityView && input.qualityScalars) {
    const range: [number, number] =
      input.qualityRange ??
      [Math.min(...input.qualityScalars), Math.max(...input.qualityScalars)];
    return { scalars: input.qualityScalars, range, label: "scaled jacobian" };
  }
  const field = input.activeField ?? input.defaultField;
  return {
    scalars: input.fields[field] ?? input.payloadScalars,
    range: input.ranges[field] ?? input.payloadRange,
    label: field.replaceAll("_", " "),
  };
}

/**
 * Default warp factor for the deformed view: 10% of the mesh diagonal at the
 * largest displacement, the usual "make it visible" scaling for FEM plots.
 */
export function autoDeformScale(
  bounds: { min: readonly number[]; max: readonly number[] },
  displacements: readonly (readonly [number, number, number])[],
): number {
  let peak = 0;
  for (const [dx, dy, dz] of displacements) {
    const magnitude = Math.hypot(dx, dy, dz);
    if (magnitude > peak) peak = magnitude;
  }
  if (peak <= 0) return 1;
  const diagonal = Math.hypot(
    bounds.max[0] - bounds.min[0],
    bounds.max[1] - bounds.min[1],
    bounds.max[2] - bounds.min[2],
  );
  return (0.1 * (diagonal || 1)) / peak;
}

/** Vertex positions offset by scaled displacements (flat xyz layout). */
export function applyDisplacements(
  positions: readonly number[],
  displacements: readonly (readonly [number, number, number])[],
  scale: number,
): number[] {
  const warped = positions.slice() as number[];
  const count = Math.min(displacements.length, Math.floor(positions.length / 3));
  for (let index = 0; index < count; index++) {
    warped[index * 3] += displacements[index][0] * scale;
    warped[index * 3 + 1] += displacements[index][1] * scale;
    warped[index * 3 + 2] += displacements[index][2] * scale;
  }
  return warped;
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
