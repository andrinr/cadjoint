/**
 * Pure helpers for the Optimize panel.
 *
 * Optimizations are declared in the scene program (`Optimization(...)`); the
 * panel edits them through /patch source operations and runs them through
 * POST /api/optimize, whose response is adopted exactly like a patch. The
 * replay player re-renders intermediate parameter snapshots by substituting
 * the literals client-side and ghost-compiling — nothing here touches the
 * DOM or app state, so it is unit-testable as-is.
 */

import type { OptimizationPayload, OptimizeTrajectoryEntry } from "./types";

/** Body for POST /patch editing a numeric optimization argument. */
export function setOptimizationValueRequest(
  optimization: OptimizationPayload,
  argument: "steps" | "learning_rate",
  value: number,
): Record<string, unknown> {
  return { op: "set_optimization_value", optimization: optimization.index, argument, value };
}

/** Body for POST /patch deleting the declaration from the program. */
export function deleteOptimizationRequest(
  optimization: OptimizationPayload,
): Record<string, unknown> {
  return { op: "delete_optimization", optimization: optimization.index };
}

/** Body for POST /api/optimize (steps only when overriding the declared). */
export function optimizeRequest(
  source: string,
  name: string,
  steps?: number,
): { source: string; name: string; steps?: number } {
  const body: { source: string; name: string; steps?: number } = { source, name };
  if (steps !== undefined) body.steps = steps;
  return body;
}

/** Format a parameter value the way the source patcher writes literals. */
export function formatParameterValue(value: number | number[]): string {
  const num = (item: number) => {
    const rounded = Number(item.toFixed(6));
    return Object.is(rounded, -0) ? "0.0" : `${rounded}${Number.isInteger(rounded) ? ".0" : ""}`;
  };
  return Array.isArray(value) ? `[${value.map(num).join(", ")}]` : num(value);
}

/**
 * Write a set of named parameter values back into the program text.
 *
 * Free parameters are declared as `name = Scalar(1.2, ...)`,
 * `name = Vector2(value=[x, y], ...)` or `name = Vector([x, y, z], ...)`;
 * each assignment's first literal is replaced with the snapshot's value.
 * Purely lexical on purpose — the replay player needs many cheap rewrites,
 * and a name that does not match simply stays at its current literal.
 */
export function substituteParameters(
  source: string,
  parameters: Record<string, number | number[]>,
): string {
  let text = source;
  for (const [name, value] of Object.entries(parameters)) {
    const pattern = new RegExp(
      // name = Scalar( [value=] <number or [..]>
      String.raw`(\b${name}\s*=\s*(?:Scalar|Vector2|Vector)\(\s*(?:value\s*=\s*)?)` +
        String.raw`(\[[^\]]*\]|[-+]?[0-9][0-9_]*\.?[0-9]*(?:[eE][-+]?[0-9]+)?)`,
    );
    text = text.replace(pattern, `$1${formatParameterValue(value)}`);
  }
  return text;
}

/**
 * Indices into a trajectory for playback, thinned to at most `cap` frames.
 *
 * The first and last frames are always kept, so the replay starts at the
 * initial design and ends on the optimized one.
 */
export function playbackFrames(length: number, cap = 16): number[] {
  if (length <= 0) return [];
  if (length === 1) return [0];
  const count = Math.min(length, Math.max(2, cap));
  const frames: number[] = [];
  for (let index = 0; index < count; index++) {
    frames.push(Math.round((index * (length - 1)) / (count - 1)));
  }
  // Rounding can collide neighbours for tiny trajectories; dedupe keeps order.
  return frames.filter((frame, index) => index === 0 || frame !== frames[index - 1]);
}

/** Replay player state: which frame shows, and whether it advances. */
export interface PlayerState {
  frame: number;
  playing: boolean;
}

/** Advance one frame; playback stops (rather than loops) at the end. */
export function advancePlayer(state: PlayerState, frameCount: number): PlayerState {
  if (!state.playing || frameCount === 0) return { ...state, playing: false };
  const next = state.frame + 1;
  if (next >= frameCount) return { frame: frameCount - 1, playing: false };
  return { frame: next, playing: true };
}

/** Start playback, rewinding when the cursor already sits at the end. */
export function startPlayer(state: PlayerState, frameCount: number): PlayerState {
  if (frameCount === 0) return { frame: 0, playing: false };
  const atEnd = state.frame >= frameCount - 1;
  return { frame: atEnd ? 0 : state.frame, playing: true };
}

/**
 * SVG polyline points for an objective-history sparkline.
 *
 * Y is flipped (SVG grows downward) and padded a hair so the extremes stay
 * inside the stroke. A flat history draws a centered horizontal line.
 */
export function sparklinePoints(
  values: readonly number[],
  width: number,
  height: number,
): string {
  if (values.length === 0) return "";
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min;
  const pad = 1.5;
  const innerHeight = height - 2 * pad;
  const step = values.length > 1 ? width / (values.length - 1) : 0;
  return values
    .map((value, index) => {
      const t = span > 0 ? (value - min) / span : 0.5;
      const x = values.length > 1 ? index * step : width / 2;
      const y = pad + (1 - t) * innerHeight;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

/** X position of the moving cursor over the sparkline for a given step. */
export function sparklineCursorX(
  step: number,
  count: number,
  width: number,
): number {
  if (count <= 1) return width / 2;
  return (Math.min(step, count - 1) / (count - 1)) * width;
}

/** Objective value of a trajectory frame, for the player readout. */
export function frameObjective(
  trajectory: readonly OptimizeTrajectoryEntry[],
  frame: number,
): number | null {
  const entry = trajectory[frame];
  return entry ? entry.objective : null;
}
