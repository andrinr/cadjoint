/**
 * The viewport graticule: what one division is worth, and where the detents are.
 *
 * The graticule is a *faceplate*, not scenery. It is a fixed screen-space rule
 * — eight square divisions down the viewport height — and the geometry moves
 * behind it, exactly as a trace moves behind a CRT's etched graticule
 * (`research/design-language.md` §16.3). That choice is what makes it honest
 * for free: there is no second coordinate system to keep registered with the
 * camera under orbit or pan. The whole truth claim lives in one number, the
 * *gain* — how much world a division is worth — and this file is that number.
 *
 * Under an orthographic camera the gain is exact everywhere in the frame,
 * because world-per-pixel is constant and isotropic. Under a perspective
 * camera it is exact only on the plane through the orbit target, so the
 * readout is marked uncalibrated with the `>` prefix a Tektronix 2465 uses for
 * an uncalibrated scale factor.
 *
 * Everything here is pure arithmetic over a camera — no DOM, no GPU — so the
 * ladder and the scale derivation are unit tested in `test/graticule.test.ts`.
 */

import { FOV_SCALE, orthoHeightFor, type Projection } from "./math";
import { VIEW_PRESETS } from "./display";

/**
 * Divisions down the viewport height.
 *
 * Tektronix's vertical count (475A: "8 × 10 cm display"). The horizontal count
 * is whatever fits at the same *square* division, rather than a forced ten:
 * the app's viewport is not a 5:4 CRT, and stretching divisions to 10 × 8 on a
 * 16:10 pane would make H and V gains differ by 28% — graph paper that is not
 * square is not graph paper.
 */
export const DIVISIONS = 8;

/**
 * Millimetres per world unit.
 *
 * The repository declares its length unit in exactly one place: the STEP
 * writer (`cadjoint/meshing/export.py`) stamps `SI_UNIT($,.METRE.)` and states
 * "one mesh unit = 1 m". The graticule reads the scene through that
 * declaration rather than inventing a scale of its own — a grid whose units
 * are made up is decoration.
 */
export const MM_PER_UNIT = 1000;

/** The 1-2-5 ladder's mantissas, per §16.3's "20 / 10 / 5 / 2 mm/div". */
const MANTISSAS = [1, 2, 5] as const;

/** Every 1-2-5 rung in the decades around `mm`, ascending. */
function rungsAround(mm: number, spread = 2): number[] {
  const exponent = Math.floor(Math.log10(mm));
  const rungs: number[] = [];
  for (let decade = exponent - spread; decade <= exponent + spread; decade++) {
    for (const mantissa of MANTISSAS) rungs.push(mantissa * 10 ** decade);
  }
  return rungs.sort((a, b) => a - b);
}

/** The 1-2-5 rung closest to `mm`, measured on a log scale. */
export function nearestDetent(mm: number): number {
  if (!Number.isFinite(mm) || mm <= 0) return Number.NaN;
  let best = Number.NaN;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const rung of rungsAround(mm, 1)) {
    const distance = Math.abs(Math.log(mm / rung));
    if (distance < bestDistance) {
      bestDistance = distance;
      best = rung;
    }
  }
  return best;
}

/**
 * The next rung above (`+1`) or below (`-1`) `mm`.
 *
 * A gain already sitting on a rung steps a full rung, so repeated detented
 * zoom walks 20 → 10 → 5 → 2 rather than stalling on the rung it is on.
 */
export function stepDetent(mm: number, direction: 1 | -1): number {
  if (!Number.isFinite(mm) || mm <= 0) return Number.NaN;
  // Relative, because the rungs span decades: an absolute epsilon that is
  // right at 10 000 mm swallows an entire rung at 0.1 mm.
  const epsilon = mm * 1e-9;
  const rungs = rungsAround(mm);
  if (direction > 0) return rungs.find((rung) => rung > mm + epsilon) ?? mm;
  const below = rungs.filter((rung) => rung < mm - epsilon);
  return below.length ? below[below.length - 1] : mm;
}

/** Whether a gain sits on the ladder, i.e. whether the zoom is in detent. */
export function isDetented(mm: number): boolean {
  const rung = nearestDetent(mm);
  return Number.isFinite(rung) && Math.abs(mm - rung) <= rung * 1e-6;
}

/** Millimetres one division is worth, given the framed world height. */
export function gainFor(worldHeight: number): number {
  return (worldHeight / DIVISIONS) * MM_PER_UNIT;
}

/** The framed world height that makes one division worth `mm`. */
export function worldHeightFor(mm: number): number {
  return (mm / MM_PER_UNIT) * DIVISIONS;
}

/** The orbit distance that puts the gain exactly on `mm` per division. */
export function distanceForGain(mm: number): number {
  return worldHeightFor(mm) / FOV_SCALE;
}

export interface Gain {
  /** Millimetres per division. Exact under an orthographic camera. */
  mm: number;
  /**
   * False when the number is not a stateable one: off the 1-2-5 ladder, or
   * — under perspective — not uniform over the frame. Drives the `>` prefix.
   */
  calibrated: boolean;
}

/** The gain a camera is currently showing. */
export function gainOf(distance: number, projection: Projection): Gain {
  const mm = gainFor(orthoHeightFor(distance));
  // A perspective frustum's world-per-pixel varies with depth, so one
  // division is only worth `mm` on the plane through the orbit target. That
  // is exactly the 2465's "uncalibrated scale factor" condition.
  return { mm, calibrated: projection === "orthographic" && isDetented(mm) };
}

export interface GainReadout {
  /** The rounded magnitude, e.g. `"10.0"`. */
  value: string;
  unit: "mm" | "m";
  calibrated: boolean;
  /** Value with the uncalibrated prefix applied: `"10.0"` or `">8.63"`. */
  text: string;
}

/**
 * Format a gain for the readout.
 *
 * Three significant figures — ASME Y14.5's "a dimension shall be expressed to
 * the same number of decimal places as its tolerance" has no tolerance to
 * quote here, so the rule becomes "state what the projection actually
 * resolves and no more". The unit switches to metres at a metre, the way an
 * instrument switches mV to V, so the field never carries five digits.
 */
export function formatGain(mm: number, calibrated: boolean): GainReadout {
  if (!Number.isFinite(mm) || mm <= 0) {
    return { value: "—", unit: "mm", calibrated: false, text: "—" };
  }
  const metres = mm >= 1000;
  const magnitude = metres ? mm / 1000 : mm;
  const value = magnitude.toPrecision(3);
  return {
    value,
    unit: metres ? "m" : "mm",
    calibrated,
    text: calibrated ? value : `>${value}`,
  };
}

/** Signed octant the camera sits in, e.g. `"+X+Y+Z"` or `"+Z"` for Front. */
export function octant(yaw: number, pitch: number): string {
  const cp = Math.cos(pitch);
  const axes: [string, number][] = [
    ["X", cp * Math.sin(yaw)],
    ["Y", Math.sin(pitch)],
    ["Z", cp * Math.cos(yaw)],
  ];
  const parts = axes
    .filter(([, component]) => Math.abs(component) > 1e-3)
    .map(([name, component]) => `${component > 0 ? "+" : "−"}${name}`);
  return parts.join("") || "—";
}

/**
 * Name of the standard view the camera is on, or `FREE`.
 *
 * Derived from the angles rather than remembered from the last preset click,
 * so the readout cannot claim FRONT after the user has orbited away.
 */
export function viewLabel(yaw: number, pitch: number): string {
  const wrapped = Math.atan2(Math.sin(yaw), Math.cos(yaw));
  for (const [name, preset] of Object.entries(VIEW_PRESETS)) {
    const presetYaw = Math.atan2(Math.sin(preset.yaw), Math.cos(preset.yaw));
    const yawDelta = Math.abs(
      Math.atan2(Math.sin(wrapped - presetYaw), Math.cos(wrapped - presetYaw)),
    );
    // Looking straight up or down, yaw is a spin about the view axis and does
    // not change which view it is.
    const polar = Math.abs(Math.abs(preset.pitch) - Math.PI / 2) < 1e-6;
    if (Math.abs(pitch - preset.pitch) < 0.01 && (polar || yawDelta < 0.01)) {
      return name.toUpperCase();
    }
  }
  return "FREE";
}
