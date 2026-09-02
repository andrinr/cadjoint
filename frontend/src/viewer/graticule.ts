/**
 * The viewport's ground grid: how far apart its lines are, and where the
 * detents on that spacing lie.
 *
 * This used to be a *faceplate* — a fixed screen-space rule, eight divisions
 * down the viewport, with the geometry moving behind it the way a trace moves
 * behind a CRT's etched graticule. It was honest but it was inert: pinned to
 * the frame, it never moved with the scene, so it said nothing about where
 * anything was in space. What replaced it is the CAD convention instead: a
 * square grid ruled on the ground plane (z = 0 — the world is Z-up), drawn in
 * projection so it recedes, which is what actually tells you where the floor
 * is and which way you are looking.
 *
 * The number the readout states is now simply the grid's spacing, on the same
 * 1-2-5 ladder an instrument's scale switch has, chosen so a cell is about one
 * eighth of the frame height. It is a world distance, so it is exactly true
 * whatever the camera does — the thing it can no longer promise is that you can
 * *measure* with it on screen, because under perspective a cell shrinks with
 * depth. That is what `calibrated` means now, and it is the same condition the
 * Tektronix 2465 marks with a `>` prefix on an uncalibrated scale factor.
 *
 * Everything here is pure arithmetic over a camera — no DOM, no GPU — so the
 * ladder and the scale derivation are unit tested in `test/graticule.test.ts`.
 */

import { FOV_SCALE, orthoHeightFor, type Projection } from "./math";
import { VIEW_PRESETS } from "./display";

/**
 * Cells down the viewport height, at the framing the spacing is chosen for.
 *
 * Tektronix's vertical count (475A: "8 × 10 cm display"), kept because it is
 * still the right density: eight cells is enough to judge a distance against
 * and few enough that the far half of the plane has not turned to noise. It no
 * longer fixes anything on screen — the grid is in the world now — it only
 * decides which rung of the ladder the spacing lands on as you zoom.
 */
export const DIVISIONS = 8;

/**
 * Millimetres per world unit.
 *
 * The repository declares its length unit in exactly one place: the STEP
 * writer (`cadjoint/meshing/export.py`) stamps `SI_UNIT($,.METRE.)` and states
 * "one mesh unit = 1 m". The grid reads the scene through that declaration
 * rather than inventing a scale of its own — a grid whose units are made up is
 * decoration.
 */
export const MM_PER_UNIT = 1000;

/**
 * How firmly each part of the grid is printed on the paper.
 *
 * These are alphas, not tones: the three colours come from the token layer
 * (`graticule-line`, `graticule-axis`), and what varies here is how much of
 * each lands on the sheet. Composited over `--surface-viewport` they measure
 * 1.36 : 1 for a minor line, 1.58 : 1 for a major one and 1.69 : 1 for a
 * centre axis — deliberately *below* the 1.6–2.8 band structure is held to,
 * because the floor is not structure. It is a spatial cue, and it has to stop
 * existing the moment you attend to the geometry standing on it.
 * `test/graticule.test.ts` measures the composited values.
 */
export const GRID_ALPHA = {
  minor: 0.55,
  major: 0.8,
  axis: 0.7,
  /**
   * Multiplier applied while a sketch is being edited on a plane that is not
   * the floor. The sketch's own plane is the reference then, and a floor grid
   * arguing with it is one grid too many — but removing it entirely would
   * take the orientation cue away at exactly the moment a 2D view needs it,
   * so it steps back rather than leaving.
   */
  offPlane: 0.6,
} as const;

/** Minor lines between one major line and the next. */
export const GRID_MAJOR_EVERY = 5;

/**
 * Where the floor starts and finishes fading, as multiples of the orbit
 * distance, measured outward from the orbit target on the plane.
 */
export const GRID_FADE = { start: 1.6, end: 7 } as const;

/** The 1-2-5 ladder's mantissas — the detent ladder §10.1 states. */
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
  /** Millimetres between grid lines. Always a rung of the 1-2-5 ladder. */
  mm: number;
  /**
   * Whether a *screen* measurement against the grid is to scale.
   *
   * The spacing itself is a world distance and is exact either way. Under
   * perspective a cell shrinks with depth, so the grid cannot be used as a
   * ruler on the image — which is the 2465's "uncalibrated scale factor"
   * condition, and drives the `>` prefix.
   */
  calibrated: boolean;
}

/**
 * The grid spacing for a camera, in millimetres.
 *
 * Chosen so a cell is about one eighth of the framed height and then snapped
 * to the nearest 1-2-5 rung, which is why the readout never has to hedge: the
 * lines really are 10 mm apart, not 8.63.
 */
export function gainOf(distance: number, projection: Projection): Gain {
  const mm = nearestDetent(gainFor(orthoHeightFor(distance)));
  return { mm, calibrated: projection === "orthographic" };
}

/** The grid spacing for a camera, in world units — what the shader rules. */
export function gridSpacing(distance: number): number {
  return gainOf(distance, "orthographic").mm / MM_PER_UNIT;
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
 * Format a spacing for the readout.
 *
 * Three significant figures — ASME Y14.5's "a dimension shall be expressed to
 * the same number of decimal places as its tolerance" has no tolerance to
 * quote here, so the rule becomes "state what the projection actually
 * resolves and no more". A ladder rung never needs more than two anyway. The
 * unit switches to metres at a metre, the way an instrument switches mV to V,
 * so the field never carries five digits.
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

/** Signed octant the camera sits in, e.g. `"+X−Y+Z"` or `"−Y"` for Front. */
export function octant(yaw: number, pitch: number): string {
  const cp = Math.cos(pitch);
  // The same offset `cameraPosition` builds, in a Z-up world: azimuth about
  // +Z measured from −Y, elevation toward +Z.
  const axes: [string, number][] = [
    ["X", cp * Math.sin(yaw)],
    ["Y", -cp * Math.cos(yaw)],
    ["Z", Math.sin(pitch)],
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
