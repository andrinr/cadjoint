/**
 * The viewport's ground grid: how far apart its lines are, and where the
 * detents on that spacing lie.
 *
 * This used to be a *faceplate* — a fixed screen-space rule, eight divisions
 * down the viewport, with the geometry moving behind it the way a trace moves
 * behind a CRT's etched graticule. It was honest but it was inert: pinned to
 * the frame, it never moved with the scene, so it said nothing about where
 * anything was in space. What replaced it is the CAD convention instead: a
 * square grid ruled on a world coordinate plane — the ground, z = 0, wherever
 * the ground is worth ruling, since the world is Z-up — drawn in projection so
 * it recedes, which is what actually tells you where the floor is and which way
 * you are looking. In the shallow views, where the floor is edge-on and rules
 * as a single line, the grid moves to the wall the camera faces instead; see
 * `GRID_FLOOR_PREFERENCE`.
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

import { FOV_SCALE, orthoHeightFor, type Projection, type Vec3 } from "./math";
import { sameView, VIEW_PRESETS } from "./display";

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

/**
 * The world coordinate plane a grid is ruled on, named the way a CAD user
 * names one: by the two axes that span it.
 */
export type GridPlane = "XY" | "XZ" | "YZ";

/** The three, in the order `gridPlaneWeights` returns them. */
export const GRID_PLANES: readonly GridPlane[] = ["XY", "XZ", "YZ"];

/** The normal of each, in the same order. `XY` is the floor. */
export const GRID_PLANE_NORMALS: readonly Vec3[] = [
  [0, 0, 1],
  [0, 1, 0],
  [1, 0, 0],
];

/**
 * How much the floor is preferred over a wall, as a multiple of face-on-ness.
 *
 * On z = 0 the grid is right in Top and Bottom and useless in Front, Back,
 * Left and Right, where it is exactly edge-on and draws as a single line —
 * which is precisely the set of views someone reaches for when they want to
 * measure something square. So the grid moves to the world plane the camera
 * most nearly faces, scored by |forward · n|.
 *
 * Plain argmax would be wrong, because the floor is not just one of three
 * planes: it is the model's ground, the plane a sketch defaults to, and the
 * thing the slice contours and the graticule ladder are ruled against. At the
 * ISO view all three planes are equally oblique (|f · n| = 1/√3 each) and an
 * unbiased argmax would flip between them on the first pixel of orbit, at the
 * app's own default standpoint. Two is the smallest whole preference that
 * settles it with room to spare: the floor holds down to a pitch of 26.6° in
 * an axis view and 19.5° at the worst azimuth, so every corner and edge view
 * keeps the ground and only the shallow views — the ones with no floor worth
 * drawing — hand it over.
 */
export const GRID_FLOOR_PREFERENCE = 2;

/**
 * Half-width of the crossover, in score units.
 *
 * The alternative is to swap only when the camera snaps to a preset, and it is
 * wrong: a camera at pitch 0 and yaw 20° is on no preset at all, so it would
 * keep an edge-on floor and show no grid in exactly the situation the swap
 * exists for. Swapping on the geometry instead means the swap can land
 * anywhere, so it has to dissolve rather than jump. 0.06 is about three
 * degrees of orbit either side of the crossover — long enough not to strobe on
 * a slow drag, short enough that "which plane is this" is never a question you
 * have time to ask, and the readout answers it anyway.
 */
export const GRID_PLANE_BAND = 0.06;

/** Unit view direction for an orbit, matching `cameraPosition` in `math.ts`. */
export function viewDirection(yaw: number, pitch: number): Vec3 {
  const cp = Math.cos(pitch);
  return [-cp * Math.sin(yaw), cp * Math.cos(yaw), -Math.sin(pitch)];
}

/**
 * How strongly each of the three world planes is drawn, in `GRID_PLANES` order.
 *
 * The weights sum to one, and outside the crossover bands exactly one of them
 * is one — so the grid is on a single, named plane almost everywhere, and the
 * few degrees where two are up are a dissolve between them rather than two
 * grids arguing. Kept in step with `plane_weights` in `graticule.wgsl`, which
 * is the same arithmetic on the GPU; `test/graticule.test.ts` pins the
 * behaviour both sides have to show.
 */
export function gridPlaneWeights(forward: Vec3): [number, number, number] {
  const scores: [number, number, number] = [
    Math.abs(forward[2]) * GRID_FLOOR_PREFERENCE,
    Math.abs(forward[1]),
    Math.abs(forward[0]),
  ];
  const raw = scores.map((score, index) => {
    const rival = Math.max(...scores.filter((_, other) => other !== index));
    return smoothstep(-GRID_PLANE_BAND, GRID_PLANE_BAND, score - rival);
  });
  const total = raw.reduce((sum, weight) => sum + weight, 0);
  if (!(total > 1e-6)) return [1, 0, 0];
  return raw.map((weight) => weight / total) as [number, number, number];
}

/** The classic Hermite step, matching WGSL's `smoothstep`. */
function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

/**
 * The plane the grid is on, for the readout.
 *
 * The dominant one, which is what the picture reads as: inside a crossover the
 * two weights are within a hair of each other and either name is as true as
 * the other, so the field names the leader rather than hedging. A grid whose
 * plane is not stated is worse than no grid, which is the whole reason this is
 * printed at all.
 */
export function gridPlane(yaw: number, pitch: number): GridPlane {
  const weights = gridPlaneWeights(viewDirection(yaw, pitch));
  let best = 0;
  for (let index = 1; index < weights.length; index++) {
    if (weights[index] > weights[best]) best = index;
  }
  return GRID_PLANES[best];
}

/**
 * Closest and furthest the orbit camera may sit from its target.
 *
 * Here rather than beside the gestures that clamp against it, because the
 * detent ladder is the other half of the same question: a rung outside this
 * range is a spacing the zoom can never actually sit on, and both the wheel
 * and the view presets have to refuse it rather than clamp off the ladder.
 */
export const MIN_DISTANCE = 0.4;
export const MAX_DISTANCE = 60;

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

/**
 * A signed world distance, on the same scale rule as the gain readout.
 *
 * `formatGain` is for *spacings* — always positive, never zero, and an absent
 * one is an error worth printing as an em dash. A coordinate is none of those
 * things: zero is where the slice plane starts, and half its travel is
 * negative. Same three significant figures, same switch to metres at a metre,
 * same typographic minus as the octant field, so the two readouts still look
 * like one instrument.
 */
export function formatDistance(mm: number): { value: string; unit: "mm" | "m" } {
  if (!Number.isFinite(mm)) return { value: "—", unit: "mm" };
  const metres = Math.abs(mm) >= 1000;
  const magnitude = metres ? mm / 1000 : mm;
  const rounded = Number(magnitude.toPrecision(3));
  return {
    // `toPrecision` on a metre value keeps the trailing zeros that say how far
    // the number is resolved; below a metre they would be noise on an integer.
    value: (metres ? magnitude.toPrecision(3) : String(rounded)).replace("-", "−"),
    unit: metres ? "m" : "mm",
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
 *
 * A corner view reports ISO in all eight octants. That is the honest name: a
 * 1:1:1 direction *is* an isometric direction, and which octant it points into
 * is already stated, exactly, by the octant field beside it. What ISO never
 * means here is a projection — the readout is describing where the camera
 * stands, and orthographic-versus-perspective is a separate control.
 */
export function viewLabel(yaw: number, pitch: number): string {
  for (const [name, preset] of Object.entries(VIEW_PRESETS)) {
    if (sameView(yaw, pitch, preset)) return preset.label ?? name.toUpperCase();
  }
  return "FREE";
}
