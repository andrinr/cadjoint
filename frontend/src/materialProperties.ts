/**
 * The physical half of a material, as the inspector reads it.
 *
 * A `Material` carries two populations and they answer to different people:
 * colour, roughness and metallic are what the *renderer* uses, and density,
 * conductivity and the elastic constants are what the *solver* uses. The
 * inspector shows them as two sections for exactly that reason, and this
 * module owns the second one: which properties there are, in what order, what
 * they are called in prose, and how a number is printed.
 *
 * ── Stated, and unstated ─────────────────────────────────────────────────
 * Every physical property is *optional* on a `Material`, and the three states
 * a row can be in are not two:
 *
 *   stated    — the call carries the keyword, the payload carries a number
 *               and a span, and the inspector edits it in place;
 *   unstated  — the call does not carry the keyword. The payload says so with
 *               a null *and* no span. This is not "zero" and not "unknown":
 *               it is a property this declaration does not make, and the only
 *               thing to offer is to start making it;
 *   opaque    — the keyword is there but its value is an expression rather
 *               than a literal (a parameter, an arithmetic). The payload has
 *               a span but no number it can round-trip; the row shows what it
 *               can and sends the reader to the code.
 *
 * The rows used to be built by dropping everything that was not the first
 * case, which made the inspector a read-only list of whatever happened to be
 * declared — the one thing you could not do from it was declare something.
 *
 * Pure; unit tested in `test/materialProperties.test.ts`.
 */

import type { MaterialDefinition } from "./types";

/**
 * The properties, in reading order.
 *
 * Ordered by what a reader is looking for rather than alphabetically: mass
 * first, then the thermal three that a conduction study needs, then the
 * mechanical three an elastic one does.
 */
export const PHYSICAL_PROPERTIES: { key: string; label: string }[] = [
  { key: "density", label: "Density" },
  { key: "conductivity", label: "Conductivity" },
  { key: "specific_heat", label: "Specific heat" },
  { key: "youngs_modulus", label: "Young's modulus" },
  { key: "poisson_ratio", label: "Poisson ratio" },
  { key: "thermal_expansion", label: "Thermal expansion" },
  { key: "yield_strength", label: "Yield strength" },
];

/** How much of a property this declaration actually says. */
export type PhysicalState = "stated" | "unstated" | "opaque";

/** One row of the physical section: a label, what it says, and its unit. */
export interface PhysicalRow {
  key: string;
  label: string;
  state: PhysicalState;
  /** The SI value, or null when the row is not a number this can edit. */
  value: number | null;
  /** `value` printed, or an em dash for a row with no number to print. */
  text: string;
  /** Empty for a dimensionless ratio, which the payload reports as `-`. */
  unit: string;
  /** True when an optimization is allowed to move this number. */
  free: boolean;
}

/**
 * Print one SI value.
 *
 * The server has already rounded to six significant figures for display, so
 * this only decides between plain and exponential notation: a Young's modulus
 * is 7e10 and a thermal expansion is 2.3e-5, and both are unreadable in the
 * other form. The threshold is where a plain number stops fitting a panel
 * column without becoming a digit-counting exercise.
 */
export function formatPhysical(value: number): string {
  if (!Number.isFinite(value)) return "—";
  if (value === 0) return "0";
  const magnitude = Math.abs(value);
  if (magnitude >= 1e5 || magnitude < 1e-3) {
    // `7e+10` reads worse than `7e10`, and the sign of a negative exponent is
    // the only one that carries information.
    return value.toExponential(4).replace(/e\+?(-?)/, "e$1").replace(/\.?0+e/, "e");
  }
  return String(Number(value.toPrecision(6)));
}

/**
 * The rows to draw for one material: all seven, in reading order.
 *
 * All seven, because an unstated property is a row the inspector has an offer
 * for — "state it" — and a row that is not drawn is an offer nobody can take.
 * What varies is the row's `state`, and the panel draws each one differently.
 * An older server sends no `physical` block at all, and that is a different
 * answer: it cannot tell us anything about any of them, so there is nothing
 * to draw and `hasPhysicalBlock` says so.
 */
export function physicalRows(material: MaterialDefinition): PhysicalRow[] {
  const physical = material.physical;
  if (!physical) return [];
  const units = material.units ?? {};
  const free = material.free ?? {};
  const spans = material.spans ?? {};
  return PHYSICAL_PROPERTIES.map(({ key, label }) => {
    const value = physical[key];
    const numeric = typeof value === "number" && Number.isFinite(value) ? value : null;
    // A span with no number is a keyword whose value is an expression; no
    // span and no number is a keyword that is simply not there.
    const state: PhysicalState =
      numeric !== null ? "stated" : key in spans ? "opaque" : "unstated";
    const unit = units[key];
    return {
      key,
      label,
      state,
      value: numeric,
      text: numeric === null ? "—" : formatPhysical(numeric),
      unit: unit && unit !== "-" ? unit : "",
      free: free[key] === true,
    };
  });
}

/** The rows this material actually states — what a reader sees as content. */
export const statedRows = (material: MaterialDefinition): PhysicalRow[] =>
  physicalRows(material).filter((row) => row.state === "stated");

/** Whether the inspector should draw a physical section at all. */
export const hasPhysicalBlock = (material: MaterialDefinition): boolean =>
  material.physical !== undefined;
