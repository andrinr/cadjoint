/**
 * The physical half of a material, as the inspector reads it.
 *
 * A `Material` carries two populations and they answer to different people:
 * colour, roughness and metallic are what the *renderer* uses, and density,
 * conductivity and the elastic constants are what the *solver* uses. The
 * inspector shows them as two numbered sections for exactly that reason, and
 * this module owns the second one: which properties there are, in what order,
 * what they are called in prose, and how a number is printed.
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

/** One row of the physical section: a label, a printed value, and its unit. */
export interface PhysicalRow {
  key: string;
  label: string;
  value: string;
  /** Empty for a dimensionless ratio, which the payload reports as `-`. */
  unit: string;
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
 * The rows to draw for one material: every property it actually states.
 *
 * A property the Material does not declare is not "zero" and not "unknown to
 * the solver" — it simply is not part of this declaration, so it is left out
 * rather than shown empty. An older server sends no `physical` block at all,
 * and that is the same answer: nothing to draw.
 */
export function physicalRows(material: MaterialDefinition): PhysicalRow[] {
  const physical = material.physical;
  if (!physical) return [];
  const units = material.units ?? {};
  const rows: PhysicalRow[] = [];
  for (const { key, label } of PHYSICAL_PROPERTIES) {
    const value = physical[key];
    if (typeof value !== "number" || !Number.isFinite(value)) continue;
    const unit = units[key];
    rows.push({
      key,
      label,
      value: formatPhysical(value),
      unit: unit && unit !== "-" ? unit : "",
    });
  }
  return rows;
}

/** Whether the inspector should draw a physical section at all. */
export const hasPhysicalBlock = (material: MaterialDefinition): boolean =>
  material.physical !== undefined;
