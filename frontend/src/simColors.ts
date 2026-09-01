/**
 * The single source of truth for every simulation/meshing color role.
 *
 * The viewport is dark and two scalar ramps live on it, so each role is
 * chosen for a specific legibility contract (asserted numerically in
 * test/simColors.test.ts):
 *
 * - FIELD ramp — viridis, for solved nodal fields (temperature, stress).
 *   Its yellow-green→yellow high end is *reserved*: no overlay hue may sit
 *   there, or a hot region would masquerade as an annotation.
 * - QUALITY ramp — magma, for mesh-quality views. Visually distinct from
 *   viridis end to end, so a quality heatmap is never mistaken for a
 *   temperature field.
 * - BC overlay hues — four saturated colors, mutually distinguishable,
 *   ≥3:1 contrast against the dark viewport background, and clear of the
 *   Simulate amber accent, the selection lime, and both ramps' high ends.
 * - Proposal cyan — the builder's live selection preview.
 * - Element edges — near-black charcoal: crisp over both ramps' mid/high
 *   ranges (where nearly all pixels of a well-formed mesh sit); over the
 *   darkest field regions edge legibility falls back to facet shading.
 *
 * The two ramp polynomials are duplicated as WGSL constants in
 * viewer/simulation.wgsl — keep the tables in sync.
 */

import type { StudyBcType } from "./types";

export type Rgb = readonly [number, number, number];

/** Degree-6 per-channel polynomial fit of matplotlib's viridis. */
export const VIRIDIS_COEFFICIENTS: readonly Rgb[] = [
  [0.2744554245, 0.0057679624, 0.3326638811],
  [0.1077083262, 1.3964696839, 1.3867705979],
  [-0.3272410968, 0.2148135645, 0.0919768808],
  [-4.5999315182, -5.7582381893, -19.2918089503],
  [6.2037359013, 14.1539649474, 56.6562995652],
  [4.7517868889, -13.7494394044, -65.3209678276],
  [-5.432077171, 4.641571316, 26.2721076045],
] as const;

/** Degree-6 per-channel polynomial fit of matplotlib's magma (err ≤ 0.024). */
export const MAGMA_COEFFICIENTS: readonly Rgb[] = [
  [-0.0020666453, -0.0006875655, -0.0095482507],
  [0.2504864448, 0.6944550333, 2.4952869139],
  [8.3459009063, -3.5960313696, 0.3290570684],
  [-27.6669694889, 14.2538530831, -13.6465831585],
  [52.1706837385, -27.9445843529, 12.8810906346],
  [-50.7585722964, 29.0538803789, 4.2699357345],
  [18.6642528253, -11.4900266123, -5.5707689618],
] as const;

/** Evaluate a ramp polynomial at t ∈ [0, 1] (clamped, channels clamped). */
function evaluateRamp(coefficients: readonly Rgb[], t: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, t));
  const color: [number, number, number] = [0, 0, 0];
  for (let channel = 0; channel < 3; channel++) {
    let value = coefficients[coefficients.length - 1][channel];
    for (let order = coefficients.length - 2; order >= 0; order--) {
      value = value * x + coefficients[order][channel];
    }
    color[channel] = Math.min(1, Math.max(0, value));
  }
  return color;
}

/** The FIELD ramp (viridis) at t. */
export const fieldRamp = (t: number): [number, number, number] =>
  evaluateRamp(VIRIDIS_COEFFICIENTS, t);

/** The QUALITY ramp (magma) at t. */
export const qualityRamp = (t: number): [number, number, number] =>
  evaluateRamp(MAGMA_COEFFICIENTS, t);

/** CSS color string from a normalized rgb triple. */
export const cssColor = (rgb: Rgb): string =>
  `rgb(${rgb.map((channel) => Math.round(Math.min(1, Math.max(0, channel)) * 255)).join(", ")})`;

/** CSS linear-gradient sampling one of the ramps, for legends. */
function rampGradient(ramp: (t: number) => [number, number, number], stops = 12): string {
  const colors: string[] = [];
  for (let index = 0; index <= stops; index++) {
    colors.push(cssColor(ramp(index / stops)));
  }
  return `linear-gradient(to right, ${colors.join(", ")})`;
}

export const fieldRampCss = (stops = 12): string => rampGradient(fieldRamp, stops);
export const qualityRampCss = (stops = 12): string => rampGradient(qualityRamp, stops);

/**
 * The viewport's effective dark background, for contrast assertions.
 *
 * The raymarched environment is a dark gradient; this is its brightest
 * representative sample, so a ratio passing here passes everywhere.
 */
export const VIEWPORT_BACKGROUND: Rgb = [0.09, 0.1, 0.09];

/**
 * BC-type overlay hues (area tint + the panel's row swatches and legend).
 *
 * azure / orange / violet / red: mutually distinguishable, ≥3:1 against
 * the dark viewport, and none sits on either ramp's reserved high end.
 */
export const BC_TYPE_COLORS: Record<StudyBcType, Rgb> = {
  dirichlet: [0.24, 0.545, 1.0],
  heat_flux: [1.0, 0.54, 0.15],
  fixed: [0.72, 0.42, 1.0],
  traction: [1.0, 0.3, 0.37],
};

/**
 * The builder's live proposal preview: bright cyan. Not on either ramp's
 * high end, far from the selection lime and the Simulate amber accent.
 */
export const PROPOSAL_COLOR: Rgb = [0.25, 0.9, 1.0];

/**
 * Element-edge lines, dark tone: drawn where the ramp value under the
 * hairline is bright. Mirrored as a constant in simulation.wgsl.
 */
export const ELEMENT_EDGE_COLOR: Rgb = [0.05, 0.05, 0.06];

/**
 * Element-edge lines, light tone: drawn where the ramp value under the
 * hairline is dark (viridis' purple end swallows a charcoal line). The
 * shader picks between the two by the ramp colour's relative luminance, so
 * the wireframe stays legible across the whole field. Mirrored in
 * simulation.wgsl.
 */
export const ELEMENT_EDGE_COLOR_LIGHT: Rgb = [0.86, 0.9, 0.94];

/** Quality accent for panel chrome tied to the quality ramp (histogram). */
export const QUALITY_ACCENT: Rgb = qualityRamp(0.75);

// ── contrast math (WCAG relative luminance), used by the color tests ──────

function linearChannel(value: number): number {
  return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
}

/** WCAG relative luminance of a normalized rgb triple. */
export function relativeLuminance(rgb: Rgb): number {
  return (
    0.2126 * linearChannel(rgb[0]) +
    0.7152 * linearChannel(rgb[1]) +
    0.0722 * linearChannel(rgb[2])
  );
}

/** WCAG contrast ratio between two colors (≥ 1, order-independent). */
export function contrastRatio(a: Rgb, b: Rgb): number {
  const lighter = Math.max(relativeLuminance(a), relativeLuminance(b));
  const darker = Math.min(relativeLuminance(a), relativeLuminance(b));
  return (lighter + 0.05) / (darker + 0.05);
}
