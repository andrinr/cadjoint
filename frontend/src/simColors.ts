/**
 * The single source of truth for every simulation/meshing color role.
 *
 * The viewport is a light paper ground and two scalar ramps live on it, so
 * each role is chosen for a specific legibility contract (asserted
 * numerically in test/simColors.test.ts):
 *
 * - FIELD ramp — viridis, for solved nodal fields (temperature, stress).
 *   Its yellow-green→yellow high end is *reserved*: no overlay hue may sit
 *   there, or a hot region would masquerade as an annotation.
 * - QUALITY ramp — magma, for mesh-quality views. Visually distinct from
 *   viridis end to end, so a quality heatmap is never mistaken for a
 *   temperature field.
 * - BC overlay hues — four saturated colors, mutually distinguishable,
 *   ≥3:1 contrast against the paper viewport background, and clear of the
 *   Simulate amber accent, the selection lime, and both ramps' high ends.
 *   All four are *darker* than paper: on a light ground an overlay reads by
 *   being heavier than its surround, not brighter.
 * - Proposal teal — the builder's live selection preview.
 * - Element edges — near-black charcoal: crisp over both ramps' mid/high
 *   ranges (where nearly all pixels of a well-formed mesh sit); over the
 *   darkest field regions edge legibility falls back to facet shading.
 *
 * The WGSL shaders get the ramp polynomials from `wgslTokens()` (tokens.ts).
 */

import { MAGMA_COEFFICIENTS, VIRIDIS_COEFFICIENTS, type Rgb } from "./tokens";

// The values live in tokens.ts, the one source; this module evaluates them.
export {
  BC_TYPE_COLORS,
  ELEMENT_EDGE_COLOR,
  ELEMENT_EDGE_COLOR_LIGHT,
  MAGMA_COEFFICIENTS,
  PROPOSAL_COLOR,
  VIEWPORT_BACKGROUND,
  VIRIDIS_COEFFICIENTS,
  contrastRatio,
  relativeLuminance,
  type Rgb,
} from "./tokens";

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

/** CSS linear-gradient sampling a ramp over t ∈ [0, 1], for legends. */
export function rampGradient(ramp: (t: number) => [number, number, number], stops = 12): string {
  const colors: string[] = [];
  for (let index = 0; index <= stops; index++) {
    colors.push(cssColor(ramp(index / stops)));
  }
  return `linear-gradient(to right, ${colors.join(", ")})`;
}

export const fieldRampCss = (stops = 12): string => rampGradient(fieldRamp, stops);
export const qualityRampCss = (stops = 12): string => rampGradient(qualityRamp, stops);

