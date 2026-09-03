/**
 * The diverging ramp the SDF views are drawn with.
 *
 * ── Why a third ramp at all ──────────────────────────────────────────────
 * The viewport already carries two: viridis for solved nodal fields and magma
 * for mesh quality (`simColors.ts`). Both are *sequential* — they run from a
 * low end to a high end — and a signed distance is not that shape. It has a
 * distinguished value, zero, which is the surface itself, and the two sides of
 * it mean opposite things. A sequential ramp puts zero at an arbitrary point
 * along its length and buries the one value the reader is looking for.
 *
 * ── The two hues ─────────────────────────────────────────────────────────
 * Violet `#6400fd` inside the solid (f < 0), ochre `#8a5601` outside it
 * (f > 0), both taken at the most chroma sRGB holds at OKLab L 0.50 —
 * `oklch(0.50 0.292 285°)` and `oklch(0.50 0.108 70°)`, 145° apart in hue.
 *
 * They were chosen by scanning every hue at that lightness for the largest
 * minimum distance to viridis and magma, measured as Euclidean sRGB distance
 * against 201 samples of each ramp — the same measure `test/simColors.test.ts`
 * uses to hold the two existing ramps apart, where the passing threshold is
 * 0.25. Measured here:
 *
 *              vs viridis   vs magma   vs paper ground
 *   violet        0.545       0.487        5.69 : 1
 *   ochre         0.538       0.505        4.94 : 1
 *
 * Both ends therefore clear the existing threshold roughly twice over, and
 * both are *darker* than the paper viewport, which is the rule every overlay
 * on this ground follows: a mark reads by being heavier than its surround.
 *
 * The centre is the paper itself rather than white. A diverging ramp's pale
 * middle is the one place it approaches magma's pale-yellow high end (0.21 at
 * the closest), and that is unavoidable for any ramp with a neutral centre —
 * it is also where the achromatic zero isoline is drawn, so the value that
 * matters at the centre is carried by a line, not by a hue.
 *
 * These constants are mirrored as literals in the WGSL that
 * `cadjoint/viewer/_webgpu.py` generates; `test/sdfRamp.test.ts` reads that
 * file and holds the two copies to the same numbers.
 */

import type { Rgb } from "../simColors";

/** f < 0 — inside the solid. `oklch(0.50 0.292 285°)`. */
export const SDF_INSIDE: Rgb = [0.3924, 0.0015, 0.9925];

/** f > 0 — outside the solid. `oklch(0.50 0.108 70°)`. */
export const SDF_OUTSIDE: Rgb = [0.5398, 0.3384, 0.0033];

/**
 * The ramp's neutral centre: the viewport's own paper.
 *
 * Mirrors `VIEWPORT_BACKGROUND`, so a field value at zero is the value of the
 * ground it is drawn on and the isoline is the only thing at the crossing.
 */
export const SDF_CENTRE: Rgb = [0.902, 0.902, 0.914];

/**
 * The ramp at a normalized signed value, t ∈ [-1, 1].
 *
 * The interpolation is in the square root of |t| rather than in |t| itself:
 * near the surface — where every question about an SDF is asked — a linear
 * ramp spends almost none of its range, and the band either side of zero comes
 * out the same colour as the band either side of that.
 */
export function sdfRamp(t: number): [number, number, number] {
  const clamped = Math.min(1, Math.max(-1, t));
  const end = clamped < 0 ? SDF_INSIDE : SDF_OUTSIDE;
  const weight = Math.sqrt(Math.abs(clamped));
  return [0, 1, 2].map(
    (channel) => SDF_CENTRE[channel] + (end[channel] - SDF_CENTRE[channel]) * weight,
  ) as [number, number, number];
}

/** CSS `linear-gradient` across the whole ramp, for the legend. */
export function sdfRampCss(stops = 16): string {
  const colors: string[] = [];
  for (let index = 0; index <= stops; index++) {
    const [r, g, b] = sdfRamp((index / stops) * 2 - 1);
    colors.push(
      `rgb(${[r, g, b].map((channel) => Math.round(channel * 255)).join(", ")})`,
    );
  }
  return `linear-gradient(to right, ${colors.join(", ")})`;
}
