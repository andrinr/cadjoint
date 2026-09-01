/**
 * BC colours, translated from the renderer's convention to CSS.
 *
 * `selectionEval.ts` keeps the per-type colours as 0..1 float triples,
 * because that is what the overlay buffer wants. The panel needs the same
 * colour as an `rgb()` string for the row swatches and the legend, and both
 * must agree with what the mesh is painted with — so the conversion lives
 * here once instead of being inlined at each swatch.
 */

import { BC_TYPE_COLORS } from "../../selectionEval";
import type { StudyBcType } from "../../types";

/** The BC type's overlay colour as a CSS `rgb()` string. */
export const bcSwatch = (type: StudyBcType): string =>
  `rgb(${BC_TYPE_COLORS[type].map((channel) => Math.round(channel * 255)).join(", ")})`;
