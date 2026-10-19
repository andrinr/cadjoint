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
import { cssColor } from "../../simColors";
import type { BoundaryConditionType } from "../../types";

/** The BC type's overlay colour as a CSS `rgb()` string. */
export const bcSwatch = (type: BoundaryConditionType): string => cssColor(BC_TYPE_COLORS[type]);
