/**
 * Client-side evaluation of serialized Nodes selections for BC previews.
 *
 * Mirrors cadjoint/fem/selection.py over the simulation render payload: the
 * payload ships exactly the boundary-surface vertices, so the server's
 * implicit "boundary nodes only" restriction is already satisfied and the
 * complement (`not`) is a plain negation. `side` reproduces the server's
 * axis-extreme semantics: the extreme is taken over the same boundary
 * vertices, with a default tolerance of half the smallest cell spacing
 * (fallback: 1e-3 of the bounding-box diagonal, as in _default_side_tol).
 *
 * Predicate selections run arbitrary Python and cannot be previewed here.
 */

import type { StudySelection } from "./types";

/** Whether every leaf of a selection can be evaluated client-side. */
export function selectionEvaluable(selection: StudySelection): boolean {
  switch (selection.kind) {
    case "predicate":
      return false;
    case "and":
    case "or":
      return selection.operands.every(selectionEvaluable);
    case "not":
      return selectionEvaluable(selection.operand);
    default:
      return true;
  }
}

/** Grid summary needed for `side` tolerances (mesh_info.grid). */
export interface GridSpacing {
  spacing: number[];
}

/** Default side tolerance, matching selection.py's _default_side_tol. */
export function defaultSideTol(
  positions: readonly number[],
  grid: GridSpacing | null,
): number {
  if (grid && grid.spacing.length > 0) return 0.5 * Math.min(...grid.spacing);
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let index = 0; index + 2 < positions.length; index += 3) {
    for (let axis = 0; axis < 3; axis++) {
      const value = positions[index + axis];
      if (value < min[axis]) min[axis] = value;
      if (value > max[axis]) max[axis] = value;
    }
  }
  if (!Number.isFinite(min[0])) return 0;
  return 1e-3 * Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
}

/**
 * Evaluate a selection against flat xyz `positions` (boundary vertices).
 *
 * Returns one boolean per vertex, or null when the selection contains a
 * predicate leaf and cannot be previewed.
 */
export function evaluateSelection(
  selection: StudySelection,
  positions: readonly number[],
  grid: GridSpacing | null,
): boolean[] | null {
  if (!selectionEvaluable(selection)) return null;
  const count = Math.floor(positions.length / 3);

  const evaluate = (node: StudySelection): boolean[] => {
    const mask = new Array<boolean>(count);
    switch (node.kind) {
      case "box": {
        const low = node.min_corner;
        const high = node.max_corner;
        for (let index = 0; index < count; index++) {
          const x = positions[index * 3];
          const y = positions[index * 3 + 1];
          const z = positions[index * 3 + 2];
          mask[index] =
            x >= low[0] && x <= high[0] &&
            y >= low[1] && y <= high[1] &&
            z >= low[2] && z <= high[2];
        }
        return mask;
      }
      case "sphere": {
        const radiusSq = node.radius * node.radius;
        for (let index = 0; index < count; index++) {
          const dx = positions[index * 3] - node.center[0];
          const dy = positions[index * 3 + 1] - node.center[1];
          const dz = positions[index * 3 + 2] - node.center[2];
          mask[index] = dx * dx + dy * dy + dz * dz <= radiusSq;
        }
        return mask;
      }
      case "halfspace": {
        for (let index = 0; index < count; index++) {
          const dx = positions[index * 3] - node.point[0];
          const dy = positions[index * 3 + 1] - node.point[1];
          const dz = positions[index * 3 + 2] - node.point[2];
          mask[index] =
            dx * node.normal[0] + dy * node.normal[1] + dz * node.normal[2] >= 0;
        }
        return mask;
      }
      case "side": {
        const axis = "xyz".indexOf(node.side[1]);
        const positive = node.side[0] === "+";
        let extreme = positive ? -Infinity : Infinity;
        for (let index = 0; index < count; index++) {
          const value = positions[index * 3 + axis];
          extreme = positive ? Math.max(extreme, value) : Math.min(extreme, value);
        }
        const tol = node.tol ?? defaultSideTol(positions, grid);
        for (let index = 0; index < count; index++) {
          const value = positions[index * 3 + axis];
          mask[index] = positive ? value >= extreme - tol : value <= extreme + tol;
        }
        return mask;
      }
      case "and":
      case "or": {
        const operands = node.operands.map(evaluate);
        for (let index = 0; index < count; index++) {
          mask[index] =
            node.kind === "and"
              ? operands.every((operand) => operand[index])
              : operands.some((operand) => operand[index]);
        }
        return mask;
      }
      case "not": {
        const operand = evaluate(node.operand);
        for (let index = 0; index < count; index++) mask[index] = !operand[index];
        return mask;
      }
      case "predicate":
        // Unreachable: selectionEvaluable() rejected the tree above.
        return mask.fill(false);
    }
  };

  return evaluate(selection);
}

// Overlay hues live in the central color-role module; re-exported here so
// the evaluator and its consumers share one import site.
export { BC_TYPE_COLORS, PROPOSAL_COLOR } from "./simColors";

/** How strongly a highlighted vertex is pulled toward its BC hue. */
const OVERLAY_STRENGTH = 0.8;

/** One overlay layer: a node mask tinted with a color. */
export interface OverlayLayer {
  mask: readonly boolean[];
  color: readonly [number, number, number];
}

/**
 * Flatten overlay layers into per-vertex RGBA (rgb hue + blend strength).
 *
 * Later layers win where masks overlap, matching the panel's list order —
 * the visually active row is pushed last.
 */
export function overlayColors(vertexCount: number, layers: OverlayLayer[]): Float32Array {
  const colors = new Float32Array(vertexCount * 4);
  for (const layer of layers) {
    for (let index = 0; index < vertexCount; index++) {
      if (!layer.mask[index]) continue;
      colors[index * 4] = layer.color[0];
      colors[index * 4 + 1] = layer.color[1];
      colors[index * 4 + 2] = layer.color[2];
      colors[index * 4 + 3] = OVERLAY_STRENGTH;
    }
  }
  return colors;
}
