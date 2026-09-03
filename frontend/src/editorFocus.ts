/**
 * Which characters the editor reveals when the viewport selection changes.
 *
 * Selecting something in the 3D view should show it in the source, and the
 * rule for *what* to show has two levels, because a selection has two levels:
 *
 * - **A precise selection points at its own literal.** A sketch vertex is
 *   `[0.75, 0.85]` in the text, and that is what the reader wants to see and
 *   edit — not the twenty-two-line sketch it belongs to.
 * - **A whole object points at its declaration.** `board = Solid.box(...)`,
 *   the statement, because "where is this object in my program" is the
 *   question a whole-object selection asks.
 *
 * The second level is the one that was missing. The memo used to answer with
 * `spans.position`, which is the position *literal* — so selecting a primitive
 * scrolled the editor to three numbers, and selecting a sketch did nothing at
 * all, because a profile publishes no argument spans. Sketches are exactly
 * what a user selects while working on sketch placement, so "nothing at all"
 * was the common case.
 *
 * `statementSpan` on the payload is the fix; this module is the rule that
 * chooses between it and the precise span, kept out of `App.tsx` so it can be
 * tested without a component.
 */

import type { ConstructionNode } from "./types";

/** A character range in the program text, and how precisely it was chosen. */
export interface FocusSpan {
  from: number;
  to: number;
  /**
   * Whether this is the exact literal for the selection.
   *
   * A precise span is small and worth tinting whole. A statement span can run
   * to twenty lines, and the editor tints only its first line — the reveal
   * still lands on the declaration, without painting a block over a fifth of
   * the file.
   */
  precise: boolean;
}

/**
 * The span to reveal for one selection, or null when the source cannot say.
 *
 * @param node The selected construction node, if it still exists.
 * @param vertexIndex The selected vertex, or null for the whole object.
 * @returns The span and its precision, or null when nothing can be shown —
 *   an object built in a loop, whose statement the mapper refuses to pin.
 */
export function focusSpan(
  node: ConstructionNode | undefined,
  vertexIndex: number | null,
): FocusSpan | null {
  if (!node) return null;
  if (vertexIndex !== null) {
    const vertex = node.vertices[vertexIndex]?.span;
    if (vertex) return { from: vertex[0], to: vertex[1], precise: true };
    // A vertex the source cannot pin still belongs to a sketch that can be
    // shown; falling through is better than refusing to move at all.
  }
  const statement = node.statementSpan;
  if (statement) return { from: statement[0], to: statement[1], precise: false };
  // Older payloads, and anything the statement locator could not place: the
  // position literal is still better than nothing for a primitive.
  const position = node.spans?.position;
  return position ? { from: position[0], to: position[1], precise: true } : null;
}
