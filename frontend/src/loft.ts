/**
 * Pure helpers for the two-click loft flow.
 *
 * Kept free of DOM and state imports so the second-sketch validation can be
 * unit tested without a browser, mirroring `constraints.ts`.
 */

import type { ConstructionNode } from "./types";

/** First pick of a loft: the sketch whose panel armed the flow. */
export interface PendingLoft {
  nodeId: string;
  line: number;
}

/**
 * Why a picked node cannot complete a pending loft, or null when it can.
 *
 * The vertex-count equality is enforced server-side; the client only rejects
 * what it can already see (not a sketch, the same sketch, not editable, or a
 * sketch that already drives an operation).
 */
export function loftPickError(
  pending: PendingLoft,
  node: Pick<ConstructionNode, "id" | "kind" | "editable" | "line" | "operators"> | null | undefined,
): string | null {
  if (!node || node.kind !== "profile") return "Loft: click a second sketch.";
  if (node.id === pending.nodeId) return "Loft: choose a different sketch.";
  if (!node.editable || node.line === null) {
    return "That sketch cannot be edited from source.";
  }
  if (node.operators.length > 0) return "That sketch already drives an operation.";
  return null;
}
