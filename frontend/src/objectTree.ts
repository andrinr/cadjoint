/**
 * Pure derivation of the visual scene-composition tree.
 *
 * Built only from the construction payload the compiler already reports — no
 * extra backend state. The Object tree panel renders this read-only; editing
 * always happens through the source or the viewport tools.
 */

import type { ConstructionNode } from "./types";

export interface SceneTreeRow {
  /** Stable key for rendering and collapse state. */
  key: string;
  /** Construction node this row selects, or null for scene/operator rows. */
  nodeId: string | null;
  depth: number;
  /** Icon/label vocabulary: scene, profile, box, sphere, cylinder, operator. */
  kind: "scene" | "profile" | "box" | "sphere" | "cylinder" | "operator";
  label: string;
  /** Compact dimensions or point-count summary, when derivable. */
  detail: string | null;
  /** Named material assigned to the node, when any. */
  material: string | null;
  /** Sketch-level constraint count (0 hides the badge). */
  constraintCount: number;
  /** True when the row has children and can collapse. */
  group: boolean;
}

/** Format one number the way the source panes do: short, no trailing zeros. */
function formatNumber(value: number): string {
  const rounded = Number(value.toFixed(2));
  return Object.is(rounded, -0) ? "0" : String(rounded);
}

/** Compact dimension summary for a solid, from its transform payload. */
export function dimensionSummary(node: ConstructionNode): string | null {
  if (node.kind === "profile") {
    const count = node.vertices.length;
    return count > 0 ? `${count} point${count === 1 ? "" : "s"}` : null;
  }
  const dimensions = node.transform?.dimensions;
  if (!dimensions) return null;
  const parts: string[] = [];
  const size = dimensions.size;
  if (Array.isArray(size)) parts.push(size.map(formatNumber).join(" × "));
  if (typeof dimensions.radius === "number") {
    parts.push(`r ${formatNumber(dimensions.radius)}`);
  }
  if (typeof dimensions.height === "number") {
    parts.push(`h ${formatNumber(dimensions.height)}`);
  }
  return parts.length > 0 ? parts.join(" · ") : null;
}

/**
 * Flatten the construction payload into scene → object → operator rows.
 *
 * The order mirrors the program: nodes appear as the compiler reported them,
 * and a sketch's operators nest beneath it in source order.
 */
export function buildSceneTree(nodes: ConstructionNode[]): SceneTreeRow[] {
  const rows: SceneTreeRow[] = [
    {
      key: "scene",
      nodeId: null,
      depth: 0,
      kind: "scene",
      label: "scene",
      detail: nodes.length > 0 ? `${nodes.length} object${nodes.length === 1 ? "" : "s"}` : "empty",
      material: null,
      constraintCount: 0,
      group: nodes.length > 0,
    },
  ];
  for (const node of nodes) {
    rows.push({
      key: node.id,
      nodeId: node.id,
      depth: 1,
      kind: node.kind,
      label: node.name ?? node.kind,
      detail: dimensionSummary(node),
      material: node.material,
      constraintCount: node.constraints.length,
      group: node.operators.length > 0,
    });
    for (const operator of node.operators) {
      rows.push({
        key: `${node.id}-op-${operator.kind}-${operator.line}`,
        nodeId: null,
        depth: 2,
        kind: "operator",
        label: operator.kind,
        detail: `line ${operator.line}`,
        material: null,
        constraintCount: 0,
        group: false,
      });
    }
  }
  return rows;
}

/** Rows visible given a set of collapsed group keys. */
export function visibleRows(
  rows: SceneTreeRow[],
  collapsed: ReadonlySet<string>,
): SceneTreeRow[] {
  const out: SceneTreeRow[] = [];
  let hideDeeperThan: number | null = null;
  for (const row of rows) {
    if (hideDeeperThan !== null && row.depth > hideDeeperThan) continue;
    hideDeeperThan = null;
    out.push(row);
    if (row.group && collapsed.has(row.key)) hideDeeperThan = row.depth;
  }
  return out;
}
