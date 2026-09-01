/**
 * Pure helpers shared by the constraint tools and the sketch panel.
 *
 * Kept free of DOM and state imports so labels and edge-index arithmetic can be
 * unit tested without a browser.
 */

import type { ConstructionConstraint, ToolMode } from "./types";

/** Constraint tools that pick two sketch vertices. */
export const VERTEX_CONSTRAINT_TOOLS = [
  "distance",
  "horizontal",
  "vertical",
  "coincident",
] as const;

/** Constraint tools that pick two sketch edges. */
export const EDGE_CONSTRAINT_TOOLS = ["parallel", "perpendicular"] as const;

export type VertexConstraintTool = (typeof VERTEX_CONSTRAINT_TOOLS)[number];
export type EdgeConstraintTool = (typeof EDGE_CONSTRAINT_TOOLS)[number];

export function isVertexConstraintTool(tool: ToolMode): tool is VertexConstraintTool {
  return (VERTEX_CONSTRAINT_TOOLS as readonly string[]).includes(tool);
}

export function isEdgeConstraintTool(tool: ToolMode): tool is EdgeConstraintTool {
  return (EDGE_CONSTRAINT_TOOLS as readonly string[]).includes(tool);
}

/** Short user-facing name of a constraint tool, for hints and status lines. */
export const CONSTRAINT_TOOL_NAMES: Record<VertexConstraintTool | EdgeConstraintTool, string> = {
  distance: "Distance",
  horizontal: "Horizontal",
  vertical: "Vertical",
  coincident: "Coincident",
  parallel: "Parallel",
  perpendicular: "Perpendicular",
};

/**
 * Vertex indices of the picked edge, from a `pickEdge` insertion index.
 *
 * The profile loop is closed, so the edge starting at the last vertex wraps
 * back to vertex 0.
 */
export function edgeVertexIndices(
  insertIndex: number,
  vertexCount: number,
): [number, number] {
  const start = insertIndex - 1;
  return [start, (start + 1) % Math.max(vertexCount, 1)];
}

/** Compact chip label for one constraint, e.g. `horiz · P1–P2`. */
export function constraintLabel(
  constraint: Pick<ConstructionConstraint, "kind" | "vertices" | "value">,
): string {
  const point = (index: number) => `P${index + 1}`;
  const [a, b, c, d] = constraint.vertices;
  switch (constraint.kind) {
    case "fixed":
      return `fix · ${point(a)}`;
    case "distance": {
      const value =
        typeof constraint.value === "number"
          ? ` = ${Number(constraint.value.toPrecision(4))}`
          : "";
      return `distance · ${point(a)}–${point(b)}${value}`;
    }
    case "horizontal":
      return `horiz · ${point(a)}–${point(b)}`;
    case "vertical":
      return `vert · ${point(a)}–${point(b)}`;
    case "coincident":
      return `coinc · ${point(a)}–${point(b)}`;
    case "equal_length":
      return `equal · ${point(a)}${point(b)}–${point(c)}${point(d)}`;
    case "parallel":
      return `∥ · ${point(a)}${point(b)}–${point(c)}${point(d)}`;
    case "perpendicular":
      return `⊥ · ${point(a)}${point(b)}–${point(c)}${point(d)}`;
  }
}
