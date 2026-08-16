import { describe, expect, it } from "vitest";
import { buildSceneTree, dimensionSummary, visibleRows } from "../src/objectTree";
import type { ConstructionNode } from "../src/types";

/** Minimal construction node with overridable fields. */
function node(overrides: Partial<ConstructionNode>): ConstructionNode {
  return {
    id: "node_0",
    kind: "box",
    name: null,
    line: 1,
    editable: true,
    edges: [],
    plane: null,
    vertices: [],
    transform: null,
    spans: {},
    constraints: [],
    operators: [],
    material: null,
    ...overrides,
  };
}

const vertex = {
  name: null,
  free: true,
  uv: [0, 0] as [number, number],
  world: [0, 0, 0] as [number, number, number],
  span: null,
};

const transform = (dimensions: Record<string, number | number[]>) => ({
  position: [0, 0, 0] as [number, number, number],
  rotation: [0, 0, 0] as [number, number, number],
  dimensions,
  line: 1,
  call: "box",
  positionArgument: "position",
  canRotate: true,
});

describe("buildSceneTree", () => {
  it("derives a scene root and one row per construction node", () => {
    const rows = buildSceneTree([
      node({ id: "profile_0", kind: "profile", name: "house", vertices: [vertex, vertex] }),
      node({ id: "sphere_1", kind: "sphere", name: "glass" }),
    ]);

    expect(rows.map((row) => row.label)).toEqual(["scene", "house", "glass"]);
    expect(rows[0]).toMatchObject({ kind: "scene", depth: 0, group: true, nodeId: null });
    expect(rows[0].detail).toBe("2 objects");
    expect(rows[1]).toMatchObject({ kind: "profile", depth: 1, nodeId: "profile_0" });
    expect(rows[2]).toMatchObject({ kind: "sphere", depth: 1, nodeId: "sphere_1" });
  });

  it("nests operator rows under their sketch", () => {
    const rows = buildSceneTree([
      node({
        id: "profile_0",
        kind: "profile",
        name: "section",
        operators: [
          { kind: "extrude", line: 12 },
          { kind: "loft", line: 30 },
        ],
      }),
    ]);

    expect(rows.map((row) => [row.label, row.depth])).toEqual([
      ["scene", 0],
      ["section", 1],
      ["extrude", 2],
      ["loft", 2],
    ]);
    expect(rows[1].group).toBe(true);
    // Operator rows describe history, not selectable geometry.
    expect(rows[2].nodeId).toBeNull();
    expect(rows[2].detail).toBe("line 12");
  });

  it("carries constraint counts and material names onto rows", () => {
    const rows = buildSceneTree([
      node({
        id: "profile_0",
        kind: "profile",
        material: "clay",
        constraints: [
          { kind: "fixed", vertices: [0], value: [0, 0] },
          { kind: "distance", vertices: [0, 1], value: 2 },
        ],
      }),
    ]);

    expect(rows[1].constraintCount).toBe(2);
    expect(rows[1].material).toBe("clay");
  });

  it("labels an empty scene", () => {
    const rows = buildSceneTree([]);
    expect(rows).toHaveLength(1);
    expect(rows[0].detail).toBe("empty");
    expect(rows[0].group).toBe(false);
  });
});

describe("dimensionSummary", () => {
  it("summarizes box, sphere, and cylinder dimensions", () => {
    expect(dimensionSummary(node({ transform: transform({ size: [0.5, 1, 2.25] }) }))).toBe(
      "0.5 × 1 × 2.25",
    );
    expect(
      dimensionSummary(node({ kind: "sphere", transform: transform({ radius: 0.5 }) })),
    ).toBe("r 0.5");
    expect(
      dimensionSummary(
        node({ kind: "cylinder", transform: transform({ radius: 0.4, height: 0.55 }) }),
      ),
    ).toBe("r 0.4 · h 0.55");
  });

  it("summarizes a profile by its point count", () => {
    expect(
      dimensionSummary(node({ kind: "profile", vertices: [vertex, vertex, vertex] })),
    ).toBe("3 points");
    expect(dimensionSummary(node({ kind: "profile" }))).toBeNull();
  });

  it("returns null when no transform is available", () => {
    expect(dimensionSummary(node({}))).toBeNull();
  });
});

describe("visibleRows", () => {
  const rows = buildSceneTree([
    node({ id: "profile_0", kind: "profile", operators: [{ kind: "extrude", line: 5 }] }),
    node({ id: "sphere_1", kind: "sphere" }),
  ]);

  it("shows everything when nothing is collapsed", () => {
    expect(visibleRows(rows, new Set())).toHaveLength(4);
  });

  it("hides descendants of a collapsed group", () => {
    const visible = visibleRows(rows, new Set(["profile_0"]));
    expect(visible.map((row) => row.label)).toEqual(["scene", "profile", "sphere"]);
  });

  it("collapsing the root hides all rows below it", () => {
    const visible = visibleRows(rows, new Set(["scene"]));
    expect(visible.map((row) => row.label)).toEqual(["scene"]);
  });
});
