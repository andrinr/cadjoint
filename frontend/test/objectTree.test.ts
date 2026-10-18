import { describe, expect, it } from "vitest";
import {
  buildSceneTree,
  descendantNodes,
  dimensionSummary,
  visibleRows,
  type SceneTree,
  type SceneTreeItem,
} from "../src/objectTree";
import type { ConstructionNode } from "../src/types";

/** Minimal construction node with overridable fields. */
function node(overrides: Partial<ConstructionNode>): ConstructionNode {
  return {
    id: "node_0",
    stableId: null,
    kind: "box",
    name: null,
    line: 1,
    editable: true,
    edges: [],
    plane: null,
    faces: [],
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
  stableId: null,
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

    expect(rows.map((row) => row.label)).toEqual(["scene.py", "house", "glass"]);
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
      ["scene.py", 0],
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
          { kind: "fixed", vertices: [0], value: [0, 0], index: 0, stableId: null },
          { kind: "distance", vertices: [0, 1], value: 2, index: 1, stableId: null },
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
    expect(visible.map((row) => row.label)).toEqual(["scene.py", "profile", "sphere"]);
  });

  it("collapsing the root hides all rows below it", () => {
    const visible = visibleRows(rows, new Set(["scene"]));
    expect(visible.map((row) => row.label)).toEqual(["scene.py"]);
  });
});

/** A scene-tree item with overridable fields. */
function item(id: string, overrides: Partial<SceneTreeItem> = {}): SceneTreeItem {
  return {
    id,
    kind: "node",
    call: "Translate",
    name: null,
    node: null,
    element: null,
    operands: [],
    ...overrides,
  };
}

describe("buildSceneTree from the evaluated scene tree", () => {
  // scene = Union(body, Union(bolt, Mirror(bolt))), with `spare` built but unused.
  const NODES = [
    node({ id: "box_0", kind: "box", name: "body" }),
    node({
      id: "profile_1",
      kind: "profile",
      name: "bolt",
      vertices: [vertex, vertex, vertex],
      operators: [{ kind: "extrude", line: 4 }],
    }),
    node({ id: "sphere_2", kind: "sphere", name: "spare" }),
  ];
  const TREE: SceneTree = {
    roots: ["boolean:scene"],
    items: [
      item("boolean:scene", {
        kind: "boolean",
        call: "Union",
        element: "boolean:scene",
        operands: ["box_0", "boolean@7"],
      }),
      item("box_0", { node: "box_0" }),
      item("boolean@7", {
        kind: "boolean",
        call: "Union",
        element: "boolean@7",
        operands: ["profile_1", "transform:Mirror"],
      }),
      item("profile_1", { node: "profile_1" }),
      item("transform:Mirror", { kind: "transform", call: "Mirror", operands: ["profile_1"] }),
    ],
  };
  const ELEMENTS = [
    {
      id: "boolean:scene",
      stableId: null,
      kind: "boolean" as const,
      call: "Union",
      line: 9,
      span: [0, 0] as [number, number],
      name: null,
      variable: "scene",
      owner: null,
      arguments: [],
    },
  ];
  const rows = buildSceneTree(NODES, ELEMENTS, "demo.py", TREE);
  const byTest = (key: string) => rows.find((row) => row.testKey === key)!;

  it("nests operands under their boolean, in argument order", () => {
    expect(rows.map((row) => [row.depth, row.label])).toEqual([
      [0, "demo.py"],
      [1, "scene"],
      [2, "body"],
      [2, "Union of bolt"],
      [3, "bolt"],
      [4, "extrude"],
      [3, "Mirror of bolt"],
      [4, "bolt"],
      [1, "Not in scene"],
      [2, "spare"],
    ]);
  });

  it("counts operands from the tree, not from the source text", () => {
    expect(byTest("boolean:scene").detail).toBe("Union · 2 operands");
    // The call is already in the label, so the detail does not repeat it.
    expect(byTest("boolean@7").detail).toBe("2 operands");
  });

  it("draws a second occurrence as a shared reference without children", () => {
    const occurrences = rows.filter((row) => row.nodeId === "profile_1");
    expect(occurrences).toHaveLength(2);
    expect(occurrences.map((row) => row.reference)).toEqual([false, true]);
    expect(occurrences.every((row) => row.uses === 2)).toBe(true);
    expect(occurrences[1]).toMatchObject({ group: false, detail: "shared" });
    // Unique test ids: the first is the node, the reference its path.
    expect(occurrences[0].testKey).toBe("profile_1");
    expect(occurrences[1].testKey).not.toBe("profile_1");
    expect(new Set(rows.map((row) => row.key)).size).toBe(rows.length);
  });

  it("points boolean rows at their element and item", () => {
    expect(byTest("boolean:scene")).toMatchObject({
      kind: "boolean",
      elementId: "boolean:scene",
      itemId: "boolean:scene",
      group: true,
    });
    expect(byTest("transform:Mirror")).toMatchObject({ kind: "transform", elementId: null });
  });

  it("collapses a boolean with everything under it", () => {
    const shown = visibleRows(rows, new Set([byTest("boolean@7").key]));
    expect(shown.map((row) => row.label)).toEqual([
      "demo.py",
      "scene",
      "body",
      "Union of bolt",
      "Not in scene",
      "spare",
    ]);
  });

  it("hints at up to two operands and elides the rest", () => {
    const wide: SceneTree = {
      roots: ["u"],
      items: [
        item("u", { kind: "boolean", call: "Union", operands: ["a", "b", "c"] }),
        item("a", { kind: "sdf", call: "Cylinder", name: "rail_left" }),
        item("b", { kind: "sdf", call: "Cylinder", name: "rail_right" }),
        item("c", { kind: "sdf", call: "Cylinder", name: "rail_mid" }),
      ],
    };
    const union = buildSceneTree([], [], "x.py", wide)[1];
    expect(union.label).toBe("Union of rail_left, rail_right…");
    expect(union.detail).toBe("3 operands");
  });

  it("falls back to the flat list without a tree", () => {
    const flat = buildSceneTree(NODES, ELEMENTS, "demo.py", null);
    expect(flat.filter((row) => row.depth === 1).map((row) => row.label)).toEqual([
      "body",
      "bolt",
      "spare",
      "scene",
    ]);
  });
});

describe("descendantNodes", () => {
  const TREE: SceneTree = {
    roots: ["u"],
    items: [
      item("u", { kind: "boolean", call: "Union", operands: ["a", "m"] }),
      item("a", { node: "profile_0" }),
      item("m", { kind: "transform", call: "Mirror", operands: ["a", "b"] }),
      item("b", { node: "box_1" }),
    ],
  };

  it("collects every construction node under an item, once", () => {
    expect([...descendantNodes(TREE, "u")].sort()).toEqual(["box_1", "profile_0"]);
    expect([...descendantNodes(TREE, "a")]).toEqual(["profile_0"]);
  });

  it("is empty without a tree or for an unknown item", () => {
    expect(descendantNodes(null, "u").size).toBe(0);
    expect(descendantNodes(TREE, "nope").size).toBe(0);
  });
});
