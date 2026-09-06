import { describe, expect, it } from "vitest";
import {
  elementForNode,
  elementForOperator,
  elementTitle,
  formatArgument,
  kindLabel,
  resolveSections,
  runtimePlaneRows,
} from "../src/properties";
import { buildSceneTree } from "../src/objectTree";
import type { ConstructionArgument, ConstructionElement, ConstructionNode } from "../src/types";

const argument = (overrides: Partial<ConstructionArgument>): ConstructionArgument => ({
  name: "depth",
  kind: "number",
  value: 0.5,
  text: "0.5",
  span: [10, 13],
  parameter: null,
  patch: { op: "set_value", name: "extrude", argument: "depth", line: 4, id: "assign:body" },
  ...overrides,
});

const element = (overrides: Partial<ConstructionElement>): ConstructionElement => ({
  id: "assign:body",
  stableId: "assign:body",
  kind: "feature",
  call: "extrude",
  line: 4,
  span: [0, 10],
  name: null,
  variable: "body",
  owner: null,
  arguments: [],
  ...overrides,
});

const node = (overrides: Partial<ConstructionNode>): ConstructionNode => ({
  id: "profile_0",
  stableId: "assign:sketch",
  kind: "profile",
  name: "s",
  line: 3,
  editable: true,
  edges: [],
  plane: {
    origin: [1, 2, 3],
    u: [1, 0, 0],
    v: [0, 1, 0],
    normal: [0, 0, 1],
    stableId: "plane:sketch",
    reference: null,
  },
  faces: [],
  vertices: [],
  transform: null,
  spans: {},
  constraints: [],
  operators: [{ kind: "extrude", line: 4 }],
  material: null,
  ...overrides,
});

const SKETCH = element({ id: "assign:sketch", stableId: "assign:sketch", kind: "sketch", call: "PolygonProfile", line: 3, name: "s", variable: "sketch" });
const PLANE = element({ id: "plane:sketch", stableId: "plane:sketch", kind: "plane", call: "SketchPlane", line: 3, owner: "assign:sketch", variable: null });
const BODY = element({});
const BLOCK = element({ id: "assign:block", stableId: "assign:block", kind: "primitive", call: "box", line: 6, variable: "block", name: "block" });
const SCENE = element({
  id: "boolean:scene",
  stableId: null,
  kind: "boolean",
  call: "Union",
  line: 7,
  variable: "scene",
  arguments: [argument({ name: "operands", kind: "reference", value: "body, block", text: "body, block", patch: null })],
});
const ELEMENTS = [SKETCH, PLANE, BODY, BLOCK, SCENE];
const NODES = [node({}), node({ id: "box_1", stableId: "assign:block", kind: "box", line: 6, plane: null, operators: [] })];

describe("resolveSections", () => {
  it("shows nothing with nothing selected", () => {
    expect(resolveSections(ELEMENTS, NODES, null, null)).toEqual([]);
  });

  it("shows a sketch with its plane under it", () => {
    const sections = resolveSections(ELEMENTS, NODES, { nodeId: "profile_0", vertexIndex: null }, null);
    expect(sections.map((item) => item.id)).toEqual(["assign:sketch", "plane:sketch"]);
  });

  it("puts the plane first when the plane was taken hold of", () => {
    const sections = resolveSections(
      ELEMENTS,
      NODES,
      { nodeId: "profile_0", vertexIndex: null, part: "plane" },
      null,
    );
    expect(sections.map((item) => item.id)).toEqual(["plane:sketch", "assign:sketch"]);
  });

  it("shows a primitive on its own", () => {
    const sections = resolveSections(ELEMENTS, NODES, { nodeId: "box_1", vertexIndex: null }, null);
    expect(sections.map((item) => item.id)).toEqual(["assign:block"]);
  });

  it("lets an inspected element win over the selection", () => {
    const sections = resolveSections(
      ELEMENTS,
      NODES,
      { nodeId: "profile_0", vertexIndex: null },
      "assign:body",
    );
    expect(sections.map((item) => item.id)).toEqual(["assign:body"]);
  });

  it("falls back to the selection when the inspected element is gone", () => {
    const sections = resolveSections(
      ELEMENTS,
      NODES,
      { nodeId: "box_1", vertexIndex: null },
      "boolean:vanished",
    );
    expect(sections.map((item) => item.id)).toEqual(["assign:block"]);
  });

  it("matches a node to its element by line when it has no stable id", () => {
    const anonymous = node({ stableId: null, line: 3 });
    expect(elementForNode(ELEMENTS, anonymous)?.id).toBe("assign:sketch");
    expect(elementForNode(ELEMENTS, undefined)).toBeNull();
  });

  it("finds the feature an operator chip names", () => {
    expect(elementForOperator(ELEMENTS, "extrude", 4)?.id).toBe("assign:body");
    expect(elementForOperator(ELEMENTS, "revolve", 4)).toBeNull();
  });
});

describe("labels", () => {
  it("names kinds and titles the way the tree does", () => {
    expect(kindLabel(SKETCH)).toBe("Sketch");
    expect(kindLabel(PLANE)).toBe("Sketch plane");
    expect(kindLabel(element({ ...PLANE, call: "SketchPlane.on" }))).toBe("Derived sketch plane");
    expect(kindLabel(BLOCK)).toBe("Box");
    expect(kindLabel(BODY)).toBe("Extrude");
    expect(kindLabel(SCENE)).toBe("Union");
    expect(elementTitle(SKETCH)).toBe("s");
    expect(elementTitle(BODY)).toBe("body");
    expect(elementTitle(element({ name: null, variable: null }))).toBe("extrude");
  });

  it("formats values compactly and expressions as written", () => {
    expect(formatArgument(argument({ value: 0.30000000000000004 }))).toBe("0.3");
    expect(formatArgument(argument({ kind: "vector", value: [1, 0.5, -0] }))).toBe("[1, 0.5, 0]");
    expect(formatArgument(argument({ kind: "reference", value: "steel" }))).toBe("steel");
    expect(
      formatArgument(argument({ kind: "expression", value: null, text: "SketchPlane.on(a.cap('+'))" })),
    ).toBe("SketchPlane.on(a.cap('+'))");
    expect(formatArgument(argument({ kind: "expression", value: "16 points", text: "[...]" }))).toBe(
      "16 points",
    );
  });
});

describe("runtimePlaneRows", () => {
  it("fills origin and normal from the compiled frame for a derived plane", () => {
    const derived = element({
      ...PLANE,
      call: "SketchPlane.on",
      arguments: [argument({ name: "reference", kind: "expression", value: null, patch: null })],
    });
    expect(runtimePlaneRows(derived, NODES)).toEqual([
      { name: "origin", text: "[1, 2, 3]" },
      { name: "normal", text: "[0, 0, 1]" },
    ]);
  });

  it("adds nothing when the plane states both", () => {
    const stated = element({
      ...PLANE,
      arguments: [
        argument({ name: "origin", kind: "vector", value: [0, 0, 0] }),
        argument({ name: "normal", kind: "vector", value: [0, 0, 1] }),
      ],
    });
    expect(runtimePlaneRows(stated, NODES)).toEqual([]);
    expect(runtimePlaneRows(BODY, NODES)).toEqual([]);
  });
});

describe("the object tree with elements", () => {
  it("points operator rows at their feature and lists booleans", () => {
    const rows = buildSceneTree(NODES, ELEMENTS);
    const operator = rows.find((row) => row.kind === "operator")!;
    expect(operator.elementId).toBe("assign:body");
    const boolean = rows.find((row) => row.kind === "boolean")!;
    expect(boolean.elementId).toBe("boolean:scene");
    expect(boolean.label).toBe("scene");
    expect(boolean.detail).toBe("Union · 2 operands");
    expect(rows[0].detail).toBe("3 objects");
  });

  it("still builds without elements", () => {
    const rows = buildSceneTree(NODES);
    expect(rows.find((row) => row.kind === "operator")!.elementId).toBeNull();
    expect(rows.some((row) => row.kind === "boolean")).toBe(false);
  });
});
