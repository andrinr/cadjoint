import { describe, expect, it } from "vitest";
import {
  CONSTRAINT_TOOL_NAMES,
  EDGE_CONSTRAINT_TOOLS,
  VERTEX_CONSTRAINT_TOOLS,
  constraintLabel,
  edgeVertexIndices,
  isEdgeConstraintTool,
  isVertexConstraintTool,
} from "../src/constraints";

describe("constraintLabel", () => {
  it("labels each constraint kind with 1-based point names", () => {
    expect(constraintLabel({ kind: "fixed", vertices: [0], value: [1, 2] })).toBe(
      "fix · P1",
    );
    expect(
      constraintLabel({ kind: "horizontal", vertices: [0, 1], value: null }),
    ).toBe("horiz · P1–P2");
    expect(
      constraintLabel({ kind: "vertical", vertices: [2, 3], value: null }),
    ).toBe("vert · P3–P4");
    expect(
      constraintLabel({ kind: "coincident", vertices: [1, 4], value: null }),
    ).toBe("coinc · P2–P5");
    expect(
      constraintLabel({ kind: "equal_length", vertices: [0, 1, 2, 3], value: null }),
    ).toBe("equal · P1P2–P3P4");
    expect(
      constraintLabel({ kind: "parallel", vertices: [0, 1, 2, 3], value: null }),
    ).toBe("∥ · P1P2–P3P4");
    expect(
      constraintLabel({
        kind: "perpendicular",
        vertices: [1, 2, 3, 0],
        value: null,
      }),
    ).toBe("⊥ · P2P3–P4P1");
  });

  it("shows a distance constraint's target value", () => {
    expect(
      constraintLabel({ kind: "distance", vertices: [0, 1], value: 1.25 }),
    ).toBe("distance · P1–P2 = 1.25");
    // A distance whose value could not be resolved still gets a label.
    expect(
      constraintLabel({ kind: "distance", vertices: [0, 1], value: null }),
    ).toBe("distance · P1–P2");
  });
});

describe("edgeVertexIndices", () => {
  it("maps an insertion index to the picked edge's endpoints", () => {
    expect(edgeVertexIndices(1, 4)).toEqual([0, 1]);
    expect(edgeVertexIndices(3, 4)).toEqual([2, 3]);
  });

  it("wraps the closing edge of the loop back to vertex 0", () => {
    expect(edgeVertexIndices(4, 4)).toEqual([3, 0]);
  });
});

describe("constraint tool classification", () => {
  it("splits the tools into vertex-pair and edge-pair flows", () => {
    for (const tool of VERTEX_CONSTRAINT_TOOLS) {
      expect(isVertexConstraintTool(tool)).toBe(true);
      expect(isEdgeConstraintTool(tool)).toBe(false);
      expect(CONSTRAINT_TOOL_NAMES[tool]).toBeTruthy();
    }
    for (const tool of EDGE_CONSTRAINT_TOOLS) {
      expect(isEdgeConstraintTool(tool)).toBe(true);
      expect(isVertexConstraintTool(tool)).toBe(false);
      expect(CONSTRAINT_TOOL_NAMES[tool]).toBeTruthy();
    }
    expect(isVertexConstraintTool("select")).toBe(false);
    expect(isEdgeConstraintTool("polygon")).toBe(false);
  });
});
