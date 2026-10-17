import { describe, expect, it } from "vitest";
import { removalLabel, type RemovalLookups } from "../src/shell/removal";

const lookups: RemovalLookups = {
  nodes: [{ id: "n1", stableId: "s1", line: 12, name: "bush_a", kind: "cylinder" }],
  studies: [{ index: 0, name: "sink-conduction" }],
  meshes: [{ index: 0, name: "sink-mesh" }],
  optimizations: [{ index: 0, name: "cool-sink" }],
};

describe("what a removal says", () => {
  it("names a removed object in the program's words", () => {
    expect(removalLabel({ op: "delete_object", id: "s1", line: 12 }, lookups)).toBe("removed bush_a");
    expect(removalLabel({ op: "delete_object", line: 12 }, lookups)).toBe("removed bush_a");
  });

  it("names studies, meshes and optimizations by index", () => {
    expect(removalLabel({ op: "delete_study", study: 0 }, lookups)).toBe("removed sink-conduction");
    expect(removalLabel({ op: "delete_mesh", mesh: 0 }, lookups)).toBe("removed sink-mesh");
    expect(removalLabel({ op: "delete_optimization", optimization: 0 }, lookups)).toBe("removed cool-sink");
  });

  it("says what kind when there is no name", () => {
    expect(removalLabel({ op: "delete_constraint", line: 3, index: 1 }, lookups)).toBe("removed a constraint");
    expect(removalLabel({ op: "delete_study_bc", study: 0, bc: 1 }, lookups)).toBe("removed a boundary condition");
  });

  it("is silent for anything that is not a removal", () => {
    expect(removalLabel({ op: "set_value", line: 3 }, lookups)).toBeNull();
  });
});
