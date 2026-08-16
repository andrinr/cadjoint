import { describe, expect, it } from "vitest";
import { loftPickError, type PendingLoft } from "../src/loft";
import type { ConstructionNode } from "../src/types";

const pending: PendingLoft = { nodeId: "profile_0", line: 3 };

function sketch(
  overrides: Partial<
    Pick<ConstructionNode, "id" | "kind" | "editable" | "line" | "operators">
  > = {},
) {
  return {
    id: "profile_1",
    kind: "profile" as const,
    editable: true,
    line: 7,
    operators: [],
    ...overrides,
  };
}

describe("loftPickError", () => {
  it("accepts a second editable sketch without an operator", () => {
    expect(loftPickError(pending, sketch())).toBeNull();
  });

  it("rejects a miss or a non-sketch node", () => {
    expect(loftPickError(pending, null)).toMatch(/second sketch/);
    expect(loftPickError(pending, sketch({ kind: "box" }))).toMatch(
      /second sketch/,
    );
  });

  it("rejects picking the armed sketch again", () => {
    expect(loftPickError(pending, sketch({ id: "profile_0" }))).toMatch(
      /different sketch/,
    );
  });

  it("rejects sketches that cannot be edited from source", () => {
    expect(loftPickError(pending, sketch({ editable: false }))).toMatch(
      /cannot be edited/,
    );
    expect(loftPickError(pending, sketch({ line: null }))).toMatch(
      /cannot be edited/,
    );
  });

  it("rejects a sketch that already drives an operation", () => {
    expect(
      loftPickError(
        pending,
        sketch({ operators: [{ kind: "extrude", line: 9 }] }),
      ),
    ).toMatch(/already drives/);
  });
});
