import { describe, expect, it } from "vitest";
import {
  EDITING_MODES,
  autoEnterSketchMode,
  createToolVisibleInMode,
  cycleEditingMode,
  groupVisibleInMode,
  loadEditingMode,
  persistEditingMode,
  railGroupsForMode,
} from "../src/editingMode";
import type { ConstructionNode } from "../src/types";

function memoryStorage(initial: string | null = null) {
  let value = initial;
  return {
    getItem: () => value,
    setItem: (_key: string, next: string) => {
      value = next;
    },
    value: () => value,
  };
}

const profileNode = (id: string): ConstructionNode =>
  ({ id, kind: "profile" }) as ConstructionNode;
const boxNode = (id: string): ConstructionNode =>
  ({ id, kind: "box" }) as ConstructionNode;

describe("editing mode cycling", () => {
  it("cycles forward through model, sketch, simulate, render and wraps", () => {
    expect(cycleEditingMode("model")).toBe("sketch");
    expect(cycleEditingMode("sketch")).toBe("simulate");
    expect(cycleEditingMode("simulate")).toBe("render");
    expect(cycleEditingMode("render")).toBe("model");
  });

  it("cycles backwards and wraps the other way", () => {
    expect(cycleEditingMode("model", -1)).toBe("render");
    expect(cycleEditingMode("render", -1)).toBe("simulate");
    expect(cycleEditingMode("simulate", -1)).toBe("sketch");
  });

  it("visits each mode exactly once per full cycle", () => {
    const seen: string[] = [];
    let mode = EDITING_MODES[0];
    for (let index = 0; index < EDITING_MODES.length; index++) {
      seen.push(mode);
      mode = cycleEditingMode(mode);
    }
    expect(new Set(seen).size).toBe(EDITING_MODES.length);
    expect(mode).toBe(EDITING_MODES[0]);
  });
});

describe("mode filtering of tool groups", () => {
  it("model mode offers solids and transforms but no sketch-only groups", () => {
    const groups = railGroupsForMode("model");
    expect(groups).toContain("create");
    expect(groups).toContain("transform");
    expect(groups).not.toContain("annotate");
    expect(groups).not.toContain("modify");
  });

  it("sketch mode adds constraint and operator groups", () => {
    const groups = railGroupsForMode("sketch");
    expect(groups).toContain("annotate");
    expect(groups).toContain("modify");
    expect(groups).toContain("select");
  });

  it("simulate and render modes hide every tool group", () => {
    expect(railGroupsForMode("simulate")).toEqual([]);
    expect(groupVisibleInMode("delete", "simulate")).toBe(false);
    expect(railGroupsForMode("render")).toEqual([]);
    expect(groupVisibleInMode("select", "render")).toBe(false);
  });

  it("filters create tools per mode: solids are model-only", () => {
    expect(createToolVisibleInMode("box", "model")).toBe(true);
    expect(createToolVisibleInMode("box", "sketch")).toBe(false);
    expect(createToolVisibleInMode("sketch", "sketch")).toBe(true);
    expect(createToolVisibleInMode("polygon", "model")).toBe(true);
  });
});

describe("sketch-mode auto-entry", () => {
  it("enters sketch mode when a profile is newly selected", () => {
    const next = autoEnterSketchMode("model", profileNode("s1"), null);
    expect(next.mode).toBe("sketch");
    expect(next.lastAutoNodeId).toBe("s1");
  });

  it("is cancelable: the same profile does not re-enter after a manual exit", () => {
    const entered = autoEnterSketchMode("model", profileNode("s1"), null);
    // The user pressed Escape / switched to model; s1 is still selected.
    const afterExit = autoEnterSketchMode("model", profileNode("s1"), entered.lastAutoNodeId);
    expect(afterExit.mode).toBe("model");
  });

  it("re-arms once the selection leaves the sketch", () => {
    const cleared = autoEnterSketchMode("model", null, "s1");
    expect(cleared.lastAutoNodeId).toBeNull();
    const reentered = autoEnterSketchMode("model", profileNode("s1"), cleared.lastAutoNodeId);
    expect(reentered.mode).toBe("sketch");
  });

  it("selecting a solid never changes the mode", () => {
    const next = autoEnterSketchMode("model", boxNode("b1"), null);
    expect(next.mode).toBe("model");
    expect(next.lastAutoNodeId).toBeNull();
  });

  it("never hijacks simulate or render mode", () => {
    const simulate = autoEnterSketchMode("simulate", profileNode("s1"), null);
    expect(simulate.mode).toBe("simulate");
    const render = autoEnterSketchMode("render", profileNode("s1"), null);
    expect(render.mode).toBe("render");
  });
});

describe("editing mode persistence", () => {
  it("round-trips through storage", () => {
    const storage = memoryStorage();
    persistEditingMode("sketch", storage);
    expect(loadEditingMode(storage)).toBe("sketch");
    persistEditingMode("render", storage);
    expect(loadEditingMode(storage)).toBe("render");
  });

  it("falls back to model for unknown or missing values", () => {
    expect(loadEditingMode(memoryStorage())).toBe("model");
    expect(loadEditingMode(memoryStorage("banana"))).toBe("model");
    expect(loadEditingMode(undefined)).toBe("model");
  });
});
