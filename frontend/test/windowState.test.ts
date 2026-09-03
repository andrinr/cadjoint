/**
 * The window state machine.
 *
 * The dock library owns geometry; these three states — open, parked, closed —
 * are ours, and every rule about them is asserted here rather than discovered
 * by dragging things around in a browser.
 */

import { describe, expect, it } from "vitest";
import {
  defaultWindowsForMode,
  DEFAULT_LAYOUTS,
  isPermanent,
  isWindowId,
  windowTitle,
  WINDOW_IDS,
} from "../src/windows/panels";
import {
  defaultModeWindows,
  defaultWindowStates,
  minimisedWindows,
  reduceModeWindows,
  reopenableWindows,
} from "../src/windows/windowState";

describe("window definitions", () => {
  it("gives every mode a default layout that starts with the viewport", () => {
    for (const mode of ["model", "sketch", "simulate"] as const) {
      expect(DEFAULT_LAYOUTS[mode][0].id).toBe("viewport");
      expect(defaultWindowsForMode(mode)).toContain("viewport");
      expect(defaultWindowsForMode(mode)).toContain("editor");
    }
  });

  it("only ever attaches a window to one already placed before it", () => {
    for (const mode of ["model", "sketch", "simulate"] as const) {
      const placed = new Set<string>();
      for (const step of DEFAULT_LAYOUTS[mode]) {
        if (step.reference) expect(placed.has(step.reference)).toBe(true);
        placed.add(step.id);
      }
    }
  });

  it("treats the viewport, and only the viewport, as permanent", () => {
    expect(isPermanent("viewport")).toBe(true);
    for (const id of WINDOW_IDS) {
      if (id !== "viewport") expect(isPermanent(id)).toBe(false);
    }
  });

  it("rejects ids it does not know", () => {
    expect(isWindowId("objects")).toBe(true);
    expect(isWindowId("console")).toBe(false);
    expect(windowTitle("objects")).toBe("Objects");
  });
});

describe("default statuses", () => {
  it("opens a mode's own windows and closes the rest", () => {
    const model = defaultModeWindows("model");
    expect(model.objects).toBe("open");
    expect(model.materials).toBe("open");
    expect(model.studies).toBe("closed");
    expect(model.results).toBe("closed");
    expect(model.sketch).toBe("closed");

    // Simulate is a desk, not a window: what it opens is the four simulation
    // windows, and Optimize is the one shared by both desks.
    const simulate = defaultModeWindows("simulate");
    expect(simulate.studies).toBe("open");
    expect(simulate.meshes).toBe("open");
    expect(simulate.results).toBe("open");
    expect(simulate.optimize).toBe("open");
    expect(simulate.materials).toBe("closed");
  });

  it("gives every mode its own record", () => {
    const states = defaultWindowStates();
    expect(Object.keys(states).sort()).toEqual(["model", "simulate", "sketch"]);
    // Optimize is one window with two homes: the Model desk tabs it behind
    // Materials, the Simulate desk behind Results.
    expect(states.model.optimize).toBe("open");
    expect(states.sketch.optimize).toBe("closed");
  });
});

describe("transitions", () => {
  const model = defaultModeWindows("model");

  it("closes an open window and reopens it", () => {
    const closed = reduceModeWindows(model, { kind: "close", id: "materials" });
    expect(closed.materials).toBe("closed");
    expect(reduceModeWindows(closed, { kind: "open", id: "materials" }).materials).toBe("open");
  });

  it("parks an open window and restores it", () => {
    const parked = reduceModeWindows(model, { kind: "minimise", id: "objects" });
    expect(parked.objects).toBe("minimised");
    // Scenes and Processes are parked in every desk by default — a document
    // browser and a monitor, both worth one click away and neither worth a
    // column of the desk before it was asked for.
    expect(minimisedWindows(parked)).toEqual(["objects", "scenes", "processes"]);
    expect(reduceModeWindows(parked, { kind: "restore", id: "objects" }).objects).toBe("open");
  });

  it("has nothing to park when a window is already closed", () => {
    const closed = reduceModeWindows(model, { kind: "close", id: "materials" });
    const parked = reduceModeWindows(closed, { kind: "minimise", id: "materials" });
    expect(parked.materials).toBe("closed");
    expect(parked).toBe(closed);
  });

  it("refuses to close or park the viewport", () => {
    expect(reduceModeWindows(model, { kind: "close", id: "viewport" })).toBe(model);
    expect(reduceModeWindows(model, { kind: "minimise", id: "viewport" })).toBe(model);
    expect(reduceModeWindows(model, { kind: "toggle", id: "viewport" })).toBe(model);
  });

  it("toggles from either not-open state back to open", () => {
    const closed = reduceModeWindows(model, { kind: "close", id: "materials" });
    expect(reduceModeWindows(closed, { kind: "toggle", id: "materials" }).materials).toBe("open");
    const parked = reduceModeWindows(model, { kind: "minimise", id: "materials" });
    expect(reduceModeWindows(parked, { kind: "toggle", id: "materials" }).materials).toBe("open");
    expect(reduceModeWindows(model, { kind: "toggle", id: "materials" }).materials).toBe("closed");
  });

  it("returns the same object when nothing changes, so callers can skip a write", () => {
    expect(reduceModeWindows(model, { kind: "open", id: "objects" })).toBe(model);
  });

  it("resets a mode back to its default desk", () => {
    const messed = reduceModeWindows(
      reduceModeWindows(model, { kind: "close", id: "objects" }),
      { kind: "minimise", id: "materials" },
    );
    expect(reduceModeWindows(messed, { kind: "reset", mode: "model" })).toEqual(model);
  });

  it("lists everything the Window menu can reopen", () => {
    const messed = reduceModeWindows(
      reduceModeWindows(model, { kind: "close", id: "objects" }),
      { kind: "minimise", id: "materials" },
    );
    expect(reopenableWindows(messed)).toContain("objects");
    expect(reopenableWindows(messed)).toContain("materials");
    expect(reopenableWindows(messed)).not.toContain("viewport");
  });
});
