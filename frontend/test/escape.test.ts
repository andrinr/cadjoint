/**
 * Escape backs out one step at a time.
 *
 * The old handler cleared the pending pick, the pending loft, the probe, the
 * selection, the tool and the mode in one statement, so a single press threw
 * the user to the top from wherever they were. These cases pin the ladder:
 * each press names exactly one rung, and the rungs below it are untouched.
 */

import { describe, expect, it } from "vitest";
import { escapeLevel, type EscapeState } from "../src/components/viewer/escape";

const nothing: EscapeState = {
  gesture: false,
  pendingConstraint: false,
  pendingLoft: false,
  toolArmed: false,
  bcPickArmed: false,
  selection: false,
  simProbe: false,
  awayFromModel: false,
};

/** Every rung set at once — the state a single press used to flatten. */
const everything: EscapeState = {
  gesture: true,
  pendingConstraint: true,
  pendingLoft: true,
  toolArmed: true,
  bcPickArmed: true,
  selection: true,
  simProbe: true,
  awayFromModel: true,
};

describe("escapeLevel", () => {
  it("does nothing when there is nothing to cancel", () => {
    expect(escapeLevel(nothing)).toBeNull();
  });

  it("cancels a gesture before anything else", () => {
    expect(escapeLevel(everything)).toBe("gesture");
  });

  it("walks down one rung per press", () => {
    // Simulate repeated presses by clearing exactly what each rung owns.
    const state = { ...everything };
    expect(escapeLevel(state)).toBe("gesture");
    state.gesture = false;
    expect(escapeLevel(state)).toBe("pending");
    state.pendingConstraint = false;
    state.pendingLoft = false;
    expect(escapeLevel(state)).toBe("tool");
    state.toolArmed = false;
    state.bcPickArmed = false;
    expect(escapeLevel(state)).toBe("selection");
    state.selection = false;
    state.simProbe = false;
    expect(escapeLevel(state)).toBe("mode");
    state.awayFromModel = false;
    expect(escapeLevel(state)).toBeNull();
  });

  it("treats either half-finished command as the same rung", () => {
    expect(escapeLevel({ ...nothing, pendingConstraint: true })).toBe("pending");
    expect(escapeLevel({ ...nothing, pendingLoft: true })).toBe("pending");
  });

  it("treats an armed BC pick as an armed tool", () => {
    expect(escapeLevel({ ...nothing, bcPickArmed: true })).toBe("tool");
    expect(escapeLevel({ ...nothing, toolArmed: true })).toBe("tool");
  });

  it("treats a probe readout as a selection", () => {
    expect(escapeLevel({ ...nothing, simProbe: true })).toBe("selection");
  });

  it("keeps the hint bar's promise once nothing else is pending", () => {
    // "Esc returns to model" is still true — it is just no longer the first
    // thing Escape does.
    expect(escapeLevel({ ...nothing, awayFromModel: true })).toBe("mode");
  });

  it("does not skip a rung when a lower one is also set", () => {
    expect(escapeLevel({ ...nothing, selection: true, awayFromModel: true })).toBe(
      "selection",
    );
    expect(escapeLevel({ ...nothing, toolArmed: true, selection: true })).toBe("tool");
  });
});
