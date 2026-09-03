/**
 * Which characters a selection reveals in the editor.
 *
 * The bug this pins is a silent one: selecting a sketch used to reveal
 * nothing, because the memo asked for `spans.position` and a profile has no
 * argument spans at all. Nothing happened, no error, and sketches are exactly
 * what a user selects while placing one. So the cases below are mostly about
 * *always returning something* — and about the precision flag, which is what
 * stops a twenty-two-line declaration from being painted over whole.
 */

import { describe, expect, it } from "vitest";
import { focusSpan } from "../src/editorFocus";
import type { ConstructionNode } from "../src/types";

const sketch = {
  id: "profile_0",
  kind: "profile",
  spans: {},
  statementSpan: [100, 420],
  vertices: [
    { span: [140, 152] },
    { span: null },
  ],
} as unknown as ConstructionNode;

const primitive = {
  id: "box_1",
  kind: "box",
  spans: { position: [60, 78], size: [30, 50] },
  statementSpan: [10, 95],
  vertices: [],
} as unknown as ConstructionNode;

describe("focusSpan", () => {
  it("reveals a sketch's declaration, which used to reveal nothing", () => {
    expect(focusSpan(sketch, null)).toEqual({ from: 100, to: 420, precise: false });
  });

  it("reveals a primitive's declaration rather than its position literal", () => {
    // The old rule answered [60, 78] here — three numbers, not the object.
    expect(focusSpan(primitive, null)).toEqual({ from: 10, to: 95, precise: false });
  });

  it("keeps a vertex on its own literal, precisely", () => {
    expect(focusSpan(sketch, 0)).toEqual({ from: 140, to: 152, precise: true });
  });

  it("falls back to the declaration for a vertex the source cannot pin", () => {
    // A sketch built in a loop has no per-vertex spans; showing the sketch is
    // better than refusing to move.
    expect(focusSpan(sketch, 1)).toEqual({ from: 100, to: 420, precise: false });
  });

  it("falls back to the position literal when there is no statement span", () => {
    const older = { ...primitive, statementSpan: null } as unknown as ConstructionNode;
    expect(focusSpan(older, null)).toEqual({ from: 60, to: 78, precise: true });
  });

  it("returns null when nothing can be shown", () => {
    expect(focusSpan(undefined, null)).toBeNull();
    const unplaceable = {
      spans: {},
      statementSpan: null,
      vertices: [],
    } as unknown as ConstructionNode;
    expect(focusSpan(unplaceable, null)).toBeNull();
    expect(focusSpan(unplaceable, 0)).toBeNull();
  });
});
