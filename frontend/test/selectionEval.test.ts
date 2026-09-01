import { describe, expect, it } from "vitest";
import {
  BC_TYPE_COLORS,
  defaultSideTol,
  evaluateSelection,
  overlayColors,
  selectionEvaluable,
} from "../src/selectionEval";
import type { StudySelection } from "../src/types";

/** 2×2×2 cube corners, a tiny stand-in for a boundary vertex set. */
const CORNERS: [number, number, number][] = [
  [-1, -1, -1],
  [1, -1, -1],
  [1, 1, -1],
  [-1, 1, -1],
  [-1, -1, 1],
  [1, -1, 1],
  [1, 1, 1],
  [-1, 1, 1],
];
const positions = CORNERS.flat();
const grid = { spacing: [0.5, 0.25, 0.5] };

const evaluate = (selection: StudySelection) =>
  evaluateSelection(selection, positions, grid);

describe("selectionEvaluable", () => {
  it("rejects predicates anywhere in the tree", () => {
    expect(selectionEvaluable({ kind: "predicate", name: "custom" })).toBe(false);
    expect(
      selectionEvaluable({
        kind: "and",
        operands: [
          { kind: "side", side: "+x", tol: null },
          { kind: "not", operand: { kind: "predicate", name: "custom" } },
        ],
      }),
    ).toBe(false);
    expect(selectionEvaluable({ kind: "sphere", center: [0, 0, 0], radius: 1 })).toBe(true);
  });

  it("returns null instead of a mask for predicate trees", () => {
    expect(evaluate({ kind: "predicate", name: "custom" })).toBeNull();
  });
});

describe("primitive selections", () => {
  it("box is inclusive on its faces", () => {
    const mask = evaluate({
      kind: "box",
      min_corner: [0, -1, -1],
      max_corner: [1, 1, 1],
    })!;
    // Exactly the four +x corners (x = 1) qualify; x = -1 does not.
    expect(mask).toEqual([false, true, true, false, false, true, true, false]);
  });

  it("sphere selects by squared distance, boundary inclusive", () => {
    const radius = Math.sqrt(3);
    const mask = evaluate({ kind: "sphere", center: [1, 1, 1], radius })!;
    // Corner [1,1,1] at distance 0; adjacent corners at 2; opposite at 2√3.
    expect(mask[6]).toBe(true);
    expect(mask[0]).toBe(false);
    expect(mask[2]).toBe(false); // distance 2 > √3
  });

  it("halfspace keeps the plane itself (>= 0)", () => {
    const mask = evaluate({
      kind: "halfspace",
      point: [1, 0, 0],
      normal: [1, 0, 0],
    })!;
    expect(mask).toEqual([false, true, true, false, false, true, true, false]);
  });
});

describe("side selections", () => {
  it("uses half the min cell spacing as the default tolerance", () => {
    expect(defaultSideTol(positions, grid)).toBeCloseTo(0.125);
    // Without a grid: 1e-3 of the bbox diagonal.
    expect(defaultSideTol(positions, null)).toBeCloseTo(1e-3 * Math.sqrt(12));
  });

  it("selects the extreme plane within tolerance", () => {
    const mask = evaluate({ kind: "side", side: "+z", tol: null })!;
    expect(mask).toEqual([false, false, false, false, true, true, true, true]);
    const negative = evaluate({ kind: "side", side: "-y", tol: null })!;
    expect(negative).toEqual([true, true, false, false, true, true, false, false]);
  });

  it("honours an explicit tolerance wide enough to catch both planes", () => {
    const mask = evaluate({ kind: "side", side: "+x", tol: 2 })!;
    expect(mask.every(Boolean)).toBe(true);
  });
});

describe("composite selections", () => {
  it("intersects, unions, and complements", () => {
    const posX: StudySelection = { kind: "side", side: "+x", tol: null };
    const posZ: StudySelection = { kind: "side", side: "+z", tol: null };
    const both = evaluate({ kind: "and", operands: [posX, posZ] })!;
    expect(both).toEqual([false, false, false, false, false, true, true, false]);
    const either = evaluate({ kind: "or", operands: [posX, posZ] })!;
    expect(either).toEqual([false, true, true, false, true, true, true, true]);
    const rest = evaluate({ kind: "not", operand: posX })!;
    expect(rest).toEqual([true, false, false, true, true, false, false, true]);
  });
});

describe("overlayColors", () => {
  it("tints masked vertices and lets later layers win overlaps", () => {
    const colors = overlayColors(3, [
      { mask: [true, true, false], color: [1, 0, 0] },
      { mask: [false, true, false], color: [0, 1, 0] },
    ]);
    // Vertex 0: red; vertex 1: overwritten green; vertex 2: untouched.
    expect(colors[0]).toBe(1);
    expect(colors[3]).toBeGreaterThan(0); // strength set
    expect(colors[4]).toBe(0);
    expect(colors[5]).toBe(1);
    expect(colors[8]).toBe(0);
    expect(colors[11]).toBe(0); // no strength → no tint
  });

  it("has a distinct hue per BC type", () => {
    const hues = Object.values(BC_TYPE_COLORS).map((color) => color.join(","));
    expect(new Set(hues).size).toBe(hues.length);
  });
});
