import { describe, expect, it } from "vitest";

import { minimalChange } from "../src/editor/minimalChange";

const apply = (text: string, change: { from: number; to: number; insert: string }) =>
  text.slice(0, change.from) + change.insert + text.slice(change.to);

describe("minimalChange", () => {
  it("is null for equal texts", () => {
    expect(minimalChange("a = 1\n", "a = 1\n")).toBeNull();
  });

  it("touches only the literal a patch rewrote", () => {
    const before = "x = Vector2(value=[0.85, 1.2], free=True)\ny = 2\n";
    const after = "x = Vector2(value=[0.7912, 1.3305], free=True)\ny = 2\n";
    const change = minimalChange(before, after)!;
    expect(before.slice(change.from, change.to)).toBe("85, 1.2");
    expect(change.insert).toBe("7912, 1.3305");
    expect(apply(before, change)).toBe(after);
  });

  it("round-trips insertions, deletions and a total rewrite", () => {
    for (const [before, after] of [
      ["", "abc"],
      ["abc", ""],
      ["abc", "abXc"],
      ["abXc", "abc"],
      ["aaa", "aaaa"],
      ["aaaa", "aaa"],
      ["one\ntwo\n", "three\n"],
    ]) {
      const change = minimalChange(before, after);
      expect(change === null ? before : apply(before, change)).toBe(after);
    }
  });
});
