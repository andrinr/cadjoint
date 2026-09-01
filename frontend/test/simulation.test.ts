import { describe, expect, it } from "vitest";
import {
  DEFAULT_SLICE,
  formatScalar,
  meshBounds,
  rampCss,
  slicePlane,
  viridis,
} from "../src/simulation";

describe("viridis ramp", () => {
  it("matches the colormap's endpoints and midpoint", () => {
    const [r0, g0, b0] = viridis(0);
    expect(r0).toBeCloseTo(0.267, 1);
    expect(g0).toBeCloseTo(0.005, 1);
    expect(b0).toBeCloseTo(0.329, 1);
    const [r1, g1, b1] = viridis(1);
    expect(r1).toBeCloseTo(0.993, 1);
    expect(g1).toBeCloseTo(0.906, 1);
    expect(b1).toBeCloseTo(0.144, 0);
    const [, gMid] = viridis(0.5);
    expect(gMid).toBeGreaterThan(0.4);
    expect(gMid).toBeLessThan(0.75);
  });

  it("clamps out-of-range inputs", () => {
    expect(viridis(-2)).toEqual(viridis(0));
    expect(viridis(3)).toEqual(viridis(1));
    for (const channel of viridis(0.999)) {
      expect(channel).toBeGreaterThanOrEqual(0);
      expect(channel).toBeLessThanOrEqual(1);
    }
  });

  it("builds a CSS gradient from the same ramp", () => {
    const css = rampCss(4);
    expect(css.startsWith("linear-gradient(to right, rgb(")).toBe(true);
    expect(css.match(/rgb\(/g)).toHaveLength(5);
  });
});

describe("legend formatting", () => {
  it("uses fixed digits for ordinary values and exponents outside", () => {
    expect(formatScalar(0)).toBe("0");
    expect(formatScalar(12.5)).toBe("12.5");
    expect(formatScalar(100)).toBe("100");
    expect(formatScalar(123456)).toBe("1.23e+5");
    expect(formatScalar(0.0000042)).toBe("4.20e-6");
    expect(formatScalar(Number.NaN)).toBe("–");
  });
});

describe("slice plane", () => {
  const bounds = { min: [-1, -2, -3], max: [1, 2, 3] };

  it("is perpendicular to the chosen axis", () => {
    expect(slicePlane({ axis: 0, fraction: 0.5, enabled: true }, bounds).normal).toEqual([
      1, 0, 0,
    ]);
    expect(slicePlane({ axis: 2, fraction: 0.5, enabled: true }, bounds).normal).toEqual([
      0, 0, 1,
    ]);
  });

  it("sweeps linearly across the bounds with margin at the ends", () => {
    const at = (fraction: number) =>
      slicePlane({ axis: 1, fraction, enabled: true }, bounds).offset;
    expect(at(0.5)).toBeCloseTo(0, 3);
    expect(at(0)).toBeLessThan(-2); // Just before the near face: all clipped.
    expect(at(1)).toBeGreaterThan(2); // Just past the far face: none clipped.
    expect(at(0.75)).toBeGreaterThan(at(0.25));
  });

  it("clamps the fraction", () => {
    const low = slicePlane({ axis: 0, fraction: -1, enabled: true }, bounds).offset;
    const high = slicePlane({ axis: 0, fraction: 5, enabled: true }, bounds).offset;
    expect(low).toBeCloseTo(slicePlane({ axis: 0, fraction: 0, enabled: true }, bounds).offset);
    expect(high).toBeCloseTo(slicePlane({ axis: 0, fraction: 1, enabled: true }, bounds).offset);
  });
});

describe("mesh bounds", () => {
  it("computes per-axis extremes of a flat position array", () => {
    const positions = [0, 0, 0, 1, -2, 5, -3, 4, 2];
    expect(meshBounds(positions)).toEqual({ min: [-3, -2, 0], max: [1, 4, 5] });
  });

  it("degrades to a zero box for empty input", () => {
    expect(meshBounds([])).toEqual({ min: [0, 0, 0], max: [0, 0, 0] });
  });

  it("defaults leave slicing off at full extent", () => {
    expect(DEFAULT_SLICE).toEqual({ axis: 0, fraction: 1, enabled: false });
  });
});
