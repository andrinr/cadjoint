import { describe, expect, it } from "vitest";
import {
  DEFAULT_SLICE,
  applyDisplacements,
  autoDeformScale,
  formatScalar,
  meshBounds,
  rampCss,
  resolveResultView,
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

describe("deformed view scaling", () => {
  const bounds = { min: [0, 0, 0], max: [3, 4, 0] }; // diagonal 5

  it("targets 10% of the mesh diagonal at the peak displacement", () => {
    const displacements: [number, number, number][] = [
      [0, 0, 0],
      [0, 0.25, 0],
      [0.25, 0, 0],
    ];
    expect(autoDeformScale(bounds, displacements)).toBeCloseTo(2); // 0.5 / 0.25
  });

  it("degrades safely for zero displacement", () => {
    expect(autoDeformScale(bounds, [[0, 0, 0]])).toBe(1);
  });

  it("offsets vertices by scale × displacement", () => {
    const warped = applyDisplacements(
      [1, 1, 1, 2, 2, 2],
      [
        [0.1, 0, -0.1],
        [0, 0.2, 0],
      ],
      2,
    );
    expect(warped[0]).toBeCloseTo(1.2);
    expect(warped[2]).toBeCloseTo(0.8);
    expect(warped[4]).toBeCloseTo(2.4);
    expect(warped[5]).toBeCloseTo(2);
  });

  it("ignores trailing vertices without a displacement entry", () => {
    const warped = applyDisplacements([0, 0, 0, 5, 5, 5], [[1, 1, 1]], 1);
    expect(warped.slice(3)).toEqual([5, 5, 5]);
  });
});

describe("result view resolution", () => {
  const base = {
    defaultField: "von_mises",
    activeField: null as string | null,
    qualityView: false,
    fields: {
      von_mises: [1, 2, 3],
      displacement_magnitude: [0.1, 0.2, 0.3],
    },
    ranges: {
      von_mises: [1, 3] as [number, number],
      displacement_magnitude: [0.1, 0.3] as [number, number],
    },
    payloadScalars: [9, 9, 9],
    payloadRange: [9, 9] as [number, number],
    qualityScalars: null as number[] | null,
    qualityRange: null as [number, number] | null,
  };

  it("shows the default field before any switch", () => {
    const view = resolveResultView(base);
    expect(view.scalars).toEqual([1, 2, 3]);
    expect(view.range).toEqual([1, 3]);
    expect(view.label).toBe("von mises");
  });

  it("switches fields without re-solving", () => {
    const view = resolveResultView({ ...base, activeField: "displacement_magnitude" });
    expect(view.scalars).toEqual([0.1, 0.2, 0.3]);
    expect(view.label).toBe("displacement magnitude");
  });

  it("falls back to the payload scalars for an unknown field", () => {
    const view = resolveResultView({ ...base, activeField: "temperature" });
    expect(view.scalars).toEqual([9, 9, 9]);
    expect(view.range).toEqual([9, 9]);
  });

  it("only enters the quality view once its scalars exist", () => {
    const pending = resolveResultView({ ...base, qualityView: true });
    expect(pending.label).toBe("von mises");
    const ready = resolveResultView({
      ...base,
      qualityView: true,
      qualityScalars: [0.4, 0.9, 1],
      qualityRange: [0.4, 1],
    });
    expect(ready.scalars).toEqual([0.4, 0.9, 1]);
    expect(ready.range).toEqual([0.4, 1]);
    expect(ready.label).toBe("scaled jacobian");
  });

  it("derives the quality range from the scalars when no summary exists", () => {
    const view = resolveResultView({
      ...base,
      qualityView: true,
      qualityScalars: [0.5, 0.7],
    });
    expect(view.range).toEqual([0.5, 0.7]);
  });
});
