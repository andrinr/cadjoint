/**
 * The SDF views' diverging ramp, held to numbers.
 *
 * Three claims. The two hues stay clear of the viewport's existing ramps, by
 * the same measure `simColors.test.ts` uses to keep viridis and magma apart.
 * They stay heavier than the paper they are drawn on, which is the rule every
 * mark on this ground follows. And the WGSL copy in `_webgpu.py` is the same
 * ramp, coefficient for coefficient — the shader is the only place a reader
 * ever sees these colours, so a drift there is a legend that lies.
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  SDF_CENTRE,
  SDF_INSIDE,
  SDF_OUTSIDE,
  sdfRamp,
  sdfRampCss,
} from "../src/viewer/sdfRamp";
import {
  VIEWPORT_BACKGROUND,
  contrastRatio,
  fieldRamp,
  qualityRamp,
  type Rgb,
} from "../src/simColors";

const distance = (a: Rgb, b: Rgb): number =>
  Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

/** Closest approach to a ramp, over 201 samples of it. */
const nearest = (color: Rgb, ramp: (t: number) => [number, number, number]): number => {
  let best = Number.POSITIVE_INFINITY;
  for (let index = 0; index <= 200; index++) {
    best = Math.min(best, distance(color, ramp(index / 200)));
  }
  return best;
};

const SHADER = join(
  dirname(fileURLToPath(import.meta.url)),
  "../../cadjoint/viewer/_webgpu.py",
);

describe("the diverging SDF ramp", () => {
  it("keeps both ends clear of viridis and of magma", () => {
    // 0.25 is the threshold the existing ramps and overlay hues are held to
    // (test/simColors.test.ts); both ends clear it about twice over.
    for (const [name, color] of [
      ["inside", SDF_INSIDE],
      ["outside", SDF_OUTSIDE],
    ] as [string, Rgb][]) {
      expect(nearest(color, fieldRamp), `${name} vs viridis`).toBeGreaterThan(0.45);
      expect(nearest(color, qualityRamp), `${name} vs magma`).toBeGreaterThan(0.45);
    }
  });

  it("keeps its two ends far apart from each other", () => {
    expect(distance(SDF_INSIDE, SDF_OUTSIDE)).toBeGreaterThan(0.9);
  });

  it("draws darker than the paper it sits on", () => {
    // On a light ground a mark reads by being heavier than its surround.
    expect(contrastRatio(SDF_INSIDE, VIEWPORT_BACKGROUND)).toBeGreaterThan(4.5);
    expect(contrastRatio(SDF_OUTSIDE, VIEWPORT_BACKGROUND)).toBeGreaterThan(4.5);
  });

  it("puts the viewport's own ground at zero, and the ends at ±1", () => {
    expect(sdfRamp(0)).toEqual([...SDF_CENTRE]);
    expect(distance(sdfRamp(-1), SDF_INSIDE)).toBeLessThan(1e-9);
    expect(distance(sdfRamp(1), SDF_OUTSIDE)).toBeLessThan(1e-9);
    // The centre is the viewport background, so a zero crossing is the ground.
    expect(distance(SDF_CENTRE, VIEWPORT_BACKGROUND)).toBe(0);
  });

  it("clamps beyond the ends rather than running off the ramp", () => {
    expect(distance(sdfRamp(-4), SDF_INSIDE)).toBeLessThan(1e-9);
    expect(distance(sdfRamp(9), SDF_OUTSIDE)).toBeLessThan(1e-9);
  });

  it("spends real range close to the surface", () => {
    // The sqrt weighting: a tenth of the way out is already a third of the
    // way to the end, which is what makes the band around zero readable.
    const near = distance(sdfRamp(0.01), SDF_CENTRE);
    const full = distance(SDF_OUTSIDE, SDF_CENTRE);
    expect(near / full).toBeCloseTo(0.1, 6);
  });

  it("is monotone in |t| on both sides", () => {
    for (let step = 1; step <= 20; step++) {
      const t = step / 20;
      expect(distance(sdfRamp(t), SDF_CENTRE)).toBeGreaterThan(
        distance(sdfRamp(t - 0.05), SDF_CENTRE),
      );
      expect(distance(sdfRamp(-t), SDF_CENTRE)).toBeGreaterThan(
        distance(sdfRamp(-t + 0.05), SDF_CENTRE),
      );
    }
  });

  it("renders a legend gradient that starts and ends on the two hues", () => {
    const css = sdfRampCss(8);
    expect(css.startsWith("linear-gradient(to right, rgb(100, 0, 253)")).toBe(true);
    expect(css.endsWith("rgb(138, 86, 1))")).toBe(true);
  });

  it("matches the WGSL copy the shader draws with", () => {
    const shader = readFileSync(SHADER, "utf8");
    const constant = (name: string): number[] => {
      const match = shader.match(
        new RegExp(`const ${name}: vec3<f32> = vec3<f32>\\(([^)]*)\\);`),
      );
      expect(match, name).not.toBeNull();
      return match![1].split(",").map((part) => Number(part.trim()));
    };
    expect(constant("SDF_INSIDE")).toEqual([...SDF_INSIDE]);
    expect(constant("SDF_OUTSIDE")).toEqual([...SDF_OUTSIDE]);
    expect(constant("SDF_CENTRE")).toEqual([...SDF_CENTRE]);
    // And the same interpolation, not just the same endpoints.
    expect(shader).toContain("mix(SDF_CENTRE, end, sqrt(abs(clamped)))");
  });
});
