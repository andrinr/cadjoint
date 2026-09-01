/**
 * Numeric legibility contract for every simulation/mesh color role
 * (src/simColors.ts documents the intent; this file holds it to numbers):
 *
 * - the two ramps stay visually distinct along their whole length,
 * - BC overlay hues clear 3:1 WCAG contrast against the paper viewport,
 *   stay mutually distinguishable, and keep off both ramps' high ends,
 * - element-edge charcoal stays crisp over the ramps' mid/high ranges,
 * - the WGSL copies of the ramp polynomials and the edge color match the
 *   TypeScript source of truth coefficient for coefficient.
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  BC_TYPE_COLORS,
  ELEMENT_EDGE_COLOR,
  ELEMENT_EDGE_COLOR_LIGHT,
  MAGMA_COEFFICIENTS,
  PROPOSAL_COLOR,
  VIEWPORT_BACKGROUND,
  VIRIDIS_COEFFICIENTS,
  contrastRatio,
  fieldRamp,
  qualityRamp,
  type Rgb,
} from "../src/simColors";

const distance = (a: Rgb, b: Rgb): number =>
  Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

const bcEntries = Object.entries(BC_TYPE_COLORS) as [string, Rgb][];

describe("field vs quality ramps", () => {
  it("stay visually distinct end to end", () => {
    for (let t = 0; t <= 1.0001; t += 0.1) {
      expect(distance(fieldRamp(t), qualityRamp(t)), `t=${t.toFixed(1)}`)
        .toBeGreaterThan(0.15);
    }
  });

  it("both start dark: the cold end of a field separates from paper", () => {
    // The polarity of this contract flips with the ground. On a dark viewport
    // the assertion was that the ramps' *high* ends read (13.8 and 16.3:1
    // there) while their cold ends vanished into the background at 1.17 and
    // 1.19:1. On paper the mirror holds: the cold ends carry 12.1 and 16.9:1
    // and the hot ends measure 1.02 and 1.15:1.
    //
    // Which end merges is a real trade, not a wash — the hot end is the end
    // anyone is looking at — and it is why the FEM surface now draws a
    // silhouette contour (see fs_sim in viewer/simulation.wgsl). Luminance is
    // no longer what separates a hot region from the ground; hue is, at 19.2
    // ΔOKLab, plus a drawn edge.
    expect(contrastRatio(fieldRamp(0), VIEWPORT_BACKGROUND)).toBeGreaterThan(7);
    expect(contrastRatio(qualityRamp(0), VIEWPORT_BACKGROUND)).toBeGreaterThan(7);
  });
});

describe("BC overlay hues", () => {
  it("clear 3:1 contrast against the paper viewport", () => {
    for (const [type, color] of bcEntries) {
      expect(contrastRatio(color, VIEWPORT_BACKGROUND), type).toBeGreaterThanOrEqual(3);
    }
    expect(contrastRatio(PROPOSAL_COLOR, VIEWPORT_BACKGROUND)).toBeGreaterThanOrEqual(3);
  });

  it("stay mutually distinguishable", () => {
    for (let a = 0; a < bcEntries.length; a++) {
      for (let b = a + 1; b < bcEntries.length; b++) {
        expect(
          distance(bcEntries[a][1], bcEntries[b][1]),
          `${bcEntries[a][0]} vs ${bcEntries[b][0]}`,
        ).toBeGreaterThan(0.3);
      }
    }
  });

  it("keep off both ramps' reserved high ends", () => {
    for (const [type, color] of bcEntries) {
      expect(distance(color, fieldRamp(1)), `${type} vs field high`).toBeGreaterThan(0.25);
      expect(distance(color, qualityRamp(1)), `${type} vs quality high`).toBeGreaterThan(0.25);
    }
    expect(distance(PROPOSAL_COLOR, fieldRamp(1))).toBeGreaterThan(0.25);
    expect(distance(PROPOSAL_COLOR, qualityRamp(1))).toBeGreaterThan(0.25);
  });
});

describe("element-edge charcoal", () => {
  it("reads over the mid/high range of both ramps (where meshes live)", () => {
    for (const t of [0.5, 0.75, 1]) {
      expect(contrastRatio(ELEMENT_EDGE_COLOR, fieldRamp(t)), `field t=${t}`)
        .toBeGreaterThanOrEqual(3);
      expect(contrastRatio(ELEMENT_EDGE_COLOR, qualityRamp(t)), `quality t=${t}`)
        .toBeGreaterThanOrEqual(3);
    }
  });
});

describe("WGSL constants stay in sync with simColors.ts", () => {
  const wgsl = readFileSync(
    join(dirname(fileURLToPath(import.meta.url)), "../src/viewer/simulation.wgsl"),
    "utf8",
  );

  /** The 7 vec3 coefficient rows inside one WGSL ramp function body. */
  const wgslCoefficients = (fn: "viridis" | "magma"): number[][] => {
    const body = wgsl.split(`fn ${fn}(`)[1]?.split("}")[0] ?? "";
    return [...body.matchAll(/vec3<f32>\(([^)]*)\)/g)].map((match) =>
      match[1].split(",").map((part) => Number(part.trim())),
    );
  };

  it("viridis (field ramp) coefficients match", () => {
    const rows = wgslCoefficients("viridis");
    expect(rows).toHaveLength(VIRIDIS_COEFFICIENTS.length);
    rows.forEach((row, index) =>
      row.forEach((value, channel) =>
        expect(value, `c${index}[${channel}]`).toBeCloseTo(
          VIRIDIS_COEFFICIENTS[index][channel],
          9,
        ),
      ),
    );
  });

  it("magma (quality ramp) coefficients match", () => {
    const rows = wgslCoefficients("magma");
    expect(rows).toHaveLength(MAGMA_COEFFICIENTS.length);
    rows.forEach((row, index) =>
      row.forEach((value, channel) =>
        expect(value, `c${index}[${channel}]`).toBeCloseTo(
          MAGMA_COEFFICIENTS[index][channel],
          9,
        ),
      ),
    );
  });

  it("both element-edge tones match", () => {
    // The edge shader picks between a dark and a light hairline by the ramp
    // luminance underneath it; both constants must stay in sync.
    const body = wgsl.slice(wgsl.indexOf("fn fs_sim_edge"));
    const tones = [...body.matchAll(/return vec4<f32>\(([^)]*)\);/g)].map((match) =>
      match[1].split(",").slice(0, 3).map((part) => Number(part.trim())),
    );
    expect(tones).toHaveLength(2);
    for (const [index, expected] of [ELEMENT_EDGE_COLOR, ELEMENT_EDGE_COLOR_LIGHT].entries()) {
      expect(tones[index][0]).toBeCloseTo(expected[0], 9);
      expect(tones[index][1]).toBeCloseTo(expected[1], 9);
      expect(tones[index][2]).toBeCloseTo(expected[2], 9);
    }
  });
});
