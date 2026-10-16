/**
 * Numeric legibility contract for every simulation/mesh color role
 * (src/simColors.ts documents the intent; this file holds it to numbers):
 *
 * - the two ramps stay visually distinct along their whole length,
 * - BC overlay hues clear 3:1 WCAG contrast against the paper viewport,
 *   stay mutually distinguishable, and keep off both ramps' high ends,
 * - element-edge charcoal stays crisp over the ramps' mid/high ranges,
 * - the WGSL shaders carry no colour of their own: they name the constants
 *   `wgslTokens()` prepends, and those equal the TypeScript source.
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
import { AXIS_RGB, CHROME, hexToRgb, wgslTokens } from "../src/tokens";

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

describe("WGSL colours come from tokens.ts and nowhere else", () => {
  const here = dirname(fileURLToPath(import.meta.url));
  const read = (name: string) => readFileSync(join(here, `../src/viewer/${name}`), "utf8");
  const shaders = ["simulation.wgsl", "overlay.wgsl", "graticule.wgsl"] as const;
  const sim = read("simulation.wgsl");

  /** Every `vec3<f32>(a, b, c)` literal with three numeric components. */
  const literals = (code: string): number[][] =>
    [...code.replace(/\/\/.*$/gm, "").matchAll(/vec[34]<f32>\(([^()]*)\)/g)]
      .map((match) => match[1].split(",").map((part) => Number(part.trim())))
      .filter((parts) => parts.length >= 3 && parts.every((value) => Number.isFinite(value)));

  /** The `const NAME = vec3<f32>(…)` table wgslTokens() emits. */
  const emitted = new Map(
    [...wgslTokens().matchAll(/const (\w+) = vec3<f32>\(([^)]*)\);/g)].map((match) => [
      match[1],
      match[2].split(",").map((part) => Number(part.trim())),
    ]),
  );

  const expectRgb = (name: string, expected: Rgb) => {
    const actual = emitted.get(name);
    expect(actual, name).toBeDefined();
    expected.forEach((value, channel) =>
      expect(actual![channel], `${name}[${channel}]`).toBeCloseTo(value, 9),
    );
  };

  it("wgslTokens() emits the ramp tables exactly as tokens.ts holds them", () => {
    VIRIDIS_COEFFICIENTS.forEach((row, index) => expectRgb(`VIRIDIS_C${index}`, row));
    MAGMA_COEFFICIENTS.forEach((row, index) => expectRgb(`MAGMA_C${index}`, row));
  });

  it("wgslTokens() emits the edge tones, ink, sheet, paper and axes from tokens.ts", () => {
    expectRgb("EDGE_DARK", ELEMENT_EDGE_COLOR);
    expectRgb("EDGE_LIGHT", ELEMENT_EDGE_COLOR_LIGHT);
    expectRgb("INK", hexToRgb(CHROME.ink));
    expectRgb("SHEET", hexToRgb(CHROME["surface-float"]));
    expectRgb("PAPER", VIEWPORT_BACKGROUND);
    AXIS_RGB.forEach((rgb, index) => expectRgb(`AXIS_${"XYZ"[index]}`, rgb));
  });

  it("the ramps are evaluated from the named coefficients", () => {
    for (const ramp of ["viridis", "magma"] as const) {
      const body = sim.split(`fn ${ramp}(`)[1]?.split("\n}")[0] ?? "";
      const prefix = ramp === "viridis" ? "VIRIDIS" : "MAGMA";
      for (let index = 0; index < 7; index++) {
        expect(body, `${ramp} uses ${prefix}_C${index}`).toMatch(new RegExp(`\\b${prefix}_C${index}\\b`));
      }
    }
  });

  it("the element edges return the named edge tones", () => {
    const body = sim.slice(sim.indexOf("fn fs_sim_edge"));
    expect(body).toContain("vec4<f32>(EDGE_DARK, 1.0)");
    expect(body).toContain("vec4<f32>(EDGE_LIGHT, 1.0)");
  });

  it("hover never tints the field", () => {
    // V2: over a solved field, hover is drawn (an outline), not mixed in.
    const body = sim.slice(sim.indexOf("fn fs_sim("), sim.indexOf("fn vs_sim_edge"));
    expect(body).not.toMatch(/s\.params\.z/);
  });

  it("no shader carries a copy of a token colour", () => {
    // Every colour token, as the WGSL would spell it. A literal within 1e-3
    // of one of them in any shader is a copy that can drift.
    const tokens = [...emitted.values()];
    for (const name of shaders) {
      const code = read(name);
      expect(code, `${name} declares no token itself`).not.toMatch(
        /const (INK|INK_2|INK_3|PAPER|SHEET|AXIS_[XYZ]|EDGE_DARK|EDGE_LIGHT|VIRIDIS_C\d|MAGMA_C\d)\b/,
      );
      for (const literal of literals(code)) {
        for (const token of tokens) {
          const same = token.every((value, channel) => Math.abs(value - literal[channel]) < 1e-3);
          expect(same, `${name}: vec(${literal.join(", ")}) copies a token`).toBe(false);
        }
      }
    }
  });

  it("the old colour literals are gone", () => {
    const all = shaders.map(read).join("\n");
    // overlay keyline, warm hover tint, old edge tones.
    for (const literal of ["0.025, 0.035, 0.03", "1.0, 0.62, 0.25", "0.05, 0.05, 0.06", "0.86, 0.90, 0.94"]) {
      expect(all).not.toContain(literal);
    }
  });

  it("the overlay keyline and the free handle use INK and SHEET", () => {
    const overlay = read("overlay.wgsl");
    expect(overlay).toMatch(/mix\(input\.color\.rgb, INK,/);
    expect(overlay).toMatch(/mix\(SHEET, input\.color\.rgb,/);
  });
});
