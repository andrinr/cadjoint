/**
 * The design system held to numbers.
 *
 * src/tokens.ts is the source of truth and src/styles.css is its mirror, so
 * the first job here is proving the two cannot drift: every token declared in
 * TypeScript must appear in the stylesheet's :root block with the same value,
 * and the stylesheet must not reintroduce the literals the tokens replaced.
 *
 * The second job is legibility. simColors.test.ts already covers the data
 * palette (the ramps, the BC hues, the element edges); this file covers the
 * chrome around it — WCAG AA for anything used as text, ≥3:1 for the tones
 * that carry meaning rather than decoration — and the seam between the two
 * populations, so a chrome accent can never be mistaken for a hot region of
 * the field ramp.
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  BC_TYPE_COLORS,
  PROPOSAL_COLOR,
  VIEWPORT_BACKGROUND,
  contrastRatio,
  fieldRamp,
  qualityRamp,
  relativeLuminance,
  type Rgb,
} from "../src/simColors";
import {
  CHROME,
  CONTROL_HEIGHTS,
  DURATIONS,
  EASINGS,
  LEADING,
  MEANINGFUL_NON_TEXT,
  MODE_ACCENTS,
  RADII,
  SPACE,
  TEXT_SURFACES,
  TEXT_TONES,
  VIEWPORT_TONES,
  TYPE_SCALE,
  WEIGHTS,
  chromeContrast,
  hexToRgb,
  type ChromeToken,
} from "../src/tokens";
import { MODE_ACCENTS as EDITING_MODE_ACCENTS } from "../src/editingMode";

const here = dirname(fileURLToPath(import.meta.url));
const css = readFileSync(join(here, "../src/styles.css"), "utf8");

/** The stylesheet's :root declaration block, where every token is defined. */
const rootBlock = (() => {
  const match = css.match(/^:root \{[\s\S]*?\n\}/m);
  if (!match) throw new Error("styles.css has no :root token block");
  return match[0];
})();

/**
 * The stylesheet with its token block and its fenced artwork removed — what
 * is left is ordinary rules, which may not contain raw design values.
 *
 * The @artwork fences hold the material swatch and the render-preset
 * thumbnails: those gradients are pictures of materials and render modes, not
 * chrome, so they keep literal colours on purpose.
 */
const ruleBody = css
  .replace(rootBlock, "")
  .replace(/\/\* @artwork[\s\S]*?\/\* @end-artwork \*\//g, "");

/** Value declared for `--name` in :root, or undefined. */
const token = (name: string): string | undefined =>
  rootBlock.match(new RegExp(`--${name}:\\s*([^;]+);`))?.[1].trim();

describe("styles.css mirrors src/tokens.ts", () => {
  it("declares every chrome colour with the same value", () => {
    for (const [name, value] of Object.entries(CHROME)) {
      expect(token(name), `--${name}`).toBe(value);
    }
  });

  it("declares every scale with the same values", () => {
    const scales: Record<string, number>[] = [
      TYPE_SCALE,
      SPACE,
      RADII,
      CONTROL_HEIGHTS,
    ];
    for (const scale of scales) {
      for (const [name, value] of Object.entries(scale)) {
        expect(token(name), `--${name}`).toBe(`${value}px`);
      }
    }
    for (const [name, value] of Object.entries(WEIGHTS)) {
      expect(token(name), `--${name}`).toBe(String(value));
    }
    for (const [name, value] of Object.entries(LEADING)) {
      expect(token(name), `--${name}`).toBe(String(value));
    }
    for (const [name, value] of Object.entries(DURATIONS)) {
      expect(token(name), `--${name}`).toBe(`${value}ms`);
    }
    for (const [name, value] of Object.entries(EASINGS)) {
      expect(token(name), `--${name}`).toBe(value);
    }
  });

  it("binds each editing mode to its accent token", () => {
    for (const [mode, accent] of Object.entries(MODE_ACCENTS)) {
      const block = css.match(
        new RegExp(`\\.app\\[data-mode="${mode}"\\] \\{[\\s\\S]*?\\n\\}`),
      );
      expect(block, mode).not.toBeNull();
      expect(block![0]).toContain(`--mode-accent: var(--accent-${mode})`);
      expect(token(`accent-${mode}`), mode).toBe(accent);
    }
  });

  it("is the same record editingMode.ts hands the rest of the app", () => {
    expect(EDITING_MODE_ACCENTS).toEqual(MODE_ACCENTS);
  });
});

describe("no design value escapes the token layer", () => {
  it("has no raw hex colour outside :root and the artwork fences", () => {
    expect([...ruleBody.matchAll(/#[0-9a-fA-F]{3,8}\b/g)].map((m) => m[0])).toEqual([]);
  });

  it("has no literal font-size outside :root", () => {
    expect(
      [...ruleBody.matchAll(/font-size:\s*[0-9][^;]*/g)].map((m) => m[0]),
    ).toEqual([]);
  });

  it("has no literal px size inside a `font:` shorthand", () => {
    expect(
      [...ruleBody.matchAll(/font:\s*[^;]*?\d+px[^;]*/g)].map((m) => m[0]),
    ).toEqual([]);
  });

  it("keeps every type size on the scale", () => {
    const sizes = Object.values(TYPE_SCALE);
    for (const size of sizes) expect(size).toBeGreaterThanOrEqual(9);
    // Strictly ascending, so "one step up" is always unambiguous.
    expect([...sizes].sort((a, b) => a - b)).toEqual(sizes);
    expect(new Set(sizes).size).toBe(sizes.length);
  });

  it("names one easing family and three durations", () => {
    expect(Object.keys(EASINGS)).toHaveLength(2);
    expect(Object.keys(DURATIONS)).toHaveLength(3);
    for (const duration of Object.values(DURATIONS)) {
      expect(duration).toBeLessThanOrEqual(300);
    }
  });
});

describe("chrome legibility", () => {
  it("clears WCAG AA for every tone used as text, on every surface", () => {
    for (const tone of TEXT_TONES) {
      for (const surface of TEXT_SURFACES) {
        expect(chromeContrast(tone, surface), `${tone} on ${surface}`)
          .toBeGreaterThanOrEqual(4.5);
      }
    }
  });

  it("clears 3:1 for every tone that carries meaning, on every surface", () => {
    for (const tone of MEANINGFUL_NON_TEXT) {
      for (const surface of TEXT_SURFACES) {
        expect(chromeContrast(tone, surface), `${tone} on ${surface}`)
          .toBeGreaterThanOrEqual(3);
      }
    }
  });

  it("separates the three ink levels enough to read as a hierarchy", () => {
    // Each level is a visible step down, not a rounding difference.
    expect(chromeContrast("ink", "ink-2")).toBeGreaterThan(1.4);
    expect(chromeContrast("ink-2", "ink-3")).toBeGreaterThan(1.4);
  });

  it("keeps the on-accent ink readable on every mode accent", () => {
    for (const mode of Object.keys(MODE_ACCENTS) as (keyof typeof MODE_ACCENTS)[]) {
      expect(chromeContrast("ink-on-accent", `accent-${mode}` as ChromeToken), mode)
        .toBeGreaterThanOrEqual(4.5);
    }
  });

  it("clears WCAG AA for every viewport tone, on paper", () => {
    // The viewport is the one light surface in the app, so it gets its own
    // ink and its own assertion; chrome ink measures 1.06:1 here.
    for (const tone of VIEWPORT_TONES) {
      expect(chromeContrast(tone, "surface-viewport"), tone).toBeGreaterThanOrEqual(4.5);
    }
  });

  it("keeps chrome ink off the paper viewport", () => {
    // Not a style rule — a bound. If any of these ever became legible on
    // paper it would mean the viewport had stopped being paper.
    for (const tone of ["ink", "ink-2", "ink-3"] as ChromeToken[]) {
      expect(chromeContrast(tone, "surface-viewport"), tone).toBeLessThan(3);
    }
  });

  it("keeps the mode accents distinguishable from each other", () => {
    const accents = Object.values(MODE_ACCENTS).map(hexToRgb);
    for (let a = 0; a < accents.length; a++) {
      for (let b = a + 1; b < accents.length; b++) {
        expect(distance(accents[a], accents[b])).toBeGreaterThan(0.3);
      }
    }
  });
});

const distance = (a: Rgb, b: Rgb): number =>
  Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

describe("chrome stays clear of the data palette", () => {
  const chromeAccents: ChromeToken[] = [
    "accent-model",
    "accent-sketch",
    "accent-simulate",
    "danger",
    "info",
    "ok",
  ];

  it("puts no chrome accent on either ramp's reserved high end", () => {
    for (const name of chromeAccents) {
      const colour = hexToRgb(CHROME[name]);
      expect(distance(colour, fieldRamp(1)), `${name} vs field high`)
        .toBeGreaterThan(0.2);
      expect(distance(colour, qualityRamp(1)), `${name} vs quality high`)
        .toBeGreaterThan(0.2);
    }
  });

  it("keeps every chrome accent distinguishable from every BC hue", () => {
    const data: [string, Rgb][] = [
      ...(Object.entries(BC_TYPE_COLORS) as [string, Rgb][]),
      ["proposal", PROPOSAL_COLOR],
    ];
    for (const name of chromeAccents) {
      const colour = hexToRgb(CHROME[name]);
      for (const [dataName, dataColour] of data) {
        expect(distance(colour, dataColour), `${name} vs ${dataName}`)
          .toBeGreaterThan(0.2);
      }
    }
  });

  it("keeps every chrome surface below the viewport's paper ground", () => {
    // Panels frame the viewport; they are never the brighter thing. With a
    // paper ground that stops being a contrast bound and becomes an ordering
    // one — the chrome sits under the paper, and by a wide margin, so the
    // frame reads as a frame and the field keeps the eye.
    const paper = relativeLuminance(VIEWPORT_BACKGROUND);
    for (const surface of TEXT_SURFACES) {
      expect(relativeLuminance(hexToRgb(CHROME[surface])), surface).toBeLessThan(paper);
      expect(
        contrastRatio(hexToRgb(CHROME[surface]), VIEWPORT_BACKGROUND),
        surface,
      ).toBeGreaterThan(10);
    }
  });
});
