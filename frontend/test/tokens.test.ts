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
  ACCENT_FILL,
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
  TRACKING,
  GRATICULE_TONES,
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
    for (const [name, value] of Object.entries(TRACKING)) {
      expect(token(name), `--${name}`).toBe(value);
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

  it("binds every editing mode to the one accent token", () => {
    // Was: three modes, three hues, one `--accent-<mode>` token each. The
    // light direction has a single accent — mode is read from the position of
    // the filled cell and from the word in the hint bar (§6) — so what is
    // asserted now is that each mode selector still resolves `--mode-accent`,
    // and that all three resolve it to the same fill.
    for (const mode of Object.keys(MODE_ACCENTS)) {
      const selector = css.match(
        new RegExp(`\\.app\\[data-mode="${mode}"\\][,\\s][\\s\\S]*?\\n\\}`),
      );
      expect(selector, mode).not.toBeNull();
      expect(selector![0]).toContain("--mode-accent: var(--accent)");
    }
    expect(new Set(Object.values(MODE_ACCENTS)).size).toBe(1);
    expect(token("accent")).toBe(CHROME.accent);
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

  it("names one radius, and it is zero", () => {
    // Was implicit in a four-step radius scale. Radius 0 is the shape rule of
    // this direction, so it is asserted rather than left to a convention.
    expect(Object.values(RADII)).toEqual([0]);
    expect(token("radius")).toBe("0px");
  });

  it("keeps tracking a function of size, not one global value", () => {
    // Was a single --tracking-caps. A 9px label and a 15px title cannot share
    // a tracking, so the scale is asserted to descend with size.
    const values = Object.values(TRACKING).map((em) => parseFloat(em));
    expect(values.length).toBeGreaterThanOrEqual(5);
    expect([...values].sort((a, b) => b - a)).toEqual(values);
    expect(ruleBody).not.toContain("--tracking-caps");
  });

  it("casts no shadow", () => {
    // Elevation here is luminance plus a rule. A drop shadow on paper draws a
    // card lying on a desk, which is a metaphor this instrument refuses.
    expect([...ruleBody.matchAll(/box-shadow:[^;]*/g)].map((m) => m[0])).toEqual([]);
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

  it("keeps the editor's warning tone a mark of its own", () => {
    // The gutter shows three severities side by side, so `warn` owes more
    // than a contrast ratio: it has to be findable against the sheet, and
    // separable from the two tones it sits between. Against `danger` that
    // separation is hue (both are dark, so luminance says almost nothing);
    // against `ink-3` it is chroma, since the info marker is achromatic.
    expect(chromeContrast("warn", "surface-viewport")).toBeGreaterThanOrEqual(3);
    expect(chromeContrast("warn", "surface-panel")).toBeGreaterThanOrEqual(3);
    expect(distance(hexToRgb(CHROME.warn), hexToRgb(CHROME.danger))).toBeGreaterThan(0.2);
    const [r, g, b] = hexToRgb(CHROME.warn);
    expect(Math.max(r, g, b) - Math.min(r, g, b)).toBeGreaterThan(0.4);
    const [ir, ig, ib] = hexToRgb(CHROME["ink-3"]);
    expect(Math.max(ir, ig, ib) - Math.min(ir, ig, ib)).toBeLessThan(0.04);
  });

  it("makes the accent a fill and refuses to make it a mark", () => {
    // Was: "the on-accent ink is readable on every mode accent" — three
    // accents, one assertion each. There is one accent now, and the pair of
    // numbers below is the whole reason it is used the way it is: as a ground
    // it clears AAA, as ink on paper it fails even the 3:1 a non-text mark
    // owes. The design is not preferring one; one passes and one fails.
    expect(chromeContrast(ACCENT_FILL.ink, ACCENT_FILL.ground)).toBeGreaterThanOrEqual(7);
    expect(chromeContrast("accent", "surface-viewport")).toBeLessThan(3);
    expect(chromeContrast("accent", "surface-base")).toBeLessThan(3);
    // The pressed tone is the escape hatch, and it is held to the mark bar.
    expect(chromeContrast("accent-press", "surface-base")).toBeGreaterThanOrEqual(3);
  });

  it("clears WCAG AA for every viewport tone, on paper", () => {
    for (const tone of VIEWPORT_TONES) {
      expect(chromeContrast(tone, "surface-viewport"), tone).toBeGreaterThanOrEqual(4.5);
    }
  });

  it("draws nothing coloured inside the viewport rectangle", () => {
    // Replaces "keeps chrome ink off the paper viewport", which asserted that
    // --ink measured under 3:1 on paper. That was true only because chrome was
    // dark; with one paper ground for both, chrome ink is perfectly legible in
    // there and the old bound is meaningless. What still has to hold is the
    // §3.7 zoning rule, and it was always the real one: the *field* owns colour,
    // so every DOM tone drawn inside the rectangle is achromatic and the ramp
    // is the only hue a reader can see there.
    for (const tone of [...VIEWPORT_TONES, ...GRATICULE_TONES]) {
      const [r, g, b] = hexToRgb(CHROME[tone]);
      expect(Math.max(r, g, b) - Math.min(r, g, b), `${tone} chroma`).toBeLessThan(0.04);
    }
    // …and the accent, which is the loudest thing in the chrome, is not on
    // that list.
    const [ar, ag, ab] = hexToRgb(CHROME.accent);
    expect(Math.max(ar, ag, ab) - Math.min(ar, ag, ab)).toBeGreaterThan(0.5);
  });
});

const distance = (a: Rgb, b: Rgb): number =>
  Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

describe("chrome stays clear of the data palette", () => {
  const chromeAccents: ChromeToken[] = ["accent", "danger", "info", "ok", "warn"];

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

  it("puts the chrome on the same sheet as the viewport", () => {
    // Was: "keeps every chrome surface below the viewport's paper ground" —
    // chrome darker than paper by more than 10:1, which is what a dark shell
    // around a light viewport means. The light direction removes the step
    // entirely (measurements.txt: "dL 0.0000 · contrast 1.00:1"), so the
    // assertion inverts: no chrome surface is *darker* than the viewport, the
    // base sheet is the same value, and the seam is therefore something that
    // has to be drawn rather than something you can see for free.
    // simColors stores the paper as a 3-decimal triple, so compare against the
    // token itself and hold the two to agreement separately.
    const paper = relativeLuminance(hexToRgb(CHROME["surface-viewport"]));
    expect(relativeLuminance(VIEWPORT_BACKGROUND)).toBeCloseTo(paper, 3);
    for (const surface of TEXT_SURFACES) {
      expect(
        relativeLuminance(hexToRgb(CHROME[surface])),
        surface,
      ).toBeGreaterThanOrEqual(paper);
      expect(
        contrastRatio(hexToRgb(CHROME[surface]), VIEWPORT_BACKGROUND),
        surface,
      ).toBeLessThan(1.25);
    }
    expect(CHROME["surface-base"]).toBe(CHROME["surface-viewport"]);
    // The rule that has to carry the seam instead clears the non-text bar
    // against both sides of it.
    expect(chromeContrast("rule-strong", "surface-base")).toBeGreaterThanOrEqual(3);
    expect(chromeContrast("rule-strong", "surface-viewport")).toBeGreaterThanOrEqual(3);
  });
});
