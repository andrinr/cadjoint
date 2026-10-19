/**
 * The design system held to numbers.
 *
 * src/tokens.ts is the source of truth and src/tokens.css its generated
 * mirror, so the first job here is proving the two cannot drift, and that no
 * stylesheet reintroduces the literals the tokens stand for.
 *
 * The second job is legibility. simColors.test.ts already covers the data
 * palette (the ramps, the BC hues, the element edges); this file covers the
 * chrome around it — WCAG AA for anything used as text, ≥3:1 for the tones
 * that carry meaning rather than decoration — and the seam between the two
 * populations, so a chrome accent can never be mistaken for a hot region of
 * the field ramp.
 */

import { readFileSync, readdirSync } from "node:fs";
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
  RADII,
  SPACE,
  TEXT_SURFACES,
  TEXT_TONES,
  TRACKING,
  GRATICULE_TONES,
  OVERLAY_TONES,
  VIEWPORT_TONES,
  TYPE_SCALE,
  WEIGHTS,
  chromeContrast,
  cssProperties,
  hexToRgb,
  tokensCss,
  tokensScss,
  wgslTokens,
  type ChromeToken,
} from "../src/tokens";
import { EDITING_MODES } from "../src/editingMode";
import { distance } from "./helpers";

const here = dirname(fileURLToPath(import.meta.url));
const css = readFileSync(join(here, "../src/styles.css"), "utf8");

/** The generated token block. */
const rootBlock = readFileSync(join(here, "../src/tokens.css"), "utf8");

/** Every stylesheet in src/, generated block excluded. */
const stylesheets = (() => {
  const files: string[] = [];
  const walk = (dir: string) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const path = join(dir, entry.name);
      if (entry.isDirectory()) walk(path);
      else if (entry.name.endsWith(".css") && entry.name !== "tokens.css") files.push(path);
    }
  };
  walk(join(here, "../src"));
  return files.map((path) => ({ path, text: readFileSync(path, "utf8") }));
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
  .replace(/\/\* @artwork[\s\S]*?\/\* @end-artwork \*\//g, "");

/** Value declared for `--name` in :root, or undefined. */
const token = (name: string): string | undefined =>
  rootBlock.match(new RegExp(`--${name}:\\s*([^;]+);`))?.[1].trim();

describe("the generated files are current", () => {
  it("src/tokens.css is what tokensCss() emits", () => {
    expect(rootBlock, "run `npm run tokens`").toBe(tokensCss());
  });

  it("docs/assets/tokens.scss is what tokensScss() emits", () => {
    const scss = readFileSync(join(here, "../../docs/assets/tokens.scss"), "utf8");
    expect(scss, "run `npm run tokens`").toBe(tokensScss());
  });

  it("src/viewer/tokens.wgsl is what wgslTokens() emits", () => {
    const wgsl = readFileSync(join(here, "../src/viewer/tokens.wgsl"), "utf8");
    expect(wgsl, "run `npm run tokens`").toBe(wgslTokens());
  });

  it("styles.css declares no token of its own", () => {
    expect(css).not.toMatch(/^:root \{/m);
  });

  it("uses no custom property that the token layer does not declare", () => {
    // A var() of an undeclared name falls back silently to nothing. Local
    // properties a stylesheet declares itself (--mode-accent and friends) are
    // allowed.
    const declared = new Set(cssProperties().map(([name]) => name));
    for (const { text } of stylesheets) {
      for (const match of text.matchAll(/--([\w-]+)\s*:/g)) declared.add(match[1]);
    }
    const unknown: string[] = [];
    for (const { path, text } of stylesheets) {
      for (const match of text.matchAll(/var\(--([\w-]+)/g)) {
        if (!declared.has(match[1])) unknown.push(`${path.split("/src/")[1]}: --${match[1]}`);
      }
    }
    expect([...new Set(unknown)]).toEqual([]);
  });
});

describe("tokens.css mirrors src/tokens.ts", () => {
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
    // Mode is read from the position of the filled cell and the word in the
    // hint bar, not from a hue, so every mode resolves the same fill.
    for (const mode of EDITING_MODES) {
      const selector = css.match(
        new RegExp(`\\.app\\[data-mode="${mode}"\\][,\\s][\\s\\S]*?\\n\\}`),
      );
      expect(selector, mode).not.toBeNull();
      expect(selector![0]).toContain("--mode-accent-rgb: var(--accent-rgb)");
    }
    expect(token("accent")).toBe(CHROME.accent);
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
    expect(Object.values(RADII)).toEqual([0]);
    expect(token("radius")).toBe("0px");
  });

  it("keeps tracking a function of size, not one global value", () => {
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
    // As a ground it clears AAA; as ink on paper it fails even the 3:1 a
    // non-text mark owes.
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
    // The field owns colour (§3.7), so every DOM tone drawn inside the
    // rectangle is achromatic and the ramp is the only hue there.
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

describe("the construction skeleton", () => {
  // The part is grey on paper, so the skeleton drawn over it is told apart
  // by hue: every state of it but `selected` (which is ink, C4) is chromatic,
  // and none of it shares a family with `param`, which marks a free handle
  // on the same overlay.
  it("is drawn in the construction hue, opaque, in every unselected state", () => {
    for (const name of ["edge", "edgeHover", "edgeLocked", "handle", "handleHover", "handleLocked"]) {
      const tone = OVERLAY_TONES[name];
      expect(tone.token, name).toMatch(/^construction/);
      expect(tone.alpha, name).toBe(1);
      const [r, g, b] = hexToRgb(CHROME[tone.token]);
      expect(Math.max(r, g, b) - Math.min(r, g, b), `${name} chroma`).toBeGreaterThan(0.3);
    }
    expect(OVERLAY_TONES.edgeSelected.token).toBe("ink");
    expect(OVERLAY_TONES.handleSelected.token).toBe("ink");
  });

  it("keeps its states apart from each other and from param", () => {
    expect(chromeContrast("construction", "construction-strong")).toBeGreaterThan(1.5);
    expect(chromeContrast("construction", "construction-locked")).toBeGreaterThan(1.3);
    for (const tone of ["construction", "construction-strong", "construction-locked"] as const) {
      expect(distance(hexToRgb(CHROME[tone]), hexToRgb(CHROME.param)), tone).toBeGreaterThan(0.4);
    }
  });
});

describe("chrome stays clear of the data palette", () => {
  const chromeAccents: ChromeToken[] = [
    "accent",
    "danger",
    "param",
    "ok",
    "warn",
    "construction",
    "construction-strong",
    "construction-locked",
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

  it("puts the chrome on the same sheet as the viewport", () => {
    // No chrome surface is darker than the viewport and the base sheet is the
    // same value, so the seam has to be drawn. simColors stores the paper as a
    // 3-decimal triple, so it is held to the token separately.
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

describe("text is set only in the four roles, spaced only by the three distances", () => {
  // Every app stylesheet; the design page's own sheet and the fenced artwork
  // (pictures of materials and render modes) are held to the same rules.
  const sheets = stylesheets.map(({ path, text }) => ({
    name: path.split("/src/")[1],
    text: text
      .replace(/\/\* @artwork[\s\S]*?\/\* @end-artwork \*\//g, "")
      .replace(/\/\*[\s\S]*?\*\//g, ""),
  }));
  const offenders = (pattern: RegExp, allowed: RegExp) => {
    const found: string[] = [];
    for (const { name, text } of sheets) {
      for (const match of text.matchAll(pattern)) {
        const value = match[1].trim();
        if (!allowed.test(value)) found.push(`${name}: ${match[0].trim()}`);
      }
    }
    return [...new Set(found)];
  };

  it("sets font only through a text role", () => {
    expect(offenders(/\bfont:\s*([^;]+);/g, /^(var\(--text-[a-z]+\)|inherit)$/)).toEqual([]);
  });

  it("never sets size, weight, tracking or case outside a role", () => {
    expect(offenders(/\bfont-size:\s*([^;]+);/g, /^(inherit|var\(--text-[a-z]+\))$/)).toEqual([]);
    expect(offenders(/\bfont-weight:\s*([^;]+);/g, /^inherit$/)).toEqual([]);
    expect(offenders(/\bletter-spacing:\s*([^;]+);/g, /^(normal|inherit|var\(--track-[a-z]+\))$/)).toEqual([]);
    expect(offenders(/\btext-transform:\s*([^;]+);/g, /^(none|inherit|var\(--case-[a-z]+\))$/)).toEqual([]);
  });

  it("draws every border and outline with one of the three lines", () => {
    // A line is weight, style and colour together (rule, edge, frame, active,
    // pending); a state may recolour an edge with border-color, but a new kind
    // of line is not written inline.
    expect(
      offenders(
        /\b(?:border(?:-top|-right|-bottom|-left)?|outline):\s*([^;]+);/g,
        /^(var\(--line(-[a-z]+)?\)|0|none|0 none)$/,
      ),
    ).toEqual([]);
  });

  it("spaces layout with the named distances, not the raw scale", () => {
    expect(offenders(/(var\(--space-\d\))/g, /^$/)).toEqual([]);
  });
});
