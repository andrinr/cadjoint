/**
 * The design system: one source of truth for every visual decision.
 *
 * The playground has two colour populations and they must not compete:
 *
 * - DATA lives in `simColors.ts` — the viridis field ramp, the magma quality
 *   ramp, the four BC hues, the proposal cyan, the element-edge tones. Those
 *   are saturated and bright because they carry measurements.
 * - CHROME lives here — surfaces, ink, hairlines, the three mode accents, and
 *   the handful of status tones. Chrome is deliberately quiet: the viewport is
 *   the subject and the panels frame it, so chrome never reaches the intensity
 *   the ramps use, and no chrome hue sits where a ramp's high end sits.
 *
 * `test/tokens.test.ts` holds both halves to numbers (WCAG AA for text, ≥3:1
 * for meaningful non-text, chrome-vs-data separation) and asserts that
 * `styles.css` declares exactly these values, so the CSS and this file cannot
 * drift apart. Everything a component needs at runtime (the mode accents) is
 * read from here; everything the stylesheet needs is mirrored as a custom
 * property with the same name.
 *
 * Scales are deliberately short. Six type sizes, seven spacing steps, four
 * radii, four control heights: if a value is not on a scale it is a bug, not a
 * nuance.
 */

import { contrastRatio, type Rgb } from "./simColors";

// ── colour ────────────────────────────────────────────────────────────────

/**
 * Semantic chrome roles, keyed by the CSS custom property they mirror
 * (without the leading `--`). Names describe the job, never the hue, so a
 * repaint is a change here and nowhere else.
 */
export const CHROME = {
  // The viewport is paper. It is the one surface in the app that is light,
  // and it is light because it is not chrome: it is the ground the field is
  // measured against, and viridis' closest sample to it sits 19.2 ΔOKLab away
  // (`VIEWPORT_BACKGROUND` in src/simColors.ts carries that argument). Chrome
  // ink never lands here — VIEWPORT_INK below is what is drawn on it.
  "surface-viewport": "#e6e6e9",

  // Surfaces, darkest (furthest back) to lightest (nearest the pointer).
  "surface-base": "#0a0c0b",
  "surface-bar": "#0b0d0c",
  "surface-bar-alt": "#0d0f0e",
  "surface-float": "#0e110f",
  "surface-panel": "#111312",
  "surface-raised": "#171a17",
  "surface-raised-hover": "#1b1f1b",

  // Ink. Three levels, and nothing else: text is never faded with `opacity`,
  // because an opacity fade has no assertable contrast ratio.
  ink: "#e9e8e2",
  "ink-2": "#b2b1a9",
  "ink-3": "#8f8e87",
  "ink-on-accent": "#10120d",

  // Lines. `line` is the structural hairline (panel edges, control borders),
  // `divider` is the weaker in-panel separation, `line-strong` is hover.
  //
  // Both sit below 3:1 against the surfaces they cross, and deliberately so:
  // in this design a resting border is decoration, not the thing that says a
  // control exists or what state it is in. That job belongs to the accent
  // tokens below (9–17:1) plus the focus ring, which is why `line` is absent
  // from MEANINGFUL_NON_TEXT. Pushing a hairline to 3:1 needs roughly
  // #5c6159, which turns every panel into a drawn box.
  line: "#383c36",
  "line-strong": "#4c5149",

  // Mode accents. Model owns the brand lime, so the two are one token.
  "accent-model": "#d9ff57",
  "accent-sketch": "#7fd6f5",
  "accent-simulate": "#ffb25c",
  /** Lime at reading weight — text on a lime-tinted surface. */
  "accent-model-ink": "#cbd99a",

  // Status and taxonomy. Four tones total; the kind chips reuse them rather
  // than inventing hues of their own.
  danger: "#ff8167",
  "danger-ink": "#ffb3a1",
  info: "#9adcf4",
  "info-ink": "#c9efff",
  ok: "#9fe7bd",

  // ── viewport ink ────────────────────────────────────────────────────────
  // Everything drawn *inside* the viewport rectangle by the DOM: dimension
  // labels, the hint bar, the mode cue on the viewport border. On paper the
  // polarity flips — annotations are dark ink with a light halo, not light
  // ink with a dark one — so chrome's three ink levels cannot be reused here
  // (`ink` measures 1.06:1 on `#e6e6e9`). Every tone here clears AA on paper
  // — 14.4 / 7.0 / 4.6 / 4.6 / 4.6 / 4.6 — and the three mode tones are the
  // mode accents at the weight paper needs: the lime, unchanged, measured
  // 1.01:1 there and was simply not visible.
  "viewport-ink": "#17171b",
  "viewport-ink-2": "#4a4a53",
  "viewport-mark": "#1769a9",
  "viewport-model": "#5b6d15",
  "viewport-sketch": "#18707d",
  "viewport-simulate": "#8f5a16",
} as const;

export type ChromeToken = keyof typeof CHROME;

/** Ink drawn directly on the viewport's paper ground; owes WCAG AA there. */
export const VIEWPORT_TONES: ChromeToken[] = [
  "viewport-ink",
  "viewport-ink-2",
  "viewport-mark",
  "viewport-model",
  "viewport-sketch",
  "viewport-simulate",
];

/** Editing modes, in switcher and keyboard-cycling order. */
export const MODE_ACCENTS = {
  model: CHROME["accent-model"],
  sketch: CHROME["accent-sketch"],
  simulate: CHROME["accent-simulate"],
} as const;

/**
 * Chrome tones that carry meaning rather than decoration, so they owe ≥3:1
 * against the surface they sit on (WCAG non-text contrast).
 */
export const MEANINGFUL_NON_TEXT: ChromeToken[] = [
  "accent-model",
  "accent-sketch",
  "accent-simulate",
  "danger",
  "info",
  "ok",
];

/** Chrome tones used as text, so they owe WCAG AA (4.5:1). */
export const TEXT_TONES: ChromeToken[] = [
  "ink",
  "ink-2",
  "ink-3",
  "accent-model",
  "accent-model-ink",
  "accent-sketch",
  "accent-simulate",
  "danger-ink",
  "info",
  "info-ink",
  "ok",
];

/**
 * Chrome surfaces text and marks are drawn on, brightest last (worst case).
 *
 * `surface-viewport` is deliberately absent: it is paper, chrome ink is never
 * drawn on it, and VIEWPORT_TONES is measured against it separately.
 */
export const TEXT_SURFACES: ChromeToken[] = [
  "surface-base",
  "surface-bar",
  "surface-bar-alt",
  "surface-float",
  "surface-panel",
  "surface-raised",
];

// ── type ──────────────────────────────────────────────────────────────────

/**
 * Six sizes. Anything smaller than 9px is unreadable on this background and
 * anything above 15px belongs to the viewport, not the chrome.
 */
export const TYPE_SCALE = {
  "text-3xs": 9,
  "text-2xs": 10,
  "text-xs": 11,
  "text-sm": 12,
  "text-md": 13,
  "text-lg": 15,
} as const;

export const WEIGHTS = {
  "weight-regular": 400,
  "weight-medium": 500,
  "weight-semibold": 600,
  "weight-bold": 700,
} as const;

export const LEADING = {
  "leading-flat": 1,
  "leading-tight": 1.25,
  "leading-snug": 1.4,
  "leading-normal": 1.55,
} as const;

// ── space, shape, depth, motion ───────────────────────────────────────────

/** 2 · 4 · 6 · 8 · 12 · 16 · 24 — dense at the bottom, where panels live. */
export const SPACE = {
  "space-1": 2,
  "space-2": 4,
  "space-3": 6,
  "space-4": 8,
  "space-5": 12,
  "space-6": 16,
  "space-7": 24,
} as const;

export const RADII = {
  "radius-xs": 4,
  "radius-sm": 6,
  "radius-md": 8,
  "radius-lg": 12,
} as const;

/**
 * Control heights. Every interactive box picks one, which is what keeps
 * labels, inputs and icons on a shared baseline inside a row.
 */
export const CONTROL_HEIGHTS = {
  "control-xs": 22,
  "control-sm": 26,
  "control-md": 30,
  "control-lg": 34,
  "control-xl": 36,
} as const;

/** Durations, in ms: a tint, a state change, a layout move. */
export const DURATIONS = {
  "dur-fast": 90,
  "dur-base": 160,
  "dur-slow": 260,
} as const;

/**
 * One easing family: a decelerating curve for anything that arrives, and its
 * symmetric sibling for anything that moves and settles in place.
 */
export const EASINGS = {
  ease: "cubic-bezier(0.2, 0, 0, 1)",
  "ease-inout": "cubic-bezier(0.4, 0, 0.2, 1)",
} as const;

// ── helpers ───────────────────────────────────────────────────────────────

/** A `#rrggbb` string as the normalized triple the contrast math wants. */
export function hexToRgb(hex: string): Rgb {
  const body = hex.replace("#", "");
  const full =
    body.length === 3
      ? body
          .split("")
          .map((channel) => channel + channel)
          .join("")
      : body;
  return [0, 2, 4].map((offset) =>
    parseInt(full.slice(offset, offset + 2), 16) / 255,
  ) as unknown as Rgb;
}

/** WCAG contrast between two chrome tokens. */
export const chromeContrast = (a: ChromeToken, b: ChromeToken): number =>
  contrastRatio(hexToRgb(CHROME[a]), hexToRgb(CHROME[b]));
