/**
 * The design system: one source of truth for every visual decision.
 *
 * The playground has two colour populations and they must not compete:
 *
 * - DATA lives in `simColors.ts` — the viridis field ramp, the magma quality
 *   ramp, the four BC hues, the proposal cyan, the element-edge tones. Those
 *   are saturated and bright because they carry measurements.
 * - CHROME lives here — surfaces, ink, rules, the one accent, and the handful
 *   of status tones. Chrome is deliberately quiet: the viewport is the subject
 *   and the panels frame it, so chrome never reaches the intensity the ramps
 *   use, and no chrome hue sits where a ramp's high end sits.
 *
 * The ground is paper. Chrome and viewport share one value (`#e6e6e9`), so the
 * boundary between them is 1.00:1 — not a step in luminance but a rule. That
 * is the whole structural idea: **structure comes from rules, not boxes.** One
 * hairline weight, three contrasts, radius 0, no shadow.
 *
 * `test/tokens.test.ts` holds both halves to numbers (WCAG AA for text, ≥3:1
 * for meaningful non-text, chrome-vs-data separation) and asserts that
 * `styles.css` declares exactly these values, so the CSS and this file cannot
 * drift apart. Everything a component needs at runtime is read from here;
 * everything the stylesheet needs is mirrored as a custom property with the
 * same name.
 *
 * Scales are deliberately short. Six type sizes, five trackings, seven spacing
 * steps, one radius, four control heights: if a value is not on a scale it is
 * a bug, not a nuance.
 */

import { contrastRatio, type Rgb } from "./simColors";

// ── colour ────────────────────────────────────────────────────────────────

/**
 * Semantic chrome roles, keyed by the CSS custom property they mirror
 * (without the leading `--`). Names describe the job, never the hue, so a
 * repaint is a change here and nowhere else.
 */
export const CHROME = {
  // The viewport is paper, and so is the chrome around it: one ground, one
  // value, measured off `research/design/combined/measurements.txt`
  // where it covers 83% of the frame. `measurements.txt` records the seam as
  // "dL 0.0000 · contrast 1.00:1" — the viewport is not a darker or lighter
  // well, it is the same sheet, and what marks its edge is a rule.
  "surface-viewport": "#e6e6e9",

  // Surfaces. Not a depth ladder — a paper ladder: the page is the ground and
  // anything above it is a lighter sheet laid on top. Two steps, because the
  // reference uses two (`#edecee` for sub-bars, `#f8f7f8` for the dock sheet
  // and for floating chrome) and a third would be below a JND.
  "surface-base": "#e6e6e9",
  "surface-bar": "#e6e6e9",
  "surface-bar-alt": "#edecee",
  "surface-float": "#f8f7f8",
  "surface-panel": "#f8f7f8",
  "surface-raised": "#f8f7f8",
  // On paper, hover is a step *down* in lightness: pressure darkens the sheet,
  // it does not make it glow. A light UI that brightens on hover has nowhere
  // left to go once the sheet is already white.
  "surface-raised-hover": "#edecee",

  // Ink. Three levels, and nothing else: text is never faded with `opacity`,
  // because an opacity fade has no assertable contrast ratio. 14.4 / 7.5 / 5.1
  // against the darkest ground, so even `ink-3` clears AA on every surface.
  ink: "#18161a",
  "ink-2": "#48464d",
  "ink-3": "#605e65",
  // The one ink that is drawn on the accent rather than on paper: 7.02:1 on
  // `accent`, which is the number the whole accent rule turns on.
  "ink-on-accent": "#0a0a0c",

  // Rules. One weight — 1px, always — and three contrasts, because on a sheet
  // the hierarchy is carried by how dark a line is, not by how thick it is:
  //
  //   rule         2.15:1 on paper — separation *within* a panel
  //   rule-strong  3.81:1          — between sections, and the viewport seam
  //   rule-heavy   9.85:1          — the viewport frame and its corner marks
  //
  // `rule` is deliberately under 3:1 and is absent from MEANINGFUL_NON_TEXT: a
  // resting hairline inside a panel is structure, not state. `rule-strong` is
  // the one that has to be found — it is what says "the viewport starts here"
  // when there is no luminance step to say it — so it clears the non-text bar.
  rule: "#9f9da5",
  "rule-strong": "#747278",
  "rule-heavy": "#36343b",

  // The accent. One hue, and it has exactly one job: **a fill behind near-black
  // type.** It measures 7.02:1 as a ground under `ink-on-accent` and 2.26:1 as
  // ink on paper, so the two uses are not a preference — one passes and one
  // fails. Anywhere the old dark chrome would have drawn accent-coloured text
  // or an accent hairline, this design draws a filled block instead.
  //
  // There is one accent and not three because the modes are a pipeline, not
  // three worlds (design-language.md §6): which mode you are in is read from
  // the position of the filled cell in the switcher and from the word in the
  // hint bar, both of which survive greyscale and colour-blindness.
  accent: "#f87318",
  // Pressed and hovered accent fills, and the only tone allowed to draw an
  // accent-coloured *mark*: 3.36:1 on paper, so it clears the non-text bar the
  // accent itself cannot.
  "accent-press": "#c85d00",

  // Status and taxonomy, re-authored for paper: on a light ground a tone has
  // to be darkened, not brightened, to be read. Four tones total; the kind
  // chips reuse them rather than inventing hues of their own.
  danger: "#a8341c",
  "danger-ink": "#8a1f10",
  info: "#0065b4",
  "info-ink": "#004a85",
  ok: "#00734c",

  // ── viewport ink ────────────────────────────────────────────────────────
  // Everything drawn *inside* the viewport rectangle by the DOM: dimension
  // labels, the hint bar, the mode cue on the viewport border. These are now
  // the same values chrome uses, because chrome and viewport share a ground —
  // but they stay a separate list because they owe a separate rule:
  // **inside the rectangle, nothing is coloured.** The field ramp is the only
  // hue in the viewport, which is what `measurements.txt` scores as
  // "FIELD WINS", and an achromatic annotation cannot be mistaken for a value.
  "viewport-ink": "#18161a",
  "viewport-ink-2": "#605e65",
  "viewport-mark": "#48464d",
  // The mode cue drawn on the viewport border. One tone for all three modes:
  // the mode is named in words beside it, and a hue here would be the one
  // chrome signal crossing into the field's rectangle.
  "viewport-mode": "#605e65",

  // ── graticule ───────────────────────────────────────────────────────────
  // The instrument faceplate drawn *under* the scene: eight square divisions,
  // minor ticks on the two centre axes, four corner brackets. Furniture, not
  // data, so these deliberately sit far below the 3:1 a meaningful mark owes
  // — they are measured against paper in `test/graticule.test.ts` and held
  // inside a 1.6–2.8:1 band. Above that the grid competes with the field;
  // below it, it is invisible. Ordered weakest to strongest.
  "graticule-line": "#adadb3",
  "graticule-axis": "#9c9ca2",
  "graticule-frame": "#8f8f95",
} as const;

export type ChromeToken = keyof typeof CHROME;

/**
 * Ink drawn directly on the viewport's paper ground; owes WCAG AA there, and
 * owes being achromatic — see the note above `viewport-ink`.
 */
export const VIEWPORT_TONES: ChromeToken[] = [
  "viewport-ink",
  "viewport-ink-2",
  "viewport-mark",
  "viewport-mode",
];

/**
 * Graticule furniture drawn on paper.
 *
 * Absent from VIEWPORT_TONES on purpose: these are structure, never text, and
 * a hairline pushed to AA would turn the viewport into squared paper that
 * competes with the part. `test/graticule.test.ts` holds them to a band
 * instead of a floor.
 */
export const GRATICULE_TONES: ChromeToken[] = [
  "graticule-line",
  "graticule-axis",
  "graticule-frame",
];

/**
 * Editing modes, in switcher and keyboard-cycling order.
 *
 * All three name the same accent, on purpose. Colour is not how this UI says
 * which mode you are in — the filled cell's *position* in the switcher is, and
 * the word in the hint bar is (design-language.md §6). The record keeps its
 * mode-shaped signature so `editingMode.ts` and the CSS mode blocks are
 * unchanged; what changed is that there is nothing left to tell apart.
 */
export const MODE_ACCENTS = {
  model: CHROME.accent,
  sketch: CHROME.accent,
  simulate: CHROME.accent,
} as const;

/**
 * Chrome tones that carry meaning rather than decoration, so they owe ≥3:1
 * against the surface they sit on (WCAG non-text contrast).
 *
 * `accent` is absent, and that absence is the design: at 2.26:1 on paper it
 * cannot be a mark, which is why it is only ever a *fill* — see ACCENT_FILL
 * below, and the assertion in test/tokens.test.ts that holds both halves.
 * `rule` is absent for the same kind of reason, stated above its declaration.
 */
export const MEANINGFUL_NON_TEXT: ChromeToken[] = [
  "accent-press",
  "rule-strong",
  "rule-heavy",
  "danger",
  "info",
  "ok",
];

/**
 * The accent's only legal composition: `ink-on-accent` printed on `accent`.
 *
 * Stated as a pair rather than a tone because the accent has no meaning on its
 * own — it is a ground, and the test asserts the pair clears AA (7.02:1) while
 * the same hue used as ink on paper does not (2.26:1).
 */
export const ACCENT_FILL: { ground: ChromeToken; ink: ChromeToken } = {
  ground: "accent",
  ink: "ink-on-accent",
};

/** Chrome tones used as text, so they owe WCAG AA (4.5:1). */
export const TEXT_TONES: ChromeToken[] = [
  "ink",
  "ink-2",
  "ink-3",
  "danger",
  "danger-ink",
  "info",
  "info-ink",
  "ok",
];

/**
 * Chrome surfaces text and marks are drawn on, darkest last (worst case).
 *
 * On paper the worst case inverts: ink loses contrast on the *darkest* sheet,
 * not the brightest, so the list ends where the assertions bite.
 * `surface-viewport` is deliberately absent — it is the viewport's own ground
 * and VIEWPORT_TONES is measured against it separately — even though it now
 * carries the same value as `surface-base`.
 */
export const TEXT_SURFACES: ChromeToken[] = [
  "surface-float",
  "surface-panel",
  "surface-raised",
  "surface-bar-alt",
  "surface-raised-hover",
  "surface-bar",
  "surface-base",
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

/**
 * Tracking, one value per size that is ever set in uppercase.
 *
 * Tracking is a function of size, not a house style: at 9px the counters need
 * 0.16em to stay open and at 15px the same value would fall apart into
 * letters. A single `--tracking-caps` is the tell of a system that has not
 * measured its own labels, and it is what this replaces.
 */
export const TRACKING = {
  "tracking-3xs": "0.16em",
  "tracking-2xs": "0.13em",
  "tracking-xs": "0.1em",
  "tracking-sm": "0.08em",
  "tracking-md": "0.06em",
  "tracking-lg": "0.04em",
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

/**
 * One radius, and it is zero.
 *
 * Kept as a token rather than deleted so the decision has a name and one
 * place to change. A rounded corner is a softness this instrument does not
 * claim: everything here is a cell on a ruled sheet, and cells are square.
 */
export const RADII = {
  radius: 0,
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
