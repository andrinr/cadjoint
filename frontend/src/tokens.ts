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

import type { StudyBcType } from "./types";

// ── colour math ───────────────────────────────────────────────────────────
// Kept here, with no runtime imports, so `tools/emit-tokens.ts` can load this
// file directly under Node and so every module that needs a ratio reads it
// from the same place the values live.

export type Rgb = readonly [number, number, number];

function linearChannel(value: number): number {
  return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
}

/** WCAG relative luminance of a normalized rgb triple. */
export function relativeLuminance(rgb: Rgb): number {
  return (
    0.2126 * linearChannel(rgb[0]) +
    0.7152 * linearChannel(rgb[1]) +
    0.0722 * linearChannel(rgb[2])
  );
}

/** WCAG contrast ratio between two colours (≥ 1, order-independent). */
export function contrastRatio(a: Rgb, b: Rgb): number {
  const lighter = Math.max(relativeLuminance(a), relativeLuminance(b));
  const darker = Math.min(relativeLuminance(a), relativeLuminance(b));
  return (lighter + 0.05) / (darker + 0.05);
}

// ── colour ────────────────────────────────────────────────────────────────

/**
 * Semantic chrome roles, keyed by the CSS custom property they mirror
 * (without the leading `--`). Names describe the job, never the hue, so a
 * repaint is a change here and nowhere else.
 */
export const CHROME = {
  // The viewport is paper, and so is the chrome around it: one ground, one
  // value, measured off the all-light study (since deleted; the numbers it
  // settled are quoted here) where it covers 83% of the frame: it recorded
  // the seam as
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
  // fails. Where another design would draw accent-coloured text or an accent
  // hairline, this one draws a filled block instead.
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
  // The editor's warning severity, and the only tone added for it. Derived
  // from the accent family rather than invented: `accent` is the family's
  // fill (2.26:1 on paper, unusable as a mark), `accent-press` is the same
  // hue darkened until it just clears the non-text bar (3.36:1), and `warn`
  // carries that one step further and shifts toward amber so it cannot be
  // read as `danger`'s red. It measures **4.76:1 on paper** — comfortably
  // past the 3:1 a non-text mark owes, and past AA as well, which is what
  // lets the same tone label a diagnostic and underline it.
  warn: "#8a5a00",
  ok: "#00734c",
  // The free-parameter role, and nothing else: a value the optimiser or a
  // drag may move. Was `info`, which also meant a constraint count, a
  // residual line, a plane label and the elastic chip.
  param: "#0065b4",
  "param-ink": "#004a85",

  // ── direction ───────────────────────────────────────────────────────────
  // World X, Y, Z. One set for the gizmo and the view-cube triad, darkened
  // until the letters clear 4.5:1 on paper and the strokes 3:1 on the
  // lightest SDF facet (`#c8c8cb`).
  "axis-x": "#c81e2a",
  "axis-y": "#17742a",
  "axis-z": "#1f5fd2",

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
  // and minor ticks on the two centre axes. Furniture, not data, so these
  // deliberately sit far below the 3:1 a meaningful mark owes — they are
  // measured against paper in `test/graticule.test.ts` and held inside a
  // 1.6–2.8:1 band. Above that the grid competes with the field; below it, it
  // is invisible. Ordered weakest to strongest.
  "graticule-line": "#adadb3",
  "graticule-axis": "#9c9ca2",
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
export const GRATICULE_TONES: ChromeToken[] = ["graticule-line", "graticule-axis"];

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
  "warn",
  "rule-strong",
  "rule-heavy",
  "danger",
  "param",
  "ok",
  "axis-x",
  "axis-y",
  "axis-z",
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
  "param",
  "param-ink",
  "ok",
  "warn",
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
 * Control heights, five to nine units. Every box picks one, which is what
 * keeps labels, inputs and icons on a shared baseline inside a row. 24px is
 * the WCAG 2.2 (2.5.8) minimum target, so `control-tag` is for things that
 * cannot be clicked and everything clickable is `control-sm` or taller.
 */
export const CONTROL_HEIGHTS = {
  "control-tag": 20,
  "control-sm": 24,
  "control-md": 28,
  "control-lg": 32,
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

/** Lengths outside the scales: the unit, and the two line weights. */
export const LINES = {
  unit: 4,
  hairline: 1,
  "stroke-active": 2,
} as const;

export const FONTS = {
  "font-sans": 'system-ui, -apple-system, "Segoe UI", sans-serif',
  "font-mono": "ui-monospace, SFMono-Regular, Menlo, monospace",
} as const;

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

// ── the guide's tables ────────────────────────────────────────────────────
// What `design.html` renders. Kept here, beside the values, so the page that
// states a rule and the code that obeys it read one record.

/** The unit every length is a multiple of (axiom A6). */
export const UNIT = 4;

/** The control heights, as the design page names them. */
export const GRID_CONTROL_HEIGHTS = CONTROL_HEIGHTS;

/** Where a role's hue may appear. */
export type RoleZone = "chrome" | "editor" | "viewport" | "legend";

export interface ColorRole {
  role: string;
  /** Chrome token, or null for a data palette that lives in simColors.ts. */
  token: ChromeToken | null;
  means: string;
  /** The non-hue carrier that says the same thing (axiom A4). */
  carrier: string;
  zones: RoleZone[];
}

/**
 * The closed table of hues (axiom A3). A hue that is not on this list is a
 * bug; a new hue needs a new row first.
 */
export const COLOR_ROLES: ColorRole[] = [
  {
    role: "current",
    token: "accent",
    means: "the thing you are acting on or in: the active mode, the selected row, the primary action, the source span being edited",
    carrier: "a filled block, and its position",
    zones: ["chrome", "editor"],
  },
  {
    role: "free",
    token: "param",
    means: "a free parameter: a value the optimiser or a drag may move, through which gradients flow",
    carrier: "the word free, a count",
    zones: ["chrome", "editor"],
  },
  {
    role: "error",
    token: "danger",
    means: "failed, violated, destructive",
    carrier: "the word, a filled square",
    zones: ["chrome", "editor", "viewport"],
  },
  {
    role: "warning",
    token: "warn",
    means: "stale, approximate, needs attention",
    carrier: "the word, a triangle",
    zones: ["chrome", "editor"],
  },
  {
    role: "done",
    token: "ok",
    means: "converged, satisfied, connected",
    carrier: "the word, a dot",
    zones: ["chrome"],
  },
  {
    role: "direction x",
    token: "axis-x",
    means: "world X",
    carrier: "the letter X",
    zones: ["viewport"],
  },
  {
    role: "direction y",
    token: "axis-y",
    means: "world Y",
    carrier: "the letter Y",
    zones: ["viewport"],
  },
  {
    role: "direction z",
    token: "axis-z",
    means: "world Z",
    carrier: "the letter Z",
    zones: ["viewport"],
  },
];

/**
 * Overlay tones the renderer draws inside the viewport, as the guide
 * proposes them: the ink ladder for state, shape for free versus held.
 */
export const VIEWPORT_STATES = {
  rest: "ink-3",
  hover: "ink-2",
  selected: "ink",
  locked: "rule",
} as const satisfies Record<string, ChromeToken>;


// ── data palettes ─────────────────────────────────────────────────────────
// The colours that carry readings. simColors.ts and sdfRamp.ts evaluate and
// test them; the WGSL shaders receive them through `wgslTokens()`; nothing
// declares them a second time.

/** Degree-6 per-channel polynomial fit of matplotlib's viridis. */
export const VIRIDIS_COEFFICIENTS: readonly Rgb[] = [
  [0.2744554245, 0.0057679624, 0.3326638811],
  [0.1077083262, 1.3964696839, 1.3867705979],
  [-0.3272410968, 0.2148135645, 0.0919768808],
  [-4.5999315182, -5.7582381893, -19.2918089503],
  [6.2037359013, 14.1539649474, 56.6562995652],
  [4.7517868889, -13.7494394044, -65.3209678276],
  [-5.432077171, 4.641571316, 26.2721076045],
] as const;

/** Degree-6 per-channel polynomial fit of matplotlib's magma (err ≤ 0.024). */
export const MAGMA_COEFFICIENTS: readonly Rgb[] = [
  [-0.0020666453, -0.0006875655, -0.0095482507],
  [0.2504864448, 0.6944550333, 2.4952869139],
  [8.3459009063, -3.5960313696, 0.3290570684],
  [-27.6669694889, 14.2538530831, -13.6465831585],
  [52.1706837385, -27.9445843529, 12.8810906346],
  [-50.7585722964, 29.0538803789, 4.2699357345],
  [18.6642528253, -11.4900266123, -5.5707689618],
] as const;

/**
 * The viewport's paper ground (`surface-viewport`, `#e6e6e9`) as the shaders
 * carry it: three decimals, the same literal the Python-generated raymarch
 * WGSL uses, so the SDF ramp's centre and the ground are the same number.
 * test/tokens.test.ts holds it to the hex.
 */
export const VIEWPORT_BACKGROUND: Rgb = [0.902, 0.902, 0.914];

/**
 * BC-type overlay hues (role: boundary). Mutually distinguishable, ≥3:1 on
 * paper, clear of both ramps' high ends, the proposal teal and every chrome
 * tone — held to those numbers by test/simColors.test.ts.
 */
export const BC_TYPE_COLORS: Record<StudyBcType, Rgb> = {
  dirichlet: [0.122, 0.482, 0.961],
  heat_flux: [0.753, 0.408, 0.114],
  fixed: [0.667, 0.322, 0.965],
  traction: [0.953, 0.161, 0.286],
  inlet: [0.342, 0.51, 0.09],
  outlet: [0.09, 0.216, 0.51],
  walls: [0.008, 0.118, 0.832],
  heat_source: [0.347, 0.218, 0.862],
  held_temperature: [0.359, 0.006, 0.594],
};

/** The builder's live selection preview (role: proposal). 4.1:1 on paper. */
export const PROPOSAL_COLOR: Rgb = [0.106, 0.475, 0.529];

/**
 * Element-edge lines over a field: ink where the ramp under the line is
 * bright, the sheet where it is dark. Both achromatic (C4).
 */
export const ELEMENT_EDGE_COLOR: Rgb = hexToRgb(CHROME.ink);
export const ELEMENT_EDGE_COLOR_LIGHT: Rgb = hexToRgb(CHROME["surface-float"]);

/** The diverging SDF ramp (role: distance): inside, outside, and the paper centre. */
export const SDF_INSIDE: Rgb = [0.3924, 0.0015, 0.9925];
export const SDF_OUTSIDE: Rgb = [0.5398, 0.3384, 0.0033];
export const SDF_CENTRE: Rgb = VIEWPORT_BACKGROUND;

// ── viewport overlays ─────────────────────────────────────────────────────

export type Rgba = [number, number, number, number];

/**
 * Every overlay tone the renderer draws, as a role token and an alpha.
 *
 * The ink ladder carries state (V1): rest `ink-3`, hover `ink-2`, selected
 * `ink`, locked `rule`. Free versus held is carried by the handle's shape,
 * not a hue, so nothing in this table is chromatic except the axes.
 */
export const OVERLAY_TONES: Record<string, { token: ChromeToken; alpha: number }> = {
  edge: { token: "ink-3", alpha: 0.95 },
  edgeHover: { token: "ink-2", alpha: 1 },
  edgeSelected: { token: "ink", alpha: 1 },
  edgeLocked: { token: "rule-strong", alpha: 0.7 },
  handle: { token: "ink-2", alpha: 1 },
  handleHover: { token: "ink", alpha: 1 },
  handleSelected: { token: "ink", alpha: 1 },
  handleLocked: { token: "rule", alpha: 0.9 },
  meshWire: { token: "ink-3", alpha: 0.3 },
  meshSharp: { token: "ink", alpha: 0.95 },
  faceOutline: { token: "ink", alpha: 0.9 },
  faceFill: { token: "ink", alpha: 0.12 },
  faceOutlineLocked: { token: "ink", alpha: 0.45 },
  faceFillLocked: { token: "ink", alpha: 0.05 },
  planeFill: { token: "ink-3", alpha: 0.08 },
  planeOutline: { token: "ink-3", alpha: 0.65 },
  planeFillHover: { token: "ink-2", alpha: 0.12 },
  planeOutlineHover: { token: "ink-2", alpha: 0.9 },
  planeFillSelected: { token: "ink", alpha: 0.12 },
  planeOutlineSelected: { token: "ink", alpha: 1 },
  planeFillLocked: { token: "rule", alpha: 0.06 },
  planeOutlineLocked: { token: "rule", alpha: 0.5 },
  planeFillPreview: { token: "ink-2", alpha: 0.1 },
  planeOutlinePreview: { token: "ink-2", alpha: 0.85 },
};

/** An overlay tone as the renderer's straight-alpha rgba. */
export function overlayRgba(name: string): Rgba {
  const tone = OVERLAY_TONES[name];
  const [r, g, b] = hexToRgb(CHROME[tone.token]);
  return [r, g, b, tone.alpha];
}

/** World X, Y, Z as triples, for the gizmo and the triad. */
export const AXIS_RGB: [Rgb, Rgb, Rgb] = [
  hexToRgb(CHROME["axis-x"]),
  hexToRgb(CHROME["axis-y"]),
  hexToRgb(CHROME["axis-z"]),
];

// ── derived CSS values ────────────────────────────────────────────────────
// The stylesheet's non-scale custom properties: alpha ladders and the rgb
// triples a rule tints with. Each is a function of a token above, so none
// is a free value.

const triple = (token: ChromeToken) =>
  hexToRgb(CHROME[token]).map((channel) => Math.round(channel * 255)).join(", ");
const rgba = (token: ChromeToken, alpha: number) => `rgba(${triple(token)}, ${alpha})`;

/**
 * The accent and ink alpha ladder. Six strengths, each with a job: a wash is
 * a sheet barely touched, a tint an active row's fill, a veil a decorative
 * hairline, an edge an active control's border, a mark something to find.
 */
export const ALPHAS = {
  "alpha-wash": 0.06,
  "alpha-tint": 0.12,
  "alpha-tint-strong": 0.18,
  "alpha-veil": 0.28,
  "alpha-edge": 0.55,
  "alpha-mark": 0.85,
} as const;

export const DERIVED: Record<string, string> = {
  // Chrome floating over the viewport: opaque at rest (no text on a
  // data-dependent backdrop), with a transparent end for fades.
  "surface-viewport-94": rgba("surface-viewport", 0.94),
  "surface-viewport-70": rgba("surface-viewport", 0.7),
  "surface-float-00": rgba("surface-float", 0),
  // Shade is a fade of ink, never of black.
  "shade-none": rgba("ink", 0),
  "shade-soft": rgba("ink", 0.1),
  "shade-strong": rgba("ink", 0.16),
  "shade-scroll": rgba("ink", 0.22),
  scrim: rgba("ink", 0.42),
  // Separation between rows of one list: weaker than `rule`.
  divider: rgba("ink", 0.09),
  "accent-rgb": triple("accent"),
  "ink-rgb": triple("ink"),
  "danger-rgb": triple("danger"),
  "param-rgb": triple("param"),
  "ok-rgb": triple("ok"),
  "viewport-mark-rgb": triple("viewport-mark"),
  "viewport-mode-rgb": triple("viewport-mode"),
  // Type composites for the combinations that repeat everywhere.
  "type-kicker": "var(--weight-semibold) var(--text-3xs) / var(--leading-flat) var(--font-mono)",
  "type-mono": "var(--weight-medium) var(--text-2xs) / var(--leading-tight) var(--font-mono)",
  "type-mono-strong": "var(--weight-semibold) var(--text-2xs) / var(--leading-tight) var(--font-mono)",
  "type-body-small": "var(--weight-regular) var(--text-xs) / var(--leading-snug) var(--font-sans)",
};

// ── emitters ──────────────────────────────────────────────────────────────
// `npm run tokens` writes these to src/tokens.css and docs/assets/tokens.scss;
// test/tokens.test.ts fails when either file differs from what they return.

const GENERATED = "Generated from frontend/src/tokens.ts by `npm run tokens`. Do not edit.";

/** Every custom property, as `name: value` pairs, in a stable order. */
export function cssProperties(): [string, string][] {
  const px = (value: number) => `${value}px`;
  const ms = (value: number) => `${value}ms`;
  return [
    ...Object.entries(CHROME),
    ...Object.entries(FONTS),
    ...Object.entries(TYPE_SCALE).map(([k, v]) => [k, px(v)] as [string, string]),
    ...Object.entries(WEIGHTS).map(([k, v]) => [k, String(v)] as [string, string]),
    ...Object.entries(LEADING).map(([k, v]) => [k, String(v)] as [string, string]),
    ...Object.entries(TRACKING),
    ...Object.entries(LINES).map(([k, v]) => [k, px(v)] as [string, string]),
    ...Object.entries(SPACE).map(([k, v]) => [k, px(v)] as [string, string]),
    ...Object.entries(RADII).map(([k, v]) => [k, px(v)] as [string, string]),
    ...Object.entries(CONTROL_HEIGHTS).map(([k, v]) => [k, px(v)] as [string, string]),
    ...Object.entries(DURATIONS).map(([k, v]) => [k, ms(v)] as [string, string]),
    ...Object.entries(EASINGS),
    ...Object.entries(ALPHAS).map(([k, v]) => [k, String(v)] as [string, string]),
    ...Object.entries(DERIVED),
  ];
}

/** The `:root` block the stylesheet imports. */
export function tokensCss(): string {
  const lines = cssProperties().map(([name, value]) => `  --${name}: ${value};`);
  return `/* ${GENERATED} */\n\n:root {\n${lines.join("\n")}\n  color-scheme: light;\n}\n`;
}

/** The same values as Sass variables, for the Quarto docs theme. */
export function tokensScss(): string {
  const lines = cssProperties()
    .filter(([, value]) => !value.includes("var("))
    .map(([name, value]) => `$cj-${name}: ${value};`);
  // A Quarto theme layer: listed after docs/assets/cadjoint.scss in
  // _quarto.yml, so its defaults are emitted ahead of the theme that reads them.
  return `/*-- scss:defaults --*/\n// ${GENERATED}\n\n${lines.join("\n")}\n`;
}

const vec3 = (rgb: Rgb) => `vec3<f32>(${rgb.map((c) => Number(c.toFixed(10))).join(", ")})`;

/**
 * Constants every WGSL module is prefixed with, so a shader names a role
 * (`INK`, `AXIS_X`, `VIRIDIS_C3`) instead of carrying a literal.
 */
export function wgslTokens(): string {
  const lines = [
    `// ${GENERATED.replace("`npm run tokens`", "wgslTokens()")}`,
    `const INK = ${vec3(hexToRgb(CHROME.ink))};`,
    `const INK_2 = ${vec3(hexToRgb(CHROME["ink-2"]))};`,
    `const INK_3 = ${vec3(hexToRgb(CHROME["ink-3"]))};`,
    `const PAPER = ${vec3(VIEWPORT_BACKGROUND)};`,
    `const SHEET = ${vec3(hexToRgb(CHROME["surface-float"]))};`,
    `const AXIS_X = ${vec3(AXIS_RGB[0])};`,
    `const AXIS_Y = ${vec3(AXIS_RGB[1])};`,
    `const AXIS_Z = ${vec3(AXIS_RGB[2])};`,
    `const EDGE_DARK = ${vec3(ELEMENT_EDGE_COLOR)};`,
    `const EDGE_LIGHT = ${vec3(ELEMENT_EDGE_COLOR_LIGHT)};`,
    ...VIRIDIS_COEFFICIENTS.map((c, i) => `const VIRIDIS_C${i} = ${vec3(c)};`),
    ...MAGMA_COEFFICIENTS.map((c, i) => `const MAGMA_C${i} = ${vec3(c)};`),
  ];
  return `${lines.join("\n")}\n`;
}
