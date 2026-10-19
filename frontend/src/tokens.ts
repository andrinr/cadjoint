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
 * Scales are deliberately short. Six type sizes, six trackings, seven spacing
 * steps, one radius, four control heights: if a value is not on a scale it is
 * a bug, not a nuance.
 */

import type { BoundaryConditionType } from "./types";

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
  // value. The viewport is not a darker or lighter well, it is the same
  // sheet, and what marks its edge is a rule.
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
  // because an opacity fade has no assertable contrast ratio. 14.2 / 7.5 / 5.1
  // against the darkest ground, so even `ink-3` clears AA on every surface.
  ink: "#1c1814",
  "ink-2": "#4b4642",
  "ink-3": "#635e5a",
  // The one ink that is drawn on the accent rather than on paper: 10.8:1 on
  // `accent`, which is the number the whole accent rule turns on.
  "ink-on-accent": "#0a0a0c",

  // Rules. One weight — 1px, always — and three contrasts, because on a sheet
  // the hierarchy is carried by how dark a line is, not by how thick it is:
  //
  //   rule         1.7:1 on paper — separation *within* a panel
  //   rule-strong  3.1:1          — between sections, and the viewport seam
  //   rule-heavy   7.2:1          — the viewport frame and its corner marks
  //
  // `rule` is deliberately under 3:1 and is absent from MEANINGFUL_NON_TEXT: a
  // resting hairline inside a panel is structure, not state. `rule-strong` is
  // the one that has to be found — it is what says "the viewport starts here"
  // when there is no luminance step to say it — so it clears the non-text bar.
  rule: "#b4b2b9",
  "rule-strong": "#848289",
  "rule-heavy": "#4a484f",

  // The accent. One hue, and it has exactly one job: **a fill behind near-black
  // type.** It measures 10.8:1 as a ground under `ink-on-accent` and 1.5:1 as
  // ink on paper, so the two uses are not a preference — one passes and one
  // fails. Where another design would draw accent-coloured text or an accent
  // hairline, this one draws a filled block instead.
  //
  // There is one accent and not three because the modes are a pipeline, not
  // three worlds (design-language.md §6): which mode you are in is read from
  // the position of the filled cell in the switcher and from the word in the
  // hint bar, both of which survive greyscale and colour-blindness.
  accent: "#ffb000",
  // Pressed and hovered accent fills, and the only tone allowed to draw an
  // accent-coloured *mark*: 3.3:1 on paper, so it clears the non-text bar the
  // accent itself cannot.
  "accent-press": "#a87400",

  // Status and taxonomy, re-authored for paper: on a light ground a tone has
  // to be darkened, not brightened, to be read. Four tones total; the kind
  // chips reuse them rather than inventing hues of their own.
  danger: "#a8341c",
  "danger-ink": "#8a1f10",
  // The editor's warning severity: the accent family darkened past
  // `accent-press` and shifted toward amber so it cannot be read as
  // `danger`'s red. 4.76:1 on paper clears AA, so the same tone can label a
  // diagnostic and underline it.
  warn: "#8a5a00",
  ok: "#00734c",
  // The free-parameter role, and nothing else: a value the optimiser or a
  // drag may move.
  param: "#0065b4",
  "param-ink": "#004a85",

  // ── direction ───────────────────────────────────────────────────────────
  // World X, Y, Z. One set for the gizmo and the view-cube triad, darkened
  // until the letters clear 4.5:1 on paper and the strokes 3:1 on the
  // lightest SDF facet (`#c8c8cb`).
  "axis-x": "#c81e2a",
  "axis-y": "#17742a",
  "axis-z": "#1f5fd2",

  // ── construction ────────────────────────────────────────────────────────
  // The skeleton the part is built from — sketch edges, axes, construction
  // circles — and its vertex handles, drawn by the GPU over the solid. On
  // paper the part is grey, so a grey skeleton — a 2px dark-grey line through
  // a mid-grey solid, a pale grey square on a pale grey facet — reads as
  // shading. A
  // drawing is two things at once, the part and how it was made, and the
  // only channel the part does not already use is hue. So the skeleton takes
  // one, and it takes it alone.
  //
  // Magenta, because every other family is spoken for: blue is `param`
  // (free) and the Z axis, red and green the X and Y axes, amber the accent,
  // teal the proposal, and the BC hues fill the violets and the reds either
  // side. 4.9:1 on paper, and separated from the grey part by hue rather
  // than value, so it reads over a light facet and a dark one alike. Held to
  // ≥0.2 from every ramp end and BC hue by test/tokens.test.ts.
  //
  //   construction         the skeleton and a held handle, at rest
  //   construction-strong  the same under the pointer
  //   construction-locked  a node the source does not let you edit: the hue
  //                        lifted and softened, still ≥3:1 — greyed instead,
  //                        it went back to reading as the part
  construction: "#b4138f",
  "construction-strong": "#7a0a61",
  "construction-locked": "#b0559c",

  // ── viewport ink ────────────────────────────────────────────────────────
  // Everything drawn *inside* the viewport rectangle by the DOM: dimension
  // labels, the hint bar, the mode cue on the viewport border. They stay a
  // separate list from the chrome inks because they owe a separate rule:
  // **inside the rectangle, nothing is coloured.** The field ramp is the only
  // hue in the viewport, so an achromatic annotation cannot be mistaken for a
  // value.
  "viewport-ink": "#1c1814",
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
 * Chrome tones that carry meaning rather than decoration, so they owe ≥3:1
 * against the surface they sit on (WCAG non-text contrast).
 *
 * `accent` is absent, and that absence is the design: at 1.5:1 on paper it
 * cannot be a mark, which is why it is only ever a *fill* — see ACCENT_FILL
 * below, and the assertion in test/tokens.test.ts that holds both halves.
 * `rule` is absent for the same kind of reason, stated above its declaration.
 */
export const MEANINGFUL_NON_TEXT: ChromeToken[] = [
  "accent-press",
  "construction",
  "construction-strong",
  "construction-locked",
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
 * own — it is a ground, and the test asserts the pair clears AAA while the
 * same hue used as ink on paper fails even the non-text bar.
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
 * `surface-viewport` is absent: VIEWPORT_TONES are measured against it
 * separately.
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
 * letters, so there is no single `--tracking-caps`.
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
 * Control heights, four to ten units: each a line box plus a distance each
 * side. 24px is the WCAG 2.2 (2.5.8) minimum target, so `control-tag` is for
 * things that cannot be clicked and everything clickable is `control-sm` or
 * taller.
 */
export const CONTROL_HEIGHTS = {
  "control-tag": 16,
  "control-sm": 24,
  "control-lg": 32,
  "control-xl": 40,
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
  // IBM Plex Mono, bundled with the app (@fontsource, OFL): the playground
  // runs locally and cannot count on a font CDN.
  "font-mono": '"IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace',
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

/** A normalized triple as a `#rrggbb` string, channels clamped to 0…1. */
export function rgbToHex(rgb: readonly number[]): string {
  return `#${rgb
    .map((channel) =>
      Math.round(Math.min(1, Math.max(0, channel)) * 255)
        .toString(16)
        .padStart(2, "0"),
    )
    .join("")}`;
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
    role: "construction",
    token: "construction",
    means: "how the part is made: sketch edges, axes, construction circles, and a held vertex",
    carrier: "a line or a filled square, cased in sheet; dashed where the part hides it",
    zones: ["viewport"],
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
 * The construction overlay's state ladder (V1): the construction hue at
 * rest, its darker step under the pointer, `ink` once selected — the one
 * tone that reads over any field value (C4) — and the greyed hue when the
 * source locks the node. Free versus held is shape (and `param`), not this.
 */
export const VIEWPORT_STATES = {
  rest: "construction",
  hover: "construction-strong",
  selected: "ink",
  locked: "construction-locked",
} as const satisfies Record<string, ChromeToken>;

/**
 * The same ladder while a solved field is on screen (V2, C7), in the ink
 * steps rather than the skeleton's hue. Over a field the field is the only
 * colour, and the construction magenta sits beside magma's middle — over the
 * quality view it would read as a value.
 */
export const FIELD_VIEWPORT_STATES = {
  rest: "ink-3",
  hover: "ink-2",
  selected: "ink",
  locked: "rule-strong",
} as const satisfies Record<keyof typeof VIEWPORT_STATES, ChromeToken>;


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
export const BC_TYPE_COLORS: Record<BoundaryConditionType, Rgb> = {
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

/**
 * The B-rep inspector's face palette (role: identity). Categorical, not a
 * ramp: a face's colour says *which* face, never how much of anything, so —
 * like the field ramp — it is data, and the one other place a hue is allowed
 * inside the viewport rectangle. Nine light tones, each a step from its
 * neighbours in hue, light enough that every edge tone below clears 3:1 on
 * the darkest of them once shaded; neighbouring faces are given different
 * entries by `colourFaces` (src/brep.ts). No grey: a grey face would read as
 * the solid the inspector replaces.
 */
export const BREP_FACE_PALETTE: readonly Rgb[] = [
  "#a0cbe8",
  "#ffbe7d",
  "#ff9d9a",
  "#86bcb6",
  "#8cd17d",
  "#f1ce63",
  "#d4a6c8",
  "#fabfd2",
  "#d7b5a6",
].map(hexToRgb);

/**
 * How the inspector strokes an edge, by the kind the private tier gave it.
 *
 * Keyed by the tier's own vocabulary and open-ended: a kind missing here
 * (a new edge state) is drawn in `other` and listed under its own name in
 * the legend, so the inspector never hides an edge for not knowing its word.
 * `traced` is the exact curve the feature-edge overlay draws, so it takes
 * ink; everything that is *not* drawn as a crease takes a hue, so the eye
 * goes to what the overlay would get wrong.
 */
export const BREP_EDGE_TONES: Record<string, { token: ChromeToken; label: string }> = {
  traced: { token: "ink", label: "Traced" },
  sampled: { token: "accent-press", label: "Sampled" },
  undrawn: { token: "danger", label: "Undrawn crossing" },
  blend: { token: "param", label: "Blend boundary" },
  tangent: { token: "ok", label: "Tangent" },
  coincident: { token: "construction-locked", label: "Coincident" },
  contact: { token: "construction-locked", label: "Contact" },
  other: { token: "ink-3", label: "Other" },
};

/** The inspector's off-surface highlight: the offending edges, at full strength. */
export const BREP_OFF_SURFACE: ChromeToken = "danger-ink";

// ── viewport overlays ─────────────────────────────────────────────────────

export type Rgba = [number, number, number, number];

/**
 * Every overlay tone the renderer draws, as a role token and an alpha.
 *
 * The construction ladder carries the skeleton's state (V1, VIEWPORT_STATES):
 * rest `construction`, hover `construction-strong`, selected `ink`, locked
 * `construction-locked`. All four are opaque — an alpha would mix the hue
 * with whatever grey facet is under it and hand back a grey. Mesh edges,
 * face highlights and planes are not the skeleton and stay on the ink ladder.
 */
export const OVERLAY_TONES: Record<string, { token: ChromeToken; alpha: number }> = {
  edge: { token: VIEWPORT_STATES.rest, alpha: 1 },
  edgeHover: { token: VIEWPORT_STATES.hover, alpha: 1 },
  edgeSelected: { token: VIEWPORT_STATES.selected, alpha: 1 },
  edgeLocked: { token: VIEWPORT_STATES.locked, alpha: 1 },
  handle: { token: VIEWPORT_STATES.rest, alpha: 1 },
  handleHover: { token: VIEWPORT_STATES.hover, alpha: 1 },
  handleSelected: { token: VIEWPORT_STATES.selected, alpha: 1 },
  handleLocked: { token: VIEWPORT_STATES.locked, alpha: 1 },
  meshWire: { token: "ink-3", alpha: 0.3 },
  meshSharp: { token: "ink", alpha: 0.95 },
  faceOutline: { token: "ink", alpha: 0.9 },
  faceFill: { token: "ink", alpha: 0.12 },
  faceOutlineLocked: { token: "ink", alpha: 0.45 },
  faceFillLocked: { token: "ink", alpha: 0.05 },
  // The plane is the ground a sketch is drawn on, so it has to read against
  // the grey model behind it: a fill a step heavier than a face highlight and
  // a frame in ink-2. Hover and selection still step up from there.
  planeFill: { token: "ink-3", alpha: 0.14 },
  planeOutline: { token: "ink-2", alpha: 0.9 },
  planeFillHover: { token: "ink-2", alpha: 0.18 },
  planeOutlineHover: { token: "ink", alpha: 0.95 },
  planeFillSelected: { token: "ink", alpha: 0.2 },
  planeOutlineSelected: { token: "ink", alpha: 1 },
  planeFillLocked: { token: "rule-strong", alpha: 0.1 },
  planeOutlineLocked: { token: "rule-strong", alpha: 0.8 },
  planeFillPreview: { token: "ink-2", alpha: 0.16 },
  planeOutlinePreview: { token: "ink-2", alpha: 0.95 },
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

// ── text roles ────────────────────────────────────────────────────────────

/**
 * The only four ways text is set (guide T0). Every piece of text in the app
 * is one of these, whole: a stylesheet picks a role, it does not mix a size
 * from one with a weight from another.
 *
 *   caps   11px 600 tracked capitals  buttons, tabs, labels, column heads
 *   body   12px 400 as written        lists, names, values, details
 *   title  15px 600 tracked capitals  a panel's title, and nothing else
 *   prose  12px 400 sans              sentences: hints, help, empty states
 *
 * Values are body: in a monospace face every digit is already one width. Code
 * is body on the editor's taller line. Colour is not part of a role: ink level
 * is chosen separately (ink, ink-2, ink-3).
 *
 * Leadings are whole units (12, 16, 20px), not ratios, so a stack of rows is
 * a stack of grid lines.
 */
export const TEXT_ROLES = {
  caps: { size: 11, weight: 600, leading: "12px", family: "mono", tracking: "0.08em", case: "uppercase", numeric: "normal" },
  body: { size: 12, weight: 400, leading: "16px", family: "mono", tracking: "normal", case: "none", numeric: "tabular-nums" },
  title: { size: 15, weight: 600, leading: "20px", family: "mono", tracking: "0.04em", case: "uppercase", numeric: "normal" },
  prose: { size: 12, weight: 400, leading: "20px", family: "sans", tracking: "normal", case: "none", numeric: "normal" },
} as const;

export type TextRole = keyof typeof TEXT_ROLES;

// ── layout distances ──────────────────────────────────────────────────────

/**
 * The only distances layout uses (guide S0), each named by its job. The
 * spacing scale is the raw material; components use these.
 *
 *   gap      4   between cells of one group; a line box to its cell's edge
 *   inset    8   inside a sheet; a cell's text to its edge, horizontally
 *   section  16  between blocks and sections; the viewport's furniture from its edges
 */
export const LAYOUT = {
  gap: 4,
  inset: 8,
  section: 16,
} as const;

// ── atoms: lines, planes, blocks ──────────────────────────────────────────

/**
 * The three lines everything is drawn with (guide B1): the hairline, the
 * active line, the pending line.
 */
export const LINE_ATOMS = {
  // One hairline for all structure (separators, edges, the viewport frame).
  // It clears 3:1, so an edge can be found.
  line: "var(--hairline) solid var(--rule-strong)",
  active: "var(--stroke-active) solid var(--ink)",
  pending: "var(--hairline) dashed var(--rule-strong)",
} as const;

/**
 * The three planes (guide B2): what a thing is drawn on.
 *
 *   ground  the page and the viewport
 *   sheet   panels, popovers, fields, cards: one step up
 *   step    hover, and sub-bars: one step down from the sheet
 */
export const PLANE_ATOMS = {
  ground: "var(--surface-base)",
  sheet: "var(--surface-float)",
  step: "var(--surface-bar-alt)",
} as const;

/**
 * The two blocks (guide B3): a filled cell that says something by its fill.
 *
 *   current  the accent, with ink-on-accent type: the thing you act in or on
 *   kind     ink, with sheet type: a label of what something is
 */
export const BLOCK_ATOMS = {
  current: { fill: "var(--accent)", ink: "var(--ink-on-accent)" },
  kind: { fill: "var(--ink)", ink: "var(--surface-float)" },
} as const;

/** Layout columns shared across panels so value columns line up (row). */
export const COLUMNS = {
  "col-label": 96,
  "col-unit": 40,
} as const;

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
  "surface-float-00": rgba("surface-float", 0),
  // Shade is a fade of ink, never of black.
  "shade-none": rgba("ink", 0),
  "shade-soft": rgba("ink", 0.1),
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
};

// ── emitters ──────────────────────────────────────────────────────────────
// `npm run tokens` writes these to src/tokens.css, docs/assets/tokens.scss and
// src/viewer/tokens.wgsl; test/tokens.test.ts fails when any differs.

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
    ...Object.entries(LAYOUT).map(([k, v]) => [k, px(v)] as [string, string]),
    ...Object.entries(COLUMNS).map(([k, v]) => [k, px(v)] as [string, string]),
    ...Object.entries(LINE_ATOMS).map(([k, v]) => [k === "line" ? "line" : `line-${k}`, v] as [string, string]),
    ...Object.entries(PLANE_ATOMS).map(([k, v]) => [`plane-${k}`, v] as [string, string]),
    ...Object.entries(BLOCK_ATOMS).flatMap(([k, v]) => [
      [`block-${k}`, v.fill] as [string, string],
      [`block-${k}-ink`, v.ink] as [string, string],
    ]),
    ...Object.entries(TEXT_ROLES).flatMap(([role, r]) => [
      [
        `text-${role}`,
        `${r.weight} ${r.size}px / ${r.leading} var(--font-${r.family})`,
      ] as [string, string],
      [`track-${role}`, r.tracking] as [string, string],
      [`case-${role}`, r.case] as [string, string],
      [`numeric-${role}`, r.numeric] as [string, string],
    ]),
    ...Object.entries(DERIVED),
  ];
}

/** One class per text role, so markup can take a role whole (`class="t-caps"`). */
function roleClasses(): string {
  return Object.keys(TEXT_ROLES)
    .map(
      (role) =>
        `.t-${role} {\n  font: var(--text-${role});\n  letter-spacing: var(--track-${role});\n  text-transform: var(--case-${role});\n  font-variant-numeric: var(--numeric-${role});\n}`,
    )
    .join("\n");
}

/**
 * The layout primitives (guide L1), as classes: every arrangement in the app
 * is one of these, so a distance is chosen by naming the arrangement.
 */
const LAYOUT_CLASSES = `/* Layout primitives (guide L1). */
.l-inset {
  padding: var(--inset);
}
.l-stack {
  display: grid;
  align-content: start;
  gap: var(--section);
}
.l-ruled {
  display: grid;
}
.l-ruled > * + * {
  border-top: var(--line);
}
.l-cluster {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--gap);
}
.l-row {
  display: grid;
  grid-template-columns: var(--col-label) minmax(0, 1fr) var(--col-unit);
  align-items: baseline;
  gap: var(--inset);
  min-height: var(--control-sm);
}
.l-split {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--inset);
}`;

/** The `:root` block the stylesheet imports, the role and layout classes. */
export function tokensCss(): string {
  const lines = cssProperties().map(([name, value]) => `  --${name}: ${value};`);
  return `/* ${GENERATED} */\n\n:root {\n${lines.join("\n")}\n  color-scheme: light;\n}\n\n${roleClasses()}\n\n${LAYOUT_CLASSES}\n`;
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
