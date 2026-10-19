/**
 * Variations: try the system's choices on this page, then download them.
 *
 * Every choice is a set of CSS custom properties — the same names tokens.ts
 * generates — written onto :root, so every specimen on the page restyles at
 * once. Download writes the selections and every resolved value to a JSON
 * file; tokens.ts can be updated from it.
 *
 * The page only: the playground keeps its tokens until a file is applied.
 * Selections are remembered in this browser (a convenience; the file is the
 * record).
 */

import { For, Show, createMemo, createSignal, onMount } from "solid-js";
import { CHROME, contrastRatio, hexToRgb, rgbToHex } from "../tokens";

type Vars = Record<string, string>;

interface Choice {
  id: string;
  label: string;
  note?: string;
  vars: () => Vars;
  /** Google Fonts family to load when chosen. */
  font?: string;
}

interface Axis {
  id: string;
  title: string;
  choices: Choice[];
}

// ── colour helpers ────────────────────────────────────────────────────────

const triple = (hex: string) => hexToRgb(hex).map((c) => Math.round(c * 255)).join(", ");
const darken = (hex: string, by: number) => rgbToHex(hexToRgb(hex).map((c) => c * (1 - by)));
const DARK_INK = "#0a0a0c";
const LIGHT_INK = "#ffffff";
const inkOn = (hex: string) =>
  contrastRatio(hexToRgb(DARK_INK), hexToRgb(hex)) >= contrastRatio(hexToRgb(LIGHT_INK), hexToRgb(hex))
    ? DARK_INK
    : LIGHT_INK;
const ratioOn = (hex: string) => contrastRatio(hexToRgb(inkOn(hex)), hexToRgb(hex));

const accent = (hex: string, press?: string): Vars => {
  const ink = inkOn(hex);
  return {
    accent: hex,
    "accent-press": press ?? darken(hex, 0.2),
    "ink-on-accent": ink,
    "block-current": hex,
    "block-current-ink": ink,
    "accent-rgb": triple(hex),
  };
};

const ground = (base: string, step: string, sheet: string): Vars => ({
  "surface-viewport": base,
  "surface-base": base,
  "surface-bar": base,
  "surface-bar-alt": step,
  "surface-float": sheet,
  "surface-panel": sheet,
  "surface-raised": sheet,
  "surface-raised-hover": step,
  "plane-ground": base,
  "plane-step": step,
  "plane-sheet": sheet,
  "block-kind-ink": sheet,
});

const inks = (ink: string, ink2: string, ink3: string): Vars => ({
  ink,
  "ink-2": ink2,
  "ink-3": ink3,
  "ink-rgb": triple(ink),
  "block-kind": ink,
  "viewport-ink": ink,
});

const lines = (rule: string, strong: string, heavy: string): Vars => ({
  rule,
  "rule-strong": strong,
  "rule-heavy": heavy,
});

/** Module: the line box, and every height derived from it. */
const module = (line: number, label: string): Vars => ({
  "control-tag": `${line}px`,
  "control-sm": `${line + 8}px`,
  "control-lg": `${line + 16}px`,
  "control-xl": `${line + 24}px`,
  "--line-box": `${line}px`,
  "--module": label,
});

/** Type scale: three sizes; leadings come from the module's line box. */
const typeScale = (caps: number, text: number, title: number) => (line: number): Vars => ({
  "text-caps": `600 ${caps}px / ${line - 4}px var(--font-mono)`,
  "text-body": `400 ${text}px / ${line}px var(--font-mono)`,
  "text-title": `600 ${title}px / ${line + 4}px var(--font-mono)`,
  "text-prose": `400 ${text}px / ${line + 4}px var(--font-sans)`,
});

const MONO_FALLBACK = "ui-monospace, SFMono-Regular, Menlo, monospace";
const SANS_FALLBACK = 'system-ui, -apple-system, "Segoe UI", sans-serif';

const AXES: Axis[] = [
  {
    id: "mono",
    title: "Mono font",
    choices: [
      { id: "system", label: "System", note: "SF Mono, Menlo", vars: () => ({ "font-mono": MONO_FALLBACK }) },
      { id: "jetbrains", label: "JetBrains Mono", font: "JetBrains+Mono:wght@400;500;600", vars: () => ({ "font-mono": `"JetBrains Mono", ${MONO_FALLBACK}` }) },
      { id: "plex-mono", label: "IBM Plex Mono", note: "the system's; bundled with the app", vars: () => ({ "font-mono": `"IBM Plex Mono", ${MONO_FALLBACK}` }) },
      { id: "space-mono", label: "Space Mono", font: "Space+Mono:wght@400;700", vars: () => ({ "font-mono": `"Space Mono", ${MONO_FALLBACK}` }) },
      { id: "fira-code", label: "Fira Code", font: "Fira+Code:wght@400;500;600", vars: () => ({ "font-mono": `"Fira Code", ${MONO_FALLBACK}` }) },
    ],
  },
  {
    id: "sans",
    title: "Sans font",
    choices: [
      { id: "system", label: "System", vars: () => ({ "font-sans": SANS_FALLBACK }) },
      { id: "inter", label: "Inter", font: "Inter:wght@400;500;600", vars: () => ({ "font-sans": `"Inter", ${SANS_FALLBACK}` }) },
      { id: "plex-sans", label: "IBM Plex Sans", font: "IBM+Plex+Sans:wght@400;500;600", vars: () => ({ "font-sans": `"IBM Plex Sans", ${SANS_FALLBACK}` }) },
      { id: "source-sans", label: "Source Sans 3", font: "Source+Sans+3:wght@400;500;600", vars: () => ({ "font-sans": `"Source Sans 3", ${SANS_FALLBACK}` }) },
    ],
  },
  {
    id: "accent",
    title: "Accent",
    choices: [
      { id: "orange", label: "Orange", vars: () => accent("#f87318", "#c85d00") },
      { id: "vermilion", label: "Vermilion", vars: () => accent("#ff5a1f") },
      { id: "amber", label: "Amber", note: "the system's", vars: () => accent("#ffb000", "#a87400") },
      { id: "magenta", label: "Magenta", vars: () => accent("#e5007d") },
    ],
  },
  {
    id: "ground",
    title: "Ground",
    choices: [
      { id: "cool-paper", label: "Cool paper", vars: () => ground(CHROME["surface-base"], CHROME["surface-bar-alt"], CHROME["surface-float"]) },
      { id: "warm-paper", label: "Warm paper", vars: () => ground("#e9e6df", "#efece6", "#f9f7f3") },
      { id: "white", label: "White", vars: () => ground("#f1f1f2", "#e9e9eb", "#ffffff") },
    ],
  },
  {
    id: "ink",
    title: "Ink",
    choices: [
      { id: "neutral", label: "Neutral", vars: () => inks("#18161a", "#48464d", "#605e65") },
      { id: "warm", label: "Warm", note: "the system's; held under 0.04 chroma", vars: () => inks(CHROME.ink, CHROME["ink-2"], CHROME["ink-3"]) },
      { id: "cool", label: "Cool", vars: () => inks("#12161c", "#3f4752", "#59616d") },
    ],
  },
  {
    id: "lines",
    title: "Lines",
    choices: [
      { id: "light", label: "Light", note: "the system's; edge held at 3:1", vars: () => lines(CHROME.rule, CHROME["rule-strong"], CHROME["rule-heavy"]) },
      { id: "standard", label: "Standard", vars: () => lines("#9f9da5", "#747278", "#36343b") },
      { id: "strong", label: "Strong", vars: () => lines("#86848c", "#5c5a61", "#242228") },
    ],
  },
  {
    id: "module",
    title: "Module",
    choices: [
      { id: "l16", label: "16 · 24 · 32 · 40", note: "the system: line box 16", vars: () => module(16, "l16") },
      { id: "l20", label: "20 · 28 · 36 · 44", note: "roomier: line box 20", vars: () => module(20, "l20") },
      { id: "current", label: "20 · 28 · 32 · 36", note: "off the module: 28 and 36 are not a line box plus a distance", vars: () => ({ "control-tag": "20px", "control-sm": "28px", "control-lg": "32px", "control-xl": "36px", "--line-box": "16px", "--module": "current" }) },
    ],
  },
  {
    id: "type",
    title: "Type scale",
    choices: [
      { id: "compact", label: "10 · 11 · 14", vars: () => ({}) },
      { id: "proposed", label: "11 · 12 · 15", vars: () => ({}) },
      { id: "large", label: "12 · 13 · 16", vars: () => ({}) },
    ],
  },
  {
    id: "corners",
    title: "Corners",
    choices: [
      { id: "square", label: "Square", vars: () => ({ radius: "0px" }) },
      { id: "soft", label: "2px", vars: () => ({ radius: "2px" }) },
      { id: "round", label: "4px", vars: () => ({ radius: "4px" }) },
    ],
  },
];

const TYPE_SCALES: Record<string, [number, number, number]> = {
  compact: [10, 11, 14],
  proposed: [11, 12, 15],
  large: [12, 13, 16],
};

/** The system's selection: what tokens.ts carries. */
const DEFAULTS: Record<string, string> = {
  mono: "plex-mono",
  sans: "system",
  accent: "amber",
  ground: "cool-paper",
  ink: "warm",
  lines: "light",
  module: "l16",
  type: "proposed",
  corners: "square",
};

const STORAGE_KEY = "cadjoint.design.variations.v2";

function load(): Record<string, string> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? { ...DEFAULTS, ...JSON.parse(raw) } : { ...DEFAULTS };
  } catch {
    return { ...DEFAULTS };
  }
}

const loadedFonts = new Set<string>();
function loadFont(family: string) {
  if (loadedFonts.has(family)) return;
  loadedFonts.add(family);
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = `https://fonts.googleapis.com/css2?family=${family}&display=swap`;
  document.head.appendChild(link);
}

export function Variations() {
  const [open, setOpen] = createSignal(false);
  const [picked, setPicked] = createSignal<Record<string, string>>(load());

  const choice = (axis: Axis) => axis.choices.find((c) => c.id === picked()[axis.id]) ?? axis.choices[0];

  /** Every custom property the current selection sets, resolved. */
  const resolved = createMemo<Vars>(() => {
    const out: Vars = {};
    for (const axis of AXES) Object.assign(out, choice(axis).vars());
    const line = parseInt(out["--line-box"] ?? "16", 10);
    const [caps, text, title] = TYPE_SCALES[picked().type] ?? TYPE_SCALES.proposed;
    Object.assign(out, typeScale(caps, text, title)(line));
    return out;
  });

  const apply = () => {
    const root = document.documentElement;
    for (const [name, value] of Object.entries(resolved())) {
      if (name.startsWith("--")) continue;
      root.style.setProperty(`--${name}`, value);
    }
    for (const axis of AXES) {
      const font = choice(axis).font;
      if (font) loadFont(font);
    }
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(picked()));
    } catch {
      // A private window: the selection lives until the page closes.
    }
  };

  onMount(apply);

  const pick = (axis: string, id: string) => {
    setPicked({ ...picked(), [axis]: id });
    apply();
  };

  const reset = () => {
    for (const name of Object.keys(resolved())) document.documentElement.style.removeProperty(`--${name}`);
    setPicked({ ...DEFAULTS });
    apply();
  };

  const download = () => {
    const values = resolved();
    const line = parseInt(values["--line-box"] ?? "16", 10);
    const [caps, text, title] = TYPE_SCALES[picked().type] ?? TYPE_SCALES.proposed;
    const file = {
      kind: "cadjoint-design-choices",
      version: 1,
      created: new Date().toISOString(),
      selections: Object.fromEntries(
        AXES.map((axis) => [axis.id, { id: choice(axis).id, label: choice(axis).label }]),
      ),
      fonts: { mono: values["font-mono"], sans: values["font-sans"] },
      colours: Object.fromEntries(
        Object.entries(values).filter(([, value]) => value.startsWith("#")),
      ),
      accentContrast: Number(ratioOn(values.accent ?? CHROME.accent).toFixed(2)),
      module: {
        unit: 4,
        lineBox: line,
        heights: { tag: line, cell: line + 8, bar: line + 16, toolbar: line + 24 },
      },
      distances: { gap: 4, inset: 8, section: 16 },
      type: {
        caps: { size: caps, weight: 600, leading: line - 4 },
        body: { size: text, weight: 400, leading: line },
        title: { size: title, weight: 600, leading: line + 4 },
        prose: { size: text, weight: 400, leading: line + 4, family: "sans" },
      },
      radius: values.radius,
      cssVariables: Object.fromEntries(
        Object.entries(values)
          .filter(([name]) => !name.startsWith("--"))
          .map(([name, value]) => [`--${name}`, value]),
      ),
    };
    const blob = new Blob([JSON.stringify(file, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "cadjoint-design-choices.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <>
      <button
        type="button"
        class={open() ? "dl-btn h-sm pressed" : "dl-btn h-sm"}
        onClick={() => setOpen(!open())}
        aria-expanded={open()}
        data-testid="variations-toggle"
      >
        Variations
      </button>
      <Show when={open()}>
        <aside class="dl-variations" aria-label="Variations">
          <div class="l-split">
            <span class="t-title">Variations</span>
            <button type="button" class="dl-btn quiet h-sm" onClick={() => setOpen(false)}>
              Close
            </button>
          </div>
          <p class="t-prose" style={{ color: "var(--ink-2)", margin: "0" }}>
            Each choice restyles the whole page. Download keeps the selection and every resolved
            value; tokens.ts can be set from the file.
          </p>
          <For each={AXES}>
            {(axis) => (
              <section class="dl-variations-axis">
                <span class="t-caps" style={{ color: "var(--ink-3)" }}>
                  {axis.title}
                </span>
                <div class="dl-variations-choices">
                  <For each={axis.choices}>
                    {(c) => (
                      <button
                        type="button"
                        class={picked()[axis.id] === c.id ? "active" : ""}
                        onClick={() => pick(axis.id, c.id)}
                        title={c.note}
                      >
                        <Show when={axis.id === "accent"}>
                          <i style={{ background: c.vars().accent }} />
                        </Show>
                        <Show when={axis.id === "ground"}>
                          <i style={{ background: c.vars()["surface-float"], outline: "var(--line)" }} />
                        </Show>
                        {c.label}
                      </button>
                    )}
                  </For>
                </div>
                <Show when={axis.id === "accent"}>
                  <small class="t-body" style={{ color: "var(--ink-3)" }}>
                    ink on accent {ratioOn(resolved().accent ?? CHROME.accent).toFixed(2)}:1
                  </small>
                </Show>
              </section>
            )}
          </For>
          <div class="l-split">
            <button type="button" class="dl-btn h-sm quiet" onClick={reset}>
              Reset
            </button>
            <button type="button" class="dl-btn h-sm primary" onClick={download} data-testid="variations-download">
              Download choices
            </button>
          </div>
        </aside>
      </Show>
    </>
  );
}
