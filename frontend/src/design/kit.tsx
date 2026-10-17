/**
 * What the design page is written with: a section, a rule and its plates,
 * and small specimens of the app's components that several rules show.
 *
 * Every number the page states about colour is computed here from
 * `tokens.ts`, so the page cannot quote a contrast that has stopped holding.
 */

import { For, Show, children, type JSX } from "solid-js";
import { contrastRatio } from "../simColors";
import { CHROME, hexToRgb, type ChromeToken } from "../tokens";

export const v = (token: ChromeToken | string): string => `var(--${token})`;

/**
 * Literal values the rules forbid. They appear only inside Don't plates, as
 * specimens of the pattern a rule excludes.
 */
export const DONT = {
  hoverTint: "#ff9e40",
  className: CHROME.danger,
  string: CHROME.ok,
  number: CHROME.warn,
  keyword: "#0065b4",
} as const;

// ── page structure ────────────────────────────────────────────────────────

/** One section of the page; its number comes from its place in the shell. */
export interface SectionDef {
  id: string;
  title: string;
  intro?: string;
  body: () => JSX.Element;
}

export function Section(props: { id: string; num: string; title: string; intro?: string; children: JSX.Element }) {
  return (
    <section class="dl-section" id={props.id}>
      <h2>
        <span class="dl-kicker">{props.num}</span>
        {props.title}
      </h2>
      <Show when={props.intro}>
        <p class="dl-prose">{props.intro}</p>
      </Show>
      {props.children}
    </section>
  );
}

/**
 * One rule: its id in the margin, the statement, the reason, and the plates
 * that show it. One plate spans the column; several sit side by side.
 */
export function Rule(props: { id: string; children: JSX.Element; why?: JSX.Element; examples?: JSX.Element }) {
  const examples = children(() => props.examples);
  return (
    <div class="dl-rule" id={`rule-${props.id}`}>
      <code>{props.id}</code>
      <div>
        <p>{props.children}</p>
        <Show when={props.why}>
          <p>{props.why}</p>
        </Show>
        <Show when={examples.toArray().length}>
          <div class={examples.toArray().length === 1 ? "dl-examples single" : "dl-examples"}>{examples()}</div>
        </Show>
      </div>
    </div>
  );
}

export type PlateKind = "example" | "do" | "dont" | "try";

const PLATE_WORDS: Record<PlateKind, string> = {
  example: "Example",
  do: "Do",
  dont: "Don't",
  try: "Try it",
};

/** An example: a sheet the reader looks at or works in. */
export function Plate(props: {
  kind?: PlateKind;
  label: string;
  /** Set the stage on the ground rather than the sheet. */
  ground?: boolean;
  /** Reserve room under a menu bar for an open dropdown. */
  menu?: boolean;
  children: JSX.Element;
}) {
  const kind = () => props.kind ?? "example";
  return (
    <figure class="dl-plate">
      <header>
        <span class={kind() === "do" || kind() === "dont" ? kind() : undefined}>{PLATE_WORDS[kind()]}</span>
        <span>· {props.label}</span>
      </header>
      <div classList={{ "dl-stage": true, ground: !!props.ground, "with-menu": !!props.menu }}>{props.children}</div>
    </figure>
  );
}

// ── measured colour ───────────────────────────────────────────────────────

/** WCAG contrast between two tokens (or raw hex specimens). */
export function ratio(fg: ChromeToken | string, bg: ChromeToken | string): number {
  const hex = (value: string) => (value.startsWith("#") ? value : CHROME[value as ChromeToken]);
  return contrastRatio(hexToRgb(hex(fg)), hexToRgb(hex(bg)));
}

/**
 * A measured ratio and its verdict, as a word as well as a hue (A4). `need`
 * is the bar the pair owes: 4.5 for text, 3 for a mark, a band for furniture.
 */
export function Ratio(props: { fg: ChromeToken | string; bg: ChromeToken | string; need?: number | [number, number] }) {
  const value = () => ratio(props.fg, props.bg);
  const verdict = () => {
    const need = props.need;
    if (need === undefined) return null;
    if (Array.isArray(need)) {
      return value() >= need[0] && value() <= need[1] ? { word: "BAND", cls: "band" } : { word: "OFF BAND", cls: "fail" };
    }
    return value() >= need ? { word: "PASS", cls: "pass" } : { word: "FAIL", cls: "fail" };
  };
  return (
    <span class="dl-ratio">
      {value().toFixed(2)}:1
      <Show when={verdict()}>{(result) => <em class={result().cls}>{result().word}</em>}</Show>
    </span>
  );
}

/** A colour swatch with its token, its value and its measured contrasts. */
export function Swatch(props: { token: ChromeToken; job: string; need?: number | [number, number]; ink?: ChromeToken }) {
  return (
    <div class="dl-swatch">
      <i style={{ background: v(props.token), color: v(props.ink ?? "ink") }}>{props.ink ? "Aa" : ""}</i>
      <div>
        <b>{props.token}</b>
        <small>{CHROME[props.token]}</small>
        <small>
          ground <Ratio fg={props.token} bg="surface-base" need={props.need} />
        </small>
        <small>
          sheet <Ratio fg={props.token} bg="surface-float" need={props.need} />
        </small>
        <small>{props.job}</small>
      </div>
    </div>
  );
}

// ── specimens of the app's components ─────────────────────────────────────

/** The × glyph a removable chip carries, drawn so it follows the ink. */
export const Cross = () => (
  <svg width="8" height="8" viewBox="0 0 8 8" stroke="currentColor" stroke-width="1.25">
    <path d="M1 1L7 7M7 1L1 7" />
  </svg>
);

/** A value chip with its remove control (K4). */
export function Chip(props: { children: JSX.Element; violated?: boolean; title?: string; onRemove?: () => void }) {
  return (
    <span class={props.violated ? "dl-chip violated" : "dl-chip"}>
      {props.children}
      <button type="button" title={props.title} onClick={() => props.onRemove?.()}>
        <Cross />
      </button>
    </span>
  );
}

/** The status line: a mark, then words (K7). */
export const Status = (props: { stale?: boolean; children: JSX.Element }) => (
  <span class={props.stale ? "dl-status stale" : "dl-status"}>
    <i />
    {props.children}
  </span>
);

/** label | value | unit, with an optional second line under the value. */
export type RowSpec = readonly [label: string, value: string, unit?: string, sub?: string];

/** Ruled rows on one baseline, values right-hung (T3). */
export function Rows(props: { rows: readonly RowSpec[]; free?: string }) {
  return (
    <div class="dl-rows">
      <For each={props.rows}>
        {([label, value, unit, sub]) => (
          <div class={label === props.free ? "free" : undefined}>
            <span>{label}</span>
            <b>{value}</b>
            <small>{unit}</small>
            <Show when={sub}>
              <sub>{sub}</sub>
            </Show>
          </div>
        )}
      </For>
    </div>
  );
}

/** A panel: tab strip, kicker and title, then its body (K1). */
export function MiniPanel(props: {
  kicker: string;
  title: string;
  tabs?: string[];
  /** A forbidden fill for the kicker, for Don't plates. */
  kickerFill?: ChromeToken;
  children: JSX.Element;
}) {
  return (
    <div class="dl-panel">
      <nav>
        <For each={props.tabs ?? [props.title]}>
          {(tab, index) => (
            <button type="button" class={index() === 0 ? "active" : ""}>
              {tab}
            </button>
          )}
        </For>
      </nav>
      <div>
        <div class="dl-panel-head">
          <Show when={props.kickerFill} fallback={<span class="dl-kicker">{props.kicker}</span>}>
            <span class="dl-dont-chip" style={{ background: v(props.kickerFill!), color: v("ink-on-accent") }}>
              {props.kicker}
            </span>
          </Show>
          <strong>{props.title}</strong>
        </div>
        {props.children}
      </div>
    </div>
  );
}

/** One menu row: the leading mark column, the label, the key column (K5). */
export function MenuItem(props: {
  mark?: string;
  label: JSX.Element;
  keys?: string;
  disabled?: boolean;
  title?: string;
  hover?: boolean;
}) {
  return (
    <button
      type="button"
      disabled={props.disabled}
      title={props.title}
      style={props.hover ? { background: v("surface-bar-alt") } : undefined}
    >
      <span>{props.mark}</span>
      <span>{props.label}</span>
      <kbd>{props.keys}</kbd>
    </button>
  );
}

/** The mode switcher, with one segment current. */
export const Seg = (props: { items?: readonly string[]; active: string }) => (
  <div class="dl-seg">
    <For each={props.items ?? ["Model", "Sketch", "Simulate"]}>
      {(item) => (
        <button type="button" class={item === props.active ? "active" : ""}>
          {item}
        </button>
      )}
    </For>
  </div>
);

/** One numbered line of a source excerpt, inside `pre.dl-code`. */
export const Line = (props: { n: number; children: JSX.Element }) => (
  <div>
    <span>{props.n}</span>
    <span>{props.children}</span>
  </div>
);

/** A ramp legend: name, low end, the ramp, high end (C7). */
export const Legend = (props: { name: string; from: string; to: string; ramp: string }) => (
  <div class="dl-legend">
    <b>{props.name}</b>
    <span>{props.from}</span>
    <i style={{ background: props.ramp }} />
    <span>{props.to}</span>
  </div>
);
