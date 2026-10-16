/**
 * The building blocks the design page is written with: a section, a rule
 * (its id, its sentence, its examples) and an example plate.
 *
 * Every number the page states about colour is computed here from
 * `tokens.ts`, never typed: a ratio on this page is the ratio of the values
 * the app ships, so the page cannot quote a contrast that has stopped
 * holding.
 */

import { Show, type JSX } from "solid-js";
import { contrastRatio } from "../simColors";
import { CHROME, hexToRgb, type ChromeToken } from "../tokens";

export const v = (token: ChromeToken | string): string => `var(--${token})`;

/** WCAG contrast between two tokens (or raw hex specimens). */
export function ratio(fg: ChromeToken | string, bg: ChromeToken | string): number {
  const hex = (value: string) =>
    value.startsWith("#") ? value : CHROME[value as ChromeToken];
  return contrastRatio(hexToRgb(hex(fg)), hexToRgb(hex(bg)));
}

/**
 * A measured ratio and its verdict, as a word as well as a hue (A4).
 *
 * `need` is the bar the pair owes: 4.5 for text, 3 for a meaningful mark,
 * or a band [low, high] for furniture.
 */
export function Ratio(props: {
  fg: ChromeToken | string;
  bg: ChromeToken | string;
  need?: number | [number, number];
}) {
  const value = () => ratio(props.fg, props.bg);
  const verdict = () => {
    const need = props.need;
    if (need === undefined) return null;
    if (Array.isArray(need)) {
      return value() >= need[0] && value() <= need[1]
        ? { word: "BAND", cls: "band" }
        : { word: "OFF BAND", cls: "fail" };
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

export function Section(props: {
  id: string;
  num: string;
  title: string;
  intro?: JSX.Element;
  children: JSX.Element;
}) {
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
 * One rule: an id in the margin, the statement, an optional reason, and the
 * examples that show it. A rule without an example is allowed only when the
 * rule is about time or keys, which a still plate cannot show.
 */
export function Rule(props: {
  id: string;
  children: JSX.Element;
  why?: JSX.Element;
  examples?: JSX.Element;
  single?: boolean;
}) {
  return (
    <div class="dl-rule" id={`rule-${props.id}`}>
      <code>{props.id}</code>
      <div>
        <p>{props.children}</p>
        <Show when={props.why}>
          <p>{props.why}</p>
        </Show>
        <Show when={props.examples}>
          <div class={props.single ? "dl-examples single" : "dl-examples"}>
            {props.examples}
          </div>
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
  label?: string;
  caption?: JSX.Element;
  ground?: boolean;
  /** Reserve room under a menu bar for an open dropdown. */
  menu?: boolean;
  children: JSX.Element;
}) {
  const kind = () => props.kind ?? "example";
  const wordClass = () => (kind() === "do" ? "do" : kind() === "dont" ? "dont" : undefined);
  return (
    <figure class="dl-plate">
      <header>
        <span class={wordClass()}>{PLATE_WORDS[kind()]}</span>
        <Show when={props.label}>
          <span>· {props.label}</span>
        </Show>
      </header>
      <div classList={{ "dl-stage": true, ground: !!props.ground, "with-menu": !!props.menu }}>{props.children}</div>
      <Show when={props.caption}>
        <figcaption>{props.caption}</figcaption>
      </Show>
    </figure>
  );
}

/** A colour swatch with its token, its value and its measured contrasts. */
export function Swatch(props: {
  token: ChromeToken;
  job: string;
  need?: number | [number, number];
  ink?: ChromeToken;
}) {
  return (
    <div class="dl-swatch">
      <i style={{ background: v(props.token), color: v(props.ink ?? "ink") }}>
        {props.ink ? "Aa" : ""}
      </i>
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

/** The × glyph a removable chip carries, drawn so it follows the ink. */
export const Cross = () => (
  <svg width="8" height="8" viewBox="0 0 8 8" stroke="currentColor" stroke-width="1.25">
    <path d="M1 1L7 7M7 1L1 7" />
  </svg>
);
