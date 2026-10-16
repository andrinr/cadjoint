/**
 * The layout grid: a guide drawn over the whole UI, to see whether things
 * sit on the unit (guide A6, S0) and to find the ones that do not.
 *
 * Toggled by View › Show › Layout grid (⇧G). It draws the 4px unit, a firmer
 * line every 16px (the section distance), and outlines in `danger` every
 * control, row and sheet whose box is off the grid by more than a hairline —
 * a 1px rule between rows shifts what follows by 1px, which is the hairline
 * the rules allow, so the tolerance is one pixel. The count sits in the
 * corner; clicking it lists the first offenders by selector, so a fix can
 * start from the stylesheet rule that placed them.
 *
 * It is a measuring tool, not part of the interface: it never takes pointer
 * events except on its own counter, and it costs nothing while off.
 */

import { For, Show, createSignal, onCleanup, onMount } from "solid-js";
import { UNIT } from "../tokens";

export const [layoutGridOn, setLayoutGridOn] = createSignal(false);

/** What the guide measures: things that are placed, not the text inside them. */
const PLACED = [
  "button",
  "input",
  "textarea",
  "[role=menu]",
  "[role=listbox]",
  "[role=treeitem]",
  "nav",
  "aside",
  "header",
  "section",
  "dl",
  "li",
  "label",
  "[class*=panel]",
  "[class*=row]",
  "[class*=chip]",
  "[class*=badge]",
  "[class*=card]",
  "[class*=tag]",
].join(",");

/** Things whose width is set by the layout, not by their text or the user. */
const FIXED_WIDTH = [
  ".tool-rail",
  ".tool-rail button",
  ".view-controls button",
  "input",
  ".ui-select",
  "[class*=badge]",
  ".view-cube",
  ".perf-readout",
].join(",");

/** How far a box edge may sit from the grid: one hairline. */
const TOLERANCE = 1;

interface Offender {
  rect: { left: number; top: number; width: number; height: number };
  what: string;
  why: string;
}

const offBy = (value: number, tolerance: number) => {
  const r = ((value % UNIT) + UNIT) % UNIT;
  return Math.min(r, UNIT - r) > tolerance;
};
/** A position may sit one hairline off (a rule above it); a size two (an edge each side). */
const off = (value: number) => offBy(value, TOLERANCE);
const offSize = (value: number) => offBy(value, TOLERANCE * 2);

/**
 * Where an element is measured from: its window's content, when it is in
 * one. A pane is as wide and as tall as the user dragged it, so the grid
 * inside it starts at its own corner, not the screen's.
 */
function originOf(element: Element): { left: number; top: number; right: number } {
  const pane = element.parentElement?.closest(".dv-content-container, .dv-groupview, .pane");
  if (!pane) return { left: 0, top: 0, right: window.innerWidth };
  const box = pane.getBoundingClientRect();
  return { left: box.left, top: box.top, right: box.right };
}

function describe(element: Element): string {
  const id = element.getAttribute("data-testid");
  const cls = [...element.classList].slice(0, 2).join(".");
  return `${element.tagName.toLowerCase()}${cls ? `.${cls}` : ""}${id ? `[${id}]` : ""}`;
}

function measure(): Offender[] {
  const found: Offender[] = [];
  const width = window.innerWidth;
  const height = window.innerHeight;
  for (const element of document.querySelectorAll(PLACED)) {
    if (element.closest("[data-layout-grid]") || element.closest(".cm-editor")) continue;
    const box = element.getBoundingClientRect();
    if (box.width < 2 || box.height < 2) continue;
    if (box.right < 0 || box.bottom < 0 || box.left > width || box.top > height) continue;
    const style = getComputedStyle(element);
    if (style.visibility === "hidden" || style.display === "none" || Number(style.opacity) === 0) continue;
    // Dock containers are sized by the dock, and scroll containers fill
    // whatever room they are given; neither height is the layout's choice.
    const fills =
      element.matches(".pane, .dv-groupview, .dv-content-container") ||
      /auto|scroll/.test(style.overflowY);
    const origin = originOf(element);
    // Vertical rhythm is on the grid everywhere. Horizontally only fixed
    // things are: a cell sized by its own text and a pane as wide as the user
    // dragged it are not measured across (guide S0).
    const fixedWidth = element.matches(FIXED_WIDTH);
    // A badge follows its text and a field its track: their widths are
    // fixed, their x is not theirs to choose.
    const fixedX = fixedWidth && !element.matches("[class*=badge], input, .ui-select");
    // Furniture anchored to a pane's right edge is on the grid from that
    // edge: either side counts.
    const x = box.left - origin.left;
    const fromRight = origin.right - box.right;
    const y = box.top - origin.top;
    const reasons = [
      fixedX && off(x) && off(fromRight) && `x ${x.toFixed(1)}`,
      !fills && off(y) && `y ${y.toFixed(1)}`,
      fixedWidth && offSize(box.width) && `w ${box.width.toFixed(1)}`,
      !fills && offSize(box.height) && `h ${box.height.toFixed(1)}`,
    ].filter(Boolean);
    if (reasons.length) {
      found.push({
        rect: { left: box.left, top: box.top, width: box.width, height: box.height },
        what: describe(element),
        why: reasons.join(" · "),
      });
    }
  }
  return found;
}

export function LayoutGrid() {
  const [offenders, setOffenders] = createSignal<Offender[]>([]);
  const [listOpen, setListOpen] = createSignal(false);
  let timer: number | undefined;
  const refresh = () => setOffenders(layoutGridOn() ? measure() : []);
  onMount(() => {
    timer = window.setInterval(() => layoutGridOn() && refresh(), 700);
    window.addEventListener("resize", refresh);
  });
  onCleanup(() => {
    clearInterval(timer);
    window.removeEventListener("resize", refresh);
  });

  return (
    <Show when={layoutGridOn()}>
      <div class="layout-grid" data-layout-grid aria-hidden="true">
        <For each={offenders()}>
          {(item) => (
            <i
              class="layout-grid-off"
              style={{
                left: `${item.rect.left}px`,
                top: `${item.rect.top}px`,
                width: `${item.rect.width}px`,
                height: `${item.rect.height}px`,
              }}
            />
          )}
        </For>
      </div>
      <div class="layout-grid-count" data-layout-grid data-testid="layout-grid-count">
        <button type="button" onClick={() => setListOpen(!listOpen())}>
          {offenders().length === 0 ? "ON GRID" : `OFF GRID ${offenders().length}`} · {UNIT}PX
        </button>
        <Show when={listOpen() && offenders().length > 0}>
          <ol>
            <For each={offenders().slice(0, 24)}>
              {(item) => (
                <li>
                  <code>{item.what}</code>
                  <small>{item.why}</small>
                </li>
              )}
            </For>
          </ol>
        </Show>
      </div>
    </Show>
  );
}
