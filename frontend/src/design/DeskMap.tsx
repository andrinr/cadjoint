/**
 * A map of the playground's desk, drawn from the data the dock is built from.
 *
 * The regions come from the shell's fixed furniture; the right-hand column
 * comes from `DEFAULT_LAYOUTS` and `WINDOW_DEFS` in `windows/panels.ts`, so
 * when a mode's default arrangement changes, this map changes with it.
 * Pointing at a region names the question it answers.
 */

import { For, Show, createMemo, createSignal } from "solid-js";
import { EDITING_MODES, type EditingMode } from "../editingMode";
import { DEFAULT_LAYOUTS, WINDOW_DEFS, windowTitle, type WindowId } from "../windows/panels";

interface Region {
  id: string;
  name: string;
  answers: string;
}

const REGIONS: Record<string, Region> = {
  menu: { id: "menu", name: "Menu bar", answers: "What can the application do? File, Edit, Window (every window, by its tab name), Help." },
  modes: { id: "modes", name: "Mode switcher", answers: "Where am I in the pipeline? Model → Sketch → Simulate." },
  program: { id: "program", name: "Program status · Run", answers: "Does the screen match the code? The running-work chip, the status, Run." },
  editor: { id: "editor", name: "scene.py", answers: "What is true? The program; every edit lands here." },
  viewport: { id: "viewport", name: "Viewport", answers: "What does the program make? The picture, its tools and how it is drawn in the bar across its top, and direct manipulation of it." },
  column: { id: "column", name: "Right column", answers: "What is selected, and what follows from it? Setup above, outcomes below." },
  tray: { id: "tray", name: "Tray", answers: "What is parked? Monitors and browsers, one click away: Scenes, Processes." },
};

/** The right column as groups of tabs, top to bottom, for one mode. */
function columnGroups(mode: EditingMode): WindowId[][] {
  const groups: WindowId[][] = [];
  const groupOf = new Map<WindowId, number>();
  for (const step of DEFAULT_LAYOUTS[mode]) {
    if (step.id === "viewport" || step.id === "editor") continue;
    if (step.reference === "viewport" || step.direction === "below") {
      groups.push([step.id]);
      groupOf.set(step.id, groups.length - 1);
    } else if (step.direction === "within" && step.reference && groupOf.has(step.reference)) {
      const index = groupOf.get(step.reference)!;
      groups[index].push(step.id);
      groupOf.set(step.id, index);
    }
  }
  return groups;
}

const PARKED = WINDOW_DEFS.filter((def) => def.parked).map((def) => def.title);

export function DeskMap() {
  const [mode, setMode] = createSignal<EditingMode>("model");
  const [focus, setFocus] = createSignal<string>("column");
  const groups = createMemo(() => columnGroups(mode()));
  const cls = (id: string, extra = "") => `dl-map-region ${extra}${focus() === id ? " focus" : ""}`;
  return (
    <div class="dl-stack">
      <div class="dl-map">
        <div class={cls("menu", "bar")} tabIndex={0} onMouseEnter={() => setFocus("menu")} onFocus={() => setFocus("menu")} style={{ "grid-area": "menu" }}>
          <span>File · Edit · Window · Help</span>
          <small>scene name · saved state</small>
        </div>
        <div class={cls("modes")} tabIndex={0} onMouseEnter={() => setFocus("modes")} onFocus={() => setFocus("modes")} style={{ "grid-area": "modes" }}>
          <span class="dl-map-modes">
            <For each={EDITING_MODES}>
              {(name) => (
                <button type="button" class={mode() === name ? "on" : ""} onClick={() => setMode(name)}>
                  {name}
                </button>
              )}
            </For>
          </span>
        </div>
        <div class={cls("program", "bar")} tabIndex={0} onMouseEnter={() => setFocus("program")} onFocus={() => setFocus("program")} style={{ "grid-area": "program" }}>
          <span>jobs · status · run</span>
        </div>
        <div class={cls("editor", "sheet")} tabIndex={0} onMouseEnter={() => setFocus("editor")} onFocus={() => setFocus("editor")} style={{ "grid-area": "editor" }}>
          <b>scene.py</b>
          <small>the program</small>
        </div>
        <div class={cls("viewport", "viewport")} tabIndex={0} onMouseEnter={() => setFocus("viewport")} onFocus={() => setFocus("viewport")} style={{ "grid-area": "viewport" }}>
          <b>Viewport</b>
          <small class="tl">bar: tools · · · projection · render</small>
          <small class="bl">legend</small>
          <small class="br">performance</small>
          <small class="bc">hint</small>
        </div>
        <div class={cls("column", "sheet column")} tabIndex={0} onMouseEnter={() => setFocus("column")} onFocus={() => setFocus("column")} style={{ "grid-area": "column" }}>
          <For each={groups()}>
            {(group) => (
              <div class="dl-map-group">
                <span>
                  <For each={group}>{(id, index) => <i class={index() === 0 ? "on" : ""}>{windowTitle(id)}</i>}</For>
                </span>
              </div>
            )}
          </For>
        </div>
        <div class={cls("tray")} tabIndex={0} onMouseEnter={() => setFocus("tray")} onFocus={() => setFocus("tray")} style={{ "grid-area": "tray" }}>
          <span>parked: {PARKED.join(" · ")}</span>
        </div>
      </div>
      <Show when={REGIONS[focus()]}>
        {(r) => (
          <p class="dl-map-answer">
            <b>{r().name}</b>
            {r().answers}
          </p>
        )}
      </Show>
    </div>
  );
}
