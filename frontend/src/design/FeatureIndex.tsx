/**
 * The feature index and the menu bar, both rendered from `features.ts`.
 */

import { For, Show, createSignal, onCleanup, onMount } from "solid-js";
import { EDITING_MODES, type EditingMode } from "../editingMode";
import { FEATURES, FEATURE_GROUPS, MENU_ORDER, proposedMenus, type Homes } from "./features";

type Column = { title: string; fields: (keyof Homes)[] };

const COLUMNS: Column[] = [
  { title: "menu", fields: ["menu"] },
  { title: "toolbar · rail", fields: ["toolbar", "rail"] },
  { title: "key", fields: ["key"] },
  { title: "window · viewport", fields: ["panel", "viewport"] },
];

const join = (homes: Homes, fields: (keyof Homes)[]) =>
  fields
    .map((field) => homes[field])
    .filter(Boolean)
    .join(" · ");

export function FeatureIndex() {
  return (
    <table class="dl-table dl-features">
      <thead>
        <tr>
          <th>feature</th>
          <For each={COLUMNS}>{(column) => <th>{column.title}</th>}</For>
        </tr>
      </thead>
      <For each={FEATURE_GROUPS}>
        {(group) => {
          const rows = FEATURES.filter((feature) => feature.group === group);
          return (
            <Show when={rows.length}>
              <tbody>
                <tr class="group">
                  <th colSpan={COLUMNS.length + 1}>{group}</th>
                </tr>
                <For each={rows}>
                  {(feature) => (
                    <tr>
                      <td>
                        {feature.name}
                        <Show when={feature.modes}>
                          <small> {feature.modes!.join(" · ")}</small>
                        </Show>
                      </td>
                      <For each={COLUMNS}>{(column) => <td>{join(feature.rule, column.fields) || "—"}</td>}</For>
                    </tr>
                  )}
                </For>
              </tbody>
            </Show>
          );
        }}
      </For>
    </table>
  );
}

/** The menu bar: every command, with its key, disabled outside its mode. */
export function MenuBarSpecimen() {
  const menus = proposedMenus();
  const [open, setOpen] = createSignal<string | null>("View");
  const [mode, setMode] = createSignal<EditingMode>("sketch");
  let bar: HTMLDivElement | undefined;
  onMount(() => {
    const close = (event: MouseEvent) => {
      if (bar && !bar.contains(event.target as Node)) setOpen(null);
    };
    document.addEventListener("click", close);
    onCleanup(() => document.removeEventListener("click", close));
  });
  const applies = (modes?: EditingMode[]) => !modes || modes.includes(mode());
  return (
    <div class="dl-stack">
      <div class="dl-row-of">
        <span class="dl-status">current mode</span>
        <div class="dl-seg">
          <For each={EDITING_MODES}>
            {(name) => (
              <button type="button" class={mode() === name ? "active" : ""} onClick={() => setMode(name)}>
                {name}
              </button>
            )}
          </For>
        </div>
      </div>
      <div class="dl-menu-stage">
        <div class="dl-menubar" ref={bar}>
          <For each={MENU_ORDER}>
            {(name) => (
              <div class="dl-menubar-item">
                <button
                  type="button"
                  class={open() === name ? "open" : ""}
                  aria-expanded={open() === name}
                  onClick={() => setOpen(open() === name ? null : name)}
                  onMouseEnter={() => open() !== null && setOpen(name)}
                >
                  {name}
                </button>
                <Show when={open() === name}>
                  <div class="dl-menu dl-dropdown" role="menu">
                    <For each={menus.get(name) ?? []}>
                      {(section, index) => (
                        <>
                          <Show when={index() > 0}>
                            <hr />
                          </Show>
                          <Show when={section.title}>
                            <span class="dl-menu-label">{section.title}</span>
                          </Show>
                          <For each={section.items}>
                            {(item) => (
                              <button
                                type="button"
                                disabled={!applies(item.modes)}
                                title={applies(item.modes) ? undefined : `Only in ${item.modes!.join(" and ")} mode`}
                              >
                                <span />
                                <span>{item.label}</span>
                                <kbd>{item.key ?? ""}</kbd>
                              </button>
                            )}
                          </For>
                        </>
                      )}
                    </For>
                  </div>
                </Show>
              </div>
            )}
          </For>
        </div>
      </div>
    </div>
  );
}
