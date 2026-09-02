/**
 * Source-backed material browser.
 *
 * Cards preview the appearance declared in Python. Editing a control patches
 * that Material(...) call; dragging a card onto the viewport assigns its
 * Python variable to the object under the pointer.
 *
 * The inspector is two numbered sections because a Material answers to two
 * readers. **01 Appearance** is what the renderer uses — colour, roughness,
 * metallic, opacity — and every control there writes back to the source.
 * **02 Physical** is what the *solver* uses — density, conductivity, the
 * elastic constants — stated in SI with the unit printed beside the number,
 * and read-only for now: these are the numbers a study's result depends on,
 * so they are shown exactly as declared rather than nudged with a slider.
 */

import { For, Show, createEffect, createMemo, createSignal } from "solid-js";
import { busy, materials, nodes } from "../state";
import { windowManager } from "../windows/manager";
import { hasPhysicalBlock, physicalRows } from "../materialProperties";
import type { MaterialDefinition } from "../types";

export const MATERIAL_DRAG_TYPE = "application/x-cadjoint-material";

export interface MaterialPanelProps {
  onCreate: () => Promise<void>;
  onSetValue: (
    line: number,
    argument: string,
    value: number | number[],
  ) => Promise<void>;
}

function colorHex(color: readonly number[]): string {
  return (
    "#" +
    color
      .map((value) =>
        Math.round(Math.max(0, Math.min(1, value)) * 255)
          .toString(16)
          .padStart(2, "0"),
      )
      .join("")
  );
}

function colorRgb(hex: string): [number, number, number] {
  const value = hex.replace("#", "");
  return [0, 2, 4].map((offset) => Number.parseInt(value.slice(offset, offset + 2), 16) / 255) as [
    number,
    number,
    number,
  ];
}

function previewStyle(material: MaterialDefinition): string {
  const [red, green, blue] = material.color.map((value) =>
    Math.round(Math.max(0, Math.min(1, value)) * 255),
  );
  return [
    `--material-rgb: ${red}, ${green}, ${blue}`,
    `--material-roughness: ${material.roughness}`,
    `--material-metallic: ${material.metallic}`,
    `--material-opacity: ${Math.max(0.16, material.opacity)}`,
  ].join(";");
}

export function MaterialPanel(props: MaterialPanelProps) {
  const [activeId, setActiveId] = createSignal<string | null>(null);
  const active = createMemo(
    () => materials().find((material) => material.id === activeId()) ?? materials()[0],
  );

  createEffect(() => {
    const available = materials();
    if (available.length > 0 && !available.some((item) => item.id === activeId())) {
      setActiveId(available[0].id);
    }
  });

  const assignmentCount = (name: string) =>
    nodes().filter((node) => node.material === name).length;

  const setScalar = (material: MaterialDefinition, argument: string, raw: string) =>
    props.onSetValue(material.line, argument, Number(raw));

  return (
    <aside class="material-panel" data-testid="material-panel">
        <header>
          <div class="material-panel-title">
            <span class="material-panel-icon" aria-hidden="true" />
            <span>
              <small>Library</small>
              Materials
            </span>
            <b>{materials().length}</b>
          </div>
          <button
            type="button"
            class="material-add"
            disabled={busy()}
            onClick={() => void props.onCreate()}
            title="Create a new source-backed material"
            data-testid="material-add"
          >
            +
          </button>
          <button
            type="button"
            class="material-close"
            onClick={() => windowManager()?.minimise("materials")}
            title="Park the material browser in the tray"
            aria-label="Park the material browser"
            data-testid="material-close"
          >
            —
          </button>
        </header>

        <p class="material-help">Drag a swatch onto an object</p>
        <div class="material-browser">
          <For each={materials()}>
            {(material) => (
              <article
                class={`material-card ${active()?.id === material.id ? "active" : ""}`}
                draggable={material.editable}
                tabIndex={0}
                role="button"
                aria-label={`${material.name} material`}
                data-testid={`material-${material.name}`}
                onClick={() => setActiveId(material.id)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    setActiveId(material.id);
                  }
                }}
                onDragStart={(event) => {
                  event.dataTransfer?.setData(MATERIAL_DRAG_TYPE, material.name);
                  event.dataTransfer?.setData("text/plain", material.name);
                  if (event.dataTransfer) event.dataTransfer.effectAllowed = "copy";
                  setActiveId(material.id);
                }}
              >
                <div class="material-preview" style={previewStyle(material)}>
                  <span />
                </div>
                <span class="material-name">{material.name.replaceAll("_", " ")}</span>
                <small>
                  {assignmentCount(material.name)
                    ? `${assignmentCount(material.name)} applied`
                    : "unused"}
                </small>
              </article>
            )}
          </For>
        </div>

        <Show
          when={active()}
          fallback={<p class="material-empty">Create a material to begin.</p>}
        >
          {(material) => (
            <section class="material-inspector" data-testid="material-inspector">
              <div class="material-inspector-head">
                <strong>{material().name}</strong>
                <label title="Base color">
                  <span>Color</span>
                  <input
                    type="color"
                    value={colorHex(material().color)}
                    disabled={busy() || !material().editable}
                    onChange={(event) =>
                      void props.onSetValue(
                        material().line,
                        "color",
                        colorRgb(event.currentTarget.value),
                      )
                    }
                    data-testid="material-color"
                  />
                </label>
              </div>
              <div class="material-section-head">
                <i aria-hidden="true">01</i>
                <span>Appearance</span>
              </div>
              <label>
                <span>
                  Roughness <b>{material().roughness.toFixed(2)}</b>
                </span>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={material().roughness}
                  disabled={busy() || !material().editable}
                  onChange={(event) =>
                    void setScalar(material(), "roughness", event.currentTarget.value)
                  }
                  data-testid="material-roughness"
                />
              </label>
              <label>
                <span>
                  Metallic <b>{material().metallic.toFixed(2)}</b>
                </span>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={material().metallic}
                  disabled={busy() || !material().editable}
                  onChange={(event) =>
                    void setScalar(material(), "metallic", event.currentTarget.value)
                  }
                  data-testid="material-metallic"
                />
              </label>
              <label>
                <span>
                  Opacity <b>{material().opacity.toFixed(2)}</b>
                </span>
                <input
                  type="range"
                  min="0.05"
                  max="1"
                  step="0.05"
                  value={material().opacity}
                  disabled={busy() || !material().editable}
                  onChange={(event) =>
                    void setScalar(material(), "opacity", event.currentTarget.value)
                  }
                  data-testid="material-opacity"
                />
              </label>

              <Show when={hasPhysicalBlock(material())}>
                <div class="material-section-head">
                  <i aria-hidden="true">02</i>
                  <span>Physical</span>
                </div>
                <Show
                  when={physicalRows(material()).length > 0}
                  fallback={
                    <p class="material-physical-empty" data-testid="material-physical-empty">
                      — states no physical properties
                    </p>
                  }
                >
                  <dl class="material-physical" data-testid="material-physical">
                    <For each={physicalRows(material())}>
                      {(row) => (
                        <div class="material-physical-row" data-testid={`material-physical-${row.key}`}>
                          <dt>{row.label}</dt>
                          <dd>
                            <b>{row.value}</b>
                            <Show when={row.unit}>
                              <small>{row.unit}</small>
                            </Show>
                          </dd>
                        </div>
                      )}
                    </For>
                  </dl>
                </Show>
              </Show>
            </section>
          )}
        </Show>
    </aside>
  );
}
