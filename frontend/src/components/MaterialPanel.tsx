/**
 * Source-backed material browser.
 *
 * Cards preview the appearance declared in Python. Editing a control patches
 * that Material(...) call; dragging a card onto the viewport assigns its
 * Python variable to the object under the pointer.
 *
 * The inspector is two sections because a Material answers to two readers.
 * **Appearance** is what the renderer uses — colour, roughness, metallic,
 * opacity — and every control there writes back to the source. **Physical**
 * is what the *solver* uses: density, conductivity, the elastic constants,
 * stated in SI with the unit printed beside the number.
 *
 * ── Why the physical rows are boxes and not sliders ──────────────────────
 * These were read-only, on the argument that a study's result depends on them
 * and they should be shown as declared rather than nudged. Half of that is
 * right and the conclusion was wrong: a number a solve depends on is exactly
 * the number you want to be able to *type*, to a stated significant figure,
 * rather than approach with a slider — 7e10 and 6.9e10 are different materials
 * and no slider distinguishes them. So each stated row is a number box that
 * commits on change, and each unstated one is a button that starts stating it.
 * Nothing here is a drag: an appearance you judge by eye gets a slider, a
 * quantity you judge by its value gets a field.
 *
 * ── Errors belong on the row ─────────────────────────────────────────────
 * The server holds every property to the same bracket the optimizer does, and
 * refuses outside it with a message naming the unit ("`density` must be a
 * number from 1 to 25000 kg/m^3."). That is an answer about one row, so it is
 * shown under that row rather than in the status line at the other end of the
 * window, where it would read as "something, somewhere, went wrong".
 */

import { For, Show, createEffect, createMemo, createSignal } from "solid-js";
import * as api from "../api";
import { busy, materials, nodes, source } from "../state";
import { NumberInput } from "./ui";
import { hasPhysicalBlock, physicalRows, type PhysicalRow } from "../materialProperties";
import type { MaterialDefinition } from "../types";

export const MATERIAL_DRAG_TYPE = "application/x-cadjoint-material";

export interface MaterialPanelProps {
  onCreate: () => Promise<void>;
  onSetValue: (
    line: number,
    argument: string,
    value: number | number[],
  ) => Promise<void>;
  /**
   * Adopt server-produced source, exactly as a patch response is adopted.
   *
   * The physical rows do their own `/patch` round trip rather than going
   * through the shell's `applyPatch`, for one reason: `applyPatch` turns a
   * refusal into a status-line message and returns nothing, and this panel
   * needs the refusal itself to put beside the row that caused it. What it
   * does *not* need is its own idea of what a committed edit means, so the
   * source it gets back is handed straight back to the shell.
   */
  onAdoptSource: (source: string) => Promise<void>;
}

/**
 * How a physical value is stepped in its own box.
 *
 * A density is counted in hundreds and a Poisson ratio in hundredths; one
 * `step` cannot serve both, and `any` — which is what a single box would have
 * to use — disables the spinner entirely. The magnitude of the number already
 * on the row is the best available answer, and for an empty row the property's
 * own scale is: both come out as "one part in a hundred of what this is".
 */
function stepFor(row: PhysicalRow): string {
  const magnitude = Math.abs(row.value ?? 0);
  if (magnitude === 0) return "any";
  const decade = Math.pow(10, Math.floor(Math.log10(magnitude)) - 2);
  return String(decade);
}

/** A first number for a property nobody has stated: the row's own scale. */
const SEED_VALUE: Record<string, number> = {
  density: 1000,
  conductivity: 1,
  specific_heat: 1000,
  youngs_modulus: 1e9,
  poisson_ratio: 0.3,
  thermal_expansion: 1e-5,
  yield_strength: 1e8,
};

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

  /** The property whose edit was refused, and what the server said. */
  const [refusal, setRefusal] = createSignal<{ key: string; message: string } | null>(null);
  const [pending, setPending] = createSignal<string | null>(null);

  /**
   * Write one physical property, or take it back out of the call.
   *
   * `null` removes the keyword — the property returns to unstated, which is a
   * real state and not the same as writing a zero into it. `expand` converts a
   * catalogue-built material (`Material.copper()`, which has no keywords at
   * all) into the literal `Material(...)` an edit needs; it is never sent on
   * the first try, because rewriting somebody's one-line declaration into a
   * twelve-line one is a thing to be asked for, not assumed. The server's own
   * refusal says so, and the row offers the retry.
   */
  const writeProperty = async (
    material: MaterialDefinition,
    key: string,
    value: number | null,
    expand = false,
  ) => {
    setPending(key);
    setRefusal(null);
    try {
      const result = await api.patch({
        source: source(),
        op: "set_material_property",
        // Both halves of `Targeted`: `stableId` is the durable name and the
        // one the server prefers, `line` is what the last compile reported and
        // the only handle a material that never earned an identity has. An
        // `expand: true` retry rewrites a one-line `Material.copper()` into a
        // multi-line literal, which moves every material declared below it —
        // so addressing by line alone is addressing by a number the edit
        // itself invalidates.
        ...(material.stableId ? { id: material.stableId } : {}),
        line: material.line,
        property: key,
        value,
        ...(expand ? { expand: true } : {}),
      });
      if (!result.ok || !result.source) {
        setRefusal({ key, message: result.error ?? "The edit was refused." });
        return;
      }
      await props.onAdoptSource(result.source);
    } catch (error) {
      setRefusal({
        key,
        message: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setPending(null);
    }
  };

  /** A refusal the reader can act on: the material is catalogue-built. */
  const canExpand = () => (refusal()?.message ?? "").includes("expand: true");

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
                  <span>Physical</span>
                </div>
                <dl class="material-physical" data-testid="material-physical">
                  <For each={physicalRows(material())}>
                    {(row) => (
                      <div
                        class="material-physical-row"
                        classList={{
                          unstated: row.state === "unstated",
                          opaque: row.state === "opaque",
                          free: row.free,
                        }}
                        data-state={row.state}
                        data-testid={`material-physical-${row.key}`}
                      >
                        <dt title={row.free ? "An optimization may move this" : undefined}>
                          {row.label}
                        </dt>
                        <dd>
                          <Show
                            when={row.state === "stated"}
                            fallback={
                              <Show
                                when={row.state === "unstated"}
                                fallback={
                                  // A keyword whose value is an expression: the
                                  // number is not ours to overwrite, and saying
                                  // so is more use than an inert box.
                                  <em title="Stated as an expression — edit it in the code">
                                    in code
                                  </em>
                                }
                              >
                                <button
                                  type="button"
                                  class="material-physical-state"
                                  disabled={
                                    busy() || !material().editable || pending() !== null
                                  }
                                  title={`State a ${row.label.toLowerCase()} for ${material().name}`}
                                  onClick={() =>
                                    void writeProperty(
                                      material(),
                                      row.key,
                                      SEED_VALUE[row.key] ?? 1,
                                    )
                                  }
                                  data-testid={`material-state-${row.key}`}
                                >
                                  state
                                </button>
                              </Show>
                            }
                          >
                            <NumberInput
                              value={row.value ?? 0}
                              step={stepFor(row)}
                              disabled={busy() || !material().editable || pending() !== null}
                              title={`${row.label} in ${row.unit || "a dimensionless ratio"}`}
                              testId={`material-value-${row.key}`}
                              onCommit={(value) =>
                                void writeProperty(material(), row.key, value)
                              }
                            />
                            <button
                              type="button"
                              class="material-physical-clear"
                              disabled={busy() || !material().editable || pending() !== null}
                              title={`Remove ${row.label.toLowerCase()} from the declaration`}
                              aria-label={`Remove ${row.label}`}
                              onClick={() => void writeProperty(material(), row.key, null)}
                              data-testid={`material-clear-${row.key}`}
                            >
                              ×
                            </button>
                          </Show>
                          <Show when={row.unit}>
                            <small>{row.unit}</small>
                          </Show>
                        </dd>
                        <Show when={refusal()?.key === row.key}>
                          <p class="material-physical-error" data-testid={`material-error-${row.key}`}>
                            {refusal()!.message}
                            <Show when={canExpand()}>
                              <button
                                type="button"
                                onClick={() =>
                                  void writeProperty(
                                    material(),
                                    row.key,
                                    row.value ?? SEED_VALUE[row.key] ?? 1,
                                    true,
                                  )
                                }
                                data-testid={`material-expand-${row.key}`}
                              >
                                Write it out as a Material(…)
                              </button>
                            </Show>
                          </p>
                        </Show>
                      </div>
                    )}
                  </For>
                </dl>
              </Show>
            </section>
          )}
        </Show>
    </aside>
  );
}
