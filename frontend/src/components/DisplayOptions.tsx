/**
 * Render settings, grouped the way a viewport panel usually is.
 *
 * Mutually exclusive choices (shading, shadows, quality) are segmented
 * controls rather than checkboxes, so the current state reads at a glance and
 * costs one click to change; independent effects stay as switches.
 */

import { For, Show, createMemo, createSignal, onCleanup } from "solid-js";
import {
  renderPresetMatches,
  type RenderPreset,
  type RenderPresetId,
} from "../renderPresets";
import type { DisplaySettings, ShadowMode } from "../viewer/renderer";
import { QUALITY_PRESETS } from "../viewer/renderer";
import { DisplayIcon } from "./icons";

const SHADOWS: { value: ShadowMode; label: string; hint: string }[] = [
  { value: "off", label: "Off", hint: "No shadow rays" },
  { value: "hard", label: "Hard", hint: "One crisp, lifted shadow ray" },
  { value: "soft", label: "Soft", hint: "Penumbra from multiple samples" },
];

const SWITCHES: { key: keyof DisplaySettings; label: string; hint: string }[] = [
  { key: "reflections", label: "Reflections", hint: "Environment reflections" },
  { key: "hideSolid", label: "Hide solid", hint: "Construction geometry only" },
  { key: "showMeshEdges", label: "Feature edges", hint: "Sharp creases, corners, and CSG seams" },
  { key: "showMeshWireframe", label: "Mesh wireframe", hint: "Full dual-contour quad wireframe" },
];

export interface DisplayOptionsProps {
  display: DisplaySettings;
  presets: RenderPreset[];
  selectedPreset: RenderPresetId;
  pathTracing: boolean;
  quality: string;
  onChange: (patch: Partial<DisplaySettings>) => void;
  onQualityChange: (key: string) => void;
  onPresetActivate: (id: RenderPresetId) => void;
  onPresetSave: (id: RenderPresetId) => void;
  onPresetReset: (id: RenderPresetId) => void;
  onPathTracingChange: (enabled: boolean) => void;
}

export function DisplayOptions(props: DisplayOptionsProps) {
  const [open, setOpen] = createSignal(false);
  const [editing, setEditing] = createSignal(false);
  const selectedPreset = createMemo(
    () =>
      props.presets.find((preset) => preset.id === props.selectedPreset) ??
      props.presets[0],
  );

  const closeOnOutside = (event: MouseEvent) => {
    if (!(event.target as HTMLElement).closest(".display-options")) setOpen(false);
  };
  document.addEventListener("click", closeOnOutside);
  onCleanup(() => document.removeEventListener("click", closeOnOutside));

  // Escape closes the popover without also clearing the viewer selection —
  // capture phase runs before the viewer's window-level Escape handler.
  const closeOnEscape = (event: KeyboardEvent) => {
    if (event.key !== "Escape" || !open()) return;
    event.stopPropagation();
    setOpen(false);
  };
  document.addEventListener("keydown", closeOnEscape, true);
  onCleanup(() => document.removeEventListener("keydown", closeOnEscape, true));

  return (
    <div class="display-options">
      <button
        type="button"
        class={`icon ${open() ? "active" : ""}`}
        onClick={() => setOpen(!open())}
        title="Render settings"
        aria-label="Render settings"
        data-testid="display-options"
      >
        <DisplayIcon />
      </button>

      <Show when={open()}>
        <div class="popover" role="menu">
          <section class="render-preset-library">
            <div class="render-preset-heading">
              <h4>Render presets</h4>
              <small>Click to activate</small>
            </div>
            <div class="render-preset-grid">
              <For each={props.presets}>
                {(preset) => {
                  const active = () =>
                    renderPresetMatches(
                      preset,
                      props.display,
                      props.quality,
                      props.pathTracing,
                    );
                  return (
                    <button
                      type="button"
                      class="render-preset-card"
                      classList={{
                        active: active(),
                        selected: preset.id === props.selectedPreset,
                      }}
                      aria-pressed={active()}
                      onClick={() => props.onPresetActivate(preset.id)}
                      data-preset={preset.id}
                      data-testid={`render-preset-${preset.id}`}
                    >
                      <span class="render-preset-preview" aria-hidden="true">
                        <i />
                        <i />
                      </span>
                      <strong>{preset.name}</strong>
                      <small>{preset.hint}</small>
                    </button>
                  );
                }}
              </For>
            </div>
            <button
              type="button"
              class={`render-customize ${editing() ? "active" : ""}`}
              onClick={() => setEditing(!editing())}
              aria-expanded={editing()}
              data-testid="render-customize"
            >
              <span>
                <b>{editing() ? "Editing" : "Customize"}</b>
                <small>{selectedPreset().name}</small>
              </span>
              <i>{editing() ? "−" : "+"}</i>
            </button>
          </section>

          <Show when={editing()}>
            <div class="render-preset-editor" data-testid="render-preset-editor">
              <div class="render-preset-actions">
                <span>
                  {renderPresetMatches(
                    selectedPreset(),
                    props.display,
                    props.quality,
                    props.pathTracing,
                  )
                    ? "Preset is active"
                    : "Current settings are unsaved"}
                </span>
                <button
                  type="button"
                  disabled={renderPresetMatches(
                    selectedPreset(),
                    props.display,
                    props.quality,
                    props.pathTracing,
                  )}
                  onClick={() => props.onPresetSave(selectedPreset().id)}
                  data-testid="render-preset-save"
                >
                  Save
                </button>
                <button
                  type="button"
                  onClick={() => props.onPresetReset(selectedPreset().id)}
                  data-testid="render-preset-reset"
                >
                  Reset
                </button>
              </div>

              <section>
                <h4>Shading</h4>
                <div class="segmented">
                  <button
                    type="button"
                    class={props.display.flatShading ? "" : "active"}
                    onClick={() => props.onChange({ flatShading: false })}
                    data-testid="shading-full"
                  >
                    Full
                  </button>
                  <button
                    type="button"
                    class={props.display.flatShading ? "active" : ""}
                    onClick={() => props.onChange({ flatShading: true })}
                    title="Albedo only, no specular or environment"
                    data-testid="shading-flat"
                  >
                    Flat
                  </button>
                </div>
              </section>

              <section>
                <h4>Shadows</h4>
                <div class="segmented">
                  <For each={SHADOWS}>
                    {(mode) => (
                      <button
                        type="button"
                        class={props.display.shadows === mode.value ? "active" : ""}
                        onClick={() => props.onChange({ shadows: mode.value })}
                        title={mode.hint}
                        data-testid={`shadows-${mode.value}`}
                      >
                        {mode.label}
                      </button>
                    )}
                  </For>
                </div>
              </section>

              <section>
                <h4>Quality</h4>
                <div class="segmented">
                  <For each={Object.entries(QUALITY_PRESETS)}>
                    {([key, preset]) => (
                      <button
                        type="button"
                        class={props.quality === key ? "active" : ""}
                        onClick={() => props.onQualityChange(key)}
                        data-testid={`quality-${key}`}
                      >
                        {preset.label}
                      </button>
                    )}
                  </For>
                </div>
              </section>

              <section>
                <h4>Effects</h4>
                <label class="switch">
                  <input
                    type="checkbox"
                    checked={props.pathTracing}
                    onChange={(event) =>
                      props.onPathTracingChange(event.currentTarget.checked)
                    }
                    data-testid="toggle-path-tracing"
                  />
                  <span>
                    Path tracing
                    <small>Progressive physically based rendering</small>
                  </span>
                </label>
                <label class="switch">
                  <input
                    type="checkbox"
                    checked={props.display.xray > 0}
                    onChange={(event) =>
                      props.onChange({
                        xray: event.currentTarget.checked ? 1 : 0,
                      })
                    }
                    data-testid="toggle-xray"
                  />
                  <span>
                    X-ray
                    <small>See construction through the solid</small>
                  </span>
                </label>
                <For each={SWITCHES}>
                  {(entry) => (
                    <label class="switch">
                      <input
                        type="checkbox"
                        checked={Boolean(props.display[entry.key])}
                        onChange={(event) =>
                          props.onChange({
                            [entry.key]: event.currentTarget.checked,
                          })
                        }
                        data-testid={`toggle-${entry.key}`}
                      />
                      <span>
                        {entry.label}
                        <small>{entry.hint}</small>
                      </span>
                    </label>
                  )}
                </For>
              </section>

              <section>
                <h4>Annotations</h4>
                <label class="switch">
                  <input
                    type="checkbox"
                    checked={props.display.showSketches}
                    onChange={(event) =>
                      props.onChange({
                        showSketches: event.currentTarget.checked,
                      })
                    }
                    data-testid="toggle-showSketches"
                  />
                  <span>
                    Sketch geometry
                    <small>Edges and editable point handles</small>
                  </span>
                </label>
                <label class="switch">
                  <input
                    type="checkbox"
                    checked={props.display.showConstraints}
                    onChange={(event) =>
                      props.onChange({
                        showConstraints: event.currentTarget.checked,
                      })
                    }
                    data-testid="toggle-constraints"
                  />
                  <span>
                    Constraints
                    <small>Viewport dimensions and fixed badges</small>
                  </span>
                </label>
                <div
                  class="annotation-options"
                  classList={{ disabled: !props.display.showConstraints }}
                >
                  <label class="switch compact">
                    <input
                      type="checkbox"
                      checked={props.display.showFixedConstraints}
                      disabled={!props.display.showConstraints}
                      onChange={(event) =>
                        props.onChange({
                          showFixedConstraints: event.currentTarget.checked,
                        })
                      }
                      data-testid="toggle-fixed-constraints"
                    />
                    <span>Fixed badges</span>
                  </label>
                  <label class="switch compact">
                    <input
                      type="checkbox"
                      checked={props.display.showDistanceConstraints}
                      disabled={!props.display.showConstraints}
                      onChange={(event) =>
                        props.onChange({
                          showDistanceConstraints: event.currentTarget.checked,
                        })
                      }
                      data-testid="toggle-distance-constraints"
                    />
                    <span>Distance dimensions</span>
                  </label>
                  <label class="switch compact">
                    <input
                      type="checkbox"
                      checked={props.display.showConstraintValues}
                      disabled={
                        !props.display.showConstraints ||
                        !props.display.showDistanceConstraints
                      }
                      onChange={(event) =>
                        props.onChange({
                          showConstraintValues: event.currentTarget.checked,
                        })
                      }
                      data-testid="toggle-constraint-values"
                    />
                    <span>Dimension values</span>
                  </label>
                </div>
              </section>
            </div>
          </Show>
        </div>
      </Show>
    </div>
  );
}
