/**
 * The full render settings, hosted in the eye-icon popover in the top bar
 * (usable from any editing mode; formerly a dock panel of the retired
 * Render mode).
 *
 * Mutually exclusive choices (shading, shadows, quality) are segmented
 * controls rather than checkboxes, so the current state reads at a glance and
 * costs one click to change; independent effects stay as switches. The preset
 * library sits on top; Customize expands the per-setting editor.
 */

import { For, Show, createMemo, createSignal } from "solid-js";
import {
  renderPresetMatches,
  type RenderPreset,
  type RenderPresetId,
} from "../renderPresets";
import type { DisplaySettings, ShadowMode } from "../viewer/renderer";
import { QUALITY_PRESETS } from "../viewer/renderer";
import { Segmented, ToggleSwitch, type SegmentedOption } from "./ui";

const SHADOWS: SegmentedOption<ShadowMode>[] = [
  { value: "off", label: "Off", title: "No shadow rays", testId: "shadows-off" },
  {
    value: "hard",
    label: "Hard",
    title: "One crisp, lifted shadow ray",
    testId: "shadows-hard",
  },
  {
    value: "soft",
    label: "Soft",
    title: "Penumbra from multiple samples",
    testId: "shadows-soft",
  },
];

/** Shading is a two-way choice; `flatShading` is the underlying flag. */
const SHADING: SegmentedOption<boolean>[] = [
  { value: false, label: "Full", testId: "shading-full" },
  {
    value: true,
    label: "Flat",
    title: "Albedo only, no specular or environment",
    testId: "shading-flat",
  },
];

const QUALITIES: SegmentedOption<string>[] = Object.entries(QUALITY_PRESETS).map(
  ([key, preset]) => ({ value: key, label: preset.label, testId: `quality-${key}` }),
);

const SWITCHES: { key: keyof DisplaySettings; label: string; hint: string }[] = [
  { key: "reflections", label: "Reflections", hint: "Environment reflections" },
  { key: "hideSolid", label: "Hide solid", hint: "Construction geometry only" },
  { key: "showMeshEdges", label: "Feature edges", hint: "Sharp creases, corners, and CSG seams" },
  { key: "showMeshWireframe", label: "Mesh wireframe", hint: "Full dual-contour quad wireframe" },
];

export interface RenderPanelProps {
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

export function RenderPanel(props: RenderPanelProps) {
  const [editing, setEditing] = createSignal(false);
  const selectedPreset = createMemo(
    () =>
      props.presets.find((preset) => preset.id === props.selectedPreset) ??
      props.presets[0],
  );
  const selectedMatches = () =>
    renderPresetMatches(
      selectedPreset(),
      props.display,
      props.quality,
      props.pathTracing,
    );

  return (
    <aside class="render-panel" data-testid="render-panel">
      <header>
        {/* Kicker first, title second — see ObjectTree. */}
        <span>
          <small>presets &amp; display</small>
          Render
        </span>
      </header>
      <div class="render-settings">
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
                {selectedMatches()
                  ? "Preset is active"
                  : "Current settings are unsaved"}
              </span>
              <button
                type="button"
                disabled={selectedMatches()}
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
              <Segmented
                options={SHADING}
                value={props.display.flatShading}
                onSelect={(flatShading) => props.onChange({ flatShading })}
              />
            </section>

            <section>
              <h4>Shadows</h4>
              <Segmented
                options={SHADOWS}
                value={props.display.shadows}
                onSelect={(shadows) => props.onChange({ shadows })}
              />
            </section>

            <section>
              <h4>Quality</h4>
              <Segmented
                options={QUALITIES}
                value={props.quality}
                onSelect={props.onQualityChange}
              />
            </section>

            <section>
              <h4>Effects</h4>
              <ToggleSwitch
                checked={props.pathTracing}
                onChange={props.onPathTracingChange}
                testId="toggle-path-tracing"
              >
                Path tracing
                <small>Progressive physically based rendering</small>
              </ToggleSwitch>
              <ToggleSwitch
                checked={props.display.xray > 0}
                onChange={(checked) => props.onChange({ xray: checked ? 1 : 0 })}
                testId="toggle-xray"
              >
                X-ray
                <small>See construction through the solid</small>
              </ToggleSwitch>
              <For each={SWITCHES}>
                {(entry) => (
                  <ToggleSwitch
                    checked={Boolean(props.display[entry.key])}
                    onChange={(checked) => props.onChange({ [entry.key]: checked })}
                    testId={`toggle-${entry.key}`}
                  >
                    {entry.label}
                    <small>{entry.hint}</small>
                  </ToggleSwitch>
                )}
              </For>
            </section>

            <section>
              <h4>Annotations</h4>
              <ToggleSwitch
                checked={props.display.showSketches}
                onChange={(showSketches) => props.onChange({ showSketches })}
                testId="toggle-showSketches"
              >
                Sketch geometry
                <small>Edges and editable point handles</small>
              </ToggleSwitch>
              <ToggleSwitch
                checked={props.display.showConstraints}
                onChange={(showConstraints) => props.onChange({ showConstraints })}
                testId="toggle-constraints"
              >
                Constraints
                <small>Viewport dimensions and fixed badges</small>
              </ToggleSwitch>
              <div
                class="annotation-options"
                classList={{ disabled: !props.display.showConstraints }}
              >
                <ToggleSwitch
                  compact
                  checked={props.display.showFixedConstraints}
                  disabled={!props.display.showConstraints}
                  onChange={(showFixedConstraints) =>
                    props.onChange({ showFixedConstraints })
                  }
                  testId="toggle-fixed-constraints"
                >
                  Fixed badges
                </ToggleSwitch>
                <ToggleSwitch
                  compact
                  checked={props.display.showDistanceConstraints}
                  disabled={!props.display.showConstraints}
                  onChange={(showDistanceConstraints) =>
                    props.onChange({ showDistanceConstraints })
                  }
                  testId="toggle-distance-constraints"
                >
                  Distance dimensions
                </ToggleSwitch>
                <ToggleSwitch
                  compact
                  checked={props.display.showConstraintValues}
                  disabled={
                    !props.display.showConstraints ||
                    !props.display.showDistanceConstraints
                  }
                  onChange={(showConstraintValues) =>
                    props.onChange({ showConstraintValues })
                  }
                  testId="toggle-constraint-values"
                >
                  Dimension values
                </ToggleSwitch>
              </div>
            </section>
          </div>
        </Show>
      </div>
    </aside>
  );
}
