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
import type { DisplaySettings, SdfView, ShadowMode } from "../viewer/renderer";
import {
  QUALITY_PRESETS,
  SDF_SLICE_RANGE,
  isSliceView,
  slicePosition,
} from "../viewer/renderer";
import { sdfRampCss } from "../viewer/sdfRamp";
import { MM_PER_UNIT, formatDistance } from "../viewer/graticule";
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

/**
 * The three ways of looking at the field.
 *
 * Solid is the raymarched surface; the other two cut the field open on a
 * plane. They are not a quality setting or an effect — they answer a different
 * question about the same scene, which is why they sit above the preset
 * editor rather than inside it.
 */
const SDF_VIEWS: SegmentedOption<SdfView>[] = [
  { value: "solid", label: "Solid", title: "The raymarched surface", testId: "sdf-solid" },
  {
    value: "slice",
    label: "Slice",
    title: "Signed distance on a plane through the scene",
    testId: "sdf-slice",
  },
  {
    value: "gradient",
    label: "∇f",
    title: "Gradient magnitude: where the field stops being a metric distance",
    testId: "sdf-gradient",
  },
  {
    value: "normal",
    label: "N",
    title: "World-space surface normals, n × 0.5 + 0.5",
    testId: "sdf-normal",
  },
  {
    value: "depth",
    label: "Z",
    title: "Linear camera depth across the framed volume",
    testId: "sdf-depth",
  },
];

const SLICE_AXES: SegmentedOption<0 | 1 | 2>[] = [
  { value: 0, label: "X", testId: "sdf-axis-x" },
  { value: 1, label: "Y", testId: "sdf-axis-y" },
  { value: 2, label: "Z", testId: "sdf-axis-z" },
];

/** A world distance, spoken the way the graticule's readout speaks one. */
const distanceLabel = (units: number): string => {
  const { value, unit } = formatDistance(units * MM_PER_UNIT);
  return `${value} ${unit}`;
};

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

        <section class="render-sdf" data-testid="render-sdf">
          <div class="render-preset-heading">
            <h4>Distance field</h4>
            <small>What the viewport draws</small>
          </div>
          <Segmented
            options={SDF_VIEWS}
            value={props.display.sdfView}
            onSelect={(sdfView) => props.onChange({ sdfView })}
          />

          <Show when={isSliceView(props.display.sdfView)}>
            <div class="sim-slice">
              <Segmented
                options={SLICE_AXES}
                value={props.display.sdfAxis}
                onSelect={(sdfAxis) => props.onChange({ sdfAxis })}
              />
              <input
                type="range"
                min="0"
                max="1"
                step="0.005"
                value={props.display.sdfFraction}
                onInput={(event) =>
                  props.onChange({ sdfFraction: Number(event.currentTarget.value) })
                }
                aria-label="Slice position"
                data-testid="sdf-fraction"
              />
            </div>
            {/* The plane's coordinate, not its fraction: a fraction of a slab
                the reader cannot see is not a number they can act on. */}
            <div class="sim-legend-values">
              <span>{["X", "Y", "Z"][props.display.sdfAxis]} ={" "}
                {distanceLabel(slicePosition(props.display.sdfFraction))}
              </span>
              <span>± {distanceLabel(SDF_SLICE_RANGE)}</span>
            </div>

            <div class="sim-legend" data-testid="sdf-legend">
              <small>
                {props.display.sdfView === "gradient"
                  ? "|∇f| — 1.0 is an exact distance field"
                  : "f — signed distance, inside and out"}
              </small>
              <div class="sim-ramp" style={{ background: sdfRampCss() }} />
              <div class="sim-legend-values">
                <span>{props.display.sdfView === "gradient" ? "0.5" : "inside"}</span>
                <span>{props.display.sdfView === "gradient" ? "1.0" : "0"}</span>
                <span>{props.display.sdfView === "gradient" ? "1.5" : "outside"}</span>
              </div>
              {/* The intervals themselves are a function of the camera, so
                  the numbers are printed over the viewport beside the GRID
                  readout they are taken from; what belongs here is the rule. */}
              <small>
                {props.display.sdfView === "gradient"
                  ? "Contours every 0.1, heaviest at 1.0"
                  : `Contours at the grid spacing and a fifth of it, ${
                      "densest within two intervals of the surface"
                    }`}
              </small>
            </div>
          </Show>

          <Show when={props.display.sdfView === "normal"}>
            <div class="sim-legend" data-testid="sdf-normal-legend">
              <small>n × 0.5 + 0.5, in world axes</small>
              <div class="sim-ramp sdf-axis-ramp" />
              <div class="sim-legend-values">
                <span>+X red</span>
                <span>+Y green</span>
                <span>+Z blue</span>
              </div>
            </div>
          </Show>

          <Show when={props.display.sdfView === "depth"}>
            <div class="sim-legend" data-testid="sdf-depth-legend">
              <small>Linear depth along the primary ray</small>
              <div class="sim-ramp sdf-depth-ramp" />
              <div class="sim-legend-values">
                <span>near</span>
                <span>far</span>
              </div>
              {/* The two ends follow the zoom, so their values are printed
                  over the viewport where the camera is. */}
              <small>Half a frame either side of the orbit target</small>
            </div>
          </Show>

          {/* Applies to every view, because it is not a view: it moves the
              surface the tracer resolves, so the solid, the slice's zero
              contour and the shadows all follow it together. */}
          <div class="sim-slice">
            <span class="pane-hint">Offset</span>
            <input
              type="range"
              min="-0.4"
              max="0.4"
              step="0.005"
              value={props.display.isoOffset}
              onInput={(event) =>
                props.onChange({ isoOffset: Number(event.currentTarget.value) })
              }
              aria-label="Isosurface offset"
              data-testid="sdf-offset"
            />
          </div>
          <div class="sim-legend-values">
            <span>f = {distanceLabel(props.display.isoOffset)}</span>
            <button
              type="button"
              class="render-sdf-zero"
              onClick={() => props.onChange({ isoOffset: 0 })}
              disabled={props.display.isoOffset === 0}
              data-testid="sdf-offset-zero"
            >
              f = 0
            </button>
          </div>
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
                checked={props.display.showGraticule}
                onChange={(showGraticule) => props.onChange({ showGraticule })}
                testId="toggle-graticule"
              >
                Ground grid
                <small>Floor plane, scale readout, and title block</small>
              </ToggleSwitch>
              {/* The master switch over everything drawn *about* the model.
                  First in the section, because the finer switches below are
                  all inside it. */}
              <ToggleSwitch
                checked={props.display.showOverlays}
                onChange={(showOverlays) => props.onChange({ showOverlays })}
                testId="toggle-construction-overlay"
              >
                Construction overlay
                <small>Sketches, handles, gizmo, constraints, BC preview</small>
              </ToggleSwitch>
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
