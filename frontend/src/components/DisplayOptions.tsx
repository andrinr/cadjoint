/**
 * Render settings, grouped the way a viewport panel usually is.
 *
 * Mutually exclusive choices (shading, shadows, quality) are segmented
 * controls rather than checkboxes, so the current state reads at a glance and
 * costs one click to change; independent effects stay as switches.
 */

import { For, Show, createSignal, onCleanup } from "solid-js";
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
  { key: "showSketches", label: "Show sketches", hint: "Draw construction overlays" },
];

export interface DisplayOptionsProps {
  display: DisplaySettings;
  quality: string;
  onChange: (patch: Partial<DisplaySettings>) => void;
  onQualityChange: (key: string) => void;
}

export function DisplayOptions(props: DisplayOptionsProps) {
  const [open, setOpen] = createSignal(false);

  const closeOnOutside = (event: MouseEvent) => {
    if (!(event.target as HTMLElement).closest(".display-options")) setOpen(false);
  };
  document.addEventListener("click", closeOnOutside);
  onCleanup(() => document.removeEventListener("click", closeOnOutside));

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
                checked={props.display.xray > 0}
                onChange={(event) =>
                  props.onChange({ xray: event.currentTarget.checked ? 1 : 0 })
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
                      props.onChange({ [entry.key]: event.currentTarget.checked })
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
        </div>
      </Show>
    </div>
  );
}
