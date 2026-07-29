/** Popover of viewport display toggles: shading, shadows, x-ray, overlays. */

import { For, Show, createSignal, onCleanup } from "solid-js";
import type { DisplaySettings } from "../viewer/renderer";

interface Toggle {
  key: keyof DisplaySettings;
  label: string;
  hint: string;
}

const TOGGLES: Toggle[] = [
  { key: "shadows", label: "Shadows", hint: "Soft shadow rays" },
  { key: "reflections", label: "Reflections", hint: "Environment reflections" },
  { key: "flatShading", label: "Flat shading", hint: "Albedo only, no specular" },
  { key: "hideSolid", label: "Hide solid", hint: "Show construction geometry only" },
  { key: "showSketches", label: "Show sketches", hint: "Draw construction overlays" },
];

export interface DisplayOptionsProps {
  display: DisplaySettings;
  onChange: (patch: Partial<DisplaySettings>) => void;
}

export function DisplayOptions(props: DisplayOptionsProps) {
  const [open, setOpen] = createSignal(false);

  const closeOnOutside = (event: MouseEvent) => {
    const target = event.target as HTMLElement;
    if (!target.closest(".display-options")) setOpen(false);
  };
  document.addEventListener("click", closeOnOutside);
  onCleanup(() => document.removeEventListener("click", closeOnOutside));

  const xrayOn = () => props.display.xray > 0;

  return (
    <div class="display-options">
      <button
        type="button"
        class={open() ? "active" : ""}
        onClick={() => setOpen(!open())}
        title="Display options"
        data-testid="display-options"
      >
        Display
      </button>
      <Show when={open()}>
        <div class="popover" role="menu">
          <label class="option">
            <input
              type="checkbox"
              checked={xrayOn()}
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
          <For each={TOGGLES}>
            {(toggle) => (
              <label class="option">
                <input
                  type="checkbox"
                  checked={Boolean(props.display[toggle.key])}
                  onChange={(event) =>
                    props.onChange({ [toggle.key]: event.currentTarget.checked })
                  }
                  data-testid={`toggle-${toggle.key}`}
                />
                <span>
                  {toggle.label}
                  <small>{toggle.hint}</small>
                </span>
              </label>
            )}
          </For>
        </div>
      </Show>
    </div>
  );
}
