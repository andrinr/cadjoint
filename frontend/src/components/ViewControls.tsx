/**
 * Standard view presets and the projection toggle.
 *
 * Picking a side switches to orthographic, which is what makes it a flat CAD
 * elevation rather than a perspective shot from that angle. Iso goes back to
 * perspective.
 */

import { For } from "solid-js";
import type { Projection } from "../viewer/math";

const PRESETS: { key: string; label: string; title: string }[] = [
  { key: "iso", label: "Iso", title: "Isometric (perspective)" },
  { key: "front", label: "Front", title: "Look along +Z" },
  { key: "back", label: "Back", title: "Look along -Z" },
  { key: "left", label: "Left", title: "Look along -X" },
  { key: "right", label: "Right", title: "Look along +X" },
  { key: "top", label: "Top", title: "Look straight down" },
  { key: "bottom", label: "Bottom", title: "Look straight up" },
];

export interface ViewControlsProps {
  active: string;
  projection: Projection;
  onPreset: (key: string) => void;
  onProjection: (projection: Projection) => void;
}

export function ViewControls(props: ViewControlsProps) {
  return (
    <div class="view-controls" role="group" aria-label="Standard views">
      <select
        value={props.active}
        onChange={(event) => props.onPreset(event.currentTarget.value)}
        aria-label="Standard view"
        data-testid="view-preset"
      >
        <For each={PRESETS}>
          {(preset) => (
            <option value={preset.key} title={preset.title}>
              {preset.label}
            </option>
          )}
        </For>
      </select>
      <button
        type="button"
        class={props.projection === "orthographic" ? "active" : ""}
        onClick={() =>
          props.onProjection(
            props.projection === "orthographic" ? "perspective" : "orthographic",
          )
        }
        title="Toggle perspective / orthographic projection"
        data-testid="projection-toggle"
      >
        {props.projection === "orthographic" ? "Ortho" : "Persp"}
      </button>
    </div>
  );
}
