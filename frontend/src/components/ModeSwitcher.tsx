/**
 * Prominent segmented mode switcher: Model / Sketch / Simulate / Render.
 *
 * Lives in the top toolbar next to the brand, the way Blender and Fusion put
 * their workspace picker top-left. Each segment shows icon + label and lights
 * up in its mode's accent color; the same accent is threaded through the rest
 * of the chrome via `.app[data-mode=…]` in the stylesheet.
 *
 * Keyboard equivalents stay untouched: M cycles the modes, Escape returns to
 * model (both handled by the viewer's key handler).
 */

import { Dynamic } from "solid-js/web";
import { For, type Component } from "solid-js";
import { editingMode, setEditingMode } from "../state";
import type { EditingMode } from "../editingMode";
import {
  ModelModeIcon,
  RenderModeIcon,
  SimulateModeIcon,
  SketchModeIcon,
} from "./icons";

const MODES: { key: EditingMode; label: string; hint: string; icon: Component }[] = [
  { key: "model", label: "Model", hint: "Model mode — solids and transforms  (M cycles)", icon: ModelModeIcon },
  { key: "sketch", label: "Sketch", hint: "Sketch mode — 2D profiles and constraints  (M cycles)", icon: SketchModeIcon },
  { key: "simulate", label: "Simulate", hint: "Simulate mode — FEM setup  (M cycles)", icon: SimulateModeIcon },
  { key: "render", label: "Render", hint: "Render mode — presets, shading, and quality  (M cycles)", icon: RenderModeIcon },
];

export function ModeSwitcher() {
  return (
    <div class="mode-switcher" role="group" aria-label="Editing mode">
      <For each={MODES}>
        {(mode) => (
          <button
            type="button"
            data-mode={mode.key}
            classList={{ active: editingMode() === mode.key }}
            onClick={() => setEditingMode(mode.key)}
            title={mode.hint}
            aria-label={`${mode.label} mode`}
            aria-pressed={editingMode() === mode.key}
            data-testid={`editmode-${mode.key}`}
          >
            <Dynamic component={mode.icon} />
            <span>{mode.label}</span>
          </button>
        )}
      </For>
    </div>
  );
}
