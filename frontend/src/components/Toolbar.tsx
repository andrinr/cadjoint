/** Top bar: brand, the mode switcher, source controls, and status. */

import { Show } from "solid-js";
import {
  busy,
  dirty,
  editingMode,
  nodeById,
  selection,
  setEditingMode,
  status,
} from "../state";
import { ModeSwitcher } from "./ModeSwitcher";
import { CodeIcon, DisplayIcon, PlayIcon, ResetIcon } from "./icons";

export interface ToolbarProps {
  onRun: () => void;
  onReset: () => void;
  onShowWgsl: () => void;
  wgslReady: boolean;
}

export function Toolbar(props: ToolbarProps) {
  return (
    <header class="toolbar">
      <div class="brand">
        <span class="mark">cj</span>
        <span>CADJOINT</span>
      </div>

      <ModeSwitcher />

      <div class="spacer" />

      <Show when={selection()}>
        <span class="selection-chip" data-testid="selection-chip">
          {selection()!.vertexIndex === null
            ? (nodeById(selection()!.nodeId)?.name ?? "solid")
            : `vertex ${selection()!.vertexIndex}`}
        </span>
      </Show>

      <span class={`status ${status().kind}`} data-testid="status">
        <i class="dot" />
        {status().text}
      </span>

      {/* The old render-settings popover became Render mode; the eye is now a
          shortcut into it (and back out), so muscle memory keeps working. */}
      <button
        type="button"
        class={`icon ${editingMode() === "render" ? "active" : ""}`}
        onClick={() =>
          setEditingMode(editingMode() === "render" ? "model" : "render")
        }
        title="Render settings — opens Render mode"
        aria-label="Render settings"
        aria-pressed={editingMode() === "render"}
        data-testid="display-options"
      >
        <DisplayIcon />
      </button>
      <button
        type="button"
        class="icon"
        onClick={props.onShowWgsl}
        disabled={!props.wgslReady}
        title="Show the generated WGSL"
        aria-label="Generated WGSL"
      >
        <CodeIcon />
      </button>
      <button
        type="button"
        class="icon"
        onClick={props.onReset}
        title="Reset to the starter program"
        aria-label="Reset"
      >
        <ResetIcon />
      </button>
      <button
        type="button"
        class="primary"
        onClick={props.onRun}
        disabled={busy()}
        data-testid="run"
      >
        <PlayIcon />
        {busy() ? "Compiling…" : dirty() ? "Run •" : "Run"}
      </button>
    </header>
  );
}
