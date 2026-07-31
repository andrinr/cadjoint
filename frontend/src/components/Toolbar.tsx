/** Top bar: source controls and the compact entry point to render presets. */

import { Show } from "solid-js";
import { busy, dirty, nodeById, selection, status } from "../state";
import { DisplayOptions } from "./DisplayOptions";
import { CodeIcon, PlayIcon, ResetIcon } from "./icons";
import type { DisplaySettings } from "../viewer/renderer";
import type { RenderPreset, RenderPresetId } from "../renderPresets";

export interface ToolbarProps {
  display: DisplaySettings;
  renderPresets: RenderPreset[];
  selectedRenderPreset: RenderPresetId;
  onDisplayChange: (patch: Partial<DisplaySettings>) => void;
  onRenderPresetActivate: (id: RenderPresetId) => void;
  onRenderPresetSave: (id: RenderPresetId) => void;
  onRenderPresetReset: (id: RenderPresetId) => void;
  onPathTracingChange: (enabled: boolean) => void;
  onRun: () => void;
  onReset: () => void;
  onQualityChange: (key: string) => void;
  onShowWgsl: () => void;
  pathTracing: boolean;
  quality: string;
  wgslReady: boolean;
}

export function Toolbar(props: ToolbarProps) {
  return (
    <header class="toolbar">
      <div class="brand">
        <span class="mark">jx</span>
        <span>JAXCAD</span>
      </div>

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

      <DisplayOptions
        display={props.display}
        presets={props.renderPresets}
        selectedPreset={props.selectedRenderPreset}
        quality={props.quality}
        onChange={props.onDisplayChange}
        onQualityChange={props.onQualityChange}
        onPresetActivate={props.onRenderPresetActivate}
        onPresetSave={props.onRenderPresetSave}
        onPresetReset={props.onRenderPresetReset}
        pathTracing={props.pathTracing}
        onPathTracingChange={props.onPathTracingChange}
      />
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
