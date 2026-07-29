/** Top bar: run controls, tool modes, quality, and the generated-WGSL view. */

import { Show } from "solid-js";
import { busy, dirty, nodeById, selection, status } from "../state";
import { DisplayOptions } from "./DisplayOptions";
import { CodeIcon, PlayIcon, ResetIcon, TraceIcon } from "./icons";
import type { DisplaySettings } from "../viewer/renderer";

export interface ToolbarProps {
  display: DisplaySettings;
  onDisplayChange: (patch: Partial<DisplaySettings>) => void;
  onRun: () => void;
  onReset: () => void;
  onToggleTrace: () => void;
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
        quality={props.quality}
        onChange={props.onDisplayChange}
        onQualityChange={props.onQualityChange}
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
        class={`icon ${props.pathTracing ? "active" : ""}`}
        onClick={props.onToggleTrace}
        title={props.pathTracing ? "Back to preview" : "Progressive path trace"}
        aria-label="Path trace"
      >
        <TraceIcon />
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
