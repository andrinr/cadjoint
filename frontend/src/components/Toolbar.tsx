/** Top bar: run controls, tool modes, quality, and the generated-WGSL view. */

import { For, Show } from "solid-js";
import { busy, dirty, selection, setTool, status, tool } from "../state";
import { QUALITY_PRESETS } from "../viewer/renderer";

export interface ToolbarProps {
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

      <div class="tool-group" role="group" aria-label="Sketch tools">
        <button
          type="button"
          class={tool() === "select" ? "active" : ""}
          onClick={() => setTool("select")}
          title="Select and drag sketch vertices"
          data-testid="tool-select"
        >
          Select
        </button>
        <button
          type="button"
          class={tool() === "add" ? "active" : ""}
          onClick={() => setTool("add")}
          title="Click a sketch edge to insert a vertex"
          data-testid="tool-add"
        >
          Add vertex
        </button>
      </div>

      <div class="spacer" />

      <Show when={selection()}>
        <span class="selection-chip" data-testid="selection-chip">
          vertex {selection()!.vertexIndex}
        </span>
      </Show>

      <span class={`status ${status().kind}`} data-testid="status">
        <i class="dot" />
        {status().text}
      </span>

      <button type="button" onClick={props.onShowWgsl} disabled={!props.wgslReady}>
        WGSL
      </button>
      <select
        value={props.quality}
        onChange={(event) => props.onQualityChange(event.currentTarget.value)}
        aria-label="Render quality"
      >
        <For each={Object.entries(QUALITY_PRESETS)}>
          {([key, preset]) => <option value={key}>{preset.label}</option>}
        </For>
      </select>
      <button
        type="button"
        class={props.pathTracing ? "active" : ""}
        onClick={props.onToggleTrace}
      >
        {props.pathTracing ? "Preview" : "Path trace"}
      </button>
      <button type="button" onClick={props.onReset}>
        Reset
      </button>
      <button
        type="button"
        class="primary"
        onClick={props.onRun}
        disabled={busy()}
        data-testid="run"
      >
        {busy() ? "Compiling…" : dirty() ? "Run •" : "Run"}
      </button>
    </header>
  );
}
