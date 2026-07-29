/** Top bar: run controls, tool modes, quality, and the generated-WGSL view. */

import { For, Show } from "solid-js";
import {
  busy,
  dirty,
  gizmoMode,
  nodeById,
  selection,
  setGizmoMode,
  setTool,
  status,
  tool,
} from "../state";
import { DisplayOptions } from "./DisplayOptions";
import { ViewControls } from "./ViewControls";
import type { Projection } from "../viewer/math";
import { QUALITY_PRESETS, type DisplaySettings } from "../viewer/renderer";

export interface ToolbarProps {
  display: DisplaySettings;
  viewPreset: string;
  onDisplayChange: (patch: Partial<DisplaySettings>) => void;
  onViewPreset: (key: string) => void;
  onProjection: (projection: Projection) => void;
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
          class={tool() === "polygon" ? "active" : ""}
          onClick={() => setTool(tool() === "polygon" ? "select" : "polygon")}
          title="Polygon: click sketch edges to add vertices (Esc to finish)"
          data-testid="tool-polygon"
        >
          Polygon
        </button>
        <For each={["box", "sphere", "cylinder"] as const}>
          {(kind) => (
            <button
              type="button"
              class={tool() === kind ? "active" : ""}
              onClick={() => setTool(tool() === kind ? "select" : kind)}
              title={`Place a ${kind} where you click`}
              data-testid={`tool-${kind}`}
            >
              {kind[0].toUpperCase() + kind.slice(1)}
            </button>
          )}
        </For>
      </div>

      <Show when={selection() && selection()!.vertexIndex === null}>
        <div class="tool-group" role="group" aria-label="Gizmo mode">
          <For each={["translate", "rotate"] as const}>
            {(mode) => (
              <button
                type="button"
                class={gizmoMode() === mode ? "active" : ""}
                onClick={() => setGizmoMode(mode)}
                data-testid={`gizmo-${mode}`}
              >
                {mode === "translate" ? "Move" : "Rotate"}
              </button>
            )}
          </For>
        </div>
      </Show>

      <ViewControls
        active={props.viewPreset}
        projection={props.display.projection}
        onPreset={props.onViewPreset}
        onProjection={props.onProjection}
      />
      <DisplayOptions display={props.display} onChange={props.onDisplayChange} />

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
