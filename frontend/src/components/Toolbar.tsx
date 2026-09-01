/**
 * Top bar: brand, the mode switcher, source controls, status — and the
 * render-settings popover anchored to the eye icon. Rendering (presets,
 * shading, quality, annotations) is orthogonal to what you are editing, so
 * the popover opens from any mode rather than being a mode itself.
 */

import { Show, createSignal, onCleanup } from "solid-js";
import { busy, dirty, nodeById, selection, status } from "../state";
import { ModeSwitcher } from "./ModeSwitcher";
import { RenderPanel, type RenderPanelProps } from "./RenderPanel";
import { CodeIcon, DisplayIcon, PlayIcon, ResetIcon } from "./icons";

export interface ToolbarProps {
  onRun: () => void;
  onReset: () => void;
  onShowWgsl: () => void;
  wgslReady: boolean;
  /** Everything the render-settings popover forwards to RenderPanel. */
  render: RenderPanelProps;
}

export function Toolbar(props: ToolbarProps) {
  const [renderOpen, setRenderOpen] = createSignal(false);
  let anchor: HTMLDivElement | undefined;

  // Escape closes the popover in the capture phase, so the viewer's own
  // Escape handling (clear selection, return to model) never sees the key.
  const onKeyDown = (event: KeyboardEvent) => {
    if (event.key === "Escape" && renderOpen()) {
      event.preventDefault();
      event.stopPropagation();
      setRenderOpen(false);
    }
  };
  // Clicking anywhere outside the anchor dismisses like any popover.
  const onPointerDown = (event: PointerEvent) => {
    if (renderOpen() && anchor && !anchor.contains(event.target as Node)) {
      setRenderOpen(false);
    }
  };
  window.addEventListener("keydown", onKeyDown, true);
  window.addEventListener("pointerdown", onPointerDown, true);
  onCleanup(() => {
    window.removeEventListener("keydown", onKeyDown, true);
    window.removeEventListener("pointerdown", onPointerDown, true);
  });

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

      <div class="render-popover-anchor" ref={anchor}>
        <button
          type="button"
          class={`icon ${renderOpen() ? "active" : ""}`}
          onClick={() => setRenderOpen(!renderOpen())}
          title="Render settings — presets, shading, and quality"
          aria-label="Render settings"
          aria-expanded={renderOpen()}
          data-testid="display-options"
        >
          <DisplayIcon />
        </button>
        <Show when={renderOpen()}>
          <div class="render-popover" data-testid="render-popover">
            <RenderPanel {...props.render} />
          </div>
        </Show>
      </div>
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
