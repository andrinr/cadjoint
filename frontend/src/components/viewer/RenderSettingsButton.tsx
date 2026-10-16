/**
 * The render-settings popover's trigger, in the viewport bar.
 *
 * It used to sit in the app toolbar between the status line and Run, but
 * everything it changes is the picture — presets, shading, overlays, the
 * distance-field views — so it lives with the other controls of the picture,
 * beside the projection toggle (guide N5: an action lives on what it acts on).
 * The open state is the shared `renderPopoverOpen` signal, so View › Render ›
 * Settings… opens the same popover.
 */

import { Show, onCleanup } from "solid-js";
import { DisplayIcon } from "../icons";
import { RenderPanel, type RenderPanelProps } from "../RenderPanel";
import { renderPopoverOpen, setRenderPopoverOpen } from "../Toolbar";

export function RenderSettingsButton(props: { render: RenderPanelProps }) {
  let anchor: HTMLDivElement | undefined;

  // Escape closes the popover in the capture phase, so the viewer's own
  // Escape ladder never sees the key.
  const onKeyDown = (event: KeyboardEvent) => {
    if (event.key === "Escape" && renderPopoverOpen()) {
      event.preventDefault();
      event.stopPropagation();
      setRenderPopoverOpen(false);
    }
  };
  // Clicking anywhere outside the anchor dismisses like any popover.
  const onPointerDown = (event: PointerEvent) => {
    if (renderPopoverOpen() && anchor && !anchor.contains(event.target as Node)) {
      setRenderPopoverOpen(false);
    }
  };
  window.addEventListener("keydown", onKeyDown, true);
  window.addEventListener("pointerdown", onPointerDown, true);
  onCleanup(() => {
    window.removeEventListener("keydown", onKeyDown, true);
    window.removeEventListener("pointerdown", onPointerDown, true);
  });

  return (
    <div class="render-popover-anchor" ref={anchor}>
      <button
        type="button"
        class={`viewport-bar-button ${renderPopoverOpen() ? "active" : ""}`}
        onClick={() => setRenderPopoverOpen(!renderPopoverOpen())}
        title="Render settings — presets, shading, overlays and quality"
        aria-label="Render settings"
        aria-expanded={renderPopoverOpen()}
        data-testid="display-options"
      >
        <DisplayIcon size={16} />
        <span class="rail-word">Render</span>
      </button>
      <Show when={renderPopoverOpen()}>
        <div class="render-popover" data-testid="render-popover">
          <RenderPanel {...props.render} />
        </div>
      </Show>
    </div>
  );
}
