/**
 * The render-settings popover's trigger, in the viewport bar beside the
 * projection toggle: everything it changes is the picture (guide N5).
 */

import { Show, createSignal } from "solid-js";
import { DisplayIcon } from "../icons";
import { RenderPanel, type RenderPanelProps } from "../RenderPanel";
import { createDismiss } from "../ui/dismiss";

/** Module state so View › Render › Settings… opens the same popover. */
export const [renderPopoverOpen, setRenderPopoverOpen] = createSignal(false);

export function RenderSettingsButton(props: { render: RenderPanelProps }) {
  let anchor: HTMLDivElement | undefined;
  createDismiss({
    open: renderPopoverOpen,
    close: () => setRenderPopoverOpen(false),
    within: () => anchor,
    escape: true,
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
