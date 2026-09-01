/**
 * DOM chrome floating over the canvas.
 *
 * The renderer draws the scene; these four things are drawn by the browser on
 * top of it because they are text, or because they are transient feedback
 * that should never cost a frame: the BC rubber band, the probe chip anchored
 * to a picked vertex, the compile indicator, and the dismissible viewer error.
 *
 * They are grouped only by where they sit. Each reads the shared state it
 * reports on; the rubber band is the exception, since it belongs to the
 * gesture in flight and is handed down from the pane.
 */

import { Show } from "solid-js";
import { formatScalar } from "../../simulation";
import {
  busy,
  dismissViewerError,
  editingMode,
  simProbe,
  viewerError,
} from "../../state";

export interface PickRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface ViewerOverlaysProps {
  /** BC box-pick rubber band, in CSS pixels over the canvas. */
  pickRect: PickRect | null;
}

export function ViewerOverlays(props: ViewerOverlaysProps) {
  return (
    <>
      <Show when={props.pickRect}>
        {(rect) => (
          <div
            class="bc-pick-rect"
            style={{
              left: `${rect().left}px`,
              top: `${rect().top}px`,
              width: `${rect().width}px`,
              height: `${rect().height}px`,
            }}
            data-testid="bc-pick-rect"
          />
        )}
      </Show>
      <Show when={editingMode() === "simulate" && simProbe()}>
        {(probe) => (
          <div
            class="sim-probe"
            style={{ left: `${probe().x}px`, top: `${probe().y}px` }}
            data-testid="sim-probe"
          >
            <b>{formatScalar(probe().value)}</b>
            <span>{probe().label}</span>
            <small>
              [{probe().world.map((component) => component.toFixed(3)).join(", ")}]
            </small>
          </div>
        )}
      </Show>
      <Show when={busy()}>
        <span
          class="viewer-compile-indicator"
          role="status"
          aria-label="Compiling scene"
          data-testid="viewer-compiling"
        >
          <i />
          Compiling
        </span>
      </Show>
      {viewerError() && (
        <div class="viewer-error">
          <p>{viewerError()}</p>
          <button type="button" onClick={dismissViewerError}>
            Dismiss
          </button>
        </div>
      )}
    </>
  );
}
