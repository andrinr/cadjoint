/**
 * The name tags on the sketch planes.
 *
 * A renderer for the marks `planeMarks.ts` projects: one small label at the
 * +u,+v corner of each plane's frame, and a ghost label on the placement
 * preview. Clicking a label selects the plane, which is the one thing a tag
 * can offer that the frame under it cannot: a target that stays the same
 * size however far out the camera is.
 */

import { For, Show } from "solid-js";
import { setSelection } from "../../state";
import type { PlaneMarkGeometry } from "./planeMarks";

export interface PlaneLabelsProps {
  /** Master switch: the whole layer is absent when the overlay is off. */
  show: boolean;
  geometry: PlaneMarkGeometry;
}

export function PlaneLabels(props: PlaneLabelsProps) {
  return (
    <Show when={props.show}>
      <div class="plane-labels" data-testid="plane-labels" aria-label="Sketch planes">
        <For each={props.geometry.marks}>
          {(mark) => (
            <button
              type="button"
              class="plane-label"
              classList={{ derived: mark.derived, preview: mark.preview }}
              style={{ left: `${mark.x}px`, top: `${mark.y}px` }}
              data-testid={mark.preview ? "plane-label-preview" : "plane-label"}
              data-node={mark.nodeId ?? undefined}
              title={
                mark.preview
                  ? "Where the new sketch will land"
                  : mark.derived
                    ? `${mark.label} — plane derived from a face; edit the reference in the code`
                    : `${mark.label} — sketch plane; click to select, drag the gizmo to move`
              }
              disabled={mark.preview}
              onClick={() => {
                if (mark.nodeId) {
                  setSelection({ nodeId: mark.nodeId, vertexIndex: null, part: "plane" });
                }
              }}
            >
              {mark.label}
            </button>
          )}
        </For>
      </div>
    </Show>
  );
}
