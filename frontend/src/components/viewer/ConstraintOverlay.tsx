/**
 * The SVG layer that draws constraint annotations over the canvas.
 *
 * Purely a renderer for the marks `constraintMarks.ts` projects: the three
 * display switches decide what is shown, and each mark becomes the same glyph
 * every time. Anything that has to decide *where* a mark goes belongs in the
 * projection module, not here.
 */

import { For, Show } from "solid-js";
import type { ConstraintOverlayGeometry } from "./constraintMarks";

export interface ConstraintOverlayProps {
  /** Master switch: the whole layer is absent when off. */
  show: boolean;
  showDistance: boolean;
  showFixed: boolean;
  showValues: boolean;
  geometry: ConstraintOverlayGeometry;
}

export function ConstraintOverlay(props: ConstraintOverlayProps) {
  return (
    <Show when={props.show}>
      <svg
        class="constraint-overlay"
        viewBox={`0 0 ${props.geometry.width} ${props.geometry.height}`}
        aria-label="Construction constraints"
        data-testid="constraint-overlay"
      >
        <For each={props.showDistance ? props.geometry.distance : []}>
          {(mark) => (
            <g
              class="constraint-distance"
              data-scope={mark.scope}
              data-testid="constraint-distance-overlay"
            >
              <line x1={mark.ax} y1={mark.ay} x2={mark.dax} y2={mark.day} />
              <line x1={mark.bx} y1={mark.by} x2={mark.dbx} y2={mark.dby} />
              <line
                class="dimension"
                x1={mark.dax}
                y1={mark.day}
                x2={mark.dbx}
                y2={mark.dby}
              />
              <circle cx={mark.dax} cy={mark.day} r="2.2" />
              <circle cx={mark.dbx} cy={mark.dby} r="2.2" />
              <Show when={props.showValues}>
                <text x={mark.labelX} y={mark.labelY}>
                  {mark.label}
                </text>
              </Show>
            </g>
          )}
        </For>
        <For each={props.showFixed ? props.geometry.fixed : []}>
          {(mark) => (
            <g
              class="constraint-fixed"
              transform={`translate(${mark.x} ${mark.y})`}
              data-scope={mark.scope}
              data-testid="constraint-fixed-overlay"
            >
              <circle r="8" />
              <rect x="-4" y="-1" width="8" height="6" rx="1.5" />
              <path d="M-2.7-1v-2a2.7 2.7 0 0 1 5.4 0v2" />
            </g>
          )}
        </For>
      </svg>
    </Show>
  );
}
