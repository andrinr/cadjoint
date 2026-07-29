/**
 * ViewCube: the orientation widget CAD viewports put in a corner.
 *
 * A real CSS 3D cube that tracks the camera, so it doubles as an orientation
 * readout and a set of shortcuts — click a face to look down that axis.
 *
 * The container rotation is the inverse of the camera's, which is what keeps
 * the face pointing at you the same one you are looking at:
 *   yaw +90° puts the camera on +X, so the cube turns -90° about Y to bring
 *   its +X ("Right") face to the front.
 */

import { For, Show, createSignal, onCleanup } from "solid-js";
import type { Projection } from "../viewer/math";

interface Face {
  key: string;
  label: string;
  /** CSS transform placing this face on the cube. */
  transform: string;
}

const HALF = 30;

const FACES: Face[] = [
  { key: "front", label: "FRONT", transform: `translateZ(${HALF}px)` },
  { key: "back", label: "BACK", transform: `rotateY(180deg) translateZ(${HALF}px)` },
  { key: "right", label: "RIGHT", transform: `rotateY(90deg) translateZ(${HALF}px)` },
  { key: "left", label: "LEFT", transform: `rotateY(-90deg) translateZ(${HALF}px)` },
  { key: "top", label: "TOP", transform: `rotateX(90deg) translateZ(${HALF}px)` },
  { key: "bottom", label: "BOTTOM", transform: `rotateX(-90deg) translateZ(${HALF}px)` },
];

const degrees = (radians: number) => (radians * 180) / Math.PI;

export interface ViewCubeProps {
  yaw: number;
  pitch: number;
  projection: Projection;
  active: string;
  onPreset: (key: string) => void;
  onProjection: (projection: Projection) => void;
}

export function ViewCube(props: ViewCubeProps) {
  const [menuOpen, setMenuOpen] = createSignal(false);
  const closeOnOutside = (event: MouseEvent) => {
    if (!(event.target as HTMLElement).closest(".view-cube")) setMenuOpen(false);
  };
  document.addEventListener("click", closeOnOutside);
  onCleanup(() => document.removeEventListener("click", closeOnOutside));

  const choose = (key: string) => {
    props.onPreset(key);
    setMenuOpen(false);
  };

  return (
    <div class="view-cube" data-testid="view-cube">
      <div class="cube-stage">
        <div
          class="cube"
          style={{
            transform: `rotateX(${degrees(props.pitch)}deg) rotateY(${-degrees(props.yaw)}deg)`,
          }}
        >
          <For each={FACES}>
            {(face) => (
              <button
                type="button"
                class={`cube-face ${props.active === face.key ? "active" : ""}`}
                style={{ transform: face.transform }}
                onClick={() => choose(face.key)}
                title={`${face.label[0]}${face.label.slice(1).toLowerCase()} view`}
                data-testid={`view-${face.key}`}
              >
                {face.label}
              </button>
            )}
          </For>
        </div>
      </div>

      <div class="cube-actions">
        <button
          type="button"
          class={props.active === "iso" ? "active" : ""}
          onClick={() => props.onPreset("iso")}
          title="Isometric view"
          data-testid="view-iso"
        >
          ISO
        </button>
        <button
          type="button"
          class={menuOpen() ? "active" : ""}
          onClick={() => setMenuOpen(!menuOpen())}
          title="Standard views"
          aria-label="Standard views"
          data-testid="view-menu"
        >
          VIEWS
        </button>
        <button
          type="button"
          class={props.projection === "orthographic" ? "active" : ""}
          onClick={() =>
            props.onProjection(
              props.projection === "orthographic" ? "perspective" : "orthographic",
            )
          }
          title={
            props.projection === "orthographic"
              ? "Orthographic — click for perspective"
              : "Perspective — click for orthographic"
          }
          data-testid="projection-toggle"
        >
          {props.projection === "orthographic" ? "ORTHO" : "PERSP"}
        </button>
      </div>

      <Show when={menuOpen()}>
        <ul class="view-menu" role="menu">
          <For each={FACES}>
            {(face) => (
              <li>
                <button
                  type="button"
                  class={props.active === face.key ? "active" : ""}
                  onClick={() => choose(face.key)}
                  data-testid={`view-menu-${face.key}`}
                >
                  {face.label[0] + face.label.slice(1).toLowerCase()}
                </button>
              </li>
            )}
          </For>
        </ul>
      </Show>
    </div>
  );
}
