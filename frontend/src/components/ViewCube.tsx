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

import { For } from "solid-js";
import type { Projection } from "../viewer/math";
import { IsoIcon, OrthographicIcon, PerspectiveIcon } from "./icons";

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

const ORBIT_SPEED = 0.011;
const PITCH_LIMIT = 1.45;
/** Pointer travel, in pixels, below which the gesture counts as a click. */
const CLICK_SLOP = 4;

export interface ViewCubeProps {
  yaw: number;
  pitch: number;
  projection: Projection;
  active: string;
  onPreset: (key: string) => void;
  onProjection: (projection: Projection) => void;
  /** Drag the cube to orbit, as in other 3D viewports. */
  onOrbit: (yaw: number, pitch: number) => void;
}

export function ViewCube(props: ViewCubeProps) {
  let dragging = false;
  let travelled = 0;
  let lastX = 0;
  let lastY = 0;

  const onPointerDown = (event: PointerEvent) => {
    dragging = true;
    travelled = 0;
    lastX = event.clientX;
    lastY = event.clientY;
  };

  const onPointerMove = (event: PointerEvent) => {
    if (!dragging) return;
    const dx = event.clientX - lastX;
    const dy = event.clientY - lastY;
    travelled += Math.abs(dx) + Math.abs(dy);
    lastX = event.clientX;
    lastY = event.clientY;
    // Capture only once this is really a drag: capturing on pointerdown would
    // retarget the click and a face would never receive it.
    if (travelled > CLICK_SLOP) {
      const element = event.currentTarget as HTMLElement;
      if (!element.hasPointerCapture(event.pointerId)) {
        element.setPointerCapture(event.pointerId);
      }
    }
    props.onOrbit(
      props.yaw - dx * ORBIT_SPEED,
      Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, props.pitch + dy * ORBIT_SPEED)),
    );
  };

  const onPointerUp = (event: PointerEvent) => {
    dragging = false;
    const element = event.currentTarget as HTMLElement;
    if (element.hasPointerCapture(event.pointerId)) {
      element.releasePointerCapture(event.pointerId);
    }
  };

  /** A face click only snaps the view when the pointer barely moved. */
  const choose = (key: string) => {
    if (travelled > CLICK_SLOP) return;
    props.onPreset(key);
  };

  return (
    <div class="view-cube" data-testid="view-cube">
      <div
        class="cube-stage"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        title="Drag to orbit, click a face to snap"
      >
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
          aria-label="Isometric view"
          data-testid="view-iso"
        >
          <IsoIcon />
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
          aria-label="Toggle projection"
          data-testid="projection-toggle"
        >
          {props.projection === "orthographic" ? <OrthographicIcon /> : <PerspectiveIcon />}
        </button>
      </div>

    </div>
  );
}
