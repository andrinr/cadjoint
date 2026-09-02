/**
 * Orbit-camera arithmetic for the viewport's navigation gestures.
 *
 * Drag-to-orbit, pan and wheel-zoom are the three ways a pointer moves the
 * camera, and each is a pure function from the current camera plus a pixel
 * delta to the next camera. Keeping them here — free of DOM events and of the
 * renderer — means the feel of navigation (speeds, the pitch clamp, the zoom
 * range) has one home, and the pane is left holding only the event plumbing.
 */

import { orthoHeightFor, type CameraState, type Vec3 } from "../../viewer/math";
import { distanceForGain, gainFor, stepDetent } from "../../viewer/graticule";

/** Pitch clamp, radians: stops the orbit tumbling over the poles. */
export const PITCH_LIMIT = 1.45;
/** Radians of orbit per pixel dragged. */
export const ORBIT_SPEED = 0.008;
/** Fraction of the orbit distance panned per pixel dragged. */
export const PAN_SPEED = 0.0022;

/** Yaw/pitch from a drag, with the pitch clamped away from the poles. */
export function orbitCamera(
  camera: CameraState,
  deltaX: number,
  deltaY: number,
): CameraState {
  return {
    ...camera,
    yaw: camera.yaw - deltaX * ORBIT_SPEED,
    pitch: Math.max(
      -PITCH_LIMIT,
      Math.min(PITCH_LIMIT, camera.pitch + deltaY * ORBIT_SPEED),
    ),
  };
}

/** Shift the orbit target within the camera's screen plane. */
export function panCamera(
  camera: CameraState,
  deltaX: number,
  deltaY: number,
): CameraState {
  const dx = deltaX * PAN_SPEED * camera.distance;
  const dy = deltaY * PAN_SPEED * camera.distance;
  const { yaw, pitch, target } = camera;
  const right: [number, number, number] = [Math.cos(yaw), 0, -Math.sin(yaw)];
  const up: [number, number, number] = [
    -Math.sin(yaw) * Math.sin(pitch),
    Math.cos(pitch),
    -Math.cos(yaw) * Math.sin(pitch),
  ];
  return {
    ...camera,
    target: [
      target[0] - right[0] * dx + up[0] * dy,
      target[1] - right[1] * dx + up[1] * dy,
      target[2] - right[2] * dx + up[2] * dy,
    ] as Vec3,
  };
}

/** Closest and furthest the orbit camera may sit from its target. */
export const MIN_DISTANCE = 0.4;
export const MAX_DISTANCE = 60;

const clampDistance = (distance: number): number =>
  Math.max(MIN_DISTANCE, Math.min(MAX_DISTANCE, distance));

/** Wheel zoom: exponential in the wheel delta, clamped to a usable range. */
export function zoomCamera(camera: CameraState, wheelDelta: number): CameraState {
  return {
    ...camera,
    distance: clampDistance(camera.distance * Math.exp(wheelDelta * 0.001)),
  };
}

/**
 * Wheel zoom that lands on the ground grid's 1-2-5 spacing ladder.
 *
 * One notch is one rung, so a cell is always an exact, stateable number of
 * millimetres across and the framing lines up with it. Held behind a modifier
 * because quantized zoom is unusual in CAD (§10.1): free zoom stays the
 * default, and the grid is ruled on a rung either way — what the modifier buys
 * is a cell that fills exactly an eighth of the frame.
 *
 * The snap is instantaneous rather than a `dur-base` glide (§8). A glide would
 * pass the spacing through every value between two rungs, and the readout is
 * supposed to change *at* the detent crossing, never between — the only ways
 * to honour both are to hold a second, pinned copy of the spacing for the
 * duration, or to not move through the intermediate values at all. The second
 * is cheaper, is what `prefers-reduced-motion` would force anyway, and
 * preserves the grid's zero-per-frame cost.
 */
export function detentZoomCamera(camera: CameraState, wheelDelta: number): CameraState {
  if (wheelDelta === 0) return camera;
  const gain = gainFor(orthoHeightFor(camera.distance));
  const stepped = stepDetent(gain, wheelDelta > 0 ? 1 : -1);
  const distance = distanceForGain(stepped);
  // Refuse a half-step at the ends of the range: a detent that lands off the
  // ladder because it was clamped would show the `>` it exists to remove.
  if (distance < MIN_DISTANCE || distance > MAX_DISTANCE) return camera;
  return { ...camera, distance };
}
