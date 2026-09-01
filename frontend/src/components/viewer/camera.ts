/**
 * Orbit-camera arithmetic for the viewport's navigation gestures.
 *
 * Drag-to-orbit, pan and wheel-zoom are the three ways a pointer moves the
 * camera, and each is a pure function from the current camera plus a pixel
 * delta to the next camera. Keeping them here — free of DOM events and of the
 * renderer — means the feel of navigation (speeds, the pitch clamp, the zoom
 * range) has one home, and the pane is left holding only the event plumbing.
 */

import type { CameraState, Vec3 } from "../../viewer/math";

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

/** Wheel zoom: exponential in the wheel delta, clamped to a usable range. */
export function zoomCamera(camera: CameraState, wheelDelta: number): CameraState {
  return {
    ...camera,
    distance: Math.max(0.4, Math.min(60, camera.distance * Math.exp(wheelDelta * 0.001))),
  };
}
