/**
 * The transform gizmo's share of the pointer.
 *
 * The gizmo sits on top of everything it can move, so it gets first refusal
 * on a press, first refusal on hover, and it owns the drag until release.
 * That whole life cycle — decide whether a handle was grabbed, follow the
 * pointer along the chosen axis, then write the finished transform back to
 * the source — is one concern, and it is the only part of the pane that
 * reasons about axes, angles and dimension scaling.
 *
 * A drag never mutates the scene as it goes: the live transform lands in the
 * shared `gizmoDrag` signal for the renderer to preview, and only the release
 * emits a patch.
 */

import {
  displayProfiles,
  gizmoDrag,
  hover,
  nodeById,
  selection,
  setGizmoDrag,
} from "../../state";
import {
  AXES,
  angleAroundAxis,
  angleDelta,
  closestPointOnAxis,
  gizmoScale,
  pickGizmoAxis,
  scaleDimensions,
} from "../../viewer/gizmo";
import { add, rayFromPixel, scale, type Vec3 } from "../../viewer/math";
import type { PickView } from "../../viewer/hittest";
import type { Renderer } from "../../viewer/renderer";
import type { Gesture } from "./gestures";
import type { ViewerPaneProps } from "./props";

/** The in-flight gizmo drag: the only gesture variant this module handles. */
type GizmoGesture = Extract<Gesture, { kind: "gizmo" }>;

export interface GizmoDragContext {
  renderer: Renderer;
  pickView: () => PickView;
  props: ViewerPaneProps;
}

export function createGizmoDrag(context: GizmoDragContext) {
  /**
   * Whether the cursor is over a handle, refreshing the highlighted axis.
   *
   * Returns false when there is no gizmo or the pointer missed it, in which
   * case the caller falls through to ordinary picking.
   */
  const hoverAxis = (x: number, y: number, view: PickView): boolean => {
    const target = context.renderer.gizmoTarget();
    if (!target) return false;
    const mode = context.renderer.gizmoModeFor(target.node);
    const axis = pickGizmoAxis(
      target.origin,
      gizmoScale(view, target.origin),
      mode,
      x,
      y,
      view,
    );
    if (axis !== context.renderer.gizmoAxis) {
      context.renderer.gizmoAxis = axis;
      context.renderer.setConstruction(displayProfiles(), selection(), hover());
    }
    return axis !== null;
  };

  /**
   * Start a gizmo drag, or return null so the press falls through.
   *
   * Null covers both "no handle under the pointer" and "the handle's axis is
   * edge-on", where there is no stable drag parameter to start from.
   */
  const begin = (x: number, y: number): GizmoGesture | null => {
    const target = context.renderer.gizmoTarget();
    if (!target) return null;
    const view = context.pickView();
    const size = gizmoScale(view, target.origin);
    const mode = context.renderer.gizmoModeFor(target.node);
    const axis = pickGizmoAxis(target.origin, size, mode, x, y, view);
    if (axis === null) return null;
    const transform = target.node.transform!;
    const ray = rayFromPixel(x, y, view);
    const start =
      mode === "rotate"
        ? angleAroundAxis(ray, target.origin, AXES[axis])
        : closestPointOnAxis(ray, target.origin, AXES[axis]);
    if (start === null) return null;
    return {
      kind: "gizmo",
      nodeId: target.node.id,
      axis,
      mode,
      start,
      position: [...transform.position] as [number, number, number],
      rotation: [...transform.rotation] as [number, number, number],
      dimensions: Object.fromEntries(
        Object.entries(transform.dimensions).map(([key, value]) => [
          key,
          Array.isArray(value) ? [...value] : value,
        ]),
      ),
      scaleLength: size,
      gizmoOrigin: [...target.origin] as [number, number, number],
      moved: false,
    };
  };

  /** Follow the pointer, publishing the previewed transform as it moves. */
  const update = (gesture: GizmoGesture, x: number, y: number) => {
    const view = context.pickView();
    const ray = rayFromPixel(x, y, view);
    const origin = gesture.gizmoOrigin as Vec3;
    const axis = AXES[gesture.axis];
    if (gesture.mode === "translate") {
      const now = closestPointOnAxis(ray, origin, axis);
      const delta = now - gesture.start;
      const position = add(gesture.position, scale(axis, delta)) as [
        number,
        number,
        number,
      ];
      gesture.moved = true;
      setGizmoDrag({
        ...gesture,
        position,
        rotation: gesture.rotation,
        dimensions: gesture.dimensions,
      });
    } else if (gesture.mode === "rotate") {
      const now = angleAroundAxis(ray, origin, axis);
      if (now === null) return;
      const rotation = [...gesture.rotation] as [number, number, number];
      rotation[gesture.axis] += angleDelta(gesture.start, now);
      gesture.moved = true;
      setGizmoDrag({
        ...gesture,
        position: gesture.position,
        rotation,
        dimensions: gesture.dimensions,
      });
    } else {
      const now = closestPointOnAxis(ray, origin, axis);
      const factor = 1 + (now - gesture.start) / gesture.scaleLength;
      const node = nodeById(gesture.nodeId);
      if (!node) return;
      const dimensions = scaleDimensions(
        node.kind,
        gesture.dimensions,
        gesture.axis,
        factor,
      );
      gesture.moved = true;
      setGizmoDrag({
        ...gesture,
        position: gesture.position,
        rotation: gesture.rotation,
        dimensions,
      });
    }
  };

  /**
   * Release: write the dragged transform back to the source.
   *
   * A drag that never moved, or whose object vanished under it, commits
   * nothing — the preview is simply dropped.
   */
  const commit = async (finished: GizmoGesture) => {
    const active = gizmoDrag();
    const node = nodeById(finished.nodeId);
    setGizmoDrag(null);
    context.renderer.gizmoAxis = null;
    const placement = node?.transform;
    if (finished.moved && active && placement) {
      if (finished.mode === "translate") {
        await context.props.onSetValue(
          placement.line,
          placement.call,
          placement.positionArgument,
          active.position,
        );
      } else if (finished.mode === "rotate") {
        await context.props.onSetValue(
          placement.line,
          placement.call,
          "rotation",
          active.rotation,
        );
      } else {
        const argument =
          node?.kind === "box"
            ? "size"
            : node?.kind === "cylinder" && finished.axis === 2
              ? "height"
              : "radius";
        const value = active.dimensions[argument];
        if (value !== undefined) {
          await context.props.onSetValue(placement.line, placement.call, argument, value);
        }
      }
    }
  };

  return { hoverAxis, begin, update, commit };
}
