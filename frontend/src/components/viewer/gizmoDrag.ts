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
 *
 * The *solid* follows too, when it can. The value a gizmo moves is often a
 * free design parameter — `bushing_a`, `board_size` — and the scene's shaders
 * read those out of a uniform buffer, so a pointer move can be answered with
 * a few hundred bytes and a redraw instead of a source rewrite and a
 * recompile. `dragBinding` decides: if every parameter behind the argument
 * being dragged has a slot in the installed program, the drag is live and
 * emits nothing until release; if any of them does not — a pinned catalog
 * radius, a rotation the SDF folded away — the whole drag falls back to the
 * old behaviour rather than half-applying, and only the wireframe moves.
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
import { argumentValue, overridesFor } from "../../viewer/dragBinding";
import { add, rayFromPixel, scale, type Vec3 } from "../../viewer/math";
import type { PickView } from "../../viewer/hittest";
import type { Renderer } from "../../viewer/renderer";
import type { ConstructionNode } from "../../types";
import type { Gesture } from "./gestures";
import type { ViewerPaneProps } from "./props";

/** The in-flight gizmo drag: the only gesture variant this module handles. */
type GizmoGesture = Extract<Gesture, { kind: "gizmo" }>;

export interface GizmoDragContext {
  renderer: Renderer;
  pickView: () => PickView;
  props: ViewerPaneProps;
}

/** The transform a drag is currently showing. */
type Placement = Pick<GizmoGesture, "position" | "rotation" | "dimensions">;

/**
 * Which source argument this drag writes back, and its value right now.
 *
 * One function, used twice: the pointer move asks it what to put in the
 * uniform buffer, and the release asks it what to patch. They were two
 * copies of the same three-way branch, and a drag whose preview and whose
 * commit disagreed about *which* argument it was moving would show one thing
 * and write another.
 */
export function draggedArgument(
  gesture: Pick<GizmoGesture, "mode" | "axis">,
  node: ConstructionNode | undefined,
  placement: Placement,
): { argument: string; value: number[] } | null {
  const transform = node?.transform;
  if (!transform) return null;
  if (gesture.mode === "translate") {
    return { argument: transform.positionArgument, value: [...placement.position] };
  }
  if (gesture.mode === "rotate") {
    return { argument: "rotation", value: [...placement.rotation] };
  }
  const argument =
    node.kind === "box"
      ? "size"
      : node.kind === "cylinder" && gesture.axis === 2
        ? "height"
        : "radius";
  const value = argumentValue(placement.dimensions[argument]);
  return value === null ? null : { argument, value };
}

export function createGizmoDrag(context: GizmoDragContext) {
  /**
   * Show the dragged value on the solid, if the shader can be told about it.
   *
   * Returns whether it took the fast path — which is the same answer as "will
   * the release be a buffer write or a fresh module" — for a caller that wants
   * to say so. Nothing in the drag itself branches on it: the wireframe
   * preview and the patch on release are identical either way.
   */
  const preview = (gesture: GizmoGesture, placement: Placement): boolean => {
    const node = nodeById(gesture.nodeId);
    const dragged = draggedArgument(gesture, node, placement);
    const overrides = dragged
      ? overridesFor(
          node?.transform?.bindings?.[dragged.argument],
          dragged.value,
          context.renderer.parameterProgram,
        )
      : null;
    // No binding, or a parameter the shader folded away: leave the buffer
    // alone entirely. Half a transform written live would draw a solid that
    // is not the one the release is about to produce.
    if (!overrides) return false;
    return context.renderer.setParameterOverrides(overrides);
  };

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
      const placement = {
        position,
        rotation: gesture.rotation,
        dimensions: gesture.dimensions,
      };
      setGizmoDrag({ ...gesture, ...placement });
      preview(gesture, placement);
    } else if (gesture.mode === "rotate") {
      const now = angleAroundAxis(ray, origin, axis);
      if (now === null) return;
      const rotation = [...gesture.rotation] as [number, number, number];
      rotation[gesture.axis] += angleDelta(gesture.start, now);
      gesture.moved = true;
      const placement = {
        position: gesture.position,
        rotation,
        dimensions: gesture.dimensions,
      };
      setGizmoDrag({ ...gesture, ...placement });
      preview(gesture, placement);
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
      const placement = {
        position: gesture.position,
        rotation: gesture.rotation,
        dimensions,
      };
      setGizmoDrag({ ...gesture, ...placement });
      preview(gesture, placement);
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
    // A transform whose call could not be located in the source has no line
    // to rewrite; the payload says so with a null, and there is nothing to
    // commit for it.
    const dragged = active ? draggedArgument(finished, node, active) : null;
    if (finished.moved && dragged && placement && placement.line !== null) {
      // The commit is unchanged by the live path: the source is still the
      // truth, and the recompile that follows clears the overrides by
      // installing the very numbers they were standing in for.
      await context.props.onSetValue(
        placement.line,
        placement.call,
        dragged.argument,
        dragged.argument === "radius" || dragged.argument === "height"
          ? dragged.value[0]
          : dragged.value,
      );
      // A patch the server refused recompiles nothing, and a live preview
      // with nothing behind it is worse than no preview.
      context.renderer.dropStaleParameterOverrides();
    } else if (finished.moved) {
      // Nothing will be recompiled, so nothing will clear a live preview.
      context.renderer.setParameterOverrides(null);
    }
  };

  return { hoverAxis, begin, update, commit };
}
