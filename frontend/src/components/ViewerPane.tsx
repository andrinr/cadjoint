/**
 * The 3D viewer and every pointer interaction on it.
 *
 * Pointer priority: a press that lands on a sketch handle starts an edit (drag
 * or select); anything else orbits the camera. In "add vertex" mode a click is
 * intersected with the sketch plane instead. Edits are applied to the Python
 * source through `/patch`, never to a private scene graph.
 */

import { createEffect, onCleanup, onMount } from "solid-js";
import {
  displayProfiles,
  drag,
  gizmoDrag,
  gizmoMode,
  setGizmoDrag,
  hover,
  nodeById,
  profiles,
  selection,
  setDrag,
  setHover,
  dismissViewerError,
  setSelection,
  setStatus,
  setTool,
  tool,
  viewerError,
} from "../state";
import { add, intersectPlane, rayFromPixel, scale, worldToPlane, type Vec3 } from "../viewer/math";
import {
  nearestInsertIndex,
  pickEdge,
  pickNode,
  pickVertex,
  type PickView,
} from "../viewer/hittest";
import {
  AXES,
  angleAroundAxis,
  angleDelta,
  closestPointOnAxis,
  gizmoScale,
  pickGizmoAxis,
  type AxisIndex,
} from "../viewer/gizmo";
import type { GizmoMode } from "../types";
import { Renderer } from "../viewer/renderer";

const PITCH_LIMIT = 1.45;
const ORBIT_SPEED = 0.008;
const PAN_SPEED = 0.0022;

export interface ViewerPaneProps {
  renderer: Renderer;
  /** Apply a patch operation to the source and recompile. */
  onPatch: (
    op: "set_vertex" | "insert_vertex" | "delete_vertex",
    line: number,
    index: number,
    xy?: [number, number],
  ) => Promise<void>;
  /** Rewrite a primitive's placement keyword. */
  onSetValue: (line: number, name: string, argument: string, value: number[]) => Promise<void>;
  /** Insert a new solid into the program. */
  onAddPrimitive: (
    kind: string,
    position: [number, number, number],
    dimensions: Record<string, number | number[]>,
  ) => Promise<void>;
}

type Gesture =
  | { kind: "none" }
  | { kind: "orbit"; x: number; y: number }
  | { kind: "pan"; x: number; y: number }
  | { kind: "drag"; nodeId: string; vertexIndex: number; moved: boolean }
  | {
      kind: "gizmo";
      nodeId: string;
      axis: AxisIndex;
      mode: GizmoMode;
      /** Drag origin: axis parameter for translate, angle for rotate. */
      start: number;
      position: [number, number, number];
      rotation: [number, number, number];
      moved: boolean;
    };

export function ViewerPane(props: ViewerPaneProps) {
  let canvas!: HTMLCanvasElement;
  let gesture: Gesture = { kind: "none" };

  const renderer = props.renderer;

  /** Convert a pointer event to framebuffer pixel coordinates. */
  const toPixels = (event: PointerEvent): [number, number] => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / Math.max(rect.width, 1);
    const scaleY = canvas.height / Math.max(rect.height, 1);
    return [(event.clientX - rect.left) * scaleX, (event.clientY - rect.top) * scaleY];
  };

  const pickView = (): PickView => ({
    position: renderer.cameraPosition,
    target: renderer.camera.target,
    width: renderer.viewport.width,
    height: renderer.viewport.height,
  });

  /** Sketch-plane coordinates under the pointer for a given profile. */
  const planePointAt = (nodeId: string, x: number, y: number): [number, number] | null => {
    const profile = nodeById(nodeId);
    if (!profile?.plane) return null;
    const { origin, normal, u, v } = profile.plane;
    const hit = intersectPlane(rayFromPixel(x, y, pickView()), origin, normal);
    return hit ? worldToPlane(hit, origin, u, v) : null;
  };

  /** Drop a new primitive where the pointer meets the ground plane. */
  const handlePlacePrimitive = async (kind: "box" | "sphere" | "cylinder", x: number, y: number) => {
    const view = pickView();
    const ray = rayFromPixel(x, y, view);
    // Place on the world XY plane, falling back to a point in front of the
    // camera when the view is edge-on to it.
    const hit =
      intersectPlane(ray, [0, 0, 0], [0, 0, 1]) ??
      add(ray.origin, scale(ray.direction, Math.max(1, renderer.camera.distance)));
    const position: [number, number, number] = [hit[0], hit[1], hit[2]];
    const dimensions: Record<string, number | number[]> =
      kind === "box"
        ? { size: [0.5, 0.5, 0.5] }
        : kind === "sphere"
          ? { radius: 0.5 }
          : { radius: 0.4, height: 0.5 };
    await props.onAddPrimitive(kind, position, dimensions);
    setTool("select");
  };

  /** Insert one vertex where the user clicked; the tool stays active. */
  const handleAddVertex = async (x: number, y: number) => {
    const view = pickView();
    const editable = profiles().filter((profile) => profile.editable);
    if (editable.length === 0) {
      setStatus({ kind: "error", text: "No editable sketch in this scene." });
      return;
    }
    // Prefer the edge under the cursor; otherwise fall back to the only sketch
    // (or the selected one) and insert next to its nearest vertex.
    const edge = pickEdge(profiles(), x, y, view);
    const target =
      (edge && nodeById(edge.nodeId)) ??
      (selection() && nodeById(selection()!.nodeId)) ??
      editable[0];
    if (!target.editable) {
      setStatus({ kind: "error", text: "That sketch is not editable from the viewer." });
      return;
    }
    const xy = planePointAt(target.id, x, y);
    if (!xy || target.line === null) {
      setStatus({ kind: "error", text: "Click nearer the sketch plane to place a vertex." });
      return;
    }
    const index =
      edge && edge.nodeId === target.id
        ? edge.insertIndex
        : nearestInsertIndex(target, x, y, view);
    await props.onPatch("insert_vertex", target.line, index, xy);
    setSelection({ nodeId: target.id, vertexIndex: Math.min(index, target.vertices.length) });
  };

  const onPointerDown = (event: PointerEvent) => {
    canvas.setPointerCapture(event.pointerId);
    const [x, y] = toPixels(event);

    if (event.button === 0 && tool() !== "select" && tool() !== "polygon") {
      void handlePlacePrimitive(tool() as "box" | "sphere" | "cylinder", x, y);
      return;
    }

    if (tool() === "polygon" && event.button === 0) {
      void handleAddVertex(x, y);
      return;
    }

    // The gizmo sits on top of everything it can move, so it gets first refusal.
    const target = renderer.gizmoTarget();
    if (event.button === 0 && target) {
      const view = pickView();
      const size = gizmoScale(view, target.origin);
      const axis = pickGizmoAxis(target.origin, size, gizmoMode(), x, y, view);
      if (axis !== null) {
        const transform = target.node.transform!;
        const ray = rayFromPixel(x, y, view);
        const start =
          gizmoMode() === "translate"
            ? closestPointOnAxis(ray, target.origin, AXES[axis])
            : angleAroundAxis(ray, target.origin, AXES[axis]);
        if (start !== null) {
          gesture = {
            kind: "gizmo",
            nodeId: target.node.id,
            axis,
            mode: gizmoMode(),
            start,
            position: [...transform.position] as [number, number, number],
            rotation: [...transform.rotation] as [number, number, number],
            moved: false,
          };
          renderer.gizmoAxis = axis;
          return;
        }
      }
    }

    const hit = event.button === 0 ? pickVertex(displayProfiles(), x, y, pickView()) : null;
    if (hit) {
      const profile = nodeById(hit.nodeId);
      setSelection({ nodeId: hit.nodeId, vertexIndex: hit.vertexIndex });
      if (profile?.editable) {
        gesture = {
          kind: "drag",
          nodeId: hit.nodeId,
          vertexIndex: hit.vertexIndex,
          moved: false,
        };
      } else {
        gesture = { kind: "none" };
      }
      return;
    }

    if (event.button === 0) {
      const node = pickNode(displayProfiles(), x, y, pickView());
      if (node) {
        setSelection({ nodeId: node.nodeId, vertexIndex: null });
        gesture = { kind: "none" };
        return;
      }
      setSelection(null);
    }
    gesture =
      event.button === 2 || event.shiftKey
        ? { kind: "pan", x: event.clientX, y: event.clientY }
        : { kind: "orbit", x: event.clientX, y: event.clientY };
    renderer.interacting = true;
  };

  const onPointerMove = (event: PointerEvent) => {
    const [x, y] = toPixels(event);

    if (gesture.kind === "none") {
      const hit = pickVertex(displayProfiles(), x, y, pickView());
      const next = hit ? { nodeId: hit.nodeId, vertexIndex: hit.vertexIndex } : null;
      const current = hover();
      if (next?.nodeId !== current?.nodeId || next?.vertexIndex !== current?.vertexIndex) {
        setHover(next);
      }
      canvas.style.cursor = hit ? "pointer" : tool() === "polygon" ? "crosshair" : "grab";
      return;
    }

    if (gesture.kind === "gizmo") {
      const view = pickView();
      const ray = rayFromPixel(x, y, view);
      const origin = gesture.position as Vec3;
      const axis = AXES[gesture.axis];
      if (gesture.mode === "translate") {
        const now = closestPointOnAxis(ray, origin, axis);
        const delta = now - gesture.start;
        const position = add(origin, scale(axis, delta)) as [number, number, number];
        gesture.moved = true;
        setGizmoDrag({ ...gesture, position, rotation: gesture.rotation });
      } else {
        const now = angleAroundAxis(ray, origin, axis);
        if (now === null) return;
        const rotation = [...gesture.rotation] as [number, number, number];
        rotation[gesture.axis] += angleDelta(gesture.start, now);
        gesture.moved = true;
        setGizmoDrag({ ...gesture, position: gesture.position, rotation });
      }
      return;
    }

    if (gesture.kind === "drag") {
      const xy = planePointAt(gesture.nodeId, x, y);
      if (!xy) return;
      gesture.moved = true;
      setDrag({ nodeId: gesture.nodeId, vertexIndex: gesture.vertexIndex, xy });
      return;
    }

    if (gesture.kind === "orbit") {
      renderer.camera = {
        ...renderer.camera,
        yaw: renderer.camera.yaw - (event.clientX - gesture.x) * ORBIT_SPEED,
        pitch: Math.max(
          -PITCH_LIMIT,
          Math.min(PITCH_LIMIT, renderer.camera.pitch + (event.clientY - gesture.y) * ORBIT_SPEED),
        ),
      };
      gesture.x = event.clientX;
      gesture.y = event.clientY;
      renderer.invalidate();
      return;
    }

    // Pan: shift the orbit target within the camera's screen plane.
    const dx = (event.clientX - gesture.x) * PAN_SPEED * renderer.camera.distance;
    const dy = (event.clientY - gesture.y) * PAN_SPEED * renderer.camera.distance;
    const { yaw, pitch, target } = renderer.camera;
    const right: [number, number, number] = [Math.cos(yaw), 0, -Math.sin(yaw)];
    const up: [number, number, number] = [
      -Math.sin(yaw) * Math.sin(pitch),
      Math.cos(pitch),
      -Math.cos(yaw) * Math.sin(pitch),
    ];
    renderer.camera = {
      ...renderer.camera,
      target: [
        target[0] - right[0] * dx + up[0] * dy,
        target[1] - right[1] * dx + up[1] * dy,
        target[2] - right[2] * dx + up[2] * dy,
      ],
    };
    gesture.x = event.clientX;
    gesture.y = event.clientY;
    renderer.invalidate();
  };

  const finishGesture = async (event: PointerEvent) => {
    if (canvas.hasPointerCapture(event.pointerId)) {
      canvas.releasePointerCapture(event.pointerId);
    }
    const finished = gesture;
    gesture = { kind: "none" };
    renderer.interacting = false;

    if (finished.kind === "gizmo") {
      const active = gizmoDrag();
      const node = nodeById(finished.nodeId);
      setGizmoDrag(null);
      renderer.gizmoAxis = null;
      if (finished.moved && active && node?.line != null) {
        const argument = finished.mode === "translate" ? "position" : "rotation";
        const value = finished.mode === "translate" ? active.position : active.rotation;
        await props.onSetValue(node.line, node.kind, argument, value);
      }
      return;
    }

    if (finished.kind === "drag") {
      const active = drag();
      const profile = nodeById(finished.nodeId);
      setDrag(null);
      if (finished.moved && active && profile?.line != null) {
        await props.onPatch("set_vertex", profile.line, finished.vertexIndex, active.xy);
      }
      return;
    }
    if (finished.kind !== "none") renderer.invalidate();
  };

  const onWheel = (event: WheelEvent) => {
    event.preventDefault();
    renderer.camera = {
      ...renderer.camera,
      distance: Math.max(
        0.4,
        Math.min(60, renderer.camera.distance * Math.exp(event.deltaY * 0.001)),
      ),
    };
    renderer.invalidate();
  };

  onMount(() => {
    // init() binds the canvas synchronously before its first await, so the
    // viewport is sized even when WebGPU initialisation later fails.
    void renderer.init(canvas);
    renderer.resize();

    const observer = new ResizeObserver(() => {
      renderer.resize();
      renderer.invalidate();
    });
    observer.observe(canvas);

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setSelection(null);
        setTool("select");
      }
      const active = selection();
      if ((event.key === "Delete" || event.key === "Backspace") && active) {
        const target = document.activeElement;
        if (target && (target.tagName === "TEXTAREA" || target.closest(".cm-editor"))) return;
        const profile = nodeById(active.nodeId);
        if (profile?.editable && profile.line !== null && active.vertexIndex !== null) {
          event.preventDefault();
          void props.onPatch("delete_vertex", profile.line, active.vertexIndex);
          setSelection(null);
        }
      }
    };
    window.addEventListener("keydown", onKeyDown);

    onCleanup(() => {
      observer.disconnect();
      window.removeEventListener("keydown", onKeyDown);
      renderer.destroy();
    });
  });

  // Keep the GPU overlay buffers in step with the construction tree.
  createEffect(() => {
    renderer.setConstruction(displayProfiles(), selection(), hover());
  });

  return (
    <section class="pane viewer-pane">
      <canvas
        ref={canvas}
        class="viewer-canvas"
        data-testid="viewer-canvas"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={(event) => void finishGesture(event)}
        onPointerCancel={(event) => void finishGesture(event)}
        onWheel={onWheel}
        onContextMenu={(event) => event.preventDefault()}
        onDblClick={() => {
          renderer.camera = { ...renderer.camera, target: [0, 0, 0] };
          renderer.invalidate();
        }}
      />
      {viewerError() && (
        <div class="viewer-error">
          <p>{viewerError()}</p>
          <button type="button" onClick={dismissViewerError}>
            Dismiss
          </button>
        </div>
      )}
      <p class="viewer-hint">
        {tool() === "polygon"
          ? "Polygon: click sketch edges to add vertices · Esc to finish"
          : tool() !== "select"
            ? `Click to place a ${tool()} · Esc to cancel`
            : "Drag handles or the gizmo · Drag empty space to orbit · Shift/right-drag to pan"}
      </p>
    </section>
  );
}
