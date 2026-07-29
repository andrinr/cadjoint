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
  hover,
  profileById,
  profiles,
  selection,
  setDrag,
  setHover,
  setSelection,
  setStatus,
  setTool,
  setViewerError,
  tool,
  viewerError,
} from "../state";
import { intersectPlane, rayFromPixel, worldToPlane } from "../viewer/math";
import { nearestInsertIndex, pickEdge, pickVertex, type PickView } from "../viewer/hittest";
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
}

type Gesture =
  | { kind: "none" }
  | { kind: "orbit"; x: number; y: number }
  | { kind: "pan"; x: number; y: number }
  | { kind: "drag"; profileId: string; vertexIndex: number; moved: boolean };

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
  const planePointAt = (profileId: string, x: number, y: number): [number, number] | null => {
    const profile = profileById(profileId);
    if (!profile) return null;
    const view = pickView();
    const ray = rayFromPixel(x, y, view);
    const hit = intersectPlane(ray, profile.plane.origin, profile.plane.normal);
    if (!hit) return null;
    return worldToPlane(hit, profile.plane.origin, profile.plane.u, profile.plane.v);
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
      (edge && profileById(edge.profileId)) ??
      (selection() && profileById(selection()!.profileId)) ??
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
      edge && edge.profileId === target.id
        ? edge.insertIndex
        : nearestInsertIndex(target, x, y, view);
    await props.onPatch("insert_vertex", target.line, index, xy);
    setSelection({ profileId: target.id, vertexIndex: Math.min(index, target.vertices.length) });
  };

  const onPointerDown = (event: PointerEvent) => {
    canvas.setPointerCapture(event.pointerId);
    const [x, y] = toPixels(event);

    if (tool() === "polygon" && event.button === 0) {
      void handleAddVertex(x, y);
      return;
    }

    const hit = event.button === 0 ? pickVertex(displayProfiles(), x, y, pickView()) : null;
    if (hit) {
      const profile = profileById(hit.profileId);
      setSelection({ profileId: hit.profileId, vertexIndex: hit.vertexIndex });
      if (profile?.editable) {
        gesture = {
          kind: "drag",
          profileId: hit.profileId,
          vertexIndex: hit.vertexIndex,
          moved: false,
        };
      } else {
        gesture = { kind: "none" };
      }
      return;
    }

    if (event.button === 0) setSelection(null);
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
      const next = hit ? { profileId: hit.profileId, vertexIndex: hit.vertexIndex } : null;
      const current = hover();
      if (next?.profileId !== current?.profileId || next?.vertexIndex !== current?.vertexIndex) {
        setHover(next);
      }
      canvas.style.cursor = hit ? "pointer" : tool() === "polygon" ? "crosshair" : "grab";
      return;
    }

    if (gesture.kind === "drag") {
      const xy = planePointAt(gesture.profileId, x, y);
      if (!xy) return;
      gesture.moved = true;
      setDrag({ profileId: gesture.profileId, vertexIndex: gesture.vertexIndex, xy });
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

    if (finished.kind === "drag") {
      const active = drag();
      const profile = profileById(finished.profileId);
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
        const profile = profileById(active.profileId);
        if (profile?.editable && profile.line !== null) {
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
          <button type="button" onClick={() => setViewerError("")}>
            Dismiss
          </button>
        </div>
      )}
      <p class="viewer-hint">
        {tool() === "polygon"
          ? "Polygon: click sketch edges to add vertices · Esc to finish"
          : "Drag handles to edit · Drag empty space to orbit · Shift/right-drag to pan"}
      </p>
    </section>
  );
}
