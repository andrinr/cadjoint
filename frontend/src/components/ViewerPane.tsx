/**
 * The 3D viewer and every pointer interaction on it.
 *
 * Pointer priority: a press that lands on a sketch handle starts an edit (drag
 * or select); anything else orbits the camera. In "add vertex" mode a click is
 * intersected with the sketch plane instead. Edits are applied to the Python
 * source through `/patch`, never to a private scene graph.
 *
 * What stays here is the dispatch: which gesture a press begins, how a move
 * advances it, and what a release commits. Everything a gesture then *does*
 * lives beside it in `components/viewer/` — the tool handlers, the gizmo's
 * life cycle, the FEM probe, the camera arithmetic, the keyboard table, and
 * the DOM layers drawn over the canvas — so this file stays a state machine
 * rather than a pile of features.
 */

import { createEffect, createMemo, createSignal, onCleanup, onMount } from "solid-js";
import { MATERIAL_DRAG_TYPE } from "./MaterialPanel";
import {
  displayProfiles,
  editingMode,
  meshEdges,
  drag,
  cameraAngles,
  busy,
  bcPickArmed,
  faceHover,
  setFaceHover,
  selectionMode,
  setCameraAngles,
  setBcProposal,
  hover,
  nodeById,
  pendingLoft,
  relations,
  selection,
  setDrag,
  setHover,
  setGizmoMode,
  setSelection,
  setSimProbe,
  setStatus,
  simView,
  tool,
} from "../state";
import { rectAabbProposal } from "../bcPick";
import { intersectPlane, rayFromPixel, worldToPlane } from "../viewer/math";
import { GRID_ALPHA } from "../viewer/graticule";
import { pickEdge, pickNode, pickVertex, type PickView } from "../viewer/hittest";
import {
  CONSTRAINT_TOOL_NAMES,
  isEdgeConstraintTool,
  isVertexConstraintTool,
} from "../constraints";
import { detentZoomCamera, orbitCamera, panCamera, zoomCamera } from "./viewer/camera";
import { buildConstraintOverlay } from "./viewer/constraintMarks";
import { ConstraintOverlay } from "./viewer/ConstraintOverlay";
import { Graticule } from "./viewer/Graticule";
import { createGizmoDrag } from "./viewer/gizmoDrag";
import { createSimInteraction } from "./viewer/simInteraction";
import { createViewerKeyboard } from "./viewer/keyboard";
import { createViewerTools } from "./viewer/tools";
import { ViewerHint } from "./viewer/ViewerHint";
import { ViewerOverlays, type PickRect } from "./viewer/ViewerOverlays";
import type { Gesture, PendingConstraint } from "./viewer/gestures";
import type { ViewerPaneProps } from "./viewer/props";
import { DOCK_REBUILT_EVENT } from "../windows/events";

export type { ViewerPaneProps } from "./viewer/props";

export function ViewerPane(props: ViewerPaneProps) {
  let canvas!: HTMLCanvasElement;
  let gesture: Gesture = { kind: "none" };
  const [pendingConstraint, setPendingConstraint] =
    createSignal<PendingConstraint | null>(null);
  /** Held space turns any drag into a pan, as other 3D viewports do. */
  let panHeld = false;
  const [materialDropActive, setMaterialDropActive] = createSignal(false);
  const [overlayRevision, setOverlayRevision] = createSignal(0);
  /** BC box-pick rubber band, in CSS pixels over the canvas. */
  const [pickRect, setPickRect] = createSignal<PickRect | null>(null);

  const renderer = props.renderer;

  /** Convert a pointer event to framebuffer pixel coordinates. */
  const toPixels = (event: { clientX: number; clientY: number }): [number, number] => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / Math.max(rect.width, 1);
    const scaleY = canvas.height / Math.max(rect.height, 1);
    return [(event.clientX - rect.left) * scaleX, (event.clientY - rect.top) * scaleY];
  };

  // Use the renderer's complete descriptor so orthographic projection and its
  // viewport scale are identical for drawing, object picking, and gizmo hits.
  const pickView = (): PickView => renderer.view;

  /** Reproject DOM annotations after camera or viewport changes. */
  const refreshOverlays = () => setOverlayRevision((revision) => revision + 1);

  /** Sketch-plane coordinates under the pointer for a given profile. */
  const planePointAt = (nodeId: string, x: number, y: number): [number, number] | null => {
    const profile = nodeById(nodeId);
    if (!profile?.plane) return null;
    const { origin, normal, u, v } = profile.plane;
    const hit = intersectPlane(rayFromPixel(x, y, pickView()), origin, normal);
    return hit ? worldToPlane(hit, origin, u, v) : null;
  };

  const tools = createViewerTools({
    renderer,
    pickView,
    planePointAt,
    pendingConstraint,
    setPendingConstraint,
    props,
  });
  const gizmo = createGizmoDrag({ renderer, pickView, props });
  const sim = createSimInteraction({ canvas: () => canvas, renderer, pickView });

  /**
   * The camera as the graticule's labels see it.
   *
   * `overlayRevision` is bumped by every gesture that moves the camera, which
   * is exactly the set of events that can change the gain — so the readout is
   * driven by the same signal that reprojects the DOM annotations, and cannot
   * drift from what the shader drew.
   */
  const graticuleCamera = createMemo(() => {
    overlayRevision();
    const { distance, yaw, pitch } = renderer.camera;
    return { distance, yaw, pitch, projection: props.display.projection };
  });

  const constraintOverlay = createMemo(() => {
    overlayRevision();
    return buildConstraintOverlay(
      pickView(),
      canvas?.clientWidth ?? 0,
      canvas?.clientHeight ?? 0,
      displayProfiles(),
      relations(),
    );
  });

  /**
   * Reflect what the pointer can act on right now.
   *
   * Cursor vocabulary: crosshair places, grab drags a handle or gizmo, pointer
   * selects, and the default arrow means "nothing here" — dragging there orbits.
   */
  const updateHover = (x: number, y: number) => {
    if (panHeld) {
      canvas.style.cursor = "move";
      return;
    }
    // The face tool resolves an analytic face rather than a pixel: the hover
    // has to answer "which face" before a click can, and the highlight the
    // renderer draws is that same answer.
    if (tool() === "face") {
      const pick = tools.resolveFace(x, y);
      if (pick?.face.id !== faceHover()?.face.id) setFaceHover(pick);
      canvas.style.cursor = pick ? "crosshair" : "cell";
      setHover(null);
      renderer.gizmoAxis = null;
      return;
    }
    if (isVertexConstraintTool(tool())) {
      const hit = pickVertex(displayProfiles(), x, y, pickView());
      const next = hit ? { nodeId: hit.nodeId, vertexIndex: hit.vertexIndex } : null;
      setHover(next);
      canvas.style.cursor = hit ? "crosshair" : "default";
      renderer.gizmoAxis = null;
      return;
    }
    if (isEdgeConstraintTool(tool())) {
      const hit = pickEdge(displayProfiles(), x, y, pickView());
      setHover(null);
      canvas.style.cursor = hit ? "crosshair" : "default";
      renderer.gizmoAxis = null;
      return;
    }
    if (tool() !== "select") {
      canvas.style.cursor = "crosshair";
      setHover(null);
      renderer.gizmoAxis = null;
      return;
    }
    const view = pickView();

    if (gizmo.hoverAxis(x, y, view)) {
      canvas.style.cursor = "grab";
      return;
    }

    if (selectionMode() === "vertex") {
      const hit = pickVertex(displayProfiles(), x, y, view);
      const next = hit ? { nodeId: hit.nodeId, vertexIndex: hit.vertexIndex } : null;
      const current = hover();
      if (next?.nodeId !== current?.nodeId || next?.vertexIndex !== current?.vertexIndex) {
        setHover(next);
      }
      canvas.style.cursor = hit ? "grab" : "default";
      return;
    }

    const node = pickNode(displayProfiles(), x, y, view, 10, true);
    const next = node ? { nodeId: node.nodeId, vertexIndex: null } : null;
    const current = hover();
    if (next?.nodeId !== current?.nodeId || next?.vertexIndex !== current?.vertexIndex) {
      setHover(next);
    }
    canvas.style.cursor = node ? "pointer" : "default";
  };

  const onPointerDown = (event: PointerEvent) => {
    canvas.setPointerCapture(event.pointerId);
    const [x, y] = toPixels(event);

    // Simulate-mode surface interactions come first: with a FEM mesh shown,
    // a left press is a pending probe/pick tap (drags fall through to orbit)
    // and Shift-drag with armed picking rubber-bands a Nodes.box proposal.
    if (sim.simInteractive() && event.button === 0 && !panHeld) {
      if (bcPickArmed() && event.shiftKey) {
        gesture = { kind: "bcrect", x0: x, y0: y, x1: x, y1: y };
        setPickRect(sim.rectFromGesture(gesture));
        return;
      }
      if (!event.shiftKey) {
        gesture = { kind: "simtap", x, y, clientX: event.clientX, clientY: event.clientY };
        return;
      }
    }

    if (tool() === "sketch" && event.button === 0) {
      void tools.handlePlaceSketch(x, y);
      return;
    }

    if (tool() === "face" && event.button === 0) {
      void tools.handleSketchOnFace(x, y);
      return;
    }

    // A pending loft claims the next object pick, like the constraint flows.
    if (pendingLoft() && event.button === 0) {
      void tools.handleLoftPick(x, y);
      return;
    }

    const activeTool = tool();
    if (isVertexConstraintTool(activeTool) && event.button === 0) {
      void tools.handleVertexConstraint(activeTool, x, y);
      return;
    }

    if (isEdgeConstraintTool(activeTool) && event.button === 0) {
      void tools.handleEdgeConstraint(activeTool, x, y);
      return;
    }

    if (
      event.button === 0 &&
      (tool() === "box" || tool() === "sphere" || tool() === "cylinder")
    ) {
      void tools.handlePlacePrimitive(tool() as "box" | "sphere" | "cylinder", x, y);
      return;
    }

    if (tool() === "polygon" && event.button === 0) {
      void tools.handleAddVertex(x, y);
      return;
    }

    // The gizmo sits on top of everything it can move, so it gets first refusal.
    if (event.button === 0) {
      const grabbed = gizmo.begin(x, y);
      if (grabbed !== null) {
        gesture = grabbed;
        renderer.gizmoAxis = grabbed.axis;
        renderer.interacting = true;
        renderer.invalidate();
        return;
      }
    }

    const hit =
      event.button === 0 && selectionMode() === "vertex"
        ? pickVertex(displayProfiles(), x, y, pickView())
        : null;
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
        renderer.interacting = true;
        renderer.invalidate();
      } else {
        gesture = { kind: "none" };
      }
      return;
    }

    if (event.button === 0 && selectionMode() === "object") {
      const node = pickNode(displayProfiles(), x, y, pickView(), 10, true);
      if (node) {
        if (nodeById(node.nodeId)?.kind === "profile") {
          setGizmoMode("translate");
        }
        setSelection({ nodeId: node.nodeId, vertexIndex: null });
        gesture = { kind: "none" };
        return;
      }
      setSelection(null);
    } else if (event.button === 0) {
      setSelection(null);
    }
    gesture =
      event.button === 2 || event.shiftKey || panHeld
        ? { kind: "pan", x: event.clientX, y: event.clientY }
        : { kind: "orbit", x: event.clientX, y: event.clientY };
    renderer.interacting = true;
  };

  const onPointerMove = (event: PointerEvent) => {
    const [x, y] = toPixels(event);

    if (gesture.kind === "none") {
      updateHover(x, y);
      return;
    }

    // A sim tap that travels becomes a plain orbit; the probe chip clears so
    // it does not float detached from the point it annotated.
    if (gesture.kind === "simtap") {
      const travel = Math.hypot(
        event.clientX - gesture.clientX,
        event.clientY - gesture.clientY,
      );
      if (travel > 4) {
        setSimProbe(null);
        gesture = { kind: "orbit", x: event.clientX, y: event.clientY };
        renderer.interacting = true;
      }
      return;
    }

    if (gesture.kind === "bcrect") {
      gesture.x1 = x;
      gesture.y1 = y;
      setPickRect(sim.rectFromGesture(gesture));
      return;
    }

    if (gesture.kind === "gizmo") {
      canvas.style.cursor = "grabbing";
      gizmo.update(gesture, x, y);
      return;
    }

    if (gesture.kind === "drag") {
      canvas.style.cursor = "grabbing";
      const xy = planePointAt(gesture.nodeId, x, y);
      if (!xy) return;
      gesture.moved = true;
      setDrag({ nodeId: gesture.nodeId, vertexIndex: gesture.vertexIndex, xy });
      return;
    }

    if (gesture.kind === "orbit") {
      canvas.style.cursor = "grabbing";
      renderer.camera = orbitCamera(
        renderer.camera,
        event.clientX - gesture.x,
        event.clientY - gesture.y,
      );
      gesture.x = event.clientX;
      gesture.y = event.clientY;
      setCameraAngles({ yaw: renderer.camera.yaw, pitch: renderer.camera.pitch });
      refreshOverlays();
      renderer.invalidate();
      return;
    }

    canvas.style.cursor = "move";
    renderer.camera = panCamera(
      renderer.camera,
      event.clientX - gesture.x,
      event.clientY - gesture.y,
    );
    gesture.x = event.clientX;
    gesture.y = event.clientY;
    refreshOverlays();
    renderer.invalidate();
  };

  const finishGesture = async (event: PointerEvent) => {
    if (canvas.hasPointerCapture(event.pointerId)) {
      canvas.releasePointerCapture(event.pointerId);
    }
    const finished = gesture;
    gesture = { kind: "none" };
    renderer.interacting = false;
    canvas.style.cursor = "default";

    if (finished.kind === "simtap") {
      sim.handleSimTap(finished.x, finished.y);
      return;
    }

    if (finished.kind === "bcrect") {
      setPickRect(null);
      const view = simView();
      if (view) {
        const proposal = rectAabbProposal(view.payload.positions, finished, pickView());
        if (proposal) setBcProposal(proposal);
        else setStatus({ kind: "error", text: "Box pick: drag the rectangle over the mesh." });
      }
      return;
    }

    if (finished.kind === "gizmo") {
      await gizmo.commit(finished);
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

  // Free zoom is the default; Alt holds the graticule's 1-2-5 detent, so one
  // division is an exact number of millimetres and the gain readout drops its
  // uncalibrated `>` prefix.
  const onWheel = (event: WheelEvent) => {
    event.preventDefault();
    renderer.camera = event.altKey
      ? detentZoomCamera(renderer.camera, event.deltaY)
      : zoomCamera(renderer.camera, event.deltaY);
    refreshOverlays();
    renderer.invalidate();
  };

  const onMaterialDrop = async (event: DragEvent) => {
    event.preventDefault();
    setMaterialDropActive(false);
    const material =
      event.dataTransfer?.getData(MATERIAL_DRAG_TYPE) ??
      event.dataTransfer?.getData("text/plain");
    if (!material) return;
    const [x, y] = toPixels(event);
    // A generous radius makes dropping on the shaded interior feel natural
    // even though construction picking is based on projected wireframes.
    const hit = pickNode(displayProfiles(), x, y, pickView(), 64, true);
    const node = hit && nodeById(hit.nodeId);
    if (!node?.editable || node.line === null) {
      setStatus({ kind: "error", text: "Drop closer to an editable object." });
      return;
    }
    setSelection({ nodeId: node.id, vertexIndex: null });
    setStatus({ kind: "", text: `Applying ${material}…` });
    await props.onAssignMaterial(node.line, material);
  };

  const keyboard = createViewerKeyboard({
    props,
    clearPendingConstraint: () => setPendingConstraint(null),
    setPanHeld: (held) => {
      panHeld = held;
      canvas.style.cursor = panHeld ? "move" : "default";
    },
  });

  onMount(() => {
    // init() binds the canvas synchronously before its first await, so the
    // viewport is sized even when WebGPU initialisation later fails.
    void renderer.init(canvas);
    renderer.resize();

    const observer = new ResizeObserver(() => {
      renderer.resize();
      refreshOverlays();
      renderer.invalidate();
    });
    observer.observe(canvas);

    // A dock rebuild moves this pane between the library's own wrappers. The
    // canvas element survives that (which is why the GPU context does), but
    // the swap chain is re-attached here in case the new layout happens to
    // hand the viewport the same rectangle and `resize()` short-circuits.
    const onDockRebuilt = () => {
      renderer.reconfigure();
      renderer.resize();
      refreshOverlays();
      renderer.invalidate();
    };

    window.addEventListener("keydown", keyboard.onKeyDown);
    window.addEventListener("keydown", keyboard.onPanKey);
    window.addEventListener("keyup", keyboard.onPanKey);
    window.addEventListener(DOCK_REBUILT_EVENT, onDockRebuilt);

    onCleanup(() => {
      observer.disconnect();
      window.removeEventListener("keydown", keyboard.onKeyDown);
      window.removeEventListener("keydown", keyboard.onPanKey);
      window.removeEventListener("keyup", keyboard.onPanKey);
      window.removeEventListener(DOCK_REBUILT_EVENT, onDockRebuilt);
      renderer.destroy();
    });
  });

  // Keep the GPU overlay buffers in step with the construction tree.
  createEffect(() => {
    renderer.setConstruction(displayProfiles(), selection(), hover());
  });

  createEffect(() => {
    renderer.setMeshEdges(meshEdges());
  });

  // The highlight is a readout of the hover, and it never outlives the tool
  // that armed it: leaving the face tool clears both.
  createEffect(() => {
    if (tool() !== "face") {
      if (faceHover() !== null) setFaceHover(null);
      renderer.setFaceHighlight(null);
      return;
    }
    renderer.setFaceHighlight(faceHover()?.face ?? null);
  });

  /**
   * Step the floor grid back while sketching off the floor.
   *
   * A sketch on the XZ plane — which is where the starter's fin comb lives —
   * is drawn standing up, and a floor ruled underneath it argues with it for
   * the eye. The floor still has a job (it says which way up the world is), so
   * it dims rather than disappearing. On the XY plane itself the sketch and
   * the floor are the same plane and there is nothing to arbitrate.
   */
  createEffect(() => {
    const active = selection();
    const node = active ? nodeById(active.nodeId) : null;
    const normal = node?.plane?.normal;
    const onFloor = !normal || Math.abs(normal[2]) > 0.999;
    const dim = editingMode() === "sketch" && !onFloor;
    const next = dim ? GRID_ALPHA.offPlane : 1;
    if (renderer.groundEmphasis === next) return;
    renderer.groundEmphasis = next;
    renderer.invalidate();
  });

  createEffect(() => {
    cameraAngles();
    props.display.projection;
    refreshOverlays();
  });

  createEffect(() => {
    const activeTool = tool();
    if (!isVertexConstraintTool(activeTool) && !isEdgeConstraintTool(activeTool)) {
      setPendingConstraint(null);
      return;
    }
    // A first pick made with another constraint tool does not carry over.
    if (pendingConstraint() && pendingConstraint()!.kind !== activeTool) {
      setPendingConstraint(null);
    }
    if (!isVertexConstraintTool(activeTool)) return;
    // Activating a vertex-pair tool with a point selected uses it as the start.
    const active = selection();
    if (!pendingConstraint() && active?.vertexIndex != null) {
      const node = nodeById(active.nodeId);
      if (node?.kind === "profile") {
        setPendingConstraint({
          kind: activeTool,
          first: { nodeId: active.nodeId, vertexIndex: active.vertexIndex },
        });
        setStatus({
          kind: "",
          text: `${CONSTRAINT_TOOL_NAMES[activeTool]}: choose the second point`,
        });
      }
    }
  });

  return (
    <section
      class="pane viewer-pane"
      classList={{
        "has-graticule": props.display.showGraticule,
        // Every readout over the canvas needs a paper plate once the viewport
        // is showing data rather than a lit scene.
        "sdf-data": props.display.sdfView !== "solid",
      }}
      aria-busy={busy()}
    >
      {props.overlay}
      <canvas
        ref={canvas}
        class="viewer-canvas"
        classList={{ "material-drop-active": materialDropActive() }}
        data-testid="viewer-canvas"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={(event) => void finishGesture(event)}
        onPointerCancel={(event) => void finishGesture(event)}
        onWheel={onWheel}
        onDragEnter={(event) => {
          if (event.dataTransfer?.types.includes(MATERIAL_DRAG_TYPE)) {
            setMaterialDropActive(true);
          }
        }}
        onDragOver={(event) => {
          if (!event.dataTransfer?.types.includes(MATERIAL_DRAG_TYPE)) return;
          event.preventDefault();
          if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
        }}
        onDragLeave={() => setMaterialDropActive(false)}
        onDrop={(event) => void onMaterialDrop(event)}
        onContextMenu={(event) => event.preventDefault()}
        onDblClick={() => {
          renderer.camera = { ...renderer.camera, target: [0, 0, 0] };
          refreshOverlays();
          renderer.invalidate();
        }}
      />
      <ConstraintOverlay
        show={props.display.showOverlays && props.display.showConstraints}
        showDistance={props.display.showDistanceConstraints}
        showFixed={props.display.showFixedConstraints}
        showValues={props.display.showConstraintValues}
        geometry={constraintOverlay()}
      />
      <Graticule
        show={props.display.showGraticule}
        camera={graticuleCamera()}
        sdfView={props.display.sdfView}
        sdfAxis={props.display.sdfAxis}
        sdfFraction={props.display.sdfFraction}
      />
      <ViewerOverlays pickRect={props.display.showOverlays ? pickRect() : null} />
      <ViewerHint pendingConstraint={pendingConstraint()} />
    </section>
  );
}
