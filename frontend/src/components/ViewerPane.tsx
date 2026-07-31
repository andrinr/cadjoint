/**
 * The 3D viewer and every pointer interaction on it.
 *
 * Pointer priority: a press that lands on a sketch handle starts an edit (drag
 * or select); anything else orbits the camera. In "add vertex" mode a click is
 * intersected with the sketch plane instead. Edits are applied to the Python
 * source through `/patch`, never to a private scene graph.
 */

import {
  For,
  Show,
  createEffect,
  createMemo,
  createSignal,
  onCleanup,
  onMount,
  type JSX,
} from "solid-js";
import { MATERIAL_DRAG_TYPE } from "./MaterialPanel";
import {
  displayProfiles,
  drag,
  gizmoDrag,
  cameraAngles,
  busy,
  selectionMode,
  setCameraAngles,
  setGizmoDrag,
  setGizmoMode,
  hover,
  nodeById,
  nodes,
  profiles,
  relations,
  selection,
  setDrag,
  setHover,
  dismissViewerError,
  setSelection,
  setSelectionMode,
  setStatus,
  setTool,
  tool,
  viewerError,
} from "../state";
import {
  add,
  intersectPlane,
  projectPoint,
  rayFromPixel,
  scale,
  worldToPlane,
  type Vec3,
} from "../viewer/math";
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
  scaleDimensions,
  type AxisIndex,
} from "../viewer/gizmo";
import type { GizmoMode } from "../types";
import { Renderer, type DisplaySettings } from "../viewer/renderer";

const PITCH_LIMIT = 1.45;
const ORBIT_SPEED = 0.008;
const PAN_SPEED = 0.0022;

export interface ViewerPaneProps {
  renderer: Renderer;
  /** View-only settings, including construction annotation visibility. */
  display: DisplaySettings;
  /** Overlays rendered on top of the canvas (tool rail, ViewCube). */
  overlay?: JSX.Element;
  /** Apply a patch operation to the source and recompile. */
  onPatch: (
    op: "set_vertex" | "insert_vertex" | "delete_vertex",
    line: number,
    index: number,
    xy?: [number, number],
  ) => Promise<void>;
  /** Rewrite a primitive's placement keyword. */
  onSetValue: (
    line: number,
    name: string,
    argument: string,
    value: number | number[],
  ) => Promise<void>;
  /** Insert a new solid into the program. */
  onAddPrimitive: (
    kind: string,
    position: [number, number, number],
    dimensions: Record<string, number | number[]>,
  ) => Promise<void>;
  /** Insert a standalone sketch at a world-space origin. */
  onAddSketch: (origin: [number, number, number]) => Promise<void>;
  /** Attach a source-level constraint to sketch vertices. */
  onAddConstraint: (
    line: number,
    kind: "fixed" | "distance",
    indices: number[],
    value: number | number[],
  ) => Promise<void>;
  /** Remove a whole construction object from the program. */
  onDeleteObject: (line: number) => Promise<void>;
  /** Assign a named Python material to the object under a drop. */
  onAssignMaterial: (line: number, material: string) => Promise<void>;
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
      dimensions: Record<string, number | number[]>;
      /** World length represented by the on-screen scale handle. */
      scaleLength: number;
      /** World position where the visible gizmo was placed. */
      gizmoOrigin: [number, number, number];
      moved: boolean;
    };

export function ViewerPane(props: ViewerPaneProps) {
  let canvas!: HTMLCanvasElement;
  let gesture: Gesture = { kind: "none" };
  let distanceStart: { nodeId: string; vertexIndex: number } | null = null;
  /** Held space turns any drag into a pan, as other 3D viewports do. */
  let panHeld = false;
  const [materialDropActive, setMaterialDropActive] = createSignal(false);
  const [overlayRevision, setOverlayRevision] = createSignal(0);

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

  const constraintOverlay = createMemo(() => {
    overlayRevision();
    const width = canvas?.clientWidth ?? 0;
    const height = canvas?.clientHeight ?? 0;
    if (width <= 0 || height <= 0) return { width, height, fixed: [], distance: [] };
    const view = pickView();
    const scaleX = width / Math.max(view.width, 1);
    const scaleY = height / Math.max(view.height, 1);
    const screen = (world: Vec3) => {
      const point = projectPoint(world, view);
      return {
        x: point.x * scaleX,
        y: point.y * scaleY,
        visible: point.visible,
      };
    };
    const fixed: {
      x: number;
      y: number;
      key: string;
      scope: "sketch" | "object";
    }[] = [];
    const distance: {
      ax: number;
      ay: number;
      bx: number;
      by: number;
      dax: number;
      day: number;
      dbx: number;
      dby: number;
      labelX: number;
      labelY: number;
      label: string;
      key: string;
      scope: "sketch" | "object";
    }[] = [];

    const addDistance = (
      first: Vec3,
      second: Vec3,
      value: number,
      key: string,
      scope: "sketch" | "object",
    ) => {
      const a = screen(first);
      const b = screen(second);
      if (!a.visible || !b.visible) return;
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const length = Math.hypot(dx, dy);
      if (length < 2) return;
      const offsetX = (-dy / length) * 17;
      const offsetY = (dx / length) * 17;
      const dax = a.x + offsetX;
      const day = a.y + offsetY;
      const dbx = b.x + offsetX;
      const dby = b.y + offsetY;
      distance.push({
        ax: a.x,
        ay: a.y,
        bx: b.x,
        by: b.y,
        dax,
        day,
        dbx,
        dby,
        labelX: (dax + dbx) * 0.5,
        labelY: (day + dby) * 0.5 - 5,
        label: Number(value.toPrecision(4)).toString(),
        key,
        scope,
      });
    };

    const displayedNodes = displayProfiles();
    for (const profile of displayedNodes) {
      if (profile.kind !== "profile") continue;
      for (let index = 0; index < profile.constraints.length; index++) {
        const constraint = profile.constraints[index];
        if (constraint.kind === "fixed") {
          const vertex = profile.vertices[constraint.vertices[0]];
          if (!vertex) continue;
          const point = screen(vertex.world);
          if (point.visible) {
            fixed.push({
              x: point.x + 9,
              y: point.y - 9,
              key: `${profile.id}-fixed-${index}`,
              scope: "sketch",
            });
          }
          continue;
        }

        const first = profile.vertices[constraint.vertices[0]];
        const second = profile.vertices[constraint.vertices[1]];
        if (!first || !second || typeof constraint.value !== "number") continue;
        addDistance(
          first.world,
          second.world,
          constraint.value,
          `${profile.id}-distance-${index}`,
          "sketch",
        );
      }
    }

    const displayedById = new Map(displayedNodes.map((node) => [node.id, node]));
    for (let index = 0; index < relations().length; index++) {
      const relation = relations()[index];
      if (relation.kind === "fixed") {
        const node = displayedById.get(relation.nodes[0]);
        if (!node?.transform) continue;
        const point = screen(node.transform.position);
        if (point.visible) {
          fixed.push({
            x: point.x + 9,
            y: point.y - 9,
            key: `object-fixed-${index}`,
            scope: "object",
          });
        }
        continue;
      }
      const first = displayedById.get(relation.nodes[0]);
      const second = displayedById.get(relation.nodes[1]);
      if (!first?.transform || !second?.transform || typeof relation.value !== "number") {
        continue;
      }
      addDistance(
        first.transform.position,
        second.transform.position,
        relation.value,
        `object-distance-${index}`,
        "object",
      );
    }
    return { width, height, fixed, distance };
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
    setSelectionMode("object");
    // Select what was just placed so the gizmo is ready without a second click.
    const placed = nodes().filter((node) => node.kind === kind);
    const newest = placed[placed.length - 1];
    if (newest) setSelection({ nodeId: newest.id, vertexIndex: null });
  };

  /** Place a new identity-oriented sketch where the pointer meets world XY. */
  const handlePlaceSketch = async (x: number, y: number) => {
    const view = pickView();
    const ray = rayFromPixel(x, y, view);
    const hit =
      intersectPlane(ray, [0, 0, 0], [0, 0, 1]) ??
      add(ray.origin, scale(ray.direction, Math.max(1, renderer.camera.distance)));
    await props.onAddSketch([hit[0], hit[1], hit[2]]);
    setTool("select");
    setSelectionMode("object");
    const sketches = profiles();
    const newest = sketches[sketches.length - 1];
    if (newest) setSelection({ nodeId: newest.id, vertexIndex: null });
  };

  /** Pick two points and preserve their current sketch-plane distance. */
  const handleDistanceConstraint = async (x: number, y: number) => {
    const hit = pickVertex(displayProfiles(), x, y, pickView());
    if (!hit) {
      setStatus({ kind: "error", text: "Distance: click a sketch point." });
      return;
    }
    const profile = nodeById(hit.nodeId);
    if (!profile?.editable || profile.line === null) {
      setStatus({ kind: "error", text: "That sketch cannot be edited from source." });
      return;
    }
    if (!distanceStart) {
      distanceStart = { nodeId: hit.nodeId, vertexIndex: hit.vertexIndex };
      setSelection({ nodeId: hit.nodeId, vertexIndex: hit.vertexIndex });
      setStatus({ kind: "", text: "Distance: choose the second point" });
      return;
    }
    if (distanceStart.nodeId !== hit.nodeId || distanceStart.vertexIndex === hit.vertexIndex) {
      setStatus({ kind: "error", text: "Choose a different point in the same sketch." });
      return;
    }
    const first = profile.vertices[distanceStart.vertexIndex].uv;
    const second = profile.vertices[hit.vertexIndex].uv;
    const distance = Math.hypot(second[0] - first[0], second[1] - first[1]);
    const indices = [distanceStart.vertexIndex, hit.vertexIndex];
    distanceStart = null;
    await props.onAddConstraint(profile.line, "distance", indices, distance);
    setTool("select");
    setSelection({ nodeId: hit.nodeId, vertexIndex: hit.vertexIndex });
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
    if (tool() === "distance") {
      const hit = pickVertex(displayProfiles(), x, y, pickView());
      const next = hit ? { nodeId: hit.nodeId, vertexIndex: hit.vertexIndex } : null;
      setHover(next);
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

    const target = renderer.gizmoTarget();
    if (target) {
      const mode = renderer.gizmoModeFor(target.node);
      const axis = pickGizmoAxis(
        target.origin,
        gizmoScale(view, target.origin),
        mode,
        x,
        y,
        view,
      );
      if (axis !== renderer.gizmoAxis) {
        renderer.gizmoAxis = axis;
        renderer.setConstruction(displayProfiles(), selection(), hover());
      }
      if (axis !== null) {
        canvas.style.cursor = "grab";
        return;
      }
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

    if (tool() === "sketch" && event.button === 0) {
      void handlePlaceSketch(x, y);
      return;
    }

    if (tool() === "distance" && event.button === 0) {
      void handleDistanceConstraint(x, y);
      return;
    }

    if (
      event.button === 0 &&
      (tool() === "box" || tool() === "sphere" || tool() === "cylinder")
    ) {
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
      const mode = renderer.gizmoModeFor(target.node);
      const axis = pickGizmoAxis(target.origin, size, mode, x, y, view);
      if (axis !== null) {
        const transform = target.node.transform!;
        const ray = rayFromPixel(x, y, view);
        const start =
          mode === "rotate"
            ? angleAroundAxis(ray, target.origin, AXES[axis])
            : closestPointOnAxis(ray, target.origin, AXES[axis]);
        if (start !== null) {
          gesture = {
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
          renderer.gizmoAxis = axis;
          renderer.interacting = true;
          renderer.invalidate();
          return;
        }
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

    if (gesture.kind === "gizmo") {
      canvas.style.cursor = "grabbing";
      const view = pickView();
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
      setCameraAngles({ yaw: renderer.camera.yaw, pitch: renderer.camera.pitch });
      refreshOverlays();
      renderer.invalidate();
      return;
    }

    canvas.style.cursor = "move";
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

    if (finished.kind === "gizmo") {
      const active = gizmoDrag();
      const node = nodeById(finished.nodeId);
      setGizmoDrag(null);
      renderer.gizmoAxis = null;
      const placement = node?.transform;
      if (finished.moved && active && placement) {
        if (finished.mode === "translate") {
          await props.onSetValue(
            placement.line,
            placement.call,
            placement.positionArgument,
            active.position,
          );
        } else if (finished.mode === "rotate") {
          await props.onSetValue(
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
            await props.onSetValue(placement.line, placement.call, argument, value);
          }
        }
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

    const onKeyDown = (event: KeyboardEvent) => {
      const target = document.activeElement;
      const typing = target && (target.tagName === "TEXTAREA" || target.closest(".cm-editor"));
      if (event.key === "Escape") {
        distanceStart = null;
        setSelection(null);
        setTool("select");
      }
      if (!typing && !event.metaKey && !event.ctrlKey && !event.altKey) {
        const shortcuts: Record<string, () => void> = {
          "1": () => (setTool("select"), setSelectionMode("object")),
          "2": () => (setTool("select"), setSelectionMode("vertex")),
          p: () => setTool("polygon"),
          b: () => setTool("box"),
          s: () => setTool("sphere"),
          c: () => setTool("cylinder"),
          g: () => setGizmoMode("translate"),
          r: () => setGizmoMode("rotate"),
        };
        const action = shortcuts[event.key.toLowerCase()];
        if (action) {
          event.preventDefault();
          action();
        }
      }
      const active = selection();
      if ((event.key === "Delete" || event.key === "Backspace") && active) {
        if (typing) return;
        const node = nodeById(active.nodeId);
        if (node?.editable && node.line !== null) {
          event.preventDefault();
          if (active.vertexIndex !== null) {
            void props.onPatch("delete_vertex", node.line, active.vertexIndex);
          } else {
            void props.onDeleteObject(node.line);
          }
          setSelection(null);
        }
      }
    };
    const onPanKey = (event: KeyboardEvent) => {
      if (event.code !== "Space") return;
      const target = document.activeElement;
      if (target && (target.tagName === "TEXTAREA" || target.closest(".cm-editor"))) return;
      // Stop the page scrolling or a focused button firing.
      event.preventDefault();
      panHeld = event.type === "keydown";
      canvas.style.cursor = panHeld ? "move" : "default";
    };
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keydown", onPanKey);
    window.addEventListener("keyup", onPanKey);

    onCleanup(() => {
      observer.disconnect();
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keydown", onPanKey);
      window.removeEventListener("keyup", onPanKey);
      renderer.destroy();
    });
  });

  // Keep the GPU overlay buffers in step with the construction tree.
  createEffect(() => {
    renderer.setConstruction(displayProfiles(), selection(), hover());
  });

  createEffect(() => {
    cameraAngles();
    props.display.projection;
    refreshOverlays();
  });

  createEffect(() => {
    if (tool() !== "distance") {
      distanceStart = null;
      return;
    }
    const active = selection();
    if (!distanceStart && active?.vertexIndex != null) {
      const node = nodeById(active.nodeId);
      if (node?.kind === "profile") {
        distanceStart = { nodeId: active.nodeId, vertexIndex: active.vertexIndex };
        setStatus({ kind: "", text: "Distance: choose the second point" });
      }
    }
  });

  return (
    <section class="pane viewer-pane" aria-busy={busy()}>
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
      <Show when={props.display.showConstraints}>
        <svg
          class="constraint-overlay"
          viewBox={`0 0 ${constraintOverlay().width} ${constraintOverlay().height}`}
          aria-label="Construction constraints"
          data-testid="constraint-overlay"
        >
          <For
            each={
              props.display.showDistanceConstraints
                ? constraintOverlay().distance
                : []
            }
          >
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
                <Show when={props.display.showConstraintValues}>
                  <text x={mark.labelX} y={mark.labelY}>
                    {mark.label}
                  </text>
                </Show>
              </g>
            )}
          </For>
          <For
            each={
              props.display.showFixedConstraints ? constraintOverlay().fixed : []
            }
          >
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
      <Show when={busy()}>
        <span
          class="viewer-compile-indicator"
          role="status"
          aria-label="Compiling scene"
          data-testid="viewer-compiling"
        >
          <i />
          Compiling
        </span>
      </Show>
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
          ? "Point: click sketch edges to add vertices · Esc to finish"
          : tool() === "distance"
            ? distanceStart
              ? "Distance: choose a second point in the same sketch"
              : "Distance: choose the first sketch point"
          : tool() !== "select"
            ? `Click to place a ${tool()} · Esc to cancel`
            : "Drag handles or the gizmo · Drag to orbit · Space, Shift or right-drag to pan · Del removes"}
      </p>
    </section>
  );
}
