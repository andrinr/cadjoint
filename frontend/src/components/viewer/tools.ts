/**
 * What a click means while a construction tool is armed.
 *
 * Each handler here owns one tool's whole click-to-source story: intersect
 * the pointer ray with something meaningful, refuse politely through the
 * status line when the click missed, then emit the edit through the pane's
 * callbacks and leave the user with the new object selected. They are grouped
 * because they share that shape and that vocabulary, and because pulling them
 * out leaves the pane holding only gesture dispatch.
 *
 * The two-click constraint flows keep their half-finished state in the pane's
 * `pendingConstraint` signal, which is handed in rather than owned here: the
 * pane also has to clear it on Escape and on a tool change.
 */

import type { Accessor, Setter } from "solid-js";
import {
  displayProfiles,
  editingMode,
  nodeById,
  nodes,
  pendingLoft,
  profiles,
  selection,
  setPendingLoft,
  setSelection,
  setSelectionMode,
  setStatus,
  setTool,
  setFaceHover,
  sketchPlane,
} from "../../state";
import {
  CONSTRAINT_TOOL_NAMES,
  edgeVertexIndices,
  type EdgeConstraintTool,
  type VertexConstraintTool,
} from "../../constraints";
import { loftPickError } from "../../loft";
import {
  quickPlaneEmission,
  type SketchPlaneEmission,
} from "../../sketchPlanes";
import { pickFace, resolveSurfaceHit, type FacePick, type FaceTarget } from "../../faces";
import { add, intersectPlane, rayFromPixel, scale } from "../../viewer/math";
import { nearestInsertIndex, pickEdge, pickNode, pickVertex } from "../../viewer/hittest";
import type { PickView } from "../../viewer/hittest";
import type { Renderer } from "../../viewer/renderer";
import type { PendingConstraint } from "./gestures";
import type { ViewerPaneProps } from "./props";

export interface ViewerToolContext {
  renderer: Renderer;
  /** The renderer's complete view descriptor, so picking matches drawing. */
  pickView: () => PickView;
  /** Sketch-plane coordinates under the pointer for a given profile. */
  planePointAt: (nodeId: string, x: number, y: number) => [number, number] | null;
  pendingConstraint: Accessor<PendingConstraint | null>;
  setPendingConstraint: Setter<PendingConstraint | null>;
  /**
   * The pane's props object, passed whole: Solid's props are getters, so
   * reading `props.onAddPrimitive` here stays as live as it is in the pane.
   */
  props: ViewerPaneProps;
}

export function createViewerTools(context: ViewerToolContext) {
  /** Drop a new primitive where the pointer meets the ground plane. */
  const handlePlacePrimitive = async (
    kind: "box" | "sphere" | "cylinder",
    x: number,
    y: number,
  ) => {
    const view = context.pickView();
    const ray = rayFromPixel(x, y, view);
    // Place on the world XY plane, falling back to a point in front of the
    // camera when the view is edge-on to it.
    const hit =
      intersectPlane(ray, [0, 0, 0], [0, 0, 1]) ??
      add(ray.origin, scale(ray.direction, Math.max(1, context.renderer.camera.distance)));
    const position: [number, number, number] = [hit[0], hit[1], hit[2]];
    const dimensions: Record<string, number | number[]> =
      kind === "box"
        ? { size: [0.5, 0.5, 0.5] }
        : kind === "sphere"
          ? { radius: 0.5 }
          : { radius: 0.4, height: 0.5 };
    await context.props.onAddPrimitive(kind, position, dimensions);
    setTool("select");
    setSelectionMode("object");
    // Select what was just placed so the gizmo is ready without a second click.
    const placed = nodes().filter((node) => node.kind === kind);
    const newest = placed[placed.length - 1];
    if (newest) setSelection({ nodeId: newest.id, vertexIndex: null });
  };

  /**
   * Place a new sketch on the chosen plane.
   *
   * Quick picks intersect the pointer ray with a world plane; "on face"
   * ray-casts the solids under the cursor and adopts the surface point and
   * its normal. A non-default normal is written with a second patch
   * (`set_value planeNormal`) once the sketch exists in the source.
   */
  const handlePlaceSketch = async (x: number, y: number) => {
    const view = context.pickView();
    const ray = rayFromPixel(x, y, view);
    const choice = sketchPlane();
    let emission: SketchPlaneEmission;
    if (choice === "face") {
      const hit = resolveSurfaceHit(nodes(), ray);
      if (!hit) {
        setStatus({ kind: "error", text: "On face: click a solid's surface." });
        return;
      }
      emission = { origin: hit.point, normal: hit.normal };
    } else {
      emission = quickPlaneEmission(choice, ray, context.renderer.camera.distance);
    }
    await context.props.onAddSketch(emission.origin);
    let sketches = profiles();
    let newest = sketches[sketches.length - 1];
    if (emission.normal && newest?.line != null) {
      await context.props.onSetValue(
        newest.line,
        "PolygonProfile",
        "planeNormal",
        emission.normal,
      );
      sketches = profiles();
      newest = sketches[sketches.length - 1];
    }
    setTool("select");
    setSelectionMode("object");
    if (newest) setSelection({ nodeId: newest.id, vertexIndex: null });
  };

  /**
   * Resolve the face under a pixel, in two steps.
   *
   * Where the surface is (`resolveSurfaceHit`) and which declared face that
   * is (`pickFace`) are asked separately, because "the pointer is over the
   * part but over no analytic face" is a real and common answer — a revolve's
   * curved wall, a blended union — and it is the answer that makes a click
   * fall back to a tangent plane instead of snapping to a face nearby.
   */
  const resolveFace = (x: number, y: number): FacePick | null => {
    const ray = rayFromPixel(x, y, context.pickView());
    const hit = resolveSurfaceHit(nodes(), ray);
    return hit ? pickFace(nodes(), hit.point, hit.normal) : null;
  };

  /**
   * Plant a sketch on the face (or the curved surface) under the pointer.
   *
   * Which sketch is the question a CAD user actually asks here, and the
   * answer is the one already open: in Sketch mode with a profile selected,
   * that profile is re-planted; otherwise a new sketch is created and planted
   * in one action. A face whose feature has no name in the source highlights
   * but refuses — the reference would have nothing to write.
   */
  const handleSketchOnFace = async (x: number, y: number) => {
    const ray = rayFromPixel(x, y, context.pickView());
    const hit = resolveSurfaceHit(nodes(), ray);
    if (!hit) {
      setStatus({ kind: "error", text: "Sketch on face: click a solid's surface." });
      return;
    }
    const pick = pickFace(nodes(), hit.point, hit.normal);
    if (pick && !pick.face.usable) {
      setStatus({
        kind: "error",
        text: "That face belongs to a feature with no name in the source; assign it to a variable first.",
      });
      return;
    }
    const target: FaceTarget = {
      faceId: pick ? pick.face.id : null,
      nodeId: pick ? pick.nodeId : hit.nodeId,
      near: hit.point,
    };
    const active = selection();
    const node = active ? nodeById(active.nodeId) : null;
    const existing =
      editingMode() === "sketch" && node?.kind === "profile" && node.line !== null
        ? node.line
        : null;
    setFaceHover(null);
    setTool("select");
    setSelectionMode("object");
    await context.props.onSketchOnFace(target, existing);
  };

  /**
   * Pick two points for a vertex-pair constraint.
   *
   * A distance constraint records the current sketch-plane distance as its
   * target; the relational kinds carry no value.
   */
  const handleVertexConstraint = async (
    kind: VertexConstraintTool,
    x: number,
    y: number,
  ) => {
    const name = CONSTRAINT_TOOL_NAMES[kind];
    const hit = pickVertex(displayProfiles(), x, y, context.pickView());
    if (!hit) {
      setStatus({ kind: "error", text: `${name}: click a sketch point.` });
      return;
    }
    const profile = nodeById(hit.nodeId);
    if (!profile?.editable || profile.line === null) {
      setStatus({ kind: "error", text: "That sketch cannot be edited from source." });
      return;
    }
    const pending = context.pendingConstraint();
    if (pending?.kind !== kind) {
      context.setPendingConstraint({
        kind,
        first: { nodeId: hit.nodeId, vertexIndex: hit.vertexIndex },
      });
      setSelection({ nodeId: hit.nodeId, vertexIndex: hit.vertexIndex });
      setStatus({ kind: "", text: `${name}: choose the second point` });
      return;
    }
    const start = pending.first as { nodeId: string; vertexIndex: number };
    if (start.nodeId !== hit.nodeId || start.vertexIndex === hit.vertexIndex) {
      setStatus({ kind: "error", text: "Choose a different point in the same sketch." });
      return;
    }
    const indices = [start.vertexIndex, hit.vertexIndex];
    let value: number | undefined;
    if (kind === "distance") {
      const first = profile.vertices[start.vertexIndex].uv;
      const second = profile.vertices[hit.vertexIndex].uv;
      value = Math.hypot(second[0] - first[0], second[1] - first[1]);
    }
    context.setPendingConstraint(null);
    await context.props.onAddConstraint(profile.line, kind, indices, value);
    setTool("select");
    setSelection({ nodeId: hit.nodeId, vertexIndex: hit.vertexIndex });
  };

  /** Pick two edges for an edge-pair constraint (parallel/perpendicular). */
  const handleEdgeConstraint = async (
    kind: EdgeConstraintTool,
    x: number,
    y: number,
  ) => {
    const name = CONSTRAINT_TOOL_NAMES[kind];
    const hit = pickEdge(displayProfiles(), x, y, context.pickView());
    if (!hit) {
      setStatus({ kind: "error", text: `${name}: click a sketch edge.` });
      return;
    }
    const profile = nodeById(hit.nodeId);
    if (!profile?.editable || profile.line === null) {
      setStatus({ kind: "error", text: "That sketch cannot be edited from source." });
      return;
    }
    const [start, end] = edgeVertexIndices(hit.insertIndex, profile.vertices.length);
    const pending = context.pendingConstraint();
    if (pending?.kind !== kind) {
      context.setPendingConstraint({ kind, first: { nodeId: hit.nodeId, start, end } });
      setSelection({ nodeId: hit.nodeId, vertexIndex: null });
      setStatus({ kind: "", text: `${name}: choose the second edge` });
      return;
    }
    const firstEdge = pending.first as { nodeId: string; start: number; end: number };
    if (firstEdge.nodeId !== hit.nodeId || firstEdge.start === start) {
      setStatus({ kind: "error", text: "Choose a different edge in the same sketch." });
      return;
    }
    const indices = [firstEdge.start, firstEdge.end, start, end];
    context.setPendingConstraint(null);
    await context.props.onAddConstraint(profile.line, kind, indices);
    setTool("select");
    setSelection({ nodeId: hit.nodeId, vertexIndex: null });
  };

  /** Complete a pending loft with the sketch under the pointer. */
  const handleLoftPick = async (x: number, y: number) => {
    const pending = pendingLoft();
    if (!pending) return;
    const hit = pickNode(displayProfiles(), x, y, context.pickView(), 10, true);
    const node = hit && nodeById(hit.nodeId);
    const error = loftPickError(pending, node);
    if (error) {
      setStatus({ kind: "error", text: error });
      return;
    }
    setPendingLoft(null);
    setSelection({ nodeId: node!.id, vertexIndex: null });
    await context.props.onAddLoft(pending.line, node!.line!);
  };

  /** Insert one vertex where the user clicked; the tool stays active. */
  const handleAddVertex = async (x: number, y: number) => {
    const view = context.pickView();
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
    const xy = context.planePointAt(target.id, x, y);
    if (!xy || target.line === null) {
      setStatus({ kind: "error", text: "Click nearer the sketch plane to place a vertex." });
      return;
    }
    const index =
      edge && edge.nodeId === target.id
        ? edge.insertIndex
        : nearestInsertIndex(target, x, y, view);
    await context.props.onPatch("insert_vertex", target.line, index, xy);
    setSelection({ nodeId: target.id, vertexIndex: Math.min(index, target.vertices.length) });
  };

  return {
    handlePlacePrimitive,
    handlePlaceSketch,
    resolveFace,
    handleSketchOnFace,
    handleVertexConstraint,
    handleEdgeConstraint,
    handleLoftPick,
    handleAddVertex,
  };
}
