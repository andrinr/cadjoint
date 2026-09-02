/**
 * Application state.
 *
 * The Python source is the single source of truth: viewer edits go through
 * `/patch`, the patched text replaces the editor contents, and a recompile
 * regenerates the construction tree. Nothing about a sketch is stored outside
 * the program text except the transient drag override below.
 */

import { createSignal } from "solid-js";
import type {
  BcProposal,
  ConstructionNode,
  ConstructionRelation,
  ConstraintSolverRun,
  GizmoMode,
  MaterialDefinition,
  MeshEdgePayload,
  MeshInspectInfo,
  OptimizationPayload,
  OptimizeHistoryEntry,
  OptimizeTrajectoryEntry,
  Selection,
  SelectionMode,
  SimMeshPayload,
  SimulationMeshPayload,
  SimulationResultSummary,
  StudyPayload,
  ToolMode,
} from "./types";
import { placeEdges } from "./viewer/gizmo";
import type { PlayerState } from "./optimize";
import type { PendingLoft } from "./loft";
import {
  autoEnterSketchMode,
  cycleEditingMode,
  loadEditingMode,
  persistEditingMode,
  type EditingMode,
} from "./editingMode";
import { DEFAULT_SKETCH_PLANE, type SketchPlaneChoice } from "./sketchPlanes";

export const [source, setSource] = createSignal("");
export const [nodes, setNodes] = createSignal<ConstructionNode[]>([]);
export const [relations, setRelations] = createSignal<ConstructionRelation[]>([]);
export const [solverRuns, setSolverRuns] = createSignal<ConstraintSolverRun[]>([]);
export const [materials, setMaterials] = createSignal<MaterialDefinition[]>([]);
export const [studies, setStudies] = createSignal<StudyPayload[]>([]);
export const [simMeshes, setSimMeshes] = createSignal<SimMeshPayload[]>([]);
export const [optimizations, setOptimizations] = createSignal<OptimizationPayload[]>([]);

/**
 * The FEM surface currently displayed in the viewport, if any.
 *
 * Owned by the SimulatePanel (a solve or a mesh inspection puts it here) and
 * read by the ViewerPane for screen-space picking: the click probe, and the
 * BC region proposals. `scalars`/`range`/`fieldLabel` describe the field the
 * heatmap is showing right now (a field switch updates them without
 * re-solving); `info` carries the mesh inspection report whose grid spacing
 * sizes proposed sphere selections and side tolerances.
 */
export interface SimViewState {
  /** Render payload with the base (undeformed) vertex positions. */
  payload: SimulationMeshPayload;
  info: MeshInspectInfo | null;
  /** Scalars currently mapped through the ramp (field, or mesh quality). */
  scalars: readonly number[];
  range: [number, number];
  fieldLabel: string;
  /** Study whose BCs apply to this surface, or null for a bare mesh. */
  studyName: string | null;
}

export const [simView, setSimView] = createSignal<SimViewState | null>(null);

/**
 * The solved field of a finished study-backed optimization run.
 *
 * Set by the optimize cards (wherever they are mounted) and consumed by the
 * SimulatePanel's Results tab, so an optimization run ends showing the
 * optimized part with its field — even when the run was started from the
 * Model-mode panel and the user enters Simulate mode later.
 */
export interface OptimizeSimulateResult {
  /** The optimization's name, for the results header. */
  name: string;
  field: string;
  mesh: SimulationMeshPayload;
  result: SimulationResultSummary | null;
  meshInfo: MeshInspectInfo | null;
}

export const [optimizeSimulate, setOptimizeSimulate] =
  createSignal<OptimizeSimulateResult | null>(null);

/**
 * The last completed optimization run, shared so the trajectory player can
 * mount wherever the user lands (the Optimize card, or the Simulate
 * panel's Results tab for study-backed runs) with one state.
 */
export interface OptimizeRunState {
  name: string;
  /** The adopted program with the final literals written back. */
  source: string;
  history: OptimizeHistoryEntry[];
  trajectory: OptimizeTrajectoryEntry[];
  parameters: Record<string, number | number[]>;
  initial: Record<string, number | number[]>;
  /** The backing study's name for study-backed runs, else null. */
  study: string | null;
}

export const [optimizeRun, setOptimizeRun] = createSignal<OptimizeRunState | null>(null);

/** Replay cursor over the shared run (frame index + playing flag). */
export const [optimizePlayer, setOptimizePlayer] = createSignal<PlayerState>({
  frame: 0,
  playing: false,
});

/** One-shot post-run replay request, consumed by the mounted player. */
export const [optimizeAutoPlay, setOptimizeAutoPlay] = createSignal(false);

/** Whether viewport clicks propose BC selections (armed by the builder). */
export const [bcPickArmed, setBcPickArmed] = createSignal(false);

/** Selection proposed from the viewport, consumed by the add-BC builder. */
export const [bcProposal, setBcProposal] = createSignal<BcProposal | null>(null);

/** Click-probe readout over the simulation surface (display only). */
export interface SimProbe {
  /** CSS pixel position of the chip inside the viewer pane. */
  x: number;
  y: number;
  world: [number, number, number];
  value: number;
  label: string;
}

export const [simProbe, setSimProbe] = createSignal<SimProbe | null>(null);
export const [meshEdges, setMeshEdges] = createSignal<MeshEdgePayload | null>(null);
export const [selection, setSelection] = createSignal<Selection | null>(null);
export const [hover, setHover] = createSignal<Selection | null>(null);
export const [tool, setTool] = createSignal<ToolMode>("select");
/** Armed first sketch of a two-click loft; the viewport picks the second. */
export const [pendingLoft, setPendingLoft] = createSignal<PendingLoft | null>(null);
export const [status, setStatus] = createSignal({ kind: "", text: "Starting…" });
export const [viewerError, setViewerError] = createSignal("");
const [dismissedError, setDismissedError] = createSignal("");

/**
 * Show a viewer error unless the user already dismissed that same message.
 *
 * Every recompile re-reports a persistent problem (no WebGPU adapter, say), and
 * a banner that keeps returning would sit over the viewport blocking clicks.
 */
export function reportViewerError(message: string): void {
  if (message && message !== dismissedError()) setViewerError(message);
}

export function dismissViewerError(): void {
  setDismissedError(viewerError());
  setViewerError("");
}
export const [consoleText, setConsoleText] = createSignal("");
export const [busy, setBusy] = createSignal(false);
export const [dirty, setDirty] = createSignal(false);
/** Saved scene file name, or null for an unsaved buffer. */
export const [sceneName, setSceneName] = createSignal<string | null>(null);

/** Which shell panels are shown, toggled from the Window menu. */
export interface PanelVisibility {
  editor: boolean;
  objectTree: boolean;
  materials: boolean;
  sketch: boolean;
}

export const DEFAULT_PANELS: PanelVisibility = {
  editor: true,
  objectTree: true,
  materials: true,
  sketch: true,
};

export const [panels, setPanels] = createSignal<PanelVisibility>({ ...DEFAULT_PANELS });

/**
 * Who actually opens and closes windows.
 *
 * The dock (`windows/WindowLayout`) is the source of truth: it owns the
 * arrangement, the per-mode layouts, and the browser-local record. It
 * registers here on mount so the menu bar's coarse Window items keep working
 * unchanged, and mirrors the result back into `panels()` for their check
 * marks. Before it mounts — and in unit tests, which never build a dock —
 * this falls back to the plain signal.
 */
type PanelVisibilityHandler = (key: keyof PanelVisibility, visible: boolean) => void;

let panelVisibilityHandler: PanelVisibilityHandler | null = null;

export function setPanelVisibilityHandler(handler: PanelVisibilityHandler | null): void {
  panelVisibilityHandler = handler;
}

export function setPanelVisible(key: keyof PanelVisibility, visible: boolean): void {
  if (panelVisibilityHandler) {
    panelVisibilityHandler(key, visible);
    return;
  }
  setPanels({ ...panels(), [key]: visible });
}

/** Vertex being dragged, with its live sketch-plane position. */
export interface DragState {
  nodeId: string;
  vertexIndex: number;
  xy: [number, number];
}

/** A gizmo drag in flight, with the placement it has produced so far. */
export interface GizmoDrag {
  nodeId: string;
  mode: GizmoMode;
  axis: 0 | 1 | 2;
  position: [number, number, number];
  rotation: [number, number, number];
  dimensions: Record<string, number | number[]>;
}

/** Top-level editing mode: what family of tools the rail offers. */
const [editingModeSignal, setEditingModeSignal] = createSignal<EditingMode>(loadEditingMode());
export const editingMode = editingModeSignal;

/** Sketch profile that last auto-entered sketch mode, for cancelability. */
let lastAutoSketchId: string | null = null;

export function setEditingMode(mode: EditingMode): void {
  setEditingModeSignal(mode);
  persistEditingMode(mode);
  if (mode !== "sketch") {
    // A manual exit must not bounce back while the same sketch is selected.
    const active = selection();
    lastAutoSketchId = active ? active.nodeId : null;
  }
}

/** Keyboard cycling between the editing modes. */
export function cycleMode(step: 1 | -1 = 1): void {
  setEditingMode(cycleEditingMode(editingMode(), step));
}

/**
 * Feed a selection change into the sketch-mode auto-entry rule.
 *
 * Called from an effect in App: selecting a sketch profile enters sketch
 * mode once; Escape or the mode switcher can leave again without fighting it.
 */
export function reactToSelectionForMode(): void {
  const active = selection();
  const node = active ? nodeById(active.nodeId) : null;
  const next = autoEnterSketchMode(editingMode(), node, lastAutoSketchId);
  lastAutoSketchId = next.lastAutoNodeId;
  if (next.mode !== editingMode()) {
    setEditingModeSignal(next.mode);
    persistEditingMode(next.mode);
  }
}

/** Plane the next placed sketch lands on (quick pick or a solid's face). */
export const [sketchPlane, setSketchPlane] =
  createSignal<SketchPlaneChoice>(DEFAULT_SKETCH_PLANE);

export const [drag, setDrag] = createSignal<DragState | null>(null);
export const [gizmoDrag, setGizmoDrag] = createSignal<GizmoDrag | null>(null);
export const [gizmoMode, setGizmoMode] = createSignal<GizmoMode>("translate");
export const [selectionMode, setSelectionMode] = createSignal<SelectionMode>("object");

/** Camera angles mirrored into a signal so the ViewCube can track them. */
export const [cameraAngles, setCameraAngles] = createSignal({ yaw: 0.75, pitch: 0.32 });

/** Find a construction node by id. */
export function nodeById(id: string): ConstructionNode | undefined {
  return nodes().find((node) => node.id === id);
}

/** Sketch profiles only — the nodes that carry editable vertices. */
export function profiles(): ConstructionNode[] {
  return nodes().filter((node) => node.kind === "profile");
}

/**
 * Profiles as they should be drawn right now.
 *
 * While a drag is in flight the moved vertex is patched in locally so the
 * sketch follows the pointer; the solid only rebuilds once the drag ends and
 * the source is recompiled.
 */
export function displayProfiles(): ConstructionNode[] {
  const active = drag();
  const gizmo = gizmoDrag();
  const all = nodes();
  if (gizmo) {
    return all.map((node) =>
      node.id === gizmo.nodeId && node.transform
        ? withPlacement(node, gizmo.position, gizmo.rotation, gizmo.dimensions)
        : node,
    );
  }
  if (!active) return all;
  return all.map((profile) => {
    if (profile.id !== active.nodeId || !profile.plane) return profile;
    const { origin, u, v } = profile.plane;
    const [x, y] = active.xy;
    const world: [number, number, number] = [
      origin[0] + u[0] * x + v[0] * y,
      origin[1] + u[1] * x + v[1] * y,
      origin[2] + u[2] * x + v[2] * y,
    ];
    const vertices = profile.vertices.map((vertex, index) =>
      index === active.vertexIndex ? { ...vertex, uv: active.xy, world } : vertex,
    );
    return { ...profile, vertices };
  });
}


/**
 * A primitive re-placed at a new position/rotation, for live drag feedback.
 *
 * Its outline is recomputed on the client so the wireframe tracks the pointer;
 * the solid itself only catches up when the patched source recompiles.
 */
function withPlacement(
  node: ConstructionNode,
  position: [number, number, number],
  rotation: [number, number, number],
  dimensions: Record<string, number | number[]>,
): ConstructionNode {
  if (!node.transform) return node;
  const move = (point: [number, number, number]): [number, number, number] => [
    point[0] + (position[0] - node.transform!.position[0]),
    point[1] + (position[1] - node.transform!.position[1]),
    point[2] + (position[2] - node.transform!.position[2]),
  ];
  return {
    ...node,
    edges: placeEdges(node.edges, node.transform, position, rotation, dimensions),
    // A sketch's handles ride along with its plane.
    vertices: node.vertices.map((vertex) => ({ ...vertex, world: move(vertex.world) })),
    transform: { ...node.transform, position, rotation, dimensions },
  };
}
