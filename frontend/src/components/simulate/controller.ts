/**
 * Everything the Simulate panel knows, with none of what it looks like.
 *
 * The panel is one long-lived piece of state — which tab is open, what was
 * last solved or inspected, which field/deformation/slice the viewport shows,
 * which study's BC builder is open and what it is proposing — plus the
 * request round-trips that change it (solve, inspect, patch). Holding that
 * here lets each tab be a plain view over an accessor bag, and keeps the one
 * place that talks to the renderer and the shared simView/bcProposal signals
 * from being spread across four files.
 *
 * `createSimulateController` must be called from the panel's component body:
 * it opens signals, effects and lifecycle hooks that need the panel's
 * reactive owner, and it is the panel's `onCleanup` that hands the viewport
 * back to the raymarched scene when Simulate mode is left.
 */

import { createEffect, createSignal, onCleanup, onMount } from "solid-js";
import * as api from "../../api";
import {
  DEFAULT_SLICE,
  applyDisplacements,
  autoDeformScale,
  meshBounds,
  resolveResultView,
  type SliceState,
} from "../../simulation";
import {
  bcProposal,
  nodes,
  optimizeSimulate,
  setBcPickArmed,
  setBcProposal,
  setOptimizeSimulate,
  setSimProbe,
  setSimView,
  simMeshes,
  simView,
  source,
  studies,
} from "../../state";
import {
  addBcRequest,
  defaultDraft,
  draftSelection,
  type BcDraft,
} from "../../studies";
import { qualityHistogram } from "../../meshes";
import {
  BC_TYPE_COLORS,
  PROPOSAL_COLOR,
  evaluateSelection,
  overlayColors,
  selectionEvaluable,
  type OverlayLayer,
} from "../../selectionEval";
import type {
  MeshInspectInfo,
  SimMeshPayload,
  SimulationMeshPayload,
  SimulationResultSummary,
  StudyPayload,
} from "../../types";
import type { SimulatePanelProps } from "../SimulatePanel";

/** The panel's tab bar: setup on the left, outcomes on the right. */
export type PanelTab = "meshes" | "studies" | "optimize" | "results";

/** Which of the two things the viewport is showing right now. */
export type ViewportMode = "scene" | "mesh";

const HISTOGRAM_BINS = 24;

/** Everything a completed solve leaves behind for results browsing. */
export interface SolveState {
  name: string;
  payload: SimulationMeshPayload;
  result: SimulationResultSummary | null;
  meshInfo: MeshInspectInfo | null;
  /** Per-vertex fields for display switching (may only hold the default). */
  fields: Record<string, number[]>;
  ranges: Record<string, [number, number]>;
  /** The field the response mesh scalars carry. */
  defaultField: string;
  displacements: [number, number, number][] | null;
  /** Lazily fetched per-vertex quality for the "mesh quality" view. */
  qualityScalars: number[] | null;
}

/** A standalone mesh inspection (no solve). */
export interface InspectState {
  name: string;
  payload: SimulationMeshPayload;
  info: MeshInspectInfo;
  qualityScalars: number[];
}

/** The accessor bag every Simulate tab is rendered from. */
export type SimulateController = ReturnType<typeof createSimulateController>;

export function createSimulateController(props: SimulatePanelProps) {
  const [tab, setTab] = createSignal<PanelTab>("studies");
  /** Which study's add-BC builder is open, by study index. */
  const [building, setBuilding] = createSignal<number | null>(null);
  const [draft, setDraft] = createSignal<BcDraft>(defaultDraft("thermal"));
  const [solving, setSolving] = createSignal<string | null>(null);
  const [inspecting, setInspecting] = createSignal<string | null>(null);
  const [error, setError] = createSignal("");
  const [unavailable, setUnavailable] = createSignal(false);
  const [result, setResult] = createSignal<SolveState | null>(null);
  const [inspected, setInspected] = createSignal<InspectState | null>(null);
  const [slice, setSlice] = createSignal<SliceState>({ ...DEFAULT_SLICE });

  // Results browsing state, layered over the solved payload.
  const [activeField, setActiveField] = createSignal<string | null>(null);
  const [qualityView, setQualityView] = createSignal(false);
  const [deformed, setDeformed] = createSignal(false);
  /** Multiplier over the auto warp scale (1 = 10% of the mesh diagonal). */
  const [deformFactor, setDeformFactor] = createSignal(1);

  // BC visualization: per-study "show all" toggle and the hovered row.
  const [showBcs, setShowBcs] = createSignal<number | null>(null);
  const [hoveredBc, setHoveredBc] = createSignal<{ study: number; bc: number } | null>(
    null,
  );
  const [picking, setPicking] = createSignal(false);

  // Viewport control: a loaded mesh/result never hijacks the viewport for
  // good — "Scene" returns to the raymarched SDF while keeping the loaded
  // state, so switching back is instant. ("Both" is deliberately absent:
  // the mesh sits inside the opaque raymarched solid, so depth-correct
  // compositing would hide it, and the raymarcher has no opacity path.)
  const [viewportMode, setViewportMode] = createSignal<ViewportMode>("mesh");
  // Element-edge hairlines: ON for generated-mesh views (a mesh you are
  // inspecting should look like a mesh), OFF for solved-field views.
  const [showEdges, setShowEdges] = createSignal(true);

  createEffect(() => {
    props.renderer.simulationEdgesVisible = showEdges();
  });

  const setViewport = (mode: ViewportMode) => {
    setViewportMode(mode);
    props.renderer.simulationActive = mode === "mesh";
  };

  /** Whether the displayed payload ships element boundary edges. */
  const hasEdges = (): boolean => {
    const payload = inspected()?.payload ?? result()?.payload;
    return (payload?.edges?.length ?? 0) >= 2;
  };

  const applySlice = (patch: Partial<SliceState>) => {
    const next = { ...slice(), ...patch };
    setSlice(next);
    props.renderer.setSimulationClip(next);
  };

  // Entering simulate mode mounts the panel; hand the viewport to the mesh
  // view when something is displayed, and give it back to the raymarched
  // scene when the mode is left.
  onMount(() => {
    if (simView()) setViewport(viewportMode());
  });
  onCleanup(() => {
    props.renderer.simulationActive = false;
    props.renderer.setSimulationOverlay(null);
    setBcPickArmed(false);
    setSimProbe(null);
  });

  /** The scalars/range/label the current view settings select. */
  const activeScalars = (): {
    scalars: readonly number[];
    range: [number, number];
    label: string;
  } | null => {
    const inspect = inspected();
    if (inspect) {
      const quality = inspect.info.quality.scaled_jacobian;
      return {
        scalars: inspect.qualityScalars,
        range: quality ? [quality.min, quality.max] : inspect.payload.range,
        label: "scaled jacobian",
      };
    }
    const solved = result();
    if (!solved) return null;
    const quality = solved.meshInfo?.quality.scaled_jacobian;
    return resolveResultView({
      defaultField: solved.defaultField,
      activeField: activeField(),
      qualityView: qualityView(),
      fields: solved.fields,
      ranges: solved.ranges,
      payloadScalars: solved.payload.scalars,
      payloadRange: solved.payload.range,
      qualityScalars: solved.qualityScalars,
      qualityRange: quality ? [quality.min, quality.max] : null,
    });
  };

  /** Push the current field/quality/deformed view into the renderer. */
  const applyView = () => {
    const solved = result();
    const inspect = inspected();
    const payload = inspect?.payload ?? solved?.payload;
    const active = activeScalars();
    if (!payload || !active) return;
    props.renderer.setSimulationScalars(
      active.scalars === payload.scalars ? null : (active.scalars as number[]),
      active.range,
    );
    if (solved && !inspect && deformed() && solved.displacements) {
      const auto = autoDeformScale(meshBounds(payload.positions), solved.displacements);
      props.renderer.setSimulationPositions(
        applyDisplacements(payload.positions, solved.displacements, auto * deformFactor()),
      );
    } else {
      props.renderer.setSimulationPositions(null);
    }
    setSimView({
      payload,
      info: inspect?.info ?? solved?.meshInfo ?? null,
      scalars: active.scalars,
      range: active.range,
      fieldLabel: active.label,
      studyName: inspect ? null : (solved?.name ?? null),
    });
  };

  const solve = async (study: StudyPayload) => {
    setSolving(study.name);
    setError("");
    try {
      const response = await api.simulateStudy({
        source: source(),
        kind: "study",
        name: study.name,
      });
      if (response.error_kind === "fem_unavailable") {
        setUnavailable(true);
        setError(response.error ?? "The jax-fem extra is not installed.");
        return;
      }
      if (!response.ok || !response.mesh) {
        setError(response.error ?? "The solve failed.");
        return;
      }
      const defaultField = response.field ?? "field";
      setInspected(null);
      // The field catalog rides on the mesh payload (top level is the
      // legacy home; coalesce both while the backend contract settles).
      setResult({
        name: study.name,
        payload: response.mesh,
        result: response.result ?? null,
        meshInfo: response.mesh_info ?? null,
        fields:
          response.mesh.fields ??
          response.fields ?? { [defaultField]: response.mesh.scalars },
        ranges:
          response.mesh.ranges ??
          response.ranges ?? { [defaultField]: response.mesh.range },
        defaultField,
        displacements: response.mesh.displacements ?? response.displacements ?? null,
        qualityScalars: null,
      });
      setActiveField(defaultField);
      setQualityView(false);
      setDeformed(false);
      setDeformFactor(1);
      // Solved fields read best clean; edges stay one toggle away.
      setShowEdges(false);
      props.renderer.setSimulationMesh(response.mesh);
      setViewport("mesh");
      props.renderer.setSimulationClip(slice());
      setSimProbe(null);
      applyView();
      setTab("results");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setSolving(null);
    }
  };

  // A finished study-backed optimization publishes the optimized design's
  // solved field; adopt it as a result so the run ends on the Results tab
  // showing the end-to-end outcome (optimization + simulation together).
  createEffect(() => {
    const finished = optimizeSimulate();
    if (!finished) return;
    setOptimizeSimulate(null);
    setInspected(null);
    setResult({
      name: finished.name,
      payload: finished.mesh,
      result: finished.result,
      meshInfo: finished.meshInfo,
      fields: finished.mesh.fields ?? { [finished.field]: finished.mesh.scalars },
      ranges: finished.mesh.ranges ?? { [finished.field]: finished.mesh.range },
      defaultField: finished.field,
      displacements: finished.mesh.displacements ?? null,
      qualityScalars: null,
    });
    setActiveField(finished.field);
    setQualityView(false);
    setDeformed(false);
    setShowEdges(false);
    props.renderer.setSimulationMesh(finished.mesh);
    setViewport("mesh");
    props.renderer.setSimulationClip(slice());
    setSimProbe(null);
    applyView();
    setTab("results");
  });

  const inspect = async (mesh: SimMeshPayload | { name: string }) => {
    setInspecting(mesh.name);
    setError("");
    try {
      const response = await api.meshInspect(source(), mesh.name);
      if (!response.ok || !response.mesh || !response.info) {
        setError(response.error ?? "Mesh inspection failed.");
        return;
      }
      setResult(null);
      setInspected({
        name: response.name ?? mesh.name,
        payload: response.mesh,
        info: response.info,
        qualityScalars: response.quality_scalars ?? response.mesh.scalars,
      });
      // A mesh you are inspecting should look like a mesh: edges on.
      setShowEdges(true);
      props.renderer.setSimulationMesh(response.mesh);
      setViewport("mesh");
      props.renderer.setSimulationClip(slice());
      setSimProbe(null);
      applyView();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setInspecting(null);
    }
  };

  /** The quality view over a solved mesh needs per-vertex quality once. */
  const toggleQualityView = async () => {
    const solved = result();
    if (!solved) return;
    if (qualityView()) {
      setQualityView(false);
      applyView();
      return;
    }
    if (!solved.qualityScalars) {
      // The result's mesh name may be a synthesized implicit-mesh label
      // ("study::mesh"); only a declared SimMesh resolves by that name, so
      // fall back to the study name, which /api/mesh_inspect also accepts.
      const declared = simMeshes().some((mesh) => mesh.name === solved.result?.mesh);
      const target = declared ? solved.result!.mesh! : solved.name;
      setInspecting(target);
      try {
        const response = await api.meshInspect(source(), target);
        if (!response.ok || !response.quality_scalars) {
          setError(response.error ?? "Mesh quality is unavailable for this result.");
          return;
        }
        setResult({ ...solved, qualityScalars: response.quality_scalars });
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : String(caught));
        return;
      } finally {
        setInspecting(null);
      }
    }
    setQualityView(true);
    applyView();
  };

  /** Route one patch body through the app queue, surfacing failures here. */
  const patch = async (body: Record<string, unknown>) => {
    setError("");
    await props.onPatch(body);
  };

  const openBuilder = (study: StudyPayload) => {
    setDraft(defaultDraft(study.kind));
    setBuilding(study.index);
  };

  const closeBuilder = () => {
    setBuilding(null);
    setPicking(false);
  };

  const submitBc = async (study: StudyPayload) => {
    await patch(addBcRequest(study, draft()));
    closeBuilder();
  };

  // Viewport BC picking is armed exactly while the builder's pick toggle is
  // on; the ViewerPane reads the shared signal to reroute clicks and drags.
  createEffect(() => {
    setBcPickArmed(building() !== null && picking());
  });

  // A proposal picked in the viewport pre-fills the builder (opening it on
  // the displayed study if needed); the user still picks type/value and
  // confirms — nothing is committed from the viewport directly.
  createEffect(() => {
    const proposal = bcProposal();
    if (!proposal) return;
    setBcProposal(null);
    let target = studies().find((study) => study.index === building());
    if (!target) {
      const view = simView();
      target =
        studies().find((study) => study.name === view?.studyName) ?? studies()[0];
      if (!target || !target.editable) return;
      openBuilder(target);
    }
    if (proposal.kind === "sphere") {
      setDraft({
        ...draft(),
        selectionKind: "sphere",
        center: [...proposal.center],
        radius: proposal.radius,
      });
    } else {
      setDraft({
        ...draft(),
        selectionKind: "box",
        minCorner: [...proposal.min],
        maxCorner: [...proposal.max],
      });
    }
  });

  // BC visualization: recompute the per-vertex overlay whenever the shown
  // mesh, the toggles, the hovered row, or the builder draft change.
  createEffect(() => {
    const view = simView();
    if (!view) {
      props.renderer.setSimulationOverlay(null);
      return;
    }
    const positions = view.payload.positions;
    const grid = view.info?.grid ?? null;
    const count = view.payload.vertex_count;
    const layers: OverlayLayer[] = [];
    const pushBc = (study: StudyPayload, bcIndex: number) => {
      const bc = study.bcs[bcIndex];
      if (!bc || !selectionEvaluable(bc.nodes)) return;
      const mask = evaluateSelection(bc.nodes, positions, grid);
      if (mask) layers.push({ mask, color: BC_TYPE_COLORS[bc.type] });
    };
    const shown = studies().find((study) => study.index === showBcs());
    if (shown) {
      for (let index = 0; index < shown.bcs.length; index++) pushBc(shown, index);
    }
    const hovered = hoveredBc();
    if (hovered) {
      const study = studies().find((item) => item.index === hovered.study);
      if (study) pushBc(study, hovered.bc);
    }
    if (building() !== null) {
      const mask = evaluateSelection(draftSelection(draft()), positions, grid);
      if (mask) layers.push({ mask, color: PROPOSAL_COLOR });
    }
    props.renderer.setSimulationOverlay(
      layers.length > 0 ? overlayColors(count, layers) : null,
    );
  });

  /** Write one component of a vector-valued field in the BC draft. */
  const setVector = (
    key: "minCorner" | "maxCorner" | "center" | "point" | "normal" | "vector",
    component: number,
    value: number,
  ) => {
    const next = { ...draft() };
    const vector = [...next[key]] as [number, number, number];
    vector[component] = value;
    next[key] = vector;
    setDraft(next);
  };

  /** Named scene objects, for the study/mesh domain pickers. */
  const namedObjects = () =>
    nodes()
      .map((node) => node.name)
      .filter((name): name is string => name !== null);

  const histogram = () => {
    const inspect = inspected();
    if (!inspect) return null;
    return qualityHistogram(inspect.qualityScalars, HISTOGRAM_BINS);
  };

  return {
    // The panel's own props, so tabs can reach the app-shell callbacks.
    props,
    // State.
    tab,
    setTab,
    building,
    draft,
    setDraft,
    solving,
    inspecting,
    error,
    unavailable,
    result,
    inspected,
    slice,
    activeField,
    setActiveField,
    qualityView,
    setQualityView,
    deformed,
    setDeformed,
    deformFactor,
    setDeformFactor,
    showBcs,
    setShowBcs,
    hoveredBc,
    setHoveredBc,
    picking,
    setPicking,
    viewportMode,
    showEdges,
    setShowEdges,
    // Derived.
    activeScalars,
    hasEdges,
    histogram,
    namedObjects,
    // Actions.
    applySlice,
    applyView,
    closeBuilder,
    inspect,
    openBuilder,
    patch,
    setVector,
    setViewport,
    solve,
    submitBc,
    toggleQualityView,
  };
}
