/**
 * FEM simulation panel — a patch layer over meshes and studies in the code.
 *
 * The compile payload lists every SimMesh and ThermalStudy/ElasticStudy in
 * the scene program; this panel renders them and edits them exclusively
 * through /patch source operations (add_mesh, set_mesh_value, add_study_bc,
 * …), the same round-trip the sketch tools use. Solving posts the study's
 * *name*; the server re-derives everything from the declaration and returns
 * the nodal fields, the result summary, and the mesh inspection report.
 *
 * On top of the solve view the panel drives the deeper inspection tools:
 * mesh quality heatmaps (via /api/mesh_inspect) with a histogram, a field
 * picker that swaps the displayed scalars without re-solving, a deformed
 * view for elastic results, and per-vertex BC previews evaluated client-side
 * against the render payload (src/selectionEval.ts). Viewport interactions
 * (probe clicks, BC region proposals) meet the panel through the shared
 * simView/bcProposal signals in state.ts.
 */

import { For, Index, Show, createEffect, createSignal, onCleanup, onMount } from "solid-js";
import * as api from "../api";
import {
  DEFAULT_SLICE,
  applyDisplacements,
  autoDeformScale,
  formatScalar,
  meshBounds,
  rampCss,
  resolveResultView,
  type SliceState,
} from "../simulation";
import {
  bcProposal,
  nodes,
  optimizeRun,
  optimizeSimulate,
  setBcPickArmed,
  setBcProposal,
  setOptimizeSimulate,
  setSimProbe,
  setSimView,
  simMeshes,
  simProbe,
  simView,
  source,
  studies,
} from "../state";
import { OptimizeCards } from "./OptimizeCards";
import { TrajectoryPlayer } from "./TrajectoryPlayer";
import {
  BC_LABELS,
  addBcRequest,
  addStudyRequest,
  bcTypesFor,
  bcValue,
  defaultDraft,
  deleteBcRequest,
  deleteStudyRequest,
  describeSelection,
  draftSelection,
  setArgumentRequest,
  setBcValueRequest,
  studyArguments,
  type BcDraft,
  type BuilderSelectionKind,
} from "../studies";
import {
  addMeshRequest,
  deleteMeshRequest,
  meshArguments,
  qualityHistogram,
  setMeshValueRequest,
  setStudyDomainRequest,
  setStudyMeshRequest,
} from "../meshes";
import {
  BC_TYPE_COLORS,
  PROPOSAL_COLOR,
  evaluateSelection,
  overlayColors,
  selectionEvaluable,
  type OverlayLayer,
} from "../selectionEval";
import type {
  MeshInspectInfo,
  QualitySummary,
  SimMeshPayload,
  SimulationMeshPayload,
  SimulationResultSummary,
  StudyPayload,
} from "../types";
import type { Renderer } from "../viewer/renderer";

export interface SimulatePanelProps {
  renderer: Renderer;
  /** Serialized /patch queue owned by the app shell. */
  onPatch: (body: Record<string, unknown>) => Promise<void>;
  /** Adopt server-produced source like a patch response (optimize runs). */
  onAdoptSource: (source: string) => Promise<void>;
  /** Compile-and-render a transient program without committing it. */
  onGhostCompile: (source: string) => Promise<boolean>;
}

/** The panel's tab bar: setup on the left, outcomes on the right. */
type PanelTab = "meshes" | "studies" | "optimize" | "results";

const AXIS_LABELS = ["X", "Y", "Z"] as const;
const SIDES = ["+x", "-x", "+y", "-y", "+z", "-z"] as const;
const SELECTION_KINDS: { value: BuilderSelectionKind; label: string }[] = [
  { value: "side", label: "Side" },
  { value: "box", label: "Box" },
  { value: "sphere", label: "Sphere" },
  { value: "halfspace", label: "Half-space" },
];

const HISTOGRAM_BINS = 24;
const HISTOGRAM_WIDTH = 216;
const HISTOGRAM_HEIGHT = 42;

/** Numeric input helper: commit only finite values. */
const parse = (raw: string): number | null => {
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
};

/** Everything a completed solve leaves behind for results browsing. */
interface SolveState {
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
interface InspectState {
  name: string;
  payload: SimulationMeshPayload;
  info: MeshInspectInfo;
  qualityScalars: number[];
}

export function SimulatePanel(props: SimulatePanelProps) {
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
  const [hoveredBc, setHoveredBc] = createSignal<{ study: number; bc: number } | null>(null);
  const [picking, setPicking] = createSignal(false);

  // Viewport control: a loaded mesh/result never hijacks the viewport for
  // good — "Scene" returns to the raymarched SDF while keeping the loaded
  // state, so switching back is instant. ("Both" is deliberately absent:
  // the mesh sits inside the opaque raymarched solid, so depth-correct
  // compositing would hide it, and the raymarcher has no opacity path.)
  const [viewportMode, setViewportMode] = createSignal<"scene" | "mesh">("mesh");
  // Element-edge hairlines: ON for generated-mesh views (a mesh you are
  // inspecting should look like a mesh), OFF for solved-field views.
  const [showEdges, setShowEdges] = createSignal(true);

  createEffect(() => {
    props.renderer.simulationEdgesVisible = showEdges();
  });

  const setViewport = (mode: "scene" | "mesh") => {
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

  const setVector = (
    key: "minCorner" | "maxCorner" | "center" | "point" | "normal" | "vector",
    component: number,
    raw: string,
  ) => {
    const value = parse(raw);
    if (value === null) return;
    const next = { ...draft() };
    const vector = [...next[key]] as [number, number, number];
    vector[component] = value;
    next[key] = vector;
    setDraft(next);
  };

  const vectorRow = (
    key: "minCorner" | "maxCorner" | "center" | "point" | "normal",
    label: string,
  ) => (
    <label class="sim-builder-vector">
      <span>{label}</span>
      <Index each={[0, 1, 2]}>
        {(component) => (
          <input
            type="number"
            step="0.1"
            value={draft()[key][component()]}
            onChange={(event) => setVector(key, component(), event.currentTarget.value)}
            title={`${label} ${AXIS_LABELS[component()]}`}
          />
        )}
      </Index>
    </label>
  );

  /** Editable triplet row for a mesh argument (resolution, bounds, size). */
  const meshVectorRow = (
    mesh: SimMeshPayload,
    key: "resolution" | "bounds" | "size",
    value: number[],
  ) => (
    <label class="sim-builder-vector">
      <span>{key}</span>
      <Index each={[0, 1, 2]}>
        {(component) => (
          <input
            type="number"
            step={key === "resolution" ? "1" : "0.1"}
            value={value[component()]}
            disabled={solving() !== null}
            onChange={(event) => {
              const next = parse(event.currentTarget.value);
              if (next === null) return;
              const triplet = [...value];
              triplet[component()] = key === "resolution" ? Math.round(next) : next;
              void patch(setMeshValueRequest(mesh, key, triplet));
            }}
            title={`${key} ${AXIS_LABELS[component()]}`}
            data-testid={`mesh-arg-${mesh.name}-${key}-${component()}`}
          />
        )}
      </Index>
    </label>
  );

  const quality = (summary: QualitySummary | undefined) =>
    summary
      ? `${formatScalar(summary.min)} / ${formatScalar(summary.mean)} / ${formatScalar(summary.max)}`
      : "–";

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

  const tabButton = (key: PanelTab, label: string) => (
    <button
      type="button"
      role="tab"
      aria-selected={tab() === key}
      classList={{ active: tab() === key }}
      onClick={() => setTab(key)}
      data-testid={`sim-tab-${key}`}
    >
      {label}
      <Show when={key === "results" && result() !== null}>
        <i class="sim-tab-badge" />
      </Show>
    </button>
  );

  return (
    <aside class="sim-panel" data-testid="simulate-panel">
      <header>
        <span>
          <small>FEM</small>
          Simulate
        </span>
      </header>

      <div class="sim-tabs" role="tablist" data-testid="sim-tabs">
        {tabButton("meshes", "Meshes")}
        {tabButton("studies", "Studies")}
        {tabButton("optimize", "Optimize")}
        {tabButton("results", "Results")}
      </div>

      {/* ── Meshes tab: declare, generate, and judge discretizations ────── */}
      <Show when={tab() === "meshes"}>
      <div class="sim-section-head" data-testid="mesh-panel">
        <b>Meshes</b>
        <button
          type="button"
          class="sim-add-inline"
          onClick={() => void patch(addMeshRequest())}
          title="Declare a SimMesh in the code"
          data-testid="mesh-add"
        >
          + Mesh
        </button>
      </div>
      <Show
        when={simMeshes().length > 0}
        fallback={
          <div class="sim-help" data-testid="mesh-empty">
            <p>
              No meshes declared — each study builds its own default mesh from
              its resolution and bounds when you solve. Declare a SimMesh(...)
              in the program to name it, share it between studies, choose
              hex/tet10, and inspect its quality here.
            </p>
            <For each={studies()}>
              {(study) => (
                <button
                  type="button"
                  class="sim-add-inline"
                  disabled={inspecting() !== null || solving() !== null}
                  onClick={() => void inspect({ name: study.name })}
                  title={`Build and display the mesh ${study.name} would solve on`}
                  data-testid={`mesh-generate-study-${study.name}`}
                >
                  {inspecting() === study.name
                    ? "Generating…"
                    : `Generate ${study.name}'s mesh`}
                </button>
              )}
            </For>
            <Show when={inspected()}>
              {(current) => (
                <div class="sim-inspect" data-testid="mesh-stats">
                  <div class="sim-stats">
                    <span>
                      nodes <b>{current().info.nodes}</b>
                    </span>
                    <span>
                      elements <b>{current().info.elements}</b>
                    </span>
                    <Show when={current().info.method}>
                      <span>
                        method <b>{current().info.method}</b>
                      </span>
                    </Show>
                  </div>
                  <div class="sim-stats">
                    <span>
                      jacobian <b>{quality(current().info.quality.scaled_jacobian)}</b>
                    </span>
                  </div>
                  <div class="sim-stats">
                    <span>
                      aspect <b>{quality(current().info.quality.aspect_ratio)}</b>
                    </span>
                  </div>
                </div>
              )}
            </Show>
          </div>
        }
      >
        <ul class="sim-studies" data-testid="mesh-list">
          <For each={simMeshes()}>
            {(mesh) => (
              <li class="sim-study" data-testid={`mesh-${mesh.name}`}>
                <div class="sim-study-head">
                  <span class="sim-kind sim-kind-mesh">mesh</span>
                  <strong>{mesh.name}</strong>
                  <button
                    type="button"
                    class="sim-delete"
                    onClick={() => void patch(deleteMeshRequest(mesh))}
                    title="Delete this mesh from the code"
                    aria-label={`Delete mesh ${mesh.name}`}
                    data-testid={`mesh-delete-${mesh.name}`}
                  >
                    ×
                  </button>
                </div>

                <Show
                  when={mesh.editable}
                  fallback={
                    <p class="sim-note">Defined dynamically in code — edit it there.</p>
                  }
                >
                  <For each={meshArguments(mesh)}>
                    {(argument) =>
                      Array.isArray(argument.value) ? (
                        meshVectorRow(
                          mesh,
                          argument.key as "resolution" | "bounds" | "size",
                          argument.value,
                        )
                      ) : (
                        <label class="sim-builder-vector">
                          <span>{argument.key}</span>
                          <input
                            type="number"
                            step={argument.key === "resolution" ? "1" : "0.05"}
                            value={argument.value as number}
                            disabled={solving() !== null}
                            onChange={(event) => {
                              const value = parse(event.currentTarget.value);
                              if (value !== null) {
                                void patch(
                                  setMeshValueRequest(
                                    mesh,
                                    argument.key,
                                    argument.key === "resolution" ? Math.round(value) : value,
                                  ),
                                );
                              }
                            }}
                            data-testid={`mesh-arg-${mesh.name}-${argument.key}`}
                          />
                        </label>
                      )
                    }
                  </For>
                  <Show when={mesh.domain}>
                    {(domain) => (
                      <label class="sim-builder-vector">
                        <span>domain</span>
                        <select
                          value={domain().name ?? ""}
                          disabled={solving() !== null}
                          onChange={(event) => {
                            const value = event.currentTarget.value;
                            if (value) void patch(setMeshValueRequest(mesh, "domain", value));
                          }}
                          data-testid={`mesh-domain-${mesh.name}`}
                        >
                          <Show when={domain().name === null}>
                            <option value="">{`(${domain().type})`}</option>
                          </Show>
                          <For each={namedObjects()}>
                            {(name) => <option value={name}>{name}</option>}
                          </For>
                        </select>
                      </label>
                    )}
                  </Show>

                  {/* Element type: hex is the lattice-native default; tet10
                      resolves curved boundaries better at a solve cost; tet4
                      exists but measures stiff (locking) — prefer tet10. */}
                  <div class="segmented sim-method" data-testid={`mesh-method-${mesh.name}`}>
                    <For each={["hex", "tet4", "tet10"] as const}>
                      {(method) => (
                        <button
                          type="button"
                          classList={{ active: (mesh.method ?? "hex") === method }}
                          disabled={solving() !== null}
                          onClick={() => void patch(setMeshValueRequest(mesh, "method", method))}
                          title={
                            method === "hex"
                              ? "Hexahedra: fast, lattice-aligned"
                              : method === "tet4"
                                ? "Linear tets: stiff — prefers Tet10"
                                : "Quadratic tets: accurate boundary, slower"
                          }
                          data-testid={`mesh-method-${mesh.name}-${method}`}
                        >
                          {method === "hex" ? "Hex" : method === "tet4" ? "Tet4" : "Tet10"}
                        </button>
                      )}
                    </For>
                  </div>
                  <p class="sim-note sim-method-hint">
                    Hex: fast, lattice-aligned · Tet10: accurate boundary, slower
                  </p>
                </Show>

                <button
                  type="button"
                  class="sim-run"
                  disabled={inspecting() !== null || solving() !== null}
                  onClick={() => void inspect(mesh)}
                  title="Build this mesh and show it with its quality heatmap"
                  data-testid={`mesh-inspect-${mesh.name}`}
                >
                  {inspecting() === mesh.name
                    ? "Generating…"
                    : inspected()?.name === mesh.name
                      ? "Regenerate"
                      : "Generate mesh"}
                </button>

                <Show when={inspected()?.name === mesh.name}>
                  {(_) => {
                    const info = () => inspected()!.info;
                    return (
                      <div class="sim-inspect" data-testid="mesh-stats">
                        <div class="sim-stats">
                          <span>
                            nodes <b>{info().nodes}</b>
                          </span>
                          <span>
                            elements <b>{info().elements}</b>
                          </span>
                          <Show when={info().method}>
                            <span>
                              method <b>{info().method}</b>
                            </span>
                          </Show>
                        </div>
                        <div class="sim-stats">
                          <span>
                            jacobian <b>{quality(info().quality.scaled_jacobian)}</b>
                          </span>
                        </div>
                        <div class="sim-stats">
                          <span>
                            aspect <b>{quality(info().quality.aspect_ratio)}</b>
                          </span>
                        </div>
                        <Show when={histogram()}>
                          {(bins) => (
                            <svg
                              class="sim-histogram"
                              viewBox={`0 0 ${HISTOGRAM_WIDTH} ${HISTOGRAM_HEIGHT}`}
                              preserveAspectRatio="none"
                              role="img"
                              aria-label="Element quality histogram"
                              data-testid="mesh-histogram"
                            >
                              <For each={bins().counts}>
                                {(count, index) => {
                                  const width = HISTOGRAM_WIDTH / bins().counts.length;
                                  const height =
                                    bins().peak > 0
                                      ? (count / bins().peak) * (HISTOGRAM_HEIGHT - 2)
                                      : 0;
                                  return (
                                    <rect
                                      x={index() * width + 0.5}
                                      y={HISTOGRAM_HEIGHT - height}
                                      width={Math.max(width - 1, 0.5)}
                                      height={height}
                                    />
                                  );
                                }}
                              </For>
                            </svg>
                          )}
                        </Show>
                        <Show when={histogram()}>
                          {(bins) => (
                            <div class="sim-legend-values">
                              <span>{formatScalar(bins().min)}</span>
                              <span>{formatScalar(bins().max)}</span>
                            </div>
                          )}
                        </Show>
                      </div>
                    );
                  }}
                </Show>
              </li>
            )}
          </For>
        </ul>
      </Show>

      </Show>

      {/* ── Studies tab: declarations, BCs, and the solve action ────────── */}
      <Show when={tab() === "studies"}>
      <Show
        when={studies().length > 0}
        fallback={
          <p class="sim-help" data-testid="simulate-empty">
            No studies declared. Add one — it becomes a ThermalStudy or
            ElasticStudy call in the code, and stays editable from either side.
          </p>
        }
      >
        <ul class="sim-studies" data-testid="simulate-studies">
          <For each={studies()}>
            {(study) => (
              <li class="sim-study" data-testid={`simulate-study-${study.name}`}>
                <div class="sim-study-head">
                  <span class={`sim-kind sim-kind-${study.kind}`}>{study.kind}</span>
                  <strong>{study.name}</strong>
                  <button
                    type="button"
                    class="sim-delete"
                    onClick={() => void patch(deleteStudyRequest(study))}
                    title="Delete this study from the code"
                    aria-label={`Delete study ${study.name}`}
                    data-testid={`simulate-delete-${study.name}`}
                  >
                    ×
                  </button>
                </div>

                <Show
                  when={study.editable}
                  fallback={
                    <p class="sim-note">
                      Defined dynamically in code — edit it there.
                    </p>
                  }
                >
                  <div class="sim-args">
                    <For each={studyArguments(study)}>
                      {(argument) => (
                        <label>
                          <span>{argument.key}</span>
                          <input
                            type="number"
                            value={argument.value}
                            disabled={solving() !== null}
                            onChange={(event) => {
                              const value = parse(event.currentTarget.value);
                              if (value !== null) {
                                void patch(setArgumentRequest(study, argument.key, value));
                              }
                            }}
                            data-testid={`simulate-arg-${study.name}-${argument.key}`}
                          />
                        </label>
                      )}
                    </For>
                    {/* The mesh/domain a study discretizes: a declared
                        SimMesh by name, or a named object restricting the
                        implicit mesh. Both are plain source rewrites. */}
                    <Show when={simMeshes().length > 0}>
                      <label>
                        <span>mesh</span>
                        <select
                          value={study.mesh ?? ""}
                          disabled={solving() !== null}
                          onChange={(event) => {
                            const value = event.currentTarget.value;
                            if (value) void patch(setStudyMeshRequest(study, value));
                          }}
                          data-testid={`simulate-mesh-${study.name}`}
                        >
                          <Show when={study.mesh === null}>
                            <option value="">(implicit)</option>
                          </Show>
                          <For each={simMeshes()}>
                            {(mesh) => <option value={mesh.name}>{mesh.name}</option>}
                          </For>
                        </select>
                      </label>
                    </Show>
                    <Show when={study.mesh === null && namedObjects().length > 0}>
                      <label>
                        <span>domain</span>
                        <select
                          value={study.domain?.name ?? ""}
                          disabled={solving() !== null}
                          onChange={(event) => {
                            const value = event.currentTarget.value;
                            if (value) void patch(setStudyDomainRequest(study, value));
                          }}
                          data-testid={`simulate-domain-${study.name}`}
                        >
                          <Show when={study.domain === null}>
                            <option value="">(whole scene)</option>
                          </Show>
                          <Show when={study.domain && study.domain.name === null}>
                            <option value="">{`(${study.domain!.type})`}</option>
                          </Show>
                          <For each={namedObjects()}>
                            {(name) => <option value={name}>{name}</option>}
                          </For>
                        </select>
                      </label>
                    </Show>
                  </div>

                  <Show when={simView()}>
                    <label class="switch compact sim-show-bcs">
                      <input
                        type="checkbox"
                        checked={showBcs() === study.index}
                        onChange={(event) =>
                          setShowBcs(event.currentTarget.checked ? study.index : null)
                        }
                        data-testid={`simulate-show-bcs-${study.name}`}
                      />
                      <span>Show BCs on mesh</span>
                    </label>
                    <Show when={showBcs() === study.index}>
                      <div class="sim-bc-legend" data-testid="simulate-bc-legend">
                        <For each={bcTypesFor(study.kind)}>
                          {(type) => (
                            <span>
                              <i
                                style={{
                                  background: `rgb(${BC_TYPE_COLORS[type]
                                    .map((channel) => Math.round(channel * 255))
                                    .join(", ")})`,
                                }}
                              />
                              {BC_LABELS[type]}
                            </span>
                          )}
                        </For>
                      </div>
                    </Show>
                  </Show>

                  <ul class="sim-bcs" data-testid={`simulate-bcs-${study.name}`}>
                    <For each={study.bcs}>
                      {(bc, bcIndex) => (
                        <li
                          classList={{ "sim-bc-readonly": !bc.serializable }}
                          onMouseEnter={() =>
                            setHoveredBc({ study: study.index, bc: bcIndex() })
                          }
                          onMouseLeave={() => setHoveredBc(null)}
                        >
                          <div class="sim-bc-main">
                            <span class="sim-bc-type">
                              <i
                                class="sim-bc-swatch"
                                style={{
                                  background: `rgb(${BC_TYPE_COLORS[bc.type]
                                    .map((channel) => Math.round(channel * 255))
                                    .join(", ")})`,
                                }}
                              />
                              {BC_LABELS[bc.type]}
                            </span>
                            <code title={describeSelection(bc.nodes)}>
                              {describeSelection(bc.nodes)}
                            </code>
                            <Show when={!selectionEvaluable(bc.nodes)}>
                              <small class="sim-note">no preview (predicate)</small>
                            </Show>
                          </div>
                          <Show
                            when={bc.serializable}
                            fallback={<small class="sim-note">edit in code</small>}
                          >
                            <Show when={bcValue(bc) !== null}>
                              <Show
                                when={bc.type === "traction"}
                                fallback={
                                  <input
                                    type="number"
                                    value={bcValue(bc) as number}
                                    disabled={solving() !== null}
                                    onChange={(event) => {
                                      const value = parse(event.currentTarget.value);
                                      if (value !== null) {
                                        void patch(setBcValueRequest(study, bcIndex(), value));
                                      }
                                    }}
                                    title={BC_LABELS[bc.type]}
                                    data-testid={`simulate-bc-value-${study.name}-${bcIndex()}`}
                                  />
                                }
                              >
                                <span class="sim-vector">
                                  <Index each={bcValue(bc) as number[]}>
                                    {(component, index) => (
                                      <input
                                        type="number"
                                        value={component()}
                                        disabled={solving() !== null}
                                        onChange={(event) => {
                                          const value = parse(event.currentTarget.value);
                                          if (value === null) return;
                                          const vector = [...(bcValue(bc) as number[])];
                                          vector[index] = value;
                                          void patch(
                                            setBcValueRequest(study, bcIndex(), vector),
                                          );
                                        }}
                                        title={`Traction ${AXIS_LABELS[index]}`}
                                      />
                                    )}
                                  </Index>
                                </span>
                              </Show>
                            </Show>
                            <button
                              type="button"
                              class="sim-delete"
                              onClick={() => void patch(deleteBcRequest(study, bcIndex()))}
                              title="Remove this boundary condition"
                              aria-label="Remove boundary condition"
                              data-testid={`simulate-bc-delete-${study.name}-${bcIndex()}`}
                            >
                              ×
                            </button>
                          </Show>
                        </li>
                      )}
                    </For>
                  </ul>

                  <Show
                    when={building() === study.index}
                    fallback={
                      <button
                        type="button"
                        class="sim-add-bc"
                        onClick={() => openBuilder(study)}
                        data-testid={`simulate-add-bc-${study.name}`}
                      >
                        + Boundary condition
                      </button>
                    }
                  >
                    <div class="sim-builder" data-testid="simulate-builder">
                      <div class="sim-builder-row">
                        <label>
                          <span>Type</span>
                          <select
                            value={draft().bcType}
                            onChange={(event) =>
                              setDraft({
                                ...draft(),
                                bcType: event.currentTarget
                                  .value as BcDraft["bcType"],
                              })
                            }
                            data-testid="simulate-builder-type"
                          >
                            <For each={bcTypesFor(study.kind)}>
                              {(type) => <option value={type}>{BC_LABELS[type]}</option>}
                            </For>
                          </select>
                        </label>
                        <label>
                          <span>Select</span>
                          <select
                            value={draft().selectionKind}
                            onChange={(event) =>
                              setDraft({
                                ...draft(),
                                selectionKind: event.currentTarget
                                  .value as BuilderSelectionKind,
                              })
                            }
                            data-testid="simulate-builder-selection"
                          >
                            <For each={SELECTION_KINDS}>
                              {(kind) => <option value={kind.value}>{kind.label}</option>}
                            </For>
                          </select>
                        </label>
                      </div>

                      {/* Viewport picking: while armed and a mesh is shown,
                          a click proposes Nodes.sphere at the picked point
                          and a shift-drag rectangle proposes Nodes.box. */}
                      <Show when={simView()}>
                        <button
                          type="button"
                          class="sim-pick"
                          classList={{ active: picking() }}
                          onClick={() => setPicking(!picking())}
                          title="Click the mesh to propose a sphere; shift-drag for a box"
                          data-testid="simulate-builder-pick"
                        >
                          {picking() ? "Picking… click the mesh" : "Pick in viewport"}
                        </button>
                      </Show>

                      <Show when={draft().selectionKind === "side"}>
                        <div class="sim-sides" data-testid="simulate-builder-sides">
                          <For each={SIDES}>
                            {(side) => (
                              <button
                                type="button"
                                classList={{ active: draft().side === side }}
                                onClick={() => setDraft({ ...draft(), side })}
                              >
                                {side}
                              </button>
                            )}
                          </For>
                        </div>
                      </Show>
                      <Show when={draft().selectionKind === "box"}>
                        {vectorRow("minCorner", "Min")}
                        {vectorRow("maxCorner", "Max")}
                      </Show>
                      <Show when={draft().selectionKind === "sphere"}>
                        {vectorRow("center", "Center")}
                        <label class="sim-builder-vector">
                          <span>Radius</span>
                          <input
                            type="number"
                            step="0.1"
                            value={draft().radius}
                            onChange={(event) => {
                              const value = parse(event.currentTarget.value);
                              if (value !== null) setDraft({ ...draft(), radius: value });
                            }}
                            data-testid="simulate-builder-radius"
                          />
                        </label>
                      </Show>
                      <Show when={draft().selectionKind === "halfspace"}>
                        {vectorRow("point", "Point")}
                        {vectorRow("normal", "Normal")}
                      </Show>

                      <Show when={draft().bcType === "dirichlet" || draft().bcType === "heat_flux"}>
                        <label class="sim-builder-vector">
                          <span>{BC_LABELS[draft().bcType]}</span>
                          <input
                            type="number"
                            value={draft().value}
                            onChange={(event) => {
                              const value = parse(event.currentTarget.value);
                              if (value !== null) setDraft({ ...draft(), value });
                            }}
                            data-testid="simulate-builder-value"
                          />
                        </label>
                      </Show>
                      <Show when={draft().bcType === "traction"}>
                        <label class="sim-builder-vector">
                          <span>Vector</span>
                          <Index each={[0, 1, 2]}>
                            {(component) => (
                              <input
                                type="number"
                                step="0.1"
                                value={draft().vector[component()]}
                                onChange={(event) =>
                                  setVector("vector", component(), event.currentTarget.value)
                                }
                                title={`Traction ${AXIS_LABELS[component()]}`}
                              />
                            )}
                          </Index>
                        </label>
                      </Show>

                      <div class="sim-builder-actions">
                        <button
                          type="button"
                          onClick={() => void submitBc(study)}
                          data-testid="simulate-builder-add"
                        >
                          Add
                        </button>
                        <button type="button" onClick={closeBuilder}>
                          Cancel
                        </button>
                      </div>
                    </div>
                  </Show>
                </Show>

                <button
                  type="button"
                  class="sim-run"
                  disabled={solving() !== null || unavailable() || study.bcs.length === 0}
                  onClick={() => void solve(study)}
                  title={
                    study.bcs.length > 0
                      ? "Mesh the scene and run this study"
                      : "Add at least one boundary condition first"
                  }
                  data-testid={`simulate-run-${study.name}`}
                >
                  {solving() === study.name ? "Meshing + solving…" : "Solve"}
                </button>
              </li>
            )}
          </For>
        </ul>
      </Show>

      <div class="sim-row sim-add-study">
        <button
          type="button"
          onClick={() => void patch(addStudyRequest("thermal"))}
          data-testid="simulate-add-thermal"
        >
          + Thermal study
        </button>
        <button
          type="button"
          onClick={() => void patch(addStudyRequest("elastic"))}
          data-testid="simulate-add-elastic"
        >
          + Elastic study
        </button>
      </div>
      </Show>

      {/* ── Optimize tab: the shared optimization cards, next to the
          simulation they drive — a study-backed run lands in Results. ───── */}
      <Show when={tab() === "optimize"}>
        <OptimizeCards
          onPatch={props.onPatch}
          onAdoptSource={props.onAdoptSource}
          onGhostCompile={props.onGhostCompile}
        />
      </Show>

      {/* ── Results tab: browse the solved surface ──────────────────────── */}
      <Show when={tab() === "results" && !result()}>
        <p class="sim-help" data-testid="simulate-results-empty">
          No results yet — solve a study or run a study-backed optimization.
        </p>
      </Show>
      <Show when={tab() === "results" && result()}>
        {(current) => (
          <>
            <Show when={Object.keys(current().fields).length > 1}>
              <div class="segmented sim-field-picker" data-testid="simulate-fields">
                <For each={Object.keys(current().fields)}>
                  {(field) => (
                    <button
                      type="button"
                      classList={{
                        active: !qualityView() && (activeField() ?? current().defaultField) === field,
                      }}
                      onClick={() => {
                        setQualityView(false);
                        setActiveField(field);
                        applyView();
                      }}
                      data-testid={`simulate-field-${field}`}
                    >
                      {field.replaceAll("_", " ")}
                    </button>
                  )}
                </For>
              </div>
            </Show>

            <div class="sim-legend" data-testid="simulate-legend">
              <small>
                {current().name} · {activeScalars()?.label ?? ""}
              </small>
              <div class="sim-ramp" style={{ background: rampCss() }} />
              <div class="sim-legend-values">
                <span>{formatScalar(activeScalars()?.range[0] ?? NaN)}</span>
                <span>{formatScalar(activeScalars()?.range[1] ?? NaN)}</span>
              </div>
            </div>

            <Show when={current().result}>
              {(summary) => (
                <div class="sim-result-summary" data-testid="simulate-result-summary">
                  <div class="sim-stats">
                    <span>
                      nodes <b>{summary().nodes}</b>
                    </span>
                    <span>
                      elements <b>{summary().elements}</b>
                    </span>
                    <Show when={summary().mesh}>
                      <span>
                        mesh <b>{summary().mesh}</b>
                      </span>
                    </Show>
                  </div>
                  <table class="sim-field-table">
                    <thead>
                      <tr>
                        <th>field</th>
                        <th>min</th>
                        <th>mean</th>
                        <th>max</th>
                      </tr>
                    </thead>
                    <tbody>
                      <For each={Object.entries(summary().fields)}>
                        {([field, values]) => (
                          <tr>
                            <td>{field.replaceAll("_", " ")}</td>
                            <td>{formatScalar(values.min)}</td>
                            <td>{formatScalar(values.mean)}</td>
                            <td>{formatScalar(values.max)}</td>
                          </tr>
                        )}
                      </For>
                    </tbody>
                  </table>
                </div>
              )}
            </Show>

            {/* The step-through control lands with the user: when this
                result came from an optimization run, its trajectory player
                mounts here too (same shared state as the Optimize card).
                Replay hands the viewport to the raymarched scene so the
                geometry visibly morphs, then restores the mesh view. */}
            <Show when={optimizeRun()?.name === current().name ? optimizeRun() : null}>
              {(run) => (
                <TrajectoryPlayer
                  onGhostCompile={props.onGhostCompile}
                  sparkTestId="results-optimize-history"
                  fieldNote={run().study !== null}
                  onReplayStart={() => setViewport("scene")}
                  onReplayEnd={() => setViewport("mesh")}
                />
              )}
            </Show>

            <label class="switch compact">
              <input
                type="checkbox"
                checked={qualityView()}
                disabled={inspecting() !== null}
                onChange={() => void toggleQualityView()}
                data-testid="simulate-quality-toggle"
              />
              <span>View mesh quality</span>
            </label>

            <Show when={current().displacements}>
              <div class="sim-row sim-deform">
                <label class="switch compact">
                  <input
                    type="checkbox"
                    checked={deformed()}
                    onChange={(event) => {
                      setDeformed(event.currentTarget.checked);
                      applyView();
                    }}
                    data-testid="simulate-deformed"
                  />
                  <span>Deformed</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="3"
                  step="0.05"
                  value={deformFactor()}
                  disabled={!deformed()}
                  onInput={(event) => {
                    setDeformFactor(Number(event.currentTarget.value));
                    applyView();
                  }}
                  title="Warp scale, relative to the automatic 10%-of-diagonal"
                  data-testid="simulate-deformed-scale"
                />
              </div>
            </Show>

            {/* The last probed point, mirrored from the viewport chip. */}
            <Show when={simProbe()}>
              {(probe) => (
                <div class="sim-stats sim-probe-row" data-testid="simulate-probe-row">
                  <span>
                    probe <b>{formatScalar(probe().value)}</b>
                  </span>
                  <span>
                    at{" "}
                    <b>
                      [{probe().world.map((component) => component.toFixed(3)).join(", ")}]
                    </b>
                  </span>
                </div>
              )}
            </Show>
          </>
        )}
      </Show>

      <Show when={result() || inspected()}>
        {/* Viewport master switch: the loaded mesh/result never owns the
            viewport for good — "Scene" returns to the raymarched SDF and
            keeps the loaded state one click away. ("Both" is deliberately
            absent: the mesh sits inside the opaque raymarched solid, so a
            composite would only hide it.) */}
        <div class="sim-row sim-view-control">
          <div class="segmented sim-viewport" data-testid="simulate-viewport">
            <button
              type="button"
              classList={{ active: viewportMode() === "scene" }}
              onClick={() => setViewport("scene")}
              title="Show the raymarched scene; the loaded mesh stays ready"
              data-testid="simulate-viewport-scene"
            >
              Scene
            </button>
            <button
              type="button"
              classList={{ active: viewportMode() === "mesh" }}
              onClick={() => setViewport("mesh")}
              title="Show the loaded mesh or solved field"
              data-testid="simulate-viewport-mesh"
            >
              Mesh
            </button>
          </div>
          <label
            class="switch compact"
            title={
              hasEdges()
                ? "Hairline element boundary edges over the surface"
                : "This payload carries no element edges"
            }
          >
            <input
              type="checkbox"
              checked={showEdges() && hasEdges()}
              disabled={!hasEdges()}
              onChange={(event) => setShowEdges(event.currentTarget.checked)}
              data-testid="simulate-edges"
            />
            <span>Element edges</span>
          </label>
        </div>
        <div class="sim-row sim-slice">
          <label class="sim-slice-toggle">
            <input
              type="checkbox"
              checked={slice().enabled}
              onChange={(event) => applySlice({ enabled: event.currentTarget.checked })}
              data-testid="simulate-slice-enabled"
            />
            <span>Slice</span>
          </label>
          <For each={[0, 1, 2] as const}>
            {(axis) => (
              <button
                type="button"
                class={slice().axis === axis ? "active" : ""}
                onClick={() => applySlice({ axis })}
                title={`Slice along ${AXIS_LABELS[axis]}`}
                data-testid={`simulate-slice-${AXIS_LABELS[axis].toLowerCase()}`}
              >
                {AXIS_LABELS[axis]}
              </button>
            )}
          </For>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={slice().fraction}
            disabled={!slice().enabled}
            onInput={(event) => applySlice({ fraction: Number(event.currentTarget.value) })}
            data-testid="simulate-slice-fraction"
          />
        </div>
      </Show>

      <Show when={unavailable()}>
        <p class="sim-note" data-testid="simulate-unavailable">
          FEM solves need the optional jax-fem extra on the server. Install it
          with <code> pip install cadjoint[fem]</code> and restart the playground.
          Study declarations still compile and stay editable without it.
        </p>
      </Show>
      <Show when={error() && !unavailable()}>
        <p class="sim-error" data-testid="simulate-error">
          {error()}
        </p>
      </Show>
    </aside>
  );
}
