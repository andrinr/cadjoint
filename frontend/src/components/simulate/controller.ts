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
 * The four simulation windows — Meshes, Studies, Optimize, Results — are
 * separate windows in the dock now, and the dock gives each one its own Solid
 * root. One shared controller therefore cannot live in any of them: it is
 * created once in the app shell and handed to each window, and each window
 * calls {@link SimulateController.attach} for as long as it is mounted. That
 * count is what used to be the panel's own lifetime — the first window to
 * mount restores the last result and hands the viewport to the mesh view, and
 * the last one to unmount hands it back to the raymarched scene. It also
 * keeps a solve that lands while no simulation window is open (a study-backed
 * optimization run from Model mode) from silently taking over the viewport.
 */

import { createEffect, createSignal, onCleanup } from "solid-js";
import * as api from "../../api";
import {
  awaitJobResult,
  cancelJob,
  dropJobRef,
  findMatchingJob,
  findRunningJob,
  isStale,
  jobsSnapshot,
  loadJobRef,
  saveJobRef,
  sceneKey,
  sourceHash,
  takeRequestedJob,
  requestedJob,
  watchJobs,
  type JobRef,
} from "../../jobs";
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
  sceneName,
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
  MeshInspectResponse,
  SimMeshPayload,
  SimulateResponse,
  SimulationMeshPayload,
  SimulationResultSummary,
  StudyPayload,
} from "../../types";
import { windowManager } from "../../windows/manager";
import type { Renderer } from "../../viewer/renderer";

/** What the app shell hands the controller when it creates it. */
export interface SimulateControllerProps {
  renderer: Renderer;
  /** Serialized /patch queue owned by the app shell. */
  onPatch: (body: Record<string, unknown>) => Promise<void>;
  /** Adopt server-produced source like a patch response (optimize runs). */
  onAdoptSource: (source: string) => Promise<void>;
  /** Compile-and-render a transient program without committing it. */
  onGhostCompile: (source: string) => Promise<boolean>;
}

/** The simulation windows that share this controller's state. */
export type SimWindow = "meshes" | "studies" | "results";

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

export function createSimulateController(props: SimulateControllerProps) {
  /**
   * How many simulation windows are mounted, and which.
   *
   * `attach` is the windows' half of the old panel lifecycle: a count that
   * rises as they mount and falls as they unmount, with the transitions
   * through zero standing in for the panel's `onMount` and `onCleanup`.
   */
  const [mounted, setMounted] = createSignal<readonly SimWindow[]>([]);
  const attached = () => mounted().length > 0;
  const resultsMounted = () => mounted().includes("results");
  /** Which study's add-BC builder is open, by study index. */
  const [building, setBuilding] = createSignal<number | null>(null);
  const [draft, setDraft] = createSignal<BcDraft>(defaultDraft("thermal"));
  const [solving, setSolving] = createSignal<string | null>(null);
  const [inspecting, setInspecting] = createSignal<string | null>(null);
  const [error, setError] = createSignal("");
  /**
   * The job behind the message in {@link error}, when a job produced it.
   *
   * A failed solve is not a one-line status: TetGen refusing a self-
   * intersecting surface is three sentences the user needs, and the registry
   * already keeps them on the job. Holding the id here is what lets the
   * window that asked show the whole message *and* offer the row it came
   * from, instead of the failure going only to the server log.
   */
  const [errorJob, setErrorJob] = createSignal<string | null>(null);
  const [unavailable, setUnavailable] = createSignal(false);
  const [result, setResult] = createSignal<SolveState | null>(null);
  const [inspected, setInspected] = createSignal<InspectState | null>(null);
  /**
   * Persistence across the panel's own lifetime.
   *
   * Leaving Simulate mode disposes this panel's Solid root, so before the
   * job registry a solved field died with the mode switch and had to be paid
   * for again. What is kept now is four fields — the job id, the hash of the
   * program it ran on, its kind and its name — and the payload is fetched
   * back from `/api/jobs/<id>/result` when the panel mounts again.
   */
  const [restoring, setRestoring] = createSignal(false);
  /** The reference to whatever is displayed, or null when nothing is. */
  const [heldRef, setHeldRef] = createSignal<JobRef | null>(null);
  /** A stored result the server no longer has: the work has to be redone. */
  const [expired, setExpired] = createSignal<string | null>(null);
  /** The running job behind the in-flight solve, once the poll has found it. */
  const [solveJob, setSolveJob] = createSignal<string | null>(null);
  /** sha256 of the document, recomputed as it changes, for staleness. */
  const [documentHash, setDocumentHash] = createSignal<string | null>(null);
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

  // ── results that outlive this panel ──────────────────────────────────────

  /** Results are stored per scene: two documents keep their own last solve. */
  const scene = () => sceneKey(sceneName());
  /** Set once the panel is torn down, so a slow restore stops waiting. */
  let unmounted = false;
  onCleanup(() => {
    unmounted = true;
  });

  // The document's hash, recomputed as the program changes. It is what tells
  // a result that still describes the editor's text from one the last edit
  // invalidated, and it is cheap: sha256 of a few kilobytes, per edit.
  createEffect(() => {
    const text = source();
    void sourceHash(text).then((hash) => setDocumentHash(hash));
  });

  /** Whether what is displayed was solved from a program that has changed. */
  const stale = (): boolean => {
    const ref = heldRef();
    return ref ? isStale(ref, documentHash()) : false;
  };

  /** Keep the reference to a finished job, in memory and in storage. */
  const remember = async (
    jobId: string | undefined,
    kind: "simulate" | "mesh_inspect",
    posted: string,
    fields: Record<string, string | number | boolean>,
  ) => {
    if (!jobId) return;
    const ref: JobRef = { job_id: jobId, source_hash: await sourceHash(posted), kind, fields };
    setHeldRef(ref);
    saveJobRef(scene(), ref);
  };

  /**
   * Display the payload a job produced, fetching it by id.
   *
   * Three outcomes are all ordinary: the payload arrives and is adopted; the
   * job is still running, in which case this waits it out rather than
   * failing; or the server no longer has the result — the registry is a
   * bounded window — and the panel says so instead of showing nothing.
   */
  const restore = async (ref: JobRef, options: { onlyIfEmpty?: boolean } = {}) => {
    setRestoring(true);
    try {
      const outcome = await awaitJobResult<SimulateResponse & MeshInspectResponse>(
        ref.job_id,
        { stopped: () => unmounted },
      );
      if (unmounted) return;
      if (outcome.state === "gone") {
        dropJobRef(scene(), ref.kind);
        if (heldRef()?.job_id === ref.job_id) setHeldRef(null);
        setExpired("That result is no longer stored on the server — run it again.");
        return;
      }
      if (outcome.state !== "ok") return;
      // A restore that nobody asked for must not overwrite something that
      // arrived while it was in flight — a finished optimization publishes
      // its solved field through `optimizeSimulate` on the same mount.
      if (options.onlyIfEmpty && (result() || inspected())) return;
      const name =
        typeof ref.fields?.name === "string" ? ref.fields.name : (outcome.payload.name ?? ref.kind);
      const adopted =
        ref.kind === "mesh_inspect"
          ? adoptInspect(name, outcome.payload)
          : adoptSolve(name, outcome.payload);
      if (!adopted) {
        dropJobRef(scene(), ref.kind);
        setExpired("That result could not be re-opened — run it again.");
        return;
      }
      setExpired(null);
      setHeldRef(ref);
      saveJobRef(scene(), ref);
    } finally {
      setRestoring(false);
    }
  };

  /**
   * What this panel shows when it mounts with nothing in hand: whatever it
   * was showing the last time it was mounted, for this scene.
   *
   * Only the reference this browser stored — the result the *user* asked
   * for. The server-wide lookup below is deliberately not done here: a
   * playground shared with an earlier session would otherwise open the
   * Simulate panel on somebody else's result and on the Results tab, and a
   * panel that greets you with an answer you did not ask a question for is
   * not a helpful panel.
   */
  const restoreOnMount = async () => {
    if (result() || inspected()) return;
    // Both a solve and an inspection can be stored for one scene, and only
    // the more recent one was on screen. Job ids are zero-padded and issued
    // in order, so the larger id is the later job — which is exactly the
    // question being asked.
    const refs = [loadJobRef(scene(), "simulate"), loadJobRef(scene(), "mesh_inspect")].filter(
      (ref): ref is JobRef => ref !== null,
    );
    const stored = refs.sort((a, b) => b.job_id.localeCompare(a.job_id))[0];
    if (stored) await restore(stored, { onlyIfEmpty: true });
  };

  /** The document text the server-wide lookup has already been tried for. */
  let searchedHash: string | null = null;

  /**
   * Opening an empty Results window asks the server whether it has one.
   *
   * This is the answer to "I solved this yesterday, why is the tab empty":
   * the newest finished solve whose source hash matches the document in the
   * editor is fetched and shown. It runs when the tab is *opened* rather
   * than when the panel mounts, so it can never pull the user onto Results
   * behind their back, and once per document text, so an empty window does
   * not re-ask on every render.
   */
  createEffect(() => {
    if (!resultsMounted() || result() || inspected() || restoring()) return;
    const hash = documentHash();
    if (!hash || searchedHash === hash) return;
    searchedHash = hash;
    void (async () => {
      const match = await findMatchingJob("simulate", hash);
      if (!match || unmounted) return;
      await restore(
        {
          job_id: match.job_id,
          source_hash: match.source_hash,
          kind: "simulate",
          fields: match.fields ?? {},
        },
        { onlyIfEmpty: true },
      );
    })();
  });

  // A row clicked in the Processes window arrives here: this is the panel
  // that knows how to draw a solved field, so it consumes the request.
  createEffect(() => {
    requestedJob();
    const ref = takeRequestedJob("simulate") ?? takeRequestedJob("mesh_inspect");
    if (ref) void restore(ref);
  });

  // ── stopping work that is already running ────────────────────────────────

  /**
   * Poll the registry while a request is in flight, to learn its job id.
   *
   * The id only comes back with the response, which is exactly too late to
   * cancel with — so while a solve or an inspection is running, the panel
   * watches the job list and picks out the running job of that kind. The
   * poll is reference-counted and shared with the Processes window, and it
   * stops the moment the request settles.
   */
  createEffect(() => {
    if (!solving() && !inspecting()) return;
    onCleanup(watchJobs());
  });

  createEffect(() => {
    const kind = solving() ? "simulate" : inspecting() ? "mesh_inspect" : null;
    if (!kind) {
      setSolveJob(null);
      return;
    }
    const snap = jobsSnapshot();
    const found = snap ? findRunningJob(snap.jobs, kind, documentHash()) : null;
    if (found) setSolveJob(found.job_id);
  });

  /** Kill the in-flight solve or inspection; its request answers by itself. */
  const cancelActive = async () => {
    const jobId = solveJob();
    if (!jobId) return;
    await cancelJob(jobId);
  };

  /**
   * Choose what the viewport draws — but only while a window is watching.
   *
   * A study-backed optimization run from Model mode publishes a solved field
   * with no simulation window open. Adopting it is right; seizing the
   * viewport for it is not, so the choice is remembered and applied when the
   * first simulation window mounts.
   */
  const setViewport = (mode: ViewportMode) => {
    setViewportMode(mode);
    if (attached()) props.renderer.simulationActive = mode === "mesh";
  };

  /**
   * Raise the Results window: a solve just landed in it.
   *
   * The old panel switched its own tab. A window cannot do that to itself, so
   * this asks the dock — and only when a simulation window is mounted at all,
   * so an optimization run from the Model desk does not push a Results window
   * into a desk that never asked for one.
   */
  const showResults = () => {
    if (!attached()) return;
    windowManager()?.open("results");
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

  /**
   * Register a mounted simulation window; the returned function unregisters.
   *
   * The first window to arrive picks up whatever was last displayed for this
   * scene and hands the viewport to the mesh view; the last one to leave
   * hands it back. Everything between is bookkeeping.
   */
  const attach = (which: SimWindow): (() => void) => {
    const first = mounted().length === 0;
    setMounted((current) => [...current, which]);
    if (first) {
      if (simView()) setViewport(viewportMode());
      // Coming back after a mode switch: the windows' Solid roots were
      // disposed, the result was not.
      void restoreOnMount();
    }
    let released = false;
    return () => {
      if (released) return;
      released = true;
      let dropped = false;
      setMounted((current) =>
        current.filter((entry) => {
          if (dropped || entry !== which) return true;
          dropped = true;
          return false;
        }),
      );
      if (mounted().length > 0) return;
      props.renderer.simulationActive = false;
      props.renderer.setSimulationOverlay(null);
      setBcPickArmed(false);
      setSimProbe(null);
    };
  };

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

  /**
   * Take a solved response as the displayed result.
   *
   * Shared by the three ways a solve can arrive: the user pressing Solve, a
   * stored result fetched back by job id after the panel remounted, and a
   * row re-opened from the Processes window. All three land in exactly the
   * same state, which is the point — a restored result is not a lesser copy
   * of a fresh one.
   */
  const adoptSolve = (name: string, response: SimulateResponse): boolean => {
    if (!response.mesh) return false;
    const defaultField = response.field ?? "field";
    setInspected(null);
    // The field catalog rides on the mesh payload (top level is the
    // legacy home; coalesce both while the backend contract settles).
    setResult({
      name,
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
    showResults();
    return true;
  };

  const solve = async (study: StudyPayload) => {
    const posted = source();
    setSolving(study.name);
    setExpired(null);
    setError("");
    setErrorJob(null);
    try {
      const response = await api.simulateStudy({
        source: posted,
        kind: "study",
        name: study.name,
      });
      if (response.error_kind === "cancelled") {
        // A cancellation is something the user asked for, not a failure.
        return;
      }
      if (response.error_kind === "fem_unavailable") {
        setUnavailable(true);
        fail(response.error ?? "The jax-fem extra is not installed.", response.job_id);
        return;
      }
      if (!response.ok || !response.mesh) {
        // Whatever is on screen stays on screen. A failed solve does not
        // make the last one wrong — it makes it *old*, which the stale chip
        // already says — and clearing it would replace a real answer with a
        // blank panel at the exact moment the user needs to compare them.
        fail(response.error ?? "The solve failed.", response.job_id);
        return;
      }
      adoptSolve(study.name, response);
      await remember(response.job_id, "simulate", posted, {
        kind: "study",
        name: study.name,
      });
    } catch (caught) {
      fail(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setSolving(null);
      setSolveJob(null);
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
    showResults();
  });

  /** Take an inspection response as the displayed mesh. See `adoptSolve`. */
  const adoptInspect = (name: string, response: MeshInspectResponse): boolean => {
    if (!response.mesh || !response.info) return false;
    setResult(null);
    setInspected({
      name: response.name ?? name,
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
    return true;
  };

  const inspect = async (mesh: SimMeshPayload | { name: string }) => {
    const posted = source();
    setInspecting(mesh.name);
    setExpired(null);
    setError("");
    setErrorJob(null);
    try {
      const response = await api.meshInspect(posted, mesh.name);
      if (response.error_kind === "cancelled") return;
      if (!response.ok || !response.mesh || !response.info) {
        fail(response.error ?? "Mesh inspection failed.", response.job_id);
        return;
      }
      adoptInspect(mesh.name, response);
      await remember(response.job_id, "mesh_inspect", posted, { name: mesh.name });
    } catch (caught) {
      fail(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setInspecting(null);
      setSolveJob(null);
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
          fail(
            response.error ?? "Mesh quality is unavailable for this result.",
            response.job_id,
          );
          return;
        }
        setResult({ ...solved, qualityScalars: response.quality_scalars });
      } catch (caught) {
        fail(caught instanceof Error ? caught.message : String(caught));
        return;
      } finally {
        setInspecting(null);
      }
    }
    setQualityView(true);
    applyView();
  };

  /** Report a failed request: the whole message, and the job it came from. */
  const fail = (message: string, jobId?: string | null) => {
    setError(message);
    setErrorJob(jobId ?? null);
  };

  /** Route one patch body through the app queue, surfacing failures here. */
  const patch = async (body: Record<string, unknown>) => {
    setError("");
    setErrorJob(null);
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

  /**
   * The domain picker's options, including whatever is already declared.
   *
   * A domain is written as a variable, and that variable does not have to be
   * a construction node: `housing = Difference(...)` is a perfectly ordinary
   * domain and the object tree never lists it. Offering only the tree's names
   * left the control matching no option at all, so the select rendered blank
   * for a mesh whose source plainly said `domain=housing`. The declared name
   * leads the list when the tree does not already carry it — the control then
   * shows what the code says, which is the only honest thing for it to show.
   */
  const domainOptions = (current: string | null | undefined): string[] => {
    const names = namedObjects();
    return current && !names.includes(current) ? [current, ...names] : names;
  };

  const histogram = () => {
    const inspect = inspected();
    if (!inspect) return null;
    return qualityHistogram(inspect.qualityScalars, HISTOGRAM_BINS);
  };

  return {
    // The panel's own props, so tabs can reach the app-shell callbacks.
    props,
    // State.
    attach,
    attached,
    building,
    draft,
    setDraft,
    solving,
    inspecting,
    error,
    errorJob,
    unavailable,
    result,
    inspected,
    slice,
    // Job-backed persistence.
    stale,
    restoring,
    expired,
    setExpired,
    solveJob,
    cancelActive,
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
    domainOptions,
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
