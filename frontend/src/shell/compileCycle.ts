/**
 * The compile/patch cycle: the loop that keeps the viewer and the code equal.
 *
 *   viewer edit → POST /patch → new source into the editor → POST /compile →
 *   new shaders + construction tree → viewer redraws.
 *
 * Everything that has to be serialized or de-duplicated to keep that loop
 * honest lives here rather than in the shell's JSX: the patch queue (UI
 * actions arriving during a compile must each start from the previous edit's
 * source instead of racing), the transient ghost compile used by optimization
 * replay, and the lazy mesh-edge fetch with its one-request-per-compile cache.
 *
 * ### The newest edit wins
 *
 * The ordering rule this file used to have was a latch: an edit arriving
 * mid-compile set `rerunRequested` and *waited*. On a scene whose compile is
 * twenty-five seconds that made two quick drags a fifty-second wait, during
 * which the viewport showed geometry two edits old — and the machine spent the
 * first twenty-five of those seconds computing an answer that was already
 * known to be unwanted.
 *
 * It is now the opposite, through `shell/supersede.ts`: a new request replaces
 * the one in flight, kills its worker, and starts immediately. Two properties,
 * both wanted:
 *
 * - the **revision guard** means a late answer from a superseded compile can
 *   never be applied, whether or not the cancel landed in time (correctness);
 * - the **cancel** means the superseded worker stops burning a core and a
 *   gigabyte in competition with the compile the user is waiting for
 *   (performance).
 *
 * Both compile entry points — a committed run and the optimizer's transient
 * ghost frame — share one revision counter, because they write the same
 * shaders through the same renderer and an out-of-order winner between them
 * would put geometry on screen that is in nobody's program.
 *
 * The panels never see any of this: they get `applyPatch`, `adoptSource` and
 * `ghostCompile` and stay ignorant of ordering.
 */

import { createEffect, createSignal, type Accessor } from "solid-js";
import * as api from "../api";
import { cancelClientJob, nextRequestId, watchJobs } from "../jobs";
import {
  selection,
  setBusy,
  setConsoleText,
  setDirty,
  setMaterials,
  setMeshEdges,
  setNodes,
  setOptimizations,
  setRelations,
  setSelection,
  setSimMeshes,
  setSolverRuns,
  setSource,
  setStatus,
  setStudies,
  setViewerError,
  meshEdges,
  source,
} from "../state";
import { createSuperseding, type RunToken } from "./supersede";
import type { SourceHistoryStore } from "./sourceHistory";
import type { DisplaySettings, Renderer } from "../viewer/renderer";

/** The generated shader pair, for the "Generated WGSL" dialog. */
export interface WgslPreview {
  preview: string;
  path: string;
}

/**
 * Milliseconds of quiet before a requested compile actually goes out.
 *
 * The number has to clear the cadence of the fastest legitimate burst and
 * stay under the threshold at which a first edit feels laggy. The bursts are
 * a drag's patch-on-release (a double action is tens of milliseconds apart),
 * a key-repeat nudge (33 ms on macOS after its initial delay), and a panel
 * sequence like constraint → satisfy → extrude, which the patch queue emits
 * back to back. 150 ms swallows all three, and against a compile measured in
 * tens of seconds it is not a delay anyone can perceive — the indicator is
 * already up, because `busy` is set when the edit is made rather than when
 * the request goes out.
 *
 * It is not a cap on how long coalescing may continue: a burst that never
 * ends would never compile. Nothing in the app emits one — every trigger is a
 * discrete user action, and a drag patches on release, not on move.
 */
export const COMPILE_DEBOUNCE_MS = 150;

export interface CompileCycleOptions {
  renderer: Renderer;
  /** Every committed run commits a snapshot, so undo lands on compiled states. */
  history: SourceHistoryStore;
  /** Mesh edges are fetched only while a mesh overlay is displayed. */
  display: Accessor<DisplaySettings>;
  /** Overridable so a test can drive the debounce without waiting on a clock. */
  debounceMs?: number;
}

export interface CompileCycle {
  /** Compile the current source and publish everything it produced. */
  run: () => Promise<void>;
  /** Send one edit through the serialized queue, then recompile. */
  applyPatch: (body: Record<string, unknown>) => Promise<void>;
  /** Adopt server-produced source exactly like a patch response. */
  adoptSource: (text: string) => Promise<void>;
  /** Compile-and-render a transient program without committing it. */
  ghostCompile: (text: string) => Promise<boolean>;
  wgsl: Accessor<WgslPreview | null>;
}

/** What one `/compile` came back with, before anything has been published. */
type CompileResult = Awaited<ReturnType<typeof api.compile>>;

type Attempt =
  /** A newer request owns the app; this answer is to be forgotten entirely. */
  | { state: "stale" }
  /** Somebody stopped this compile from the process chip. */
  | { state: "cancelled" }
  | { state: "failed"; message: string }
  | { state: "ok"; result: CompileResult };

export function createCompileCycle(options: CompileCycleOptions): CompileCycle {
  const [wgsl, setWgsl] = createSignal<WgslPreview | null>(null);

  // The mesh-edge overlay is fetched lazily: /compile no longer computes it,
  // and this cache requests it only while a mesh display mode is on.
  const [compiledSource, setCompiledSource] = createSignal<string | null>(null);
  let meshRequestFor: string | null = null;
  let meshInFlight: { clientId: string; controller: AbortController } | null = null;

  const compiles = createSuperseding({
    debounceMs: options.debounceMs ?? COMPILE_DEBOUNCE_MS,
  });

  /**
   * Say the picture on screen is no longer the picture of the code.
   *
   * Set when the edit is made rather than when the request goes out, so the
   * debounce window is not a window in which the app claims to be up to date.
   * Only the newest run ever clears it (see `compileInto`'s `finally`), which
   * is what keeps a superseded compile's ending from unmarking an app that is
   * still working on the edit that replaced it.
   */
  const markBusy = (): void => {
    setBusy(true);
    // And the status line says nothing while it runs. It is the *result*
    // readout — what the last work produced, what the renderer is drawing —
    // and the toolbar's running-work chip is the one indicator for work in
    // flight: it names the job, counts its seconds and can stop it. Two
    // readouts for one machine state is what this used to be.
    setStatus({ kind: "", text: "" });
    setConsoleText("");
  };

  /** Stop a mesh-edge fetch whose compile has just been replaced. */
  const dropMeshFetch = (): void => {
    const stale = meshInFlight;
    meshInFlight = null;
    if (!stale) return;
    stale.controller.abort();
    void cancelClientJob(stale.clientId);
  };

  /**
   * One `/compile`, with both halves of supersession attached to it.
   *
   * The revision guard is the load-bearing one and it is checked on every
   * path out, including the failure path: an aborted fetch of a superseded
   * request rejects, and reporting that as a compile error would put a
   * failure on screen for work nobody asked to finish.
   */
  const fetchCompile = async (token: RunToken, text: string): Promise<Attempt> => {
    const clientId = nextRequestId();
    const controller = new AbortController();
    token.onSupersede(() => {
      // Kill the worker first — that is the expensive half — and then drop
      // the connection this side is holding for an answer nobody wants.
      void cancelClientJob(clientId);
      controller.abort();
    });
    // Watch the registry for as long as the request is out, and not a poll
    // longer.
    //
    // A single nudge is not enough and the reason is a race worth naming: the
    // poll and the compile leave the browser together, so the answer can come
    // back before the worker has been registered at all. The poller then sees
    // nothing pending, nobody watching, and stops — and the chip spends the
    // whole twenty-five seconds unable to name the job or offer its ×, which
    // is precisely the compile most worth stopping. Holding a watcher instead
    // means the id turns up on the next tick, at the cost of one request a
    // second while a compile is running and none at all when it is not.
    const unwatch = watchJobs();
    try {
      const result = await api.compile(text, { clientId, signal: controller.signal });
      if (!token.current()) return { state: "stale" };
      if (result.error_kind === "cancelled") return { state: "cancelled" };
      if (!result.ok) {
        return { state: "failed", message: result.error ?? "Unknown compile error." };
      }
      return { state: "ok", result };
    } catch (error) {
      if (!token.current()) return { state: "stale" };
      return {
        state: "failed",
        message: error instanceof Error ? error.message : String(error),
      };
    } finally {
      unwatch();
    }
  };

  /** Publish everything a committed compile produced. */
  const publish = (text: string, result: CompileResult): void => {
    setDirty(false);
    setConsoleText(result.output ?? "");
    setNodes(result.construction ?? []);
    setRelations(result.relations ?? []);
    setSolverRuns(result.solver_runs ?? []);
    setMaterials(result.materials ?? []);
    setStudies(result.studies ?? []);
    setSimMeshes(result.sim_meshes ?? []);
    setOptimizations(result.optimizations ?? []);
    // Mesh edges are no longer part of the compile payload; clear the stale
    // overlay and let the lazy /api/mesh effect refill it when wanted.
    setMeshEdges(null);
    setCompiledSource(text);
    // Drop a selection that no longer exists in the rebuilt sketch.
    const active = selection();
    if (active) {
      const node = (result.construction ?? []).find((item) => item.id === active.nodeId);
      const stale =
        !node || (active.vertexIndex !== null && active.vertexIndex >= node.vertices.length);
      if (stale) setSelection(null);
    }
    setWgsl({ preview: result.preview_shader, path: result.path_shader });
    setViewerError("");
  };

  /**
   * One guarded compile, from either entry point.
   *
   * `commit` is the whole difference between them: a committed run is an edit
   * to the document — history snapshot, construction tree, studies, the lot —
   * while a ghost run is a transient frame of the optimizer's replay that
   * only swaps shaders. Busy, status and ordering are deliberately identical,
   * because the user is waiting on a compile either way and because a shared
   * counter is only sound if both sides also share the bookkeeping it drives.
   */
  const compileInto = async (
    token: RunToken,
    text: string,
    commit: boolean,
  ): Promise<boolean> => {
    try {
      if (commit) options.history.commit(text);
      // Mesh edges describe the program that is being replaced.
      dropMeshFetch();
      const attempt = await fetchCompile(token, text);
      if (attempt.state === "stale") return false;
      if (attempt.state === "cancelled") {
        setStatus({ kind: "stale", text: "Compile stopped — view is stale" });
        return false;
      }
      if (attempt.state === "failed") {
        setStatus({ kind: "error", text: "Compile failed" });
        setConsoleText(attempt.message);
        return false;
      }
      if (commit) publish(text, attempt.result);
      // The renderer replaces this as soon as it draws a frame; setting it here
      // means the status still settles on a machine without WebGPU.
      setStatus({ kind: "ready", text: "Scene compiled" });
      // Shaders are the one thing both entry points write, so the guard is
      // re-read immediately before the write rather than only before the
      // publish above: two installs racing inside the renderer is the one way
      // an older program could still end up on screen.
      if (!token.current()) return false;
      await options.renderer.setShaders({
        preview: attempt.result.preview_shader,
        path: attempt.result.path_shader,
        present: attempt.result.present_shader,
        // The uniform contract, when the worker emitted one: with it the
        // renderer can tell a parameter edit from a topology edit and skip
        // the shader module and pipelines entirely for the former.
        program: attempt.result.program ?? null,
      });
      return true;
    } catch (error) {
      if (!token.current()) return false;
      setStatus({ kind: "error", text: "Compile failed" });
      setConsoleText(error instanceof Error ? error.message : String(error));
      return false;
    } finally {
      // Only the newest run may say the app is idle. A superseded one ending
      // here is ending into an app that is still compiling the edit that
      // replaced it.
      if (token.current()) setBusy(false);
    }
  };

  const run = (): Promise<void> => {
    markBusy();
    // The source is read when the run *starts*, not when it is requested, so
    // a burst of patches coalesces into one compile of the final program.
    return compiles.request(async (token) => {
      await compileInto(token, source(), true);
    });
  };

  /**
   * Send one edit to the server, adopt the patched source, then rebuild.
   *
   * UI actions can arrive while the previous patch is compiling (constraint →
   * satisfy → extrude is a common sequence). Serialize them so every request
   * starts from the source produced by the preceding edit instead of racing
   * and letting the last network response discard another operation.
   *
   * The queue is over the *patches only*, and that is the half of chaining a
   * revision guard cannot fix on its own. When it also spanned the compile,
   * a second drag's `/patch` could not even leave the browser until the first
   * drag's twenty-five-second compile had finished — so there was no newer
   * request to supersede anything with, and the wait was the two compiles
   * added together. Now the queue advances the moment an edit has rewritten
   * the source: the next edit is sent immediately, from that new source, and
   * the compile it asks for replaces the one still running.
   *
   * The promise a caller gets still covers the compile, because a caller
   * awaiting an edit means "when the picture has caught up" — it just is no
   * longer what the *next* edit is made to wait on.
   */
  let patchQueue: Promise<void> = Promise.resolve();
  const performPatch = async (body: Record<string, unknown>): Promise<boolean> => {
    try {
      const result = await api.patch({ source: source(), ...body });
      if (!result.ok || !result.source) {
        setStatus({ kind: "error", text: result.error ?? "Edit failed" });
        return false;
      }
      setSource(result.source);
      return true;
    } catch (error) {
      setStatus({
        kind: "error",
        text: error instanceof Error ? error.message : String(error),
      });
      return false;
    }
  };
  const applyPatch = (body: Record<string, unknown>): Promise<void> => {
    const edited = patchQueue.then(() => performPatch(body));
    patchQueue = edited.then(
      () => undefined,
      () => undefined,
    );
    return edited.then((patched) => (patched ? run() : undefined));
  };

  /**
   * Adopt server-produced source exactly like a patch response.
   *
   * The optimizer is a patch layer too: a successful /api/optimize returns
   * the program with the optimized literals written back, and the app treats
   * it as one committed edit (history snapshot via run()).
   */
  const adoptSource = (text: string): Promise<void> => {
    const adopted = patchQueue.then(() => {
      setSource(text);
    });
    patchQueue = adopted.then(
      () => undefined,
      () => undefined,
    );
    return adopted.then(() => run());
  };

  /**
   * Compile-and-render a transient program without committing it.
   *
   * The optimization replay player scrubs through parameter snapshots by
   * substituting literals client-side; each frame shows in the editor and the
   * viewport but never lands in the undo history — only the adopted final
   * source does. Construction/studies state is refreshed by the caller's
   * closing adoptSource, so this only swaps the shaders.
   *
   * Scrubbing is the same problem as chained edits one level down, so it gets
   * the same answer: a frame requested while an older frame is compiling
   * replaces it, and the editor only adopts the frame that is actually going
   * to be drawn.
   */
  const ghostCompile = (text: string): Promise<boolean> => {
    markBusy();
    let landed = false;
    return compiles
      .request(async (token) => {
        setSource(text);
        landed = await compileInto(token, text, false);
      })
      .then(() => landed);
  };

  /**
   * Fetch mesh edges lazily: only while a mesh overlay is displayed, only for
   * the compiled program, and only once per compile (a "no mesh available"
   * answer is cached too, so the effect cannot loop on it).
   *
   * Superseded the same way a compile is, and for the same reason: extracting
   * the dual-contour mesh is the work that was split out of `/compile`
   * because it dominated the round trip, so a fetch describing a program the
   * user has already replaced is a worker to kill, not an answer to wait for.
   * The guard here is the compiled source rather than a revision counter —
   * same rule, expressed in the thing this cache is keyed by.
   */
  createEffect(() => {
    const display = options.display();
    const wanted = display.showMeshEdges || display.showMeshWireframe;
    const compiled = compiledSource();
    if (!wanted || compiled === null || meshEdges() !== null) return;
    if (meshRequestFor === compiled) return;
    meshRequestFor = compiled;
    const clientId = nextRequestId();
    const controller = new AbortController();
    const inFlight = { clientId, controller };
    meshInFlight = inFlight;
    // Watched for the same reason a compile is: extracting the mesh is a
    // registered job, the chip lists it, and without a watcher held for the
    // request the poller can stop before the registry has heard of it.
    const unwatch = watchJobs();
    const settle = () => {
      unwatch();
      if (meshInFlight === inFlight) meshInFlight = null;
    };
    void api
      .mesh(compiled, { clientId, signal: controller.signal })
      .then((result) => {
        settle();
        // A newer compile owns the cache now; drop the stale answer.
        if (compiledSource() !== compiled) return;
        if (result.ok) setMeshEdges(result.mesh_edges ?? null);
      })
      .catch(() => {
        // Missing mesh edges only dim an optional overlay; stay quiet.
        settle();
      });
  });

  return { run, applyPatch, adoptSource, ghostCompile, wgsl };
}
