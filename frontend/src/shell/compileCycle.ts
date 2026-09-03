/**
 * The compile/patch cycle: the loop that keeps the viewer and the code equal.
 *
 *   viewer edit → POST /patch → new source into the editor → POST /compile →
 *   new shaders + construction tree → viewer redraws.
 *
 * Everything that has to be serialized or de-duplicated to keep that loop
 * honest lives here rather than in the shell's JSX: the rerun latch (an edit
 * that lands mid-compile must not be dropped), the patch queue (UI actions
 * arriving during a compile must each start from the previous edit's source
 * instead of racing), the transient ghost compile used by optimization
 * replay, and the lazy mesh-edge fetch with its one-request-per-compile
 * cache.
 *
 * The panels never see any of this: they get `applyPatch`, `adoptSource` and
 * `ghostCompile` and stay ignorant of ordering.
 */

import { createEffect, createSignal, type Accessor } from "solid-js";
import * as api from "../api";
import { pokeJobs } from "../jobs";
import {
  busy,
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
import type { SourceHistoryStore } from "./sourceHistory";
import type { DisplaySettings, Renderer } from "../viewer/renderer";

/** The generated shader pair, for the "Generated WGSL" dialog. */
export interface WgslPreview {
  preview: string;
  path: string;
}

export interface CompileCycleOptions {
  renderer: Renderer;
  /** Every run commits a snapshot, so undo lands on compiled states. */
  history: SourceHistoryStore;
  /** Mesh edges are fetched only while a mesh overlay is displayed. */
  display: Accessor<DisplaySettings>;
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

export function createCompileCycle(options: CompileCycleOptions): CompileCycle {
  const [wgsl, setWgsl] = createSignal<WgslPreview | null>(null);

  // The mesh-edge overlay is fetched lazily: /compile no longer computes it,
  // and this cache requests it only while a mesh display mode is on.
  const [compiledSource, setCompiledSource] = createSignal<string | null>(null);
  let meshRequestFor: string | null = null;

  // A viewer edit can land while an earlier compile is still running. Dropping
  // it would leave the patched source unrendered, so remember to run again.
  let rerunRequested = false;

  const run = async (): Promise<void> => {
    if (busy()) {
      rerunRequested = true;
      return;
    }
    setBusy(true);
    // A compile is the most common piece of real work in this app and the
    // one most likely to be waited on, so tell the job poller it has
    // something to watch; it stops by itself when the worker is done.
    pokeJobs();
    setStatus({ kind: "", text: "JAX compiling…" });
    setConsoleText("");
    const text = source();
    options.history.commit(text);
    try {
      const result = await api.compile(text);
      if (!result.ok) {
        setStatus({ kind: "error", text: "Compile failed" });
        setConsoleText(result.error ?? "Unknown compile error.");
        return;
      }
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
          !node ||
          (active.vertexIndex !== null && active.vertexIndex >= node.vertices.length);
        if (stale) setSelection(null);
      }
      setWgsl({ preview: result.preview_shader, path: result.path_shader });
      setViewerError("");
      // The renderer replaces this as soon as it draws a frame; setting it here
      // means the status still settles on a machine without WebGPU.
      setStatus({ kind: "ready", text: "Scene compiled" });
      await options.renderer.setShaders({
        preview: result.preview_shader,
        path: result.path_shader,
        present: result.present_shader,
        // The uniform contract, when the worker emitted one: with it the
        // renderer can tell a parameter edit from a topology edit and skip
        // the shader module and pipelines entirely for the former.
        program: result.program ?? null,
      });
    } catch (error) {
      setStatus({ kind: "error", text: "Compile failed" });
      setConsoleText(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
      if (rerunRequested) {
        rerunRequested = false;
        await run();
      }
    }
  };

  /**
   * Send one edit to the server, adopt the patched source, then rebuild.
   *
   * UI actions can arrive while the previous patch is compiling (constraint →
   * satisfy → extrude is a common sequence). Serialize them so every request
   * starts from the source produced by the preceding edit instead of racing
   * and letting the last network response discard another operation.
   */
  let patchQueue: Promise<void> = Promise.resolve();
  const performPatch = async (body: Record<string, unknown>) => {
    try {
      const result = await api.patch({ source: source(), ...body });
      if (!result.ok || !result.source) {
        setStatus({ kind: "error", text: result.error ?? "Edit failed" });
        return;
      }
      setSource(result.source);
      await run();
    } catch (error) {
      setStatus({
        kind: "error",
        text: error instanceof Error ? error.message : String(error),
      });
    }
  };
  const applyPatch = (body: Record<string, unknown>): Promise<void> => {
    const queued = patchQueue.then(() => performPatch(body));
    patchQueue = queued.catch(() => undefined);
    return queued;
  };

  /**
   * Adopt server-produced source exactly like a patch response.
   *
   * The optimizer is a patch layer too: a successful /api/optimize returns
   * the program with the optimized literals written back, and the app treats
   * it as one committed edit (history snapshot via run()).
   */
  const adoptSource = (text: string): Promise<void> => {
    const queued = patchQueue.then(async () => {
      setSource(text);
      await run();
    });
    patchQueue = queued.catch(() => undefined);
    return queued;
  };

  /**
   * Compile-and-render a transient program without committing it.
   *
   * The optimization replay player scrubs through parameter snapshots by
   * substituting literals client-side; each frame shows in the editor and the
   * viewport but never lands in the undo history — only the adopted final
   * source does. Construction/studies state is refreshed by the caller's
   * closing adoptSource, so this only swaps the shaders.
   */
  const ghostCompile = async (text: string): Promise<boolean> => {
    setSource(text);
    try {
      const result = await api.compile(text);
      if (!result.ok) return false;
      await options.renderer.setShaders({
        preview: result.preview_shader,
        path: result.path_shader,
        present: result.present_shader,
        program: result.program ?? null,
      });
      return true;
    } catch {
      return false;
    }
  };

  /**
   * Fetch mesh edges lazily: only while a mesh overlay is displayed, only for
   * the compiled program, and only once per compile (a "no mesh available"
   * answer is cached too, so the effect cannot loop on it).
   */
  createEffect(() => {
    const display = options.display();
    const wanted = display.showMeshEdges || display.showMeshWireframe;
    const compiled = compiledSource();
    if (!wanted || compiled === null || meshEdges() !== null) return;
    if (meshRequestFor === compiled) return;
    meshRequestFor = compiled;
    void api
      .mesh(compiled)
      .then((result) => {
        // A newer compile owns the cache now; drop the stale answer.
        if (compiledSource() !== compiled) return;
        if (result.ok) setMeshEdges(result.mesh_edges ?? null);
      })
      .catch(() => {
        // Missing mesh edges only dim an optional overlay; stay quiet.
      });
  });

  return { run, applyPatch, adoptSource, ghostCompile, wgsl };
}
