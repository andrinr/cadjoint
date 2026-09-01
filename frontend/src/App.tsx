/**
 * Wires the panes together and owns the compile/patch cycle.
 *
 * The loop that gives the viewer and the code parity:
 *   viewer edit → POST /patch → new source into the editor → POST /compile →
 *   new shaders + construction tree → viewer redraws.
 */

import { createEffect, createMemo, createSignal, onCleanup, onMount, Show } from "solid-js";
import * as api from "./api";
import { EditorPane } from "./components/EditorPane";
import { MaterialPanel } from "./components/MaterialPanel";
import { MenuBar } from "./components/MenuBar";
import { ObjectTree } from "./components/ObjectTree";
import { OptimizePanel } from "./components/OptimizePanel";
import { RenderPanel } from "./components/RenderPanel";
import { SimulatePanel } from "./components/SimulatePanel";
import { SketchPanel } from "./components/SketchPanel";
import { ToolRail } from "./components/ToolRail";
import { Toolbar } from "./components/Toolbar";
import { ViewCube } from "./components/ViewCube";
import { ViewerPane } from "./components/ViewerPane";
import { SourceHistory } from "./history";
import {
  DEFAULT_RENDER_PRESETS,
  loadRenderPresetState,
  persistRenderPresetState,
  type RenderPresetId,
} from "./renderPresets";
import {
  busy,
  cameraAngles,
  dirty,
  editingMode,
  gizmoMode,
  reactToSelectionForMode,
  meshEdges,
  panels,
  setCameraAngles,
  nodeById,
  selection,
  setBusy,
  setConsoleText,
  setDirty,
  setMaterials,
  setMeshEdges,
  setNodes,
  setOptimizations,
  setPanelVisible,
  setRelations,
  setSceneName,
  setSimMeshes,
  setSolverRuns,
  setStudies,
  setSelection,
  setSource,
  reportViewerError,
  setStatus,
  setViewerError,
  source,
} from "./state";
import type { ConstraintKind } from "./types";
import {
  QUALITY_PRESETS,
  Renderer,
  type DisplaySettings,
} from "./viewer/renderer";

export function App() {
  const initialPresetState = loadRenderPresetState();
  const initialPreset =
    initialPresetState.presets.find(
      (preset) => preset.id === initialPresetState.activeId,
    ) ?? initialPresetState.presets[0];
  const [pathTracing, setPathTracing] = createSignal(initialPreset.pathTracing);
  const [quality, setQuality] = createSignal(initialPreset.quality);
  const [wgsl, setWgsl] = createSignal<{ preview: string; path: string } | null>(null);
  const [showWgsl, setShowWgsl] = createSignal(false);
  const [example, setExample] = createSignal("");
  const [display, setDisplay] = createSignal<DisplaySettings>({
    ...initialPreset.display,
  });
  const [renderPresets, setRenderPresets] = createSignal(
    initialPresetState.presets,
  );
  const [selectedRenderPreset, setSelectedRenderPreset] =
    createSignal<RenderPresetId>(initialPresetState.activeId);
  const [viewPreset, setViewPreset] = createSignal("iso");

  // Undo/redo across source edits. Snapshots are committed on every run and
  // every viewer patch (both funnel through run()); typing inside the editor
  // keeps CodeMirror's native history until the next run commits it here.
  const history = new SourceHistory();
  const [historyVersion, setHistoryVersion] = createSignal(0);
  const commitHistory = (text: string) => {
    history.commit(text);
    setHistoryVersion((version) => version + 1);
  };
  const canUndo = createMemo(() => (historyVersion(), history.canUndo()));
  const canRedo = createMemo(() => (historyVersion(), history.canRedo()));

  // Editor/viewport splitter. Null means "the stylesheet default" until the
  // user drags; the value persists like the other layout preferences.
  const EDITOR_WIDTH_KEY = "cadjoint.editorWidth.v1";
  const EDITOR_MIN = 280;
  const VIEWER_MIN = 360;
  const storedWidth = Number(localStorage.getItem(EDITOR_WIDTH_KEY));
  const [editorWidth, setEditorWidth] = createSignal<number | null>(
    Number.isFinite(storedWidth) && storedWidth >= EDITOR_MIN ? storedWidth : null,
  );
  const [resizing, setResizing] = createSignal(false);
  let panesElement: HTMLElement | undefined;

  const persistEditorWidth = (width: number | null) => {
    setEditorWidth(width);
    try {
      if (width === null) localStorage.removeItem(EDITOR_WIDTH_KEY);
      else localStorage.setItem(EDITOR_WIDTH_KEY, String(Math.round(width)));
    } catch {
      // Layout persistence is best-effort only.
    }
  };

  const paneColumns = () => {
    if (!panels().editor) return "44px 0px 1fr";
    const width = editorWidth();
    const editor =
      width === null
        ? "minmax(320px, 42%)"
        : `min(${Math.round(width)}px, calc(100% - ${VIEWER_MIN + 6}px))`;
    return `${editor} 6px 1fr`;
  };

  const onSplitterDown = (event: PointerEvent) => {
    if (!panels().editor || !panesElement) return;
    const splitter = event.currentTarget as HTMLElement;
    splitter.setPointerCapture(event.pointerId);
    setResizing(true);
    const bounds = panesElement.getBoundingClientRect();
    const move = (moveEvent: PointerEvent) => {
      const width = Math.min(
        Math.max(moveEvent.clientX - bounds.left, EDITOR_MIN),
        bounds.width - VIEWER_MIN - 6,
      );
      setEditorWidth(width);
    };
    const up = () => {
      splitter.removeEventListener("pointermove", move);
      splitter.removeEventListener("pointerup", up);
      splitter.removeEventListener("pointercancel", up);
      setResizing(false);
      persistEditorWidth(editorWidth());
    };
    splitter.addEventListener("pointermove", move);
    splitter.addEventListener("pointerup", up);
    splitter.addEventListener("pointercancel", up);
  };

  // The mesh-edge overlay is fetched lazily: /compile no longer computes it,
  // and this cache requests it only while a mesh display mode is on.
  const [compiledSource, setCompiledSource] = createSignal<string | null>(null);
  let meshRequestFor: string | null = null;

  const renderer = new Renderer({
    onStatus: (kind, text) => setStatus({ kind, text }),
    onError: (message) => reportViewerError(message),
  });
  renderer.display = { ...initialPreset.display };
  renderer.quality = QUALITY_PRESETS[initialPreset.quality];
  renderer.pathTracing = initialPreset.pathTracing;

  /** Push display settings to the renderer and redraw. */
  const applyDisplay = (patch: Partial<DisplaySettings>) => {
    const next = { ...display(), ...patch };
    setDisplay(next);
    renderer.display = next;
    renderer.invalidate();
  };

  const applyQuality = (key: string) => {
    const preset = QUALITY_PRESETS[key];
    if (!preset) return;
    setQuality(key);
    renderer.quality = preset;
    renderer.invalidate();
  };

  const applyPathTracing = (enabled: boolean) => {
    setPathTracing(enabled);
    renderer.pathTracing = enabled;
    renderer.invalidate();
  };

  const activateRenderPreset = (id: RenderPresetId) => {
    const preset = renderPresets().find((item) => item.id === id);
    if (!preset) return;
    setSelectedRenderPreset(id);
    applyDisplay(preset.display);
    applyQuality(preset.quality);
    applyPathTracing(preset.pathTracing);
    persistRenderPresetState({ presets: renderPresets(), activeId: id });
  };

  const saveRenderPreset = (id: RenderPresetId) => {
    const next = renderPresets().map((preset) =>
      preset.id === id
        ? {
            ...preset,
            pathTracing: pathTracing(),
            quality: quality(),
            display: { ...display() },
          }
        : preset,
    );
    setRenderPresets(next);
    setSelectedRenderPreset(id);
    persistRenderPresetState({ presets: next, activeId: id });
  };

  const resetRenderPreset = (id: RenderPresetId) => {
    const original = DEFAULT_RENDER_PRESETS.find((preset) => preset.id === id);
    if (!original) return;
    const next = renderPresets().map((preset) =>
      preset.id === id
        ? { ...original, display: { ...original.display } }
        : preset,
    );
    setRenderPresets(next);
    setSelectedRenderPreset(id);
    applyDisplay(original.display);
    applyQuality(original.quality);
    applyPathTracing(original.pathTracing);
    persistRenderPresetState({ presets: next, activeId: id });
  };

  const applyPreset = (key: string) => {
    setViewPreset(key);
    renderer.applyViewPreset(key);
    setDisplay({ ...renderer.display });
    setCameraAngles({ yaw: renderer.camera.yaw, pitch: renderer.camera.pitch });
  };

  /** Character span of the selected vertex's literal, for the editor. */
  // The renderer needs the gizmo mode to know which handles to draw.
  createEffect(() => {
    renderer.gizmoMode = gizmoMode();
    renderer.invalidate();
  });

  // Selecting a sketch profile auto-enters sketch mode (cancelable — the
  // rule itself lives in editingMode.ts and remembers manual exits).
  createEffect(() => {
    selection();
    reactToSelectionForMode();
  });

  const highlight = createMemo(() => {
    const active = selection();
    if (!active) return null;
    const node = nodeById(active.nodeId);
    if (!node) return null;
    if (active.vertexIndex === null) {
      // A whole primitive highlights its position literal instead.
      const span = node.spans.position;
      return span ? { from: span[0], to: span[1] } : null;
    }
    const span = node.vertices[active.vertexIndex]?.span;
    return span ? { from: span[0], to: span[1] } : null;
  });

  // A viewer edit can land while an earlier compile is still running. Dropping
  // it would leave the patched source unrendered, so remember to run again.
  let rerunRequested = false;

  const run = async (): Promise<void> => {
    if (busy()) {
      rerunRequested = true;
      return;
    }
    setBusy(true);
    setStatus({ kind: "", text: "JAX compiling…" });
    setConsoleText("");
    const text = source();
    commitHistory(text);
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
      await renderer.setShaders({
        preview: result.preview_shader,
        path: result.path_shader,
        present: result.present_shader,
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
      await renderer.setShaders({
        preview: result.preview_shader,
        path: result.path_shader,
        present: result.present_shader,
      });
      return true;
    } catch {
      return false;
    }
  };

  const patch = (
    op: "set_vertex" | "insert_vertex" | "delete_vertex",
    line: number,
    index: number,
    xy?: [number, number],
  ) => applyPatch({ op, line, index, xy });

  const setValue = (
    line: number,
    name: string,
    argument: string,
    value: number | number[],
  ) =>
    applyPatch({ op: "set_value", line, name, argument, value });

  const deleteObject = (line: number) => applyPatch({ op: "delete_object", line });

  const addPrimitive = (
    kind: string,
    position: [number, number, number],
    dimensions: Record<string, number | number[]>,
  ) => applyPatch({ op: "add_primitive", kind, position, dimensions });

  const addMaterial = () =>
    applyPatch({
      op: "add_material",
      color: [0.32, 0.72, 0.86],
      roughness: 0.35,
      metallic: 0,
      opacity: 1,
      ior: 1.45,
      reflectivity: 0,
    });

  const assignMaterial = (line: number, material: string) =>
    applyPatch({ op: "assign_material", line, material });

  const addSketch = (origin: [number, number, number]) =>
    applyPatch({ op: "add_sketch", origin });

  const addConstraint = (
    line: number,
    kind: ConstraintKind,
    indices: number[],
    value?: number | number[],
  ) => applyPatch({ op: "add_constraint", line, kind, indices, value });

  const deleteConstraint = (line: number, index: number) =>
    applyPatch({ op: "delete_constraint", line, index });

  const setConstraintValue = (line: number, index: number, value: number) =>
    applyPatch({ op: "set_constraint_value", line, index, value });

  const addExtrusion = (line: number) =>
    applyPatch({ op: "add_extrusion", line, depth: 0.5 });

  const addRevolution = (line: number) =>
    applyPatch({ op: "add_revolution", line, offset: 0 });

  /** Extrude the selected sketch — shared by the rail and the sketch panel. */
  const extrudeSelection = () => {
    const active = selection();
    const node = active && nodeById(active.nodeId);
    if (node?.kind === "profile" && node.line !== null) {
      void addExtrusion(node.line);
    }
  };

  const revolveSelection = () => {
    const active = selection();
    const node = active && nodeById(active.nodeId);
    if (node?.kind === "profile" && node.line !== null) {
      void addRevolution(node.line);
    }
  };

  const addLoft = (lineA: number, lineB: number) =>
    applyPatch({ op: "add_loft", line_a: lineA, line_b: lineB, height: 1.0 });

  const solveSketch = (
    line: number,
    method: "newton" | "adam" | "sgd",
    iterations: number,
  ) => applyPatch({ op: "solve_sketch", line, method, iterations });

  /**
   * Fetch mesh edges lazily: only while a mesh overlay is displayed, only for
   * the compiled program, and only once per compile (a "no mesh available"
   * answer is cached too, so the effect cannot loop on it).
   */
  createEffect(() => {
    const wanted = display().showMeshEdges || display().showMeshWireframe;
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

  const undo = () => {
    // Capture typed-but-unrun edits first so redo can return to them.
    commitHistory(source());
    const previous = history.undo();
    setHistoryVersion((version) => version + 1);
    if (previous === null) return;
    setSource(previous);
    setSelection(null);
    void run();
  };

  const redo = () => {
    const next = history.redo();
    setHistoryVersion((version) => version + 1);
    if (next === null) return;
    setSource(next);
    setSelection(null);
    void run();
  };

  /** Reset to the starter example, asking before unsaved work is lost. */
  const newScene = () => {
    if (dirty() && !window.confirm("Discard unsaved changes to the current scene?")) {
      return;
    }
    setSceneName(null);
    setSource(example());
    setSelection(null);
    void run();
  };

  /** Adopt a scene file loaded through File → Open. */
  const adoptScene = (name: string, text: string) => {
    setSceneName(name);
    setSource(text);
    setSelection(null);
    void run();
  };

  onMount(() => {
    // Undo/redo shortcuts, kept away from the editor: while typing there,
    // CodeMirror's own history owns Ctrl/Cmd+Z. The menu items always work.
    const onKeyDown = (event: KeyboardEvent) => {
      if (!(event.metaKey || event.ctrlKey) || event.key.toLowerCase() !== "z") return;
      const target = document.activeElement;
      const typing =
        target &&
        (target.tagName === "TEXTAREA" ||
          target.tagName === "INPUT" ||
          target.closest(".cm-editor"));
      if (typing) return;
      event.preventDefault();
      if (event.shiftKey) redo();
      else undo();
    };
    window.addEventListener("keydown", onKeyDown);
    onCleanup(() => window.removeEventListener("keydown", onKeyDown));

    // Escape closes the WGSL dialog. Capture phase, so the viewer's global
    // Escape (clear selection, reset mode) does not also fire underneath.
    const onEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || !showWgsl()) return;
      event.stopPropagation();
      setShowWgsl(false);
    };
    document.addEventListener("keydown", onEscape, true);
    onCleanup(() => document.removeEventListener("keydown", onEscape, true));
  });

  onMount(async () => {
    try {
      const session = await api.startSession();
      setExample(session.example);
      setSource(session.example);
      await run();
    } catch (error) {
      setStatus({ kind: "error", text: "Could not reach the playground server." });
      setConsoleText(error instanceof Error ? error.message : String(error));
    }
  });

  return (
    // data-mode drives the per-mode accent variables in the stylesheet, so
    // the switcher, rail, dock, hint bar, and viewport border stay in step.
    <div class="app" data-mode={editingMode()}>
      <MenuBar
        canUndo={canUndo()}
        canRedo={canRedo()}
        onUndo={undo}
        onRedo={redo}
        onNew={newScene}
        onAdoptScene={adoptScene}
      />
      <Toolbar
        onRun={() => void run()}
        onReset={() => {
          setSource(example());
          setSelection(null);
          void run();
        }}
        onShowWgsl={() => setShowWgsl(true)}
        wgslReady={wgsl() !== null}
      />

      <main
        class="panes"
        classList={{ resizing: resizing(), "editor-collapsed": !panels().editor }}
        style={{ "grid-template-columns": paneColumns() }}
        ref={panesElement}
      >
        <Show
          when={panels().editor}
          fallback={
            <div class="editor-rail">
              <button
                type="button"
                title="Expand the editor"
                aria-label="Expand the editor"
                onClick={() => setPanelVisible("editor", true)}
                data-testid="editor-expand"
              >
                ⟩
              </button>
              <span class="editor-rail-label">scene.py</span>
            </div>
          }
        >
          <EditorPane
            highlight={highlight()}
            onRun={() => void run()}
            onCollapse={() => setPanelVisible("editor", false)}
          />
        </Show>
        <div
          class="pane-splitter"
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize the editor pane"
          title="Drag to resize · double-click to reset"
          onPointerDown={onSplitterDown}
          onDblClick={() => persistEditorWidth(null)}
          data-testid="pane-splitter"
        />
        <ViewerPane
          renderer={renderer}
          display={display()}
          onPatch={patch}
          onSetValue={setValue}
          onAddPrimitive={addPrimitive}
          onAddSketch={addSketch}
          onAddConstraint={addConstraint}
          onAddLoft={addLoft}
          onDeleteObject={deleteObject}
          onAssignMaterial={assignMaterial}
          overlay={
            <>
              <ToolRail
                onDelete={() => {
                  const active = selection();
                  const node = active && nodeById(active.nodeId);
                  if (!node?.editable || node.line === null) return;
                  if (active!.vertexIndex !== null) {
                    void patch("delete_vertex", node.line, active!.vertexIndex);
                  } else {
                    void deleteObject(node.line);
                  }
                  setSelection(null);
                }}
                onExtrude={extrudeSelection}
                onRevolve={revolveSelection}
              />
              {/* Right-side dock, scoped by editing mode: Model shows Objects
                  + Materials, Sketch shows Objects + Sketch properties,
                  Simulate and Render each own the column with their panel.
                  One column; sections share the height and scroll
                  internally. */}
              <div class="dock">
                <Show
                  when={
                    (editingMode() === "model" || editingMode() === "sketch") &&
                    panels().objectTree
                  }
                >
                  <ObjectTree />
                </Show>
                <Show when={editingMode() === "sketch" && panels().sketch}>
                  <SketchPanel
                    onFix={() => {
                      const active = selection();
                      const node = active && nodeById(active.nodeId);
                      if (
                        !node ||
                        node.kind !== "profile" ||
                        node.line === null ||
                        active!.vertexIndex === null
                      ) {
                        return;
                      }
                      const vertex = node.vertices[active!.vertexIndex];
                      void addConstraint(
                        node.line,
                        "fixed",
                        [active!.vertexIndex],
                        vertex.uv,
                      );
                    }}
                    onSolve={(method, iterations) => {
                      const active = selection();
                      const node = active && nodeById(active.nodeId);
                      if (node?.kind === "profile" && node.line !== null) {
                        void solveSketch(node.line, method, iterations);
                      }
                    }}
                    onExtrude={extrudeSelection}
                    onRevolve={revolveSelection}
                    onDeleteConstraint={(line, index) =>
                      void deleteConstraint(line, index)
                    }
                    onSetConstraintValue={(line, index, value) =>
                      void setConstraintValue(line, index, value)
                    }
                  />
                </Show>
                <Show when={editingMode() === "model" && panels().materials}>
                  <MaterialPanel
                    onCreate={addMaterial}
                    onSetValue={(line, argument, value) =>
                      setValue(line, "Material", argument, value)
                    }
                  />
                </Show>
                {/* Optimize: run declared optimizations over the free
                    parameters through the differentiable path. Lives in
                    Model mode next to the object tree — it edits the same
                    design parameters the modeling tools do. */}
                <Show when={editingMode() === "model"}>
                  <OptimizePanel
                    onPatch={applyPatch}
                    onAdoptSource={adoptSource}
                    onGhostCompile={ghostCompile}
                  />
                </Show>
                {/* Simulate-mode slot: shown by the mode system (switcher, M
                    cycling, Escape returns to model); the panel internals
                    belong to the FEM feature, which may expand/collapse and
                    drive the renderer freely inside it. */}
                <Show when={editingMode() === "simulate"}>
                  <div class="mode-simulate-slot" data-testid="mode-simulate">
                    <SimulatePanel
                      renderer={renderer}
                      onPatch={applyPatch}
                      onAdoptSource={adoptSource}
                      onGhostCompile={ghostCompile}
                    />
                  </div>
                </Show>
                {/* Render mode owns the dock with the full render settings —
                    the panel the eye-icon popover grew into. */}
                <Show when={editingMode() === "render"}>
                  <RenderPanel
                    display={display()}
                    presets={renderPresets()}
                    selectedPreset={selectedRenderPreset()}
                    pathTracing={pathTracing()}
                    quality={quality()}
                    onChange={applyDisplay}
                    onQualityChange={applyQuality}
                    onPresetActivate={activateRenderPreset}
                    onPresetSave={saveRenderPreset}
                    onPresetReset={resetRenderPreset}
                    onPathTracingChange={applyPathTracing}
                  />
                </Show>
              </div>
              <ViewCube
                yaw={cameraAngles().yaw}
                pitch={cameraAngles().pitch}
                projection={display().projection}
                active={viewPreset()}
                onPreset={applyPreset}
                onProjection={(projection) => applyDisplay({ projection })}
                onOrbit={(yaw, pitch) => {
                  renderer.camera = { ...renderer.camera, yaw, pitch };
                  setCameraAngles({ yaw, pitch });
                  renderer.invalidate();
                }}
              />
            </>
          }
        />
      </main>

      <Show when={showWgsl() && wgsl()}>
        <div class="dialog-backdrop" onClick={() => setShowWgsl(false)}>
          <div class="dialog" onClick={(event) => event.stopPropagation()}>
            <header>
              <span>Generated WGSL</span>
              <button type="button" onClick={() => setShowWgsl(false)}>
                Close
              </button>
            </header>
            <pre>{pathTracing() ? wgsl()!.path : wgsl()!.preview}</pre>
          </div>
        </div>
      </Show>
    </div>
  );
}
