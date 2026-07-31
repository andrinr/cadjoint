/**
 * Wires the panes together and owns the compile/patch cycle.
 *
 * The loop that gives the viewer and the code parity:
 *   viewer edit → POST /patch → new source into the editor → POST /compile →
 *   new shaders + construction tree → viewer redraws.
 */

import { createEffect, createMemo, createSignal, onMount, Show } from "solid-js";
import * as api from "./api";
import { EditorPane } from "./components/EditorPane";
import { MaterialPanel } from "./components/MaterialPanel";
import { SketchPanel } from "./components/SketchPanel";
import { ToolRail } from "./components/ToolRail";
import { Toolbar } from "./components/Toolbar";
import { ViewCube } from "./components/ViewCube";
import { ViewerPane } from "./components/ViewerPane";
import {
  DEFAULT_RENDER_PRESETS,
  loadRenderPresetState,
  persistRenderPresetState,
  type RenderPresetId,
} from "./renderPresets";
import {
  busy,
  cameraAngles,
  gizmoMode,
  setCameraAngles,
  nodeById,
  selection,
  setBusy,
  setConsoleText,
  setDifferentiabilityDemo,
  setDirty,
  setMaterials,
  setNodes,
  setRelations,
  setSolverRuns,
  setSelection,
  setSource,
  reportViewerError,
  setStatus,
  setViewerError,
  source,
} from "./state";
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
    try {
      const result = await api.compile(source());
      if (!result.ok) {
        setStatus({ kind: "error", text: "Compile failed" });
        setConsoleText(result.error ?? "Unknown compile error.");
        setDifferentiabilityDemo(null);
        return;
      }
      setDirty(false);
      setConsoleText(result.output ?? "");
      setNodes(result.construction ?? []);
      setRelations(result.relations ?? []);
      setSolverRuns(result.solver_runs ?? []);
      setDifferentiabilityDemo(result.differentiability ?? null);
      setMaterials(result.materials ?? []);
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
      setDifferentiabilityDemo(null);
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
    kind: "fixed" | "distance",
    indices: number[],
    value: number | number[],
  ) => applyPatch({ op: "add_constraint", line, kind, indices, value });

  const addExtrusion = (line: number) =>
    applyPatch({ op: "add_extrusion", line, depth: 0.5 });

  const solveSketch = (
    line: number,
    method: "newton" | "adam" | "sgd",
    iterations: number,
  ) => applyPatch({ op: "solve_sketch", line, method, iterations });

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
    <div class="app">
      <Toolbar
        display={display()}
        renderPresets={renderPresets()}
        selectedRenderPreset={selectedRenderPreset()}
        onDisplayChange={applyDisplay}
        onRenderPresetActivate={activateRenderPreset}
        onRenderPresetSave={saveRenderPreset}
        onRenderPresetReset={resetRenderPreset}
        onPathTracingChange={applyPathTracing}
        onRun={() => void run()}
        onReset={() => {
          setSource(example());
          setSelection(null);
          void run();
        }}
        onQualityChange={applyQuality}
        onShowWgsl={() => setShowWgsl(true)}
        pathTracing={pathTracing()}
        quality={quality()}
        wgslReady={wgsl() !== null}
      />

      <main class="panes">
        <EditorPane highlight={highlight()} onRun={() => void run()} />
        <ViewerPane
          renderer={renderer}
          display={display()}
          onPatch={patch}
          onSetValue={setValue}
          onAddPrimitive={addPrimitive}
          onAddSketch={addSketch}
          onAddConstraint={addConstraint}
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
              />
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
                onExtrude={() => {
                  const active = selection();
                  const node = active && nodeById(active.nodeId);
                  if (node?.kind === "profile" && node.line !== null) {
                    void addExtrusion(node.line);
                  }
                }}
              />
              <MaterialPanel
                onCreate={addMaterial}
                onSetValue={(line, argument, value) =>
                  setValue(line, "Material", argument, value)
                }
              />
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
