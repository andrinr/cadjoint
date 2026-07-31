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
import { ToolRail } from "./components/ToolRail";
import { Toolbar } from "./components/Toolbar";
import { ViewCube } from "./components/ViewCube";
import { ViewerPane } from "./components/ViewerPane";
import {
  busy,
  cameraAngles,
  gizmoMode,
  setCameraAngles,
  nodeById,
  selection,
  setBusy,
  setConsoleText,
  setDirty,
  setNodes,
  setSelection,
  setSource,
  reportViewerError,
  setStatus,
  setViewerError,
  source,
} from "./state";
import {
  DEFAULT_DISPLAY,
  QUALITY_PRESETS,
  Renderer,
  type DisplaySettings,
} from "./viewer/renderer";

export function App() {
  const [pathTracing, setPathTracing] = createSignal(false);
  const [quality, setQuality] = createSignal("high");
  const [wgsl, setWgsl] = createSignal<{ preview: string; path: string } | null>(null);
  const [showWgsl, setShowWgsl] = createSignal(false);
  const [example, setExample] = createSignal("");
  const [display, setDisplay] = createSignal<DisplaySettings>({ ...DEFAULT_DISPLAY });
  const [viewPreset, setViewPreset] = createSignal("iso");

  const renderer = new Renderer({
    onStatus: (kind, text) => setStatus({ kind, text }),
    onError: (message) => reportViewerError(message),
  });

  /** Push display settings to the renderer and redraw. */
  const applyDisplay = (patch: Partial<DisplaySettings>) => {
    const next = { ...display(), ...patch };
    setDisplay(next);
    renderer.display = next;
    renderer.invalidate();
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
        return;
      }
      setDirty(false);
      setConsoleText(result.output ?? "");
      setNodes(result.construction ?? []);
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

  /** Send one edit to the server, adopt the patched source, then rebuild. */
  const applyPatch = async (body: Record<string, unknown>) => {
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

  const patch = (
    op: "set_vertex" | "insert_vertex" | "delete_vertex",
    line: number,
    index: number,
    xy?: [number, number],
  ) => applyPatch({ op, line, index, xy });

  const setValue = (line: number, name: string, argument: string, value: number[]) =>
    applyPatch({ op: "set_value", line, name, argument, value });

  const deleteObject = (line: number) => applyPatch({ op: "delete_object", line });

  const addPrimitive = (
    kind: string,
    position: [number, number, number],
    dimensions: Record<string, number | number[]>,
  ) => applyPatch({ op: "add_primitive", kind, position, dimensions });

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
        onDisplayChange={applyDisplay}
        onRun={() => void run()}
        onReset={() => {
          setSource(example());
          setSelection(null);
          void run();
        }}
        onToggleTrace={() => {
          const next = !pathTracing();
          setPathTracing(next);
          renderer.pathTracing = next;
          renderer.invalidate();
        }}
        onQualityChange={(key) => {
          setQuality(key);
          renderer.quality = QUALITY_PRESETS[key];
          renderer.invalidate();
        }}
        onShowWgsl={() => setShowWgsl(true)}
        pathTracing={pathTracing()}
        quality={quality()}
        wgslReady={wgsl() !== null}
      />

      <main class="panes">
        <EditorPane highlight={highlight()} onRun={() => void run()} />
        <ViewerPane
          renderer={renderer}
          onPatch={patch}
          onSetValue={setValue}
          onAddPrimitive={addPrimitive}
          onDeleteObject={deleteObject}
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
