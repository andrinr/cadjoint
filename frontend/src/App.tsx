/**
 * Wires the panes together and owns the compile/patch cycle.
 *
 * The loop that gives the viewer and the code parity:
 *   viewer edit → POST /patch → new source into the editor → POST /compile →
 *   new shaders + construction tree → viewer redraws.
 */

import { createMemo, createSignal, onMount, Show } from "solid-js";
import * as api from "./api";
import { EditorPane } from "./components/EditorPane";
import { Toolbar } from "./components/Toolbar";
import { ViewerPane } from "./components/ViewerPane";
import {
  busy,
  profileById,
  selection,
  setBusy,
  setConsoleText,
  setDirty,
  setProfiles,
  setSelection,
  setSource,
  setStatus,
  setViewerError,
  source,
} from "./state";
import { QUALITY_PRESETS, Renderer } from "./viewer/renderer";

export function App() {
  const [pathTracing, setPathTracing] = createSignal(false);
  const [quality, setQuality] = createSignal("high");
  const [wgsl, setWgsl] = createSignal<{ preview: string; path: string } | null>(null);
  const [showWgsl, setShowWgsl] = createSignal(false);
  const [example, setExample] = createSignal("");

  const renderer = new Renderer({
    onStatus: (kind, text) => setStatus({ kind, text }),
    onError: (message) => setViewerError(message),
  });

  /** Character span of the selected vertex's literal, for the editor. */
  const highlight = createMemo(() => {
    const active = selection();
    if (!active) return null;
    const span = profileById(active.profileId)?.vertices[active.vertexIndex]?.span;
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
      setProfiles(result.construction ?? []);
      // Drop a selection that no longer exists in the rebuilt sketch.
      const active = selection();
      if (active) {
        const profile = (result.construction ?? []).find((item) => item.id === active.profileId);
        if (!profile || active.vertexIndex >= profile.vertices.length) setSelection(null);
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

  /** Apply one viewer edit to the program text, then rebuild. */
  const patch = async (
    op: "set_vertex" | "insert_vertex" | "delete_vertex",
    line: number,
    index: number,
    xy?: [number, number],
  ) => {
    try {
      const result = await api.patch(source(), op, line, index, xy);
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
        <ViewerPane renderer={renderer} onPatch={patch} />
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
