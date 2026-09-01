/**
 * Wires the panes together: the shell and nothing else.
 *
 * The loop that gives the viewer and the code parity —
 *   viewer edit → POST /patch → new source into the editor → POST /compile →
 *   new shaders + construction tree → viewer redraws
 * — lives in `shell/compileCycle.ts`. What is left here is the arrangement:
 * which panel occupies the dock in each editing mode, which callbacks each
 * one gets, and the two dialogs the shell owns. Anything that needs a name
 * and a paragraph of explanation belongs in `shell/`, not in this file.
 */

import { createEffect, createMemo, createSignal, onMount, Show } from "solid-js";
import * as api from "./api";
import { EditorPane } from "./components/EditorPane";
import { MaterialPanel } from "./components/MaterialPanel";
import { MenuBar } from "./components/MenuBar";
import { ObjectTree } from "./components/ObjectTree";
import { OptimizePanel } from "./components/OptimizePanel";
import { SimulatePanel } from "./components/SimulatePanel";
import { SketchPanel } from "./components/SketchPanel";
import { ToolRail } from "./components/ToolRail";
import { Toolbar } from "./components/Toolbar";
import { ViewCube } from "./components/ViewCube";
import { ViewerPane } from "./components/ViewerPane";
import { createCompileCycle } from "./shell/compileCycle";
import { createPaneLayout } from "./shell/paneLayout";
import { createPatchOperations } from "./shell/patchOperations";
import { createRenderState } from "./shell/renderState";
import { createShellShortcuts } from "./shell/shellShortcuts";
import { createSourceActions, createSourceHistory } from "./shell/sourceHistory";
import {
  cameraAngles,
  editingMode,
  gizmoMode,
  reactToSelectionForMode,
  panels,
  setCameraAngles,
  nodeById,
  selection,
  setConsoleText,
  setPanelVisible,
  setSelection,
  setSource,
  reportViewerError,
  setStatus,
} from "./state";
import { Renderer } from "./viewer/renderer";

export function App() {
  const [wgslOpen, setWgslOpen] = createSignal(false);
  const [example, setExample] = createSignal("");

  const renderer = new Renderer({
    onStatus: (kind, text) => setStatus({ kind, text }),
    onError: (message) => reportViewerError(message),
  });

  const render = createRenderState(renderer);
  const layout = createPaneLayout();
  const history = createSourceHistory();
  const compile = createCompileCycle({
    renderer,
    history,
    display: render.display,
  });
  const ops = createPatchOperations(compile.applyPatch);
  const actions = createSourceActions({
    history,
    run: compile.run,
    example,
  });

  createShellShortcuts({
    undo: actions.undo,
    redo: actions.redo,
    wgslOpen,
    closeWgsl: () => setWgslOpen(false),
  });

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

  /** Character span of the selected vertex's literal, for the editor. */
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

  onMount(async () => {
    try {
      const session = await api.startSession();
      setExample(session.example);
      setSource(session.example);
      await compile.run();
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
        canUndo={history.canUndo()}
        canRedo={history.canRedo()}
        onUndo={actions.undo}
        onRedo={actions.redo}
        onNew={actions.newScene}
        onAdoptScene={actions.adoptScene}
      />
      <Toolbar
        onRun={() => void compile.run()}
        onReset={() => {
          setSource(example());
          setSelection(null);
          void compile.run();
        }}
        onShowWgsl={() => setWgslOpen(true)}
        wgslReady={compile.wgsl() !== null}
        render={{
          display: render.display(),
          presets: render.presets(),
          selectedPreset: render.selectedPreset(),
          pathTracing: render.pathTracing(),
          quality: render.quality(),
          onChange: render.applyDisplay,
          onQualityChange: render.applyQuality,
          onPresetActivate: render.activateRenderPreset,
          onPresetSave: render.saveRenderPreset,
          onPresetReset: render.resetRenderPreset,
          onPathTracingChange: render.applyPathTracing,
        }}
      />

      <main
        class="panes"
        classList={{
          resizing: layout.resizing(),
          "editor-collapsed": !panels().editor,
        }}
        style={{ "grid-template-columns": layout.paneColumns() }}
        ref={layout.setPanesElement}
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
            onRun={() => void compile.run()}
            onCollapse={() => setPanelVisible("editor", false)}
          />
        </Show>
        <div
          class="pane-splitter"
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize the editor pane"
          title="Drag to resize · double-click to reset"
          onPointerDown={layout.onSplitterDown}
          onDblClick={() => layout.persistEditorWidth(null)}
          data-testid="pane-splitter"
        />
        <ViewerPane
          renderer={renderer}
          display={render.display()}
          onPatch={ops.patch}
          onSetValue={ops.setValue}
          onAddPrimitive={ops.addPrimitive}
          onAddSketch={ops.addSketch}
          onAddConstraint={ops.addConstraint}
          onAddLoft={ops.addLoft}
          onDeleteObject={ops.deleteObject}
          onAssignMaterial={ops.assignMaterial}
          overlay={
            <>
              <ToolRail
                onDelete={() => {
                  const active = selection();
                  const node = active && nodeById(active.nodeId);
                  if (!node?.editable || node.line === null) return;
                  if (active!.vertexIndex !== null) {
                    void ops.patch("delete_vertex", node.line, active!.vertexIndex);
                  } else {
                    void ops.deleteObject(node.line);
                  }
                  setSelection(null);
                }}
                onExtrude={ops.extrudeSelection}
                onRevolve={ops.revolveSelection}
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
                      void ops.addConstraint(
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
                        void ops.solveSketch(node.line, method, iterations);
                      }
                    }}
                    onExtrude={ops.extrudeSelection}
                    onRevolve={ops.revolveSelection}
                    onDeleteConstraint={(line, index) =>
                      void ops.deleteConstraint(line, index)
                    }
                    onSetConstraintValue={(line, index, value) =>
                      void ops.setConstraintValue(line, index, value)
                    }
                  />
                </Show>
                <Show when={editingMode() === "model" && panels().materials}>
                  <MaterialPanel
                    onCreate={ops.addMaterial}
                    onSetValue={(line, argument, value) =>
                      ops.setValue(line, "Material", argument, value)
                    }
                  />
                </Show>
                {/* Optimize: run declared optimizations over the free
                    parameters through the differentiable path. Lives in
                    Model mode next to the object tree — it edits the same
                    design parameters the modeling tools do. */}
                <Show when={editingMode() === "model"}>
                  <OptimizePanel
                    onPatch={compile.applyPatch}
                    onAdoptSource={compile.adoptSource}
                    onGhostCompile={compile.ghostCompile}
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
                      onPatch={compile.applyPatch}
                      onAdoptSource={compile.adoptSource}
                      onGhostCompile={compile.ghostCompile}
                    />
                  </div>
                </Show>
              </div>
              <ViewCube
                yaw={cameraAngles().yaw}
                pitch={cameraAngles().pitch}
                projection={render.display().projection}
                active={render.viewPreset()}
                onPreset={render.applyPreset}
                onProjection={(projection) => render.applyDisplay({ projection })}
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

      <Show when={wgslOpen() && compile.wgsl()}>
        <div class="dialog-backdrop" onClick={() => setWgslOpen(false)}>
          <div class="dialog" onClick={(event) => event.stopPropagation()}>
            <header>
              <span>Generated WGSL</span>
              <button type="button" onClick={() => setWgslOpen(false)}>
                Close
              </button>
            </header>
            <pre>{render.pathTracing() ? compile.wgsl()!.path : compile.wgsl()!.preview}</pre>
          </div>
        </div>
      </Show>
    </div>
  );
}
