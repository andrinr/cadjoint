/**
 * Wires the panes together: the shell and nothing else.
 *
 * The loop that gives the viewer and the code parity —
 *   viewer edit → POST /patch → new source into the editor → POST /compile →
 *   new shaders + construction tree → viewer redraws
 * — lives in `shell/compileCycle.ts`. What is left here is the arrangement:
 * which Solid tree each window holds, which callbacks each one gets, and the
 * two dialogs the shell owns. The *placement* of those windows — docked,
 * tabbed, floating, parked, per mode, remembered across reloads — belongs to
 * `windows/`, not to this file.
 */

import { createMemo, createEffect, createSignal, onMount, Show } from "solid-js";
import * as api from "./api";
import { EditorPane } from "./components/EditorPane";
import { MaterialPanel } from "./components/MaterialPanel";
import { MenuBar } from "./components/MenuBar";
import { ObjectTree } from "./components/ObjectTree";
import { OptimizePanel } from "./components/OptimizePanel";
import { ScenesPanel } from "./components/ScenesPanel";
import {
  MeshesWindow,
  ResultsWindow,
  StudiesWindow,
} from "./components/simulate/SimWindows";
import { createSimulateController } from "./components/simulate/controller";
import { SketchPanel } from "./components/SketchPanel";
import { ToolRail } from "./components/ToolRail";
import { Toolbar } from "./components/Toolbar";
import { ViewCube } from "./components/ViewCube";
import { ViewerPane } from "./components/ViewerPane";
import { createCompileCycle } from "./shell/compileCycle";
import { createPatchOperations } from "./shell/patchOperations";
import { createRenderState } from "./shell/renderState";
import { createShellShortcuts } from "./shell/shellShortcuts";
import { createSourceActions, createSourceHistory } from "./shell/sourceHistory";
import { WindowLayout } from "./windows/WindowLayout";
import type { WindowId } from "./windows/panels";
import { focusSpan } from "./editorFocus";
import { referenceFor, type FaceTarget } from "./faces";
import {
  cameraAngles,
  editingMode,
  gizmoMode,
  reactToSelectionForMode,
  setCameraAngles,
  nodeById,
  nodes,
  profiles,
  selection,
  setConsoleText,
  setPanelVisible,
  setSelection,
  setSource,
  reportViewerError,
  setStatus,
} from "./state";
import { transformState, vertexState, type BindingState } from "./viewer/dragBinding";
import { Renderer, type ShaderStats } from "./viewer/renderer";

/** One draggable value, and whether dragging it writes a buffer or a module. */
export interface HandleBinding {
  nodeId: string;
  /** The sketch vertex's index, or the gizmo argument's name. */
  handle: string;
  /** The parameter the source names for it, if it names one. */
  parameter: string | null;
  state: BindingState;
}

declare global {
  interface Window {
    __cadjointShaders?: () => ShaderStats;
    __cadjointSetParameters?: (
      overrides: Record<string, readonly number[]> | null,
    ) => boolean;
    __cadjointBindings?: () => HandleBinding[];
  }
}

export function App() {
  const [wgslOpen, setWgslOpen] = createSignal(false);
  const [example, setExample] = createSignal("");

  const renderer = new Renderer({
    onStatus: (kind, text) => setStatus({ kind, text }),
    onError: (message) => reportViewerError(message),
  });
  // The shader path's counters, for the end-to-end tests and the console:
  // "a drag rebuilds no pipelines" is a negative claim, and only a counter
  // can check one. `__cadjointSetParameters` drives the same frame-rate
  // path a handle drag uses, which is the thing that claim is about.
  if (typeof window !== "undefined") {
    window.__cadjointShaders = () => renderer.shaderStats;
    window.__cadjointSetParameters = (overrides) =>
      renderer.setParameterOverrides(overrides);
    // The same classification the overlay draws each handle with, published
    // so a test can assert the mark and the path agree: a filled handle that
    // recompiled, or a hollow one that did not, is the mark lying.
    window.__cadjointBindings = () =>
      nodes().flatMap((node) => {
        const program = renderer.parameterProgram;
        const vertices = node.vertices.map((vertex, index) => ({
          nodeId: node.id,
          handle: `vertex[${index}]`,
          parameter: vertex.binding?.name ?? null,
          state: vertexState(vertex, program),
        }));
        const transform = node.transform;
        const args = transform
          ? ["position", "rotation", ...Object.keys(transform.dimensions)]
          : [];
        return [
          ...vertices,
          ...args.map((argument) => ({
            nodeId: node.id,
            handle: `gizmo ${argument}`,
            parameter: transform?.bindings?.[argument]?.[0]?.name ?? null,
            state: transformState(transform, argument, program),
          })),
        ];
      });
  }

  const render = createRenderState(renderer);
  const history = createSourceHistory();
  const compile = createCompileCycle({
    renderer,
    history,
    display: render.display,
  });
  const ops = createPatchOperations(compile.applyPatch);
  /**
   * One simulation controller for four windows.
   *
   * Meshes, Studies and Results each get their own Solid root from the dock,
   * so the state they share — the last solve, the field on screen, the BC
   * builder, the job references that outlive a mode switch — cannot live in
   * any one of them. It lives here, in the shell that outlives all of them,
   * and each window registers itself with it while it is mounted.
   */
  const sim = createSimulateController({
    renderer,
    onPatch: compile.applyPatch,
    onAdoptSource: compile.adoptSource,
    onGhostCompile: compile.ghostCompile,
  });
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
  // A vertex reveals its own literal; a whole object reveals the statement
  // that declares it. The rule itself lives in `editorFocus.ts`.
  const highlight = createMemo(() => {
    const active = selection();
    return active ? focusSpan(nodeById(active.nodeId), active.vertexIndex) : null;
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

  /**
   * Plant a sketch on a picked face.
   *
   * Two patches when there is no sketch yet, and the order matters: the new
   * sketch is created first, then the face reference is derived from the
   * *recompiled* tree. Inserting a statement renumbers every line after it,
   * and a reference names its owner by line — resolving before the insert
   * would write a plane that points at whatever moved into that slot.
   */
  const sketchOnFace = async (target: FaceTarget, sketchLine: number | null) => {
    let line = sketchLine;
    if (line === null) {
      await ops.addSketch([0, 0, 0]);
      const created = profiles();
      const newest = created[created.length - 1];
      if (!newest || newest.line === null) return;
      line = newest.line;
      setSelection({ nodeId: newest.id, vertexIndex: null });
    }
    const reference = referenceFor(nodes(), target);
    if (!reference) {
      setStatus({ kind: "error", text: "That surface has no reference the source can name." });
      return;
    }
    await ops.setSketchPlane(line, reference);
  };

  /**
   * One window's contents.
   *
   * Each of these is mounted into its own Solid root by the dock, so a window
   * that is closed takes its effects with it. Nothing here knows where its
   * window is: docked, tabbed behind another, floating or parked all render
   * the same tree.
   */
  const renderWindow = (id: WindowId) => {
    switch (id) {
      case "viewport":
        return (
          <ViewerPane
            renderer={renderer}
            display={render.display()}
            onPatch={ops.patch}
            onSetValue={ops.setValue}
            onAddPrimitive={ops.addPrimitive}
            onAddSketch={ops.addSketch}
            onSketchOnFace={sketchOnFace}
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
                <ViewCube
                  yaw={cameraAngles().yaw}
                  pitch={cameraAngles().pitch}
                  projection={render.display().projection}
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
        );

      case "editor":
        return (
          <EditorPane
            highlight={highlight()}
            onRun={() => void compile.run()}
            onCollapse={() => setPanelVisible("editor", false)}
          />
        );

      case "objects":
        return <ObjectTree />;

      case "materials":
        return (
          <MaterialPanel
            onCreate={ops.addMaterial}
            onSetValue={(line, argument, value) =>
              ops.setValue(line, "Material", argument, value)
            }
            onAdoptSource={compile.adoptSource}
          />
        );

      case "sketch":
        return (
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
            onDeleteConstraint={(line, index) => void ops.deleteConstraint(line, index)}
            onSetConstraintValue={(line, index, value) =>
              void ops.setConstraintValue(line, index, value)
            }
          />
        );

      // Optimize: run declared optimizations over the free parameters through
      // the differentiable path. It edits the same design parameters the
      // modeling tools do, so Model mode's default desk includes it.
      case "optimize":
        return (
          <OptimizePanel
            onPatch={compile.applyPatch}
            onAdoptSource={compile.adoptSource}
            onGhostCompile={compile.ghostCompile}
          />
        );

      // The three windows the Simulate desk arranges. Each is a view over
      // the one controller above; none of them owns the simulation.
      case "meshes":
        return <MeshesWindow sim={sim} />;

      case "studies":
        return <StudiesWindow sim={sim} />;

      case "results":
        return <ResultsWindow sim={sim} />;

      // The document browser: what is saved beside this one, and what is in
      // each of them, without running any of them.
      case "scenes":
        return <ScenesPanel onOpen={actions.adoptScene} />;
    }
  };

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

      <WindowLayout renderWindow={renderWindow} />

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
