/**
 * The shell's commands: what the menu bar and the keys can do to the program,
 * camera, tools and dock. Registered once by `App`; the menu bar registers
 * its own dialogs' commands. Each implementation is the same call the
 * matching button makes, so the two cannot diverge.
 */

import { createSignal, onCleanup, type Accessor } from "solid-js";
import { cancelJob, runningJobs } from "../jobs";
import { addMeshRequest } from "../meshes";
import { addStudyRequest } from "../studies";
import {
  busy,
  cameraAngles,
  cycleMode,
  dirty,
  editingMode,
  nodeById,
  selection,
  selectionMode,
  setCameraAngles,
  setEditingMode,
  setGizmoMode,
  setSelection,
  setSimProbe,
  studies,
  tool,
} from "../state";
import { CHIP_KINDS } from "../components/jobChip";
import { setRenderPopoverOpen } from "../components/viewer/RenderSettingsButton";
import { detentZoomCamera } from "../components/viewer/camera";
import { EDITING_MODES } from "../editingMode";
import { windowManager } from "../windows/manager";
import { WINDOW_DEFS, isWindowId, type WindowId } from "../windows/panels";
import type { SimulateController } from "../components/simulate/controller";
import type { DisplaySettings } from "../viewer/display";
import type { Renderer } from "../viewer/renderer";
import type { GizmoMode, SelectionMode, ToolMode } from "../types";
import { registerCommands, type CommandImpl } from "./commands";
import type { PatchOperations } from "./patchOperations";
import type { RenderState } from "./renderState";
import {
  CONSTRAINT_TOOLS,
  armCreateTool,
  constraintBlocked,
  fixBlocked,
  fixSelectedPoint,
  gizmoBlocked,
  setSelectKind,
  shapeBlocked,
  toggleConstraintTool,
  toggleLoft,
} from "./toolActions";
import { layoutGridOn, setLayoutGridOn } from "./layoutGrid";

export interface AppCommandDeps {
  renderer: Renderer;
  render: RenderState;
  ops: PatchOperations;
  sim: SimulateController;
  run: () => void;
  wgslReady: Accessor<boolean>;
  showWgsl: () => void;
  undo: () => void;
  redo: () => void;
  canUndo: Accessor<boolean>;
  canRedo: Accessor<boolean>;
  newScene: () => void;
  /** Delete the selection, as the rail's Delete and ⌫ do. */
  deleteSelection: () => void;
}

/**
 * The window the user last clicked or focused into, for Window › Float. The
 * dock does not publish one; presses in the menu bar are ignored.
 */
function trackFocusedWindow(): Accessor<WindowId | null> {
  const [focused, setFocused] = createSignal<WindowId | null>(null);
  const note = (event: Event) => {
    const target = event.target as HTMLElement | null;
    if (!target?.closest || target.closest(".menu-bar")) return;
    const body =
      target.closest<HTMLElement>(".win-body[data-window]") ??
      target.closest(".dv-groupview")?.querySelector<HTMLElement>(".win-body[data-window]");
    const id = body?.dataset.window;
    if (id && isWindowId(id)) setFocused(id);
  };
  document.addEventListener("pointerdown", note, true);
  document.addEventListener("focusin", note, true);
  onCleanup(() => {
    document.removeEventListener("pointerdown", note, true);
    document.removeEventListener("focusin", note, true);
  });
  return focused;
}

/** Stop the work a person is waiting on: the compile first, then a solve. */
function stopTarget(sim: SimulateController): string | null {
  const running = runningJobs();
  if (busy()) {
    const compile = running.find((job) => job.kind === "compile");
    if (compile?.id) return compile.id;
  }
  const solve = sim.solveJob();
  if (solve) return solve;
  return running.find((job) => job.id && CHIP_KINDS.has(job.kind))?.id ?? null;
}

export function createAppCommands(deps: AppCommandDeps): void {
  const { renderer, render, ops, sim } = deps;
  const focusedWindow = trackFocusedWindow();

  const display = () => render.display();
  const toggle = (
    on: () => boolean,
    value: (next: boolean) => Partial<DisplaySettings>,
  ): CommandImpl => ({
    run: () => render.applyDisplay(value(!on())),
    checked: on,
  });
  const inspect = (view: DisplaySettings["sdfView"]): CommandImpl => ({
    run: () => render.applyDisplay({ sdfView: display().sdfView === view ? "solid" : view }),
    checked: () => display().sdfView === view,
  });

  /** One detent of the grid: the Alt-wheel step, from a key or the menu. */
  const stepGrid = (direction: 1 | -1) => {
    renderer.camera = detentZoomCamera(renderer.camera, direction);
    // A fresh object, so everything that reprojects on a camera move redraws.
    setCameraAngles({ ...cameraAngles() });
    renderer.invalidate();
  };

  /** The study ⇧⌘↵ solves: the one whose card is being edited, else the first ready. */
  const solveTarget = () => {
    const all = studies();
    const building = sim.building();
    return (
      all.find((study) => study.index === building) ??
      all.find((study) => study.bcs.length > 0) ??
      all[0] ??
      null
    );
  };

  const selectKind = (kind: SelectionMode): CommandImpl => ({
    run: () => setSelectKind(kind),
    checked: () => tool() === "select" && selectionMode() === kind,
  });
  const create = (key: ToolMode): CommandImpl => ({
    run: () => armCreateTool(key),
    checked: () => tool() === key,
  });
  const gizmo = (mode: GizmoMode): CommandImpl => ({
    run: () => setGizmoMode(mode),
    // With nothing selected the key only sets which gizmo the next selection gets.
    blocked: () => (selection() === null ? null : gizmoBlocked(mode)),
  });
  const view = (key: string): CommandImpl => ({ run: () => render.applyPreset(key) });
  const study = (kind: "thermal" | "elastic" | "flow"): CommandImpl => ({
    run: () => void sim.patch(addStudyRequest(kind)),
  });

  const windows = Object.fromEntries(
    WINDOW_DEFS.filter((def) => !def.permanent).map((def): [string, CommandImpl] => [
      `window-${def.id}`,
      {
        run: () => windowManager()?.toggle(def.id),
        checked: () => windowManager()?.status(def.id) === "open",
        blocked: () => (windowManager() ? null : "The desk is still loading"),
      },
    ]),
  );

  const modes = Object.fromEntries(
    EDITING_MODES.map((mode): [string, CommandImpl] => [
      `mode-${mode}`,
      { run: () => setEditingMode(mode), checked: () => editingMode() === mode },
    ]),
  );

  const constraints = Object.fromEntries(
    CONSTRAINT_TOOLS.map((key): [string, CommandImpl] => [
      `constrain-${key}`,
      {
        run: () => toggleConstraintTool(key),
        blocked: constraintBlocked,
        checked: () => tool() === key,
      },
    ]),
  );

  const unregister = registerCommands({
    "file-new": { run: deps.newScene },

    "edit-undo": { run: deps.undo, blocked: () => (deps.canUndo() ? null : "Nothing to undo") },
    "edit-redo": { run: deps.redo, blocked: () => (deps.canRedo() ? null : "Nothing to redo") },
    "edit-delete": {
      run: deps.deleteSelection,
      blocked: () => {
        const active = selection();
        if (!active) return "Select something first";
        const node = nodeById(active.nodeId);
        return node?.editable && node.line !== null ? null : "Defined dynamically in code";
      },
    },
    "select-object": selectKind("object"),
    "select-vertex": selectKind("vertex"),
    "select-none": {
      run: () => {
        setSelection(null);
        setSimProbe(null);
      },
      blocked: () => (selection() === null ? "Nothing is selected" : null),
    },

    "view-front": view("front"),
    "view-right": view("right"),
    "view-top": view("top"),
    "view-opposite": {
      // The cube's 9: the octant mirrored through the orbit target.
      run: () => {
        const { yaw, pitch } = cameraAngles();
        renderer.camera = { ...renderer.camera, yaw: yaw + Math.PI, pitch: -pitch };
        setCameraAngles({ yaw: yaw + Math.PI, pitch: -pitch });
        renderer.invalidate();
      },
    },
    "view-ortho": toggle(
      () => display().projection === "orthographic",
      (next) => ({ projection: next ? "orthographic" : "perspective" }),
    ),
    "grid-finer": { run: () => stepGrid(-1) },
    "grid-coarser": { run: () => stepGrid(1) },
    "render-preset": { run: () => setRenderPopoverOpen(true) },
    "render-settings": { run: () => setRenderPopoverOpen(true) },
    "show-layout-grid": {
      run: () => setLayoutGridOn(!layoutGridOn()),
      checked: layoutGridOn,
    },
    "show-construction": toggle(
      () => display().showOverlays,
      (next) => ({ showOverlays: next }),
    ),
    "show-edges": toggle(
      () => Boolean(display().showMeshEdges),
      (next) => ({ showMeshEdges: next }),
    ),
    "show-wireframe": toggle(
      () => Boolean(display().showMeshWireframe),
      (next) => ({ showMeshWireframe: next }),
    ),
    "show-solid": toggle(
      () => !display().hideSolid,
      (next) => ({ hideSolid: !next }),
    ),
    "inspect-slice": inspect("slice"),
    "inspect-normals": inspect("normal"),
    "inspect-depth": inspect("depth"),

    "tool-sketch": create("sketch"),
    "tool-face": create("face"),
    "tool-polygon": create("polygon"),
    "tool-box": create("box"),
    "tool-sphere": create("sphere"),
    "tool-cylinder": create("cylinder"),
    "create-material": { run: () => void ops.addMaterial() },

    "shape-extrude": { run: ops.extrudeSelection, blocked: () => shapeBlocked("extrude") },
    "shape-revolve": { run: ops.revolveSelection, blocked: () => shapeBlocked("revolve") },
    "shape-loft": { run: toggleLoft, blocked: () => shapeBlocked("loft") },
    "gizmo-translate": gizmo("translate"),
    "gizmo-rotate": gizmo("rotate"),
    "gizmo-scale": gizmo("scale"),

    ...constraints,
    "constrain-fix": { run: () => fixSelectedPoint(ops.addConstraint), blocked: fixBlocked },

    "sim-new-mesh": { run: () => void sim.patch(addMeshRequest()) },
    "sim-study-thermal": study("thermal"),
    "sim-study-elastic": study("elastic"),
    "sim-study-flow": study("flow"),
    "sim-solve": {
      run: () => {
        const target = solveTarget();
        if (!target) return;
        // The field lands in the Simulate desk's windows; go where it shows.
        if (editingMode() !== "simulate") setEditingMode("simulate");
        void sim.solve(target);
      },
      blocked: () => {
        const target = solveTarget();
        if (!target) return "No study is declared";
        if (sim.solving() !== null) return "A solve is already running";
        if (sim.unavailable()) return "The solver is not available on this server";
        if (target.bcs.length === 0) return `Add a boundary condition to ${target.name} first`;
        return null;
      },
    },
    "sim-optimize": { run: () => windowManager()?.open("optimize") },

    "program-run": {
      run: deps.run,
      blocked: () => (busy() && !dirty() ? "Already compiling this program" : null),
    },
    "program-stop": {
      run: () => {
        const id = stopTarget(sim);
        if (id) void cancelJob(id);
      },
      blocked: () => (stopTarget(sim) ? null : "Nothing is running"),
    },
    "program-wgsl": {
      run: deps.showWgsl,
      blocked: () => (deps.wgslReady() ? null : "Run the program first"),
    },

    ...modes,
    "cycle-mode": { run: () => cycleMode(1) },
    "cycle-mode-back": { run: () => cycleMode(-1) },

    ...windows,
    "window-float": {
      run: () => {
        const manager = windowManager();
        const id = focusedWindow();
        if (!manager || !id) return;
        if (manager.isFloating(id)) manager.dock(id);
        else manager.float(id);
      },
      checked: () => {
        const id = focusedWindow();
        return id !== null && windowManager()?.isFloating(id) === true;
      },
      blocked: () => {
        const id = focusedWindow();
        if (!windowManager()) return "The desk is still loading";
        if (!id) return "Click in a window first";
        return windowManager()!.status(id) === "open" ? null : "That window is closed";
      },
    },
    "window-reset": {
      run: () => windowManager()?.resetLayout(),
      blocked: () => (windowManager() ? null : "The desk is still loading"),
    },

    "help-design": { run: () => void window.open("/design.html", "_blank", "noreferrer") },
  });
  onCleanup(unregister);
}
