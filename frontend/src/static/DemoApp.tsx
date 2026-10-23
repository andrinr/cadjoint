/**
 * The public demo: the playground's viewport, object tree and display panel
 * over a shipped scene, with no Python server behind them.
 *
 * Where the playground keeps the Python program as its source of truth and
 * turns every edit into a patch and a compile, the demo keeps one record of
 * parameter values (`parameters.ts`) and derives the rest in the browser:
 *
 *   edit (slider, field, handle drag) → values → buffer overrides (this frame)
 *     → scene graph at the values → WGSL + uniform contract (`../scene`)
 *     → `Renderer.setShaders`, a values-only upload since the module text
 *       does not change → construction tree at the values → overlay.
 *
 * Meshing (the wireframe overlay and the STL export) runs dual contouring on
 * jax-js, on WebGPU when there is one and on WebAssembly otherwise.
 */
import { Show, batch, createEffect, createMemo, createSignal, on, onCleanup, onMount } from "solid-js";
import { ObjectTree } from "../components/ObjectTree";
import { ProjectionToggle, SkeletonToggle, ViewCube } from "../components/ViewCube";
import { ViewerPane } from "../components/ViewerPane";
import { RenderSettingsButton } from "../components/viewer/RenderSettingsButton";
import { ObjectSelectIcon, VertexSelectIcon } from "../components/icons";
import { Select, TierTag } from "../components/ui";
import { graphBounds } from "../scene/bounds.ts";
import { sceneShaders } from "../scene/shaders.ts";
import { registerCommands } from "../shell/commands";
import { createRenderState } from "../shell/renderState";
import {
  cameraAngles,
  gizmoMode,
  nodeById,
  nodes,
  selectionMode,
  setCameraAngles,
  setElements,
  setGizmoMode,
  setMeshEdges,
  setNodes,
  setRelations,
  setSceneName,
  setSceneTree,
  setSelection,
  setSelectionMode,
  setStatus,
  status,
  viewerError,
} from "../state";
import { stateTier } from "../tier";
import type { DemoScene, SceneGraph } from "../types";
import { Renderer } from "../viewer/renderer";
import { PUBLIC_TIER, SCENE_NAMES, loadScene } from "./bundle";
import { meshEdges } from "./edges";
import { ParameterPanel } from "./ParameterPanel";
import {
  acceptValues,
  bindingValues,
  constructionAt,
  constructionBounds,
  editedValues,
  graphAt,
  handleLock,
  shippedValues,
  type Values,
} from "./parameters";
import "./demo.css";

/** Cells on the longest side of the wireframe overlay's lattice. */
const WIRE_CELLS = 64;
/** The STL export's choices, cells on the longest side. */
const EXPORT_CELLS = ["64", "96", "128", "160"];
/** Quiet time after the last edit before the wireframe is meshed again. */
const REMESH_MS = 400;

const PLAYGROUND = "https://github.com/andrinr/cadjoint#readme";

const pretty = (name: string) => name.replaceAll("_", " ");

/** Values that differ between two records, for the renderer's live overrides. */
function changed(values: Values, base: Values): Record<string, number[]> {
  const out: Record<string, number[]> = {};
  for (const [name, value] of Object.entries(values))
    if (value.some((component, i) => component !== base[name]?.[i])) out[name] = [...value];
  return out;
}

declare global {
  interface Window {
    /** For the e2e spec: the renderer's counters and the values on screen. */
    __cadjointDemo?: () => {
      scene: string | null;
      values: Values;
      uploads: number;
      pipelineBuilds: number;
      device: string | null;
    };
  }
}

export function DemoApp() {
  // The renderer reports its quality tier after every settled frame; the
  // perf readout already shows it, and here it would overwrite messages.
  // Settles once: true when the renderer has a device, false when it failed
  // first. Shaders installed before that would be dropped, and the demo
  // builds its shaders in milliseconds, well before a device is up.
  let settleGpu!: (ok: boolean) => void;
  const gpuReady = new Promise<boolean>((resolve) => (settleGpu = resolve));
  const [gpuError, setGpuError] = createSignal("");
  const renderer = new Renderer({
    onReady: () => settleGpu(true),
    onError: (message) => {
      setGpuError((current) => current || message);
      settleGpu(false);
      setStatus({ kind: "error", text: message });
    },
  });
  const render = createRenderState(renderer);
  stateTier(PUBLIC_TIER);

  const [scene, setScene] = createSignal<DemoScene | null>(null);
  const [values, setValues] = createSignal<Values>({});
  /** The values the installed shaders were built at. */
  let committed: Values = {};
  const [gpu, setGpu] = createSignal<"pending" | "ok" | "missing">("pending");
  const [compute, setCompute] = createSignal<string | null>(null);
  const [exporting, setExporting] = createSignal(false);
  const [cells, setCells] = createSignal("96");
  const [download, setDownload] = createSignal<{ url: string; name: string } | null>(null);

  const parameters = () => scene()?.parameters ?? [];
  const edited = createMemo(() => Object.keys(editedValues(parameters(), values())).length);
  const graph = (): SceneGraph | null => {
    const current = scene();
    return current ? graphAt(current.scene_graph, values()) : null;
  };

  // ── values → shaders ───────────────────────────────────────────────────
  let installing: Promise<void> = Promise.resolve();
  let revision = 0;
  /** Install shaders built at the current values; the newest call wins. */
  const commit = (): Promise<void> => {
    const current = scene();
    if (!current || gpu() !== "ok") return Promise.resolve();
    const mine = ++revision;
    const at = values();
    installing = installing.then(async () => {
      if (mine !== revision) return;
      const shaders = await sceneShaders(graphAt(current.scene_graph, at));
      if (mine !== revision) return;
      await renderer.setShaders({
        preview: shaders.preview_shader,
        path: shaders.path_shader,
        present: shaders.present_shader,
        program: shaders.program,
      });
      committed = at;
    });
    return installing;
  };
  let commitTimer: ReturnType<typeof setTimeout> | undefined;
  const scheduleCommit = () => {
    clearTimeout(commitTimer);
    commitTimer = setTimeout(() => void commit(), 120);
  };

  /** Adopt proposed values: live in the buffer now, in the shaders shortly. */
  const propose = (proposed: Values) => {
    const current = scene();
    if (!current) return;
    setValues((now) => acceptValues(current.parameters, now, proposed));
  };
  createEffect(
    on(values, (now) => {
      const current = scene();
      if (!current) return;
      setNodes(constructionAt(current.construction, current.parameters, now));
      if (gpu() !== "ok") return;
      const overrides = changed(now, committed);
      if (Object.keys(overrides).length) renderer.setParameterOverrides(overrides);
      scheduleCommit();
    }),
  );

  // ── the viewport's edits ───────────────────────────────────────────────
  const notHere = (what: string) => async (): Promise<void> => {
    setStatus({ kind: "", text: `${what} needs the Python playground; the demo edits parameters only` });
  };

  /** A sketch vertex's release: its binding's value, at once. */
  const onPatch = async (op: string, line: number, index: number, xy?: [number, number]): Promise<void> => {
    const node = nodes().find((n) => n.kind === "profile" && n.line === line);
    const binding = node?.vertices[index]?.binding;
    if (op !== "set_vertex" || !binding || !xy) return notHere("Changing a sketch's points")();
    propose(bindingValues([binding], xy));
    clearTimeout(commitTimer);
    await commit();
  };

  /** A gizmo's release: the argument's bindings, at once. */
  const onSetValue = async (line: number, call: string, argument: string, value: number | number[]): Promise<void> => {
    const node = nodes().find((n) => n.transform?.line === line && n.transform.call === call);
    const bindings = node?.transform?.bindings?.[argument];
    if (!bindings) return notHere(`Changing ${argument}`)();
    propose(bindingValues(bindings, typeof value === "number" ? [value] : value));
    clearTimeout(commitTimer);
    await commit();
  };

  const lockedHandle = (nodeId: string, handle: number | string) =>
    handleLock(nodeById(nodeId), handle, parameters(), renderer.parameterProgram);

  // The viewport keys the demo answers; every other command stays unregistered.
  onCleanup(
    registerCommands({
      "select-object": { run: () => setSelectionMode("object"), checked: () => selectionMode() === "object" },
      "select-vertex": { run: () => setSelectionMode("vertex"), checked: () => selectionMode() === "vertex" },
      "gizmo-translate": { run: () => setGizmoMode("translate") },
      "gizmo-scale": { run: () => setGizmoMode("scale") },
    }),
  );

  // The renderer draws the gizmo for the mode the keys pick.
  createEffect(() => {
    renderer.gizmoMode = gizmoMode();
    renderer.invalidate();
  });

  // ── scenes ─────────────────────────────────────────────────────────────
  const open = async (name: string) => {
    setStatus({ kind: "", text: `Loading ${pretty(name)}…` });
    const next = await loadScene(name);
    const shipped = shippedValues(next.parameters);
    committed = shipped;
    batch(() => {
      setSelection(null);
      setMeshEdges(null);
      setSceneName(`${next.name}.py`);
      setElements(next.elements);
      setSceneTree(next.scene_tree);
      setRelations(next.relations);
      setScene(next);
      setValues(shipped);
    });
    const url = new URL(location.href);
    url.searchParams.set("scene", name);
    history.replaceState(null, "", url);
    const bounds = graphBounds(next.scene_graph.root) ?? constructionBounds(next.construction);
    if (bounds) {
      renderer.camera = {
        ...renderer.camera,
        target: [...bounds.center] as [number, number, number],
        distance: Math.hypot(...bounds.half) * 2.2,
      };
    }
    if (gpu() === "ok") {
      clearTimeout(commitTimer);
      await commit();
    }
    const free = next.parameters.filter((p) => p.drivenBy === null).length;
    setStatus({
      kind: "ready",
      text: `${free} free parameter${free === 1 ? "" : "s"} · drag a ring in the viewport or move a slider`,
    });
  };

  // ── meshing: the wireframe overlay and the STL ─────────────────────────
  const mesh = async (longest: number) => {
    const at = graph();
    const bounds = at && (graphBounds(at.root) ?? constructionBounds(nodes()));
    if (!at || !bounds) throw new Error("this scene has no bounding box to mesh in");
    const { meshScene } = await import("./jaxjs.ts");
    const result = await meshScene(at, bounds, longest);
    setCompute(result.device);
    return result;
  };

  const wantsMesh = () => render.display().showMeshWireframe || render.display().showMeshEdges;
  let meshTimer: ReturnType<typeof setTimeout> | undefined;
  let meshRevision = 0;
  createEffect(
    on([wantsMesh, values, scene], ([wanted]) => {
      clearTimeout(meshTimer);
      if (!wanted) {
        setMeshEdges(null);
        return;
      }
      const mine = ++meshRevision;
      meshTimer = setTimeout(async () => {
        try {
          setStatus({ kind: "", text: "Meshing the overlay on jax-js…" });
          const result = await mesh(WIRE_CELLS);
          if (mine !== meshRevision || !wantsMesh()) return;
          const edges = meshEdges(result.vertices, result.triangles, WIRE_CELLS);
          setMeshEdges(edges);
          setStatus({
            kind: "ready",
            text:
              `Overlay: ${edges.wire.length.toLocaleString()} edges, ${edges.sharp.length.toLocaleString()} creases; ` +
              `${result.cells.join("×")} cells on ${result.device} in ${(result.sampleMs + result.meshMs).toFixed(0)} ms`,
          });
        } catch (error) {
          setStatus({ kind: "error", text: `Wireframe failed: ${error instanceof Error ? error.message : error}` });
        }
      }, REMESH_MS);
    }),
  );

  const exportStl = async () => {
    const current = scene();
    if (!current) return;
    setExporting(true);
    setStatus({ kind: "", text: `Meshing ${pretty(current.name)} on jax-js…` });
    try {
      const result = await mesh(Number(cells()));
      const { stlBlob } = await import("./jaxjs.ts");
      const previous = download();
      if (previous) URL.revokeObjectURL(previous.url);
      const name = `${current.name}${edited() ? "-edited" : ""}.stl`;
      const url = URL.createObjectURL(stlBlob(result));
      setDownload({ url, name });
      const link = Object.assign(document.createElement("a"), { href: url, download: name });
      link.click();
      setStatus({
        kind: "ready",
        text:
          `${name}: ${(result.triangles.length / 3).toLocaleString()} triangles, ` +
          `${result.cells.join("×")} cells on ${result.device}, ` +
          `${(result.sampleMs + result.meshMs).toFixed(0)} ms` +
          (result.badEdges ? `, ${result.badEdges} open edges` : ""),
      });
    } catch (error) {
      setStatus({ kind: "error", text: `Export failed: ${error instanceof Error ? error.message : error}` });
    } finally {
      setExporting(false);
    }
  };

  // ── start ──────────────────────────────────────────────────────────────
  onMount(async () => {
    setSelectionMode("vertex");
    const wanted = new URLSearchParams(location.search).get("scene");
    const first = wanted && SCENE_NAMES.includes(wanted) ? wanted : SCENE_NAMES.includes("bracket") ? "bracket" : SCENE_NAMES[0];
    void gpuReady.then((ok) => {
      setGpu(ok ? "ok" : "missing");
      if (ok) void commit();
    });
    // jax-js picks its device in the background, so the toolbar can say where meshing will run.
    void import("./jaxjs.ts")
      .then(({ computeDevice }) => computeDevice())
      .then(setCompute, () => setCompute("none"));
    try {
      await open(first);
    } catch (error) {
      setStatus({ kind: "error", text: `Could not load ${first}: ${error instanceof Error ? error.message : error}` });
    }
  });
  onCleanup(() => {
    clearTimeout(commitTimer);
    clearTimeout(meshTimer);
  });

  window.__cadjointDemo = () => ({
    scene: scene()?.name ?? null,
    values: values(),
    uploads: renderer.shaderStats.parameterUploads,
    pipelineBuilds: renderer.shaderStats.pipelineBuilds,
    device: compute(),
  });

  const renderPanelProps = () => ({
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
  });

  return (
    <div class="app demo" data-mode="model">
      <header class="toolbar demo-toolbar">
        <div class="brand">
          <span class="mark">cj</span>
          <span>CADJOINT</span>
        </div>
        <span class="demo-badge" title="A static page: no Python runs; everything below runs in your browser">
          public demo
        </span>
        <Select
          value={scene()?.name ?? ""}
          options={SCENE_NAMES.map((name) => ({ value: name, label: pretty(name) }))}
          onChange={(name) => void open(name)}
          aria-label="Scene"
          testId="scene-picker"
        />
        <TierTag kind="brep" testId="demo-tier" />
        <div class="spacer" />
        <span class={`status ${status().kind}`} data-testid="status">
          <Show when={status().text}>
            <i class="dot" />
            {status().text}
          </Show>
        </span>
        <span class="demo-device" data-testid="compute-device" title="Where jax-js runs the meshing">
          {compute() === null ? "jax-js …" : compute() === "none" ? "no jax-js device" : `jax-js · ${compute()}`}
        </span>
        <Select
          value={cells()}
          options={EXPORT_CELLS.map((value) => ({ value, label: `${value} cells` }))}
          onChange={setCells}
          aria-label="Export resolution"
          title="Cells on the longest side of the export lattice"
          testId="export-cells"
        />
        <button
          type="button"
          class="primary"
          disabled={exporting() || !scene() || compute() === "none"}
          onClick={() => void exportStl()}
          data-testid="export-stl"
        >
          {exporting() ? "Meshing…" : "Export STL"}
        </button>
        <Show when={download()}>
          {(file) => (
            <a class="demo-download" href={file().url} download={file().name} data-testid="stl-download">
              {file().name}
            </a>
          )}
        </Show>
      </header>

      <main class="demo-desk">
        <ObjectTree />
        <section class="demo-viewport">
          <ViewerPane
            renderer={renderer}
            display={render.display()}
            onPatch={onPatch}
            onLiveParameters={propose}
            lockedHandle={lockedHandle}
            onSetValue={onSetValue}
            onAddPrimitive={notHere("Adding a solid")}
            onAddSketch={notHere("Adding a sketch")}
            onSketchOnFace={notHere("Sketching on a face")}
            onAddConstraint={notHere("Adding a constraint")}
            onAddLoft={notHere("Lofting")}
            onDeleteObject={notHere("Deleting an object")}
            onAssignMaterial={notHere("Assigning a material")}
            overlay={
              <>
                <div class="view-controls" data-testid="view-controls">
                  <button
                    type="button"
                    class="viewport-bar-button"
                    classList={{ active: selectionMode() === "vertex" }}
                    onClick={() => setSelectionMode("vertex")}
                    title="Drag sketch points  (V)"
                    data-testid="pick-points"
                  >
                    <VertexSelectIcon size={16} />
                    <span class="rail-word">Points</span>
                  </button>
                  <button
                    type="button"
                    class="viewport-bar-button"
                    classList={{ active: selectionMode() === "object" }}
                    onClick={() => setSelectionMode("object")}
                    title="Select solids and drag their gizmo  (O)"
                    data-testid="pick-objects"
                  >
                    <ObjectSelectIcon size={16} />
                    <span class="rail-word">Objects</span>
                  </button>
                  <SkeletonToggle
                    shown={render.display().showOverlays}
                    onToggle={(showOverlays) => render.applyDisplay({ showOverlays })}
                  />
                  <ProjectionToggle
                    projection={render.display().projection}
                    onProjection={(projection) => render.applyDisplay({ projection })}
                  />
                  <RenderSettingsButton render={renderPanelProps()} />
                </div>
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
                <Show when={gpu() === "missing"}>
                  <div class="demo-nogpu" data-testid="no-webgpu">
                    <b>The viewport cannot draw here: it needs WebGPU.</b>
                    <p>
                      {gpuError()} The model is ray-marched on the GPU; a current Chrome or Edge, or Safari 26 and
                      later, can draw it.{" "}
                      {compute() === "wasm"
                        ? "The parameters and the STL export still work: meshing runs on WebAssembly."
                        : compute() === "none"
                          ? "This browser cannot run jax-js on WebAssembly either, so the STL export is off too."
                          : ""}
                    </p>
                  </div>
                </Show>
                <Show when={viewerError()}>
                  <p class="demo-error">{viewerError()}</p>
                </Show>
              </>
            }
          />
        </section>
        <div class="demo-side">
          <Show when={scene()}>
            {(current) => (
              <section class="demo-about" data-testid="scene-about">
                <h1>{current().title}</h1>
                <p>{current().description}</p>
                <p class="demo-note">
                  Dual contouring in the browser; the exact B-rep, simulation and optimisation run in the{" "}
                  <a href={PLAYGROUND} target="_blank" rel="noreferrer">
                    Python playground
                  </a>
                  .
                </p>
              </section>
            )}
          </Show>
          <ParameterPanel
            parameters={parameters()}
            values={values()}
            edited={edited()}
            onChange={(name, value) => propose({ [name]: value })}
            onReset={() => {
              const current = scene();
              if (current) setValues(shippedValues(current.parameters));
            }}
          />
        </div>
      </main>
    </div>
  );
}
