/**
 * The static viewer: a scene from a bundled `cadjoint-scene` v2 graph,
 * drawn with WGSL generated in the browser (`../scene`), and exported to STL
 * on jax-js (`./jaxjs.ts`). No Python server is involved, so the build is
 * plain files (`npm run build:static`).
 */
import "@fontsource/ibm-plex-mono/400.css";
import "../tokens.css";
import { orbitCamera, zoomCamera } from "../components/viewer/camera";
import { graphBounds } from "../scene/bounds.ts";
import type { SceneGraph } from "../scene/graph.ts";
import { sceneShaders } from "../scene/shaders.ts";
import { Renderer } from "../viewer/renderer";

// Read as text and parsed here: a JSON import becomes an object literal in
// the bundle, which turns every -0 into 0, and the module text spells -0.
const scenes: Record<string, SceneGraph> = Object.fromEntries(
  Object.entries(
    import.meta.glob<string>("./scenes/*.scene.json", { eager: true, query: "?raw", import: "default" }),
  ).map(([path, text]) => [path.split("/").pop()!.replace(".scene.json", ""), JSON.parse(text) as SceneGraph]),
);

const canvas = document.querySelector<HTMLCanvasElement>("#viewport")!;
const picker = document.querySelector<HTMLSelectElement>("#scene")!;
const exportButton = document.querySelector<HTMLButtonElement>("#export")!;
const status = document.querySelector<HTMLElement>("#status")!;
const download = document.querySelector<HTMLAnchorElement>("#download")!;

const say = (text: string) => (status.textContent = text);
const renderer = new Renderer({ onError: (message) => say(message) });

async function show(name: string): Promise<void> {
  const graph = scenes[name];
  const t = performance.now();
  const shaders = await sceneShaders(graph);
  const built = performance.now() - t;
  const bounds = graphBounds(graph.root);
  if (bounds) {
    const radius = Math.hypot(...bounds.half);
    renderer.camera = { ...renderer.camera, target: [...bounds.center] as [number, number, number], distance: radius * 3 };
  }
  await renderer.setShaders({
    preview: shaders.preview_shader,
    path: shaders.path_shader,
    present: shaders.present_shader,
    program: shaders.program,
  });
  say(`${name}: WGSL built in the browser in ${built.toFixed(0)} ms (${shaders.sdf.length} characters)`);
}

async function exportCurrent(): Promise<void> {
  const name = picker.value;
  const graph = scenes[name];
  const bounds = graphBounds(graph.root);
  if (!bounds) {
    say(`${name} has no bounding box to mesh in`);
    return;
  }
  exportButton.disabled = true;
  download.hidden = true;
  say(`meshing ${name} on jax-js …`);
  try {
    const { exportMesh } = await import("./jaxjs.ts");
    const result = await exportMesh(graph, bounds);
    if (download.href) URL.revokeObjectURL(download.href);
    download.href = URL.createObjectURL(result.stl);
    download.download = `${name}.stl`;
    download.textContent = `${name}.stl`;
    download.hidden = false;
    say(
      `${result.triangles} triangles, volume ${result.volume.toPrecision(6)}, ` +
        `${result.badEdges} bad edges; ${result.cells.join("×")} cells on ${result.device}, ` +
        `sampled in ${result.sampleMs.toFixed(0)} ms, meshed in ${result.meshMs.toFixed(0)} ms`,
    );
  } catch (error) {
    say(`export failed: ${error instanceof Error ? error.message : String(error)}`);
  } finally {
    exportButton.disabled = false;
  }
}

function bindGestures(): void {
  let last: { x: number; y: number } | null = null;
  canvas.addEventListener("pointerdown", (event) => {
    last = { x: event.clientX, y: event.clientY };
    canvas.setPointerCapture(event.pointerId);
    renderer.interacting = true;
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!last) return;
    renderer.camera = orbitCamera(renderer.camera, event.clientX - last.x, event.clientY - last.y);
    last = { x: event.clientX, y: event.clientY };
    renderer.invalidate();
  });
  const end = () => {
    last = null;
    renderer.interacting = false;
  };
  canvas.addEventListener("pointerup", end);
  canvas.addEventListener("pointercancel", end);
  canvas.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      renderer.camera = zoomCamera(renderer.camera, event.deltaY);
      renderer.invalidate();
    },
    { passive: false },
  );
  new ResizeObserver(() => {
    renderer.resize();
    renderer.invalidate();
  }).observe(canvas);
}

async function main(): Promise<void> {
  const names = Object.keys(scenes).sort();
  for (const name of names) picker.add(new Option(name, name));
  const wanted = new URLSearchParams(location.search).get("scene");
  if (wanted && names.includes(wanted)) picker.value = wanted;
  await renderer.init(canvas);
  bindGestures();
  picker.addEventListener("change", () => void show(picker.value));
  exportButton.addEventListener("click", () => void exportCurrent());
  await show(picker.value);
}

// For tests and benchmarks: the bundled graphs and the shader builder.
(window as unknown as Record<string, unknown>).__cadjointStatic = { scenes, sceneShaders };

void main();
