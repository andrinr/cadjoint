/**
 * The demo's bundle: one `cadjoint-demo` file per shipped scene and the
 * public tier's capabilities, written by `python -m cadjoint.viewer.demo`
 * (the schema is `cadjoint/viewer/schema/demo.py`).
 *
 * Scenes load on demand, so the page downloads only the one on screen. Each
 * is read as text and parsed here: a JSON import becomes an object literal
 * in the bundle, which turns every -0 into 0, and the WGSL built from the
 * graph spells -0.
 */
import { DEMO_BUNDLE } from "../../../cadjoint/viewer/schema/constants.ts";
import type { CapabilitiesResponse, DemoScene } from "../types";
import tierText from "./bundle/public-tier.json?raw";

const loaders = import.meta.glob<string>("./bundle/*.demo.json", { query: "?raw", import: "default" });

const nameOf = (path: string) => path.split("/").pop()!.replace(".demo.json", "");

/** The shipped scenes' names, sorted. */
export const SCENE_NAMES: readonly string[] = Object.keys(loaders).map(nameOf).sort();

/** What `cadjoint.tier` reports with the private tier absent. */
export const PUBLIC_TIER: CapabilitiesResponse = JSON.parse(tierText);

/** Check a scene file's format and version before trusting its shape. */
export function parseDemoScene(text: string): DemoScene {
  const scene = JSON.parse(text) as Partial<DemoScene>;
  if (scene.format !== DEMO_BUNDLE.format || scene.version !== DEMO_BUNDLE.version)
    throw new Error(
      `scene file is ${scene.format} v${scene.version}; this page reads ${DEMO_BUNDLE.format} v${DEMO_BUNDLE.version}`,
    );
  return scene as DemoScene;
}

/** One shipped scene, fetched and checked. */
export async function loadScene(name: string): Promise<DemoScene> {
  const load = Object.entries(loaders).find(([path]) => nameOf(path) === name)?.[1];
  if (!load) throw new Error(`no shipped scene named ${name}`);
  return parseDemoScene(await load());
}
