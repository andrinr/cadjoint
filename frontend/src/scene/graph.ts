/**
 * The `cadjoint-scene` graph, as the Python side writes it
 * (`cadjoint/backends/jaxjs/scene.py`; the format is in `SCENE_FORMAT.md`
 * there). Version 2 adds node ids, free-parameter names and materials to
 * version 1; a version 1 graph reads as one with no sharing, no free
 * parameters and the default material.
 */

export const SCENE_FORMAT = "cadjoint-scene";
export const SCENE_VERSIONS = [1, 2] as const;

export type Vec = number[];

export type SceneMaterial = {
  color: (number | null)[];
  roughness: number | null;
  metallic: number | null;
  opacity: number | null;
  ior: number | null;
  reflectivity: number | null;
};

export type SceneNode = {
  type: string;
  params: Record<string, number | Vec | Vec[]>;
  children: SceneNode[];
  /** v2: stable within the graph; a shared subtree repeats its id. */
  id?: string;
  /** v2: graph parameter → declared free name; per vertex for profiles. */
  free?: Record<string, string | (string | null)[]>;
  /** v2: parameters the node never declared, written with their default. */
  implicit?: string[];
  /** v2: a primitive's material. */
  material?: SceneMaterial;
};

export type SceneGraph = { format: string; version: number; root: SceneNode };

/** `cadjoint.render.material.Material()`'s optical properties. */
export const DEFAULT_MATERIAL: SceneMaterial = {
  color: [1, 1, 1],
  roughness: 0.5,
  metallic: 0,
  opacity: 1,
  ior: 1,
  reflectivity: 0,
};

/** The root of a scene graph, checking its format and version. */
export function sceneRoot(graph: SceneGraph): SceneNode {
  if (graph.format !== SCENE_FORMAT || !(SCENE_VERSIONS as readonly number[]).includes(graph.version))
    throw new Error(`scene is ${graph.format} v${graph.version}; this reads ${SCENE_FORMAT} v${SCENE_VERSIONS.join(", v")}`);
  return graph.root;
}
