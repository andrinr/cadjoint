/**
 * The `cadjoint-scene` graph, as the Python side writes it
 * (`cadjoint/backends/jaxjs/scene.py`; the format is in `SCENE_FORMAT.md`
 * there). Version 2 adds node ids, free-parameter names and materials to
 * version 1; a version 1 graph reads as one with no sharing, no free
 * parameters and the default material.
 */

export const SCENE_FORMAT = "cadjoint-scene";
export const SCENE_VERSIONS = [1, 2] as const;

import type {
  SceneGraph as PayloadSceneGraph,
  SceneGraphMaterial,
  SceneGraphNode,
} from "../../../cadjoint/viewer/schema/payloads";

export type Vec = number[];

/** The graph types are the compile payload's (`cadjoint/viewer/schema`). */
export type SceneMaterial = SceneGraphMaterial;
export type SceneNode = SceneGraphNode;
/** A graph of any version; `sceneRoot` checks the version before reading. */
export type SceneGraph = Omit<PayloadSceneGraph, "version"> & { version: number };

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
