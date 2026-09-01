/**
 * Pure helpers for the mesh inspector: patch-request bodies for declared
 * SimMesh objects, and the quality-histogram binning behind the inline SVG.
 *
 * Meshes are declared in the scene program; the panel edits them only through
 * /patch source operations, mirroring how studies.ts treats studies.
 */

import type { SimMeshPayload, StudyPayload } from "./types";

export function addMeshRequest(name?: string): Record<string, unknown> {
  const body: Record<string, unknown> = { op: "add_mesh" };
  if (name) body.name = name;
  return body;
}

export function deleteMeshRequest(mesh: SimMeshPayload): Record<string, unknown> {
  return { op: "delete_mesh", mesh: mesh.index };
}

export type MeshArgument =
  | "resolution"
  | "padding"
  | "bounds"
  | "size"
  | "domain"
  | "method";

export function setMeshValueRequest(
  mesh: SimMeshPayload,
  argument: MeshArgument,
  value: number | number[] | string,
): Record<string, unknown> {
  return { op: "set_mesh_value", mesh: mesh.index, argument, value };
}

/** Point a study at a declared SimMesh (or back to implicit meshing). */
export function setStudyMeshRequest(
  study: StudyPayload,
  meshName: string,
): Record<string, unknown> {
  return { op: "set_study_value", study: study.index, argument: "mesh", value: meshName };
}

/** Restrict a study's implicit mesh to a named scene object. */
export function setStudyDomainRequest(
  study: StudyPayload,
  objectName: string,
): Record<string, unknown> {
  return { op: "set_study_value", study: study.index, argument: "domain", value: objectName };
}

/** The numeric rows a mesh card offers for editing. */
export function meshArguments(
  mesh: SimMeshPayload,
): { key: MeshArgument; value: number | number[] }[] {
  const rows: { key: MeshArgument; value: number | number[] }[] = [
    { key: "resolution", value: mesh.resolution },
    { key: "padding", value: mesh.padding },
  ];
  if (mesh.bounds !== null) rows.push({ key: "bounds", value: mesh.bounds });
  if (mesh.size !== null) rows.push({ key: "size", value: mesh.size });
  return rows;
}

/** Histogram of a scalar field over fixed-width bins across its range. */
export interface Histogram {
  counts: number[];
  min: number;
  max: number;
  /** Largest bin count, for bar normalization (0 for an empty field). */
  peak: number;
}

export function qualityHistogram(values: readonly number[], bins = 24): Histogram {
  if (values.length === 0 || bins <= 0) {
    return { counts: [], min: 0, max: 0, peak: 0 };
  }
  let min = Infinity;
  let max = -Infinity;
  for (const value of values) {
    if (value < min) min = value;
    if (value > max) max = value;
  }
  const counts = new Array<number>(bins).fill(0);
  const span = max - min;
  for (const value of values) {
    // A constant field lands entirely in the first bin.
    const bin = span > 0 ? Math.min(bins - 1, Math.floor(((value - min) / span) * bins)) : 0;
    counts[bin]++;
  }
  return { counts, min, max, peak: Math.max(...counts) };
}
