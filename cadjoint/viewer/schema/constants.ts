/**
 * Generated from cadjoint (enums, viewer/_limits) — do not edit.
 *
 * Regenerate with:
 *
 *     python -m cadjoint.viewer.schema.emit
 */

import type { BoundaryConditionType, StudyKind } from "./payloads";

/** The boundary conditions each study kind accepts, in the order they are offered. */
export const STUDY_BOUNDARY_CONDITIONS: Readonly<
  Record<StudyKind, readonly BoundaryConditionType[]>
> = {
  thermal: ["dirichlet", "heat_flux"],
  elastic: ["fixed", "traction"],
  flow: ["inlet", "outlet", "walls", "heat_source", "held_temperature"],
};

/** Conditions that are faces of the flow lattice and place no region. */
export const UNPLACED_BOUNDARY_CONDITIONS: readonly BoundaryConditionType[] = ["inlet", "outlet", "walls"];

/** The export lattice's resolution bounds and default. */
export const EXPORT_RESOLUTION = {
  min: 8,
  max: 256,
  default: 64,
} as const;

/** The scene shader's parameter buffer: one `vec4<f32>` slot per parameter, then the reserved ones. */
export const SHADER_PARAMETERS = {
  group: 3,
  binding: 0,
  addressSpace: "storage, read",
  slotBytes: 16,
  reservedSlots: 2,
  cullMargin: 0.0001,
} as const;

/** The `cadjoint-scene` graph format and the versions the server writes. */
export const SCENE_GRAPH = {
  format: "cadjoint-scene",
  versions: [1, 2],
} as const;

/** The static demo's scene files (`python -m cadjoint.viewer.demo`). */
export const DEMO_BUNDLE = {
  format: "cadjoint-demo",
  version: 1,
} as const;
