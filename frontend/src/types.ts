/**
 * Shapes exchanged with the playground server, and the app's own.
 *
 * Nothing the server sends or accepts is declared here. Every request and
 * response of every route is generated from the pydantic models in
 * `cadjoint/viewer/schema` (`payloads.d.ts`, emitted by
 * `python -m cadjoint.viewer.schema.emit`, pinned by
 * `tests/viewer/test_parity_schema.py` and `test_api_contract.py`), and this
 * module re-exports them. A hand-maintained copy of a generated contract is a
 * copy that drifts, and it had: a `cylinder` selection the server sends and
 * the preview could not evaluate, an `equal_length` constraint the add-tool
 * vocabulary did not have, face groups typed non-null on tet meshes that send
 * null, and a whole simulate request shape the server stopped accepting.
 *
 * What is still declared here is what the *frontend* owns: the selection,
 * tool and gizmo state, a viewport-picked BC proposal, and a few names derived
 * from the generated types for the unions the UI switches on.
 */

import type {
  ConstructionNode,
  LintDiagnostic,
  LintFix,
  SimMeshPayload,
  StudyPayload,
} from "../../cadjoint/viewer/schema/payloads";

export type * from "../../cadjoint/viewer/schema/payloads";

/** A sketch profile or a primitive: the kinds the gizmo and the tool rail draw. */
export type ConstructionKind = ConstructionNode["kind"];

/**
 * What a study in the payload can be.
 *
 * Equal to the generated `StudyKind` (the kinds the GUI can *author*) today,
 * and kept as its own name because the two have differed before and may
 * again: a kind a program may *contain* is reported and read but only built
 * here once the patch layer can write it.
 */
export type StudyPayloadKind = StudyPayload["kind"];

/** Element type a SimMesh extracts. */
export type MeshMethod = NonNullable<SimMeshPayload["method"]>;

export type LintSeverity = LintDiagnostic["severity"];

/** Where a ruff autofix came from, i.e. how far it may be trusted. */
export type FixApplicability = LintFix["applicability"];

/** A viewport-picked BC region, pre-filling the add-BC builder. */
export type BcProposal =
  | { kind: "sphere"; center: [number, number, number]; radius: number }
  | { kind: "box"; min: [number, number, number]; max: [number, number, number] };

/** What the user has selected: a whole object, or one of its vertices. */
export interface Selection {
  nodeId: string;
  /** Null when the selection is the object itself, as for a primitive. */
  vertexIndex: number | null;
  /**
   * Which part of the object was taken hold of.
   *
   * A sketch is one object with two graspable things: its polygon and the
   * plane it sits on. Selecting the plane keeps `nodeId` on the sketch — every
   * panel that shows "the selected sketch" stays right — and puts the gizmo on
   * the plane's origin rather than the polygon's centre.
   */
  part?: "plane";
}

export type ToolMode =
  | "select"
  | "sketch"
  | "polygon"
  | "distance"
  | "horizontal"
  | "vertical"
  | "coincident"
  | "parallel"
  | "perpendicular"
  | "box"
  | "sphere"
  | "cylinder"
  | "face";

/** How a gizmo drag transforms the selected construction object. */
export type GizmoMode = "translate" | "rotate" | "scale";

/** What a click in the viewport picks. */
export type SelectionMode = "object" | "vertex";
