/**
 * Shapes exchanged with the playground server.
 *
 * The compile payload and the patch requests are **not** written here. They
 * are generated from the pydantic models the compile worker validates every
 * response against (`cadjoint/viewer/schema/payloads.d.ts`, emitted by
 * `python -m cadjoint.viewer.schema.emit`, pinned by
 * `tests/viewer/test_parity_schema.py`), and this module re-exports them
 * under the names the app already uses. A hand-maintained copy of a
 * generated contract is a copy that drifts, and it had: `line` was declared
 * non-null on a transform that sends null, `faces` optional on a node that
 * always sends it, `objective` non-null on an optimization that may omit it,
 * a study's `material` typed as numbers when it can be the `"material"`
 * sentinel, and three triplets typed as tuples when the wire sends plain
 * lists. Re-exporting is what makes those disagreements impossible rather
 * than merely fixed.
 *
 * What is still declared here is what the *frontend* owns: request bodies it
 * builds, responses from endpoints outside the compile worker (simulate,
 * optimize, lint, completion, scenes), and the two places where the app
 * refines a payload type rather than restating it — `StudySelection`, which
 * the generated model can only express as one flat bag of optional fields,
 * and the narrow literal unions the UI switches on.
 */

import type {
  ConstraintSolverRun,
  ConstructionConstraint as ConstructionConstraintPayload,
  ConstructionNode as ConstructionNodePayload,
  ConstructionRelation,
  MaterialDefinition,
  MeshEdgePayload,
  OptimizationPayload,
  SimMeshPayload,
  StudyBc as StudyBcPayload,
  StudyPayload as StudyPayloadShape,
  StudySelection as StudySelectionPayload,
  ShaderProgram,
} from "../../cadjoint/viewer/schema/payloads";

export type {
  ConstraintSolverRun,
  ConstructionFace,
  ConstructionOperator,
  ConstructionPlane,
  ConstructionRelation,
  ConstructionTransform,
  ConstructionVertex,
  DomainEntry,
  ExportFormat,
  ExportRequest,
  FaceAccessor,
  FaceOwner,
  IdentityEntry,
  MaterialDefinition,
  MeshEdgePayload,
  OptimizationPayload,
  PatchOperation,
  PatchRequest,
  ParameterBinding,
  PatchResponse,
  PlaneReference,
  ShaderParameter,
  ShaderProgram,
  SimMeshPayload,
  SketchPlaneReference,
} from "../../cadjoint/viewer/schema/payloads";

export interface SessionResponse {
  ok: boolean;
  token: string;
  example: string;
}

export type ConstraintKind =
  | "fixed"
  | "distance"
  | "horizontal"
  | "vertical"
  | "coincident"
  | "equal_length"
  | "parallel"
  | "perpendicular";

export type ConstructionKind = "profile" | "box" | "sphere" | "cylinder";

/**
 * The generated construction row, with the kind the viewer switches on.
 *
 * Same refinement as the constraint above, for the same reason: the model
 * leaves `kind` open because a new primitive must not need a schema change,
 * and the gizmo, the object tree and the tool rail all branch on the closed
 * set the app can actually draw.
 */
export interface ConstructionNode extends ConstructionNodePayload {
  kind: ConstructionKind;
  constraints: ConstructionConstraint[];
}

export type ConstraintSolverMethod = "newton" | "adam" | "sgd";

/**
 * The generated constraint row, with the kind the UI actually switches on.
 *
 * The model types `kind` as a string because the FEM and sketch layers can
 * grow constraint kinds without a schema change; the panels need the closed
 * set to render a label per kind, so the refinement lives here rather than
 * in the wire contract. Everything else — `index`, `stableId`, `value` — is
 * inherited, which is the point.
 */
export interface ConstructionConstraint extends ConstructionConstraintPayload {
  kind: ConstraintKind;
}

export interface CompileResponse {
  ok: boolean;
  error?: string;
  sdf: string;
  preview_shader: string;
  path_shader: string;
  present_shader: string;
  construction: ConstructionNode[];
  relations: ConstructionRelation[];
  solver_runs: ConstraintSolverRun[];
  materials: MaterialDefinition[];
  mesh_edges: MeshEdgePayload | null;
  /**
   * The uniform contract for the two scene shaders, when the worker built
   * them in the uniform form (its default).
   *
   * Absent or null means the literal form, where the parameters are
   * constants in the source and every edit is a fresh module. The renderer
   * uses this to tell a parameter edit from a topology edit.
   */
  program?: ShaderProgram | null;
  studies?: StudyPayload[];
  sim_meshes?: SimMeshPayload[];
  optimizations?: OptimizationPayload[];
  output: string;
}

/**
 * A serialized node selection, mirroring `cadjoint.fem.selection`.
 *
 * One pydantic model has to cover every leaf and every composite, so on the
 * wire this is a flat bag of optional fields with `kind` saying which are
 * meaningful. The app has to *narrow* it — evaluating a selection is a switch
 * over exactly these seven shapes — so the wire type is intersected with the
 * discriminated union rather than replaced by it: the app gets exhaustiveness,
 * and anything assignable here is still assignable to the payload the server
 * validates.
 */
export type StudySelection = StudySelectionPayload &
  (
    | { kind: "box"; min_corner: number[]; max_corner: number[] }
    | { kind: "sphere"; center: number[]; radius: number }
    | { kind: "halfspace"; point: number[]; normal: number[] }
    | { kind: "side"; side: string; tol: number | null }
    | { kind: "predicate"; name: string }
    | { kind: "and" | "or"; operands: StudySelection[] }
    | { kind: "not"; operand: StudySelection }
  );

export type StudyBcType = "dirichlet" | "heat_flux" | "fixed" | "traction";

/** The generated BC row, narrowed to the kinds and selections the UI draws. */
export interface StudyBc extends StudyBcPayload {
  type: StudyBcType;
  nodes: StudySelection;
}

/** The generated study, carrying the narrowed BC rows. */
export interface StudyPayload extends StudyPayloadShape {
  bcs: StudyBc[];
}

/** Element type a SimMesh extracts. */
export type MeshMethod = "hex" | "tet4" | "tet10";

/** min/mean/max summary of a per-element quality metric. */
export interface QualitySummary {
  min: number;
  mean: number;
  max: number;
}

/** JSON inspection report of a built mesh (SimMesh.inspect()). */
export interface MeshInspectInfo {
  name: string;
  nodes: number;
  elements: number;
  method?: MeshMethod;
  bounds: { min: number[]; max: number[] };
  grid: { origin: number[]; spacing: number[]; cells: number[] } | null;
  quality: Record<string, QualitySummary>;
}

/** POST /api/mesh_inspect: build a declared mesh and report its quality. */
export interface MeshInspectResponse {
  ok: boolean;
  kind?: "mesh_inspect";
  name?: string;
  /** The scalar field carried by `mesh.scalars` (scaled_jacobian). */
  field?: string;
  info?: MeshInspectInfo;
  mesh?: SimulationMeshPayload;
  /** Per-vertex min scaled Jacobian, same order as `mesh.positions`. */
  quality_scalars?: number[];
  error?: string;
  output?: string;
}

/** One Optimization(...) declared in the scene program. */
/** One recorded optimizer step. */
export interface OptimizeHistoryEntry {
  step: number;
  objective: number;
  grad_norm: number;
}

/** One replayable trajectory frame (step 0 is the initial state). */
export interface OptimizeTrajectoryEntry {
  step: number;
  objective: number;
  parameters: Record<string, number | number[]>;
}

export interface OptimizeRequest {
  source: string;
  name: string;
  steps?: number;
}

/** The final design's solved field, attached to study-backed optimize runs. */
export interface OptimizeSimulateBlock {
  field?: string | null;
  mesh?: SimulationMeshPayload;
  result?: SimulationResultSummary;
  mesh_info?: MeshInspectInfo | null;
}

/** POST /api/optimize: run a declared optimization to completion. */
export interface OptimizeResponse {
  ok: boolean;
  kind?: "optimize";
  name?: string;
  /** The program with the optimized parameter literals written back. */
  source?: string;
  history?: OptimizeHistoryEntry[];
  /** Parameter snapshots along the run, for the replay player (≤100). */
  trajectory?: OptimizeTrajectoryEntry[];
  parameters?: Record<string, number | number[]>;
  initial?: Record<string, number | number[]>;
  /** Study-backed runs: the optimized design's solved field, ready to show. */
  simulate?: OptimizeSimulateBlock;
  error?: string;
  output?: string;
}

/** A viewport-picked BC region, pre-filling the add-BC builder. */
export type BcProposal =
  | { kind: "sphere"; center: [number, number, number]; radius: number }
  | { kind: "box"; min: [number, number, number]; max: [number, number, number] };

/** Lazy mesh-edge extraction, requested only while a mesh overlay is on. */
export interface MeshResponse {
  ok: boolean;
  mesh_edges?: MeshEdgePayload | null;
  error?: string;
}

/**
 * What a scene's declarations add up to, counted by the server statically.
 *
 * `parameters` counts only the ones that carry a `name=` — an unnamed
 * `Scalar(0.07)` inside a primitive is a literal, not a design freedom — and
 * `free` is the subset an optimization is allowed to move.
 */
export interface SceneCounts {
  parameters: number;
  free: number;
  studies: number;
  meshes: number;
  optimizations: number;
  materials: number;
}

/**
 * One described scene file, as `GET /api/scenes` lists it.
 *
 * Everything here is read with `ast`, never by running the program: a browser
 * that executed every file in the directory to describe it would be running
 * arbitrary code on a directory listing. `source_hash` is the same sha256 the
 * job registry stamps on a request, which is what lets a rendered thumbnail
 * be cached against the exact bytes it was drawn from.
 */
export interface SceneEntry {
  name: string;
  path: string;
  bytes: number;
  /** ISO-8601 UTC, or null when the file could not be stat'd. */
  modified: string | null;
  source_hash: string | null;
  /** First paragraph of the module docstring, collapsed to one line. */
  summary: string;
  counts: SceneCounts;
  materials: string[];
  /** A syntax error, or a read failure: the file is still listed. */
  error: string | null;
}

export interface SceneListResponse {
  ok: boolean;
  files?: string[];
  /** The same files, described. Absent from an older server. */
  scenes?: SceneEntry[];
  error?: string;
}

export interface SceneLoadResponse {
  ok: boolean;
  name?: string;
  source?: string;
  error?: string;
}

export interface SceneSaveResponse {
  ok: boolean;
  name?: string;
  error?: string;
}

/** What the user has selected: a whole object, or one of its vertices. */
export interface Selection {
  nodeId: string;
  /** Null when the selection is the object itself, as for a primitive. */
  vertexIndex: number | null;
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

/** One boundary face group of the simulation hex mesh — a BC target. */
export interface SimulationFaceGroup {
  /** Gradient-axis id such as `+x` or `-z`. */
  id: string;
  axis: "x" | "y" | "z";
  side: "+" | "-";
  center: [number, number, number];
  area: number;
  faces: number;
  /** This group's triangle range in the payload's index buffer. */
  start: number;
  count: number;
}

/** Indexed boundary-surface triangles with one scalar per vertex. */
export interface SimulationMeshPayload {
  /** Flat xyz positions, three floats per vertex. */
  positions: number[];
  /** One scalar per vertex (temperature, von Mises, or zero for a probe). */
  scalars: number[];
  /** Flat triangle list into the compacted vertex array. */
  indices: number[];
  /** Unique element boundary-face edges, flat index pairs (no diagonals). */
  edges?: number[];
  groups: SimulationFaceGroup[];
  /** Min and max of the scalar field. */
  range: [number, number];
  vertex_count: number;
  /**
   * Solved payloads carry the full per-vertex field catalog here (the
   * backend's `_result_field_payload` attaches them to the mesh payload):
   * every nodal field for display switching, its `[min, max]` per entry,
   * and — for elastic solves — raw displacement vectors for a warped view.
   */
  fields?: Record<string, number[]>;
  ranges?: Record<string, [number, number]>;
  displacements?: [number, number, number][];
}

/** `probe` only meshes and returns the face-group catalog; the rest solve. */
export type SimulationKind = "probe" | "thermal" | "elastic";

/** One boundary condition, targeting a face group by id. */
export interface SimulationBc {
  group: string;
  type: "dirichlet" | "traction";
  value: number | [number, number, number];
}

export interface SimulateRequest {
  source: string;
  kind: SimulationKind;
  resolution: number;
  bcs: SimulationBc[];
  material: Record<string, number>;
}

/** Run a study declared in the scene program, resolved server-side by name. */
export interface SimulateStudyRequest {
  source: string;
  kind: "study";
  name: string;
}

/** JSON summary of a solved study (SimulationResult.describe()). */
export interface SimulationResultSummary {
  name: string;
  kind: "thermal" | "elastic";
  /** The display field carried by the response mesh scalars. */
  field: string;
  /** SimMesh name the study solved on, or null for an implicit mesh. */
  mesh: string | null;
  nodes: number;
  elements: number;
  range: [number, number];
  fields: Record<string, QualitySummary>;
}

export interface SimulateResponse {
  ok: boolean;
  kind?: SimulationKind | "study";
  /** Which nodal field `mesh.scalars` carries; null for a probe. */
  field?: string | null;
  mesh?: SimulationMeshPayload;
  /** The solved study's description, echoed on `kind: "study"` responses. */
  study?: StudyPayload;
  /** Solved-result summary: field ranges, element counts, source mesh. */
  result?: SimulationResultSummary;
  /** Inspection report of the mesh the study solved on. */
  mesh_info?: MeshInspectInfo | null;
  /**
   * Per-vertex nodal fields, ranges, and displacements historically sat at
   * the top level; the server ships them on `mesh` now (see
   * SimulationMeshPayload) and readers coalesce both homes.
   */
  fields?: Record<string, number[]>;
  ranges?: Record<string, [number, number]>;
  displacements?: [number, number, number][];
  error?: string;
  /** `fem_unavailable` when the jax-fem extra is not installed (HTTP 501). */
  error_kind?: string;
  output?: string;
}

/** What a click in the viewport picks. */
export type SelectionMode = "object" | "vertex";

// ── editor intelligence ─────────────────────────────────────────────────────
//
// Three endpoints read the editor's Python without running it: `/api/lint`
// (ruff, plus the last compile traceback), `/api/complete` and
// `/api/signature` (jedi). They share one coordinate convention, which is
// also CodeMirror's: **lines are 1-based, columns are 0-based**, so a
// position becomes an offset with `doc.line(from_line).from + from_col`.

export type LintSeverity = "error" | "warning" | "info";

/** Where a ruff autofix came from, i.e. how far it may be trusted. */
export type FixApplicability = "safe" | "unsafe" | "display";

/** One replacement of a ruff autofix, in the shared line/column convention. */
export interface LintFixEdit {
  from_line: number;
  from_col: number;
  to_line: number;
  to_col: number;
  content: string;
}

/** A ruff autofix: a label plus the edits that apply it in one go. */
export interface LintFix {
  message: string;
  applicability: FixApplicability;
  edits: LintFixEdit[];
}

/**
 * One diagnostic.
 *
 * `source` is `"ruff"` for static analysis and `"runtime"` for the traceback
 * of the last failed `/compile` of this exact text — the latter names the
 * line that actually blew up, which no static analyser can produce.
 */
export interface LintDiagnostic {
  from_line: number;
  from_col: number;
  to_line: number;
  to_col: number;
  severity: LintSeverity;
  message: string;
  code: string;
  source: "ruff" | "runtime";
  /** Rule documentation, rendered as a "learn more" link. */
  url: string | null;
  fix: LintFix | null;
}

export interface LintResponse {
  ok: boolean;
  /** True when a remembered traceback contributed one of the diagnostics. */
  runtime?: boolean;
  diagnostics?: LintDiagnostic[];
  error?: string;
}

/** One jedi completion, already shaped like CodeMirror's `Completion`. */
export interface CompletionItem {
  label: string;
  /** CodeMirror completion type: `function`, `class`, `property`, … */
  type: string;
  /** Jedi's own kind, shown beside the label. */
  detail: string;
  /** Signature and docstring; present only for the head of the list. */
  info: string | null;
  apply: string;
}

export interface CompleteResponse {
  ok: boolean;
  /** Caret line the completions were computed at (1-based). */
  from_line?: number;
  /** Where the already-typed prefix starts (0-based). */
  from_column?: number;
  truncated?: boolean;
  completions?: CompletionItem[];
  error?: string;
}

export interface SignatureParameter {
  name: string;
  label: string;
}

/** One call signature the caret sits inside. */
export interface SignatureInfo {
  name: string;
  label: string;
  /** Index of the argument being typed, or null between calls. */
  active_parameter: number | null;
  parameters: SignatureParameter[];
  documentation: string | null;
}

export interface SignatureResponse {
  ok: boolean;
  signatures?: SignatureInfo[];
  error?: string;
}
