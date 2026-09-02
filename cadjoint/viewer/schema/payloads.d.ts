/**
 * Generated from the pydantic models in cadjoint/viewer/schema — do not edit.
 *
 * Regenerate with:
 *
 *     python -m cadjoint.viewer.schema.emit
 *
 * The compile worker validates every payload it sends against those same
 * models, so a type here is a guarantee, not a hope. `tests/viewer/
 * test_parity_schema.py` fails when this file and the models disagree.
 */

/**
 * What ``mode="compile"`` answers with when the program ran.
 *
 * ``identities`` is the whole stable-id table for this text; every entry
 * that can carry one also carries its own ``stableId``, so nothing the
 * viewer addresses has to be remembered by line.
 */
export interface CompilePayload {
  ok: true;
  sdf: string;
  shader: string;
  scene_wgsl: string;
  preview_shader: string;
  path_shader: string;
  present_shader: string;
  construction: ConstructionNode[];
  identities: IdentityEntry[];
  relations: ConstructionRelation[];
  materials: MaterialDefinition[];
  studies: StudyPayload[];
  sim_meshes: SimMeshPayload[];
  optimizations: OptimizationPayload[];
  mesh_edges: MeshEdgePayload | null;
  solver_runs: ConstraintSolverRun[];
  output: string;
}

/** What every worker mode answers with when it raised. */
export interface WorkerFailure {
  ok: false;
  error: string;
}

/**
 * One row of the identity table the compile payload publishes.
 *
 * The table exists so the viewer can name anything the payload mentions
 * only by line — an operator chip, a face's owner — without every pinned
 * shape having to grow a field.
 */
export interface IdentityEntry {
  id: string;
  kind: string;
  token: string;
  call: string | null;
  line: number | null;
  index: number | null;
  owner: string | null;
  name: string | null;
  variable: string | null;
}

/** One construction object from the executed program. */
export interface ConstructionNode {
  id: string;
  stableId: string | null;
  kind: string;
  name: string | null;
  line: number | null;
  editable: boolean;
  edges: [number, number, number][][];
  plane: ConstructionPlane | null;
  faces: ConstructionFace[];
  vertices: ConstructionVertex[];
  transform: ConstructionTransform | null;
  spans: Record<string, [number, number]>;
  constraints: ConstructionConstraint[];
  operators: ConstructionOperator[];
  material: string | null;
}

/** The frame a sketch is drawn on, plus the id that addresses it. */
export interface ConstructionPlane {
  origin: [number, number, number];
  u: [number, number, number];
  v: [number, number, number];
  normal: [number, number, number];
  stableId: string | null;
  reference?: PlaneReference | null;
}

/**
 * How a sketch's plane is written in the source.
 *
 * ``SketchPlane.on(body.cap("+"))`` reads back as constructor ``on``,
 * owner ``body``, accessor ``cap``, argument ``'"+"'``.
 */
export interface PlaneReference {
  constructor: string | null;
  owner: string | null;
  accessor: string | null;
  argument: string | null;
}

/** One analytic face of a feature — a reference, not stored geometry. */
export interface ConstructionFace {
  id: string;
  stableId: string | null;
  ownerStableId: string | null;
  key: string;
  kind: string;
  origin: [number, number, number];
  normal: [number, number, number];
  xAxis: [number, number, number];
  yAxis: [number, number, number];
  polygon: [number, number, number][];
  tolerance: number;
  reference: FaceAccessor;
  owner: FaceOwner | null;
  usable: boolean;
  /** Fields the object's own describe() may add. */
  [key: string]: unknown;
}

/** The call that reproduces a face: ``cap("+")``, ``side(3)``. */
export interface FaceAccessor {
  call: string;
  args: (string | number)[];
  /** Fields the object's own describe() may add. */
  [key: string]: unknown;
}

/** The feature that declared a face, and the variable naming it. */
export interface FaceOwner {
  kind: string;
  line: number;
  variable: string | null;
}

/** A sketch vertex, in both sketch-plane and world coordinates. */
export interface ConstructionVertex {
  stableId: string | null;
  name: string | null;
  free: boolean;
  uv: [number, number];
  world: [number, number, number];
  span: [number, number] | null;
}

/** One constraint attached to a sketch's vertex parameters. */
export interface ConstructionConstraint {
  kind: string;
  vertices: number[];
  value?: number | number[] | null;
  index: number;
  stableId: string | null;
}

/** A constraint relating whole construction objects, not sketch points. */
export interface ConstructionRelation {
  kind: "fixed" | "distance";
  nodes: string[];
  value: number | number[];
}

/** A construction-history call consuming a sketch. */
export interface ConstructionOperator {
  kind: "extrude" | "revolve" | "loft";
  line: number;
}

/** Placement and size of a construction primitive, or a sketch's plane. */
export interface ConstructionTransform {
  position: [number, number, number];
  rotation: [number, number, number];
  dimensions: Record<string, number | number[]>;
  line: number | null;
  call: string;
  positionArgument: string;
  canRotate: boolean;
}

/** Diagnostics captured from one source-level constraint solve. */
export interface ConstraintSolverRun {
  node: string | null;
  method: "newton" | "adam" | "sgd";
  iterations: number;
  losses: number[];
}

/** A named Python ``Material`` definition shown in the material browser. */
export interface MaterialDefinition {
  id: string;
  stableId: string | null;
  name: string;
  line: number;
  editable: boolean;
  color: [number, number, number];
  roughness: number;
  metallic: number;
  opacity: number;
  ior: number;
  reflectivity: number;
  physical?: Record<string, number | null> | null;
  units?: Record<string, string> | null;
  free?: Record<string, boolean> | null;
  spans: Record<string, [number, number]>;
  /** Fields the object's own describe() may add. */
  [key: string]: unknown;
}

/** One ``ThermalStudy``/``ElasticStudy`` declared in the scene program. */
export interface StudyPayload {
  index: number;
  stableId: string | null;
  name: string;
  kind: "thermal" | "elastic";
  resolution?: number | number[] | null;
  bounds?: number[] | null;
  size?: number[] | null;
  mesh?: string | null;
  domain?: DomainEntry | null;
  material?: Record<string, number | string>;
  line: number | null;
  span: [number, number] | null;
  editable: boolean;
  mesh_span?: [number, number] | null;
  domain_span?: [number, number] | null;
  bcs: StudyBc[];
  /** Fields the object's own describe() may add. */
  [key: string]: unknown;
}

/** One boundary condition of a declared study. */
export interface StudyBc {
  type: string;
  nodes: StudySelection;
  stableId: string | null;
  serializable: boolean;
  span: [number, number] | null;
  value?: number | null;
  flux?: number | null;
  vector?: [number, number, number] | null;
  /** Fields the object's own describe() may add. */
  [key: string]: unknown;
}

/**
 * A serialized node selection, mirroring ``cadjoint.fem.selection``.
 *
 * Composite selections nest: ``and``/``or`` carry ``operands``, ``not``
 * carries ``operand``, and the leaves carry their own geometry. The
 * per-kind fields are optional here because one model has to cover all of
 * them; ``kind`` says which are meaningful.
 */
export interface StudySelection {
  kind: string;
  min_corner?: number[] | null;
  max_corner?: number[] | null;
  center?: number[] | null;
  radius?: number | null;
  point?: number[] | null;
  normal?: number[] | null;
  side?: string | null;
  tol?: number | null;
  name?: string | null;
  operands?: StudySelection[] | null;
  operand?: StudySelection | null;
  /** Fields the object's own describe() may add. */
  [key: string]: unknown;
}

/** The domain object a mesh or study discretizes, reported by name. */
export interface DomainEntry {
  name: string | null;
  type: string;
  /** Fields the object's own describe() may add. */
  [key: string]: unknown;
}

/** One ``SimMesh`` declared in the scene program. */
export interface SimMeshPayload {
  kind: "mesh";
  index: number;
  stableId: string | null;
  name: string;
  resolution: number | number[];
  bounds?: number[] | null;
  size?: number[] | null;
  padding: number;
  method?: "hex" | "tet4" | "tet10" | null;
  domain?: DomainEntry | null;
  line: number | null;
  span: [number, number] | null;
  editable: boolean;
  /** Fields the object's own describe() may add. */
  [key: string]: unknown;
}

/** One ``Optimization(...)`` declared in the scene program. */
export interface OptimizationPayload {
  kind: "optimization";
  index: number;
  stableId: string | null;
  name: string;
  steps: number;
  learning_rate: number;
  method: string;
  parameters: string[];
  objective?: string | null;
  study?: string | null;
  metric?: string | null;
  remesh_every?: number | null;
  line: number | null;
  span: [number, number] | null;
  editable: boolean;
  steps_span?: [number, number] | null;
  learning_rate_span?: [number, number] | null;
  /** Fields the object's own describe() may add. */
  [key: string]: unknown;
}

/** World-space line segments of the extracted dual-contour mesh. */
export interface MeshEdgePayload {
  wire: [number, number, number][][];
  sharp: [number, number, number][][];
  resolution: number;
}

/** What ``/patch`` answers with: the patched program, or why not. */
export interface PatchResponse {
  ok: boolean;
  source?: string | null;
  error?: string | null;
}

export interface SetVertexRequest {
  source: string;
  id?: string | null;
  line?: number | null;
  op: "set_vertex";
  index?: number | null;
  xy: [number, number];
}

export interface InsertVertexRequest {
  source: string;
  id?: string | null;
  line?: number | null;
  op: "insert_vertex";
  index?: number | null;
  xy: [number, number];
}

export interface DeleteVertexRequest {
  source: string;
  id?: string | null;
  line?: number | null;
  op: "delete_vertex";
  index?: number | null;
}

/** Rewrite one keyword of a construction call, e.g. a box's ``size``. */
export interface SetValueRequest {
  source: string;
  id?: string | null;
  line?: number | null;
  op: "set_value";
  name: string;
  argument: string;
  value: number | number[];
}

export interface AddPrimitiveRequest {
  source: string;
  op: "add_primitive";
  kind: string;
  position: [number, number, number];
  dimensions: Record<string, number | number[]>;
}

export interface AddMaterialRequest {
  source: string;
  op: "add_material";
  color: [number, number, number];
  roughness?: number;
  metallic?: number;
  opacity?: number;
  ior?: number;
  reflectivity?: number;
}

export interface AssignMaterialRequest {
  source: string;
  id?: string | null;
  line?: number | null;
  op: "assign_material";
  material: string;
}

export interface AddSketchRequest {
  source: string;
  op: "add_sketch";
  origin: [number, number, number];
}

export interface SetSketchPlaneRequest {
  source: string;
  id?: string | null;
  line?: number | null;
  op: "set_sketch_plane";
  reference: WorldPlaneReference | CapPlaneReference | SidePlaneReference | FacePlaneReference | TangentPlaneReference;
  x_axis?: [number, number, number] | null;
  flip?: boolean;
  offset?: number | null;
}

export interface AddExtrusionRequest {
  source: string;
  id?: string | null;
  line?: number | null;
  op: "add_extrusion";
  depth?: number;
}

export interface AddRevolutionRequest {
  source: string;
  id?: string | null;
  line?: number | null;
  op: "add_revolution";
  offset?: number;
}

/** A loft joins two sketches, so it names both rather than one target. */
export interface AddLoftRequest {
  source: string;
  op: "add_loft";
  id_a?: string | null;
  id_b?: string | null;
  line_a?: number | null;
  line_b?: number | null;
  height?: number;
}

export interface AddConstraintRequest {
  source: string;
  id?: string | null;
  line?: number | null;
  op: "add_constraint";
  kind: "fixed" | "distance" | "horizontal" | "vertical" | "coincident" | "parallel" | "perpendicular";
  indices: number[];
  value?: number | number[] | null;
}

export interface DeleteConstraintRequest {
  source: string;
  id?: string | null;
  line?: number | null;
  op: "delete_constraint";
  index?: number | null;
}

export interface SetConstraintValueRequest {
  source: string;
  id?: string | null;
  line?: number | null;
  op: "set_constraint_value";
  index?: number | null;
  value: number | number[];
}

export interface SolveSketchRequest {
  source: string;
  id?: string | null;
  line?: number | null;
  op: "solve_sketch";
  method?: "newton" | "adam" | "sgd";
  iterations?: number;
}

export interface DeleteObjectRequest {
  source: string;
  id?: string | null;
  line?: number | null;
  op: "delete_object";
}

export interface AddStudyRequest {
  source: string;
  op: "add_study";
  kind: "thermal" | "elastic";
  name?: string | null;
}

export interface DeleteStudyRequest {
  source: string;
  id?: string | null;
  study?: string | number | null;
  op: "delete_study";
}

export interface AddStudyBcRequest {
  source: string;
  id?: string | null;
  study?: string | number | null;
  op: "add_study_bc";
  bc_type: "dirichlet" | "heat_flux" | "fixed" | "traction";
  selection: Record<string, unknown>;
  value?: number | number[] | null;
}

export interface DeleteStudyBcRequest {
  source: string;
  id?: string | null;
  study?: string | number | null;
  op: "delete_study_bc";
  bc?: number | null;
}

/** Set one boundary condition's value, or one keyword of the study. */
export interface SetStudyValueRequest {
  source: string;
  id?: string | null;
  study?: string | number | null;
  op: "set_study_value";
  bc?: number | null;
  argument?: string | null;
  value: number | number[] | string;
}

export interface AddMeshRequest {
  source: string;
  op: "add_mesh";
  name?: string | null;
}

export interface DeleteMeshRequest {
  source: string;
  id?: string | null;
  mesh?: string | number | null;
  op: "delete_mesh";
}

export interface SetMeshValueRequest {
  source: string;
  id?: string | null;
  mesh?: string | number | null;
  op: "set_mesh_value";
  argument: string;
  value: number | number[] | string;
}

export interface DeleteOptimizationRequest {
  source: string;
  id?: string | null;
  optimization?: string | number | null;
  op: "delete_optimization";
}

export interface SetOptimizationValueRequest {
  source: string;
  id?: string | null;
  optimization?: string | number | null;
  op: "set_optimization_value";
  argument: "steps" | "learning_rate";
  value: number;
}

/** No reference at all: a stated origin and normal. */
export interface WorldPlaneReference {
  kind: "world";
  origin: [number, number, number];
  normal: [number, number, number];
}

export interface CapPlaneReference {
  owner: string | number;
  kind: "cap";
  sign: "+" | "-";
}

export interface SidePlaneReference {
  owner: string | number;
  kind: "side";
  edge: number;
}

export interface FacePlaneReference {
  owner: string | number;
  kind: "face";
  key: string;
}

/**
 * The fallback for a surface with no analytic face: read the plane off
 * the solid's own gradient at the picked point.
 */
export interface TangentPlaneReference {
  owner: string | number;
  kind: "tangent";
  near: [number, number, number];
}

/** Every accepted `/patch` request, discriminated on `op`. */
export type PatchRequest =
  | SetVertexRequest
  | InsertVertexRequest
  | DeleteVertexRequest
  | SetValueRequest
  | AddPrimitiveRequest
  | AddMaterialRequest
  | AssignMaterialRequest
  | AddSketchRequest
  | SetSketchPlaneRequest
  | AddExtrusionRequest
  | AddRevolutionRequest
  | AddLoftRequest
  | AddConstraintRequest
  | DeleteConstraintRequest
  | SetConstraintValueRequest
  | SolveSketchRequest
  | DeleteObjectRequest
  | AddStudyRequest
  | DeleteStudyRequest
  | AddStudyBcRequest
  | DeleteStudyBcRequest
  | SetStudyValueRequest
  | AddMeshRequest
  | DeleteMeshRequest
  | SetMeshValueRequest
  | DeleteOptimizationRequest
  | SetOptimizationValueRequest;

/** The operation names the server accepts. */
export type PatchOperation = PatchRequest["op"];

/** The plane a `set_sketch_plane` request plants a sketch on. */
export type SketchPlaneReference =
  | WorldPlaneReference
  | CapPlaneReference
  | SidePlaneReference
  | FacePlaneReference
  | TangentPlaneReference;

/** What `mode: "compile"` answers with. */
export type CompileResponse = CompilePayload | WorkerFailure;
