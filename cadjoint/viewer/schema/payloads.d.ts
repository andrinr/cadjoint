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
  program?: ShaderProgram | null;
  shader_hash?: string;
  construction: ConstructionNode[];
  identities: IdentityEntry[];
  relations: ConstructionRelation[];
  materials: MaterialDefinition[];
  studies: StudyPayload[];
  sim_meshes: SimMeshPayload[];
  optimizations: OptimizationPayload[];
  mesh_edges: MeshEdgePayload | null;
  tier?: Record<string, boolean> | null;
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
  statementSpan?: [number, number] | null;
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
  binding?: ParameterBinding | null;
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
  bindings?: Record<string, ParameterBinding[]>;
}

/**
 * The free design parameter behind a value a drag can move.
 *
 * The scene's shaders read every *free* parameter out of the uniform buffer
 * described by :class:`ShaderProgram`, so a drag that knows the slot behind
 * the value it is moving can answer a pointer move with a buffer write
 * instead of a source rewrite and a recompile. This is the join between the
 * two halves: ``name`` is the same name :class:`ShaderParameter` carries.
 *
 * ``index`` names the component of the payload value this parameter drives,
 * for the one case where several parameters cover one value — a primitive's
 * ``rotation`` is three separate angle scalars. ``None`` means the parameter
 * covers the whole value.
 *
 * A value with no binding is a fixed literal in the source: absent here,
 * never guessed, and dragged through the ordinary recompile.
 */
export interface ParameterBinding {
  name: string;
  components: number;
  index?: number | null;
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
  mesher?: "tetgen" | "gmsh" | null;
  frozen_geometry?: boolean;
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
  edges?: "graph" | "lattice";
}

/** What ``/patch`` answers with: the patched program, or why not. */
export interface PatchResponse {
  ok: boolean;
  source?: string | null;
  error?: string | null;
}

/**
 * What ``POST /api/export`` takes: which object, which format, how fine.
 *
 * Unlike a patch request this one is the gate as well as the description:
 * :mod:`cadjoint.viewer._export` validates against it before a worker is
 * started, and the message of a failed field is what the dialog shows.
 * The response is the file itself, not JSON — see the module.
 */
export interface ExportRequest {
  source: string;
  format: ExportFormat;
  name?: string;
  resolution?: number;
  binary?: boolean;
  analytic?: boolean;
  merge_planar?: boolean;
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

/**
 * Set, add, or remove one property keyword on a ``Material(...)`` call.
 *
 * Optical properties are always stated, so a number rewrites a literal.
 * Physical ones usually are not, so a number the call does not carry is
 * added as a new keyword; ``value: null`` removes it again.
 */
export interface SetMaterialPropertyRequest {
  source: string;
  id?: string | null;
  line?: number | null;
  op: "set_material_property";
  material?: string | number | null;
  property: "roughness" | "metallic" | "opacity" | "ior" | "reflectivity" | "density" | "conductivity" | "specific_heat" | "youngs_modulus" | "poisson_ratio" | "thermal_expansion" | "yield_strength";
  value?: number | null;
  expand?: boolean;
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
  kind: ConstraintKind;
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
  method?: ConstraintSolveMethod;
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
  kind: StudyKind;
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
  bc_type: BoundaryConditionType;
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
  argument: OptimizationArgument;
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

/**
 * The boundary conditions a study accepts, as the wire spells them.
 *
 * The thermal kinds come first, then the elastic ones; the viewer offers
 * them in this order and the ``describe()`` payload's ``type`` field is
 * exactly these values.
 */
export type BoundaryConditionType = "dirichlet" | "heat_flux" | "fixed" | "traction";

/**
 * The sketch constraints the viewer can add.
 *
 * The two valued kinds come first (they take a numeric target), then the
 * relational ones.
 */
export type ConstraintKind = "fixed" | "distance" | "horizontal" | "vertical" | "coincident" | "parallel" | "perpendicular";

/**
 * How a sketch's constraints are satisfied.
 *
 * Distinct from ``OptimizerMethod``: the constraint solver's default is a
 * minimum-norm Newton projection, which has no meaning as a descent
 * method for a design objective.
 */
export type ConstraintSolveMethod = "newton" | "adam" | "sgd";

/**
 * The file formats the viewer's ``File → Export…`` can write.
 *
 * The three geometry formats take an SDF object of the program (the
 * top-level ``scene`` by default); ``vtk`` takes a declared study instead
 * and writes its solved fields, so it only exists where a result does.
 */
export type ExportFormat = "obj" | "stl" | "step" | "vtk";

/**
 * The optimization keywords the viewer may retune.
 *
 * Everything else in an ``Optimization`` constructor is the objective
 * itself, which is code, not a control.
 */
export type OptimizationArgument = "steps" | "learning_rate";

/**
 * One design parameter's slot in the shader's uniform buffer.
 *
 * The shader source is byte-identical for every value of every parameter,
 * so an edit that moves only values is a buffer write rather than a
 * recompile.  ``offset`` is a byte offset into a buffer of 16-byte slots;
 * ``components`` says how many of the slot's four floats are read.
 */
export interface ShaderParameter {
  name: string;
  offset: number;
  components: number;
  value: (number | null)[];
  free: boolean;
}

/**
 * The parameter buffer the scene's shaders read, and where it binds.
 *
 * Present only when the worker emitted the uniform form of the shader
 * (the default); ``None`` means the parameters are literals in the source
 * and every edit needs a fresh module.
 */
export interface ShaderProgram {
  group: number;
  binding: number;
  buffer_bytes: number;
  nan_offset?: number;
  cull_margin_offset?: number | null;
  parameters: ShaderParameter[];
}

/** The physics a declared study solves. */
export type StudyKind = "thermal" | "elastic";

/** Every accepted `/patch` request, discriminated on `op`. */
export type PatchRequest =
  | SetVertexRequest
  | InsertVertexRequest
  | DeleteVertexRequest
  | SetValueRequest
  | AddPrimitiveRequest
  | AddMaterialRequest
  | AssignMaterialRequest
  | SetMaterialPropertyRequest
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
