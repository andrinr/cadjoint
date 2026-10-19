/**
 * Generated from the pydantic models in cadjoint/viewer/schema — do not edit.
 *
 * Regenerate with:
 *
 *     python -m cadjoint.viewer.schema.emit
 *
 * The server checks every worker reply against those same models before a
 * route answers (cadjoint/viewer/schema/worker.py), so a type here is a
 * guarantee, not a hope. `tests/viewer/test_parity_schema.py` fails when
 * this file and the models disagree, and `tests/viewer/test_api_contract.py`
 * when the models and the server's routes do.
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
  table_hash?: string | null;
  construction: ConstructionNode[];
  identities: IdentityEntry[];
  elements?: ConstructionElement[];
  scene_tree?: SceneTree | null;
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

/**
 * What every worker mode — and every endpoint — answers with when it fails.
 *
 * ``output`` is what the program printed before it failed, when the mode
 * captured it; ``error_kind`` is set only for the reasons in
 * :data:`ErrorKind`.
 */
export interface WorkerFailure {
  ok: false;
  error: string;
  error_kind?: "cancelled" | "fem_unavailable" | "contract" | null;
  output?: string | null;
}

/** What every route answers with when it refuses or fails. */
export interface Failure {
  ok: false;
  error: string;
  error_kind?: "cancelled" | "fem_unavailable" | "contract" | null;
  output?: string | null;
  job_id?: string | null;
}

/** What the job registry adds to a tracked route's response. */
export interface JobStamped {
  job_id: string;
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

/** The request that rewrites one argument: which op, call, keyword, and target. */
export interface PatchAddress {
  op: "set_value" | "assign_material";
  name: string;
  argument: string;
  line: number;
  id: string | null;
}

/**
 * One argument a construction call was written with, and whether it can be edited.
 *
 * ``kind`` says what the source holds: a ``number`` or ``vector`` literal
 * (possibly reached through a named parameter, in which case ``parameter``
 * names it), a ``string``, a ``reference`` to another object, a ``default``
 * the call never stated, or an ``expression`` the viewer may only show.
 * ``patch`` is present exactly when the patch layer may rewrite the value.
 */
export interface ConstructionArgument {
  name: string;
  kind: "number" | "vector" | "string" | "reference" | "expression" | "default";
  value: number | number[] | string | null;
  text: string;
  span: [number, number] | null;
  parameter: string | null;
  patch: PatchAddress | null;
}

/**
 * One construction call of the program, as the properties window lists it.
 *
 * Built statically from the source (see
 * :mod:`cadjoint.viewer.source_map.elements`): sketches and their planes,
 * primitives, features and booleans, each with the arguments it was written
 * with.  ``id`` is the stable identity where the program has one for the
 * element, else a synthetic id that is still stable under edits elsewhere.
 */
export interface ConstructionElement {
  id: string;
  stableId: string | null;
  kind: "sketch" | "plane" | "primitive" | "feature" | "boolean";
  call: string;
  line: number;
  span: [number, number];
  name: string | null;
  variable: string | null;
  owner: string | null;
  arguments: ConstructionArgument[];
}

/** One construction object from the executed program. */
export interface ConstructionNode {
  id: string;
  stableId: string | null;
  kind: "profile" | "box" | "sphere" | "cylinder";
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
  kind: "fixed" | "distance" | "horizontal" | "vertical" | "coincident" | "equal_length" | "parallel" | "perpendicular";
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
  physical: Record<string, number | null>;
  units: Record<string, string>;
  free: Record<string, boolean>;
  spans: Record<string, [number, number]>;
  /** Fields the object's own describe() may add. */
  [key: string]: unknown;
}

/**
 * A study's own ``describe()`` (:data:`cadjoint.descriptions.StudyDescription`).
 *
 * ``source`` is a thermal study's, ``gravity`` an elastic study's and
 * ``fluid`` a flow study's.  A flow study discretises a fixed lattice, so
 * it carries a ``resolution`` and no ``mesh``.
 */
export interface StudyDescription {
  name: string;
  kind: "thermal" | "elastic" | "flow";
  resolution?: number | number[] | null;
  bounds?: number[] | null;
  size?: number[] | null;
  mesh?: string | null;
  domain?: DomainEntry | null;
  material?: Record<string, number | string>;
  source?: number | null;
  gravity?: [number, number, number] | null;
  fluid?: FluidDescription | null;
  bcs: StudyBcDescription[];
  /** Fields the object's own describe() may add. */
  [key: string]: unknown;
}

/** A flow study's fluid and scheme, as :meth:`cadjoint.flow.FlowStudy.describe` reports it. */
export interface FluidDescription {
  reynolds: number | null;
  viscosity: number;
  inlet_speed: number;
  alpha_max: number;
  scheme: string;
  peclet_cell: number;
  richardson: number;
}

/**
 * One study declared in the scene program.
 *
 * ``kind`` names what a program may contain;
 * :class:`~cadjoint.enums.StudyKind` names what the patch endpoints can
 * write.  A kind here and not in the enum reports ``editable: false``.
 */
export interface StudyPayload {
  index: number;
  stableId: string | null;
  line: number | null;
  span: [number, number] | null;
  editable: boolean;
  name: string;
  kind: "thermal" | "elastic" | "flow";
  resolution?: number | number[] | null;
  bounds?: number[] | null;
  size?: number[] | null;
  mesh?: string | null;
  domain?: DomainEntry | null;
  material?: Record<string, number | string>;
  source?: number | null;
  gravity?: [number, number, number] | null;
  fluid?: FluidDescription | null;
  bcs: StudyBc[];
  mesh_span?: [number, number] | null;
  domain_span?: [number, number] | null;
  /** Fields the object's own describe() may add. */
  [key: string]: unknown;
}

/**
 * One boundary condition as its own ``describe()`` reports it.
 *
 * ``nodes`` is optional because not every condition is placed by selecting
 * nodes: a flow study's inlet, outlet and duct walls are planes of the
 * lattice, and carry an inlet velocity or a wall temperature instead.
 * ``serializable`` still says whether the condition round-trips, but it is
 * now the condition's own answer rather than its selection's.
 *
 * This is the shape a solved study echoes (``/api/simulate``'s ``study``);
 * the compile payload's :class:`StudyBc` adds where the condition is
 * written.
 */
export interface StudyBcDescription {
  type: BoundaryConditionType;
  nodes?: StudySelection | null;
  serializable: boolean;
  value?: number | null;
  flux?: number | null;
  vector?: [number, number, number] | null;
  velocity?: [number, number, number] | null;
  temperature?: number | null;
  power?: number | null;
  /** Fields the object's own describe() may add. */
  [key: string]: unknown;
}

/** One boundary condition of a declared study, with its place in the source. */
export interface StudyBc {
  type: BoundaryConditionType;
  nodes?: StudySelection | null;
  serializable: boolean;
  value?: number | null;
  flux?: number | null;
  vector?: [number, number, number] | null;
  velocity?: [number, number, number] | null;
  temperature?: number | null;
  power?: number | null;
  stableId: string | null;
  span: [number, number] | null;
  /** Fields the object's own describe() may add. */
  [key: string]: unknown;
}

/** ``Nodes.box``: the nodes inside an axis-aligned box. */
export interface BoxSelection {
  kind: "box";
  min_corner: number[];
  max_corner: number[];
}

/** ``Nodes.sphere``: the nodes inside a ball. */
export interface SphereSelection {
  kind: "sphere";
  center: number[];
  radius: number;
}

/** ``Nodes.halfspace``: the nodes on the side of a plane its normal points to. */
export interface HalfspaceSelection {
  kind: "halfspace";
  point: number[];
  normal: number[];
}

/**
 * ``Nodes.cylinder``: the nodes inside a cylinder, or a tube when ``inner > 0``.
 *
 * ``half_length`` bounds it along ``axis`` either side of ``center``; null
 * leaves it unbounded.
 */
export interface CylinderSelection {
  kind: "cylinder";
  center: number[];
  axis: number[];
  radius: number;
  inner: number;
  half_length: number | null;
}

/** ``Nodes.side``: the extreme boundary plane of the mesh on one axis side. */
export interface SideSelection {
  kind: "side";
  side: string;
  tol: number | null;
}

/** ``Nodes.predicate``: a Python callable, which the viewer can name but not evaluate. */
export interface PredicateSelection {
  kind: "predicate";
  name: string;
}

/** ``a & b`` and ``a | b``. */
export interface CombinedSelection {
  kind: "and" | "or";
  operands: StudySelection[];
}

/** ``~a``. */
export interface ComplementSelection {
  kind: "not";
  operand: StudySelection;
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
  index: number;
  stableId: string | null;
  line: number | null;
  span: [number, number] | null;
  editable: boolean;
  kind: "mesh";
  name: string;
  resolution: number | number[];
  bounds?: number[] | null;
  size?: number[] | null;
  padding: number;
  method?: "hex" | "tet4" | "tet10" | "cutfem" | null;
  mesher?: "tetgen" | "gmsh" | null;
  frozen_geometry?: boolean;
  domain?: DomainEntry | null;
  /** Fields the object's own describe() may add. */
  [key: string]: unknown;
}

/** One ``Optimization(...)`` declared in the scene program. */
export interface OptimizationPayload {
  index: number;
  stableId: string | null;
  line: number | null;
  span: [number, number] | null;
  editable: boolean;
  kind: "optimization";
  name: string;
  steps: number;
  learning_rate: number;
  method: string;
  parameters: string[];
  objective?: string | null;
  study?: string | null;
  metric?: string | null;
  remesh_every?: number | null;
  regularizer?: string | null;
  regularizer_weight?: number | null;
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

/** The derived B-rep, or why there is none (``available`` false). */
export interface BRepInspectionPayload {
  available: boolean;
  reason?: string | null;
  error?: string | null;
  positions?: number[] | null;
  triangles?: number[] | null;
  triangle_face?: number[] | null;
  faces?: BRepFacePayload[] | null;
  edges?: BRepEdgePayload[] | null;
  vertices?: BRepVertexPayload[] | null;
  diagnostics?: BRepDiagnostics | null;
}

/** One face of the derived B-rep, for the inspector's readout. */
export interface BRepFacePayload {
  id: number;
  kind: string;
  surface: string;
  analytic: boolean;
  patch: number;
  leaf: number;
  leaf_type?: string | null;
  area?: number | null;
}

/** One face-pair boundary chain, kinded by how it reached the screen. */
export interface BRepEdgePayload {
  id: number;
  kind: string;
  state: string;
  faces: number[];
  patches: number[];
  vertices: number[];
  closed: boolean;
  analytic: boolean;
  residual?: number | null;
  crossing?: number | null;
  points: number[];
  off_surface: number;
}

/** One solved corner of the derived B-rep. */
export interface BRepVertexPayload {
  id: number;
  kind: string;
  point: [number, number, number];
  faces: number[];
  residual?: number | null;
}

/** Counts, the path that produced the B-rep, and why the exact one refused. */
export interface BRepDiagnostics {
  faces: number;
  edges: number;
  vertices: number;
  source: string;
  exact: boolean;
  fallback_reason?: string | null;
  surfaces: Record<string, number>;
  edge_kinds: Record<string, number>;
  vertex_kinds: Record<string, number>;
  off_surface_tolerance: number;
  off_surface_edges: number;
  feature_curves?: string | null;
  cell: number;
  resolution: number;
  stats: Record<string, number | string | boolean>;
  timings: Record<string, number>;
}

/** ``GET /api/session``: the write token, the starter program, the contract version. */
export interface SessionResponse {
  ok: true;
  token: string;
  example: string;
  api_version: number;
}

/** One private plugin kind's standing in the server process (:mod:`cadjoint.tier`). */
export interface KindStatus {
  kind: string;
  available: boolean;
  plugin: string | null;
  version: string | null;
  contract_version: number | null;
  compatible: boolean;
  reason: string | null;
}

/** ``GET /api/capabilities``: which private plugin kinds this server can reach. */
export interface CapabilitiesResponse {
  ok: true;
  installed: boolean;
  provider: string;
  contract_version: number;
  kinds: Record<string, KindStatus>;
}

/** A request that carries nothing but the program. */
export interface SourceRequest {
  source: string;
}

/** A command that takes no arguments; the client sends ``{}``. */
export interface EmptyRequest {
}

/** ``POST /api/compile`` and ``POST /api/mesh``. */
export interface CompileRequest {
  source: string;
  client_id?: string | null;
}

/** ``POST /api/mesh``: the dual-contour mesh edges of the program. */
export interface MeshResult {
  ok: true;
  mesh_edges: MeshEdgePayload | null;
  output: string;
}

/**
 * ``POST /api/brep``: the derived B-rep, for the inspector.
 *
 * Without the private tier this is still ``ok``: ``brep.available`` is
 * false and ``brep.reason`` says why.
 */
export interface BrepResult {
  ok: true;
  brep: BRepInspectionPayload;
  output: string;
}

/**
 * ``POST /api/mesh_refresh``: move the last extracted overlay to new values.
 *
 * ``source`` is the program the overlay was extracted from — the base of a
 * values-only edit, not the edit — and ``values`` maps free-parameter names
 * to their new components, whole parameters at a time.
 */
export interface MeshRefreshRequest {
  source: string;
  values: Record<string, number[]>;
  certify?: boolean;
}

/** The overlay at the new values, and how much of it could not be certified. */
export interface MeshRefreshResult {
  ok: true;
  mesh_edges: MeshEdgePayload;
  stale: number;
}

/** ``{min, mean, max}`` of one per-element (or per-node) quantity. */
export interface QualitySummary {
  min: number;
  mean: number;
  max: number;
}

/**
 * One boundary face group of a simulation mesh — a BC target.
 *
 * ``axis``/``side`` are set for a hex mesh's gradient-axis groups (``+x``,
 * ``-z``) and null for any other group.
 */
export interface SurfaceGroup {
  id: string;
  axis: "x" | "y" | "z" | null;
  side: "+" | "-" | null;
  center: [number, number, number];
  area: number;
  faces: number;
  start: number;
  count: number;
}

/**
 * A mesh's boundary surface, triangulated, with one scalar per vertex.
 *
 * Hex and tet meshes leave the worker in this one shape.  A solved study
 * adds its whole per-vertex field catalog (``fields``, ``ranges``) and, for
 * an elastic solve, the raw ``displacements`` for a warped view.
 */
export interface SurfacePayload {
  positions: number[];
  scalars: number[];
  indices: number[];
  edges?: number[] | null;
  groups: SurfaceGroup[];
  range: [number, number];
  vertex_count: number;
  fields?: Record<string, number[]> | null;
  ranges?: Record<string, [number, number]> | null;
  displacements?: [number, number, number][] | null;
}

export interface MeshBounds {
  min: number[];
  max: number[];
}

export interface MeshGrid {
  origin: number[];
  spacing: number[];
  cells: number[];
}

/**
 * ``SimMesh.inspect()``: what was built and how good its elements are.
 *
 * Open because it is the library's own ``inspect()``; ``quality`` maps each
 * metric the method has (``scaled_jacobian`` for hexes, ``radius_ratio``
 * and ``aspect_ratio`` for tets) to its summary.
 */
export interface MeshInspectInfo {
  name: string;
  method: string;
  mesher?: string | null;
  frozen_geometry?: boolean | null;
  nodes: number;
  elements: number;
  bounds: MeshBounds;
  grid: MeshGrid | null;
  quality: Record<string, QualitySummary>;
  /** Fields the object's own describe() may add. */
  [key: string]: unknown;
}

/** How far the tet mesher had to re-dice the declared grid. */
export interface Refinement {
  declared: number[];
  used: number[];
  attempts: number;
}

/** ``SimulationResult.describe()``: the solved study, summarised. */
export interface ResultSummary {
  name: string;
  kind: "thermal" | "elastic" | "flow";
  field: string;
  mesh: string | null;
  nodes: number;
  elements: number;
  range: [number, number];
  fields: Record<string, QualitySummary>;
  mass?: number | null;
  safety_factor?: number | null;
  refinement?: Refinement | null;
  /** Fields the object's own describe() may add. */
  [key: string]: unknown;
}

/** ``POST /api/simulate``: solve one study the program declares, by name. */
export interface SimulateRequest {
  source: string;
  kind?: "study";
  name: string;
  cached?: boolean;
}

/** A solved study: its declaration, its render surface, and its summaries. */
export interface SimulateResult {
  ok: true;
  kind: "study";
  study: StudyDescription;
  field: string;
  mesh: SurfacePayload;
  result: ResultSummary;
  mesh_info: MeshInspectInfo | null;
  cached: boolean;
  output: string;
}

/**
 * ``POST /api/mesh_inspect``: build a declared mesh (or a study's) without solving.
 *
 * With no ``name``, the program's one mesh — or, failing that, its one
 * study — is used.
 */
export interface MeshInspectRequest {
  source: string;
  name?: string | null;
}

/** A built mesh: its report and a quality heatmap of its surface. */
export interface MeshInspectResult {
  ok: true;
  kind: "mesh_inspect";
  name: string;
  field: string | null;
  info: MeshInspectInfo;
  mesh: SurfacePayload;
  quality_scalars: number[];
  output: string;
}

/** ``POST /api/optimize``: run one declared ``Optimization`` by name. */
export interface OptimizeRequest {
  source: string;
  name: string;
  steps?: number | null;
}

/** One recorded optimizer evaluation. */
export interface OptimizeStep {
  step: number;
  objective: number;
  grad_norm: number;
}

/** One replayable trajectory frame; step 0 is the initial design. */
export interface OptimizeFrame {
  step: number;
  objective: number;
  parameters: Record<string, number | number[]>;
}

/** A study-backed run's final design, solved — the shapes ``/api/simulate`` uses. */
export interface OptimizeSimulateBlock {
  field: string;
  mesh: SurfacePayload;
  result: ResultSummary;
  mesh_info: MeshInspectInfo | null;
}

/** A finished run: the program with the optimized values written back. */
export interface OptimizeResult {
  ok: true;
  kind: "optimize";
  name: string;
  method: string;
  steps: number;
  source: string;
  history: OptimizeStep[];
  trajectory: OptimizeFrame[];
  parameters: Record<string, number | number[]>;
  initial: Record<string, number | number[]>;
  simulate?: OptimizeSimulateBlock | null;
  output: string;
}

/** One ``/api/optimize`` stream line per finished optimizer step. */
export interface OptimizeProgressEvent {
  event: "progress";
  step: number;
  steps: number;
  objective: number;
  grad_norm: number;
  elapsed: number;
  job_id?: string | null;
}

/** The stream's last line when the run succeeded. */
export interface OptimizeDone {
  ok: true;
  kind: "optimize";
  name: string;
  method: string;
  steps: number;
  source: string;
  history: OptimizeStep[];
  trajectory: OptimizeFrame[];
  parameters: Record<string, number | number[]>;
  initial: Record<string, number | number[]>;
  simulate?: OptimizeSimulateBlock | null;
  output: string;
  event: "done";
  job_id?: string | null;
}

/** The stream's last line when the run (or its request) failed. */
export interface OptimizeFailed {
  ok: false;
  error: string;
  error_kind?: "cancelled" | "fem_unavailable" | "contract" | null;
  output?: string | null;
  job_id?: string | null;
  event: "done";
}

/**
 * ``X-Cadjoint-Export``: what the server says about the file beside its bytes.
 *
 * ``report`` is the writer's own: ``resolution``/``bounds``/``size`` of the
 * lattice for a geometry format, ``study``/``cached`` for ``vtk``.
 */
export interface ExportSummary {
  format: ExportFormat;
  name: string;
  size: number;
  report: Record<string, unknown>;
}

export interface LintFixEdit {
  from_line: number;
  from_col: number;
  to_line: number;
  to_col: number;
  content: string;
}

/** A ruff autofix: a label and the edits that apply it in one go. */
export interface LintFix {
  message: string;
  applicability: "safe" | "unsafe" | "display";
  edits: LintFixEdit[];
}

/** One diagnostic: ruff's, or the last failed compile's traceback line. */
export interface LintDiagnostic {
  from_line: number;
  from_col: number;
  to_line: number;
  to_col: number;
  severity: "error" | "warning" | "info";
  message: string;
  code: string;
  source: "ruff" | "runtime";
  url: string | null;
  fix: LintFix | null;
}

/** ``POST /api/lint``. */
export interface LintResult {
  ok: true;
  diagnostics: LintDiagnostic[];
  runtime: boolean;
}

/** ``POST /api/complete`` and ``POST /api/signature``: a caret in the program. */
export interface CursorRequest {
  source: string;
  line: number;
  column: number;
}

/** One jedi completion, shaped like CodeMirror's ``Completion``. */
export interface CompletionItem {
  label: string;
  type: string;
  detail: string;
  info: string | null;
  apply: string;
}

/** ``POST /api/complete``. */
export interface CompleteResult {
  ok: true;
  from_line: number;
  from_column: number;
  completions: CompletionItem[];
  truncated: boolean;
}

export interface SignatureParameter {
  name: string;
  label: string;
}

export interface SignatureInfo {
  name: string;
  label: string;
  active_parameter: number | null;
  parameters: SignatureParameter[];
  documentation: string | null;
}

/** ``POST /api/signature``. */
export interface SignatureResult {
  ok: true;
  signatures: SignatureInfo[];
}

/** A scene's declarations, counted statically; ``parameters`` only the named ones. */
export interface SceneCounts {
  parameters: number;
  free: number;
  studies: number;
  meshes: number;
  optimizations: number;
  materials: number;
}

/** One scene file, described with ``ast`` and never by running it. */
export interface SceneEntry {
  name: string;
  path: string;
  bytes: number;
  modified: string | null;
  source_hash: string | null;
  summary: string;
  counts: SceneCounts;
  materials: string[];
  error: string | null;
}

/** ``GET /api/scenes``. */
export interface SceneList {
  ok: true;
  files: string[];
  scenes: SceneEntry[];
}

export interface SceneLoadRequest {
  name: string;
}

export interface SceneLoadResult {
  ok: true;
  name: string;
  source: string;
}

export interface SceneSaveRequest {
  name: string;
  source: string;
}

export interface SceneSaveResult {
  ok: true;
  name: string;
}

/** The per-step figures an optimize job mirrors out of its stream. */
export interface JobProgress {
  step?: number | null;
  steps?: number | null;
  objective?: number | null;
  grad_norm?: number | null;
  elapsed?: number | null;
}

/** One resource sample of a running worker, relative to the job's start. */
export interface JobSample {
  t: number;
  cpu_percent: number;
  rss_bytes: number;
}

/** A job as ``GET /api/jobs`` lists it: everything but the result and samples. */
export interface JobSummary {
  job_id: string;
  kind: "compile" | "mesh" | "brep" | "mesh_inspect" | "simulate" | "optimize" | "export" | "lint" | "warmup";
  status: "queued" | "running" | "done" | "failed" | "cancelled";
  fields: Record<string, string | number | boolean>;
  source_hash: string | null;
  source_bytes: number;
  submitted_at: number;
  started_at: number | null;
  finished_at: number | null;
  elapsed_s: number;
  pid: number | null;
  ok: boolean | null;
  error: string | null;
  progress: JobProgress | null;
  cpu_percent: number;
  rss_bytes: number;
  peak_cpu_percent: number;
  peak_rss_bytes: number;
  cpu_seconds: number;
  sampling: "psutil" | "rusage" | "none";
  samples: number;
  result_available: boolean;
  result_bytes: number;
}

export interface JobDetail {
  job_id: string;
  kind: "compile" | "mesh" | "brep" | "mesh_inspect" | "simulate" | "optimize" | "export" | "lint" | "warmup";
  status: "queued" | "running" | "done" | "failed" | "cancelled";
  fields: Record<string, string | number | boolean>;
  source_hash: string | null;
  source_bytes: number;
  submitted_at: number;
  started_at: number | null;
  finished_at: number | null;
  elapsed_s: number;
  pid: number | null;
  ok: boolean | null;
  error: string | null;
  progress: JobProgress | null;
  cpu_percent: number;
  rss_bytes: number;
  peak_cpu_percent: number;
  peak_rss_bytes: number;
  cpu_seconds: number;
  sampling: "psutil" | "rusage" | "none";
  samples: number;
  result_available: boolean;
  result_bytes: number;
  sample_series: JobSample[];
  progress_events: JobProgress[];
}

export interface ServerLoad {
  cpu_percent: number;
  rss_bytes: number;
  pid: number;
}

export interface HostCapacity {
  cpu_count: number | null;
  mem_total: number | null;
  mem_available: number | null;
}

export interface JobTotals {
  running: number;
  cpu_percent: number;
  rss_bytes: number;
  uptime_s: number;
  server: ServerLoad;
  host: HostCapacity;
  sampling: "psutil" | "rusage";
}

export interface JobStore {
  jobs: number;
  max_jobs: number;
  max_lint_jobs: number;
  result_bytes: number;
  max_result_bytes: number;
  evicted_jobs: number;
  evicted_results: number;
}

/** ``GET /api/jobs``: every registered job, newest first, and the live totals. */
export interface JobsSnapshot {
  ok: true;
  jobs: JobSummary[];
  totals: JobTotals;
  store: JobStore;
}

/** ``GET /api/jobs/{job_id}``. */
export interface JobDetailResult {
  ok: true;
  job: JobDetail;
}

/** A job route's refusal: unknown id (404), still running (409), evicted (410). */
export interface JobFailure {
  ok: false;
  error: string;
  error_kind?: "cancelled" | "fem_unavailable" | "contract" | null;
  output?: string | null;
  job_id: string;
  status?: "queued" | "running" | "done" | "failed" | "cancelled" | null;
}

/** ``POST /api/jobs/{job_id}/cancel``. */
export interface JobCancelled {
  ok: true;
  job_id: string;
  status: "cancelled";
}

/** ``POST /api/jobs/clear``: finished jobs dropped; running ones are kept. */
export interface JobsCleared {
  ok: true;
  cleared: number;
  remaining: number;
}

/** ``POST /api/jobs/cancel_client``. */
export interface CancelClientRequest {
  client_id: string;
}

export interface ClientCancelled {
  ok: true;
  client_id: string;
  cancelled: string[];
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
 * :mod:`cadjoint.viewer.worker.export` validates against it before a worker is
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
  reference: SketchPlaneReference;
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
  selection?: StudySelection | null;
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
export type BoundaryConditionType = "dirichlet" | "heat_flux" | "fixed" | "traction" | "inlet" | "outlet" | "walls" | "heat_source" | "held_temperature";

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

/** The evaluated composition of ``scene``: its root ids and every reachable item. */
export interface SceneTree {
  roots: string[];
  items: SceneTreeItem[];
}

/**
 * One node of the SDF graph ``scene`` evaluates to (see :mod:`cadjoint.viewer.source_map.tree`).
 *
 * ``node`` is the construction node a generated solid belongs to, and
 * ``element`` the boolean element a boolean was written as, when known.
 * ``operands`` are item ids, in argument order; an item two booleans both
 * consume is listed once and referenced twice.
 */
export interface SceneTreeItem {
  id: string;
  kind: "node" | "boolean" | "transform" | "sdf";
  call: string;
  name: string | null;
  node: string | null;
  element: string | null;
  operands: string[];
}

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
  depends?: string[];
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
export type StudyKind = "thermal" | "elastic" | "flow";

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

/** A serialized node selection, discriminated on `kind`. */
export type StudySelection =
  | BoxSelection
  | SphereSelection
  | HalfspaceSelection
  | CylinderSelection
  | SideSelection
  | PredicateSelection
  | CombinedSelection
  | ComplementSelection;

/**
 * The contract version this file was generated for (`API_VERSION`).
 * `/api/session` reports the server's; a client compares the two.
 */
export type ApiVersion = 2;

/** Why a request failed, when the client acts on the reason. */
export type ErrorKind = "cancelled" | "fem_unavailable" | "contract";

/** Every constraint kind a compiled sketch can report (wider than `ConstraintKind`). */
export type ReportedConstraintKind = ConstructionConstraint["kind"];

/** Every node-selection kind a study's boundary condition can carry. */
export type SelectionKind = StudySelection["kind"];

/** What `POST /api/compile` answers with. */
export type CompileResponse = CompilePayload | Failure;

/** What `POST /api/mesh` answers with. */
export type MeshResponse = MeshResult | Failure;

/** What `POST /api/brep` answers with. */
export type BrepResponse = BrepResult | Failure;

/** What `POST /api/mesh_refresh` answers with. */
export type MeshRefreshResponse = MeshRefreshResult | Failure;

/** What `POST /api/simulate` answers with. */
export type SimulateResponse = SimulateResult | Failure;

/** What `POST /api/mesh_inspect` answers with. */
export type MeshInspectResponse = MeshInspectResult | Failure;

/** What `POST /api/optimize` answers with. */
export type OptimizeStreamLine = OptimizeProgressEvent | OptimizeDone | OptimizeFailed;

/** What `POST /api/lint` answers with. */
export type LintResponse = LintResult | Failure;

/** What `POST /api/complete` answers with. */
export type CompleteResponse = CompleteResult | Failure;

/** What `POST /api/signature` answers with. */
export type SignatureResponse = SignatureResult | Failure;

/** What `GET /api/scenes` answers with. */
export type SceneListResponse = SceneList | Failure;

/** What `POST /api/scenes/load` answers with. */
export type SceneLoadResponse = SceneLoadResult | Failure;

/** What `POST /api/scenes/save` answers with. */
export type SceneSaveResponse = SceneSaveResult | Failure;

/** What `GET /api/jobs/{job_id}` answers with. */
export type JobDetailResponse = JobDetailResult | JobFailure;

/** What `POST /api/jobs/{job_id}/cancel` answers with. */
export type JobCancelResponse = JobCancelled | JobFailure;

/** What `POST /api/jobs/cancel_client` answers with. */
export type CancelClientResponse = ClientCancelled | Failure;

/** A buffered optimize run's answer: the stream's `done` line without `event`. */
export type OptimizeResponse = OptimizeResult | Failure;

/**
 * Every route the server answers, by path (`{job_id}` is one path segment).
 *
 * `tracked` routes carry `job_id` on every response (`JobStamped`); a
 * `stream` route answers NDJSON, one `response` per line; a `file` route
 * answers a success as an attachment and only a failure as JSON.
 */
export interface ApiRoutes {
  "/api/session": { method: "GET"; request: null; response: SessionResponse; tracked: false; stream: false; file: false };
  "/api/capabilities": { method: "GET"; request: null; response: CapabilitiesResponse; tracked: false; stream: false; file: false };
  "/api/compile": { method: "POST"; request: CompileRequest; response: CompileResponse; tracked: true; stream: false; file: false };
  "/api/patch": { method: "POST"; request: PatchRequest; response: PatchResponse; tracked: false; stream: false; file: false };
  "/api/mesh": { method: "POST"; request: CompileRequest; response: MeshResponse; tracked: true; stream: false; file: false };
  "/api/brep": { method: "POST"; request: CompileRequest; response: BrepResponse; tracked: true; stream: false; file: false };
  "/api/mesh_refresh": { method: "POST"; request: MeshRefreshRequest; response: MeshRefreshResponse; tracked: false; stream: false; file: false };
  "/api/simulate": { method: "POST"; request: SimulateRequest; response: SimulateResponse; tracked: true; stream: false; file: false };
  "/api/mesh_inspect": { method: "POST"; request: MeshInspectRequest; response: MeshInspectResponse; tracked: true; stream: false; file: false };
  "/api/optimize": { method: "POST"; request: OptimizeRequest; response: OptimizeStreamLine; tracked: true; stream: true; file: false };
  "/api/export": { method: "POST"; request: ExportRequest; response: Failure; tracked: true; stream: false; file: true };
  "/api/lint": { method: "POST"; request: SourceRequest; response: LintResponse; tracked: true; stream: false; file: false };
  "/api/complete": { method: "POST"; request: CursorRequest; response: CompleteResponse; tracked: false; stream: false; file: false };
  "/api/signature": { method: "POST"; request: CursorRequest; response: SignatureResponse; tracked: false; stream: false; file: false };
  "/api/scenes": { method: "GET"; request: null; response: SceneListResponse; tracked: false; stream: false; file: false };
  "/api/scenes/load": { method: "POST"; request: SceneLoadRequest; response: SceneLoadResponse; tracked: false; stream: false; file: false };
  "/api/scenes/save": { method: "POST"; request: SceneSaveRequest; response: SceneSaveResponse; tracked: false; stream: false; file: false };
  "/api/jobs": { method: "GET"; request: null; response: JobsSnapshot; tracked: false; stream: false; file: false };
  "/api/jobs/{job_id}": { method: "GET"; request: null; response: JobDetailResponse; tracked: false; stream: false; file: false };
  "/api/jobs/{job_id}/result": { method: "GET"; request: null; response: JobFailure; tracked: false; stream: false; file: false };
  "/api/jobs/{job_id}/cancel": { method: "POST"; request: EmptyRequest; response: JobCancelResponse; tracked: false; stream: false; file: false };
  "/api/jobs/clear": { method: "POST"; request: EmptyRequest; response: JobsCleared; tracked: false; stream: false; file: false };
  "/api/jobs/cancel_client": { method: "POST"; request: CancelClientRequest; response: CancelClientResponse; tracked: false; stream: false; file: false };
}
