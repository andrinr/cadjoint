"""The compile payload, as models rather than as nested dicts.

One model per shape the ``mode="compile"`` response carries, assembled into
:class:`CompilePayload`. The worker builds the payload as plain dicts and
hands it here to be checked, so these models describe the wire rather than
dictating it — but a field that changes shape without changing here fails
loudly at the boundary instead of quietly in the browser.

Two rules run through the file:

- **Names are wire names.** ``stableId``, ``xAxis``, ``mesh_span``: the
  payload mixes camelCase (viewer-authored fields) and snake_case (fields
  that come straight out of a ``describe()``), and the models spell each
  one the way it actually travels, so no alias layer can drift.
- **Describe-derived shapes allow extras.** A study's ``describe()`` grows
  keys as the FEM layer grows (``regularizer``, ``source``); those models
  say ``extra="allow"`` and the generated TypeScript carries an index
  signature, so the schema documents the guaranteed core without freezing
  what the solver may add.  This is deliberate and limited to shapes that
  come out of a library object's own ``describe()``/``inspect()``: the
  studies, meshes, optimizations and their conditions and domains here,
  and the solved-result and mesh-inspection summaries in
  :mod:`cadjoint.viewer.schema.api`.  Everything the viewer package builds
  itself is :class:`Strict`.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from cadjoint.enums import BoundaryConditionType

Span = Annotated[list[int], Field(min_length=2, max_length=2)]
"""``[start, end]`` character offsets into the program text."""

Vector3 = Annotated[list[float], Field(min_length=3, max_length=3)]
Vector2 = Annotated[list[float], Field(min_length=2, max_length=2)]

#: A number, several numbers, or nothing — what a constraint or study value
#: is allowed to be.
Value = float | list[float] | None


class Strict(BaseModel):
    """A shape this package fully controls: no unknown keys allowed."""

    model_config = ConfigDict(extra="forbid")


class Open(BaseModel):
    """A shape built from an object's own ``describe()``, which may grow."""

    model_config = ConfigDict(extra="allow")


# ── Identity ────────────────────────────────────────────────────────────────


class IdentityEntry(Strict):
    """One row of the identity table the compile payload publishes.

    The table exists so the viewer can name anything the payload mentions
    only by line — an operator chip, a face's owner — without every pinned
    shape having to grow a field.
    """

    id: str
    kind: str
    token: str
    call: str | None
    line: int | None
    index: int | None
    owner: str | None
    name: str | None
    variable: str | None


# ── Construction ────────────────────────────────────────────────────────────


class PlaneReference(Strict):
    """How a sketch's plane is written in the source.

    ``SketchPlane.on(body.cap("+"))`` reads back as constructor ``on``,
    owner ``body``, accessor ``cap``, argument ``'"+"'``.
    """

    constructor: str | None
    owner: str | None
    accessor: str | None
    argument: str | None


class ConstructionPlane(Strict):
    """The frame a sketch is drawn on, plus the id that addresses it."""

    origin: Vector3
    u: Vector3
    v: Vector3
    normal: Vector3
    stableId: str | None
    reference: PlaneReference | None = None


class FaceAccessor(Open):
    """The call that reproduces a face: ``cap("+")``, ``side(3)``."""

    call: str
    args: list[str | float]


class FaceOwner(Strict):
    """The feature that declared a face, and the variable naming it."""

    kind: str
    line: int
    variable: str | None


class ConstructionFace(Open):
    """One analytic face of a feature — a reference, not stored geometry."""

    id: str
    stableId: str | None
    ownerStableId: str | None
    key: str
    kind: str
    origin: Vector3
    normal: Vector3
    xAxis: Vector3
    yAxis: Vector3
    polygon: list[Vector3]
    tolerance: float
    reference: FaceAccessor
    owner: FaceOwner | None
    usable: bool


class ParameterBinding(Strict):
    """The free design parameter behind a value a drag can move.

    The scene's shaders read every *free* parameter out of the uniform buffer
    described by :class:`ShaderProgram`, so a drag that knows the slot behind
    the value it is moving can answer a pointer move with a buffer write
    instead of a source rewrite and a recompile. This is the join between the
    two halves: ``name`` is the same name :class:`ShaderParameter` carries.

    ``index`` names the component of the payload value this parameter drives,
    for the one case where several parameters cover one value — a primitive's
    ``rotation`` is three separate angle scalars. ``None`` means the parameter
    covers the whole value.

    A value with no binding is a fixed literal in the source: absent here,
    never guessed, and dragged through the ordinary recompile.
    """

    name: str
    components: int
    index: int | None = None


class ConstructionVertex(Strict):
    """A sketch vertex, in both sketch-plane and world coordinates."""

    stableId: str | None
    name: str | None
    free: bool
    uv: Vector2
    world: Vector3
    span: Span | None
    # The uniform slot the whole point lives in, when it is a free parameter.
    binding: ParameterBinding | None = None


ReportedConstraintKind = Literal[
    "fixed",
    "distance",
    "horizontal",
    "vertical",
    "coincident",
    "equal_length",
    "parallel",
    "perpendicular",
]
"""Every constraint kind a compiled sketch can *report*.

One wider than :class:`cadjoint.enums.ConstraintKind`, which names the kinds
the viewer can *add*: ``equal_length`` is read back from a program that
declares an ``EqualLengthConstraint`` but has no authoring tool.  Pinned
against ``source_map.constraints.CONSTRAINT_CLASS_KINDS`` by a test.
"""


class ConstructionConstraint(Strict):
    """One constraint attached to a sketch's vertex parameters."""

    kind: ReportedConstraintKind
    vertices: list[int]
    value: Value = None
    index: int
    stableId: str | None


class ConstructionRelation(Strict):
    """A constraint relating whole construction objects, not sketch points."""

    kind: Literal["fixed", "distance"]
    nodes: list[str]
    value: float | list[float]


class ConstructionOperator(Strict):
    """A construction-history call consuming a sketch."""

    kind: Literal["extrude", "revolve", "loft"]
    line: int


class ConstructionTransform(Strict):
    """Placement and size of a construction primitive, or a sketch's plane."""

    position: Vector3
    rotation: Vector3
    dimensions: dict[str, float | list[float]]
    line: int | None
    call: str
    positionArgument: str
    canRotate: bool
    # Free-parameter slots behind the draggable arguments, keyed by the
    # argument a drag writes back: ``position``, ``rotation``, and the kind's
    # dimension keywords. An argument appears only when *every* component of
    # it is bound, so a drag never half-applies; a sketch plane's transform
    # binds nothing at all.
    bindings: dict[str, list[ParameterBinding]] = Field(default_factory=dict)


ConstructionKind = Literal["profile", "box", "sphere", "cylinder"]
"""A sketch profile, or one of the primitive kinds of ``cadjoint.construction.solid.DIMENSIONS``."""


class ConstructionNode(Strict):
    """One construction object from the executed program."""

    id: str
    stableId: str | None
    kind: ConstructionKind
    name: str | None
    line: int | None
    editable: bool
    edges: list[list[Vector3]]
    plane: ConstructionPlane | None
    faces: list[ConstructionFace]
    vertices: list[ConstructionVertex]
    transform: ConstructionTransform | None
    spans: dict[str, Span]
    # Character span of the whole statement that declares this object —
    # ``board = Solid.box(...)``, all of it. Separate from ``spans``, which
    # maps *argument* names to their literals: an editor revealing "where is
    # this object" wants the declaration, and a consumer walking ``spans`` as
    # the set of editable arguments must not trip over it.
    statementSpan: Span | None = None
    constraints: list[ConstructionConstraint]
    operators: list[ConstructionOperator]
    material: str | None


class PatchAddress(Strict):
    """The request that rewrites one argument: which op, call, keyword, and target."""

    op: Literal["set_value", "assign_material"]
    name: str
    argument: str
    line: int
    id: str | None


class ConstructionArgument(Strict):
    """One argument a construction call was written with, and whether it can be edited.

    ``kind`` says what the source holds: a ``number`` or ``vector`` literal
    (possibly reached through a named parameter, in which case ``parameter``
    names it), a ``string``, a ``reference`` to another object, a ``default``
    the call never stated, or an ``expression`` the viewer may only show.
    ``patch`` is present exactly when the patch layer may rewrite the value.
    """

    name: str
    kind: Literal["number", "vector", "string", "reference", "expression", "default"]
    value: float | list[float] | str | None
    text: str
    span: Span | None
    parameter: str | None
    patch: PatchAddress | None


class ConstructionElement(Strict):
    """One construction call of the program, as the properties window lists it.

    Built statically from the source (see
    :mod:`cadjoint.viewer.source_map.elements`): sketches and their planes,
    primitives, features and booleans, each with the arguments it was written
    with.  ``id`` is the stable identity where the program has one for the
    element, else a synthetic id that is still stable under edits elsewhere.
    """

    id: str
    stableId: str | None
    kind: Literal["sketch", "plane", "primitive", "feature", "boolean"]
    call: str
    line: int
    span: Span
    name: str | None
    variable: str | None
    owner: str | None
    arguments: list[ConstructionArgument]


class SceneTreeItem(Strict):
    """One node of the SDF graph ``scene`` evaluates to (see :mod:`cadjoint.viewer.source_map.tree`).

    ``node`` is the construction node a generated solid belongs to, and
    ``element`` the boolean element a boolean was written as, when known.
    ``operands`` are item ids, in argument order; an item two booleans both
    consume is listed once and referenced twice.
    """

    id: str
    kind: Literal["node", "boolean", "transform", "sdf"]
    call: str
    name: str | None
    node: str | None
    element: str | None
    operands: list[str]


class SceneTree(Strict):
    """The evaluated composition of ``scene``: its root ids and every reachable item."""

    roots: list[str]
    items: list[SceneTreeItem]


class ConstraintSolverRun(Strict):
    """Diagnostics captured from one source-level constraint solve."""

    node: str | None
    method: Literal["newton", "adam", "sgd"]
    iterations: int
    losses: list[float]


# ── Materials ───────────────────────────────────────────────────────────────


class MaterialDefinition(Open):
    """A named Python ``Material`` definition shown in the material browser."""

    id: str
    stableId: str | None
    name: str
    line: int
    editable: bool
    color: Vector3
    roughness: float
    metallic: float
    opacity: float
    ior: float
    reflectivity: float
    physical: dict[str, float | None]
    units: dict[str, str]
    free: dict[str, bool]
    spans: dict[str, Span]


# ── Declarations ────────────────────────────────────────────────────────────


class BoxSelection(Strict):
    """``Nodes.box``: the nodes inside an axis-aligned box."""

    kind: Literal["box"]
    min_corner: list[float]
    max_corner: list[float]


class SphereSelection(Strict):
    """``Nodes.sphere``: the nodes inside a ball."""

    kind: Literal["sphere"]
    center: list[float]
    radius: float


class HalfspaceSelection(Strict):
    """``Nodes.halfspace``: the nodes on the side of a plane its normal points to."""

    kind: Literal["halfspace"]
    point: list[float]
    normal: list[float]


class CylinderSelection(Strict):
    """``Nodes.cylinder``: the nodes inside a cylinder, or a tube when ``inner > 0``.

    ``half_length`` bounds it along ``axis`` either side of ``center``; null
    leaves it unbounded.
    """

    kind: Literal["cylinder"]
    center: list[float]
    axis: list[float]
    radius: float
    inner: float
    half_length: float | None


class SideSelection(Strict):
    """``Nodes.side``: the extreme boundary plane of the mesh on one axis side."""

    kind: Literal["side"]
    side: str
    tol: float | None


class PredicateSelection(Strict):
    """``Nodes.predicate``: a Python callable, which the viewer can name but not evaluate."""

    kind: Literal["predicate"]
    name: str


class CombinedSelection(Strict):
    """``a & b`` and ``a | b``."""

    kind: Literal["and", "or"]
    operands: list[StudySelection]


class ComplementSelection(Strict):
    """``~a``."""

    kind: Literal["not"]
    operand: StudySelection


StudySelection = Annotated[
    BoxSelection
    | SphereSelection
    | HalfspaceSelection
    | CylinderSelection
    | SideSelection
    | PredicateSelection
    | CombinedSelection
    | ComplementSelection,
    Field(discriminator="kind"),
]
"""A serialized node selection, mirroring ``cadjoint.studies.selection``.

One model per selection class, discriminated on ``kind``, so the frontend
evaluates exactly these shapes with an exhaustive switch.  Closed, although
it is a ``describe()``: a test describes one selection of every class, so a
class added there without a model here fails rather than reaching a
frontend that cannot evaluate it (as ``cylinder`` once did).
"""

SELECTION_MODELS: tuple[type[BaseModel], ...] = (
    BoxSelection,
    SphereSelection,
    HalfspaceSelection,
    CylinderSelection,
    SideSelection,
    PredicateSelection,
    CombinedSelection,
    ComplementSelection,
)

CombinedSelection.model_rebuild()
ComplementSelection.model_rebuild()


class StudyBcDescription(Open):
    """One boundary condition as its own ``describe()`` reports it.

    ``nodes`` is optional because not every condition is placed by selecting
    nodes: a flow study's inlet, outlet and duct walls are planes of the
    lattice, and carry an inlet velocity or a wall temperature instead.
    ``serializable`` says whether the condition round-trips; it is the
    condition's own answer, not its selection's.

    This is the shape a solved study echoes (``/api/simulate``'s ``study``);
    the compile payload's :class:`StudyBc` adds where the condition is
    written.
    """

    type: BoundaryConditionType
    nodes: StudySelection | None = None
    serializable: bool
    value: float | None = None
    flux: float | None = None
    vector: Vector3 | None = None
    velocity: Vector3 | None = None
    temperature: float | None = None
    power: float | None = None


class StudyBc(StudyBcDescription):
    """One boundary condition of a declared study, with its place in the source."""

    stableId: str | None
    span: Span | None


class DomainEntry(Open):
    """The domain object a mesh or study discretizes, reported by name."""

    name: str | None
    type: str


class Located(BaseModel):
    """Where a declaration is written: the fields the worker adds to its ``describe()``."""

    index: int
    stableId: str | None
    line: int | None
    span: Span | None
    editable: bool


class FluidDescription(Strict):
    """A flow study's fluid and scheme, as :meth:`cadjoint.flow.FlowStudy.describe` reports it."""

    reynolds: float | None
    viscosity: float
    inlet_speed: float
    alpha_max: float
    scheme: str
    peclet_cell: float
    richardson: float


class StudyDescription(Open):
    """A study's own ``describe()`` (:data:`cadjoint.descriptions.StudyDescription`).

    ``source`` is a thermal study's, ``gravity`` an elastic study's and
    ``fluid`` a flow study's.  A flow study discretises a fixed lattice, so
    it carries a ``resolution`` and no ``mesh``.
    """

    name: str
    kind: Literal["thermal", "elastic", "flow"]
    resolution: int | list[int] | None = None
    bounds: list[float] | None = None
    size: list[float] | None = None
    mesh: str | None = None
    domain: DomainEntry | None = None
    material: dict[str, float | str] = Field(default_factory=dict)
    source: float | None = None
    gravity: Vector3 | None = None
    fluid: FluidDescription | None = None
    bcs: list[StudyBcDescription]


class StudyPayload(StudyDescription, Located):
    """One study declared in the scene program.

    ``kind`` names what a program may contain;
    :class:`~cadjoint.enums.StudyKind` names what the patch endpoints can
    write.  A kind here and not in the enum reports ``editable: false``.
    """

    mesh_span: Span | None = None
    domain_span: Span | None = None
    bcs: list[StudyBc]


class SimMeshPayload(Located, Open):
    """One ``SimMesh`` declared in the scene program."""

    kind: Literal["mesh"]
    name: str
    resolution: int | list[int]
    bounds: list[float] | None = None
    size: list[float] | None = None
    padding: float
    method: Literal["hex", "tet4", "tet10", "cutfem"] | None = None
    # Which volume mesher fills a tet mesh, and whether its nodes can follow
    # the design in this process: a Gmsh mesh is frozen geometry without the
    # ``node_map`` plugin kind (:mod:`cadjoint.tier`).
    mesher: Literal["tetgen", "gmsh"] | None = None
    frozen_geometry: bool = False
    domain: DomainEntry | None = None


class OptimizationPayload(Located, Open):
    """One ``Optimization(...)`` declared in the scene program."""

    kind: Literal["optimization"]
    name: str
    steps: int
    learning_rate: float
    method: str
    parameters: list[str]
    objective: str | None = None
    study: str | None = None
    metric: str | None = None
    remesh_every: int | None = None
    regularizer: str | None = None
    regularizer_weight: float | None = None
    steps_span: Span | None = None
    learning_rate_span: Span | None = None


class MeshEdgePayload(Strict):
    """World-space line segments of the extracted dual-contour mesh."""

    wire: list[list[Vector3]]
    sharp: list[list[Vector3]]
    resolution: int
    # Which layer produced ``sharp``: the derived B-rep's exact curves
    # (``"graph"``, the private tier), or the lattice feature classifier
    # (``"lattice"``, public cadjoint alone).  See :mod:`cadjoint.tier`.
    edges: Literal["graph", "lattice"] = "graph"


# ── B-rep inspector ─────────────────────────────────────────────────────────
#
# What ``mode="brep"`` serves (:mod:`cadjoint.viewer._brep_inspector`).  The
# kinds and states are plain strings on purpose: they are the private tier's
# vocabulary, passed through, so a new edge state reaches the viewer without
# a schema change.


StepSurface = Literal["plane", "cylinder", "sphere", "cone", "bspline", "faceted", "dropped"]
"""What a B-rep face is written on in the analytic STEP: an exact surface, one
fitted B-spline, its own triangles, or — on the lattice path — nothing."""


class StepFaceExport(Strict):
    """How one B-rep face goes to STEP, as the ``step_export`` kind decides it."""

    surface: StepSurface
    # For a fitted face (a found plane or a B-spline): the largest distance
    # from a tessellation vertex to the surface, and the largest ``|f|`` of
    # the scene over the certificate's probes, both in model units.
    fit_error: float | None = None
    field_error: float | None = None
    control_points: int = 0
    # For a faceted face: why no surface took it.
    reason: str = ""


class StepContents(Strict):
    """What an analytic STEP file holds: faces per surface kind, and the fit bar."""

    faces: int
    surfaces: dict[StepSurface, int]
    # The bar every fitted face is certified to, model units; ``None`` when
    # nothing was fitted (the lattice path).
    fit_tolerance: float | None = None
    # The largest fit error over the fitted faces, or ``None`` without one.
    fit_error_max: float | None = None
    length_unit: Literal["metre"] = "metre"

    @classmethod
    def from_report(cls, report: dict[str, Any]) -> StepContents:
        """The summary of a ``step_export`` report (or plan): ``surfaces``, ``faces``, ``fit_tolerance``."""
        records = report.get("faces") or []
        errors = [r["fit_error"] for r in records if r.get("fit_error") is not None]
        return cls(
            faces=len(records),
            surfaces={kind: count for kind, count in report["surfaces"].items() if count},
            fit_tolerance=report.get("fit_tolerance"),
            fit_error_max=max(errors) if errors else None,
            length_unit=report.get("length_unit", "metre"),
        )


class BRepFacePayload(Strict):
    """One face of the derived B-rep, for the inspector's readout."""

    id: int
    # The private tier's structural kind (``plane``, ``blend``, ``revolution``…).
    kind: str
    # What the viewer prints: a certified closed form, ``blend`` or ``faceted``.
    surface: str
    analytic: bool
    # Global patch index; -1 for a blend face, which lies on no patch.
    patch: int
    leaf: int
    leaf_type: str | None = None
    area: float | None = None
    # How the analytic STEP export writes this face; absent without the
    # ``step_export`` kind.
    export: StepFaceExport | None = None


class BRepEdgePayload(Strict):
    """One face-pair boundary chain, kinded by how it reached the screen."""

    id: int
    # ``traced`` / ``sampled`` when the feature-edge overlay drew it, otherwise
    # the graph's state (``blend``, ``tangent``, …) or ``undrawn``.
    kind: str
    state: str
    faces: list[int]
    patches: list[int]
    vertices: list[int]
    closed: bool
    analytic: bool
    residual: float | None = None
    crossing: float | None = None
    # Flat ``x, y, z`` polyline.
    points: list[float]
    # Largest ``|f|`` of the scene along the curve, in model units.
    off_surface: float


class BRepVertexPayload(Strict):
    """One solved corner of the derived B-rep."""

    id: int
    kind: str
    point: Vector3
    faces: list[int]
    residual: float | None = None


class BRepDiagnostics(Strict):
    """Counts, the path that produced the B-rep, and why the exact one refused."""

    faces: int
    edges: int
    vertices: int
    source: str
    exact: bool
    fallback_reason: str | None = None
    surfaces: dict[str, int]
    edge_kinds: dict[str, int]
    vertex_kinds: dict[str, int]
    off_surface_tolerance: float
    off_surface_edges: int
    feature_curves: str | None = None
    cell: float
    resolution: int
    stats: dict[str, float | int | str | bool]
    timings: dict[str, float]
    # What an analytic STEP export of this B-rep holds; absent without the
    # ``step_export`` kind or when its plan failed (see ``step_note``).
    step: StepContents | None = None
    step_note: str | None = None


class BRepInspectionPayload(Strict):
    """The derived B-rep, or why there is none (``available`` false)."""

    available: bool
    reason: str | None = None
    # The extraction raised: the failure, in one line, is what there is to see.
    error: str | None = None
    # Flat ``x, y, z`` positions, flat triangle indices into them, and the
    # face id of every triangle.
    positions: list[float] | None = None
    triangles: list[int] | None = None
    triangle_face: list[int] | None = None
    # The exact surface normal at every triangle corner, flat ``x, y, z``
    # per corner in triangle order, when the private tier supplies them: the
    # viewer shades with these rather than with the facets, so a smooth blend
    # does not read as ripples.  Absent, the faces are shaded flat.
    corner_normals: list[float] | None = None
    faces: list[BRepFacePayload] | None = None
    edges: list[BRepEdgePayload] | None = None
    vertices: list[BRepVertexPayload] | None = None
    diagnostics: BRepDiagnostics | None = None


class ShaderParameter(Strict):
    """One design parameter's slot in the shader's uniform buffer.

    The shader source is byte-identical for every value of every parameter,
    so an edit that moves only values is a buffer write rather than a
    recompile.  ``offset`` is a byte offset into a buffer of 16-byte slots;
    ``components`` says how many of the slot's four floats are read.
    """

    name: str
    offset: int
    components: int
    # ``null`` for a component that is not finite — a material property the
    # scene never set.  JSON cannot carry a NaN that a strict parser will
    # read back, so the client turns the null into one when it packs the
    # buffer; see ``ShaderParameter.as_dict`` in the WGSL backend.
    value: list[float | None]
    free: bool
    # For a slot derived from parameters rather than one of them — a culling
    # box, ``__cull<n>.center`` / ``.half`` — the parameter slots it was
    # computed from.  While a drag overrides any of them the box no longer
    # holds its operand, so the client opens it (half extents to a huge
    # value) until the compile that ends the drag brings a box that does.
    depends: list[str] = []


class ShaderProgram(Strict):
    """The parameter buffer the scene's shaders read, and where it binds.

    Present only when the worker emitted the uniform form of the shader
    (the default); ``None`` means the parameters are literals in the source
    and every edit needs a fresh module.
    """

    group: int
    binding: int
    buffer_bytes: int
    # Byte offset of a reserved slot the module reads wherever it needs a
    # NaN.  WGSL has no NaN literal a compiler will accept — Chromium's Tint
    # const-evaluates the bit-pattern bitcast and rejects the module — so the
    # value has to arrive through the buffer.  The client writes a NaN here.
    nan_offset: int = 0
    # Byte offset of a reserved slot holding the bounding-box cull margin.
    # Every skip test in the generated module is
    # `box_distance(p, bounds) >= threshold + margin`, so writing an infinite
    # margin here makes every test false and the module computes the flat
    # field.  That is how culling is a render toggle rather than a recompile:
    # the tests live inside the *generated* shader, which reads no other
    # uniform.  The client writes 1e-4 for on, +Infinity for off.
    # `None`, not 0, when no such slot was reserved: 0 names a real
    # parameter's slot, so a program that merely forgot to set this would
    # overwrite that parameter with the margin.
    cull_margin_offset: int | None = None
    parameters: list[ShaderParameter]


# ── The response itself ─────────────────────────────────────────────────────


class SceneGraphMaterial(Strict):
    """A primitive's material in the scene graph: the shader's two material rows.

    A non-finite value is ``null``, as everywhere JSON carries a float.
    """

    color: list[float | None]
    roughness: float | None
    metallic: float | None
    opacity: float | None
    ior: float | None
    reflectivity: float | None


class SceneGraphNode(Strict):
    """One node of a ``cadjoint-scene`` graph.

    The format is specified in ``cadjoint/backends/jaxjs/SCENE_FORMAT.md``;
    ``id``, ``free``, ``implicit`` and ``material`` are the version 2 fields.
    """

    type: str
    params: dict[str, float | list[float] | list[list[float]]]
    children: list[SceneGraphNode]
    id: str | None = None
    free: dict[str, str | list[str | None]] | None = None
    implicit: list[str] | None = None
    material: SceneGraphMaterial | None = None


class SceneGraph(Strict):
    """The scene as a ``cadjoint-scene`` graph, from which a client can build its own WGSL."""

    format: Literal["cadjoint-scene"]
    version: Literal[1, 2]
    root: SceneGraphNode


class CompilePayload(Strict):
    """What ``mode="compile"`` answers with when the program ran.

    ``identities`` is the whole stable-id table for this text; every entry
    that can carry one also carries its own ``stableId``, so nothing the
    viewer addresses has to be remembered by line.
    """

    ok: Literal[True]
    sdf: str
    shader: str
    scene_wgsl: str
    preview_shader: str
    path_shader: str
    present_shader: str
    # The uniform contract for the two scene shaders above, when they were
    # built in the uniform form: the frontend binds this buffer and, on an
    # edit whose sources are unchanged, uploads it instead of recompiling.
    program: ShaderProgram | None = None
    # sha256 of the two scene shaders, so the browser can key its module
    # cache without hashing megabytes of source itself.
    shader_hash: str = ""
    # The scene as a version 2 ``cadjoint-scene`` graph.  A client that builds
    # its own WGSL from it (``frontend/src/scene``) writes the same module as
    # ``scene_wgsl``.  ``None`` when the scene has a node the graph cannot
    # carry, such as a plain callable.
    scene_graph: SceneGraph | None = None
    # Of the scene's node table (:mod:`cadjoint.zeroset`) with its design
    # values left out: unchanged by a free-parameter edit, changed by any
    # other, which is what tells a refresh of the mesh-edge overlay from an
    # extraction.  ``None`` when the scene cannot be lowered to the table.
    table_hash: str | None = None
    construction: list[ConstructionNode]
    identities: list[IdentityEntry]
    # Every construction call with its written arguments, for the properties
    # window; defaults to nothing when absent.
    elements: list[ConstructionElement] = Field(default_factory=list)
    # What ``scene`` is built from, boolean by boolean; when absent the
    # Objects panel falls back to a flat list.
    scene_tree: SceneTree | None = None
    relations: list[ConstructionRelation]
    materials: list[MaterialDefinition]
    studies: list[StudyPayload]
    sim_meshes: list[SimMeshPayload]
    optimizations: list[OptimizationPayload]
    mesh_edges: MeshEdgePayload | None
    # Per private plugin kind, whether it is filled in the worker's process
    # (:func:`cadjoint.tier.status`).  Optional with a default.
    tier: dict[str, bool] | None = None
    solver_runs: list[ConstraintSolverRun]
    output: str


ErrorKind = Literal["cancelled", "fem_unavailable", "contract"]
"""Why a request failed, when the reason is one the client acts on.

``cancelled``: somebody stopped the job (HTTP 409).  ``fem_unavailable``: the
optional solver extra is missing (HTTP 501, shown as an install hint).
``contract``: a worker answered with a shape its model does not allow — a
bug, reported rather than forwarded (see :mod:`cadjoint.viewer.schema.worker`).
"""


class WorkerFailure(Strict):
    """What every worker mode — and every endpoint — answers with when it fails.

    ``output`` is what the program printed before it failed, when the mode
    captured it; ``error_kind`` is set only for the reasons in
    :data:`ErrorKind`.
    """

    ok: Literal[False]
    error: str
    error_kind: ErrorKind | None = None
    output: str | None = None


def validate_compile_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Check a built compile payload against the schema and return it.

    Args:
        payload: The dict the worker assembled.

    Returns:
        The same dict, unchanged, once it validates.

    Raises:
        pydantic.ValidationError: If the payload does not match the models.
    """
    CompilePayload.model_validate(payload)
    return payload
