"""The compile payload, as models rather than as nested dicts.

One model per shape the ``mode="compile"`` response carries, assembled into
:class:`CompilePayload`. The worker builds the payload exactly as it always
did and hands it here to be checked, so these models describe the wire
rather than dictating it — but a field that changes shape without changing
here now fails loudly at the boundary instead of quietly in the browser.

Two rules run through the file:

- **Names are wire names.** ``stableId``, ``xAxis``, ``mesh_span``: the
  payload mixes camelCase (viewer-authored fields) and snake_case (fields
  that come straight out of a ``describe()``), and the models spell each
  one the way it actually travels, so no alias layer can drift.
- **Describe-derived shapes allow extras.** A study's ``describe()`` grows
  keys as the FEM layer grows (``regularizer``, ``source``); those models
  say ``extra="allow"`` and the generated TypeScript carries an index
  signature, so the schema documents the guaranteed core without freezing
  what the solver may add.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

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


class ConstructionConstraint(Strict):
    """One constraint attached to a sketch's vertex parameters."""

    kind: str
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


class ConstructionNode(Strict):
    """One construction object from the executed program."""

    id: str
    stableId: str | None
    kind: str
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
    physical: dict[str, float | None] | None = None
    units: dict[str, str] | None = None
    free: dict[str, bool] | None = None
    spans: dict[str, Span]


# ── Declarations ────────────────────────────────────────────────────────────


class StudySelection(Open):
    """A serialized node selection, mirroring ``cadjoint.fem.selection``.

    Composite selections nest: ``and``/``or`` carry ``operands``, ``not``
    carries ``operand``, and the leaves carry their own geometry. The
    per-kind fields are optional here because one model has to cover all of
    them; ``kind`` says which are meaningful.
    """

    kind: str
    min_corner: list[float] | None = None
    max_corner: list[float] | None = None
    center: list[float] | None = None
    radius: float | None = None
    point: list[float] | None = None
    normal: list[float] | None = None
    side: str | None = None
    tol: float | None = None
    name: str | None = None
    operands: list[StudySelection] | None = None
    operand: StudySelection | None = None


class StudyBc(Open):
    """One boundary condition of a declared study."""

    type: str
    nodes: StudySelection
    stableId: str | None
    serializable: bool
    span: Span | None
    value: float | None = None
    flux: float | None = None
    vector: Vector3 | None = None


class DomainEntry(Open):
    """The domain object a mesh or study discretizes, reported by name."""

    name: str | None
    type: str


class StudyPayload(Open):
    """One ``ThermalStudy``/``ElasticStudy`` declared in the scene program."""

    index: int
    stableId: str | None
    name: str
    kind: Literal["thermal", "elastic"]
    resolution: int | list[int] | None = None
    bounds: list[float] | None = None
    size: list[float] | None = None
    mesh: str | None = None
    domain: DomainEntry | None = None
    material: dict[str, float | str] = Field(default_factory=dict)
    line: int | None
    span: Span | None
    editable: bool
    mesh_span: Span | None = None
    domain_span: Span | None = None
    bcs: list[StudyBc]


class SimMeshPayload(Open):
    """One ``SimMesh`` declared in the scene program."""

    kind: Literal["mesh"]
    index: int
    stableId: str | None
    name: str
    resolution: int | list[int]
    bounds: list[float] | None = None
    size: list[float] | None = None
    padding: float
    method: Literal["hex", "tet4", "tet10"] | None = None
    # Which volume mesher fills a tet mesh, and whether its nodes can follow
    # the design in this process: a Gmsh mesh is frozen geometry without the
    # ``node_map`` plugin kind (:mod:`cadjoint.tier`).
    mesher: Literal["tetgen", "gmsh"] | None = None
    frozen_geometry: bool = False
    domain: DomainEntry | None = None
    line: int | None
    span: Span | None
    editable: bool


class OptimizationPayload(Open):
    """One ``Optimization(...)`` declared in the scene program."""

    kind: Literal["optimization"]
    index: int
    stableId: str | None
    name: str
    steps: int
    learning_rate: float
    method: str
    parameters: list[str]
    objective: str | None = None
    study: str | None = None
    metric: str | None = None
    remesh_every: int | None = None
    line: int | None
    span: Span | None
    editable: bool
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
    construction: list[ConstructionNode]
    identities: list[IdentityEntry]
    relations: list[ConstructionRelation]
    materials: list[MaterialDefinition]
    studies: list[StudyPayload]
    sim_meshes: list[SimMeshPayload]
    optimizations: list[OptimizationPayload]
    mesh_edges: MeshEdgePayload | None
    # Per private plugin kind, whether it is filled in the worker's process
    # (:func:`cadjoint.tier.status`).  Optional with a default, so a client
    # built before the seam sees no change.
    tier: dict[str, bool] | None = None
    solver_runs: list[ConstraintSolverRun]
    output: str


class WorkerFailure(Strict):
    """What every worker mode answers with when it raised."""

    ok: Literal[False]
    error: str


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
