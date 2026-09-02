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


class ConstructionVertex(Strict):
    """A sketch vertex, in both sketch-plane and world coordinates."""

    stableId: str | None
    name: str | None
    free: bool
    uv: Vector2
    world: Vector3
    span: Span | None


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
    construction: list[ConstructionNode]
    identities: list[IdentityEntry]
    relations: list[ConstructionRelation]
    materials: list[MaterialDefinition]
    studies: list[StudyPayload]
    sim_meshes: list[SimMeshPayload]
    optimizations: list[OptimizationPayload]
    mesh_edges: MeshEdgePayload | None
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
