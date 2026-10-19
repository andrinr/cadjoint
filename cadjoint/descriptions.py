"""The shapes ``describe()`` returns.

Every declaration the viewer displays describes itself as plain JSON.  These
are those shapes, as :class:`~typing.TypedDict` so core needs nothing beyond
the standard library.  The viewer's wire models
(:mod:`cadjoint.viewer.schema.payloads`) extend them with where each one is
written in the source; ``tests/viewer/test_descriptions.py`` holds the two
in step.
"""

from __future__ import annotations

from typing import Literal, TypeAlias, TypedDict

Vector = list[float]

# ── node selections ─────────────────────────────────────────────────────────


class BoxSelectionDescription(TypedDict):
    kind: Literal["box"]
    min_corner: Vector
    max_corner: Vector


class SphereSelectionDescription(TypedDict):
    kind: Literal["sphere"]
    center: Vector
    radius: float


class HalfspaceSelectionDescription(TypedDict):
    kind: Literal["halfspace"]
    point: Vector
    normal: Vector


class CylinderSelectionDescription(TypedDict):
    kind: Literal["cylinder"]
    center: Vector
    axis: Vector
    radius: float
    inner: float
    half_length: float | None


class SideSelectionDescription(TypedDict):
    kind: Literal["side"]
    side: str
    tol: float | None


class PredicateSelectionDescription(TypedDict):
    kind: Literal["predicate"]
    name: str


class CombinedSelectionDescription(TypedDict):
    kind: Literal["and", "or"]
    operands: list[SelectionDescription]


class ComplementSelectionDescription(TypedDict):
    kind: Literal["not"]
    operand: SelectionDescription


SelectionDescription: TypeAlias = (
    BoxSelectionDescription
    | SphereSelectionDescription
    | HalfspaceSelectionDescription
    | CylinderSelectionDescription
    | SideSelectionDescription
    | PredicateSelectionDescription
    | CombinedSelectionDescription
    | ComplementSelectionDescription
)

# ── boundary conditions ─────────────────────────────────────────────────────


class DirichletDescription(TypedDict):
    type: Literal["dirichlet"]
    nodes: SelectionDescription
    value: float


class HeatFluxDescription(TypedDict):
    type: Literal["heat_flux"]
    nodes: SelectionDescription
    flux: float


class FixedDescription(TypedDict):
    type: Literal["fixed"]
    nodes: SelectionDescription


class TractionDescription(TypedDict):
    type: Literal["traction"]
    nodes: SelectionDescription
    vector: Vector


class InletDescription(TypedDict):
    type: Literal["inlet"]
    velocity: Vector
    temperature: float


class OutletDescription(TypedDict):
    type: Literal["outlet"]


class WallsDescription(TypedDict):
    type: Literal["walls"]
    temperature: float | None


class HeatSourceDescription(TypedDict):
    type: Literal["heat_source"]
    nodes: SelectionDescription
    power: float


class HeldTemperatureDescription(TypedDict):
    type: Literal["held_temperature"]
    nodes: SelectionDescription
    value: float


BoundaryConditionDescription: TypeAlias = (
    DirichletDescription
    | HeatFluxDescription
    | FixedDescription
    | TractionDescription
    | InletDescription
    | OutletDescription
    | WallsDescription
    | HeatSourceDescription
    | HeldTemperatureDescription
)

# ── meshes and studies ──────────────────────────────────────────────────────


class DomainDescription(TypedDict):
    name: str | None
    type: str


class SimMeshDescription(TypedDict):
    kind: Literal["mesh"]
    name: str
    method: str
    mesher: str
    frozen_geometry: bool
    resolution: int | list[int]
    bounds: Vector | None
    size: Vector | None
    padding: float
    domain: DomainDescription | None


class _StudyDescription(TypedDict):
    name: str
    resolution: int | list[int] | None
    bounds: Vector | None
    size: Vector | None
    mesh: str | None
    domain: DomainDescription | None
    bcs: list[BoundaryConditionDescription]


class ThermalStudyDescription(_StudyDescription):
    kind: Literal["thermal"]
    material: dict[Literal["conductivity"], float | str]
    source: float


class ElasticStudyDescription(_StudyDescription):
    kind: Literal["elastic"]
    material: dict[Literal["youngs", "poisson"], float | str]
    gravity: Vector | None


class FluidDescription(TypedDict):
    reynolds: float | None
    viscosity: float
    inlet_speed: float
    alpha_max: float
    scheme: str
    peclet_cell: float
    richardson: float


class FlowStudyDescription(_StudyDescription):
    kind: Literal["flow"]
    material: dict[Literal["conductivity_ratio", "prandtl"], float]
    fluid: FluidDescription


StudyDescription: TypeAlias = (
    ThermalStudyDescription | ElasticStudyDescription | FlowStudyDescription
)


class FieldSummary(TypedDict):
    min: float
    mean: float
    max: float


class RefinementDescription(TypedDict):
    declared: list[int]
    used: list[int]
    attempts: int


class ResultDescription(TypedDict):
    name: str
    kind: str
    field: str
    mesh: str | None
    nodes: int
    elements: int
    range: Vector
    fields: dict[str, FieldSummary]
    mass: float | None
    safety_factor: float | None
    refinement: RefinementDescription | None


class OptimizationDescription(TypedDict):
    kind: Literal["optimization"]
    name: str
    steps: int
    learning_rate: float
    method: str
    parameters: list[str]
    objective: str
    study: str | None
    metric: str | None
    remesh_every: int | None
    regularizer: str | None
    regularizer_weight: float


# ── materials and construction ──────────────────────────────────────────────


class MaterialDescription(TypedDict):
    color: Vector
    roughness: float
    metallic: float
    opacity: float
    ior: float
    reflectivity: float
    physical: dict[str, float | None]
    units: dict[str, str]
    free: dict[str, bool]


class FaceReference(TypedDict):
    call: str
    args: list[str | float]


class FaceDescription(TypedDict):
    key: str
    kind: str
    origin: Vector
    normal: Vector
    xAxis: Vector
    yAxis: Vector
    polygon: list[Vector]
    tolerance: float
    reference: FaceReference


class PlaneDescription(TypedDict):
    origin: Vector
    u: Vector
    v: Vector
    normal: Vector
