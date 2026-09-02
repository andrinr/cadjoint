"""Every ``/patch`` request the server accepts, as one union per operation.

:data:`PATCH_REQUEST_MODELS` has exactly one entry per operation in
``cadjoint.viewer._patch.OPERATIONS``, and a test pins that correspondence,
so an operation added to the registry without a model here fails the build
rather than reaching the frontend undocumented.

These models are the *description* of the wire, not the gate: the checking
that decides whether a request is applied still lives in
:mod:`cadjoint.viewer._patch_requests`, whose rejection strings are the
API. What the models buy is the generated TypeScript — the frontend can
build a request the compiler agrees with — and a cross-check that the two
descriptions of the same contract agree.

Every operation that addresses an existing object takes ``id`` (the stable
identity, resolved against the source in this same request) *or* the legacy
``line``/``index`` the payload used to report. Both are optional in the
models because either one satisfies the server; which combinations are
sufficient is what the validators decide.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from cadjoint.enums import (
    BoundaryConditionType,
    ConstraintKind,
    ConstraintSolveMethod,
    OptimizationArgument,
    StudyKind,
)

Vector2 = Annotated[list[float], Field(min_length=2, max_length=2)]
Vector3 = Annotated[list[float], Field(min_length=3, max_length=3)]
Value = float | list[float]


class PatchBase(BaseModel):
    """Fields every patch request carries."""

    model_config = ConfigDict(extra="forbid")

    source: str
    """The program text the edit is applied to."""


class Targeted(PatchBase):
    """A request that names an object that already exists in the source.

    ``id`` is the durable form and wins when both are given; ``line`` is
    what the payload reported at the last compile and still works.
    """

    id: str | None = None
    line: int | None = None


# ── Sketch vertices ─────────────────────────────────────────────────────────


class SetVertexRequest(Targeted):
    op: Literal["set_vertex"]
    index: int | None = None
    xy: Vector2


class InsertVertexRequest(Targeted):
    op: Literal["insert_vertex"]
    index: int | None = None
    xy: Vector2


class DeleteVertexRequest(Targeted):
    op: Literal["delete_vertex"]
    index: int | None = None


# ── Geometry ────────────────────────────────────────────────────────────────


class SetValueRequest(Targeted):
    """Rewrite one keyword of a construction call, e.g. a box's ``size``."""

    op: Literal["set_value"]
    name: str
    """The called function's name, e.g. ``box`` or ``PolygonProfile``."""
    argument: str
    """The keyword to rewrite; ``planeOrigin``/``planeNormal`` for a sketch."""
    value: Value


class AddPrimitiveRequest(PatchBase):
    op: Literal["add_primitive"]
    kind: str
    position: Vector3
    dimensions: dict[str, Value]


class AddMaterialRequest(PatchBase):
    op: Literal["add_material"]
    color: Vector3
    roughness: float = 0.4
    metallic: float = 0.0
    opacity: float = 1.0
    ior: float = 1.45
    reflectivity: float = 0.0


class AssignMaterialRequest(Targeted):
    op: Literal["assign_material"]
    material: str
    """A Python identifier: the variable the material is bound to."""


MaterialProperty = Literal[
    "roughness",
    "metallic",
    "opacity",
    "ior",
    "reflectivity",
    "density",
    "conductivity",
    "specific_heat",
    "youngs_modulus",
    "poisson_ratio",
    "thermal_expansion",
    "yield_strength",
]
"""Every property one request may set: the scalar optical ones, and all seven
physical ones in SI. ``color`` is a vector and keeps its own editor."""


class SetMaterialPropertyRequest(Targeted):
    """Set, add, or remove one property keyword on a ``Material(...)`` call.

    Optical properties are always stated, so a number rewrites a literal.
    Physical ones usually are not, so a number the call does not carry is
    added as a new keyword; ``value: null`` removes it again.
    """

    op: Literal["set_material_property"]
    material: str | int | None = None
    """The material's name, or its index in the material payload."""
    property: MaterialProperty
    value: float | None = None
    """SI, inside the bracket ``Material`` enforces; null removes the keyword."""
    expand: bool = False
    """Convert a catalogue-built material to a literal first, if it is cheap."""


class AddSketchRequest(PatchBase):
    op: Literal["add_sketch"]
    origin: Vector3


class WorldPlaneReference(BaseModel):
    """No reference at all: a stated origin and normal."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["world"]
    origin: Vector3
    normal: Vector3


class OwnedPlaneReference(BaseModel):
    """A plane read off one face of an existing feature.

    ``owner`` names that feature — by its stable id, or by the 1-based line
    the payload reported for it.
    """

    model_config = ConfigDict(extra="forbid")

    owner: str | int


class CapPlaneReference(OwnedPlaneReference):
    kind: Literal["cap"]
    sign: Literal["+", "-"]


class SidePlaneReference(OwnedPlaneReference):
    kind: Literal["side"]
    edge: int = Field(ge=0)


class FacePlaneReference(OwnedPlaneReference):
    kind: Literal["face"]
    key: str


class TangentPlaneReference(OwnedPlaneReference):
    """The fallback for a surface with no analytic face: read the plane off
    the solid's own gradient at the picked point."""

    kind: Literal["tangent"]
    near: Vector3


SketchPlaneReference = Annotated[
    Union[
        WorldPlaneReference,
        CapPlaneReference,
        SidePlaneReference,
        FacePlaneReference,
        TangentPlaneReference,
    ],
    Field(discriminator="kind"),
]


class SetSketchPlaneRequest(Targeted):
    op: Literal["set_sketch_plane"]
    reference: SketchPlaneReference
    x_axis: Vector3 | None = None
    flip: bool = False
    offset: float | None = None


class AddExtrusionRequest(Targeted):
    op: Literal["add_extrusion"]
    depth: float = 0.5


class AddRevolutionRequest(Targeted):
    op: Literal["add_revolution"]
    offset: float = 0.0


class AddLoftRequest(PatchBase):
    """A loft joins two sketches, so it names both rather than one target."""

    op: Literal["add_loft"]
    id_a: str | None = None
    id_b: str | None = None
    line_a: int | None = None
    line_b: int | None = None
    height: float = 1.0


class DeleteObjectRequest(Targeted):
    op: Literal["delete_object"]


# ── Constraints ─────────────────────────────────────────────────────────────
#
# The option sets a request may name — constraint kinds, study kinds, boundary
# condition types, solver methods — are the enums in :mod:`cadjoint.enums`,
# the same ones the validators check against.  Their JSON schema carries the
# member list into ``payloads.d.ts`` as a named string union.


class AddConstraintRequest(Targeted):
    op: Literal["add_constraint"]
    kind: ConstraintKind
    indices: list[int]
    """One index per point the kind takes: 1 for fixed, 2 or 4 otherwise."""
    value: Value | None = None
    """Required for ``fixed`` and ``distance``; ignored by the rest."""


class DeleteConstraintRequest(Targeted):
    op: Literal["delete_constraint"]
    index: int | None = Field(default=None, ge=0)


class SetConstraintValueRequest(Targeted):
    op: Literal["set_constraint_value"]
    index: int | None = Field(default=None, ge=0)
    value: Value


class SolveSketchRequest(Targeted):
    op: Literal["solve_sketch"]
    method: ConstraintSolveMethod = ConstraintSolveMethod.NEWTON
    iterations: int = Field(default=8, ge=1, le=512)


# ── Studies ─────────────────────────────────────────────────────────────────


class StudyTargeted(PatchBase):
    """A request naming a declared study by id, name, or source-order index."""

    id: str | None = None
    study: str | int | None = None


class AddStudyRequest(PatchBase):
    op: Literal["add_study"]
    kind: StudyKind
    name: str | None = None


class DeleteStudyRequest(StudyTargeted):
    op: Literal["delete_study"]


class AddStudyBcRequest(StudyTargeted):
    op: Literal["add_study_bc"]
    bc_type: BoundaryConditionType
    selection: dict[str, Any]
    """A serialized node selection, as ``StudySelection`` describes it."""
    value: Value | None = None
    """Absent for ``fixed``; three numbers for ``traction``; a scalar else."""


class DeleteStudyBcRequest(StudyTargeted):
    op: Literal["delete_study_bc"]
    bc: int | None = Field(default=None, ge=0)


class SetStudyValueRequest(StudyTargeted):
    """Set one boundary condition's value, or one keyword of the study."""

    op: Literal["set_study_value"]
    bc: int | None = Field(default=None, ge=0)
    argument: str | None = None
    value: float | list[float] | str


# ── Simulation meshes ───────────────────────────────────────────────────────


class MeshTargeted(PatchBase):
    """A request naming a declared mesh by id, name, or source-order index."""

    id: str | None = None
    mesh: str | int | None = None


class AddMeshRequest(PatchBase):
    op: Literal["add_mesh"]
    name: str | None = None


class DeleteMeshRequest(MeshTargeted):
    op: Literal["delete_mesh"]


class SetMeshValueRequest(MeshTargeted):
    op: Literal["set_mesh_value"]
    argument: str
    value: float | list[float] | str


# ── Optimizations ───────────────────────────────────────────────────────────


class OptimizationTargeted(PatchBase):
    """A request naming a declared optimization by id, name, or index."""

    id: str | None = None
    optimization: str | int | None = None


class DeleteOptimizationRequest(OptimizationTargeted):
    op: Literal["delete_optimization"]


class SetOptimizationValueRequest(OptimizationTargeted):
    op: Literal["set_optimization_value"]
    argument: OptimizationArgument
    value: float


#: One model per operation the server accepts.  The test suite pins this
#: against ``cadjoint.viewer._patch.OPERATIONS``, so the two tables cannot
#: diverge without failing.
PATCH_REQUEST_MODELS: dict[str, type[BaseModel]] = {
    "set_vertex": SetVertexRequest,
    "insert_vertex": InsertVertexRequest,
    "delete_vertex": DeleteVertexRequest,
    "set_value": SetValueRequest,
    "add_primitive": AddPrimitiveRequest,
    "add_material": AddMaterialRequest,
    "assign_material": AssignMaterialRequest,
    "set_material_property": SetMaterialPropertyRequest,
    "add_sketch": AddSketchRequest,
    "set_sketch_plane": SetSketchPlaneRequest,
    "add_extrusion": AddExtrusionRequest,
    "add_revolution": AddRevolutionRequest,
    "add_loft": AddLoftRequest,
    "add_constraint": AddConstraintRequest,
    "delete_constraint": DeleteConstraintRequest,
    "set_constraint_value": SetConstraintValueRequest,
    "solve_sketch": SolveSketchRequest,
    "delete_object": DeleteObjectRequest,
    "add_study": AddStudyRequest,
    "delete_study": DeleteStudyRequest,
    "add_study_bc": AddStudyBcRequest,
    "delete_study_bc": DeleteStudyBcRequest,
    "set_study_value": SetStudyValueRequest,
    "add_mesh": AddMeshRequest,
    "delete_mesh": DeleteMeshRequest,
    "set_mesh_value": SetMeshValueRequest,
    "delete_optimization": DeleteOptimizationRequest,
    "set_optimization_value": SetOptimizationValueRequest,
}

PatchRequest = Annotated[
    Union[tuple(PATCH_REQUEST_MODELS.values())],  # type: ignore[valid-type]
    Field(discriminator="op"),
]
"""Every accepted request, discriminated on ``op``."""


class PatchResponse(BaseModel):
    """What ``/patch`` answers with: the patched program, or why not."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    source: str | None = None
    error: str | None = None


def validate_patch_request(request: dict[str, Any]) -> BaseModel:
    """Parse one request against the model for its ``op``.

    Args:
        request: The raw request object.

    Returns:
        The parsed model.

    Raises:
        pydantic.ValidationError: If the request does not match its model.
        KeyError: If ``op`` names no known operation.
    """
    return PATCH_REQUEST_MODELS[request["op"]].model_validate(request)
