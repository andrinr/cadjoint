"""Every fixed option set in cadjoint, as one string enum per concept.

An option a user picks from a closed list — a mesh method, a study kind, a
gradient path — is an :class:`enum.StrEnum` here rather than a bare string
scattered across the modules that check it.  The enum is the single place
the members live: constructors normalise to it, error messages are derived
from it, the viewer's validators check against it, and the pydantic request
models carry it into the generated TypeScript as a string union.

Nothing breaks for callers who never import this module.  A ``StrEnum``
member *is* a ``str``: ``MeshMethod.HEX == "hex"``, ``f"{MeshMethod.HEX}"``
prints ``hex``, and every public constructor keeps accepting the plain
spelling::

    SimMesh(name="m", resolution=20, method="tet10")
    SimMesh(name="m", resolution=20, method=MeshMethod.TET10)  # the same mesh

The ``*Like`` aliases are what public signatures annotate: they accept the
enum and its literal spellings alike.

Two details keep the substitution invisible.  Every option set derives from
:class:`Option`, whose ``repr`` is the *value's* repr (``'hex'``, not
``<MeshMethod.HEX: 'hex'>``), so an error message that interpolates one with
``{value!r}`` reads exactly as it did.  And ``describe()`` payloads emit
``str(member)`` so a payload dict holds plain strings — JSON does not care
either way, since a ``StrEnum`` serializes as its value, but a plain string
is what the wire has always carried.

This module deliberately imports nothing from cadjoint: the viewer's schema
layer and the FEM layer both depend on it, and neither should pull the
other in through it.

Example::

    from cadjoint.enums import MeshMethod, listed, values

    values(MeshMethod)   # ("hex", "tet4", "tet10")
    listed(MeshMethod)   # "hex, tet4, tet10"
    MeshMethod("tet4")   # MeshMethod.TET4
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, TypeVar

__all__ = [
    "BoundaryConditionType",
    "BoundaryConditionTypeLike",
    "ConstraintKind",
    "ConstraintKindLike",
    "ConstraintSolveMethod",
    "ConstraintSolveMethodLike",
    "ExportFormat",
    "ExportFormatLike",
    "FemBackend",
    "GradientPath",
    "GradientPathLike",
    "MeshMethod",
    "MeshMethodLike",
    "ObjectiveMetric",
    "ObjectiveMetricLike",
    "Option",
    "OptimizationArgument",
    "OptimizationArgumentLike",
    "OptimizerMethod",
    "OptimizerMethodLike",
    "PluginKind",
    "PluginKindLike",
    "PluginTransport",
    "PluginTransportLike",
    "Side",
    "SideLike",
    "StudyKind",
    "StudyKindLike",
    "TetMesher",
    "TetMesherLike",
    "either",
    "listed",
    "parse",
    "values",
]


class Option(StrEnum):
    """Base for every option set here: a string with a fixed vocabulary.

    The one behaviour it adds to :class:`enum.StrEnum` is ``repr``, which
    reports the value (``'hex'``) rather than the member
    (``<MeshMethod.HEX: 'hex'>``).  Error messages all over cadjoint
    interpolate an offending or accepted value with ``{value!r}``, and a
    member has to read there exactly as the string it replaced.
    """

    def __repr__(self) -> str:
        """The value's repr, so a member prints like the string it is."""
        return repr(self.value)


_Option = TypeVar("_Option", bound=StrEnum)


def values(option: type[StrEnum]) -> tuple[str, ...]:
    """The option set's members as plain strings, in declaration order.

    Args:
        option: A string enum from this module.

    Returns:
        Every member's value, declaration order preserved — the tuple form
        the modules that publish their accepted spellings expose.
    """
    return tuple(member.value for member in option)


def listed(option: type[StrEnum], separator: str = ", ") -> str:
    """The option set's members joined for an error message.

    Args:
        option: A string enum from this module.
        separator: What to join the values with.

    Returns:
        The member values in declaration order, e.g. ``"hex, tet4, tet10"``.
    """
    return separator.join(values(option))


def either(option: type[StrEnum], quote: str = "`") -> str:
    """The option set as a prose alternative, in declaration order.

    Args:
        option: A string enum from this module.
        quote: What to wrap each value in; a backtick by default, because
            the viewer's rejection messages are read as markdown.

    Returns:
        ``"`thermal` or `elastic`"`` for two members, and
        ``"`newton`, `adam`, or `sgd`"`` for three or more.
    """
    quoted = [f"{quote}{value}{quote}" for value in values(option)]
    if len(quoted) < 3:
        return " or ".join(quoted)
    return ", ".join(quoted[:-1]) + f", or {quoted[-1]}"


def parse(option: type[_Option], value: Any, message: str) -> _Option:
    """Normalise one option value, raising ``message`` when it is not a member.

    The enum's own ``ValueError`` names the class rather than the accepted
    spellings, and the accepted spellings are what a caller needs, so every
    boundary in cadjoint converts through here with the message its tests
    (and its users) already know.

    Args:
        option: A string enum from this module.
        value: The user's spelling — a member, or the plain string.
        message: The error text to raise instead of the enum's own.

    Returns:
        The matching member.

    Raises:
        ValueError: If ``value`` is not one of the members.
    """
    try:
        return option(value)
    except ValueError:
        raise ValueError(message) from None


class MeshMethod(Option):
    """How a ``SimMesh`` turns an SDF into elements."""

    HEX = "hex"
    """Voxelize the sampling lattice and snap boundary vertices (HEX8)."""
    TET4 = "tet4"
    """Dual-contour the surface, then fill it with TetGen (linear tets)."""
    TET10 = "tet10"
    """The TET4 mesh promoted to quadratic straight-sided tets."""


MeshMethodLike = MeshMethod | Literal["hex", "tet4", "tet10"]
"""A mesh method, or the plain string spelling of one."""


class TetMesher(Option):
    """Which volume mesher fills a ``SimMesh`` declared ``tet4``/``tet10``.

    Both take the same dual-contour surface; they differ in who sizes the
    elements and in whether the mesh follows the design.
    """

    TETGEN = "tetgen"
    """TetGen on the dual-contour surface (the default): the lattice sizes
    the elements and every node follows the design through
    :func:`cadjoint.fem.motion.recompute_tet_points`."""
    GMSH = "gmsh"
    """Gmsh (HXT, second order) on the same surface, reached through the
    ``tet_mesher`` plugin: the part sizes the elements, each node is tagged
    with the patches that own it, and the design derivative is the
    ``node_map`` plugin's — the private tier's; without it the mesh is
    frozen geometry (see :mod:`cadjoint.tier`)."""


TetMesherLike = TetMesher | Literal["tetgen", "gmsh"]
"""A tet mesher, or the plain string spelling of one."""


class StudyKind(Option):
    """The physics a declared study solves."""

    THERMAL = "thermal"
    """Steady-state heat conduction (``ThermalStudy``)."""
    ELASTIC = "elastic"
    """Small-strain linear elasticity (``ElasticStudy``)."""


StudyKindLike = StudyKind | Literal["thermal", "elastic"]
"""A study kind, or the plain string spelling of one."""


class BoundaryConditionType(Option):
    """The boundary conditions a study accepts, as the wire spells them.

    The thermal kinds come first, then the elastic ones; the viewer offers
    them in this order and the ``describe()`` payload's ``type`` field is
    exactly these values.
    """

    DIRICHLET = "dirichlet"
    """Prescribed temperature on the selected nodes."""
    HEAT_FLUX = "heat_flux"
    """Prescribed heat inflow per area on the faces the selection spans."""
    FIXED = "fixed"
    """Fully clamped nodes (all displacement components zero)."""
    TRACTION = "traction"
    """Constant force per area on the faces the selection spans."""


BoundaryConditionTypeLike = (
    BoundaryConditionType | Literal["dirichlet", "heat_flux", "fixed", "traction"]
)
"""A boundary condition type, or the plain string spelling of one."""


class Side(Option):
    """An axis-extreme face of a mesh, for ``Nodes.side``."""

    PLUS_X = "+x"
    MINUS_X = "-x"
    PLUS_Y = "+y"
    MINUS_Y = "-y"
    PLUS_Z = "+z"
    MINUS_Z = "-z"


SideLike = Side | Literal["+x", "-x", "+y", "-y", "+z", "-z"]
"""A side, or the plain string spelling of one."""


class GradientPath(Option):
    """How a study-form optimization carries the design->points derivative."""

    DIRECT = "direct"
    """Frozen topology, Newton-reprojected node positions, solved in-process."""
    TESSERACT = "tesseract"
    """The packaged mesher+solver tesseract chain, on the sampled lattice."""
    TESSERACT_DC = "tesseract-dc"
    """Dual contouring in JAX on the true SDF; only TetGen is wrapped."""


GradientPathLike = GradientPath | Literal["direct", "tesseract", "tesseract-dc"]
"""A gradient path, or the plain string spelling of one (aliases included)."""


class ObjectiveMetric(Option):
    """What a study-form optimization minimizes about the solved field."""

    MEAN = "mean"
    """The mean of the result's objective scalar."""
    MAX = "max"
    """The maximum of the result's objective scalar."""
    COMPLIANCE = "compliance"
    """Traction work (twice the strain energy); elastic studies only."""


ObjectiveMetricLike = ObjectiveMetric | Literal["mean", "max", "compliance"]
"""An objective metric, or the plain string spelling of one."""


class OptimizerMethod(Option):
    """The optimizer an ``Optimization`` descends with."""

    ADAM = "adam"
    """Optax Adam (the default)."""
    SGD = "sgd"
    """Optax SGD."""


OptimizerMethodLike = OptimizerMethod | Literal["adam", "sgd"]
"""An optimizer method, or the plain string spelling of one."""


class OptimizationArgument(Option):
    """The optimization keywords the viewer may retune.

    Everything else in an ``Optimization`` constructor is the objective
    itself, which is code, not a control.
    """

    STEPS = "steps"
    """How many optimizer steps a run takes."""
    LEARNING_RATE = "learning_rate"
    """The optimizer step size."""


OptimizationArgumentLike = OptimizationArgument | Literal["steps", "learning_rate"]
"""An editable optimization argument, or the plain string spelling of one."""


class ConstraintSolveMethod(Option):
    """How a sketch's constraints are satisfied.

    Distinct from ``OptimizerMethod``: the constraint solver's default is a
    minimum-norm Newton projection, which has no meaning as a descent
    method for a design objective.
    """

    NEWTON = "newton"
    """Minimum-norm projection onto the constraint manifold."""
    ADAM = "adam"
    """Optax Adam on the residual."""
    SGD = "sgd"
    """Optax SGD on the residual."""


ConstraintSolveMethodLike = ConstraintSolveMethod | Literal["newton", "adam", "sgd"]
"""A constraint solve method, or the plain string spelling of one."""


class ConstraintKind(Option):
    """The sketch constraints the viewer can add.

    The two valued kinds come first (they take a numeric target), then the
    relational ones.
    """

    FIXED = "fixed"
    """Pin one point where it is."""
    DISTANCE = "distance"
    """Hold two points a stated distance apart."""
    HORIZONTAL = "horizontal"
    """Level two points in y."""
    VERTICAL = "vertical"
    """Level two points in x."""
    COINCIDENT = "coincident"
    """Hold two points together."""
    PARALLEL = "parallel"
    """Keep two segments parallel."""
    PERPENDICULAR = "perpendicular"
    """Keep two segments at right angles."""


ConstraintKindLike = (
    ConstraintKind
    | Literal[
        "fixed",
        "distance",
        "horizontal",
        "vertical",
        "coincident",
        "parallel",
        "perpendicular",
    ]
)
"""A constraint kind, or the plain string spelling of one."""


class ExportFormat(Option):
    """The file formats the viewer's ``File → Export…`` can write.

    The three geometry formats take an SDF object of the program (the
    top-level ``scene`` by default); ``vtk`` takes a declared study instead
    and writes its solved fields, so it only exists where a result does.
    """

    OBJ = "obj"
    """Wavefront OBJ: coplanar quads merged into n-gons, curved regions as triangles."""
    STL = "stl"
    """STL triangles, binary by default (ASCII on request)."""
    STEP = "step"
    """STEP AP214: analytic planes and cylinders from the derived B-rep, faceted elsewhere."""
    VTK = "vtk"
    """A study's solved mesh and fields, for ParaView."""


ExportFormatLike = ExportFormat | Literal["obj", "stl", "step", "vtk"]
"""An export format, or the plain string spelling of one."""


class FemBackend(Option):
    """The FEM backends cadjoint registers itself.

    Unlike every other option set here this one is *open*: plugins and
    downstream code register more names through
    ``cadjoint.fem.backends.register_backend``, so ``get_backend`` accepts
    any registered string.  The enum names the built-ins — the ones the
    docs promise and the defaults resolve to — not the whole registry.
    """

    JAXFEM = "jaxfem"
    """In-process jax-fem; the default, and the differentiable path."""
    TESSERACT = "tesseract"
    """Solvers resolved through :mod:`cadjoint.plugins`."""
    CALCULIX = "calculix"
    """CalculiX (``ccx``) as a subprocess."""


class PluginKind(Option):
    """The component slots cadjoint's own pipeline resolves by kind.

    Open, like :class:`FemBackend` and for the same reason: a third-party
    plugin may declare any kind it likes, and ``PluginSpec.kind`` stays a
    plain string so an unknown one registers and reports cleanly.  These
    are the slots cadjoint itself asks for.
    """

    MESHER = "mesher"
    """Lattice samples in, a mesh out."""
    TETFILL = "tetfill"
    """A surface in, tetrahedra out."""
    TET_MESHER = "tet_mesher"
    """A B-rep in, a tetrahedral mesh out (Gmsh, out-of-process)."""
    THERMAL_SOLVER = "thermal_solver"
    """A steady-state conduction solver."""
    ELASTIC_SOLVER = "elastic_solver"
    """A small-strain elasticity solver."""
    FLOW_SOLVER = "flow_solver"
    """A Brinkman-penalised flow solver."""
    NODE_MAP = "node_map"
    """Design parameters to the node positions of a Gmsh mesh, with the
    implicit-function adjoint (:class:`cadjoint.plugins.contracts.NodeMap`)."""
    FEATURE_EDGES = "feature_edges"
    """The exact feature curves of a scene, for the viewer's sharp layer
    (:class:`cadjoint.plugins.contracts.FeatureEdges`)."""
    BREP = "brep"
    """The derived boundary representation, as an opaque in-process object
    (:class:`cadjoint.plugins.contracts.BRepExtractor`)."""
    STEP_EXPORT = "step_export"
    """Analytic STEP from the derived B-rep
    (:class:`cadjoint.plugins.contracts.StepExporter`)."""
    DRAG = "drag"
    """The drag inverse problem on a B-rep handle
    (:class:`cadjoint.plugins.contracts.Drag`)."""


PluginKindLike = PluginKind | str
"""A plugin kind: one cadjoint asks for, or any string a plugin declares."""


class PluginTransport(Option):
    """Where a plugin's component actually runs."""

    LOCAL = "local"
    """In this process, from a ``tesseract_api.py`` on disk."""
    CONTAINER = "container"
    """In a container image, served on demand."""
    REMOTE = "remote"
    """Behind a URL someone else operates."""
    PYTHON = "python"
    """An importable object in this process satisfying one of the contracts
    in :mod:`cadjoint.plugins.contracts` — no Tesseract runtime at all."""


PluginTransportLike = PluginTransport | Literal["local", "container", "remote", "python"]
"""A plugin transport, or the plain string spelling of one."""
