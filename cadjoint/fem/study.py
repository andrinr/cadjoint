"""Declarative, code-first simulation studies.

Studies are first-class citizens of the scene program, exactly like
sketches and constraints: declared in code (the source of truth),
serializable for the viewer via :meth:`describe`, and runnable directly by
scripts and optimizers via :meth:`solve`.  Constructing a study inside a
:func:`capture_studies` context registers it automatically, so the compile
worker can collect the studies a user program declares — mirroring
``capture_constraint_solves`` in :mod:`cadjoint.constraints.solve`.

Boundary conditions take a :class:`~cadjoint.fem.selection.NodeSelection`
built from the :class:`~cadjoint.fem.selection.Nodes` factory — programmatic
vertex selection composed with ``&``/``|``/``~``.  Node-valued conditions
(:class:`Dirichlet`, :class:`Fixed`) apply to the selected node set
directly; area-integrated conditions (:class:`HeatFlux`, :class:`Traction`)
act on the boundary faces spanned by the selection (all four corners
selected — :func:`~cadjoint.fem.hexmesh.faces_from_nodes`).

Meshing runs through one path: a study either references a declared
:class:`~cadjoint.fem.simmesh.SimMesh` (``mesh=<SimMesh or name>`` — the
name resolves against the meshes captured in the same program) or wraps its
own ``resolution``/``bounds``/``size`` into an anonymous one.  ``domain=``
restricts which part of the scene participates in the solve.  ``solve``
returns a :class:`~cadjoint.fem.result.SimulationResult` (also stored as
``last_result``) and stays differentiable through a traced ``points=``
override, exactly like the underlying solver calls.

Example::

    mesh = SimMesh(name="bar-mesh", resolution=(22, 5, 5))
    study = ThermalStudy(
        name="bar-conduction",
        conductivity=2.0,
        bcs=[
            Dirichlet(Nodes.side("-x"), 1.0),
            Dirichlet(Nodes.side("+x"), 0.0),
        ],
        mesh=mesh,
    )
    result = study.solve(scene_sdf)
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import KW_ONLY, dataclass, field
from typing import Any

import numpy as np

from cadjoint.fem.hexmesh import HexMesh, faces_from_nodes
from cadjoint.fem.selection import NodeSelection
from cadjoint.fem.simmesh import _CAPTURED_MESHES, SimMesh, _anonymous, _domain_entry
from cadjoint.fem.tetmesh import TetMesh, tet_faces_from_nodes

__all__ = [
    "Dirichlet",
    "ElasticStudy",
    "Fixed",
    "HeatFlux",
    "ThermalStudy",
    "Traction",
    "capture_studies",
]

# Same domain convention as the viewer's simulate path (compile worker).
_DEFAULT_BOUNDS = (-3.0, -3.0, -3.0)
_DEFAULT_SIZE = (6.0, 6.0, 6.0)

_CAPTURED_STUDIES: ContextVar[list[Any] | None] = ContextVar(
    "cadjoint_captured_studies",
    default=None,
)


@contextmanager
def capture_studies() -> Iterator[list[ThermalStudy | ElasticStudy]]:
    """Collect every study constructed inside this context.

    Mirrors ``capture_constraint_solves``: the compile worker wraps user
    program execution in this context and receives the declared studies in
    construction order.
    """
    studies: list[ThermalStudy | ElasticStudy] = []
    token = _CAPTURED_STUDIES.set(studies)
    try:
        yield studies
    finally:
        _CAPTURED_STUDIES.reset(token)


def _register(study: Any) -> None:
    captured = _CAPTURED_STUDIES.get()
    if captured is not None:
        captured.append(study)


def _triplet(value: Any, label: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must contain three finite numbers, got {value!r}.")
    return (float(array[0]), float(array[1]), float(array[2]))


def _expect_selection(nodes: Any, bc_kind: str) -> NodeSelection:
    if not isinstance(nodes, NodeSelection):
        raise ValueError(
            f"{bc_kind} takes a node selection, got {type(nodes).__name__}. "
            "Build one via Nodes.box/sphere/halfspace/side/predicate "
            "(from cadjoint.fem import Nodes)."
        )
    return nodes


@dataclass(frozen=True)
class Dirichlet:
    """Prescribed temperature on a set of boundary nodes.

    Applies to the selected node set directly.  With the default direct
    backend ``value`` may be a traced JAX scalar, making the solve
    differentiable w.r.t. the prescribed value.
    """

    nodes: NodeSelection
    value: float

    def __post_init__(self):
        _expect_selection(self.nodes, "Dirichlet")

    def describe(self) -> dict[str, Any]:
        """JSON-ready description."""
        return {"type": "dirichlet", "nodes": self.nodes.describe(), "value": self.value}


@dataclass(frozen=True)
class HeatFlux:
    """Prescribed heat inflow per area on the faces spanned by a selection.

    Positive flux heats the body.  Solvable on the direct backend (Neumann
    surface integral); the tesseract schema does not carry fluxes yet.
    """

    nodes: NodeSelection
    flux: float

    def __post_init__(self):
        _expect_selection(self.nodes, "HeatFlux")

    def describe(self) -> dict[str, Any]:
        """JSON-ready description."""
        return {"type": "heat_flux", "nodes": self.nodes.describe(), "flux": float(self.flux)}


@dataclass(frozen=True)
class Fixed:
    """Fully clamped node set (all displacement components zero)."""

    nodes: NodeSelection

    def __post_init__(self):
        _expect_selection(self.nodes, "Fixed")

    def describe(self) -> dict[str, Any]:
        """JSON-ready description."""
        return {"type": "fixed", "nodes": self.nodes.describe()}


@dataclass(frozen=True)
class Traction:
    """Constant traction (force per area) on the faces spanned by a selection."""

    nodes: NodeSelection
    vector: tuple[float, float, float]

    def __post_init__(self):
        _expect_selection(self.nodes, "Traction")
        object.__setattr__(self, "vector", _triplet(self.vector, "vector"))

    def describe(self) -> dict[str, Any]:
        """JSON-ready description."""
        return {
            "type": "traction",
            "nodes": self.nodes.describe(),
            "vector": list(self.vector),
        }


def _resolve_mesh_reference(mesh: Any) -> SimMesh:
    """Turn a ``mesh=`` argument into a SimMesh (resolving names)."""
    if isinstance(mesh, SimMesh):
        return mesh
    if isinstance(mesh, str):
        captured = _CAPTURED_MESHES.get()
        declared = [candidate for candidate in (captured or []) if candidate.name == mesh]
        if len(declared) == 1:
            return declared[0]
        if len(declared) > 1:
            raise ValueError(f"The program declares more than one mesh named {mesh!r}.")
        names = ", ".join(repr(candidate.name) for candidate in (captured or [])) or "none"
        raise ValueError(
            f"No declared SimMesh named {mesh!r} (declared: {names}). "
            "Declare one before the study, or pass the SimMesh instance itself."
        )
    raise ValueError(
        f"mesh must be a SimMesh or the name of a declared one, got {type(mesh).__name__}."
    )


def _validate_common(study: Any, kind: str, allowed_bcs: tuple[type, ...]) -> None:
    if not isinstance(study.name, str) or not study.name.strip():
        raise ValueError(f"{kind} study needs a non-empty name.")
    for bc in study.bcs:
        if not isinstance(bc, allowed_bcs):
            names = ", ".join(cls.__name__ for cls in allowed_bcs)
            raise ValueError(
                f"{kind} study accepts boundary conditions of type {names}; "
                f"got {type(bc).__name__}."
            )
    if study.domain is not None and not callable(study.domain):
        raise TypeError(
            f"domain must be an SDF object or a callable field, got {type(study.domain).__name__}."
        )
    if study.mesh is not None:
        study.mesh = _resolve_mesh_reference(study.mesh)
        conflicts = [
            label
            for label, value in (
                ("resolution", study.resolution),
                ("bounds", study.bounds),
                ("size", study.size),
                ("domain", study.domain),
            )
            if value is not None
        ]
        if conflicts:
            raise ValueError(
                f"{kind} study got mesh= and {', '.join(conflicts)}; meshing intent "
                "lives on the SimMesh — set those on it instead."
            )
        return
    if study.resolution is None:
        raise ValueError(f"{kind} study needs a resolution (or a mesh=SimMesh).")
    counts = (
        (study.resolution,) * 3 if isinstance(study.resolution, int) else tuple(study.resolution)
    )
    if len(counts) != 3 or any(int(count) != count or count < 1 for count in counts):
        raise ValueError("resolution must be a positive integer or a triplet of them.")
    study.bounds = _triplet(study.bounds if study.bounds is not None else _DEFAULT_BOUNDS, "bounds")
    study.size = _triplet(study.size if study.size is not None else _DEFAULT_SIZE, "size")


def _solve_mesh(study: Any, sdf: Any, mesh: Any) -> tuple[SimMesh | None, HexMesh | TetMesh]:
    """The one meshing path for solves.

    An explicit ``HexMesh``/``TetMesh`` is used as-is (no SimMesh
    attached); a SimMesh (explicit argument, the study's own, or an
    anonymous wrap of the study's resolution/bounds/size/domain) is built
    — reusing its cache.  The mesh's method decides the solve route
    (:mod:`cadjoint.fem.simulate` dispatches hex vs tet).
    """
    if isinstance(mesh, (HexMesh, TetMesh)):
        return None, mesh
    if mesh is not None:
        target = _resolve_mesh_reference(mesh)
    elif study.mesh is not None:
        target = study.mesh
    else:
        if study._implicit_mesh is None:
            study._implicit_mesh = _anonymous(
                name=f"{study.name}::mesh",
                resolution=study.resolution,
                domain=study.domain,
                bounds=study.bounds,
                size=study.size,
            )
        target = study._implicit_mesh
    return target, target.build(sdf)


def _check_resolvable(bcs: list[Any], mesh: HexMesh | TetMesh) -> None:
    """Raise a selection-specific error before handing BCs to the solver."""
    for bc in bcs:
        indices = bc.nodes.resolve(mesh)
        if isinstance(bc, (HeatFlux, Traction)):
            if isinstance(mesh, TetMesh):
                spanned = int(tet_faces_from_nodes(mesh, indices).shape[0])
            else:
                spanned = int(faces_from_nodes(mesh, indices).nodes.shape[0])
            if spanned == 0:
                raise ValueError(
                    f"{type(bc).__name__} selection {bc.nodes.describe()} spans no complete "
                    "boundary face; area-integrated conditions need every corner of at "
                    "least one boundary face selected."
                )


def _mesh_payload(study: Any) -> dict[str, Any]:
    """The describe() entries shared by both study kinds.

    Mesh-backed studies report the SimMesh's resolution/bounds/size (bounds
    may be None when the mesh derives them automatically) plus its name;
    implicit studies report their own resolved values with ``mesh: None``.
    """
    mesh = study.mesh
    if mesh is not None:
        resolution = mesh.resolution
        bounds, size = mesh.bounds, mesh.size
        domain = mesh.domain
    else:
        resolution = study.resolution
        bounds, size = study.bounds, study.size
        domain = study.domain
    return {
        "resolution": resolution if isinstance(resolution, int) else list(resolution),
        "bounds": list(bounds) if bounds is not None else None,
        "size": list(size) if size is not None else None,
        "mesh": mesh.name if mesh is not None else None,
        "domain": _domain_entry(domain),
    }


@dataclass
class ThermalStudy:
    """Declarative steady-state heat conduction study.

    Attributes:
        name: Study identifier (unique within a scene program).
        resolution: Meshing resolution (cells per axis, int or triplet);
            leave None when ``mesh`` is given.
        conductivity: Thermal conductivity ``k`` (keyword-only).
        bcs: :class:`Dirichlet` / :class:`HeatFlux` boundary conditions
            (at least one Dirichlet is required to solve).
        source: Volumetric heat source ``q``.
        bounds: Lower corner of the meshing domain (None: default volume).
        size: Extent of the meshing domain (None: default volume).
        mesh: A declared :class:`~cadjoint.fem.simmesh.SimMesh` (or its
            name) to solve on; when given, resolution/bounds/size/domain
            live on the mesh and must be left unset here.
        domain: Optional SDF restricting which part of the scene is meshed
            (implicit-mesh studies only).
        last_result: The :class:`~cadjoint.fem.result.SimulationResult` of
            the most recent ``solve`` (None before the first).
    """

    name: str
    resolution: Any = None
    _: KW_ONLY
    conductivity: float
    bcs: list[Dirichlet | HeatFlux] = field(default_factory=list)
    source: float = 0.0
    bounds: Any = None
    size: Any = None
    mesh: SimMesh | str | None = None
    domain: Any = None
    last_result: Any = field(default=None, init=False, repr=False, compare=False)
    _implicit_mesh: SimMesh | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        _validate_common(self, "Thermal", (Dirichlet, HeatFlux))
        if float(self.conductivity) <= 0.0:
            raise ValueError("conductivity must be positive.")
        _register(self)

    def describe(self) -> dict[str, Any]:
        """JSON-ready payload: everything the viewer needs to display it."""
        return {
            "name": self.name,
            "kind": "thermal",
            **_mesh_payload(self),
            "material": {"conductivity": float(self.conductivity)},
            "source": float(self.source),
            "bcs": [bc.describe() for bc in self.bcs],
        }

    def solve(
        self,
        sdf=None,
        *,
        backend=None,
        mesh: HexMesh | TetMesh | SimMesh | str | None = None,
        points=None,
    ):
        """Mesh the field (through the study's SimMesh) and run the solve.

        Args:
            sdf: Scene SDF object or plain callable field; optional when
                the study's mesh declares a ``domain``.
            backend: Optional solver backend override (see
                :func:`cadjoint.fem.simulate.thermal_solve`).
            mesh: Optional mesh override — a pre-extracted
                :class:`~cadjoint.fem.hexmesh.HexMesh` or
                :class:`~cadjoint.fem.tetmesh.TetMesh`, a SimMesh, or a
                declared mesh name.
            points: Optional traced override of the mesh node positions
                (``recompute_points`` / ``recompute_tet_points``, matching
                the mesh's method) for differentiable frozen-topology
                solves; BC selections still resolve on the nominal points.

        Returns:
            A :class:`~cadjoint.fem.result.SimulationResult` (also stored
            as ``last_result``).
        """
        from cadjoint.fem.result import SimulationResult
        from cadjoint.fem.simulate import thermal_solve

        dirichlet = [bc for bc in self.bcs if isinstance(bc, Dirichlet)]
        fluxes = [bc for bc in self.bcs if isinstance(bc, HeatFlux)]
        if not dirichlet:
            raise ValueError("A thermal study needs at least one Dirichlet BC to solve.")
        sim_mesh, hex_mesh = _solve_mesh(self, sdf, mesh)
        _check_resolvable(self.bcs, hex_mesh)
        solution = thermal_solve(
            hex_mesh,
            conductivity=float(self.conductivity),
            dirichlet=[(bc.nodes, bc.value) for bc in dirichlet],
            neumann=[(bc.nodes, bc.flux) for bc in fluxes],
            source=float(self.source),
            backend=backend,
            points=points,
        )
        result = SimulationResult(
            name=self.name,
            kind="thermal",
            field="temperature",
            solution=solution,
            sim_mesh=sim_mesh,
        )
        self.last_result = result
        return result


@dataclass
class ElasticStudy:
    """Declarative small-strain linear elasticity study.

    Attributes:
        name: Study identifier (unique within a scene program).
        resolution: Meshing resolution (cells per axis, int or triplet);
            leave None when ``mesh`` is given.
        youngs: Young's modulus (keyword-only).
        poisson: Poisson ratio in ``[0, 0.5)`` (keyword-only).
        bcs: :class:`Fixed` / :class:`Traction` boundary conditions
            (at least one Fixed is required to solve).
        bounds: Lower corner of the meshing domain (None: default volume).
        size: Extent of the meshing domain (None: default volume).
        mesh: A declared :class:`~cadjoint.fem.simmesh.SimMesh` (or its
            name) to solve on; when given, resolution/bounds/size/domain
            live on the mesh and must be left unset here.
        domain: Optional SDF restricting which part of the scene is meshed
            (implicit-mesh studies only).
        last_result: The :class:`~cadjoint.fem.result.SimulationResult` of
            the most recent ``solve`` (None before the first).
    """

    name: str
    resolution: Any = None
    _: KW_ONLY
    youngs: float
    poisson: float
    bcs: list[Fixed | Traction] = field(default_factory=list)
    bounds: Any = None
    size: Any = None
    mesh: SimMesh | str | None = None
    domain: Any = None
    last_result: Any = field(default=None, init=False, repr=False, compare=False)
    _implicit_mesh: SimMesh | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        _validate_common(self, "Elastic", (Fixed, Traction))
        if float(self.youngs) <= 0.0:
            raise ValueError("youngs must be positive.")
        if not 0.0 <= float(self.poisson) < 0.5:
            raise ValueError("poisson must be in [0, 0.5).")
        _register(self)

    def describe(self) -> dict[str, Any]:
        """JSON-ready payload: everything the viewer needs to display it."""
        return {
            "name": self.name,
            "kind": "elastic",
            **_mesh_payload(self),
            "material": {"youngs": float(self.youngs), "poisson": float(self.poisson)},
            "bcs": [bc.describe() for bc in self.bcs],
        }

    def solve(
        self,
        sdf=None,
        *,
        backend=None,
        mesh: HexMesh | TetMesh | SimMesh | str | None = None,
        points=None,
    ):
        """Mesh the field (through the study's SimMesh) and run the solve.

        Args:
            sdf: Scene SDF object or plain callable field; optional when
                the study's mesh declares a ``domain``.
            backend: Optional solver backend override (see
                :func:`cadjoint.fem.simulate.elastic_solve`).
            mesh: Optional mesh override — a pre-extracted
                :class:`~cadjoint.fem.hexmesh.HexMesh` or
                :class:`~cadjoint.fem.tetmesh.TetMesh`, a SimMesh, or a
                declared mesh name.
            points: Optional traced override of the mesh node positions
                (``recompute_points`` / ``recompute_tet_points``, matching
                the mesh's method) for differentiable frozen-topology
                solves; BC selections still resolve on the nominal points.

        Returns:
            A :class:`~cadjoint.fem.result.SimulationResult` (also stored
            as ``last_result``).
        """
        from cadjoint.fem.result import SimulationResult
        from cadjoint.fem.simulate import elastic_solve

        fixed = [bc for bc in self.bcs if isinstance(bc, Fixed)]
        tractions = [bc for bc in self.bcs if isinstance(bc, Traction)]
        if not fixed:
            raise ValueError("An elastic study needs at least one Fixed BC to solve.")
        sim_mesh, hex_mesh = _solve_mesh(self, sdf, mesh)
        _check_resolvable(self.bcs, hex_mesh)
        solution = elastic_solve(
            hex_mesh,
            youngs=float(self.youngs),
            poisson=float(self.poisson),
            dirichlet=[bc.nodes for bc in fixed],
            tractions=[(bc.nodes, bc.vector) for bc in tractions],
            backend=backend,
            points=points,
        )
        result = SimulationResult(
            name=self.name,
            kind="elastic",
            field="von_mises",
            solution=solution,
            sim_mesh=sim_mesh,
        )
        self.last_result = result
        return result
