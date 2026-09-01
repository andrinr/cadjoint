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

Example::

    study = ThermalStudy(
        name="bar-conduction",
        resolution=(22, 5, 5),
        conductivity=2.0,
        bcs=[
            Dirichlet(Nodes.side("-x"), 1.0),
            Dirichlet(Nodes.side("+x"), 0.0),
        ],
    )
    result = study.solve(scene_sdf)
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from cadjoint.fem.hexmesh import GridSpec, HexMesh, faces_from_nodes, sdf_to_hex_mesh
from cadjoint.fem.selection import NodeSelection

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


def _validate_common(study: Any, kind: str, allowed_bcs: tuple[type, ...]) -> None:
    if not isinstance(study.name, str) or not study.name.strip():
        raise ValueError(f"{kind} study needs a non-empty name.")
    resolution = study.resolution
    counts = (resolution,) * 3 if isinstance(resolution, int) else tuple(resolution)
    if len(counts) != 3 or any(int(count) != count or count < 1 for count in counts):
        raise ValueError("resolution must be a positive integer or a triplet of them.")
    for bc in study.bcs:
        if not isinstance(bc, allowed_bcs):
            names = ", ".join(cls.__name__ for cls in allowed_bcs)
            raise ValueError(
                f"{kind} study accepts boundary conditions of type {names}; "
                f"got {type(bc).__name__}."
            )
    object.__setattr__(study, "bounds", _triplet(study.bounds, "bounds"))
    object.__setattr__(study, "size", _triplet(study.size, "size"))


def _study_grid(study: Any) -> GridSpec:
    resolution = study.resolution
    if not isinstance(resolution, int):
        resolution = tuple(int(count) for count in resolution)
    return GridSpec.from_bounds(study.bounds, study.size, resolution)


def _as_sdf(sdf_or_callable: Any) -> Callable[[Any], Any]:
    if not callable(sdf_or_callable):
        raise TypeError("solve() expects an SDF object or a callable field.")
    return sdf_or_callable


def _check_resolvable(bcs: list[Any], mesh: HexMesh) -> None:
    """Raise a selection-specific error before handing BCs to the solver."""
    for bc in bcs:
        indices = bc.nodes.resolve(mesh)
        if isinstance(bc, (HeatFlux, Traction)) and faces_from_nodes(mesh, indices).nodes.size == 0:
            raise ValueError(
                f"{type(bc).__name__} selection {bc.nodes.describe()} spans no complete "
                "boundary face; area-integrated conditions need all four corners of at "
                "least one boundary quad selected."
            )


@dataclass
class ThermalStudy:
    """Declarative steady-state heat conduction study.

    Attributes:
        name: Study identifier (unique within a scene program).
        resolution: Meshing resolution (cells per axis, int or triplet).
        conductivity: Thermal conductivity ``k``.
        bcs: :class:`Dirichlet` / :class:`HeatFlux` boundary conditions
            (at least one Dirichlet is required to solve).
        source: Volumetric heat source ``q``.
        bounds: Lower corner of the meshing domain.
        size: Extent of the meshing domain.
    """

    name: str
    resolution: Any
    conductivity: float
    bcs: list[Dirichlet | HeatFlux] = field(default_factory=list)
    source: float = 0.0
    bounds: Any = _DEFAULT_BOUNDS
    size: Any = _DEFAULT_SIZE

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
            "resolution": self.resolution
            if isinstance(self.resolution, int)
            else list(self.resolution),
            "bounds": list(self.bounds),
            "size": list(self.size),
            "material": {"conductivity": float(self.conductivity)},
            "source": float(self.source),
            "bcs": [bc.describe() for bc in self.bcs],
        }

    def solve(self, sdf, *, backend=None, mesh: HexMesh | None = None):
        """Mesh the field and run the thermal solve.

        Args:
            sdf: Scene SDF object or plain callable field.
            backend: Optional solver backend override (see
                :func:`cadjoint.fem.simulate.thermal_solve`).
            mesh: Optional pre-extracted mesh (skips ``sdf_to_hex_mesh``).

        Returns:
            A :class:`cadjoint.fem.simulate.ThermalResult`.
        """
        from cadjoint.fem.simulate import thermal_solve

        dirichlet = [bc for bc in self.bcs if isinstance(bc, Dirichlet)]
        fluxes = [bc for bc in self.bcs if isinstance(bc, HeatFlux)]
        if not dirichlet:
            raise ValueError("A thermal study needs at least one Dirichlet BC to solve.")
        field_fn = _as_sdf(sdf)
        if mesh is None:
            mesh = sdf_to_hex_mesh(field_fn, _study_grid(self))
        _check_resolvable(self.bcs, mesh)
        return thermal_solve(
            mesh,
            conductivity=float(self.conductivity),
            dirichlet=[(bc.nodes, bc.value) for bc in dirichlet],
            neumann=[(bc.nodes, bc.flux) for bc in fluxes],
            source=float(self.source),
            backend=backend,
        )


@dataclass
class ElasticStudy:
    """Declarative small-strain linear elasticity study.

    Attributes:
        name: Study identifier (unique within a scene program).
        resolution: Meshing resolution (cells per axis, int or triplet).
        youngs: Young's modulus.
        poisson: Poisson ratio (in ``[0, 0.5)``).
        bcs: :class:`Fixed` / :class:`Traction` boundary conditions
            (at least one Fixed is required to solve).
        bounds: Lower corner of the meshing domain.
        size: Extent of the meshing domain.
    """

    name: str
    resolution: Any
    youngs: float
    poisson: float
    bcs: list[Fixed | Traction] = field(default_factory=list)
    bounds: Any = _DEFAULT_BOUNDS
    size: Any = _DEFAULT_SIZE

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
            "resolution": self.resolution
            if isinstance(self.resolution, int)
            else list(self.resolution),
            "bounds": list(self.bounds),
            "size": list(self.size),
            "material": {"youngs": float(self.youngs), "poisson": float(self.poisson)},
            "bcs": [bc.describe() for bc in self.bcs],
        }

    def solve(self, sdf, *, backend=None, mesh: HexMesh | None = None):
        """Mesh the field and run the elastic solve.

        Args:
            sdf: Scene SDF object or plain callable field.
            backend: Optional solver backend override (see
                :func:`cadjoint.fem.simulate.elastic_solve`).
            mesh: Optional pre-extracted mesh (skips ``sdf_to_hex_mesh``).

        Returns:
            An :class:`cadjoint.fem.simulate.ElasticResult`.
        """
        from cadjoint.fem.simulate import elastic_solve

        fixed = [bc for bc in self.bcs if isinstance(bc, Fixed)]
        tractions = [bc for bc in self.bcs if isinstance(bc, Traction)]
        if not fixed:
            raise ValueError("An elastic study needs at least one Fixed BC to solve.")
        field_fn = _as_sdf(sdf)
        if mesh is None:
            mesh = sdf_to_hex_mesh(field_fn, _study_grid(self))
        _check_resolvable(self.bcs, mesh)
        return elastic_solve(
            mesh,
            youngs=float(self.youngs),
            poisson=float(self.poisson),
            dirichlet=[bc.nodes for bc in fixed],
            tractions=[(bc.nodes, bc.vector) for bc in tractions],
            backend=backend,
        )
