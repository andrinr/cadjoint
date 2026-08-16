"""Declarative, code-first simulation studies.

Studies are first-class citizens of the scene program, exactly like
sketches and constraints: declared in code (the source of truth),
serializable for the viewer via :meth:`describe`, and runnable directly by
scripts and optimizers via :meth:`solve`.  Constructing a study inside a
:func:`capture_studies` context registers it automatically, so the compile
worker can collect the studies a user program declares — mirroring
``capture_constraint_solves`` in :mod:`jaxcad.constraints.solve`.

Example::

    study = ThermalStudy(
        name="bar-conduction",
        resolution=(22, 5, 5),
        conductivity=2.0,
        bcs=[
            Dirichlet(FaceSelector.side("-x"), 1.0),
            Dirichlet(FaceSelector.side("+x"), 0.0),
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

from jaxcad.fem.hexmesh import FaceGroup, GridSpec, HexMesh, sdf_to_hex_mesh, select_faces

__all__ = [
    "Dirichlet",
    "ElasticStudy",
    "FaceSelector",
    "Fixed",
    "HeatFlux",
    "ThermalStudy",
    "Traction",
    "capture_studies",
]

# Same domain convention as the viewer's simulate path (compile worker).
_DEFAULT_BOUNDS = (-3.0, -3.0, -3.0)
_DEFAULT_SIZE = (6.0, 6.0, 6.0)
_SIDES = ("+x", "-x", "+y", "-y", "+z", "-z")

_CAPTURED_STUDIES: ContextVar[list[Any] | None] = ContextVar(
    "jaxcad_captured_studies",
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


@dataclass(frozen=True)
class FaceSelector:
    """Serializable selector of boundary-face patches.

    Build via the classmethods: :meth:`side` (dominant-gradient-axis group,
    e.g. ``"+x"``), :meth:`box` (face centers inside an axis-aligned
    region), or :meth:`where` (arbitrary predicate — the escape hatch;
    everything else round-trips through JSON, predicates do not).
    """

    kind: str
    side_name: str | None = None
    center: tuple[float, float, float] | None = None
    extent: tuple[float, float, float] | None = None
    predicate: Callable[..., Any] | None = None

    def __post_init__(self):
        if self.kind == "side":
            if self.side_name not in _SIDES:
                raise ValueError(f"side must be one of {_SIDES}, got {self.side_name!r}.")
        elif self.kind == "box":
            if self.center is None or self.extent is None:
                raise ValueError("box selector needs center and extent.")
        elif self.kind == "predicate":
            if not callable(self.predicate):
                raise ValueError("predicate selector needs a callable.")
        else:
            raise ValueError(f"Unknown selector kind {self.kind!r}.")

    @classmethod
    def side(cls, name: str) -> FaceSelector:
        """All boundary faces whose dominant SDF-gradient axis is ``name``."""
        return cls(kind="side", side_name=name)

    @classmethod
    def box(cls, center: Any, extent: Any) -> FaceSelector:
        """Boundary faces whose centers lie in the axis-aligned box.

        Args:
            center: Box center.
            extent: Full box widths per axis.
        """
        return cls(kind="box", center=_triplet(center, "center"), extent=_triplet(extent, "extent"))

    @classmethod
    def where(cls, predicate: Callable[..., Any]) -> FaceSelector:
        """Escape hatch: ``predicate(center)`` or ``predicate(center, normal)``."""
        return cls(kind="predicate", predicate=predicate)

    def describe(self) -> dict[str, Any]:
        """JSON-ready description (predicates are named but not serialized)."""
        if self.kind == "side":
            return {"kind": "side", "side": self.side_name}
        if self.kind == "box":
            return {"kind": "box", "center": list(self.center), "extent": list(self.extent)}
        name = getattr(self.predicate, "__name__", "<lambda>")
        return {"kind": "predicate", "callable": name}

    def resolve(self, mesh: HexMesh) -> FaceGroup:
        """The matching boundary faces of ``mesh``.

        Raises:
            ValueError: If no boundary face matches.
        """
        group = select_faces(mesh, self.to_predicate(mesh))
        if group.nodes.size == 0:
            raise ValueError(f"Selector {self.describe()} matched no boundary faces.")
        return group

    def to_predicate(self, mesh: HexMesh) -> Callable[..., Any]:
        """A ``select_faces`` predicate equivalent to this selector.

        Side selectors are membership tests against the mesh's precomputed
        gradient-axis group (exact float match — ``select_faces`` iterates
        the very same center rows), so they compose with the existing
        predicate-based ``thermal_solve``/``elastic_solve`` API unchanged.
        """
        if self.kind == "side":
            group = mesh.boundary_faces.get(self.side_name)
            centers = frozenset(map(tuple, group.centers)) if group is not None else frozenset()
            return lambda center: tuple(center) in centers
        if self.kind == "box":
            center = np.asarray(self.center)
            half = np.asarray(self.extent) / 2.0
            return lambda point: bool(np.all(np.abs(np.asarray(point) - center) <= half))
        return self.predicate


@dataclass(frozen=True)
class Dirichlet:
    """Prescribed temperature on a boundary patch."""

    selector: FaceSelector
    value: float

    def describe(self) -> dict[str, Any]:
        """JSON-ready description."""
        return {"type": "dirichlet", "selector": self.selector.describe(), "value": self.value}


@dataclass(frozen=True)
class HeatFlux:
    """Prescribed heat flux on a boundary patch.

    Declarable and serializable today; solving raises until the thermal
    backend grows Neumann support (see research/fem-integration.md).
    """

    selector: FaceSelector
    value: float

    def describe(self) -> dict[str, Any]:
        """JSON-ready description."""
        return {"type": "heat_flux", "selector": self.selector.describe(), "value": self.value}


@dataclass(frozen=True)
class Fixed:
    """Fully clamped boundary patch (all displacement components zero)."""

    selector: FaceSelector

    def describe(self) -> dict[str, Any]:
        """JSON-ready description."""
        return {"type": "fixed", "selector": self.selector.describe()}


@dataclass(frozen=True)
class Traction:
    """Constant traction (force per area) on a boundary patch."""

    selector: FaceSelector
    vector: tuple[float, float, float]

    def __post_init__(self):
        object.__setattr__(self, "vector", _triplet(self.vector, "vector"))

    def describe(self) -> dict[str, Any]:
        """JSON-ready description."""
        return {
            "type": "traction",
            "selector": self.selector.describe(),
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
                :func:`jaxcad.fem.simulate.thermal_solve`).
            mesh: Optional pre-extracted mesh (skips ``sdf_to_hex_mesh``).

        Returns:
            A :class:`jaxcad.fem.simulate.ThermalResult`.
        """
        from jaxcad.fem.simulate import thermal_solve

        if any(isinstance(bc, HeatFlux) for bc in self.bcs):
            raise NotImplementedError(
                "HeatFlux boundary conditions are declarable but not solvable yet: "
                "the thermal backend has no Neumann term. Use Dirichlet + source, "
                "or wait for the flux-enabled backend."
            )
        dirichlet = [bc for bc in self.bcs if isinstance(bc, Dirichlet)]
        if not dirichlet:
            raise ValueError("A thermal study needs at least one Dirichlet BC to solve.")
        field_fn = _as_sdf(sdf)
        if mesh is None:
            mesh = sdf_to_hex_mesh(field_fn, _study_grid(self))
        for bc in dirichlet:
            bc.selector.resolve(mesh)  # selector-specific error before solving
        return thermal_solve(
            mesh,
            conductivity=float(self.conductivity),
            dirichlet=[(bc.selector.to_predicate(mesh), bc.value) for bc in dirichlet],
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
                :func:`jaxcad.fem.simulate.elastic_solve`).
            mesh: Optional pre-extracted mesh (skips ``sdf_to_hex_mesh``).

        Returns:
            An :class:`jaxcad.fem.simulate.ElasticResult`.
        """
        from jaxcad.fem.simulate import elastic_solve

        fixed = [bc for bc in self.bcs if isinstance(bc, Fixed)]
        tractions = [bc for bc in self.bcs if isinstance(bc, Traction)]
        if not fixed:
            raise ValueError("An elastic study needs at least one Fixed BC to solve.")
        field_fn = _as_sdf(sdf)
        if mesh is None:
            mesh = sdf_to_hex_mesh(field_fn, _study_grid(self))
        for bc in self.bcs:
            bc.selector.resolve(mesh)  # selector-specific error before solving
        return elastic_solve(
            mesh,
            youngs=float(self.youngs),
            poisson=float(self.poisson),
            dirichlet=[bc.selector.to_predicate(mesh) for bc in fixed],
            tractions=[(bc.selector.to_predicate(mesh), bc.vector) for bc in tractions],
            backend=backend,
        )
