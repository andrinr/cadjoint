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
selected — :func:`~cadjoint.fem.boundary.faces_from_nodes`).

Material properties come from one of two places.  Pass an explicit scalar
(``ThermalStudy(conductivity=2.0)``) and the whole domain solves with it, as
it always has.  Leave the argument out — or pass the sentinel
:data:`~cadjoint.fem.properties.FROM_MATERIAL` (``"material"``) — and the
study samples the scene's own material field per element at solve time
(:func:`cadjoint.fem.properties.sample_cell_property`), so a copper slug
pressed into an aluminium sink solves as two materials with a smooth
transition exactly as wide as the CSG blend that joins them.  The sampling is
differentiable in both directions that matter: w.r.t. the geometry (the blend
moves when the design does) and w.r.t. any material property marked ``free``.
A study whose materials never state the property it needs fails with an error
naming the property, rather than inventing a value.

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

from cadjoint.enums import BoundaryConditionType, StudyKind
from cadjoint.fem.boundary import faces_from_nodes, tet_faces_from_nodes
from cadjoint.fem.hexmesh import HexMesh
from cadjoint.fem.properties import FROM_MATERIAL
from cadjoint.fem.selection import NodeSelection
from cadjoint.fem.simmesh import _CAPTURED_MESHES, SimMesh, _anonymous, _domain_entry
from cadjoint.fem.tetmesh import TetMesh

__all__ = [
    "FROM_MATERIAL",
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
        return {
            "type": BoundaryConditionType.DIRICHLET.value,
            "nodes": self.nodes.describe(),
            "value": self.value,
        }


@dataclass(frozen=True)
class HeatFlux:
    """Prescribed heat inflow per area on the faces spanned by a selection.

    Positive flux heats the body.  Solvable on the direct backend and the
    thermal tesseract alike (Neumann surface integral either way).
    """

    nodes: NodeSelection
    flux: float

    def __post_init__(self):
        _expect_selection(self.nodes, "HeatFlux")

    def describe(self) -> dict[str, Any]:
        """JSON-ready description."""
        return {
            "type": BoundaryConditionType.HEAT_FLUX.value,
            "nodes": self.nodes.describe(),
            "flux": float(self.flux),
        }


@dataclass(frozen=True)
class Fixed:
    """Fully clamped node set (all displacement components zero)."""

    nodes: NodeSelection

    def __post_init__(self):
        _expect_selection(self.nodes, "Fixed")

    def describe(self) -> dict[str, Any]:
        """JSON-ready description."""
        return {"type": BoundaryConditionType.FIXED.value, "nodes": self.nodes.describe()}


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
            "type": BoundaryConditionType.TRACTION.value,
            "nodes": self.nodes.describe(),
            "vector": list(self.vector),
        }


def _from_material(value: Any) -> bool:
    """True when a property argument asks to be sampled from the materials."""
    return isinstance(value, str)


def _property_argument(value: Any, label: str, check: Any) -> Any:
    """Validate a material-property argument at construction time.

    Args:
        value: The user's argument — a number, or the
            :data:`~cadjoint.fem.properties.FROM_MATERIAL` sentinel.
        label: Argument name, for error messages.
        check: Predicate a numeric value must satisfy, or None.

    Returns:
        ``FROM_MATERIAL`` unchanged, or the value as a validated ``float``.

    Raises:
        ValueError: If a string other than the sentinel is given, or a numeric
            value fails ``check``.
    """
    if _from_material(value):
        if value != FROM_MATERIAL:
            raise ValueError(
                f"{label} must be a number or {FROM_MATERIAL!r} (sample the scene's "
                f"materials per element); got {value!r}."
            )
        return FROM_MATERIAL
    number = float(value)
    if check is not None and not check(number):
        raise ValueError(f"{label} value {number} is out of range.")
    return number


def _material_source(sdf: Any, sim_mesh: SimMesh | None) -> Any:
    """The object whose ``material_at`` defines the property field for a solve."""
    if sdf is not None:
        return sdf
    return sim_mesh.domain if sim_mesh is not None else None


def _resolve_property(
    value: Any,
    key: str,
    *,
    sdf: Any,
    points: Any,
    cells: Any,
    label: str,
) -> Any:
    """A scalar property, or a per-element array sampled from the materials.

    Args:
        value: The study's stored argument (number or ``FROM_MATERIAL``).
        key: The :class:`~cadjoint.render.material.Material` property name.
        sdf: The scene SDF whose ``material_at`` is sampled.
        points: Node positions the centroids are built from (may be traced).
        cells: Element connectivity.
        label: Study name, for error messages.

    Returns:
        The float itself, or a ``(C,)`` JAX array of per-element values.

    Raises:
        ValueError: If sampling is asked for but no SDF is available.
    """
    if not _from_material(value):
        return value
    from cadjoint.fem.properties import sample_cell_property

    if sdf is None:
        raise ValueError(
            f"Study {label!r} derives {key!r} from the scene's materials but got no "
            "SDF to sample: pass the scene to solve(sdf), give the study's SimMesh a "
            f"domain=, or set an explicit {key} value on the study."
        )
    return sample_cell_property(sdf, points, cells, key, label=f"study {label!r}")


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


def _reported_property(sdf: Any, points: Any, mesh: Any, key: str) -> Any:
    """A per-element property for reporting, or None when the scene lacks it.

    Reporting is best-effort by design: a study should still solve when its
    materials say nothing about density or yield strength — it just cannot
    report a mass or a safety factor.
    """
    from cadjoint.fem.properties import maybe_sample_cell_property

    return maybe_sample_cell_property(sdf, points, mesh.cells, key, base_points=mesh.points)


def _reported_mass(sdf: Any, points: Any, mesh: Any) -> Any:
    """Mass of the solved domain, or None when the materials state no density."""
    density = _reported_property(sdf, points, mesh, "density")
    if density is None:
        return None
    from cadjoint.fem.properties import total_mass

    return total_mass(points, mesh.cells, density)


@dataclass
class ThermalStudy:
    """Declarative steady-state heat conduction study.

    Attributes:
        name: Study identifier (unique within a scene program).
        resolution: Meshing resolution (cells per axis, int or triplet);
            leave None when ``mesh`` is given.
        conductivity: Thermal conductivity ``k`` in W/(m*K), keyword-only.
            A number applies to the whole domain; the default
            :data:`~cadjoint.fem.properties.FROM_MATERIAL` sentinel samples
            the scene's material field per element instead.
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
    conductivity: Any = FROM_MATERIAL
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
        self.conductivity = _property_argument(
            self.conductivity, "conductivity", lambda value: value > 0.0
        )
        _register(self)

    def describe(self) -> dict[str, Any]:
        """JSON-ready payload: everything the viewer needs to display it."""
        return {
            "name": self.name,
            "kind": StudyKind.THERMAL.value,
            **_mesh_payload(self),
            "material": {"conductivity": self.conductivity},
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
        source_sdf = _material_source(sdf, sim_mesh)
        solve_points = hex_mesh.points if points is None else points
        conductivity = _resolve_property(
            self.conductivity,
            "conductivity",
            sdf=source_sdf,
            points=solve_points,
            cells=hex_mesh.cells,
            label=self.name,
        )
        solution = thermal_solve(
            hex_mesh,
            conductivity=conductivity,
            dirichlet=[(bc.nodes, bc.value) for bc in dirichlet],
            neumann=[(bc.nodes, bc.flux) for bc in fluxes],
            source=float(self.source),
            backend=backend,
            points=points,
        )
        result = SimulationResult(
            name=self.name,
            kind=StudyKind.THERMAL.value,
            field="temperature",
            solution=solution,
            sim_mesh=sim_mesh,
            mass=_reported_mass(source_sdf, solve_points, hex_mesh),
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
        youngs: Young's modulus in Pa (keyword-only).  A number applies to
            the whole domain; the default
            :data:`~cadjoint.fem.properties.FROM_MATERIAL` sentinel samples
            the scene's material field per element instead.
        poisson: Poisson ratio in ``[0, 0.5)`` (keyword-only), scalar or
            material-derived exactly like ``youngs``.
        gravity: Optional gravity vector in m/s^2 (e.g. ``(0, 0, -9.81)``)
            adding self-weight as the body force ``density * gravity``.  The
            scene's materials must specify a density.
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
    youngs: Any = FROM_MATERIAL
    poisson: Any = FROM_MATERIAL
    gravity: Any = None
    bcs: list[Fixed | Traction] = field(default_factory=list)
    bounds: Any = None
    size: Any = None
    mesh: SimMesh | str | None = None
    domain: Any = None
    last_result: Any = field(default=None, init=False, repr=False, compare=False)
    _implicit_mesh: SimMesh | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        _validate_common(self, "Elastic", (Fixed, Traction))
        self.youngs = _property_argument(self.youngs, "youngs", lambda value: value > 0.0)
        self.poisson = _property_argument(self.poisson, "poisson", lambda value: 0.0 <= value < 0.5)
        if self.gravity is not None:
            self.gravity = _triplet(self.gravity, "gravity")
        _register(self)

    def describe(self) -> dict[str, Any]:
        """JSON-ready payload: everything the viewer needs to display it."""
        return {
            "name": self.name,
            "kind": StudyKind.ELASTIC.value,
            **_mesh_payload(self),
            "material": {"youngs": self.youngs, "poisson": self.poisson},
            "gravity": list(self.gravity) if self.gravity is not None else None,
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
        source_sdf = _material_source(sdf, sim_mesh)
        solve_points = hex_mesh.points if points is None else points
        resolved = {
            key: _resolve_property(
                getattr(self, key),
                material_key,
                sdf=source_sdf,
                points=solve_points,
                cells=hex_mesh.cells,
                label=self.name,
            )
            for key, material_key in (("youngs", "youngs_modulus"), ("poisson", "poisson_ratio"))
        }
        solution = elastic_solve(
            hex_mesh,
            youngs=resolved["youngs"],
            poisson=resolved["poisson"],
            dirichlet=[bc.nodes for bc in fixed],
            tractions=[(bc.nodes, bc.vector) for bc in tractions],
            backend=backend,
            points=points,
            body_force=self._body_force(source_sdf, solve_points, hex_mesh),
        )
        result = SimulationResult(
            name=self.name,
            kind=StudyKind.ELASTIC.value,
            field="von_mises",
            solution=solution,
            sim_mesh=sim_mesh,
            mass=_reported_mass(source_sdf, solve_points, hex_mesh),
            yield_strength=_reported_property(source_sdf, solve_points, hex_mesh, "yield_strength"),
        )
        self.last_result = result
        return result

    def _body_force(self, sdf: Any, points: Any, mesh: Any) -> Any:
        """Self-weight ``density * gravity`` per element, or None without gravity.

        Args:
            sdf: The scene SDF whose material field carries the density.
            points: Node positions the centroids are built from.
            mesh: The solved mesh (for its connectivity).

        Returns:
            A ``(C, 3)`` body force density in N/m^3, or None when the study
            declares no gravity.

        Raises:
            ValueError: If gravity is set but the materials state no density.
        """
        if self.gravity is None:
            return None
        import jax.numpy as jnp

        from cadjoint.fem.properties import sample_cell_property

        if sdf is None:
            raise ValueError(
                f"Study {self.name!r} sets gravity but got no SDF to read densities "
                "from: pass the scene to solve(sdf) or give the SimMesh a domain=."
            )
        density = sample_cell_property(
            sdf, points, mesh.cells, "density", label=f"study {self.name!r} (gravity)"
        )
        return density[:, None] * jnp.asarray(self.gravity, dtype=density.dtype)
