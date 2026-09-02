"""Named, first-class simulation meshes.

A :class:`SimMesh` captures *meshing intent* the same way a study captures
solving intent: which part of the scene to mesh (``domain``), with which
``method`` (``"hex"`` voxelize+snap — the fast default — or the
DC-conforming ``"tet4"``/``"tet10"`` route), at what ``resolution``, over
which box (``bounds`` + ``size``, or automatically scanned from the domain
plus ``padding``), under a stable ``name``.  It is declared in the scene
program, registered by :func:`capture_sim_meshes` (mirroring
``capture_studies``), serialized for the viewer via
:meth:`SimMesh.describe`, and built on demand via :meth:`SimMesh.build` —
the one meshing path both explicit and implicit study meshing run through.

``resolution`` always means the *sampling lattice* (cells per axis of the
grid the SDF is discretized on).  For ``method="hex"`` those lattice cells
*are* the elements (inside cells kept, boundary vertices snapped).  For
``method="tet4"``/``"tet10"`` the lattice is the dual-contouring extraction
grid: the DC surface is tetrahedralized by TetGen, so element and node
counts are decided by TetGen's fill of the interior (typically several
tets per lattice cell) — same geometric feature size per resolution, not
the same element count.  ``"tet10"`` promotes the TET4 mesh to quadratic
straight-sided tets (shared midside nodes appended); it is the quality
path — accurate in bending where TET4 locks, geometry-conforming where hex
staircases alias — at a higher solve cost (see ``research/tet-vs-hex.md``).

The built mesh (:class:`~cadjoint.fem.hexmesh.HexMesh` or
:class:`~cadjoint.fem.tetmesh.TetMesh`) is cached on the instance until
the meshing parameters or the meshed field change, so a scene program can
pass the same mesh to several studies (and to
:func:`~cadjoint.fem.motion.recompute_points` /
:func:`~cadjoint.fem.motion.recompute_tet_points` for design gradients)
and mesh exactly once.  Inspection is first-class: :meth:`SimMesh.quality`
returns per-element quality arrays and :meth:`SimMesh.inspect` a JSON-ready
summary (method, counts, bounds, grid, element-quality statistics).

Example::

    mesh = SimMesh(name="bracket-mesh", resolution=(24, 17, 13), domain=bracket)
    study = ElasticStudy(name="pry", youngs=1000.0, poisson=0.3,
                         bcs=[...], mesh=mesh)
    print(mesh.inspect()["quality"]["scaled_jacobian"])
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cadjoint.enums import MeshMethod, MeshMethodLike, parse, values
from cadjoint.fem.hexmesh import GridSpec, HexMesh, sdf_to_hex_mesh
from cadjoint.fem.quality import (
    aspect_ratios,
    scaled_jacobians,
    tet_aspect_ratios,
    tet_radius_ratios,
)
from cadjoint.fem.tetmesh import TetMesh, sdf_to_tet_mesh, tet10_mesh

__all__ = ["SimMesh", "capture_sim_meshes"]

#: Supported meshing methods (the viewer round-trips these literals).  The
#: option set itself lives in :class:`cadjoint.enums.MeshMethod`; this is the
#: tuple of its spellings, in declaration order.
_METHODS = values(MeshMethod)

# Same default meshing volume as the implicit study path and the viewer's
# simulate mode; also the region the automatic domain-bounds scan samples.
_DEFAULT_BOUNDS = (-3.0, -3.0, -3.0)
_DEFAULT_SIZE = (6.0, 6.0, 6.0)
_SCAN_CELLS = 32

_CAPTURED_MESHES: ContextVar[list[SimMesh] | None] = ContextVar(
    "cadjoint_captured_sim_meshes",
    default=None,
)


@contextmanager
def capture_sim_meshes() -> Iterator[list[SimMesh]]:
    """Collect every :class:`SimMesh` constructed inside this context.

    Mirrors ``capture_studies``: the compile worker wraps user program
    execution in this context and receives the declared meshes in
    construction order.  Studies constructed inside the same context may
    refer to a captured mesh by name (``mesh="bracket-mesh"``).
    """
    meshes: list[SimMesh] = []
    token = _CAPTURED_MESHES.set(meshes)
    try:
        yield meshes
    finally:
        _CAPTURED_MESHES.reset(token)


def _register(mesh: SimMesh) -> None:
    captured = _CAPTURED_MESHES.get()
    if captured is not None:
        captured.append(mesh)


def _triplet(value: Any, label: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must contain three finite numbers, got {value!r}.")
    return (float(array[0]), float(array[1]), float(array[2]))


def _resolution_counts(resolution: Any) -> tuple[int, int, int]:
    counts = (resolution,) * 3 if isinstance(resolution, int) else tuple(resolution)
    if len(counts) != 3 or any(int(count) != count or count < 1 for count in counts):
        raise ValueError("resolution must be a positive integer or a triplet of them.")
    return tuple(int(count) for count in counts)


def _summary(values: np.ndarray) -> dict[str, float]:
    """Min/mean/max of a per-element metric, JSON-ready."""
    return {
        "min": round(float(values.min()), 6),
        "mean": round(float(values.mean()), 6),
        "max": round(float(values.max()), 6),
    }


def _scan_bounds(
    field_fn: Any, padding: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Tight bounds of the field's inside region plus padding, by coarse scan.

    Samples the default meshing volume on a ``_SCAN_CELLS``-per-axis lattice
    and takes the bounding box of the inside samples, expanded per side by
    ``padding`` plus one scan spacing (covering the discretization error of
    the scan itself).
    """
    import jax.numpy as jnp

    origin = np.asarray(_DEFAULT_BOUNDS, dtype=np.float64)
    extent = np.asarray(_DEFAULT_SIZE, dtype=np.float64)
    spacing = extent / _SCAN_CELLS
    axes = [origin[i] + (np.arange(_SCAN_CELLS + 1)) * spacing[i] for i in range(3)]
    lattice = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    values = np.asarray(field_fn(jnp.asarray(lattice)), dtype=np.float64).reshape(-1)
    inside = lattice[values < 0.0]
    if inside.size == 0:
        raise ValueError(
            "Automatic mesh bounds found no inside point in the default "
            f"volume {tuple(origin)} + {tuple(extent)}; pass bounds/size explicitly."
        )
    margin = float(padding) + spacing
    low = inside.min(axis=0) - margin
    high = inside.max(axis=0) + margin
    return _triplet(low, "bounds"), _triplet(high - low, "size")


@dataclass
class SimMesh:
    """A declarable, named meshing intent — the scene program's mesh object.

    Attributes:
        name: Mesh identifier (unique within a scene program; studies refer
            to it via ``mesh="<name>"``).
        resolution: Sampling-lattice cells per axis (positive int or
            triplet).  Elements for ``method="hex"``; the DC extraction
            grid for the tet methods (see the module docstring for the
            honest mapping).
        domain: Optional SDF (or plain callable field) selecting which part
            of the scene participates: when set, :meth:`build` meshes this
            field instead of the scene SDF handed to ``solve``/``build``.
        bounds: Lower corner of the meshing box, or None to derive it from
            the meshed field automatically (coarse inside-scan + padding).
        size: Extent of the meshing box; None exactly when ``bounds`` is.
        padding: Extra margin per side used only by the automatic bounds
            scan.
        method: Meshing method — a :class:`~cadjoint.enums.MeshMethod`
            or its plain string spelling: ``"hex"`` (voxelize+snap HEX8,
            the fast default), ``"tet4"`` (DC surface -> TetGen TET4), or
            ``"tet10"`` (the TET4 mesh promoted to quadratic tets — the
            quality path).  Normalised to the enum on construction.
    """

    name: str
    resolution: Any
    domain: Any = None
    bounds: Any = None
    size: Any = None
    padding: float = 0.1
    method: MeshMethodLike = MeshMethod.HEX

    _cache: tuple[Any, tuple, HexMesh | TetMesh] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("SimMesh needs a non-empty name.")
        self.method = parse(
            MeshMethod,
            self.method,
            f"method must be one of {list(_METHODS)}, got {self.method!r}.",
        )
        _resolution_counts(self.resolution)
        if (self.bounds is None) != (self.size is None):
            raise ValueError("bounds and size must be given together (or both omitted).")
        if self.bounds is not None:
            self.bounds = _triplet(self.bounds, "bounds")
            self.size = _triplet(self.size, "size")
            if any(extent <= 0.0 for extent in self.size):
                raise ValueError(f"size must be positive per axis, got {self.size!r}.")
        padding = float(self.padding)
        if not np.isfinite(padding) or padding < 0.0:
            raise ValueError(f"padding must be a finite non-negative number, got {self.padding!r}.")
        self.padding = padding
        if self.domain is not None and not callable(self.domain):
            raise TypeError(
                f"domain must be an SDF object or a callable field, got {type(self.domain).__name__}."
            )
        _register(self)

    # ── declaration ─────────────────────────────────────────────────────────

    def describe(self) -> dict[str, Any]:
        """JSON-ready declaration payload: everything the viewer displays.

        The ``domain`` entry records the selected object by its ``name``
        attribute when it has one (the viewer patches it by name) plus its
        type; meshes without a domain report ``None``.
        """
        return {
            "kind": "mesh",
            "name": self.name,
            "method": str(self.method),
            "resolution": self.resolution
            if isinstance(self.resolution, int)
            else list(self.resolution),
            "bounds": list(self.bounds) if self.bounds is not None else None,
            "size": list(self.size) if self.size is not None else None,
            "padding": self.padding,
            "domain": _domain_entry(self.domain),
        }

    # ── building ────────────────────────────────────────────────────────────

    def _field(self, sdf: Any) -> Any:
        """The field this mesh discretizes: the domain, else the scene SDF."""
        if self.domain is not None:
            return self.domain
        if sdf is None:
            raise ValueError(
                f"SimMesh {self.name!r} declares no domain; pass the scene SDF to build/solve."
            )
        if not callable(sdf):
            raise TypeError("build() expects an SDF object or a callable field.")
        return sdf

    def _parameters(self) -> tuple:
        return (
            self.method,
            _resolution_counts(self.resolution),
            self.bounds,
            self.size,
            self.padding,
        )

    def grid(self, sdf: Any = None) -> GridSpec:
        """The sampling grid this mesh uses (resolving automatic bounds).

        For ``method="hex"`` the grid cells are the candidate elements;
        for the tet methods it is the DC extraction lattice.

        Args:
            sdf: Scene SDF, needed only when the mesh has no ``domain`` and
                automatic bounds must scan the field.
        """
        if self.bounds is not None:
            bounds, size = self.bounds, self.size
        else:
            bounds, size = _scan_bounds(self._field(sdf), self.padding)
        return GridSpec.from_bounds(bounds, size, _resolution_counts(self.resolution))

    def build(self, sdf: Any = None, *, rebuild: bool = False) -> HexMesh | TetMesh:
        """Extract (or reuse) the volume mesh for the current parameters.

        The result is cached on the instance and reused while the meshing
        parameters and the meshed field object stay the same; pass
        ``rebuild=True`` after mutating the field's parameter values in
        place to force re-extraction.

        Args:
            sdf: Scene SDF object or callable field; ignored when the mesh
                declares a ``domain``, required otherwise.
            rebuild: Discard the cached mesh and re-extract.

        Returns:
            The extracted :class:`~cadjoint.fem.hexmesh.HexMesh`
            (``method="hex"``) or :class:`~cadjoint.fem.tetmesh.TetMesh`
            (``method="tet4"``/``"tet10"``; requires ``tetgen``).  Tet
            extraction runs :func:`~cadjoint.fem.tetmesh.sdf_to_tet_mesh`'s
            refinement ladder, which tries exact sharp-feature DC placement
            and the more robust Tikhonov placement at every rung and
            records what it tried on ``mesh.refinement``; when no rung
            works the TetGen error propagates, naming both ends of the
            ladder.  The tet grid must fully contain the zero surface
            (unlike voxelization, DC needs the closed boundary).
        """
        field_fn = self._field(sdf)
        parameters = self._parameters()
        cached = self._cache
        if not rebuild and cached is not None and cached[0] is field_fn and cached[1] == parameters:
            return cached[2]
        if self.method == MeshMethod.HEX:
            mesh: HexMesh | TetMesh = sdf_to_hex_mesh(field_fn, self.grid(sdf))
        else:
            # No sharp=True/sharp=False retry here: the ladder inside
            # sdf_to_tet_mesh already tries both placements at every rung,
            # so wrapping it in one would walk the whole ladder twice.
            mesh = sdf_to_tet_mesh(field_fn, self.grid(sdf))
            if self.method == MeshMethod.TET10:
                mesh = tet10_mesh(mesh)
        self._cache = (field_fn, parameters, mesh)
        return mesh

    # ── inspection ──────────────────────────────────────────────────────────

    def quality(self, sdf: Any = None) -> dict[str, np.ndarray]:
        """Per-element quality metrics of the built mesh.

        Builds the mesh first if needed (same caching as :meth:`build`).

        Returns:
            ``{"scaled_jacobian": (C,), "aspect_ratio": (C,)}`` float64
            arrays for hex meshes (see
            :func:`~cadjoint.fem.quality.scaled_jacobians` /
            :func:`~cadjoint.fem.quality.aspect_ratios`);
            ``{"radius_ratio": (C,), "aspect_ratio": (C,)}`` for tet
            meshes (see :func:`~cadjoint.fem.quality.tet_radius_ratios` /
            :func:`~cadjoint.fem.quality.tet_aspect_ratios`; TET10 metrics
            are those of the straight-sided corner tets).
        """
        mesh = self.build(sdf)
        if isinstance(mesh, TetMesh):
            return {
                "radius_ratio": tet_radius_ratios(mesh.points, mesh.cells),
                "aspect_ratio": tet_aspect_ratios(mesh.points, mesh.cells),
            }
        return {
            "scaled_jacobian": scaled_jacobians(mesh.points, mesh.cells),
            "aspect_ratio": aspect_ratios(mesh.points, mesh.cells),
        }

    def inspect(self, sdf: Any = None) -> dict[str, Any]:
        """JSON-ready inspection summary of the built mesh.

        Builds the mesh first if needed (same caching as :meth:`build`).

        Returns:
            ``{"name", "method", "nodes", "elements", "bounds": {"min",
            "max"}, "grid": {"origin", "spacing", "cells"}, "quality":
            {...}}`` where each quality entry is a ``{"min", "mean",
            "max"}`` summary over elements.  The shape is
            method-agnostic; only the metric names under ``"quality"``
            differ (:meth:`quality`), so a stats display can iterate them.
        """
        mesh = self.build(sdf)
        metrics = self.quality(sdf)
        grid = mesh.grid
        return {
            "name": self.name,
            "method": str(self.method),
            "nodes": mesh.num_points,
            "elements": mesh.num_cells,
            "bounds": {
                "min": [round(float(value), 6) for value in mesh.points.min(axis=0)],
                "max": [round(float(value), 6) for value in mesh.points.max(axis=0)],
            },
            "grid": {
                "origin": [round(float(value), 6) for value in grid.origin],
                "spacing": [round(float(value), 6) for value in grid.spacing],
                "cells": list(grid.cells),
            }
            if grid is not None
            else None,
            "quality": {name: _summary(values) for name, values in metrics.items()},
        }


def _domain_entry(domain: Any) -> dict[str, Any] | None:
    """Describe a domain object by name (when it has one) and type."""
    if domain is None:
        return None
    name = getattr(domain, "name", None)
    return {
        "name": name if isinstance(name, str) else None,
        "type": type(domain).__name__,
    }


def _anonymous(
    name: str,
    resolution: Any,
    domain: Any = None,
    bounds: Any = None,
    size: Any = None,
) -> SimMesh:
    """Build a SimMesh without registering it in an active capture context.

    The auto-wrap for studies declared without an explicit mesh: the study
    still meshes through the one :class:`SimMesh` path, but the anonymous
    mesh never shows up next to user-declared meshes.
    """
    token = _CAPTURED_MESHES.set(None)
    try:
        return SimMesh(name=name, resolution=resolution, domain=domain, bounds=bounds, size=size)
    finally:
        _CAPTURED_MESHES.reset(token)
