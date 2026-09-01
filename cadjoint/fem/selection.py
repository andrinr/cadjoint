"""Programmatic node (vertex) selection for boundary conditions.

Selections are small declarative values describing a set of mesh vertices.
They are built from the :class:`Nodes` factory namespace, composed with the
boolean operators ``&``, ``|`` and ``~``, and evaluated against a
:class:`~cadjoint.fem.hexmesh.HexMesh` only when a study is solved — so the
same selection works across resolutions and remeshes.

Boundary restriction is implicit: **selections always resolve to boundary
(surface) nodes**.  Boundary conditions only ever act on the surface of the
meshed part, so interior lattice nodes are never selected — and
``~selection`` therefore means "the rest of the surface", not "the mesh
interior".

Every selection except :meth:`Nodes.predicate` is serializable:
:meth:`NodeSelection.describe` emits a JSON-ready dict of the kind plus its
numeric parameters, and :func:`selection_from_description` reconstructs the
selection from that payload.  Constructor arguments are plain floats and
3-lists on purpose, so a UI can emit selections as literal source text
(``Nodes.box([0, 0, 0], [1, 1, 1])``).

Example::

    clamp = Nodes.sphere([-0.7, 0.35, 0.1], 0.33) | Nodes.sphere([0.7, 0.35, 0.1], 0.33)
    load = Nodes.side("+x") & ~Nodes.halfspace([0.0, 0.0, 0.0], [0.0, 0.0, -1.0])
    indices = load.resolve(mesh)  # int32 node indices, boundary only
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from cadjoint.fem.hexmesh import HexMesh

__all__ = [
    "NodeSelection",
    "Nodes",
    "boundary_node_mask",
    "selection_from_description",
]

_SIDES = ("+x", "-x", "+y", "-y", "+z", "-z")


def _triplet(value: Any, label: str) -> tuple[float, float, float]:
    """Validate a 3-vector of finite numbers, returning a plain float tuple."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must contain three finite numbers, got {value!r}.")
    return (float(array[0]), float(array[1]), float(array[2]))


def boundary_node_mask(mesh: HexMesh) -> np.ndarray:
    """Boolean mask over ``mesh`` nodes that lie on the boundary surface."""
    mask = np.zeros(mesh.num_points, dtype=bool)
    mask[np.unique(mesh.all_boundary_faces().nodes)] = True
    return mask


class NodeSelection:
    """A composable, mesh-independent description of a node set.

    Build concrete selections via the :class:`Nodes` factory namespace and
    combine them with ``&`` (intersection), ``|`` (union) and ``~``
    (complement within the boundary surface).  Evaluate with :meth:`mask`
    (boolean mask over all mesh nodes, true only at boundary nodes) or
    :meth:`resolve` (selected node indices, raising if empty).
    """

    @property
    def serializable(self) -> bool:
        """Whether :meth:`describe` round-trips (false only for predicates)."""
        return True

    def mask(self, mesh: HexMesh) -> np.ndarray:
        """Boolean node mask over ``mesh``, restricted to boundary nodes."""
        points = np.asarray(mesh.points, dtype=np.float64)
        geometric = np.asarray(self._geometric(points, mesh), dtype=bool).reshape(-1)
        if geometric.shape[0] != mesh.num_points:
            raise ValueError(
                f"Selection {self.describe()} produced a mask of {geometric.shape[0]} "
                f"entries for a mesh with {mesh.num_points} nodes. Predicates must be "
                "vectorized: fn(points) receives the (N, 3) node-position array and "
                "returns a boolean mask of shape (N,)."
            )
        return geometric & boundary_node_mask(mesh)

    def resolve(self, mesh: HexMesh) -> np.ndarray:
        """Selected node indices of ``mesh`` as an int32 array.

        Raises:
            ValueError: If the selection matches no boundary node.
        """
        indices = np.flatnonzero(self.mask(mesh)).astype(np.int32)
        if indices.size == 0:
            raise ValueError(f"Selection {self.describe()} matched no boundary nodes.")
        return indices

    def describe(self) -> dict[str, Any]:
        """JSON-ready description: kind plus numeric parameters."""
        raise NotImplementedError

    def _geometric(self, points: np.ndarray, mesh: HexMesh) -> np.ndarray:
        """Geometric criterion over all node positions (pre boundary cut)."""
        raise NotImplementedError

    def __and__(self, other: NodeSelection) -> NodeSelection:
        return _Combined("and", self, _expect_selection(other))

    def __or__(self, other: NodeSelection) -> NodeSelection:
        return _Combined("or", self, _expect_selection(other))

    def __invert__(self) -> NodeSelection:
        return _Complement(self)


def _expect_selection(value: Any) -> NodeSelection:
    if not isinstance(value, NodeSelection):
        raise TypeError(
            f"Expected a NodeSelection to combine with, got {type(value).__name__}. "
            "Build one via Nodes.box/sphere/halfspace/side/predicate."
        )
    return value


@dataclass(frozen=True)
class _Box(NodeSelection):
    min_corner: tuple[float, float, float]
    max_corner: tuple[float, float, float]

    def _geometric(self, points: np.ndarray, _mesh: HexMesh) -> np.ndarray:
        low = np.asarray(self.min_corner)
        high = np.asarray(self.max_corner)
        return np.all((points >= low) & (points <= high), axis=-1)

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "box",
            "min_corner": list(self.min_corner),
            "max_corner": list(self.max_corner),
        }


@dataclass(frozen=True)
class _Sphere(NodeSelection):
    center: tuple[float, float, float]
    radius: float

    def _geometric(self, points: np.ndarray, _mesh: HexMesh) -> np.ndarray:
        offsets = points - np.asarray(self.center)
        return np.einsum("nd,nd->n", offsets, offsets) <= self.radius**2

    def describe(self) -> dict[str, Any]:
        return {"kind": "sphere", "center": list(self.center), "radius": self.radius}


@dataclass(frozen=True)
class _Halfspace(NodeSelection):
    point: tuple[float, float, float]
    normal: tuple[float, float, float]

    def _geometric(self, points: np.ndarray, _mesh: HexMesh) -> np.ndarray:
        return (points - np.asarray(self.point)) @ np.asarray(self.normal) >= 0.0

    def describe(self) -> dict[str, Any]:
        return {"kind": "halfspace", "point": list(self.point), "normal": list(self.normal)}


@dataclass(frozen=True)
class _Side(NodeSelection):
    side: str
    tol: float | None = None

    def _geometric(self, points: np.ndarray, mesh: HexMesh) -> np.ndarray:
        axis = "xyz".index(self.side[1])
        positive = self.side[0] == "+"
        coords = points[:, axis]
        boundary = coords[boundary_node_mask(mesh)]
        extreme = boundary.max() if positive else boundary.min()
        tol = self.tol if self.tol is not None else _default_side_tol(mesh)
        return coords >= extreme - tol if positive else coords <= extreme + tol

    def describe(self) -> dict[str, Any]:
        return {"kind": "side", "side": self.side, "tol": self.tol}


def _default_side_tol(mesh: HexMesh) -> float:
    """Half the smallest cell spacing (fallback: 1e-3 of the bbox diagonal)."""
    if mesh.grid is not None:
        return 0.5 * float(min(mesh.grid.spacing))
    extent = mesh.points.max(axis=0) - mesh.points.min(axis=0)
    return 1e-3 * float(np.linalg.norm(extent))


@dataclass(frozen=True)
class _Predicate(NodeSelection):
    fn: Callable[[np.ndarray], Any]

    @property
    def serializable(self) -> bool:
        return False

    def _geometric(self, points: np.ndarray, _mesh: HexMesh) -> np.ndarray:
        return self.fn(points)

    def describe(self) -> dict[str, Any]:
        return {"kind": "predicate", "name": getattr(self.fn, "__name__", "<callable>")}


@dataclass(frozen=True)
class _Combined(NodeSelection):
    op: str  # "and" | "or"
    left: NodeSelection
    right: NodeSelection

    @property
    def serializable(self) -> bool:
        return self.left.serializable and self.right.serializable

    def mask(self, mesh: HexMesh) -> np.ndarray:
        left = self.left.mask(mesh)
        right = self.right.mask(mesh)
        return left & right if self.op == "and" else left | right

    def describe(self) -> dict[str, Any]:
        return {"kind": self.op, "operands": [self.left.describe(), self.right.describe()]}


@dataclass(frozen=True)
class _Complement(NodeSelection):
    operand: NodeSelection

    @property
    def serializable(self) -> bool:
        return self.operand.serializable

    def mask(self, mesh: HexMesh) -> np.ndarray:
        return boundary_node_mask(mesh) & ~self.operand.mask(mesh)

    def describe(self) -> dict[str, Any]:
        return {"kind": "not", "operand": self.operand.describe()}


class Nodes:
    """Factory namespace for :class:`NodeSelection` values.

    All constructors take plain floats and 3-lists so a UI can emit them as
    literal source text; the resulting selections are combined with ``&``,
    ``|`` and ``~`` and always resolve to boundary nodes.
    """

    def __init__(self) -> None:
        raise TypeError("Nodes is a factory namespace; call its static methods instead.")

    @staticmethod
    def box(min_corner: Any, max_corner: Any) -> NodeSelection:
        """Nodes inside the axis-aligned box ``[min_corner, max_corner]``.

        Args:
            min_corner: Lower corner, three finite numbers.
            max_corner: Upper corner, three finite numbers (componentwise
                at least ``min_corner``).
        """
        low = _triplet(min_corner, "min_corner")
        high = _triplet(max_corner, "max_corner")
        if any(lo > hi for lo, hi in zip(low, high)):
            raise ValueError(f"min_corner {low} must not exceed max_corner {high}.")
        return _Box(low, high)

    @staticmethod
    def sphere(center: Any, radius: Any) -> NodeSelection:
        """Nodes within ``radius`` of ``center``.

        Args:
            center: Sphere center, three finite numbers.
            radius: Positive finite radius.
        """
        value = float(radius)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"radius must be positive and finite, got {radius!r}.")
        return _Sphere(_triplet(center, "center"), value)

    @staticmethod
    def halfspace(point: Any, normal: Any) -> NodeSelection:
        """Nodes on the ``normal`` side of the plane through ``point``.

        Selects nodes ``x`` with ``dot(x - point, normal) >= 0``.

        Args:
            point: Any point on the plane, three finite numbers.
            normal: Plane normal pointing into the selected halfspace
                (need not be unit length, must be nonzero).
        """
        direction = _triplet(normal, "normal")
        if not any(component != 0.0 for component in direction):
            raise ValueError("normal must be nonzero.")
        return _Halfspace(_triplet(point, "point"), direction)

    @staticmethod
    def side(side: str, tol: float | None = None) -> NodeSelection:
        """Nodes on the axis-extreme boundary plane of the mesh.

        ``Nodes.side("+x")`` selects boundary nodes whose x coordinate is
        within ``tol`` of the largest x over all boundary nodes — for a bar
        this is the end face; for curved geometry it is the extreme cap.

        Args:
            side: One of ``"+x"``, ``"-x"``, ``"+y"``, ``"-y"``, ``"+z"``,
                ``"-z"``.
            tol: Capture distance from the extreme plane.  ``None`` (the
                default) resolves per mesh to half the smallest cell
                spacing.
        """
        if side not in _SIDES:
            raise ValueError(f"side must be one of {_SIDES}, got {side!r}.")
        if tol is not None:
            tol = float(tol)
            if not np.isfinite(tol) or tol < 0.0:
                raise ValueError(f"tol must be a finite non-negative number, got {tol!r}.")
        return _Side(side, tol)

    @staticmethod
    def predicate(fn: Callable[[np.ndarray], Any]) -> NodeSelection:
        """Escape hatch: select nodes by an arbitrary vectorized callable.

        ``fn`` receives the ``(N, 3)`` array of all mesh node positions and
        returns a boolean mask of shape ``(N,)``; the result is intersected
        with the boundary-node set like every other selection.  Predicate
        selections are the one non-serializable kind: :meth:`~NodeSelection.
        describe` reports only the callable's name.

        Args:
            fn: Vectorized ``(N, 3) -> (N,)`` boolean callable.
        """
        if not callable(fn):
            raise ValueError(f"predicate needs a callable, got {fn!r}.")
        return _Predicate(fn)


def selection_from_description(payload: dict[str, Any]) -> NodeSelection:
    """Rebuild a :class:`NodeSelection` from its :meth:`~NodeSelection.describe` payload.

    Args:
        payload: A description dict as produced by ``describe()``.

    Returns:
        The equivalent selection.

    Raises:
        ValueError: For unknown kinds, and for ``"predicate"`` payloads —
            predicates carry a callable and do not serialize.
    """
    kind = payload.get("kind")
    if kind == "box":
        return Nodes.box(payload["min_corner"], payload["max_corner"])
    if kind == "sphere":
        return Nodes.sphere(payload["center"], payload["radius"])
    if kind == "halfspace":
        return Nodes.halfspace(payload["point"], payload["normal"])
    if kind == "side":
        return Nodes.side(payload["side"], payload.get("tol"))
    if kind in ("and", "or"):
        left, right = payload["operands"]
        return _Combined(kind, selection_from_description(left), selection_from_description(right))
    if kind == "not":
        return _Complement(selection_from_description(payload["operand"]))
    if kind == "predicate":
        raise ValueError(
            f"Predicate selection {payload.get('name', '<callable>')!r} is not "
            "serializable; reconstruct it in code via Nodes.predicate."
        )
    raise ValueError(f"Unknown selection kind {kind!r}.")
