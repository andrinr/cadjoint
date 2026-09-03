"""The in-process plugin contracts: what the private tier implements against.

A Tesseract is the right contract for a *coarse* component — arrays in,
arrays out, one ``apply`` and one ``vector_jacobian_product`` per optimizer
step, where a ~10 ms round trip is 1–2 % of the step.  It is the wrong
grain for a component called inside a trace or once per compile: a Newton
kernel under ``vmap``, a curve tracer's step, the viewer's edge overlay.
Those cross the seam as **Python objects in this process**, and this module
is the one place their interface is written down: a typed
:class:`typing.Protocol` per plugin kind, and the frozen-dataclass payloads
that cross with them.

The five kinds here are the ones the private tier (``diff-brep``) fills.
Public cadjoint never imports it; it asks the registry for a *kind*
(:func:`cadjoint.plugins.plugin_for_kind`, or :func:`cadjoint.tier.require`
for the refusal text) and calls the object it gets back through the
Protocol below.  A ``PluginSpec`` with ``transport="python"`` names the
object; :class:`cadjoint.plugins.PythonPlugin` binds it.

===================  ===========================  ==========================
kind                 Protocol                     payload
===================  ===========================  ==========================
``node_map``         :class:`NodeMap`             :class:`OwnedNodes` in, ``(P, 3)`` positions out, differentiable in the design
``feature_edges``    :class:`FeatureEdges`        :class:`EdgeSet` out (NumPy, no derivative)
``brep``             :class:`BRepExtractor`       an opaque B-rep that never leaves the process
``step_export``      :class:`StepExporter`        a file, and a report mapping
``drag``             :class:`Drag`                :class:`DragOutcome`
===================  ===========================  ==========================

**Versioning.**  :data:`CONTRACT_VERSION` bumps whenever anything in this
module changes shape.  Every component carries ``contract_version``;
:func:`cadjoint.tier.require` refuses a mismatch with a sentence rather than
letting a stale private build fail inside a trace.

**Differentiability** is declared in the type: an argument or result
annotated ``Differentiable[...]`` (a :data:`typing.Annotated` tag) is one
the component's own ``custom_vjp`` carries a derivative through.
:func:`payload_types` reads the tags back, which is how a python-transport
plugin reports ``capabilities`` without pydantic or tesseract-core.
"""

from __future__ import annotations

import hashlib
import inspect
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Annotated, Any, Protocol, TypeVar, runtime_checkable

import numpy as np

from cadjoint.enums import PluginKind

__all__ = [
    "CONTRACT_VERSION",
    "DIFFERENTIABLE",
    "BRepExtractor",
    "Differentiable",
    "Drag",
    "DragOutcome",
    "EdgeSet",
    "FeatureEdges",
    "KIND_CONTRACTS",
    "NodeMap",
    "OwnedNodes",
    "StepExporter",
    "contract_for",
    "contract_signature",
    "payload_types",
    "primary_method",
]

#: Bumped whenever a Protocol or payload in this module changes shape.
CONTRACT_VERSION = 1

#: The :data:`typing.Annotated` tag marking a differentiable argument or result.
DIFFERENTIABLE = "cadjoint.plugins.differentiable"

_T = TypeVar("_T")

Differentiable = Annotated[_T, DIFFERENTIABLE]
"""``Differentiable[Array]``: the component carries a derivative through this."""

# The array types below are documentation for the reader and the
# capability scan, not runtime checks: JAX arrays in, JAX arrays out.
Array = Any


# ── payloads ────────────────────────────────────────────────────────────────


def _as(array: Any, dtype: Any, shape_tail: tuple[int, ...], name: str) -> np.ndarray:
    """Coerce one record field, checking its trailing shape."""
    out = np.ascontiguousarray(np.asarray(array, dtype=dtype))
    if out.ndim != 1 + len(shape_tail) or tuple(out.shape[1:]) != shape_tail:
        raise ValueError(
            f"OwnedNodes.{name} must be shaped (n, {', '.join(map(str, shape_tail))}) "
            f"or (n,); got {out.shape}."
        )
    return out


@dataclass(frozen=True)
class OwnedNodes:
    """Every node of a Gmsh mesh, with the patches that own it — the seam's record.

    Produced by the **public** Gmsh route
    (:func:`cadjoint.fem.gmsh.assign_ownership`, a residual test against
    the scene's public patch fields and nothing else) and consumed by the
    **private** ``node_map``, which re-solves each node against its owners
    under a design change.  Rows follow the
    :class:`~cadjoint.fem.tetmesh.TetMesh` layout: boundary corners, then
    interior corners, then the midside block, so ``num_surface`` and
    ``edge_parents`` describe contiguous blocks.

    Attributes:
        seeds: Node positions at the design they were meshed at, ``(P, 3)``
            float64.
        patches: Per node, the global patch indices (in
            :func:`cadjoint.meshing.patch_fields.scene_patch_fields` order,
            leaves flattened) it must satisfy, ``(P, 3)`` int32,
            ``-1``-padded.
        arity: Count of non-``-1`` entries per row, ``(P,)`` int8; ``0`` for
            a volume node and for a blend node.
        entity_dim: The Gmsh entity the node lies on — 0 vertex, 1 curve,
            2 surface, 3 volume — ``(P,)`` int8.
        blend: Boundary nodes no patch owns (``arity == 0`` and
            ``entity_dim < 3``): the scene's own zero set holds them.
        midside: Rows of the order-2 midside block, ``(P,)`` bool.
        edge_parents: Corner pairs of the midside block, ``(M, 2)`` int32;
            row ``k`` describes node ``num_corner + k``.  Empty at order 1.
        cells: The frozen connectivity, ``(T, 4 | 10)`` int32 in meshio
            order.  Carried because the map's interior follow needs the node
            adjacency, which nothing else in the record can supply.
        bar: The residual bar ``|f_patch| <= bar`` ownership was decided
            at, in model units.
        design: The free-parameter values at meshing time, name to array,
            so the map can check it is being asked about the design the
            seeds belong to.
    """

    seeds: np.ndarray
    patches: np.ndarray
    arity: np.ndarray
    entity_dim: np.ndarray
    blend: np.ndarray
    midside: np.ndarray
    edge_parents: np.ndarray
    cells: np.ndarray
    bar: float
    design: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        set_(self, "seeds", _as(self.seeds, np.float64, (3,), "seeds"))
        count = self.seeds.shape[0]
        set_(self, "patches", _as(self.patches, np.int32, (3,), "patches"))
        set_(self, "arity", _as(self.arity, np.int8, (), "arity"))
        set_(self, "entity_dim", _as(self.entity_dim, np.int8, (), "entity_dim"))
        set_(self, "blend", _as(self.blend, bool, (), "blend"))
        set_(self, "midside", _as(self.midside, bool, (), "midside"))
        set_(self, "edge_parents", _as(self.edge_parents, np.int32, (2,), "edge_parents"))
        cells = np.ascontiguousarray(np.asarray(self.cells, dtype=np.int32))
        if cells.ndim != 2 or cells.shape[1] not in (4, 10):
            raise ValueError(
                f"OwnedNodes.cells must be shaped (T, 4) or (T, 10); got {cells.shape}."
            )
        set_(self, "cells", cells)
        for name in ("patches", "arity", "entity_dim", "blend", "midside"):
            if getattr(self, name).shape[0] != count:
                raise ValueError(
                    f"OwnedNodes.{name} has {getattr(self, name).shape[0]} rows but seeds has {count}."
                )
        if int(self.midside.sum()) != self.edge_parents.shape[0]:
            raise ValueError(
                f"OwnedNodes marks {int(self.midside.sum())} midside rows but edge_parents "
                f"has {self.edge_parents.shape[0]}."
            )
        if ((self.patches >= 0).sum(axis=1) != self.arity).any():
            raise ValueError("OwnedNodes.arity must count the non -1 entries of each patches row.")
        if (self.blend & (self.arity > 0)).any() or (self.blend & (self.entity_dim >= 3)).any():
            raise ValueError("OwnedNodes.blend must mark exactly the unowned boundary nodes.")
        set_(self, "bar", float(self.bar))
        set_(
            self,
            "design",
            {str(k): np.asarray(v, dtype=np.float64) for k, v in dict(self.design).items()},
        )

    # ── layout ──────────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        """Number of nodes, midsides included."""
        return int(self.seeds.shape[0])

    @property
    def num_corner(self) -> int:
        """Number of corner nodes (the leading rows)."""
        return self.count - int(self.edge_parents.shape[0])

    @property
    def num_surface(self) -> int:
        """Number of leading boundary corner nodes."""
        return int((self.entity_dim[: self.num_corner] < 3).sum())

    @property
    def order(self) -> int:
        """Element order: 1 without a midside block, 2 with one."""
        return 2 if self.edge_parents.shape[0] else 1

    @property
    def owned(self) -> np.ndarray:
        """Mask of the patch-owned nodes (``arity > 0``)."""
        return self.arity > 0

    def arity_counts(self) -> dict[int, int]:
        """``arity -> node count`` for arities 0 to 3."""
        return {k: int((self.arity == k).sum()) for k in (0, 1, 2, 3)}

    def design_digest(self) -> str:
        """A short hash of the design the seeds were meshed at."""
        digest = hashlib.sha256()
        for name in sorted(self.design):
            digest.update(name.encode("utf-8"))
            digest.update(np.ascontiguousarray(self.design[name]).tobytes())
        return digest.hexdigest()[:16]

    # ── round trip ──────────────────────────────────────────────────────

    def to_mapping(self) -> dict[str, Any]:
        """The record as a plain mapping of arrays (what a wire form carries)."""
        return {
            **{f.name: getattr(self, f.name) for f in fields(self) if f.name != "design"},
            "design": dict(self.design),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> OwnedNodes:
        """Rebuild the record from :meth:`to_mapping`'s output."""
        return cls(**{f.name: data[f.name] for f in fields(cls)})


@dataclass(frozen=True)
class EdgeSet:
    """The feature curves of a scene, as polylines ready to draw.

    What the ``feature_edges`` kind returns.  NumPy throughout, no
    derivative: these are display segments.

    Attributes:
        polylines: One ``(k, 3)`` array per curve, ordered along it; a
            closed curve does not repeat its first point.
        closed: Per curve, whether it closes on itself, ``(n,)`` bool.
        patches: Per curve, the two global patch indices whose zero sets
            meet on it, ``(n, 2)`` int32.
        kind: Per curve, how it was produced (``"traced"``, ``"sampled"``).
        residual: Per curve, the largest ``|f|`` its points carry against
            its two patches, ``(n,)`` float64.
        vertices: Per curve, the triple-point indices at its two ends,
            ``(n, 2)`` int32, ``-1`` where an end is free.
        stats: Counts and timings.  ``stats["mesh"]``, when present, is the
            ``(points, quads)`` pair of the dual-contour pass the extraction
            ran, so the public wire layer can reuse it instead of running a
            second pass.
    """

    polylines: tuple[np.ndarray, ...]
    closed: np.ndarray
    patches: np.ndarray
    kind: tuple[str, ...]
    residual: np.ndarray
    vertices: np.ndarray
    stats: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        polylines = tuple(np.asarray(p, dtype=np.float64).reshape(-1, 3) for p in self.polylines)
        set_(self, "polylines", polylines)
        count = len(polylines)
        set_(self, "closed", np.asarray(self.closed, dtype=bool).reshape(-1))
        set_(self, "patches", np.asarray(self.patches, dtype=np.int32).reshape(-1, 2))
        set_(self, "kind", tuple(str(k) for k in self.kind))
        set_(self, "residual", np.asarray(self.residual, dtype=np.float64).reshape(-1))
        set_(self, "vertices", np.asarray(self.vertices, dtype=np.int32).reshape(-1, 2))
        for name in ("closed", "patches", "kind", "residual", "vertices"):
            if len(getattr(self, name)) != count:
                raise ValueError(
                    f"EdgeSet.{name} has {len(getattr(self, name))} rows for {count} curves."
                )

    @property
    def count(self) -> int:
        """Number of curves."""
        return len(self.polylines)

    def chords(self) -> np.ndarray:
        """Every curve as consecutive point pairs, shaped ``(m, 2, 3)``."""
        chords = []
        for points, closed in zip(self.polylines, self.closed):
            if points.shape[0] < 2:
                continue
            following = np.roll(points, -1, axis=0) if closed else points[1:]
            leading = points if closed else points[:-1]
            chords.append(np.stack([leading, following], axis=1))
        if not chords:
            return np.empty((0, 2, 3), dtype=np.float64)
        return np.concatenate(chords)

    def to_mapping(self) -> dict[str, Any]:
        """The set as a plain mapping (``stats`` excluded)."""
        return {
            "polylines": list(self.polylines),
            "closed": self.closed,
            "patches": self.patches,
            "kind": list(self.kind),
            "residual": self.residual,
            "vertices": self.vertices,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> EdgeSet:
        """Rebuild the set from :meth:`to_mapping`'s output."""
        return cls(
            polylines=tuple(data["polylines"]),
            closed=data["closed"],
            patches=data["patches"],
            kind=tuple(data["kind"]),
            residual=data["residual"],
            vertices=data["vertices"],
        )


@dataclass(frozen=True)
class DragOutcome:
    """What a drag did, and why — the ``drag`` kind's result.

    Attributes:
        handle: The handle dragged, as ``"vertex:3"`` or ``"edge:12@0.50"``.
        target: The requested position, ``(3,)``.
        achieved: Where the handle ended up, ``(3,)``.
        error: ``|achieved - target|``.
        parameters: The full solved free-parameter mapping.
        delta: Per-parameter change from the starting design.
        moved: ``(name, magnitude)`` for every parameter that moved, largest
            first.
        constraint_residual: Largest ``|c(theta)|`` after the solve.
        topology_changed: Whether the drag would take the handle off the
            solid's boundary, which a frozen graph cannot represent.
        applied: Whether ``parameters`` was written back into the scene.
        reason: A one-line explanation when the drag was refused.
        iterations: Solver steps taken.
    """

    handle: str
    target: np.ndarray
    achieved: np.ndarray
    error: float
    parameters: dict[str, Any]
    delta: dict[str, np.ndarray]
    moved: list[tuple[str, float]]
    constraint_residual: float
    topology_changed: bool
    applied: bool
    reason: str
    iterations: int


# ── the Protocols ───────────────────────────────────────────────────────────


@runtime_checkable
class NodeMap(Protocol):
    """``node_map``: design parameters to the node positions of a Gmsh mesh.

    The seam's centrepiece.  Its input is the public :class:`OwnedNodes`
    record; its output is positions, traced in the parameters, with the
    implicit-function adjoint of the component's own projection kernel.

    Contract:
        Preconditions: ``owned.seeds`` were produced at ``owned.design``;
            every ``patches`` row names patches of ``scene``'s table in
            :func:`~cadjoint.meshing.patch_fields.scene_patch_fields`
            order; midsides' parents precede them (the ``TetMesh`` layout).
        Forward: a boundary node of arity ``k >= 1`` is one ``k``-field
            Gauss–Newton from its seed onto its owning patches — the seed
            chooses the branch, never the position on it; a blend node is
            solved against the scene itself; an order-2 midside is solved
            on its own patch set, not put at the chord midpoint; interior
            nodes follow the boundary displacement by ``smooth_passes``
            Laplacian passes over the node adjacency.
        Frozen: topology, ownership, arity, adjacency, the pass count.
        Differentiable: ``params`` only, by the IFT; a node whose Gram
            fails the transversality guard carries zero derivative.
        Postconditions: at ``owned.design``, ``|positions - seeds| <= bar``
            for every patch-owned node; every boundary node satisfies
            ``|f| <= tol`` on its patches.
        Refuses: a patch index outside the table.
    """

    version: str
    contract_version: int

    def positions(
        self,
        scene: Any,
        params: Mapping[str, Any],
        owned: OwnedNodes,
        *,
        smooth_passes: int = 0,
    ) -> Differentiable[Array]:
        """``(P, 3)`` node positions at ``params``, traced in ``params``.

        Args:
            scene: The root SDF node the mesh was built from.  The object,
                not a functionalized closure: the patch table is rebuilt
                under the traced parameter values by walking it.
            params: Free-parameter mapping; a partial mapping merges over
                the scene's current values.
            owned: The public ownership record.
            smooth_passes: Interior Laplacian passes (0 leaves volume nodes
                at their seeds).
        """
        ...


@runtime_checkable
class FeatureEdges(Protocol):
    """``feature_edges``: the exact feature curves of a scene, to draw."""

    version: str
    contract_version: int

    def feature_edges(
        self,
        scene: Any,
        grid: Any,
        *,
        design_leaves: np.ndarray | None = None,
        blend_tolerance: float | None = None,
        steps: int = 4,
    ) -> EdgeSet:
        """The curves where two patch zero sets cross, on ``grid``.

        Args:
            scene: Root SDF node.
            grid: The :class:`~cadjoint.meshing.GridSpec` to discover on.
            design_leaves: World-frame leaf indices whose curves are drawn
                (both patches of a curve must belong); ``None`` draws all.
            blend_tolerance: ``|f_patch|`` above which a surface is a blend
                rather than the patch it rounds.
            steps: Newton iterations per projection.
        """
        ...


@runtime_checkable
class BRepExtractor(Protocol):
    """``brep``: the derived boundary representation, kept in-process."""

    version: str
    contract_version: int

    def extract(self, scene: Any, grid: Any, **options: Any) -> Any:
        """Derive the B-rep of ``scene`` on ``grid``; the object is opaque."""
        ...


@runtime_checkable
class StepExporter(Protocol):
    """``step_export``: analytic STEP from the derived B-rep."""

    version: str
    contract_version: int

    def step_export(self, scene: Any, grid: Any, path: Any, **options: Any) -> dict[str, Any]:
        """Write ``scene`` as STEP to ``path``; return the writer's report."""
        ...


@runtime_checkable
class Drag(Protocol):
    """``drag``: move a B-rep handle by solving for the design that fits."""

    version: str
    contract_version: int

    def drag(
        self, scene: Any, brep: Any, handle: str, index: int, target: Any, **options: Any
    ) -> DragOutcome:
        """Solve the drag of ``handle`` (``"vertex"`` / ``"edge"``) ``index`` to ``target``."""
        ...


#: kind -> (Protocol, the method a plugin's ``apply``/``as_jax`` bind to).
KIND_CONTRACTS: dict[str, tuple[type, str]] = {
    PluginKind.NODE_MAP.value: (NodeMap, "positions"),
    PluginKind.FEATURE_EDGES.value: (FeatureEdges, "feature_edges"),
    PluginKind.BREP.value: (BRepExtractor, "extract"),
    PluginKind.STEP_EXPORT.value: (StepExporter, "step_export"),
    PluginKind.DRAG.value: (Drag, "drag"),
}


def contract_for(kind: str) -> type | None:
    """The Protocol a kind is bound to, or ``None`` for a kind without one."""
    entry = KIND_CONTRACTS.get(str(kind))
    return entry[0] if entry else None


def primary_method(kind: str) -> str | None:
    """The method name a python plugin of ``kind`` exposes through ``apply``."""
    entry = KIND_CONTRACTS.get(str(kind))
    return entry[1] if entry else None


def _is_differentiable(annotation: Any) -> bool:
    return (
        typing.get_origin(annotation) is Annotated
        and DIFFERENTIABLE in typing.get_args(annotation)[1:]
    )


def payload_types(kind: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """The declared inputs and output of a kind's primary method.

    Read off the Protocol's annotations, ``Differentiable`` tags included,
    so a python-transport plugin can publish ``inputs``/``outputs`` and
    ``capabilities`` the way a Tesseract does from its schema.

    Returns:
        ``(inputs, output)`` — ``{name: {"annotation": str,
        "differentiable": bool}}`` and the same for the single result.
    """
    contract = contract_for(kind)
    method = primary_method(kind)
    if contract is None or method is None:
        return {}, {}
    function = getattr(contract, method)
    hints = typing.get_type_hints(function, include_extras=True)
    signature = inspect.signature(function)
    inputs = {}
    for name in signature.parameters:
        if name == "self":
            continue
        annotation = hints.get(name, Any)
        inputs[name] = {
            "annotation": _spell(annotation),
            "differentiable": _is_differentiable(annotation),
        }
    result = hints.get("return", Any)
    return inputs, {"annotation": _spell(result), "differentiable": _is_differentiable(result)}


def _spell(annotation: Any) -> str:
    if typing.get_origin(annotation) is Annotated:
        annotation = typing.get_args(annotation)[0]
    return getattr(annotation, "__name__", None) or str(annotation)


def contract_signature(kind: str) -> str:
    """``sha256:`` of a kind's Protocol signature — the python-transport ``schema_hash``.

    Covers the contract version, the method name, its parameters and their
    annotations, so a probe can tell that the interface a component was
    built against is the one this cadjoint speaks.
    """
    inputs, output = payload_types(kind)
    text = repr((CONTRACT_VERSION, str(kind), primary_method(kind), sorted(inputs.items()), output))
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
