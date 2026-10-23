"""The B-rep inspector: what the derived B-rep found, as data the viewer draws.

The feature-edge overlay (:mod:`cadjoint.viewer._edge_overlay`) shows the
*result* of the private tier's extraction — the curves it decided to draw.
When one of those is wrong, the overlay cannot say why: it does not show
the faces the curve was supposed to separate, the boundary chains it
dropped, or the corners it solved.  The inspector does.  It asks the
``brep`` kind for the derived B-rep on the overlay's own grid and serves
the whole of it:

* every **face** with its own triangles of the extraction's tessellation,
  its surface type and its owning patch;
* every **edge** — every face-pair boundary chain the graph built, not only
  the drawn ones — with a ``kind`` saying how it reached the screen:
  ``traced`` / ``sampled`` when the ``feature_edges`` kind drew it (the
  overlay's curves, matched back onto the graph), otherwise the graph's own
  state (``blend``, ``tangent``, ``coincident``, ``contact``, …, or
  ``undrawn`` for a crossing the overlay dropped).  The kind is a string
  passed through as the private tier spells it, so a new edge state shows
  up in the viewer without a change here;
* every **vertex** with its solved position;
* per face, when the ``step_export`` kind is installed, how the analytic
  STEP export writes it — its exact surface, one fitted B-spline with the
  error it was certified to, or its own triangles and why — read off
  ``brep.step_plan(scene)``, with the file's summary in the diagnostics;
* a **diagnostics** block: counts, which path produced the B-rep (exact or
  the dual-contouring fallback, and the reason the exact one refused), and
  the per-edge ``off_surface`` distance — the largest ``|f|`` of the
  *scene* along the curve, which is how a phantom edge under a smooth blend
  gives itself away: it sits on the intersection of two patches the blend
  has rounded off, a band-width away from the solid.

**The seam.**  The B-rep object is opaque by contract
(:class:`cadjoint.plugins.contracts.BRepExtractor`): it never leaves the
process and nothing public builds one.  The inspector is the one public
reader of it, and it reads only the attributes the private tier documents
on its ``BRep`` — ``faces``, ``edges``, ``vertices``, ``points``,
``triangles`` / ``triangle_face`` (or ``mesh.quads`` / ``quad_face``),
``patches``, ``decomposition`` and ``stats``, and the methods
``corner_normals`` and ``step_plan`` — each through ``getattr``
with a default, so a B-rep that lacks one degrades the panel rather than
the request.  ``import diff_brep`` is never written here.

Without the private tier the inspector reports ``available: False`` and
the reason, in the tier's own words, and the viewer keeps the mode off.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from cadjoint.enums import PluginKind
from cadjoint.viewer._edge_overlay import (
    _MESH_EDGE_RESOLUTION,
    _PROJECTION_STEPS,
    _overlay_grid,
)
from cadjoint.viewer.schema.payloads import StepContents

__all__ = ["brep_inspection"]

#: Analytic surface kinds the viewer names as they are.
_NAMED_SURFACES = ("plane", "cylinder", "sphere", "cone", "torus")

#: An edge whose scene residual exceeds this many grid cells is off the solid.
#: A drawn curve sits on the surface to projection precision (``1e-5`` cells
#: and below); a curve under a blend sits a band-width away, which on any
#: blend the grid can see is a large fraction of a cell.  A tenth separates
#: the two with two orders of magnitude to spare either side.
OFF_SURFACE_CELLS = 0.1

#: A drawn feature curve is matched to the graph edge whose polyline lies
#: within this many cells of it on average.
_MATCH_CELLS = 1.5

#: Digits kept on the wire for positions (model units).
_DIGITS = 5


def _round(values: np.ndarray, digits: int = _DIGITS) -> list[float]:
    """Flatten and round an array for JSON."""
    return [round(float(value), digits) for value in np.asarray(values, dtype=np.float64).ravel()]


def _finite(value: Any) -> float | None:
    """A float for JSON, or ``None`` where it is missing or not finite."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _surface_label(kind: str, analytic: bool) -> str:
    """The surface type the viewer prints: a named form, ``blend`` or ``faceted``.

    A face of an analytic kind whose closed form did not certify on its own
    samples is shown as ``faceted``: the tessellation is all there is of it.
    """
    if kind == "blend":
        return "blend"
    if analytic and kind in _NAMED_SURFACES:
        return kind
    return "faceted"


def _leaf_name(decomposition: Any, leaf: int) -> str | None:
    """The primitive type under a world-frame leaf, for the readout."""
    leaves = getattr(decomposition, "leaves", None)
    if leaves is None or not 0 <= leaf < len(leaves):
        return None
    node = leaves[leaf]
    # Walk the transform chain to the primitive: the class name a user
    # wrote (``Box``, ``Cylinder``) says more than ``Translate``.
    for _ in range(32):
        child = getattr(node, "sdf", None)
        if child is None or not hasattr(child, "patch_fields"):
            break
        node = child
    return type(node).__name__


def _tessellation(brep: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(points, triangles, triangle_face)`` of the B-rep's own tessellation.

    The analytic path tessellates into ``triangles``; the dual-contouring
    path also fills it, and where it does not, the lattice quads are split
    along their first diagonal.
    """
    points = np.asarray(getattr(brep, "points", np.zeros((0, 3))), dtype=np.float64).reshape(-1, 3)
    triangles = np.asarray(getattr(brep, "triangles", np.zeros((0, 3))), dtype=np.int64)
    owners = np.asarray(getattr(brep, "triangle_face", np.zeros(0)), dtype=np.int64)
    if triangles.size and owners.shape[0] == triangles.shape[0]:
        return points, triangles.reshape(-1, 3), owners
    mesh = getattr(brep, "mesh", None)
    quads = np.asarray(getattr(mesh, "quads", np.zeros((0, 4))), dtype=np.int64).reshape(-1, 4)
    quad_face = np.asarray(getattr(brep, "quad_face", np.zeros(0)), dtype=np.int64)
    if quads.shape[0] == 0 or quad_face.shape[0] != quads.shape[0]:
        return points, np.zeros((0, 3), dtype=np.int64), np.zeros(0, dtype=np.int64)
    split = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    return points, split, np.concatenate([quad_face, quad_face])


def _corner_normals(scene: Any, brep: Any, triangles: np.ndarray) -> np.ndarray | None:
    """The private tier's exact normal at every triangle corner, or ``None``.

    Asked of the B-rep itself (``brep.corner_normals(scene)``), which knows
    each face's surface: shading with the surface's own normal shows a smooth
    blend smooth, where the facets' normals show every facet.  Only the
    B-rep's own triangles have them — not the lattice quads split here — and
    a B-rep without the method, or one that fails, is shaded flat as before.
    """
    method = getattr(brep, "corner_normals", None)
    if method is None or triangles.shape[0] == 0:
        return None
    own = np.asarray(getattr(brep, "triangles", np.zeros((0, 3))), dtype=np.int64).reshape(-1, 3)
    if own.shape != triangles.shape or not np.array_equal(own, triangles):
        return None
    try:
        normals = np.asarray(method(scene), dtype=np.float64)
    except Exception as error:  # noqa: BLE001 - shading is a nicety, the inspector still answers
        print(f"corner normals unavailable: {type(error).__name__}: {error}")
        return None
    if normals.shape != (*triangles.shape, 3):
        return None
    return normals


def _scene_residual(scene: Any, polylines: Sequence[np.ndarray]) -> list[float]:
    """Per polyline, the largest ``|f|`` of the scene over its points.

    One program over every point of every curve.
    """
    import jax
    import jax.numpy as jnp

    sizes = [int(points.shape[0]) for points in polylines]
    if not sum(sizes):
        return [0.0] * len(polylines)
    stacked = np.concatenate([points for points in polylines if points.shape[0]])
    program = jax.jit(jax.vmap(lambda q: jnp.asarray(scene(q))))
    values = np.abs(np.asarray(program(jnp.asarray(stacked, dtype=jnp.float32)), np.float64))
    out: list[float] = []
    start = 0
    for size in sizes:
        out.append(float(values[start : start + size].max()) if size else 0.0)
        start += size
    return out


def _curve_distance(curve: np.ndarray, polyline: np.ndarray) -> float:
    """Mean distance from a drawn curve's points to a graph polyline's points."""
    if curve.shape[0] == 0 or polyline.shape[0] == 0:
        return float("inf")
    gaps = np.linalg.norm(curve[:, None, :] - polyline[None, :, :], axis=-1)
    return float(gaps.min(axis=1).mean())


def _match_curves(edges: Sequence[Any], curves: Any, cell: float) -> dict[int, int]:
    """Graph edge index -> drawn curve index, for the curves the overlay drew.

    A curve names the two patches it lies on; among the graph edges between
    the same two patches it belongs to the nearest one.  The overlay runs
    its own extraction (at its own blend tolerance), so matching is by
    geometry, never by index.
    """
    by_pair: dict[tuple[int, int], list[int]] = {}
    for position, edge in enumerate(edges):
        pair = tuple(sorted(int(p) for p in getattr(edge, "patches", (-1, -1))))
        by_pair.setdefault(pair, []).append(position)  # type: ignore[arg-type]
    matched: dict[int, int] = {}
    best: dict[int, float] = {}
    for index, curve in enumerate(curves.polylines):
        pair = tuple(sorted(int(p) for p in curves.patches[index]))
        candidates = by_pair.get(pair, [])  # type: ignore[arg-type]
        scored = [
            (_curve_distance(curve, np.asarray(edges[c].polyline, np.float64).reshape(-1, 3)), c)
            for c in candidates
        ]
        if not scored:
            continue
        distance, chosen = min(scored)
        if distance > _MATCH_CELLS * cell or distance >= best.get(chosen, float("inf")):
            continue
        matched[chosen] = index
        best[chosen] = distance
    return matched


def _step_plan(scene: Any, brep: Any) -> tuple[dict[str, Any] | None, str | None]:
    """The analytic STEP export's plan for this B-rep, or ``None`` and why.

    The plan is the ``step_export`` kind's: without it the export is the
    public tier's faceted dual contouring and there is no per-face plan to
    show.  It fits and certifies every blend face, so it is timed with the
    rest of the inspection.
    """
    from cadjoint import tier

    if tier.component(PluginKind.STEP_EXPORT) is None:
        return None, tier.message(PluginKind.STEP_EXPORT)
    method = getattr(brep, "step_plan", None)
    if method is None:
        return None, "this B-rep does not say how it exports"
    try:
        return dict(method(scene)), None
    except Exception as error:  # noqa: BLE001 - the B-rep is still worth showing
        return None, f"STEP plan failed: {_failure(error)}"


def _faces(brep: Any, plan: dict[str, Any] | None) -> list[dict[str, Any]]:
    exports = {int(record["face"]): record for record in (plan or {}).get("faces", [])}
    decomposition = getattr(brep, "decomposition", None)
    faces = []
    for position, face in enumerate(getattr(brep, "faces", [])):
        kind = str(getattr(face, "kind", "opaque"))
        analytic = bool(getattr(face, "analytic", False))
        leaf = int(getattr(face, "leaf", -1))
        faces.append(
            {
                "id": int(getattr(face, "index", position)),
                "kind": kind,
                "surface": _surface_label(kind, analytic),
                "analytic": analytic,
                "patch": int(getattr(face, "patch", -1)),
                "leaf": leaf,
                "leaf_type": _leaf_name(decomposition, leaf),
                "area": _finite(getattr(face, "area", None)),
                **({"export": _export_record(exports[position])} if position in exports else {}),
            }
        )
    return faces


def _export_record(record: dict[str, Any]) -> dict[str, Any]:
    """One face's plan entry as the payload carries it: the face index dropped, numbers finite."""
    return {
        "surface": str(record["surface"]),
        "fit_error": _finite(record.get("fit_error")),
        "field_error": _finite(record.get("field_error")),
        "control_points": int(record.get("control_points") or 0),
        "reason": str(record.get("reason") or ""),
    }


def _vertices(brep: Any) -> list[dict[str, Any]]:
    vertices = []
    for position, vertex in enumerate(getattr(brep, "vertices", [])):
        point = np.asarray(getattr(vertex, "point", np.zeros(3)), dtype=np.float64).reshape(3)
        vertices.append(
            {
                "id": int(getattr(vertex, "index", position)),
                "kind": str(getattr(vertex, "kind", "corner")),
                "point": _round(point),
                "faces": [int(f) for f in getattr(vertex, "faces", ())],
                "residual": _finite(getattr(vertex, "residual", None)),
            }
        )
    return vertices


def _edges(scene: Any, brep: Any, curves: Any, cell: float) -> list[dict[str, Any]]:
    """Every graph edge, kinded and measured, plus drawn curves no edge took."""
    graph = list(getattr(brep, "edges", []))
    matched = _match_curves(graph, curves, cell) if curves is not None else {}
    rows: list[dict[str, Any]] = []
    polylines: list[np.ndarray] = []
    for position, edge in enumerate(graph):
        state = str(getattr(edge, "state", "crossing"))
        curve = matched.get(position)
        if curve is not None:
            kind = str(curves.kind[curve])
            points = np.asarray(curves.polylines[curve], dtype=np.float64).reshape(-1, 3)
            closed = bool(curves.closed[curve])
        else:
            # A solid crossing the overlay did not draw is still worth seeing:
            # it is either debris it pruned or an edge it lost.
            kind = "undrawn" if state in ("crossing", "branch", "waterline") else state
            points = np.asarray(getattr(edge, "polyline", np.zeros((0, 3))), np.float64)
            points = points.reshape(-1, 3)
            closed = bool(getattr(edge, "closed", False))
        polylines.append(points)
        rows.append(
            {
                "id": int(getattr(edge, "index", position)),
                "kind": kind,
                "state": state,
                "faces": [int(f) for f in getattr(edge, "faces", (-1, -1))],
                "patches": [int(p) for p in getattr(edge, "patches", (-1, -1))],
                "vertices": [int(v) for v in getattr(edge, "vertices", (-1, -1))],
                "closed": closed,
                "analytic": bool(getattr(edge, "analytic", False)),
                "residual": _finite(getattr(edge, "residual", None)),
                "crossing": _finite(getattr(edge, "crossing", None)),
            }
        )
    taken = set(matched.values())
    if curves is not None:
        for index in range(curves.count):
            if index in taken:
                continue
            polylines.append(np.asarray(curves.polylines[index], np.float64).reshape(-1, 3))
            rows.append(
                {
                    "id": -1 - index,
                    "kind": str(curves.kind[index]),
                    "state": "unmatched",
                    "faces": [-1, -1],
                    "patches": [int(p) for p in curves.patches[index]],
                    "vertices": [int(v) for v in curves.vertices[index]],
                    "closed": bool(curves.closed[index]),
                    "analytic": True,
                    "residual": _finite(curves.residual[index]),
                    "crossing": None,
                }
            )
    off = _scene_residual(scene, polylines)
    for row, points, value in zip(rows, polylines, off):
        row["points"] = _round(points)
        row["off_surface"] = round(value, 7)
    return rows


def _diagnostics(
    brep: Any,
    faces: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    vertices: list[dict[str, Any]],
    curves_note: str | None,
    cell: float,
    timings: dict[str, float],
    plan: dict[str, Any] | None,
    plan_note: str | None,
) -> dict[str, Any]:
    stats = dict(getattr(brep, "stats", {}) or {})
    source = str(stats.get("source", "unknown"))
    tally = lambda rows, key: {  # noqa: E731 - a two-line counter
        value: sum(1 for row in rows if row[key] == value)
        for value in sorted({r[key] for r in rows})
    }
    tolerance = OFF_SURFACE_CELLS * cell
    scalars = {
        key: value
        for key, value in stats.items()
        if isinstance(value, (int, float, str, bool)) and key not in ("source", "ineligible_reason")
    }
    return {
        "faces": len(faces),
        "edges": len(edges),
        "vertices": len(vertices),
        "source": source,
        "exact": source == "analytic",
        "fallback_reason": stats.get("ineligible_reason"),
        "surfaces": tally(faces, "surface"),
        "edge_kinds": tally(edges, "kind"),
        "vertex_kinds": tally(vertices, "kind"),
        "off_surface_tolerance": tolerance,
        "off_surface_edges": sum(1 for row in edges if row["off_surface"] > tolerance),
        "feature_curves": curves_note,
        "cell": cell,
        "resolution": _MESH_EDGE_RESOLUTION,
        "stats": scalars,
        "timings": {key: round(value, 3) for key, value in timings.items()},
        "step": None if plan is None else StepContents.from_report(plan).model_dump(mode="json"),
        "step_note": plan_note,
    }


def _failure(error: BaseException) -> str:
    """One line for an extraction that raised, with the frame it raised in.

    The inspector exists to look at the private tier's work, so when that
    work fails the failure *is* the finding: the panel prints where.
    """
    import traceback

    frames = traceback.extract_tb(error.__traceback__)
    where = (
        f" (at {frames[-1].name}, {Path(frames[-1].filename).name}:{frames[-1].lineno})"
        if frames
        else ""
    )
    return f"{type(error).__name__}: {error}{where}"


def brep_inspection(scene: Any) -> dict[str, Any]:
    """The inspector payload for ``scene``: faces, edges, vertices, diagnostics.

    Returns:
        ``{"available": False, "reason": str}`` without the private tier;
        ``{"available": True, "error": str}`` when the extraction raised;
        otherwise ``{"available": True, "positions", "triangles",
        "triangle_face", "faces", "edges", "vertices", "diagnostics"}`` —
        ``positions`` a flat ``x, y, z`` list, ``triangles`` flat indices
        into it, ``triangle_face`` the face id of each triangle.

    Raises:
        Whatever the extraction raises; the worker turns it into an error
        response the viewer shows.
    """
    from cadjoint import tier

    component = tier.component(PluginKind.BREP)
    if component is None:
        return {"available": False, "reason": tier.message(PluginKind.BREP)}
    grid = _overlay_grid()
    cell = float(np.max(np.asarray(grid.spacing, dtype=np.float64)))
    timings: dict[str, float] = {}
    with warnings.catch_warnings():
        # The overlay's grid is a fixed view volume a scene may leave; see
        # `_edge_overlay._graph_layers`.
        warnings.filterwarnings("ignore", message="The isosurface crosses the extraction boundary")
        started = time.perf_counter()
        try:
            brep = component.extract(scene, grid)
        except Exception as error:  # noqa: BLE001 - the inspector reports it, the viewer shows it
            return {"available": True, "error": _failure(error)}
        timings["extract"] = time.perf_counter() - started
        curves = None
        curves_note: str | None = None
        drawer = tier.component(PluginKind.FEATURE_EDGES)
        if drawer is None:
            curves_note = "feature_edges kind not installed: edges are the graph's seeds"
        else:
            started = time.perf_counter()
            try:
                curves = drawer.feature_edges(scene, grid, steps=_PROJECTION_STEPS)
            except Exception as error:  # noqa: BLE001 - the graph is still worth showing
                curves_note = f"feature curves failed: {type(error).__name__}: {error}"
            timings["feature_edges"] = time.perf_counter() - started
    started = time.perf_counter()
    points, triangles, owners = _tessellation(brep)
    normals = _corner_normals(scene, brep, triangles)
    timings["inspect"] = time.perf_counter() - started
    started = time.perf_counter()
    plan, plan_note = _step_plan(scene, brep)
    timings["step_plan"] = time.perf_counter() - started
    started = time.perf_counter()
    faces = _faces(brep, plan)
    vertices = _vertices(brep)
    edges = _edges(scene, brep, curves, cell)
    timings["inspect"] += time.perf_counter() - started
    return {
        "available": True,
        "positions": _round(points),
        "triangles": [int(i) for i in triangles.ravel()],
        "triangle_face": [int(i) for i in owners],
        **({"corner_normals": _round(normals.reshape(-1, 3))} if normals is not None else {}),
        "faces": faces,
        "edges": edges,
        "vertices": vertices,
        "diagnostics": _diagnostics(
            brep, faces, edges, vertices, curves_note, cell, timings, plan, plan_note
        ),
    }
