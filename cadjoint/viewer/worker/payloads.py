"""Viewer payloads built from a solved or built FEM mesh.

The shared output shapes of the simulate, optimize, and mesh-inspect
stages: the renderable boundary surface (positions, scalars, indices,
face groups, element edges), the per-vertex field catalog of a solved
result, and :func:`_study_payload`, which packages one solved study the
single way both ``/api/simulate`` and the ``simulate`` block of a
study-backed ``/api/optimize`` response use.

Hex and tet meshes leave here in the same contract, so the frontend never
has to know which mesher ran.  Nothing in this module solves or meshes —
it only turns finished objects into JSON.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _boundary_vertex_nodes(mesh: Any) -> np.ndarray:
    """Node indices behind the compacted boundary vertex list.

    Mirrors the compaction of :func:`cadjoint.fem.render_payload.surface_render_payload`:
    faces gathered group by group in the surface's order and node ids
    deduplicated with ``np.unique`` — so position *i* of the render
    payload's vertex arrays corresponds to mesh node ``result[i]``.
    """
    return np.unique(mesh.surface().faces.reshape(-1))


def _render_surface_payload(mesh: Any, node_scalar: np.ndarray) -> dict[str, Any]:
    """The viewer's boundary-surface payload for any discretization."""
    from cadjoint.fem.render_payload import surface_render_payload

    return surface_render_payload(mesh.surface(), node_scalar)


def _element_edge_pairs(mesh: Any) -> np.ndarray:
    """Unique boundary-face element edges, in compacted-vertex indices.

    The viewer draws real element edges over the simulated surface; the
    triangulated render faces would show the quad-splitting diagonals, so
    the true face perimeters ship separately.  Each hex boundary quad
    contributes its 4 perimeter edges (tet boundary triangles their 3),
    deduplicated across shared faces.  Indices refer to the same compacted
    vertex list as the render payload's ``positions`` (faces gathered
    group by group, node ids deduplicated with ``np.unique`` — the
    :func:`_boundary_vertex_nodes` mapping).

    Returns:
        ``(E, 2)`` int64 edge pairs, each sorted, unique.
    """
    faces = np.asarray(mesh.surface().faces)
    corners = ((0, 1), (1, 2), (2, 3), (3, 0)) if faces.shape[1] == 4 else ((0, 1), (1, 2), (2, 0))
    _, remapped = np.unique(faces.reshape(-1), return_inverse=True)
    compact = remapped.reshape(faces.shape).astype(np.int64)
    edges = np.concatenate([compact[:, [a, b]] for a, b in corners], axis=0)
    return np.unique(np.sort(edges, axis=1), axis=0)


def _finite_range(values: np.ndarray) -> list[float]:
    """``[min, max]`` over the finite entries, like the render payload's range."""
    finite = values[np.isfinite(values)]
    low = float(finite.min()) if finite.size else 0.0
    high = float(finite.max()) if finite.size else 0.0
    return [round(low, 6), round(high, 6)]


def _result_field_payload(result: Any, payload: dict[str, Any]) -> None:
    """Attach the full per-vertex field catalog to a render payload.

    The base payload carries one display scalar per vertex; inspection
    wants every solved field.  Thermal results expose ``temperature``
    (identical to the display scalars, kept for a uniform shape); elastic
    results expose ``von_mises`` (the display scalars) plus
    ``displacement_magnitude``, and the raw per-vertex ``displacements``
    so the viewer can draw a warped surface.  ``ranges`` maps each field
    to its finite ``[lo, hi]``.  Mapping happens here, viewer-side, from
    the concrete SimulationResult arrays.
    """
    if result.kind == "thermal":
        payload["fields"] = {"temperature": list(payload["scalars"])}
    else:
        used = _boundary_vertex_nodes(result.mesh)
        displacement = np.asarray(result.solution.displacement, dtype=np.float64)[used]
        magnitude = np.linalg.norm(displacement, axis=-1)
        payload["fields"] = {
            "von_mises": list(payload["scalars"]),
            "displacement_magnitude": [round(float(value), 6) for value in magnitude],
        }
        payload["displacements"] = [
            [round(float(component), 6) for component in row] for row in displacement
        ]
    payload["ranges"] = {
        name: _finite_range(np.asarray(values, dtype=np.float64))
        for name, values in payload["fields"].items()
    }


def _study_payload(study: Any, result: Any, sdf: Any) -> dict[str, Any]:
    """Package one concrete study result for the viewer.

    The one packaging path for solved studies: ``/api/simulate`` responses
    and the ``simulate`` block of study-backed ``/api/optimize`` responses
    both go through here, so the frontend renders an optimized design with
    exactly the shapes a plain simulation carries — declaration (with
    per-BC serializability), display field, renderable surface (full field
    catalog, ranges, displacements), result summary, and the built mesh's
    inspection report.
    """
    scalar = np.asarray(result.nodal_scalar(), dtype=np.float64)
    described = study.describe()
    # Asked of the condition, not of a selection it is assumed to carry:
    # see :func:`cadjoint.viewer.worker.declarations._study_entries`.
    described["bcs"] = [{**bc.describe(), "serializable": bc.serializable} for bc in study.bcs]
    render_payload = _render_surface_payload(result.mesh, scalar)
    render_payload["edges"] = [int(index) for index in _element_edge_pairs(result.mesh).reshape(-1)]
    _result_field_payload(result, render_payload)
    return {
        "study": described,
        "field": result.field,
        "mesh": render_payload,
        "result": result.describe(),
        "mesh_info": result.sim_mesh.inspect(sdf) if result.sim_mesh is not None else None,
    }
