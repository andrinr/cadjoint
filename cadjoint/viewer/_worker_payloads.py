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

    Must mirror the compaction of :func:`_render_surface_payload`: hex
    quads are gathered group by group (sorted by group id) exactly like
    :func:`cadjoint.fem.render_payload.boundary_render_payload`, tet
    boundary triangles as-is, and node ids deduplicated with
    ``np.unique`` — so position *i* of the render payload's vertex arrays
    corresponds to mesh node ``result[i]``.
    """
    if hasattr(mesh, "boundary_faces"):
        faces = np.concatenate(
            [mesh.boundary_faces[group_id].nodes for group_id in sorted(mesh.boundary_faces)],
            axis=0,
        )
    else:
        faces = np.asarray(mesh.boundary_tris)
    return np.unique(faces.reshape(-1))


def _render_surface_payload(mesh: Any, node_scalar: np.ndarray) -> dict[str, Any]:
    """The viewer's boundary-surface payload for any solve mesh.

    Hex meshes go through the canonical
    :func:`cadjoint.fem.render_payload.boundary_render_payload`; tet meshes
    get the same contract built here from their outward corner triangles —
    identical keys (``positions``/``scalars``/``indices``/``groups``/
    ``range``/``vertex_count``) with one synthetic ``"surface"`` group, so
    the frontend renders both without knowing the meshing method.
    """
    from cadjoint.fem.render_payload import boundary_render_payload

    if hasattr(mesh, "boundary_faces"):
        return boundary_render_payload(mesh, node_scalar)

    scalar = np.asarray(node_scalar, dtype=np.float64).reshape(-1)
    if scalar.shape[0] != mesh.num_points:
        raise ValueError(
            f"Expected one scalar per node ({mesh.num_points}), got {scalar.shape[0]}."
        )
    tris = np.asarray(mesh.boundary_tris)
    used, remapped = np.unique(tris.reshape(-1), return_inverse=True)
    triangles = remapped.reshape(-1, 3).astype(np.int64)
    positions = np.asarray(mesh.points)[used]
    scalars = scalar[used]
    finite = scalars[np.isfinite(scalars)]
    low = float(finite.min()) if finite.size else 0.0
    high = float(finite.max()) if finite.size else 0.0
    corners = positions[triangles]
    normals = 0.5 * np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    areas = np.linalg.norm(normals, axis=-1)
    total = float(areas.sum())
    weights = areas / max(total, 1e-30)
    center = (corners.mean(axis=1) * weights[:, None]).sum(axis=0)
    groups = [
        {
            "id": "surface",
            "axis": None,
            "side": None,
            "center": [round(float(value), 5) for value in center],
            "area": round(total, 6),
            "faces": int(triangles.shape[0]),
            "start": 0,
            "count": int(triangles.size),
        }
    ]
    return {
        "positions": [round(float(value), 5) for value in positions.reshape(-1)],
        "scalars": [round(float(value), 6) for value in scalars],
        "indices": [int(value) for value in triangles.reshape(-1)],
        "groups": groups,
        "range": [round(low, 6), round(high, 6)],
        "vertex_count": int(used.shape[0]),
    }


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
    if hasattr(mesh, "boundary_faces"):
        faces = np.concatenate(
            [mesh.boundary_faces[group_id].nodes for group_id in sorted(mesh.boundary_faces)],
            axis=0,
        )
        corners = ((0, 1), (1, 2), (2, 3), (3, 0))
    else:  # tet meshes: outward corner triangles
        faces = np.asarray(mesh.boundary_tris)
        corners = ((0, 1), (1, 2), (2, 0))
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
    described["bcs"] = [
        {**bc.describe(), "serializable": bc.nodes.serializable} for bc in study.bcs
    ]
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
