"""The worker's two FEM-facing modes: solve a study, or inspect a mesh.

``mode="simulate"`` runs one study the program declares and returns the
solved field with its render surface; ``mode="mesh_inspect"`` builds one
declared (or study-implicit) ``SimMesh`` without solving and returns its
inspection report plus a quality heatmap.  Both execute the program
through :mod:`cadjoint.viewer._worker_scene` and shape their response
through :mod:`cadjoint.viewer._worker_payloads`.
"""

from __future__ import annotations

import contextlib
import io
from typing import Any

import numpy as np

from cadjoint.viewer._worker_payloads import (
    _element_edge_pairs,
    _render_surface_payload,
    _study_payload,
)
from cadjoint.viewer._worker_scene import (
    _FEM_UNAVAILABLE_MESSAGE,
    _execute_scene,
    _named_study,
)


def _simulate_study(scene: Any, studies: list[Any], request: dict[str, Any]) -> dict[str, Any]:
    """Solve one study the scene program declared, by name.

    The study is the source of truth: mesh, material, and boundary
    conditions all come from its declaration — the request only picks which
    one to run.  Non-serializable (predicate) selections solve fine here
    since the declared objects are used directly.

    With ``cached=True`` the study's ``last_result`` is served without
    re-solving when it exists.  The cache lives on the study object, so over
    the HTTP API — where every request runs a fresh worker process — it only
    ever hits when the scene program itself called ``solve()`` while it
    executed; it is a per-worker-process cache, not a server-side one.
    """
    import jax.numpy as jnp

    study = _named_study(studies, request.get("name"))

    try:
        import jax_fem  # noqa: F401
    except ImportError:
        return {"ok": False, "error_kind": "fem_unavailable", "error": _FEM_UNAVAILABLE_MESSAGE}

    sdf = lambda p: jnp.asarray(scene(p))  # noqa: E731
    cached = bool(request.get("cached")) and study.last_result is not None
    result = study.last_result if cached else study.solve(sdf)
    return {
        "ok": True,
        "kind": "study",
        **_study_payload(study, result, sdf),
        "cached": cached,
    }


def _simulate_source(request: dict[str, Any]) -> dict[str, Any]:
    """Run the FEM simulation mode: exec scene -> declared study -> payload."""
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        namespace = _execute_scene(request["source"])
        result = _simulate_study(namespace["scene"], namespace["__studies__"], request)
    result["output"] = captured.getvalue()[-8_000:]
    return result


def _inspect_mesh(
    scene: Any, sim_meshes: list[Any], studies: list[Any], request: dict[str, Any]
) -> dict[str, Any]:
    """Build one declared (or study-implicit) SimMesh and describe it.

    Resolution of the ``name`` request field:

    * a declared ``SimMesh`` name wins;
    * otherwise a declared study's name selects that study's mesh (its
      ``mesh=`` SimMesh, or the anonymous mesh implied by its
      resolution/bounds/size/domain);
    * with no name at all, a single declared mesh — or, failing that, a
      single declared study — is used.

    Returns the JSON inspection summary plus a renderable boundary surface
    whose scalars are the per-vertex scaled-jacobian quality field (each
    element's quality mapped to its 8 corners, min-combined), so the
    viewer can show a quality heatmap before anything is solved.
    """
    import jax.numpy as jnp

    from cadjoint.fem.study import _solve_mesh

    sdf = lambda p: jnp.asarray(scene(p))  # noqa: E731
    name = request.get("name")

    def implicit(study: Any) -> Any:
        if study.mesh is not None:
            return study.mesh
        sim_mesh, _ = _solve_mesh(study, sdf, None)
        return sim_mesh

    declared = ", ".join(repr(mesh.name) for mesh in sim_meshes) or "none"
    if name is not None:
        matches = [mesh for mesh in sim_meshes if mesh.name == name]
        if len(matches) > 1:
            raise ValueError(f"The program declares more than one mesh named {name!r}.")
        if matches:
            target = matches[0]
        else:
            study_matches = [study for study in studies if study.name == name]
            if len(study_matches) > 1:
                raise ValueError(f"The program declares more than one study named {name!r}.")
            if not study_matches:
                studies_declared = ", ".join(repr(study.name) for study in studies) or "none"
                raise ValueError(
                    f"No declared mesh or study named {name!r} "
                    f"(meshes: {declared}; studies: {studies_declared})."
                )
            target = implicit(study_matches[0])
    elif len(sim_meshes) == 1:
        target = sim_meshes[0]
    elif not sim_meshes and len(studies) == 1:
        target = implicit(studies[0])
    else:
        raise ValueError(
            f"Pass `name` to pick a mesh: the program declares meshes {declared} "
            f"and {len(studies)} studies."
        )

    mesh = target.build(sdf)
    # Method-agnostic quality heatmap: scaled jacobians for hex meshes, the
    # radius ratio for tet meshes — each element's quality mapped onto its
    # nodes, min-combined.
    metrics = target.quality(sdf)
    metric_name = "scaled_jacobian" if "scaled_jacobian" in metrics else "radius_ratio"
    quality = np.asarray(metrics[metric_name], dtype=np.float64)
    cells = np.asarray(mesh.cells)
    node_quality = np.full(mesh.num_points, np.inf, dtype=np.float64)
    np.minimum.at(node_quality, cells.reshape(-1), np.repeat(quality, cells.shape[1]))
    node_quality = np.where(np.isfinite(node_quality), node_quality, 1.0)
    payload = _render_surface_payload(mesh, node_quality)
    payload["edges"] = [int(index) for index in _element_edge_pairs(mesh).reshape(-1)]
    return {
        "ok": True,
        "kind": "mesh_inspect",
        "name": target.name,
        "field": metric_name,
        "info": target.inspect(sdf),
        "mesh": payload,
        "quality_scalars": payload["scalars"],
    }


def _mesh_inspect_source(request: dict[str, Any]) -> dict[str, Any]:
    """Run the mesh-inspection mode: exec scene -> build SimMesh -> payload."""
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        namespace = _execute_scene(request["source"])
        result = _inspect_mesh(
            namespace["scene"], namespace["__sim_meshes__"], namespace["__studies__"], request
        )
    result["output"] = captured.getvalue()[-8_000:]
    return result
