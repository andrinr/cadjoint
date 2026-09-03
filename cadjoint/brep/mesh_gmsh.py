"""The node map: design parameters to the positions of a Gmsh mesh.

The public Gmsh route (:mod:`cadjoint.fem.gmsh`) decides a topology once
and tags every node with the patches that own it — the
:class:`~cadjoint.plugins.OwnedNodes` record.  What this module owns is the
other half: **where the nodes go when the design changes**.  From the
record comes the arity, and from the arity the projection:

- a **surface** node solves ``f_a = 0`` — one field,
- a **curve** node solves ``f_a = f_b = 0`` — the two faces it separates,
- a **point** node solves ``f_a = f_b = f_c = 0``,
- a **blend** node lies on no patch at all and solves ``scene(x) = 0``,
- a **volume** node is mesh gauge and follows the interior Laplacian.

Every one of those is :func:`cadjoint.brep.project.project` at a different
arity, so the whole node set differentiates in the design parameters through
the same implicit-function adjoint the graph uses — *including the midside
nodes*, which is the point: re-solving a midside against its own patch is
what keeps it on the cylinder when the bore radius changes, instead of
drifting to the straight-sided midpoint the DC path is stuck with.

:func:`node_positions` is the ``node_map`` plugin kind
(:class:`cadjoint.plugins.contracts.NodeMap`), registered from
:mod:`cadjoint.brep.plugins`; :func:`gmsh_tet_mesh` is the graph-fed
convenience that hands the public mesher the *analytic* STEP instead of the
dual-contour STL, so midsides land on the true cylinder to 1e-6 rather
than on a chord.

**Topology is frozen, positions are not.**  Nothing here re-runs Gmsh; a
design step recomputes positions over the topology the record carries.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from cadjoint.brep.graph import BRep
from cadjoint.brep.project import project_batched, project_fields
from cadjoint.brep.step import save_brep_step
from cadjoint.fem.gmsh import (
    HXT,
    TET_MESHER_KIND,
    GmshMesh,
    assign_ownership,
    gmsh_available,
    gmsh_topology,
    gmsh_version,
    tet_mesh_from_gmsh,
)
from cadjoint.fem.gmsh import gmsh_tet_mesh as _gmsh_tet_mesh
from cadjoint.plugins.contracts import OwnedNodes

__all__ = [
    "HXT",
    "TET_MESHER_KIND",
    "GmshMesh",
    "assign_ownership",
    "gmsh_available",
    "gmsh_tet_mesh",
    "gmsh_topology",
    "gmsh_version",
    "node_positions",
    "parameterised_points",
    "recompute_gmsh_points",
    "tet_mesh_from_gmsh",
]


def gmsh_tet_mesh(
    brep: BRep,
    *,
    target_size: float | None = None,
    order: int = 2,
    algorithm: int = HXT,
    optimize: bool = True,
    step_path: str | Path | None = None,
    blend_tolerance: float | None = None,
    plugin: str | None = None,
    verbose: bool = False,
) -> GmshMesh:
    """Mesh the graph's exact STEP with Gmsh and give every node its owner.

    The same mesher and the same residual tagging as the public
    :func:`cadjoint.fem.gmsh.gmsh_tet_mesh`, fed the analytic STEP
    :func:`~cadjoint.brep.step.save_brep_step` writes instead of the
    dual-contour STL — the private tier's improvement of the input, not a
    different contract.  No snap: an exact surface's nodes arrive on it.

    Args:
        brep: The extracted graph.
        target_size: Uniform element size in model units; defaults to the
            graph's own smallest grid spacing.
        order: 1 for TET4, 2 for TET10.
        algorithm: Gmsh's ``Mesh.Algorithm3D``; :data:`HXT` by default.
        optimize: Run Gmsh's tet optimizer after generation.
        step_path: Where to keep the intermediate STEP; discarded when
            omitted.
        blend_tolerance: See :func:`cadjoint.fem.gmsh.gmsh_tet_mesh`.
        plugin: Run the mesher through this registered ``tet_mesher``
            plugin instead of importing Gmsh in this process.
        verbose: Let Gmsh write to the terminal.

    Returns:
        The :class:`~cadjoint.fem.gmsh.GmshMesh`; ``stats`` additionally
            carries ``step_seconds`` and the STEP face plan.
    """
    started = time.perf_counter()
    if step_path is None:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="cadjoint-gmsh-") as scratch:
            path = Path(scratch) / "brep.step"
            step_report = save_brep_step(brep, path)
            step_text = path.read_text(encoding="ascii")
    else:
        step_report = save_brep_step(brep, step_path)
        step_text = Path(step_path).read_text(encoding="ascii")
    step_seconds = time.perf_counter() - started

    mesh = _gmsh_tet_mesh(
        step_text,
        None,
        grid=brep.grid,
        geometry_format="step",
        target_size=target_size,
        order=order,
        algorithm=algorithm,
        optimize=optimize,
        blend_tolerance=blend_tolerance,
        plugin=plugin,
        fields=[patch.field for patch in brep.patches],
        snap=False,
        expected_bounds=np.stack([brep.points.min(axis=0), brep.points.max(axis=0)]),
        verbose=verbose,
    )
    mesh.stats["step_seconds"] = step_seconds
    mesh.stats["step_faces"] = step_report.get("faces", {})
    return mesh


# ── the concrete forward ─────────────────────────────────────────────────────


def recompute_gmsh_points(
    brep: BRep,
    mesh: GmshMesh,
    *,
    scene: Any = None,
    smooth_passes: int = 0,
    max_step: float | None = None,
) -> np.ndarray:
    """Re-solve every node against its own owners; the concrete forward.

    The whole node set is replaced by the solution of its own system: one
    field on a surface, two on a curve, three at a point, the scene's own
    field on a blend.  Midside nodes are solved the same way as corners,
    which is what keeps a midside on the cylinder rather than at the
    straight-sided midpoint.

    Args:
        brep: The graph whose patch table supplies the fields.
        mesh: The frozen Gmsh topology.
        scene: Root SDF node, needed only when the mesh has blend nodes.
        smooth_passes: Laplacian passes carrying boundary motion into the
            volume nodes; 0 leaves them where Gmsh put them.
        max_step: Projection clamp; defaults to the mesh's own.

    Returns:
        Node positions shaped like :attr:`GmshMesh.points`.

    Raises:
        ValueError: If the mesh has blend nodes but no ``scene`` was given.
    """
    if max_step is None:
        max_step = mesh.max_step
    owned = mesh.owned
    field_table = [patch.field for patch in brep.patches]
    solved = mesh.points.copy()
    by_arity: dict[int, list[int]] = {}
    for row in range(solved.shape[0]):
        arity = int(owned.arity[row])
        if arity:
            by_arity.setdefault(arity, []).append(row)
    for _arity, rows in sorted(by_arity.items()):
        index = np.asarray(rows, dtype=np.int64)
        members = owned.patches[index, : int(owned.arity[index[0]])].astype(np.int32)
        solved[index] = project_batched(field_table, members, solved[index], max_step=max_step)
    if owned.blend.any():
        if scene is None:
            raise ValueError(
                f"{int(owned.blend.sum())} nodes lie on blend faces, which no patch owns; "
                "pass scene= so they can be solved against the scene's own zero set."
            )
        index = np.flatnonzero(owned.blend)
        solved[index] = project_fields([scene], solved[index], max_step=max_step)
    if smooth_passes > 0:
        solved = _smoothed(mesh.cells, mesh.entity_dim < 3, solved, smooth_passes)
    return solved


def _smoothed(cells: np.ndarray, moving: np.ndarray, solved: np.ndarray, passes: int) -> np.ndarray:
    """Carry boundary motion into the volume nodes by Laplacian passes."""
    neighbours = _node_adjacency(cells)
    positions = solved.copy()
    for _ in range(passes):
        averaged = np.zeros_like(positions)
        counts = np.zeros(positions.shape[0])
        np.add.at(averaged, neighbours[:, 0], positions[neighbours[:, 1]])
        np.add.at(counts, neighbours[:, 0], 1.0)
        interior = ~moving & (counts > 0)
        positions[interior] = averaged[interior] / counts[interior, None]
    return positions


def _node_adjacency(cells: np.ndarray) -> np.ndarray:
    """Directed node-pair list from the cell connectivity, both ways."""
    cells = np.asarray(cells, dtype=np.int64)
    width = cells.shape[1]
    left = np.repeat(cells, width, axis=1).reshape(-1)
    right = np.tile(cells, (1, width)).reshape(-1)
    keep = left != right
    return np.stack([left[keep], right[keep]], axis=1)


# ── the traced forward: the node map ─────────────────────────────────────────


def _default_max_step(owned: OwnedNodes) -> float:
    """Half the median corner-edge length — a clamp the record itself sets."""
    from cadjoint.fem.elements import TET10_EDGES

    corners = np.asarray(owned.cells, dtype=np.int64)[:, :4]
    pairs = corners[:, TET10_EDGES].reshape(-1, 2)
    lengths = np.linalg.norm(owned.seeds[pairs[:, 1]] - owned.seeds[pairs[:, 0]], axis=1)
    return 0.5 * float(np.median(lengths)) if lengths.size else 1.0


def _scene_field_fn(scene: Any):
    """``field_fn(params, point) -> (1,)`` for the scene itself, under traced params."""
    import jax.numpy as jnp

    from cadjoint.extraction import extract_parameters
    from cadjoint.functionalize import functionalize

    compiled = functionalize(scene)
    baseline, fixed, _metadata = extract_parameters(scene)

    def field_fn(params: Mapping[str, Any], point: Any):
        merged = {**baseline, **dict(params or {})}
        return jnp.asarray(compiled(merged, fixed)(point)).reshape(1)

    return field_fn


def node_positions(
    scene: Any,
    params: Mapping[str, Any],
    owned: OwnedNodes,
    *,
    smooth_passes: int = 0,
    max_step: float | None = None,
    steps: int = 8,
):
    """The ``node_map`` kind: every node's position at ``params``, traced.

    One :func:`~cadjoint.brep.project.project` call per *distinct owner
    set* rather than per arity: the traced field builder
    (:func:`~cadjoint.brep.drag.patch_field_fn`) is a Python closure over a
    fixed patch list, so it cannot gather per point the way the concrete
    :func:`~cadjoint.brep.project.project_batched` does.  A hard part has a
    few dozen distinct sets, which is a few dozen calls — fine for a
    gradient, and the reason the concrete forward keeps its own fast path.

    Contract: see :class:`cadjoint.plugins.contracts.NodeMap`.

    Args:
        scene: Root SDF node the mesh was built from.
        params: Free-parameter mapping (partial mappings merge over the
            scene's current values).
        owned: The public ownership record.
        smooth_passes: Laplacian passes carrying the boundary displacement
            into the volume nodes.
        max_step: Projection clamp; half the median corner edge by default.
        steps: Newton iterations.

    Returns:
        A traced ``(P, 3)`` array of node positions, differentiable in
            ``params`` through the projection's implicit-function adjoint.
            Volume nodes carry a derivative only through the follow.

    Raises:
        ValueError: If a patch index in the record is outside the scene's
            table.
    """
    import jax.numpy as jnp

    from cadjoint.brep.drag import patch_field_fn
    from cadjoint.brep.project import project
    from cadjoint.fem.gmsh import patch_table

    table_size = len(patch_table(scene))
    if owned.patches.size and int(owned.patches.max()) >= table_size:
        raise ValueError(
            f"OwnedNodes names patch {int(owned.patches.max())} but the scene's table has "
            f"{table_size} patches; the record belongs to another scene."
        )
    if max_step is None:
        max_step = _default_max_step(owned)

    # Left at the default dtype rather than forced to float32 the way the
    # forward-only project_fields does: a finite-difference check of a
    # volume against a radius needs the x64 the FEM suite already enables.
    seeds = jnp.asarray(owned.seeds)
    positions = seeds
    groups: dict[tuple[int, ...], list[int]] = {}
    for row in np.flatnonzero(owned.arity > 0):
        arity = int(owned.arity[row])
        key = tuple(int(v) for v in owned.patches[row, :arity])
        groups.setdefault(key, []).append(int(row))
    for owners, rows in groups.items():
        index = jnp.asarray(rows, dtype=jnp.int32)
        solved = project(
            patch_field_fn(scene, owners), params, seeds[index], max_step=max_step, steps=steps
        )
        positions = positions.at[index].set(solved)
    blend_rows = np.flatnonzero(owned.blend)
    if blend_rows.size:
        index = jnp.asarray(blend_rows, dtype=jnp.int32)
        solved = project(
            _scene_field_fn(scene), params, seeds[index], max_step=max_step, steps=steps
        )
        positions = positions.at[index].set(solved)
    if smooth_passes > 0:
        positions = _followed(owned, seeds, positions, smooth_passes)
    return positions


def _followed(owned: OwnedNodes, seeds, positions, passes: int):
    """The interior Laplacian follow, in JAX: volume nodes chase the boundary."""
    import jax.numpy as jnp

    neighbours = _node_adjacency(owned.cells)
    total = owned.count
    counts = np.bincount(neighbours[:, 0], minlength=total).astype(np.float64)
    interior = np.asarray(owned.entity_dim >= 3) & (counts > 0)
    source = jnp.asarray(neighbours[:, 0])
    target = jnp.asarray(neighbours[:, 1])
    scale = jnp.asarray(np.where(counts > 0, 1.0 / np.maximum(counts, 1.0), 0.0))
    mask = jnp.asarray(interior)[:, None]
    delta = positions - seeds
    for _ in range(passes):
        averaged = jnp.zeros_like(delta).at[source].add(delta[target]) * scale[:, None]
        delta = jnp.where(mask, averaged, delta)
    return seeds + delta


def parameterised_points(
    scene: Any,
    mesh: GmshMesh,
    params: dict,
    *,
    max_step: float | None = None,
    steps: int = 8,
):
    """:func:`node_positions` over a :class:`GmshMesh` — the same solve, traced.

    Args:
        scene: Root SDF node.
        mesh: The frozen Gmsh topology.
        params: Free-parameter mapping.
        max_step: Projection clamp; defaults to the mesh's own.
        steps: Newton iterations.

    Returns:
        A traced ``(n, 3)`` array of node positions.
    """
    return node_positions(
        scene,
        params,
        mesh.owned,
        max_step=mesh.max_step if max_step is None else max_step,
        steps=steps,
    )
