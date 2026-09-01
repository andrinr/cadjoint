"""Differentiable node motion under frozen topology, hex and tet alike.

What belongs here: everything that moves the vertices of an *already
extracted* mesh as a differentiable function of a traced SDF — the shared
Newton projection onto the zero set (:func:`project_points`), the
per-family recomputation that rebuilds a frozen mesh's points for a new
design (:func:`recompute_points` on hexes, :func:`recompute_tet_points` on
tets), and the Laplacian operator that carries boundary displacement into a
frozen interior (:func:`smooth_interior_delta`).  This is the mesh half of
the design-parameter -> mesh -> FEM gradient chain; both element families
project through the *same* clamped Newton step so their gradients agree.

What does *not* belong here: mesh construction, connectivity, quality or
boundary bookkeeping.  Functions take meshes structurally (``base_points``
/ ``max_step`` plus the family's frozen-topology fields), so this module
sits *below* the mesh modules and they import from it.

Everything is pure JAX with a fixed iteration count — safe to trace,
differentiate and transpose.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from cadjoint.fem.elements import TET10_EDGES

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from cadjoint.fem.hexmesh import HexMesh
    from cadjoint.fem.tetmesh import TetMesh

__all__ = [
    "project_points",
    "recompute_points",
    "recompute_tet_points",
    "smooth_interior_delta",
]

# Below this squared gradient magnitude the field carries no usable
# direction and the Newton linearization is float noise.  A signed
# *distance* field has ``|grad| = 1`` almost everywhere, so ``|grad| < 1e-4``
# never means "a shallow field" — it means a dead subgradient: the crease of
# a smoothed union, or a flat wall a vertex has landed on bit-exactly (the
# same degeneracy ``cadjoint.meshing.edge_detection`` guards against).
_MIN_GRADIENT_SQUARED = 1e-8


def project_points(sdf: Callable[[Any], Any], points: Any, max_step: float, *, steps: int = 8):
    """Newton-project points onto the SDF zero set along the field gradient.

    Mirrors the seam projection used by the viewer's feature extraction but
    for a single field: each iteration moves ``x`` by
    ``-f(x) grad f / |grad f|^2``, and the total displacement from the
    starting point is clamped to ``max_step`` so callers can bound how far
    vertices wander (keeping interior elements well-shaped).

    Where the field's gradient underflows (:data:`_MIN_GRADIENT_SQUARED`)
    the linearization is meaningless, so the step is suppressed in both the
    forward and the backward pass — the point stays put and contributes no
    derivative, rather than moving on float noise.

    Pure JAX, fixed iteration count — safe to trace and differentiate with
    respect to SDF parameters.

    Args:
        sdf: Scalar field callable on ``(3,)`` points (batched via vmap).
        points: Starting positions, shaped ``(M, 3)``.
        max_step: Maximum total displacement per point.
        steps: Newton iterations.

    Returns:
        Projected positions as a JAX array, shaped ``(M, 3)``.
    """
    import jax
    import jax.numpy as jnp

    value_and_grad = jax.vmap(jax.value_and_grad(lambda p: jnp.asarray(sdf(p)).reshape(())))
    start = jnp.asarray(points)
    x = start
    for _ in range(steps):
        value, gradient = value_and_grad(x)
        squared = jnp.sum(gradient * gradient, axis=-1, keepdims=True)
        # Double-``where`` guard on the degenerate denominator, the repo's
        # idiom for this class of bug (cf. ``edge_detection._refine``).
        # Merely flooring ``squared`` keeps the forward step finite but
        # freezes the denominator as a *constant*, so the step's Jacobian
        # becomes ``value * Hessian / floor`` — a 1e12 amplification per
        # iteration that compounds over ``steps`` even though the forward
        # step is exactly zero.  Suppressing the step in BOTH passes leaves
        # value and derivative at zero together; the inner ``where`` keeps
        # the division itself finite so no NaN reaches the cotangent.
        usable = jax.lax.stop_gradient(squared) > _MIN_GRADIENT_SQUARED
        step = jnp.where(usable, value[:, None] * gradient / jnp.where(usable, squared, 1.0), 0.0)
        x = x - step
        displacement = x - start
        # Guarded norm: at zero displacement (vertex already on the surface)
        # a bare norm's gradient is 0/0, and the NaN would leak through
        # minimum() into otherwise-clean cotangents.
        squared_displacement = jnp.sum(displacement * displacement, axis=-1, keepdims=True)
        length = jnp.sqrt(jnp.maximum(squared_displacement, 1e-24))
        scale = jnp.minimum(1.0, max_step / length)
        x = start + displacement * scale
    return x


def recompute_points(sdf: Callable[[Any], Any], mesh: HexMesh):
    """Recompute hex-mesh vertex positions differentiably with frozen topology.

    Interior lattice vertices stay at their frozen positions; the vertices
    recorded in ``mesh.snap_mask`` are re-projected onto the (possibly
    traced) SDF's zero set with the same clamp used at extraction.  Because
    connectivity, the snapped-vertex set, and the base lattice are all
    frozen, the output is a differentiable function of the SDF's parameters
    — the mesh half of the design -> mesh -> FEM gradient chain.

    Args:
        sdf: Signed distance field, possibly closing over traced parameters.
        mesh: Mesh extracted at the nominal design by
            :func:`~cadjoint.fem.hexmesh.sdf_to_hex_mesh`.

    Returns:
        Vertex positions as a JAX array shaped like ``mesh.points``.
    """
    import jax.numpy as jnp

    base = jnp.asarray(mesh.base_points)
    indices = np.flatnonzero(mesh.snap_mask)
    if indices.size == 0:
        return base
    projected = project_points(sdf, base[indices], mesh.max_step)
    return base.at[indices].set(projected)


def _neighbor_lists(mesh: TetMesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unique undirected corner-vertex adjacency of the tet mesh, as flat pairs.

    Returns:
        ``(sources, targets, degrees)`` — for every directed adjacency
        pair ``sources[k] -> targets[k]``, plus per-corner-node degree.
    """
    edges = np.asarray(mesh.cells, dtype=np.int64)[:, :4][:, TET10_EDGES].reshape(-1, 2)
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    sources = np.concatenate([edges[:, 0], edges[:, 1]])
    targets = np.concatenate([edges[:, 1], edges[:, 0]])
    degrees = np.bincount(targets, minlength=mesh.num_corner_points)
    return sources, targets, degrees


def smooth_interior_delta(mesh: TetMesh, boundary_delta: Any, passes: int) -> Any:
    """Propagate boundary-vertex displacement into the frozen interior.

    ``passes`` Jacobi–Laplacian sweeps over the mesh's frozen corner
    adjacency, with the leading ``mesh.num_surface`` rows pinned to
    ``boundary_delta`` and the interior (Steiner) rows relaxed.  A fixed
    iteration count over frozen topology is one linear operator, so JAX
    differentiates and transposes it exactly; at zero boundary displacement
    it is the identity, which keeps a frozen mesh a fixed point of the map
    that uses it.

    Shared by the direct path (:func:`recompute_tet_points`) and the DC
    Tesseract chain, so both let the interior follow the boundary through
    the *same* operator.

    Args:
        mesh: The frozen mesh; its corner adjacency defines the operator.
        boundary_delta: Surface-vertex displacement from the frozen
            positions, ``(mesh.num_surface, 3)``.
        passes: Number of sweeps; ``<= 0`` holds the interior fixed.

    Returns:
        Corner-node displacement, ``(mesh.num_corner_points, 3)``.
    """
    import jax.numpy as jnp

    count = mesh.num_surface
    boundary_delta = jnp.asarray(boundary_delta)
    delta = (
        jnp.zeros((mesh.num_corner_points, 3), dtype=boundary_delta.dtype)
        .at[:count]
        .set(boundary_delta)
    )
    if passes <= 0:
        return delta
    sources, targets, degrees = _neighbor_lists(mesh)
    weights = 1.0 / jnp.maximum(jnp.asarray(degrees, dtype=delta.dtype), 1.0)
    for _ in range(passes):
        averaged = jnp.zeros_like(delta).at[targets].add(delta[sources]) * weights[:, None]
        delta = averaged.at[:count].set(boundary_delta)
    return delta


def recompute_tet_points(
    sdf: Callable[[Any], Any], mesh: TetMesh, *, smooth_passes: int = 0
) -> Any:
    """Recompute tet-mesh vertex positions differentiably with frozen topology.

    The leading ``num_surface`` boundary vertices are Newton re-projected
    from their frozen nominal positions onto the (possibly traced) SDF's
    zero set — the same clamped projection the hex mesher uses, so the
    output is a differentiable function of the SDF's parameters.  Interior
    Steiner vertices stay frozen by default; ``smooth_passes > 0`` runs
    that many differentiable Jacobi–Laplacian passes propagating the
    boundary *displacement* inward (boundary values pinned), which keeps
    interior elements better shaped under larger design motions.  On a
    TET10 mesh the midside nodes are recomputed as midpoints of the traced
    corner positions, so gradients flow through them too.

    Args:
        sdf: Signed distance field, possibly closing over traced parameters.
        mesh: Mesh extracted at the nominal design.
        smooth_passes: Number of interior displacement-smoothing passes.

    Returns:
        Vertex positions as a JAX array shaped like ``mesh.points``.
    """
    import jax.numpy as jnp

    corner_count = mesh.num_corner_points
    base = jnp.asarray(mesh.base_points[:corner_count])
    count = mesh.num_surface
    projected = project_points(sdf, base[:count], mesh.max_step)
    if smooth_passes <= 0:
        # Kept as a concatenation rather than base + a zero delta: the
        # boundary rows then pass through bit-for-bit, not to one ulp.
        corners = jnp.concatenate([projected, base[count:]], axis=0)
    else:
        corners = base + smooth_interior_delta(mesh, projected - base[:count], smooth_passes)
    if mesh.edge_parents is None:
        return corners
    midsides = corners[jnp.asarray(mesh.edge_parents)].mean(axis=1)
    return jnp.concatenate([corners, midsides], axis=0)
