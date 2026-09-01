"""Derived quantities read off a solved field, uniform across element families.

What belongs here: anything computed *from* a solution rather than by a
solver — stress recovery (:func:`hex_von_mises`, :func:`tet_von_mises`) and
the boundary work integrals the optimizer's compliance objective sums
(:func:`load_work_quads` on hex faces, :func:`load_work_tris` /
:func:`load_work_tri6` on tet faces).  Both families' von Mises recovery
shares one stress evaluation (:func:`_von_mises_from_gradient`); only the
displacement-gradient reconstruction differs by element.

What does *not* belong here: solving (:mod:`cadjoint.fem.jaxfem`,
:mod:`cadjoint.fem.backends`), patch resolution
(:mod:`cadjoint.fem.simulate`) or meshing.  The stress functions are pure
NumPy; the load-work integrals are pure JAX and differentiable in both
``points`` and ``displacement``, so the load surface may move with the
design.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cadjoint.fem.elements import HEX_CORNER_SIGNS, TET10_EDGES

__all__ = [
    "hex_von_mises",
    "load_work_quads",
    "load_work_tri6",
    "load_work_tris",
    "tet_von_mises",
]


def _von_mises_from_gradient(u_grad: np.ndarray, youngs: float, poisson: float) -> np.ndarray:
    """Von Mises stress of per-cell displacement gradients ``(C, 3, 3)``.

    Small-strain isotropic linear elasticity: symmetrize, apply Hooke's
    law with the Lame constants of ``(youngs, poisson)``, and take
    ``sqrt(1.5 * s : s)`` of the deviator.
    """
    strain = 0.5 * (u_grad + u_grad.transpose(0, 2, 1))
    lame_lambda = youngs * poisson / ((1 + poisson) * (1 - 2 * poisson))
    lame_mu = youngs / (2 * (1 + poisson))
    trace = np.trace(strain, axis1=1, axis2=2)
    stress = lame_lambda * trace[:, None, None] * np.eye(3) + 2.0 * lame_mu * strain
    deviator = stress - np.trace(stress, axis1=1, axis2=2)[:, None, None] / 3.0 * np.eye(3)
    return np.sqrt(1.5 * np.einsum("cab,cab->c", deviator, deviator))


def hex_von_mises(
    points: np.ndarray,
    cells: np.ndarray,
    displacement: np.ndarray,
    *,
    youngs: float,
    poisson: float,
) -> np.ndarray:
    """Per-cell von Mises stress of a HEX8 solution at element centers.

    The displacement gradient comes from the trilinear basis evaluated at
    the element center, where ``dN_i/dxi`` is
    ``HEX_CORNER_SIGNS[i] / 8``.

    Args:
        points: Node positions, ``(N, 3)``.
        cells: HEX8 connectivity, ``(C, 8)``.
        displacement: Per-node displacement, ``(N, 3)``.
        youngs: Young's modulus.
        poisson: Poisson ratio.

    Returns:
        Von Mises stress per cell, shaped ``(C,)`` float64.
    """
    points = np.asarray(points, dtype=np.float64)
    cells = np.asarray(cells)
    displacement = np.asarray(displacement, dtype=np.float64)
    grad_ref = HEX_CORNER_SIGNS / 8.0  # (8, 3): dN/dxi at the center
    corner_positions = points[cells]  # (C, 8, 3)
    corner_disp = displacement[cells]  # (C, 8, 3)
    jacobian = np.einsum("cia,ib->cab", corner_positions, grad_ref)  # dx/dxi
    grad_phys = np.einsum("ib,cab->cia", grad_ref, np.linalg.inv(jacobian).transpose(0, 2, 1))
    u_grad = np.einsum("cia,cib->cab", corner_disp, grad_phys)  # du_a/dx_b
    return _von_mises_from_gradient(u_grad, youngs, poisson)


def tet_von_mises(
    points: np.ndarray,
    cells: np.ndarray,
    displacement: np.ndarray,
    *,
    youngs: float,
    poisson: float,
) -> np.ndarray:
    """Per-cell von Mises stress of a TET4/TET10 solution at cell centroids.

    The displacement gradient is exact for TET4 (constant strain).  For
    straight-sided TET10 it is evaluated at the element centroid, where the
    corner shape-function gradients vanish (``(4 L_i - 1) grad L_i`` with
    ``L_i = 1/4``) and the midside node on edge ``(a, b)`` contributes
    ``grad L_a + grad L_b`` — so the centroid gradient depends on the
    midside displacements only.

    Args:
        points: Node positions, ``(N, 3)``.
        cells: Connectivity, ``(T, 4)`` or ``(T, 10)``.
        displacement: Per-node displacement, ``(N, 3)``.
        youngs: Young's modulus.
        poisson: Poisson ratio.

    Returns:
        Von Mises stress per cell, shaped ``(T,)`` float64.
    """
    points = np.asarray(points, dtype=np.float64)
    cells = np.asarray(cells)
    displacement = np.asarray(displacement, dtype=np.float64)
    corners = points[cells[:, :4]]  # (T, 4, 3)
    edge_matrix = corners[:, 1:] - corners[:, :1]  # rows: x1-x0, x2-x0, x3-x0
    # grad L_i (i = 1..3) are the columns of the inverse edge matrix.
    grad_l123 = np.linalg.inv(edge_matrix).transpose(0, 2, 1)  # (T, 3, 3)
    grad_l = np.concatenate([-grad_l123.sum(axis=1, keepdims=True), grad_l123], axis=1)
    if cells.shape[1] == 4:
        u_grad = np.einsum("tia,tib->tab", displacement[cells], grad_l)
    elif cells.shape[1] == 10:
        edge_grads = grad_l[:, TET10_EDGES[:, 0]] + grad_l[:, TET10_EDGES[:, 1]]  # (T, 6, 3)
        u_grad = np.einsum("tia,tib->tab", displacement[cells[:, 4:]], edge_grads)
    else:
        raise ValueError(f"cells must be (T, 4) or (T, 10), got shape {cells.shape}.")
    return _von_mises_from_gradient(u_grad, youngs, poisson)


def load_work_tris(points: Any, displacement: Any, faces: np.ndarray, traction: Any) -> Any:
    """Work of a constant traction over linear (TRI3) boundary faces.

    ``W = sum_f area_f * t . mean(u at corners)`` — exact for linear
    interpolation, differentiable in both ``points`` and ``displacement``.
    Equal to the classical compliance ``f . u`` (twice the strain energy)
    when the faces carry the only load and supports are homogeneous.
    """
    import jax.numpy as jnp

    points = jnp.asarray(points)
    displacement = jnp.asarray(displacement)
    tri = points[np.asarray(faces)]  # (M, 3, 3)
    areas = 0.5 * jnp.linalg.norm(jnp.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=-1)
    mean_u = displacement[np.asarray(faces)].mean(axis=1)  # (M, 3)
    return jnp.sum(areas * (mean_u @ jnp.asarray(traction)))


def load_work_tri6(points: Any, displacement: Any, faces6: np.ndarray, traction: Any) -> Any:
    """Work of a constant traction over straight-sided quadratic (TRI6) faces.

    On a straight triangle the exact integral of a quadratic field uses
    the midside-only rule: ``integral(u) = area * mean(u at midsides)``.
    ``faces6`` is ``(M, 6)`` — three corners then the midsides opposite
    them (any consistent order; only corner/midside split matters).
    """
    import jax.numpy as jnp

    points = jnp.asarray(points)
    displacement = jnp.asarray(displacement)
    faces6 = np.asarray(faces6)
    tri = points[faces6[:, :3]]
    areas = 0.5 * jnp.linalg.norm(jnp.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=-1)
    mean_mid = displacement[faces6[:, 3:]].mean(axis=1)
    return jnp.sum(areas * (mean_mid @ jnp.asarray(traction)))


def load_work_quads(points: Any, displacement: Any, faces: np.ndarray, traction: Any) -> Any:
    """Work of a constant traction over bilinear (QUAD4) boundary faces.

    2x2 Gauss on the isoparametric bilinear map — matching jax-fem's own
    surface integration on HEX8 faces; differentiable in both arguments.
    """
    import jax.numpy as jnp

    points = jnp.asarray(points)
    displacement = jnp.asarray(displacement)
    corners = points[np.asarray(faces)]  # (M, 4, 3)
    corner_u = displacement[np.asarray(faces)]  # (M, 4, 3)
    g = 1.0 / np.sqrt(3.0)
    total = jnp.zeros(())
    for xi, eta in ((-g, -g), (g, -g), (g, g), (-g, g)):
        shape = 0.25 * jnp.asarray(
            [(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)]
        )
        d_xi = 0.25 * jnp.asarray([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
        d_eta = 0.25 * jnp.asarray([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
        tangent_u = jnp.einsum("i,mid->md", d_xi, corners)
        tangent_v = jnp.einsum("i,mid->md", d_eta, corners)
        jacobian = jnp.linalg.norm(jnp.cross(tangent_u, tangent_v), axis=-1)  # (M,)
        u_gauss = jnp.einsum("i,mid->md", shape, corner_u)  # (M, 3)
        total = total + jnp.sum(jacobian * (u_gauss @ jnp.asarray(traction)))
    return total
