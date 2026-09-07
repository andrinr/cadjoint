"""A cut-cell finite-element solver on an implicit model, differentiable in the design.

Steady heat conduction on Ω(θ) = {f(p, θ) < 0} without a volume mesh: a
background grid of Q1 cells, the boundary handled by marching-simplices
cut cells and Nitsche's method, Burman's ghost penalty on the cut faces,
and a sparse direct solve with a hand-written adjoint.  Everything that
depends on the design is in the quadrature, so ``jax.grad`` through
:func:`solve` is the discrete shape derivative at fixed structure — the
same "frozen topology, moving positions" contract the mesh route keeps
between refreezes, with no node map to get wrong at a crease.

Where it sits in the package: :class:`CutMesh` is what a
``SimMesh(method="cutfem")`` builds — the fixed structure plus the
projected dual-contour surface the viewer colours and selections resolve
on — and :func:`solve_thermal` is what a :class:`~cadjoint.fem.study.ThermalStudy`
calls on it.  The prototype this grew from, with its convergence tables,
is ``research/cutfem/``.


The domain is Ω(θ) = {f(p, θ) < 0} for a model in the zero-set grammar,
placed in a background grid of Q1 (bilinear or trilinear) cells. The
problem is −Δu = 1 on Ω with u = 0 on Γ = {f = 0}, enforced weakly by
Nitsche's method, with Burman's ghost penalty on the faces of cut cells.

Everything that depends on θ is in the quadrature: each cut cell is
divided into a sub-grid of simplices (two triangles per sub-cell, six
Kuhn tetrahedra in 3D); the field at the sub-grid corners fixes a sign
pattern, the *structure*, and at fixed structure the crossings on the
sub-simplex edges move smoothly with θ. The volume quadrature comes in two
modes, both differentiable:

* ``"smooth"``: every sub-simplex is integrated whole, its points weighted
  by a C² Heaviside of −f/ε with ε = α h. The derivative of such a weight
  is a smeared delta on Γ, so ``jax.grad`` of the assembly is a smeared
  Hadamard boundary term; the price is an O(ε) bias in the solution.
* ``"sharp"``: each sub-simplex is clipped by the linear interpolant of f,
  a marching-simplices tessellation of {f < 0}; the bias is O(h_sub²).

The boundary integrals use the marching-simplices polyline (2D) or
triangles (3D) on the same sub-grid, with normals from ∇f, in both modes.
The linear system is assembled in COO form and solved by a sparse direct
factorisation outside the graph, with a custom VJP whose adjoint is the
same factorisation (K is symmetric): ``jax.grad`` through
:func:`objective` is the discrete shape derivative at fixed structure, and
the cost is two factorisations per objective-and-gradient.  A dense path
remains for condition numbers and small checks.

The model is anything that evaluates as ``f(θ, points)``: a
:class:`cadjoint.zeroset.table.Model` lowered from a scene, a plain
function, or a :class:`Field` — including one closing over traced
parameters, with ``theta=None``.  Gradients in the point and in θ are
taken by JAX.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

from cadjoint.fem.boundary import FaceGroup, _tri_geometry
from cadjoint.fem.discretization import Surface

# float64 is required for the solve (Nitsche's penalty and the ghost penalty
# put the system around cond 1e3-1e4, and the derivative checks are 1e-9),
# but it is switched on where the solve runs — `_x64_scope` in
# :func:`solve_thermal` — never at import: this module is imported by every
# worker that executes a scene, and a global switch would recompile every
# program in the process in a different dtype.


# ----------------------------------------------------------------- the field


@dataclass(frozen=True)
class Field:
    """``f(θ, points)`` with its point gradient and θ-Jacobian by JAX.

    Wraps any batch evaluator ``(theta, points (N, d)) -> (N,)``; the point
    dimension is whatever the caller uses (a 2D model ignores z).
    """

    f: Callable

    def __call__(self, theta, points):
        return self.f(theta, points)

    def gradient(self, theta, points):
        """``∇ₚ f`` at each point, (N, d)."""
        points = jnp.asarray(points)
        return jax.vmap(jax.grad(lambda q: self.f(theta, q[None])[0]))(points)

    def jacobian(self, theta, points):
        """``∂f/∂θ`` at each point, (N, P)."""
        points = jnp.asarray(points)
        return jax.vmap(lambda q: jax.grad(lambda th: self.f(th, q[None])[0])(theta))(points)

    @classmethod
    def from_closure(cls, g: Callable) -> Field:
        """A field ``g(points)`` that carries its design inside (traced or not); use ``theta=None``."""
        return cls(lambda _theta, points: g(points))


def as_field(model: Any) -> Field:
    """A :class:`Field` from a cadjoint ``zeroset.Model`` or a function ``f(θ, points)``."""
    if isinstance(model, Field):
        return model
    if hasattr(model, "nodes") and hasattr(model, "root"):
        from cadjoint.zeroset.evaluate import field as zeroset_field

        return Field(zeroset_field(model))
    if callable(model):
        return Field(model)
    raise TypeError(f"not a field: {type(model).__name__}")


_TINY = 1e-300


def _safe_sqrt(v):
    return jnp.sqrt(jnp.maximum(v, _TINY))


# ------------------------------------------------------------------ the grid


@dataclass(frozen=True)
class Grid:
    """The solver's own lattice: ``cells[k]`` cells of size ``h`` along axis k from ``origin``.

    Dimension-generic (the prototype's 2D tables run on it), which is why it
    is not :class:`~cadjoint.meshing.GridSpec`; the package builds it from
    one (:func:`sdf_to_cut_mesh`), at the finest of the spec's spacings.
    """

    origin: tuple[float, ...]
    cells: tuple[int, ...]
    h: float

    @property
    def dim(self) -> int:
        return len(self.cells)

    @property
    def node_shape(self) -> tuple[int, ...]:
        return tuple(n + 1 for n in self.cells)


def grid_for(lo, hi, h: float) -> Grid:
    """The grid of cell size ``h`` whose box starts at ``lo`` and reaches at least ``hi``."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return Grid(tuple(lo.tolist()), tuple(int(math.ceil(x)) for x in (hi - lo) / h), float(h))


# ---------------------------------------------------- simplices and Q1 basis


def _kuhn(dim: int) -> np.ndarray:
    """The d! Kuhn simplices of the unit cube, as (d!, d+1, d) corner offsets in {0, 1}."""
    out = []
    for perm in itertools.permutations(range(dim)):
        v = np.zeros(dim, int)
        verts = [v.copy()]
        for axis in perm:
            v[axis] = 1
            verts.append(v.copy())
        out.append(verts)
    return np.array(out)


def _prism(p, q):
    return [[p[0], p[1], p[2], q[0]], [p[1], p[2], q[0], q[1]], [p[2], q[0], q[1], q[2]]]


def _cases(dim: int):
    """Marching simplices: per sign pattern, the inside simplices and the boundary facets.

    A vertex is a pair (i, j) of simplex corners: (i, i) is the corner, (i, j)
    the crossing on edge i→j at t = f_i / (f_i − f_j), i inside and j outside.
    Bit v of the pattern is set when corner v is inside (f < 0).
    """
    n = dim + 1
    table = {}
    for pattern in range(2**n):
        inside = [v for v in range(n) if pattern >> v & 1]
        outside = [v for v in range(n) if not pattern >> v & 1]
        vol: list = []
        bnd: list = []
        if dim == 2:
            if len(inside) == 3:
                vol = [[(0, 0), (1, 1), (2, 2)]]
            elif len(inside) == 1:
                a, (b, c) = inside[0], outside
                vol = [[(a, a), (a, b), (a, c)]]
                bnd = [[(a, b), (a, c)]]
            elif len(inside) == 2:
                (a, b), c = inside, outside[0]
                vol = [[(a, a), (b, b), (b, c)], [(a, a), (b, c), (a, c)]]
                bnd = [[(b, c), (a, c)]]
        elif dim == 3:
            if len(inside) == 4:
                vol = [[(0, 0), (1, 1), (2, 2), (3, 3)]]
            elif len(inside) == 1:
                a, (b, c, d) = inside[0], outside
                vol = [[(a, a), (a, b), (a, c), (a, d)]]
                bnd = [[(a, b), (a, c), (a, d)]]
            elif len(inside) == 2:
                (a, b), (c, d) = inside, outside
                vol = _prism([(a, a), (a, c), (a, d)], [(b, b), (b, c), (b, d)])
                bnd = [[(a, c), (a, d), (b, d)], [(a, c), (b, d), (b, c)]]
            elif len(inside) == 3:
                (a, b, c), d = inside, outside[0]
                vol = _prism([(a, a), (b, b), (c, c)], [(a, d), (b, d), (c, d)])
                bnd = [[(a, d), (b, d), (c, d)]]
        else:
            raise ValueError("only 2D and 3D")
        table[pattern] = (vol, bnd)
    return table


def _padded_tables(dim: int):
    """The case table as arrays: (patterns, slots, verts, 2) descriptors and a validity mask."""
    table = _cases(dim)
    n_pat = 2 ** (dim + 1)
    max_vol = max(len(v) for v, _ in table.values())
    max_bnd = max(len(b) for _, b in table.values())
    vol = np.zeros((n_pat, max_vol, dim + 1, 2), int)
    vol_mask = np.zeros((n_pat, max_vol), bool)
    bnd = np.zeros((n_pat, max_bnd, dim, 2), int)
    bnd_mask = np.zeros((n_pat, max_bnd), bool)
    for pattern, (v, b) in table.items():
        for s, simplex in enumerate(v):
            vol[pattern, s] = simplex
            vol_mask[pattern, s] = True
        for s, facet in enumerate(b):
            bnd[pattern, s] = facet
            bnd_mask[pattern, s] = True
    return vol, vol_mask, bnd, bnd_mask


def _volume_rule(dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Barycentric quadrature on a simplex, exact for the Q1 stiffness integrand.

    2D: the mid-edge rule, degree 2, exact for the bilinear stiffness. 3D:
    Keast's 14-point degree-5 rule with positive weights, exact for the
    trilinear stiffness (degree 4); the 4-point degree-2 rule under-integrates
    it and costs an order of convergence.
    """
    if dim == 2:
        return np.array([[0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5]]), np.full(3, 1 / 3)
    pts, w = [], []
    for a, b, weight in (
        (0.0673422422100983, 0.3108859192633005, 0.1126879257180162),
        (0.7217942490673264, 0.0927352503108912, 0.0734930431163619),
    ):
        for i in range(4):
            p = [b] * 4
            p[i] = a
            pts.append(p)
            w.append(weight)
    a, b, weight = 0.4544962958743506, 0.0455037041256494, 0.0425460207770812
    for i, j in itertools.combinations(range(4), 2):
        p = [b] * 4
        p[i] = p[j] = a
        pts.append(p)
        w.append(weight)
    return np.array(pts), np.array(w)


def _facet_rule(dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Barycentric quadrature on a boundary facet: 2-point Gauss (segment), mid-edge (triangle)."""
    if dim == 2:
        g = 1 / math.sqrt(3)
        return np.array([[(1 - g) / 2, (1 + g) / 2], [(1 + g) / 2, (1 - g) / 2]]), np.full(2, 0.5)
    return _volume_rule(2)


def _bits(dim: int) -> np.ndarray:
    """Q1 node order: node b has corner offset bit k of b along axis k."""
    return (np.arange(2**dim)[:, None] >> np.arange(dim)) & 1


def q1_basis(xi, dim: int):
    """Q1 shape functions and their reference gradients at local coordinates ``xi`` (N, d)."""
    bits = jnp.asarray(_bits(dim))
    one = jnp.where(bits[None] == 1, xi[:, None, :], 1.0 - xi[:, None, :])  # (N, 2^d, d)
    sign = jnp.where(bits == 1, 1.0, -1.0)  # (2^d, d)
    vals = jnp.prod(one, axis=-1)
    grads = []
    for m in range(dim):
        others = [k for k in range(dim) if k != m]
        grads.append(sign[None, :, m] * jnp.prod(one[..., others], axis=-1))
    return vals, jnp.stack(grads, axis=-1)


def smooth_heaviside(s, eps):
    """A C² step from 0 (s ≤ −ε) to 1 (s ≥ ε), the quintic smoothstep."""
    t = jnp.clip((s + eps) / (2.0 * eps), 0.0, 1.0)
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


# ------------------------------------------------------------- the problem


@dataclass(frozen=True)
class Thermal:
    """Steady heat conduction ``−∇·(k∇T) = q`` with conditions by region.

    ``dirichlet`` and ``neumann`` are ``(region, value)`` pairs: a region is
    any ``points (N, d) -> bool`` predicate — a cadjoint
    ``NodeSelection.contains`` qualifies — or ``None`` for the whole
    boundary.  A Dirichlet value is enforced by Nitsche's method, a
    Neumann value is an inflow per area (positive heats the body), and the
    rest of the boundary is insulated.  Regions are read on the facets at
    the structure's θ, the way the mesh route resolves its conditions on
    the nominal mesh; a facet keeps its condition while θ moves.

    The default is the prototype's torsion-like problem: unit conductivity
    and source, ``T = 0`` on the whole boundary.
    """

    conductivity: float = 1.0
    source: float = 1.0
    dirichlet: tuple = ((None, 0.0),)
    neumann: tuple = ()


# ---------------------------------------------------------------- structure


@dataclass
class Structure:
    """Everything about the discretisation that is fixed while θ moves.

    The active cells and their DOFs, the sub-grid corner positions, the
    marching-simplices descriptors chosen by the sign pattern at the
    structure's θ, and the ghost-penalty faces as a COO matrix.
    """

    model: Any
    f: Field
    problem: Thermal
    grid: Grid
    n_sub: int
    mode: str
    eps: float
    active: np.ndarray  # (n_active, d) cell multi-indices
    cut: np.ndarray  # (n_active,) bool
    origins: np.ndarray  # (n_active, d)
    dofs: np.ndarray  # (n_active, 2^d) global DOF numbers
    n_dofs: int
    x_sub: np.ndarray  # (n_active, n_c, d) sub-grid corner positions
    vol_cell: np.ndarray  # (N_v,)
    vol_ij: np.ndarray  # (N_v, d+1, 2) sub-corner ids
    vol_sign: np.ndarray  # (N_v,) orientation of each simplex at the structure's θ
    bnd_cell: np.ndarray  # (N_b,)
    bnd_ij: np.ndarray  # (N_b, d, 2)
    bnd_ref: np.ndarray  # (N_b, d) unit facet normals at the structure's θ
    bnd_dirichlet: np.ndarray  # (N_b,) prescribed value, NaN where the facet is not Dirichlet
    bnd_flux: np.ndarray  # (N_b,) inflow per area, 0 where the facet is not Neumann
    centroids: np.ndarray  # (N_b, d) facet centroids at the structure's θ, for reading regions
    active_index: np.ndarray  # grid.cells-shaped: active cell index, -1 elsewhere
    ghost_rows: np.ndarray
    ghost_cols: np.ndarray
    ghost_vals: np.ndarray  # the ghost penalty for unit γ_g
    # The sparsity pattern, fixed with the structure: the matrix's nonzeros
    # as (rows, cols), each cell's 2^d × 2^d block as slots into them, and the
    # ghost entries' slots.  Assembly reduces onto this inside XLA, so the
    # host solve sees ~27 n values rather than 64 per quadrature point.
    nz_rows: np.ndarray
    nz_cols: np.ndarray
    cell_slots: np.ndarray  # (n_active, 4^d)
    ghost_slots: np.ndarray

    @property
    def dim(self) -> int:
        return self.grid.dim

    @property
    def h(self) -> float:
        return self.grid.h

    def summary(self) -> dict[str, Any]:
        return {
            "cells": int(len(self.active)),
            "cut": int(self.cut.sum()),
            "dofs": int(self.n_dofs),
            "volume_simplices": int(len(self.vol_cell)),
            "boundary_facets": int(len(self.bnd_cell)),
        }


def _sub_corners(dim: int, n_sub: int) -> np.ndarray:
    """Local coordinates of the (n_sub+1)^d sub-grid corners, in ravel order."""
    axes = [np.arange(n_sub + 1)] * dim
    k = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, dim)
    return k / n_sub


def _sub_simplices(dim: int, n_sub: int) -> np.ndarray:
    """Sub-corner ids of the n_sub^d · d! sub-simplices of a cell, (n_simp, d+1)."""
    kuhn = _kuhn(dim)
    shape = (n_sub + 1,) * dim
    cells = np.stack(np.meshgrid(*[np.arange(n_sub)] * dim, indexing="ij"), axis=-1).reshape(
        -1, dim
    )
    verts = cells[:, None, None, :] + kuhn[None]  # (n_sub^d, d!, d+1, d)
    return np.ravel_multi_index(verts.reshape(-1, dim).T, shape).reshape(-1, dim + 1)


def _cell_simplices(dim: int, n_sub: int) -> np.ndarray:
    """Sub-corner ids of the d! Kuhn simplices of the whole cell, (d!, d+1)."""
    verts = _kuhn(dim) * n_sub
    return np.ravel_multi_index(verts.reshape(-1, dim).T, (n_sub + 1,) * dim).reshape(-1, dim + 1)


def _ghost_penalty(grid: Grid, active_index: np.ndarray, cut: np.ndarray, dofs: np.ndarray):
    """The ghost penalty Σ_F h ∫_F [∂ₙu][∂ₙv] over faces of cut cells, as COO for γ_g = 1."""
    dim, h = grid.dim, grid.h
    g = 1 / math.sqrt(3)
    gauss = np.array([(1 - g) / 2, (1 + g) / 2])
    rows, cols, vals = [], [], []
    for axis in range(dim):
        shift = np.zeros(dim, int)
        shift[axis] = 1
        idx = np.stack(np.meshgrid(*[np.arange(n) for n in grid.cells], indexing="ij"), -1)
        idx = idx.reshape(-1, dim)
        idx = idx[idx[:, axis] + 1 < grid.cells[axis]]
        a = active_index[tuple(idx.T)]
        b = active_index[tuple((idx + shift).T)]
        keep = (a >= 0) & (b >= 0) & (cut[np.maximum(a, 0)] | cut[np.maximum(b, 0)])
        a, b = a[keep], b[keep]
        if len(a) == 0:
            continue
        others = [k for k in range(dim) if k != axis]
        tangential = np.stack(np.meshgrid(*[gauss] * (dim - 1), indexing="ij"), -1).reshape(
            -1, dim - 1
        )
        weight = h ** (dim - 1) / len(tangential) * h  # face measure × the h of the penalty
        for tpt in tangential:
            xa, xb = np.zeros(dim), np.zeros(dim)
            xa[axis], xb[axis] = 1.0, 0.0
            xa[others], xb[others] = tpt, tpt
            _, ga = q1_basis(jnp.asarray(xa)[None], dim)
            _, gb = q1_basis(jnp.asarray(xb)[None], dim)
            jump = np.concatenate([np.asarray(ga)[0, :, axis], -np.asarray(gb)[0, :, axis]]) / h
            local = weight * np.outer(jump, jump)  # (2·2^d, 2·2^d)
            d = np.concatenate([dofs[a], dofs[b]], axis=1)  # (n_faces, 2·2^d)
            rows.append(np.repeat(d, d.shape[1], axis=1).ravel())
            cols.append(np.tile(d, (1, d.shape[1])).ravel())
            vals.append(np.tile(local.ravel(), len(a)))
    if not rows:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def _orientations(f_sub, x_sub, vol_cell, vol_ij, bnd_cell, bnd_ij):
    """Orientation signs of the volume simplices and unit normals of the facets at θ₀.

    Measures are taken *signed* against these, so that a sub-corner whose sign
    flips while the structure is held fixed makes its simplex's measure pass
    through zero smoothly instead of folding at zero: the fixed-structure
    objective is then C¹, like the re-classified one. At θ₀ itself the signed
    and unsigned measures, and their derivatives, coincide.
    """
    dim = x_sub.shape[-1]
    f_sub, x_sub = jnp.asarray(f_sub), jnp.asarray(x_sub)
    v = np.asarray(_vertices(f_sub, x_sub, jnp.asarray(vol_cell), jnp.asarray(vol_ij)))
    det = np.linalg.det(v[:, 1:] - v[:, :1])
    vol_sign = np.where(det < 0, -1.0, 1.0)
    vb = np.asarray(_vertices(f_sub, x_sub, jnp.asarray(bnd_cell), jnp.asarray(bnd_ij)))
    e = vb[:, 1:] - vb[:, :1]  # (N_b, d-1, d)
    ref = np.stack([-e[:, 0, 1], e[:, 0, 0]], axis=-1) if dim == 2 else np.cross(e[:, 0], e[:, 1])
    norm = np.linalg.norm(ref, axis=-1, keepdims=True)
    ref = ref / np.where(norm > 0, norm, 1.0)
    return vol_sign, ref


def _scan(f: Field, theta, points: np.ndarray, chunk: int = 1 << 16) -> np.ndarray:
    """``f`` on many concrete points, jitted and in chunks.

    A node table evaluated eagerly keeps one array per node alive for the
    whole batch — a thousand nodes on a million points is gigabytes — where
    the jitted fold frees each as it goes.
    """
    jitted = jax.jit(f.f)
    parts = np.array_split(points, max(1, math.ceil(len(points) / chunk)))
    return np.concatenate([np.asarray(jitted(theta, part)) for part in parts])


def _facet_conditions(problem: Thermal, centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each facet's Dirichlet value (NaN if none) and Neumann flux (0 if none).

    A Dirichlet region wins over a Neumann one where they overlap; among
    regions of one kind the first listed wins.
    """
    n = len(centroids)
    dirichlet = np.full(n, np.nan)
    flux = np.zeros(n)
    for region, value in reversed(problem.neumann):
        mask = np.ones(n, bool) if region is None else np.asarray(region(centroids), bool)
        flux[mask] = value
    for region, value in reversed(problem.dirichlet):
        mask = np.ones(n, bool) if region is None else np.asarray(region(centroids), bool)
        dirichlet[mask] = value
        flux[mask] = 0.0
    return dirichlet, flux


def classify(
    model: Any,
    theta,
    grid: Grid,
    n_sub: int = 4,
    mode: str = "sharp",
    alpha: float = 0.5,
    problem: Thermal | None = None,
) -> Structure:
    """Fix the discrete structure at ``theta``.

    Args:
        model: The implicit model.
        theta: Parameters at which cells are classified and sign patterns read.
        grid: The background grid.
        n_sub: Sub-grid divisions per cell axis for the marching simplices.
        mode: ``"sharp"`` (clipped sub-simplices) or ``"smooth"`` (Heaviside weights).
        alpha: The Heaviside half-width ε = α h; unused in sharp mode.
        problem: The conduction problem; the prototype's torsion problem by default.

    Returns:
        The structure: active cells, DOFs, quadrature descriptors, ghost faces.
    """
    if mode not in ("sharp", "smooth"):
        raise ValueError(f"mode {mode!r}")
    dim, h = grid.dim, grid.h
    eps = alpha * h if mode == "smooth" else 0.0
    theta = jax.tree_util.tree_map(lambda v: jnp.asarray(v, dtype=float), theta)
    f = as_field(model)

    all_cells = np.stack(np.meshgrid(*[np.arange(n) for n in grid.cells], indexing="ij"), -1)
    all_cells = all_cells.reshape(-1, dim)
    local = _sub_corners(dim, n_sub)  # (n_c, d)
    n_c = len(local)
    origins_all = np.asarray(grid.origin) + h * all_cells
    x_all = origins_all[:, None, :] + h * local[None]  # (cells, n_c, d)
    f_all = _scan(f, theta, x_all.reshape(-1, dim)).reshape(len(all_cells), n_c)
    fmin, fmax = f_all.min(axis=1), f_all.max(axis=1)
    band = eps
    active_mask = fmin < band
    full_mask = fmax < -band
    if mode == "smooth":
        # A cell is active only if some quadrature weight is non-zero: the mid-edge
        # points of its sub-simplices must reach into the band.
        simp = _sub_simplices(dim, n_sub)
        rule, _ = _volume_rule(dim)
        fq = np.einsum("qk,csk->csq", rule, f_all[:, simp])
        active_mask &= (fq < band).any(axis=(1, 2))

    active = all_cells[active_mask]
    cut = ~full_mask[active_mask]
    origins = origins_all[active_mask]
    x_sub = x_all[active_mask]
    f_sub = f_all[active_mask]
    n_active = len(active)
    active_index = -np.ones(grid.cells, int)
    active_index[tuple(active.T)] = np.arange(n_active)

    # DOFs: the nodes of active cells, renumbered densely.
    bits = _bits(dim)
    node_ids = np.ravel_multi_index(
        (active[:, None, :] + bits[None]).reshape(-1, dim).T, grid.node_shape
    ).reshape(n_active, 2**dim)
    used = np.unique(node_ids)
    dof_of_node = -np.ones(int(np.prod(grid.node_shape)), int)
    dof_of_node[used] = np.arange(len(used))
    dofs = dof_of_node[node_ids]

    # Volume descriptors: full cells use the cell's own Kuhn simplices; cut cells
    # the sub-grid, clipped (sharp) or whole (smooth).
    full_ids = np.nonzero(~cut)[0]
    cut_ids = np.nonzero(cut)[0]
    cell_simp = _cell_simplices(dim, n_sub)  # (d!, d+1)
    vol_cell = [np.repeat(full_ids, len(cell_simp))]
    vol_ij = [np.tile(np.stack([cell_simp, cell_simp], -1), (len(full_ids), 1, 1))]
    simp = _sub_simplices(dim, n_sub)  # (n_simp, d+1)
    vol_tab, vol_mask, bnd_tab, bnd_mask = _padded_tables(dim)
    f_cut = f_sub[cut_ids]  # (n_cut, n_c)
    signs = (f_cut[:, simp] < 0).astype(int)  # (n_cut, n_simp, d+1)
    pattern = (signs << np.arange(dim + 1)).sum(axis=-1)  # (n_cut, n_simp)
    if mode == "sharp":
        desc = vol_tab[pattern]  # (n_cut, n_simp, max_vol, d+1, 2) local corner ids
        corners = np.take_along_axis(
            simp[None, :, None, None, :], desc.reshape(*desc.shape[:2], 1, 1, -1), axis=-1
        ).reshape(desc.shape)
        mask = vol_mask[pattern]  # (n_cut, n_simp, max_vol)
        cell_of = np.broadcast_to(cut_ids[:, None, None], mask.shape)
        vol_cell.append(cell_of[mask])
        vol_ij.append(corners[mask])
    else:
        vol_cell.append(np.repeat(cut_ids, len(simp)))
        vol_ij.append(np.tile(np.stack([simp, simp], -1), (len(cut_ids), 1, 1)))
    desc = bnd_tab[pattern]  # (n_cut, n_simp, max_bnd, d, 2)
    corners = np.take_along_axis(
        simp[None, :, None, None, :], desc.reshape(*desc.shape[:2], 1, 1, -1), axis=-1
    ).reshape(desc.shape)
    mask = bnd_mask[pattern]
    bnd_cell = np.broadcast_to(cut_ids[:, None, None], mask.shape)[mask]
    bnd_ij = corners[mask]

    vol_cell, vol_ij = np.concatenate(vol_cell), np.concatenate(vol_ij)
    vol_sign, bnd_ref = _orientations(f_sub, x_sub, vol_cell, vol_ij, bnd_cell, bnd_ij)
    problem = problem or Thermal()
    centroids = np.asarray(
        _vertices(
            jnp.asarray(f_sub), jnp.asarray(x_sub), jnp.asarray(bnd_cell), jnp.asarray(bnd_ij)
        )
    ).mean(axis=1)
    bnd_dirichlet, bnd_flux = _facet_conditions(problem, centroids)
    rows, cols, vals = _ghost_penalty(grid, active_index, cut, dofs)
    m = dofs.shape[1]
    block_rows = np.broadcast_to(dofs[:, :, None], (n_active, m, m)).reshape(-1)
    block_cols = np.broadcast_to(dofs[:, None, :], (n_active, m, m)).reshape(-1)
    all_rows = np.concatenate([block_rows, rows])
    all_cols = np.concatenate([block_cols, cols])
    pairs, slots = np.unique(all_rows * len(used) + all_cols, return_inverse=True)
    nz_rows, nz_cols = pairs // len(used), pairs % len(used)
    return Structure(
        model=model,
        f=f,
        problem=problem,
        grid=grid,
        n_sub=n_sub,
        mode=mode,
        eps=eps,
        active=active,
        cut=cut,
        origins=origins,
        dofs=dofs,
        n_dofs=len(used),
        x_sub=x_sub,
        vol_cell=vol_cell,
        vol_ij=vol_ij,
        vol_sign=vol_sign,
        bnd_cell=bnd_cell,
        bnd_ij=bnd_ij,
        bnd_ref=bnd_ref,
        bnd_dirichlet=bnd_dirichlet,
        bnd_flux=bnd_flux,
        centroids=centroids,
        active_index=active_index,
        ghost_rows=rows,
        ghost_cols=cols,
        ghost_vals=vals,
        nz_rows=nz_rows,
        nz_cols=nz_cols,
        cell_slots=slots[: n_active * m * m].reshape(n_active, m * m),
        ghost_slots=slots[n_active * m * m :],
    )


# --------------------------------------------------------------- quadrature


def _vertices(f_sub, x_sub, cell, ij):
    """Positions of descriptor vertices: corners, or edge crossings of the linear interpolant."""
    i, j = ij[..., 0], ij[..., 1]
    c = cell[:, None]
    fi, fj = f_sub[c, i], f_sub[c, j]
    xi, xj = x_sub[c, i], x_sub[c, j]
    same = i == j
    t = jnp.where(same, 0.0, fi / jnp.where(same, 1.0, fi - fj))
    return xi + t[..., None] * (xj - xi)


def quadrature(theta, s: Structure):
    """The θ-dependent quadrature: volume and boundary points, weights, cells and normals."""
    dim = s.dim
    f = s.f
    x_sub = jnp.asarray(s.x_sub)
    f_sub = f(theta, x_sub.reshape(-1, dim)).reshape(x_sub.shape[:2])

    rule, w_rule = (jnp.asarray(a) for a in _volume_rule(dim))
    v = _vertices(f_sub, x_sub, jnp.asarray(s.vol_cell), jnp.asarray(s.vol_ij))  # (N, d+1, d)
    edges = v[:, 1:] - v[:, :1]
    measure = jnp.asarray(s.vol_sign) * jnp.linalg.det(edges) / math.factorial(dim)
    pts = jnp.einsum("qk,nkd->nqd", rule, v).reshape(-1, dim)
    w = (measure[:, None] * w_rule[None]).reshape(-1)
    cell = jnp.repeat(jnp.asarray(s.vol_cell), len(w_rule))
    if s.mode == "smooth":
        w = w * smooth_heaviside(-f(theta, pts), s.eps)

    frule, fw = (jnp.asarray(a) for a in _facet_rule(dim))
    vb = _vertices(f_sub, x_sub, jnp.asarray(s.bnd_cell), jnp.asarray(s.bnd_ij))  # (N_b, d, d)
    eb = vb[:, 1:] - vb[:, :1]
    frame = jnp.concatenate([eb, jnp.asarray(s.bnd_ref)[:, None, :]], axis=1)  # (N_b, d, d)
    bmeasure = jnp.linalg.det(frame) / math.factorial(dim - 1)
    bpts = jnp.einsum("qk,nkd->nqd", frule, vb).reshape(-1, dim)
    bw = (bmeasure[:, None] * fw[None]).reshape(-1)
    bcell = jnp.repeat(jnp.asarray(s.bnd_cell), len(fw))
    # The normal from ∇f a hair inside the facet rather than on it: a signed
    # distance written as sign·√(d·d) has value 0 and derivative 0·∞ exactly
    # on a planar face, where the interpolated crossings land, and the
    # gradient of a distance field is the same a hair away.
    ref = jnp.repeat(jnp.asarray(s.bnd_ref), len(fw), axis=0)
    grad = s.f.gradient(theta, bpts - 1e-7 * s.h * ref)
    normal = grad / _safe_sqrt(jnp.sum(grad * grad, axis=-1))[:, None]
    return (pts, w, cell), (bpts, bw, bcell, normal)


def _basis_at(pts, cell, s: Structure):
    xi = (pts - jnp.asarray(s.origins)[cell]) / s.h
    vals, grads = q1_basis(xi, s.dim)
    return vals, grads / s.h, jnp.asarray(s.dofs)[cell]


def _entries(theta, s: Structure, nitsche: float, ghost: float):
    """The stiffness matrix's nonzeros ``(rows, cols, vals)`` on the fixed pattern, and F.

    The volume and boundary contributions are summed per cell first (a
    segment sum over the quadrature points) and then scattered onto the
    pattern's slots, so no 64-entries-per-point array leaves XLA.
    """
    (pts, w, cell), (bpts, bw, bcell, normal) = quadrature(theta, s)
    n = s.n_dofs
    kappa, source = s.problem.conductivity, s.problem.source
    vals, grads, dofs = _basis_at(pts, cell, s)
    k_local = (kappa * w)[:, None, None] * jnp.einsum("qid,qjd->qij", grads, grads)
    F = jnp.zeros(n).at[dofs].add((source * w)[:, None] * vals)

    # each facet's condition, repeated over its quadrature points
    nq = len(bpts) // max(len(s.bnd_cell), 1)
    g_d = jnp.repeat(jnp.asarray(s.bnd_dirichlet), nq)
    is_d = ~jnp.isnan(g_d)
    g_d = jnp.where(is_d, g_d, 0.0)
    flux = jnp.repeat(jnp.asarray(s.bnd_flux), nq)
    bvals, bgrads, bdofs = _basis_at(bpts, bcell, s)
    dn = kappa * jnp.einsum("qid,qd->qi", bgrads, normal)
    penalty = nitsche * kappa / s.h
    wd = jnp.where(is_d, bw, 0.0)
    k_b = wd[:, None, None] * (
        -dn[:, :, None] * bvals[:, None, :]
        - bvals[:, :, None] * dn[:, None, :]
        + penalty * bvals[:, :, None] * bvals[:, None, :]
    )
    # Nitsche's data terms for a nonzero prescribed value, and the inflow
    F = F.at[bdofs].add((wd * g_d)[:, None] * (penalty * bvals - dn))
    F = F.at[bdofs].add((bw * flux)[:, None] * bvals)

    n_active, block = s.cell_slots.shape
    per_cell = jax.ops.segment_sum(k_local.reshape(-1, block), cell, n_active)
    per_cell = per_cell + jax.ops.segment_sum(k_b.reshape(-1, block), bcell, n_active)
    vals = jnp.zeros(len(s.nz_rows)).at[jnp.asarray(s.cell_slots)].add(per_cell)
    if ghost and len(s.ghost_vals):
        vals = vals.at[jnp.asarray(s.ghost_slots)].add(ghost * jnp.asarray(s.ghost_vals))
    return jnp.asarray(s.nz_rows), jnp.asarray(s.nz_cols), vals, F


def assemble(theta, s: Structure, nitsche: float = 20.0, ghost: float = 0.1):
    """The stiffness matrix and load vector at θ, dense over the active DOFs.

    For condition numbers and small checks; :func:`solve` never forms it.

    Args:
        theta: Parameters.
        s: The fixed structure.
        nitsche: Nitsche's penalty γ_N, applied as γ_N / h on Γ.
        ghost: The ghost penalty γ_g; 0 switches it off.

    Returns:
        ``(K, F)`` with K symmetric.
    """
    rows, cols, vals, F = _entries(theta, s, nitsche, ghost)
    n = s.n_dofs
    return jnp.zeros((n, n)).at[rows, cols].add(vals), F


# ------------------------------------------------------------ sparse solve
#
# K is symmetric, so the adjoint solve is the same factorisation.  The
# factorisation happens on the host (SuperLU through scipy); what JAX sees
# is a primitive with the VJP written out:  u = K⁻¹F  gives
#   F̄ = K⁻¹ ū,   K̄ = −F̄ uᵀ,
# and K̄ restricted to the COO pattern is −F̄[rows]·u[cols] per entry.


def _host_solve(vals, F, rows, cols):
    def run(vals, F, rows, cols):
        n = len(F)
        K = scipy.sparse.csc_matrix(
            (np.asarray(vals), (np.asarray(rows), np.asarray(cols))), shape=(n, n)
        )
        return scipy.sparse.linalg.splu(K).solve(np.asarray(F))

    return jax.pure_callback(run, jax.ShapeDtypeStruct(F.shape, F.dtype), vals, F, rows, cols)


@jax.custom_vjp
def sparse_solve(vals, F, rows, cols):
    """``u = K⁻¹ F`` for K the COO matrix ``(rows, cols, vals)``, differentiable in vals and F."""
    return _host_solve(vals, F, rows, cols)


def _sparse_solve_fwd(vals, F, rows, cols):
    u = _host_solve(vals, F, rows, cols)
    return u, (vals, u, rows, cols)


def _sparse_solve_bwd(residuals, u_bar):
    vals, u, rows, cols = residuals
    f_bar = _host_solve(vals, u_bar, rows, cols)
    zero = jnp.zeros(rows.shape, dtype=jax.dtypes.float0)
    return -f_bar[rows] * u[cols], f_bar, zero, zero


sparse_solve.defvjp(_sparse_solve_fwd, _sparse_solve_bwd)


def solve(theta, s: Structure, nitsche: float = 20.0, ghost: float = 0.1, dense: bool = False):
    """The discrete solution u and the objective J = ∫_Ω u = F·u.

    Args:
        dense: Solve the dense matrix inside the graph instead (small cases,
            and the check that the sparse path agrees).
    """
    if dense:
        K, F = assemble(theta, s, nitsche, ghost)
        u = jnp.linalg.solve(K, F)
        return u, F @ u
    rows, cols, vals, F = _entries(theta, s, nitsche, ghost)
    u = sparse_solve(vals, F, rows, cols)
    return u, F @ u


def objective(theta, s: Structure, nitsche: float = 20.0, ghost: float = 0.1, dense: bool = False):
    """J(θ) = F·u at fixed structure — ∫_Ω u for the torsion problem; differentiate with ``jax.grad``."""
    return solve(theta, s, nitsche, ghost, dense)[1]


def _volume_mean(u, theta, s: Structure):
    (pts, w, cell), _ = quadrature(theta, s)
    vals, _, dofs = _basis_at(pts, cell, s)
    return jnp.sum(w * jnp.sum(vals * u[dofs], axis=1)) / jnp.sum(w)


def mean_temperature(theta, s: Structure, nitsche: float = 20.0, ghost: float = 0.1):
    """``∫_Ω T / |Ω|`` at fixed structure, from the quadrature the solve used."""
    u, _ = solve(theta, s, nitsche, ghost)
    return _volume_mean(u, theta, s)


def with_conditions(s: Structure, problem: Thermal) -> Structure:
    """The structure with ``problem``'s conditions read onto its facets.

    A structure is classified before the study that will run on it is
    known (a ``SimMesh`` builds, a study solves), so the regions are read
    here, from the centroids the classification kept.
    """
    dirichlet, flux = _facet_conditions(problem, s.centroids)
    return dataclasses.replace(s, problem=problem, bnd_dirichlet=dirichlet, bnd_flux=flux)


# ------------------------------------------------------- the package's seam


@dataclass
class CutMesh:
    """A cut-cell discretisation with the surface it is displayed on.

    What ``SimMesh(method="cutfem")`` builds.  ``points`` and
    ``boundary_tris`` are the projected dual-contour surface at the design
    the mesh was built for — what a selection resolves on and what the
    viewer colours — and ``structure`` is the fixed cut-cell structure the
    solve runs on.  ``display_cell``/``display_xi`` locate each surface
    vertex in an active cell, so a solved field can be read there.  There
    are no elements: ``cells`` is None, and ``num_cells`` counts the active
    lattice cells.
    """

    points: np.ndarray
    boundary_tris: np.ndarray
    structure: Structure
    grid: Any
    display_cell: np.ndarray
    display_xi: np.ndarray
    cells: Any = None
    refinement: Any = None

    @property
    def num_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def num_cells(self) -> int:
        return int(len(self.structure.active))

    def all_boundary_faces(self) -> FaceGroup:
        centers, normals = _tri_geometry(self.points, self.boundary_tris)
        return FaceGroup(nodes=self.boundary_tris, centers=centers, normals=normals)

    # ── the discretization protocol (cadjoint.fem.discretization) ─────────

    @property
    def family(self) -> str:
        return "cut"

    def surface(self) -> Surface:
        return Surface(points=self.points, groups=(("surface", np.asarray(self.boundary_tris)),))

    def quality(self) -> dict[str, np.ndarray]:
        return {}  # no elements to grade

    def node_patch(self, selection: Any) -> np.ndarray:
        return selection.resolve(self)

    def face_patch(self, selection: Any) -> tuple[np.ndarray, np.ndarray]:
        """The surface vertices the selection picks, and the facets whose centroid it contains."""
        facets = np.nonzero(np.asarray(selection.contains(self.structure.centroids), bool))[0]
        if facets.size == 0:
            raise ValueError(
                f"Selection {selection.describe()} selects no boundary facet; "
                "a condition by region needs the region to reach the surface."
            )
        return np.asarray(selection.resolve(self), dtype=np.int32), facets

    def unresolvable(self, bcs: list) -> str | None:
        return unresolvable_condition(bcs, self)

    def moved(self, field: Any, *, smooth_passes: int = 0, design: Any = None) -> Any:  # noqa: ARG002 - the protocol's signature
        """The placement of cut cells is the field itself: there is no node map to move."""
        return field

    def thermal(
        self, problem: Any, *, placement: Any = None, backend: Any = None
    ) -> CutThermalResult:
        """Conduction on the structure, with the field it was built for or the placement's.

        Materials are not sampled on cut cells yet, so the conductivity is
        a number; a prescribed temperature is read as a number too.
        """
        if backend is not None:
            raise ValueError("cut cells are their own solver and take no backend")
        if not isinstance(problem.conductivity, (int, float)):
            raise ValueError(
                "cut cells do not sample the scene's materials yet: give the study a numeric conductivity."
            )
        conditions = Thermal(
            conductivity=float(problem.conductivity),
            source=float(problem.source),
            dirichlet=tuple((bc.contains, float(value)) for bc, value in problem.dirichlet),
            neumann=tuple((bc.contains, float(flux)) for bc, flux in problem.neumann),
        )
        return solve_thermal(self, conditions, placement)

    def elastic(self, problem: Any, *, placement: Any = None, backend: Any = None) -> Any:
        raise NotImplementedError("cut cells solve thermal studies only, for now")

    def traction_work(self, positions: Any, displacement: Any, selection: Any, vector: Any) -> Any:
        raise NotImplementedError("cut cells solve thermal studies only, for now")


def _locate(s: Structure, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The active cell holding each point and its local coordinates there.

    A surface vertex lies in a cut cell up to rounding at a cell face; one
    that rounds into an inactive cell takes the nearest active neighbour.
    """
    origin, h, cells = np.asarray(s.grid.origin), s.h, np.asarray(s.grid.cells)
    ijk = np.clip(np.floor((points - origin) / h).astype(int), 0, cells - 1)
    idx = s.active_index[tuple(ijk.T)]
    if (idx < 0).any():
        offsets = np.array(list(itertools.product((-1, 0, 1), repeat=s.dim)))
        for a in np.nonzero(idx < 0)[0]:
            best, best_d = -1, np.inf
            for o in offsets:
                c = ijk[a] + o
                if (c < 0).any() or (c >= cells).any():
                    continue
                k = s.active_index[tuple(c)]
                if k < 0:
                    continue
                d = np.linalg.norm(s.origins[k] + 0.5 * h - points[a])
                if d < best_d:
                    best, best_d = k, d
            if best < 0:
                best = int(np.argmin(np.linalg.norm(s.origins + 0.5 * h - points[a], axis=1)))
            idx[a] = best
    xi = np.clip((points - s.origins[idx]) / h, 0.0, 1.0)
    return idx, xi


def sdf_to_cut_mesh(sdf: Callable, grid: Any, *, n_sub: int = 1) -> CutMesh:
    """Classify the cut cells of ``sdf`` on the lattice, and extract its surface to show.

    Args:
        sdf: The field at the design to build for, callable on ``(N, 3)`` points.
        grid: A :class:`~cadjoint.meshing.GridSpec`; its box is the cut-cell
            grid's box, at the smallest of its spacings.
        n_sub: Sub-grid divisions per cell for the marching simplices.
    """
    from cadjoint.fem.motion import project_points
    from cadjoint.meshing import extract_mesh

    try:
        surface = extract_mesh(sdf, grid, sharp=True)
    except Exception:  # noqa: BLE001 - the robust placement is the fallback the tet route uses too
        surface = extract_mesh(sdf, grid, sharp=False)
    raw = np.asarray(surface.vertices, dtype=np.float64)
    faces = np.asarray(surface.faces, dtype=np.int64).reshape(-1, 3)
    max_step = 0.5 * float(np.linalg.norm(np.asarray(grid.spacing, float)))
    points = np.asarray(project_points(sdf, raw, max_step), dtype=np.float64)
    lo = np.asarray(grid.origin, float)
    hi = lo + np.asarray(grid.spacing, float) * np.asarray(grid.cells, float)
    structure = classify(
        Field.from_closure(sdf), None, grid_for(lo, hi, float(min(grid.spacing))), n_sub=n_sub
    )
    cell, xi = _locate(structure, points)
    return CutMesh(
        points=points,
        boundary_tris=faces,
        structure=structure,
        grid=grid,
        display_cell=cell,
        display_xi=xi,
    )


@dataclass(frozen=True)
class CutThermalResult:
    """A cut-cell conduction solution: the field on the display surface, and the DOFs.

    ``temperature`` is read at the surface vertices (what the viewer colours,
    the shape :class:`~cadjoint.fem.result.SimulationResult` expects);
    ``dofs`` are the lattice values; ``mean``/``max`` are the objectives the
    optimiser's metrics use — the volume mean from the solve's quadrature
    and the largest lattice value.  Traced whenever the solve was.
    """

    temperature: Any
    mesh: CutMesh
    dofs: Any
    volume_mean: Any

    def mean(self):
        return self.volume_mean

    def max(self):
        return jnp.max(self.dofs)

    def vtk_export(self, path: str) -> None:
        raise NotImplementedError("a cut-cell result has no volume mesh to export")


def solve_thermal(
    mesh: CutMesh,
    problem: Thermal,
    field: Callable | None = None,
    *,
    nitsche: float = 20.0,
    ghost: float = 0.1,
) -> CutThermalResult:
    """Solve ``problem`` on the mesh's structure, with the field it was built for or another.

    Args:
        mesh: The cut-cell mesh.
        problem: Conductivity, source, and conditions by region.
        field: A field ``g(points)`` to solve with instead of the one the
            mesh was built for — closing over traced parameters, this is
            the design derivative at frozen structure (the cut-cell
            counterpart of ``points=recompute_points(...)``).
    """
    from cadjoint.fem.backends import _x64_scope

    s = with_conditions(mesh.structure, problem)
    if field is not None:
        s = dataclasses.replace(s, f=Field.from_closure(field))
    with _x64_scope():
        u, _ = solve(None, s, nitsche, ghost)
        vals, _ = q1_basis(jnp.asarray(mesh.display_xi), s.dim)
        at_surface = jnp.sum(vals * u[jnp.asarray(s.dofs)[jnp.asarray(mesh.display_cell)]], axis=1)
        volume_mean = _volume_mean(u, None, s)
    return CutThermalResult(temperature=at_surface, mesh=mesh, dofs=u, volume_mean=volume_mean)


def unresolvable_condition(bcs: list, mesh: CutMesh) -> str | None:
    """The first condition whose region selects no boundary facet, described; None if all do."""
    for bc in bcs:
        if not np.asarray(bc.nodes.contains(mesh.structure.centroids), bool).any():
            return f"boundary condition {type(bc).__name__} {bc.nodes.describe()} selects no boundary facet"
    return None


def hadamard(theta, s: Structure, nitsche: float = 20.0, ghost: float = 0.1):
    """The classical shape derivative ∫_Γ (∂ₙu)² Vₙ ds with Vₙ = −∂_θ f / |∇f|, from u_h.

    J = ∫_Ω u is self-adjoint (p = u), so dJ[V] = ∫_Γ ∂ₙu ∂ₙp Vₙ; this evaluates
    it with the discrete gradient of u_h on the marching-simplices boundary.
    """
    u, _ = solve(theta, s, nitsche, ghost)
    _, (bpts, bw, bcell, normal) = quadrature(theta, s)
    _, bgrads, bdofs = _basis_at(bpts, bcell, s)
    dn_u = jnp.einsum("qid,qd,qi->q", bgrads, normal, u[bdofs])
    grad = s.f.gradient(theta, bpts)
    speed = -s.f.jacobian(theta, bpts) / _safe_sqrt(jnp.sum(grad**2, -1))[:, None]
    return jnp.einsum("q,q,qp->p", bw, dn_u**2, speed)


def volume(theta, s: Structure):
    """|Ω_h| = Σ weights, the quadrature's own measure of the domain."""
    (_, w, _), _ = quadrature(theta, s)
    return jnp.sum(w)


def condition_number(K) -> tuple[float, float, float]:
    """(λ_max / λ_min, λ_min, λ_max) of a symmetric matrix, by LAPACK's subset solver."""
    K = np.asarray(K)
    K = 0.5 * (K + K.T)
    n = len(K)
    lo = scipy.linalg.eigh(K, eigvals_only=True, subset_by_index=[0, 0], driver="evr")[0]
    hi = scipy.linalg.eigh(K, eigvals_only=True, subset_by_index=[n - 1, n - 1], driver="evr")[0]
    return float(hi / lo) if lo > 0 else float("inf"), float(lo), float(hi)


def central_difference(fn, theta, delta: float = 1e-5) -> np.ndarray:
    """dfn/dθ by central differences, one parameter at a time."""
    theta = np.asarray(theta, float)
    out = np.zeros_like(theta)
    for k in range(len(theta)):
        e = np.zeros_like(theta)
        e[k] = delta
        out[k] = (float(fn(theta + e)) - float(fn(theta - e))) / (2 * delta)
    return out
