"""Tesseract packaging of the whole mesher with a surface-interpolation VJP.

Research prototype (Route 3 of ``research/tet-vs-hex.md``): the *entire*
black-box meshing pipeline — dual-contour surface extraction, Newton
projection onto the zero set, TetGen volume meshing — runs as one opaque
``apply`` endpoint, and differentiability w.r.t. the implicit field is
served by a hand-written ``vector_jacobian_product`` built from **surface
interpolation** alone, never from the mesher's internals.

The field crosses the boundary as samples on a regular lattice
(``field_values`` at the grid vertices).  Inside, the mesher operates on
the trilinear interpolant ``f(x) = sum_i w_i(x) f_i`` of those samples:
DC vertices are extracted from it and Newton-projected onto its zero set,
so every boundary vertex ``v`` satisfies ``f(v) = 0``.  The implicit
function theorem then gives the exact normal shape velocity

    dv/df_i = - w_i(v) * g / |g|^2,      g = grad f(v),

so the VJP of a cotangent ``vbar`` on the points is

    fbar_i = - sum_boundary (vbar . g / |g|^2) * w_i(v),

i.e. *the interpolation weights at the frozen vertex positions are the
rows of the VJP map*.  Interior (Steiner) vertices contribute nothing.

Gauge, stated honestly: this VJP carries only the **normal** component of
vertex motion.  The tangential component is mesher gauge — sliding a
vertex along the surface does not change the shape (Hadamard's
shape-derivative structure) — so physics objectives, which depend on the
shape, are exactly the case where dropping it is correct; a functional of
the mesh *parameterization* would need the mesher's true Jacobian.
Sharp-feature caveat: a vertex on a crease is constrained by several
smooth patches and its true velocity solves that multi-constraint system;
this v1 applies the single-field velocity there, and the resulting
gradient error is measured (not assumed) in the research note.

Topology is discrete and data-dependent, so output shapes cannot be
inferred from input shapes alone.  The contract is frozen-topology: run
``apply`` directly once (concretely) to discover ``points``/``cells``/
``surface_mask``, then pass ``point_ids = arange(N)`` and a
``cell_template`` of shape ``(T, K)`` (shape-carrying templates) so
``abstract_eval`` can promise shapes for the traced call at the same
design.  The forward is deterministic, so re-running it at that design
reproduces the promised topology.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float64, Int32, ShapeDType

_PROJECTION_STEPS = 12


class InputSchema(BaseModel):
    """Implicit field on a lattice plus meshing parameters and topology promise.

    ``element`` picks the mesher: ``0`` = DC surface + TetGen (TET4 cells,
    boundary vertices are the DC vertices), ``1`` = voxelize + Newton-snap
    (HEX8 cells, boundary vertices are the snapped lattice vertices —
    compatible with the packaged ``elastic_jaxfem`` solver tesseract),
    ``2`` = the TET4 mesh promoted to straight-sided TET10 (shared midside
    nodes appended after all corner vertices, meshio ``tetra10`` order).
    The interpolation VJP is mesher-agnostic: it only needs each movable
    boundary vertex to sit on the interpolant's zero set, which the corner
    meshers guarantee; a TET10 midside node is the affine average of its
    two corner parents, so its cotangent splits half-and-half onto them
    before the corner-level IFT pullback (exact — the promotion is linear).
    """

    field_values: Differentiable[Array[(None, None, None), Float64]]
    origin: Array[(3,), Float64]
    spacing: Array[(3,), Float64]
    element: Array[(), Int32]
    sharp: Array[(), Int32]
    min_ratio: Array[(), Float64]
    min_dihedral: Array[(), Float64]
    # Shape-carrying templates (values unused): shapes promise the frozen
    # topology for abstract_eval — ``cell_template`` is ``(T, K)`` with
    # ``K`` the nodes per cell (4 for TET4, 8 for HEX8, 10 for TET10; the
    # TET10 template thereby carries the edge structure — the deterministic
    # forward re-derives the concrete edge pairs).  Empty arrays =
    # discovery mode (direct ``apply`` only; traced calls need the real
    # shapes).
    point_ids: Array[(None,), Int32]
    cell_template: Array[(None, None), Int32]
    num_surface: Array[(), Int32]


class OutputSchema(BaseModel):
    """The volume mesh; only ``points`` is differentiable (frozen topology)."""

    points: Differentiable[Array[(None, 3), Float64]]
    cells: Array[(None, None), Int32]
    surface_mask: Array[(None,), Int32]


def make_interpolant(field_values, origin, spacing):
    """Differentiable trilinear interpolant of lattice samples.

    Args:
        field_values: Samples at grid vertices, shaped ``(nx+1, ny+1, nz+1)``.
        origin: Lattice origin, ``(3,)``.
        spacing: Cell edge lengths, ``(3,)``.

    Returns:
        A callable on ``(..., 3)`` points returning ``(...)`` values,
        differentiable in both the query points and ``field_values``.
    """
    import jax.numpy as jnp

    field = jnp.asarray(field_values)
    origin = jnp.asarray(origin, dtype=field.dtype)
    spacing = jnp.asarray(spacing, dtype=field.dtype)
    shape = np.array(field_values.shape)
    counts = jnp.asarray(shape - 1)
    strides = jnp.asarray([int(shape[1] * shape[2]), int(shape[2]), 1])
    flat = field.reshape(-1)

    def interpolant(points):
        q = (jnp.asarray(points) - origin) / spacing
        cell = jnp.clip(jnp.floor(q).astype(jnp.int32), 0, counts - 1)
        t = q - cell
        base = (cell * strides).sum(axis=-1)
        value = jnp.zeros(q.shape[:-1], dtype=field.dtype)
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    weight = (
                        (t[..., 0] if di else 1.0 - t[..., 0])
                        * (t[..., 1] if dj else 1.0 - t[..., 1])
                        * (t[..., 2] if dk else 1.0 - t[..., 2])
                    )
                    offset = di * strides[0] + dj * strides[1] + dk * strides[2]
                    value = value + weight * flat[base + offset]
        return value

    return interpolant


def _run_mesher(inputs: InputSchema):
    """The opaque forward mesher.

    Returns:
        ``(points, cells, movable_mask, edge_parents)``.  ``movable_mask``
        flags the vertices that sit on the interpolant's zero set and move
        with the field — DC surface vertices (TET4/TET10 modes) or snapped
        boundary lattice vertices (HEX8 mode); in TET10 mode it
        additionally marks the midside nodes both of whose corner parents
        are on the surface (those ride the surface too, as corner
        averages).  ``edge_parents`` is ``None`` except in TET10 mode,
        where row ``k`` holds the two corner indices whose midpoint node
        ``num_corners + k`` is.
    """

    from cadjoint.fem.backends import _x64_scope
    from cadjoint.fem.hexmesh import project_points, sdf_to_hex_mesh
    from cadjoint.fem.tetmesh import surface_to_tet_mesh, tet10_from_tet4
    from cadjoint.meshing import GridSpec, extract_mesh

    with _x64_scope():
        field = np.asarray(inputs.field_values, dtype=np.float64)
        origin = tuple(float(v) for v in np.asarray(inputs.origin))
        spacing = tuple(float(v) for v in np.asarray(inputs.spacing))
        grid = GridSpec(origin=origin, spacing=spacing, cells=tuple(n - 1 for n in field.shape))
        interpolant = make_interpolant(field, origin, spacing)
        element = int(np.asarray(inputs.element))
        if element == 1:
            hex_mesh = sdf_to_hex_mesh(interpolant, grid)
            mask = hex_mesh.snap_mask.astype(np.int32)
            return hex_mesh.points, hex_mesh.cells, mask, None
        surface = extract_mesh(interpolant, grid, sharp=bool(int(np.asarray(inputs.sharp))))
        raw = np.asarray(surface.vertices, dtype=np.float64)
        clamp = 0.5 * float(np.linalg.norm(spacing))
        projected = np.asarray(
            project_points(interpolant, raw, clamp, steps=_PROJECTION_STEPS), dtype=np.float64
        )
        mesh = surface_to_tet_mesh(
            projected,
            np.asarray(surface.faces),
            base_vertices=raw,
            grid=grid,
            min_ratio=float(np.asarray(inputs.min_ratio)),
            min_dihedral=float(np.asarray(inputs.min_dihedral)),
        )
        mask = np.zeros(mesh.num_points, dtype=np.int32)
        mask[: mesh.num_surface] = 1
        if element != 2:
            return mesh.points, mesh.cells, mask, None
        points10, cells10, parents = tet10_from_tet4(mesh.points, mesh.cells)
        mask10 = np.zeros(points10.shape[0], dtype=np.int32)
        mask10[: mesh.num_surface] = 1
        on_surface = (parents < mesh.num_surface).all(axis=1)
        mask10[mesh.num_points + np.flatnonzero(on_surface)] = 1
        return points10, cells10, mask10, parents


def apply(inputs: InputSchema) -> OutputSchema:
    """Run the black-box mesher (opaque to JAX tracing; runs concretely)."""
    points, cells, mask, _parents = _run_mesher(inputs)
    promised = int(inputs.point_ids.shape[0])
    if promised and promised != points.shape[0]:
        raise ValueError(
            f"Frozen-topology promise violated: caller promised {promised} points but the "
            f"mesher produced {points.shape[0]}. Re-run discovery apply at this design."
        )
    return OutputSchema(points=points, cells=cells, surface_mask=mask)


def abstract_eval(abstract_inputs):
    """Output shapes from the shape-carrying topology templates."""
    num_points = abstract_inputs.point_ids.shape[0]
    num_cells, nodes_per_cell = abstract_inputs.cell_template.shape
    if num_points == 0 or num_cells == 0:
        raise ValueError(
            "Traced mesher calls need the frozen topology: pass point_ids=arange(N) and "
            "cell_template=zeros((T, K)) from a prior concrete apply at the same design."
        )
    return {
        "points": ShapeDType(shape=(num_points, 3), dtype="float64"),
        "cells": ShapeDType(shape=(num_cells, nodes_per_cell), dtype="int32"),
        "surface_mask": ShapeDType(shape=(num_points,), dtype="int32"),
    }


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict,
):
    """Surface-interpolation VJP: cotangents on ``points`` -> ``field_values``.

    Re-runs the deterministic forward mesher to recover the frozen vertex
    positions, then applies the implicit-function-theorem map at the
    boundary vertices.  The weight scatter is realized as ``jax.vjp`` of
    the interpolant evaluated at the frozen boundary positions — the
    interpolation weights *are* the linear map, so this is exact.
    Interior (Steiner) vertex cotangents are dropped: interior motion is
    mesh gauge with no shape meaning (their smallness for physics
    objectives is measured in the research note).

    TET10 mode composes one exact linear step in front: each midside node
    is the affine average of its two corner parents (``m = (a + b) / 2``),
    so its cotangent splits half-and-half onto the corners before the
    corner-level IFT pullback.  Midsides between interior corners thereby
    land on interior corners and are dropped exactly as today.
    """
    import jax
    import jax.numpy as jnp

    from cadjoint.fem.backends import _x64_scope

    unsupported = set(vjp_inputs) - {"field_values"}
    if unsupported:
        raise ValueError(
            f"Non-differentiable inputs requested in vjp: {sorted(unsupported)}; "
            "the only differentiable input is field_values."
        )
    if vjp_outputs != {"points"}:
        raise ValueError(f"Only 'points' carries a vjp; requested: {sorted(vjp_outputs)}")

    with _x64_scope():
        points, _cells, mask, parents = _run_mesher(inputs)
        points = np.asarray(points, dtype=np.float64)
        cotangent_full = np.asarray(cotangent_vector["points"], dtype=np.float64)
        if parents is not None:
            # TET10: split each midside cotangent half-and-half onto its
            # two corner parents (the promotion is the exact linear map
            # m = (a + b) / 2), then pull back at the corners as usual.
            corner_count = points.shape[0] - parents.shape[0]
            corner_cotangent = cotangent_full[:corner_count].copy()
            np.add.at(corner_cotangent, parents[:, 0], 0.5 * cotangent_full[corner_count:])
            np.add.at(corner_cotangent, parents[:, 1], 0.5 * cotangent_full[corner_count:])
            points = points[:corner_count]
            mask = np.asarray(mask)[:corner_count]
            cotangent_full = corner_cotangent
        movable = np.flatnonzero(np.asarray(mask, dtype=bool))
        field = jnp.asarray(np.asarray(inputs.field_values, dtype=np.float64))
        origin = np.asarray(inputs.origin, dtype=np.float64)
        spacing = np.asarray(inputs.spacing, dtype=np.float64)
        boundary = jnp.asarray(points[movable])
        cotangent = jnp.asarray(cotangent_full[movable])

        def values_at_boundary(field_values):
            return make_interpolant(field_values, origin, spacing)(boundary)

        # Normal velocity scale per boundary vertex: s = (vbar . g) / |g|^2
        # with g the interpolant gradient at the frozen vertex position.
        interpolant = make_interpolant(field, origin, spacing)
        gradients = jax.vmap(jax.grad(lambda p: interpolant(p).reshape(())))(boundary)
        squared = jnp.sum(gradients * gradients, axis=-1)
        scale = jnp.sum(cotangent * gradients, axis=-1) / jnp.maximum(squared, 1e-12)
        _, vjp_fn = jax.vjp(values_at_boundary, field)
        (field_bar,) = vjp_fn(-scale)
        return {"field_values": np.asarray(field_bar)}
