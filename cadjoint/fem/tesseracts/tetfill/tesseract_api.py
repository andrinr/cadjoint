"""TetGen volume filling as a Tesseract — the narrow cut around the black box.

Where the ``mesher`` tesseract wraps the *whole* pipeline (dual contouring,
Newton projection and TetGen) behind one opaque ``apply``, this one wraps
only the part that is actually opaque: **TetGen**.  The user's design
insight, verbatim — *"i dont understand why the whole meshing pipeline is
in the meshing tesseract? i think its only the tet meshing that needs this
the rest should already natively be differentiable"* — is correct.  Dual
contouring in :mod:`cadjoint.meshing` is natively differentiable: frozen
crossing edges, bisection on ``stop_gradient`` values plus differentiable
Newton corrections on the **true SDF**, and QEF placement through a
differentiable linear solve.  Only the tetrahedralization is a Fortran/C++
black box, so only it needs a hand-written VJP.

Cutting there instead of around the whole pipeline has one measured
consequence (``research/tet-vs-hex.md``): the whole-pipeline tesseract
forces every boundary vertex onto the *trilinear interpolant's* zero set,
which smears creases across a cell — the source of the crease-dominated
sign flip on the bracket and of the TetGen self-intersections that stop
the starter heat sink from meshing at its declared resolution.  This
tesseract never sees the field at all: the surface arrives as points, so
the boundary stays on the true SDF's zero set.

Contract
--------

``apply`` takes a watertight triangulated surface (``points`` +
``triangles``) and returns the tet mesh of its interior.  TetGen runs in
PLC mode with boundary splitting disabled (``-Y``/``nobisect``), so the
input vertices survive **verbatim** as the leading output nodes and every
added node is strictly interior; the forward asserts that (bit-for-bit on
the leading block), because the whole VJP rests on it.

Two modes, one endpoint.  With ``interior_points`` empty, ``apply`` *runs
TetGen* — the discovery call, and the honest black-box forward.  With
``interior_points`` non-empty it *re-evaluates a frozen fill*: the
interior nodes are held at the given positions and ``cell_template``
carries the frozen connectivity verbatim, so the forward is exactly the
gather its VJP transposes.  The second mode exists because TetGen's
quality-driven Steiner insertion is not continuous in the input surface —
on the box bar below, a design perturbation of 1e-4 already changes the
Steiner count and breaks the frozen-topology promise — while the frozen
fill is the same contract ``tetmesh.recompute_tet_points`` honours on the
direct path (interior held, boundary moving).  Holding the interior is
also exactly what the VJP already asserts by dropping Steiner cotangents,
so in this mode forward and derivative are consistent by construction
rather than to a tolerance.

``vector_jacobian_product`` w.r.t. ``points`` is therefore the transpose
of a **gather**:

* preserved vertices — output node ``i < V`` *is* input vertex ``i``, so
  its cotangent passes straight through;
* Steiner nodes — dropped.  Interior node motion is mesh gauge with no
  shape meaning (Hadamard: only normal boundary motion changes the shape),
  and the discrete size of the term is measured, not assumed: on the
  bracket TET4 @ 32x24x19 the interior sensitivity is 15x (RMS) / 26x
  (max) below the boundary sensitivity — see the "Interior (Steiner) node
  sensitivity" table in ``research/tet-vs-hex.md``;
* TET10 midside nodes (``element = 2``) — the promotion ``m = (a + b)/2``
  is exactly linear, so each midside cotangent splits half-and-half onto
  its two corner parents before the corner-level pass-through (the same
  step the ``mesher`` tesseract takes).

All three cases are one table: ``parents`` is ``(P, 2)`` and every output
node contributes ``0.5`` of its cotangent to each listed input vertex
(``-1`` = no parent, dropped).  A preserved vertex lists itself twice
(``0.5 + 0.5 = 1``), a midside lists its two corners, a Steiner node lists
none.  Because the map is a gather, the VJP is **exact** — a mechanical
check against ``jax.vjp`` of the same gather agrees to ~1e-16, not to a
tolerance.

Topology is discrete and data-dependent, so output shapes cannot follow
from input shapes.  As in the ``mesher`` tesseract the contract is
frozen-topology: run ``apply`` concretely once to discover the mesh, then
pass ``node_ids = arange(P)`` and a ``(T, K)`` ``cell_template`` so
``abstract_eval`` can promise shapes for the traced call.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float64, Int32, ShapeDType


class InputSchema(BaseModel):
    """A watertight surface plus TetGen options and the topology promise.

    ``element`` mirrors the ``mesher`` tesseract's selector: ``0`` = TET4
    (the raw TetGen output), ``2`` = the same mesh promoted to
    straight-sided TET10 with shared midside nodes appended after all
    corner nodes (meshio ``tetra10`` order).  ``1`` (HEX8) has no meaning
    here — this tesseract fills a surface, it does not voxelize a field.
    """

    points: Differentiable[Array[(None, 3), Float64]]
    triangles: Array[(None, 3), Int32]
    element: Array[(), Int32]
    min_ratio: Array[(), Float64]
    min_dihedral: Array[(), Float64]
    # Frozen interior (Steiner) node positions, ``(S, 3)``.  Empty runs
    # TetGen; non-empty re-evaluates the frozen fill (see the module
    # docstring) and then ``cell_template`` must carry the real
    # connectivity, not just its shape.
    interior_points: Array[(None, 3), Float64]
    # Topology templates: their shapes promise the frozen topology to
    # abstract_eval (and ``cell_template``'s values are the connectivity in
    # frozen-fill mode).  Empty arrays = discovery mode (direct ``apply``
    # only; traced calls need the real shapes).
    node_ids: Array[(None,), Int32]
    cell_template: Array[(None, None), Int32]


class OutputSchema(BaseModel):
    """The volume mesh; only ``nodes`` is differentiable (frozen topology).

    ``parents`` is the VJP table described in the module docstring: row
    ``i`` lists the input-vertex indices that own output node ``i``, each
    with weight ``0.5`` (``-1`` = none).  ``steiner_mask`` flags the nodes
    TetGen added (and, in TET10 mode, midside nodes that are not pure
    boundary averages) — the ones whose cotangent is dropped or split.
    """

    nodes: Differentiable[Array[(None, 3), Float64]]
    cells: Array[(None, None), Int32]
    parents: Array[(None, 2), Int32]
    steiner_mask: Array[(None,), Int32]


def _tetgen_corners(inputs: InputSchema, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run TetGen on the surface (the black box) and return corner nodes/cells.

    Raises:
        RuntimeError: When TetGen rejects the surface, or (the assertion
            the whole VJP rests on) fails to preserve the input vertices.
    """
    from cadjoint.fem.tetmesh import surface_to_tet_mesh

    mesh = surface_to_tet_mesh(
        points,
        np.asarray(inputs.triangles, dtype=np.int64),
        min_ratio=float(np.asarray(inputs.min_ratio)),
        min_dihedral=float(np.asarray(inputs.min_dihedral)),
    )
    count = int(points.shape[0])
    nodes = np.asarray(mesh.points, dtype=np.float64)
    # surface_to_tet_mesh already checks -Y preservation to 1e-12; the
    # gather VJP claims more than that, so verify it exactly here.
    if mesh.num_surface != count or not np.array_equal(nodes[:count], points):
        raise RuntimeError(
            "TetGen did not preserve the input surface vertices bit-for-bit as its "
            f"leading {count} nodes (nobisect/-Y contract); the pass-through VJP of "
            "the tetfill tesseract is only exact when it does."
        )
    return nodes, np.asarray(mesh.cells, dtype=np.int32)


def _frozen_corners(inputs: InputSchema, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Re-evaluate a frozen fill: interior held, connectivity taken verbatim."""
    template = np.asarray(inputs.cell_template, dtype=np.int32)
    if template.size == 0:
        raise ValueError(
            "interior_points was given without a cell_template carrying the frozen "
            "connectivity; the frozen fill needs both (see the module docstring)."
        )
    interior = np.asarray(inputs.interior_points, dtype=np.float64)
    return np.concatenate([points, interior], axis=0), template[:, :4]


def _fill(inputs: InputSchema):
    """The forward: TetGen, or the frozen re-fill when the interior is pinned.

    Returns:
        ``(nodes, cells, parents, steiner_mask)``.  ``nodes[:V]`` are the
        input vertices verbatim (asserted); ``parents`` is the ``(P, 2)``
        half-weight ownership table and ``steiner_mask`` the complement of
        the preserved block.
    """
    from cadjoint.fem.backends import _x64_scope
    from cadjoint.fem.tetmesh import tet10_from_tet4

    with _x64_scope():
        points = np.asarray(inputs.points, dtype=np.float64)
        count = int(points.shape[0])
        frozen = int(np.asarray(inputs.interior_points).size) > 0
        if frozen:
            nodes, cells = _frozen_corners(inputs, points)
        else:
            nodes, cells = _tetgen_corners(inputs, points)
        element = int(np.asarray(inputs.element))
        if element not in (0, 2):
            raise ValueError(f"element must be 0 (TET4) or 2 (TET10), got {element}.")
        if element == 2:
            # Deterministic in the corner connectivity alone, so the frozen
            # re-fill reproduces the discovery promotion node for node.
            nodes, cells, edges = tet10_from_tet4(nodes, cells)
        else:
            edges = None
        total = int(nodes.shape[0])
        parents = np.full((total, 2), -1, dtype=np.int32)
        preserved = np.arange(count, dtype=np.int32)
        # Preserved vertices own themselves twice: 0.5 + 0.5 = pass-through.
        parents[:count, 0] = preserved
        parents[:count, 1] = preserved
        if edges is not None:
            corner_count = total - int(edges.shape[0])
            pairs = np.asarray(edges, dtype=np.int64)
            # A midside's corner parent contributes only when that corner is
            # itself a preserved input vertex; Steiner corners drop out.
            owned = np.where(pairs < count, pairs, -1).astype(np.int32)
            parents[corner_count:] = owned
        mask = np.ones(total, dtype=np.int32)
        mask[:count] = 0
        return nodes, cells, parents, mask


def apply(inputs: InputSchema) -> OutputSchema:
    """Fill the input surface with tets (opaque; runs concretely)."""
    nodes, cells, parents, mask = _fill(inputs)
    promised = int(inputs.node_ids.shape[0])
    if promised and promised != nodes.shape[0]:
        raise ValueError(
            f"Frozen-topology promise violated: caller promised {promised} nodes but the fill "
            f"produced {nodes.shape[0]}. Re-run discovery apply at this design (TetGen's "
            "Steiner insertion is not continuous in the surface; pin the interior with "
            "interior_points to hold the topology across a design step)."
        )
    template = np.asarray(inputs.cell_template, dtype=np.int32)
    if int(np.asarray(inputs.interior_points).size) > 0 and not np.array_equal(cells, template):
        raise ValueError(
            "The frozen fill's connectivity does not match cell_template; pass the discovery "
            "apply's own cells (TET10 templates must carry the promoted connectivity)."
        )
    return OutputSchema(nodes=nodes, cells=cells, parents=parents, steiner_mask=mask)


def abstract_eval(abstract_inputs):
    """Output shapes from the shape-carrying topology templates."""
    num_nodes = abstract_inputs.node_ids.shape[0]
    num_cells, nodes_per_cell = abstract_inputs.cell_template.shape
    if num_nodes == 0 or num_cells == 0:
        raise ValueError(
            "Traced tetfill calls need the frozen topology: pass node_ids=arange(P) and "
            "cell_template=zeros((T, K)) from a prior concrete apply at the same surface."
        )
    return {
        "nodes": ShapeDType(shape=(num_nodes, 3), dtype="float64"),
        "cells": ShapeDType(shape=(num_cells, nodes_per_cell), dtype="int32"),
        "parents": ShapeDType(shape=(num_nodes, 2), dtype="int32"),
        "steiner_mask": ShapeDType(shape=(num_nodes,), dtype="int32"),
    }


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict,
):
    """Transpose of the preserved-vertex gather: ``nodes`` -> ``points``.

    Re-runs the deterministic forward to recover the ownership table, then
    scatters each output node's cotangent onto its parents with weight
    ``0.5`` each.  Preserved vertices get their cotangent back unchanged,
    TET10 midsides split half-and-half onto their corner parents, and
    Steiner nodes contribute nothing.  Exact by construction — the forward
    restricted to the preserved block *is* a gather (see the module
    docstring for the measured justification of the interior drop).
    """
    unsupported = set(vjp_inputs) - {"points"}
    if unsupported:
        raise ValueError(
            f"Non-differentiable inputs requested in vjp: {sorted(unsupported)}; "
            "the only differentiable input is points."
        )
    if vjp_outputs != {"nodes"}:
        raise ValueError(f"Only 'nodes' carries a vjp; requested: {sorted(vjp_outputs)}")

    _nodes, _cells, parents, _mask = _fill(inputs)
    cotangent = np.asarray(cotangent_vector["nodes"], dtype=np.float64)
    points_bar = np.zeros_like(np.asarray(inputs.points, dtype=np.float64))
    for column in (0, 1):
        owners = parents[:, column]
        keep = owners >= 0
        np.add.at(points_bar, owners[keep].astype(np.int64), 0.5 * cotangent[keep])
    return {"points": points_bar}
