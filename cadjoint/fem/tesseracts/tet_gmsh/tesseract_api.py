"""Gmsh tet meshing of a solid — the narrow cut, and the GPL boundary.

Two things put this behind the Tesseract ABI, and only one of them is
technical.

**The licence.**  Gmsh is GPL-2.0-or-later.  cadjoint is Apache-2.0 and
must stay linkable by people who cannot take the GPL, so Gmsh may not be a
dependency of the library.  A Tesseract is a process boundary with a data
contract, so the GPL code lives in *this image* and cadjoint talks to it
over the ABI — the same reason the ccx package exists.  Nothing in
``cadjoint/`` imports ``gmsh`` at module scope, and the built-in ``tetfill``
package stays the no-dependency fallback.

**The narrow cut.**  Exactly as in ``tetfill``, the wrapper goes around the
part that is genuinely opaque and no further.  Gmsh's job is to look at a
solid — the dual-contour surface as STL, or an exact STEP — and decide *a
topology*: how many nodes, which tetrahedra, and which CAD entity owns each
node.  That decision is a black box.  What is **not** a black box is where
the nodes go: which patches own a node is a residual test against the
scene's public patch decomposition
(:func:`cadjoint.fem.gmsh.assign_ownership`), and moving the nodes under a
design change is the ``node_map`` plugin kind, whose implicit-function
adjoint carries the derivative.  Both run on the caller's side.  So this
package never sees a patch field, a scene or a parameter; it sees a file.

That is why ``bounding_surfaces`` is in the output schema.  It is the one
piece of Gmsh's answer the caller cannot recompute: for every node, the
surface tags whose closure it lies on.  A node on one surface solves one
field, a node on the curve between two solves two, a node where three meet
solves three — the arity falls straight out of that list.  Tags themselves
are *not* stable (the reader numbers them in its own order, and Gmsh
reports a cylindrical surface's type as ``Unknown``), so they are used only
to group nodes, never to identify geometry.

Contract
--------

Two modes, one endpoint, mirroring ``tetfill``.

* ``node_positions`` **empty** — the *discovery* call.  Gmsh runs; the
  topology it found comes back.  This is the honest black-box forward and
  it must run concretely.
* ``node_positions`` **non-empty** — the *frozen* call.  The topology is
  taken from ``cell_template`` and the companion arrays verbatim and
  ``nodes`` is ``node_positions`` unchanged.  Gmsh is not run at all.

The second mode is not a shortcut, it is the whole differentiable contract.
Gmsh's Delaunay refinement is not continuous in a design perturbation — a
1e-4 change in a bore radius moves a node count — so a traced call must not
re-run it.  Positions under a design change are recomputed *outside*, by
the projection kernel, and handed back in.  ``vector_jacobian_product``
w.r.t. ``node_positions`` is therefore the transpose of the identity: an
exact pass-through, not a tolerance.

Topology is discrete and data-dependent, so output shapes cannot follow
from input shapes.  As in ``mesher`` and ``tetfill`` the contract is
frozen-topology: run ``apply`` concretely once to discover the mesh, then
pass ``node_ids = arange(P)`` and the real ``cell_template`` so
``abstract_eval`` can promise shapes for the traced call.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float64, Int32, ShapeDType


class InputSchema(BaseModel):
    """A solid plus the meshing options and the topology promise.

    ``geometry`` is the file's own text, ASCII: the dual-contour surface as
    STL (:func:`cadjoint.fem.gmsh.surface_stl`, the public route) or an
    exact STEP solid (any CAD file, or the private tier's analytic writer).
    A file is what a mesher can be handed across a process boundary; a
    scene, whose patch fields are live Python callables, is not.
    """

    geometry: str
    # "stl" — classify the triangle soup into smooth regions and
    # reparametrise them (Gmsh t13) — or "step".
    geometry_format: str = "stl"
    # Uniform target element size, in the STEP file's own units.
    target_size: Array[(), Float64]
    # 1 = TET4, 2 = TET10 (the reason this package exists: Gmsh puts a
    # midside node on the CAD surface, which a straight-sided promotion of
    # a linear mesh cannot).
    order: Array[(), Int32]
    # Gmsh's Mesh.Algorithm3D; 10 is HXT.
    algorithm: Array[(), Int32]
    # Frozen node positions, ``(P, 3)``.  Empty runs Gmsh (discovery);
    # non-empty returns these verbatim over the frozen topology.
    node_positions: Differentiable[Array[(None, 3), Float64]]
    # Topology templates: their shapes promise the frozen topology to
    # abstract_eval, and in frozen mode their values *are* the topology.
    node_ids: Array[(None,), Int32]
    cell_template: Array[(None, None), Int32]
    entity_dim_template: Array[(None,), Int32]
    bounding_template: Array[(None, None), Int32]
    edge_parent_template: Array[(None, 2), Int32]


class OutputSchema(BaseModel):
    """The volume mesh and its CAD incidence; only ``nodes`` differentiates.

    Node rows are ordered the way
    :class:`cadjoint.fem.tetmesh.TetMesh` documents — boundary corners,
    then interior corners, then the shared midside block — so
    ``num_surface`` and ``edge_parents`` describe contiguous blocks.
    """

    nodes: Differentiable[Array[(None, 3), Float64]]
    cells: Array[(None, None), Int32]
    # Dimension of the CAD entity owning each node: 0 vertex, 1 curve,
    # 2 surface, 3 volume.  The node's projection arity is 3 - dim.
    entity_dim: Array[(None,), Int32]
    # Per node, the OCC surface tags whose closure it lies on, -1 padded.
    # The one part of Gmsh's answer the caller cannot recompute.
    bounding_surfaces: Array[(None, None), Int32]
    edge_parents: Array[(None, 2), Int32]
    num_surface: Array[(), Int32]
    num_corner_points: Array[(), Int32]
    bounds: Array[(2, 3), Float64]


def _discover(inputs: InputSchema) -> dict:
    """Run Gmsh on the STEP text — the black box.

    Raises:
        ImportError: If gmsh is not installed in this environment.
    """
    from cadjoint.fem.gmsh import gmsh_topology

    return gmsh_topology(
        inputs.geometry,
        geometry_format=str(inputs.geometry_format),
        target_size=float(np.asarray(inputs.target_size)),
        order=int(np.asarray(inputs.order)),
        algorithm=int(np.asarray(inputs.algorithm)),
    )


def _frozen(inputs: InputSchema) -> dict:
    """Re-serve a frozen topology at the given positions; Gmsh never runs.

    Raises:
        ValueError: If the templates do not carry a real topology.
    """
    cells = np.asarray(inputs.cell_template, dtype=np.int32)
    if cells.size == 0:
        raise ValueError(
            "node_positions was given without a cell_template carrying the frozen "
            "connectivity; the frozen call needs the whole topology (see the module "
            "docstring)."
        )
    positions = np.asarray(inputs.node_positions, dtype=np.float64)
    entity_dim = np.asarray(inputs.entity_dim_template, dtype=np.int32)
    if entity_dim.shape[0] != positions.shape[0]:
        raise ValueError(
            f"entity_dim_template has {entity_dim.shape[0]} rows but node_positions has "
            f"{positions.shape[0]}; both describe the same frozen node set."
        )
    is_corner = np.zeros(positions.shape[0], dtype=bool)
    is_corner[np.unique(cells[:, :4])] = True
    return {
        "points": positions,
        "cells": cells,
        "entity_dim": entity_dim,
        "bounding_surfaces": np.asarray(inputs.bounding_template, dtype=np.int32),
        "edge_parents": np.asarray(inputs.edge_parent_template, dtype=np.int32),
        "num_surface": int((is_corner & (entity_dim < 3)).sum()),
        "num_corner_points": int(is_corner.sum()),
        "bounds": np.stack([positions.min(axis=0), positions.max(axis=0)]),
    }


def _mesh(inputs: InputSchema) -> dict:
    """Discovery or frozen re-serve, normalised to one shape."""
    frozen = int(np.asarray(inputs.node_positions).size) > 0
    result = _frozen(inputs) if frozen else _discover(inputs)
    parents = result.get("edge_parents")
    return {
        "nodes": np.asarray(result["points"], dtype=np.float64),
        "cells": np.asarray(result["cells"], dtype=np.int32),
        "entity_dim": np.asarray(result["entity_dim"], dtype=np.int32),
        "bounding_surfaces": np.asarray(result["bounding_surfaces"], dtype=np.int32),
        "edge_parents": (
            np.zeros((0, 2), np.int32) if parents is None else np.asarray(parents, dtype=np.int32)
        ),
        "num_surface": np.int32(result["num_surface"]),
        "num_corner_points": np.int32(result["num_corner_points"]),
        "bounds": np.asarray(result["bounds"], dtype=np.float64),
    }


def apply(inputs: InputSchema) -> OutputSchema:
    """Mesh the solid, or re-serve a frozen topology (runs concretely).

    Raises:
        ValueError: If a promised node count does not match what Gmsh found
            — the frozen-topology promise, broken.
    """
    result = _mesh(inputs)
    promised = int(inputs.node_ids.shape[0])
    if promised and promised != result["nodes"].shape[0]:
        raise ValueError(
            f"Frozen-topology promise violated: caller promised {promised} nodes but the "
            f"mesh has {result['nodes'].shape[0]}. Re-run the discovery apply at this "
            "design (Gmsh's Delaunay refinement is not continuous in the geometry; pass "
            "node_positions to hold the topology across a design step)."
        )
    return OutputSchema(**result)


def abstract_eval(abstract_inputs):
    """Output shapes from the shape-carrying topology templates.

    Raises:
        ValueError: If the templates are empty — a traced call cannot
            rediscover a topology.
    """
    num_nodes = abstract_inputs.node_ids.shape[0]
    num_cells, nodes_per_cell = abstract_inputs.cell_template.shape
    if num_nodes == 0 or num_cells == 0:
        raise ValueError(
            "Traced tet_gmsh calls need the frozen topology: pass node_ids=arange(P) and "
            "the cell_template, entity_dim_template, bounding_template and "
            "edge_parent_template from a prior concrete apply on the same solid."
        )
    return {
        "nodes": ShapeDType(shape=(num_nodes, 3), dtype="float64"),
        "cells": ShapeDType(shape=(num_cells, nodes_per_cell), dtype="int32"),
        "entity_dim": ShapeDType(shape=(num_nodes,), dtype="int32"),
        "bounding_surfaces": ShapeDType(
            shape=tuple(abstract_inputs.bounding_template.shape), dtype="int32"
        ),
        "edge_parents": ShapeDType(
            shape=tuple(abstract_inputs.edge_parent_template.shape), dtype="int32"
        ),
        "num_surface": ShapeDType(shape=(), dtype="int32"),
        "num_corner_points": ShapeDType(shape=(), dtype="int32"),
        "bounds": ShapeDType(shape=(2, 3), dtype="float64"),
    }


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict,
):
    """Pass-through: ``nodes`` *is* ``node_positions``, so the VJP is identity.

    The topology is frozen and the positions were solved outside, by the
    projection kernel whose own implicit-function adjoint carries the design
    derivative.  What this endpoint transposes is therefore the identity map
    on positions — exact by construction, with no tolerance to state.

    Raises:
        ValueError: If a non-differentiable input or output is requested, or
            if the call is in discovery mode (where there is no map to
            transpose, because Gmsh's topology decision is not
            differentiable).
    """
    unsupported = set(vjp_inputs) - {"node_positions"}
    if unsupported:
        raise ValueError(
            f"Non-differentiable inputs requested in vjp: {sorted(unsupported)}; "
            "the only differentiable input is node_positions."
        )
    if vjp_outputs != {"nodes"}:
        raise ValueError(f"Only 'nodes' carries a vjp; requested: {sorted(vjp_outputs)}")
    positions = np.asarray(inputs.node_positions, dtype=np.float64)
    if positions.size == 0:
        raise ValueError(
            "vector_jacobian_product needs the frozen call: pass node_positions (and the "
            "topology templates) from a prior discovery apply. Gmsh's meshing decision "
            "itself is discrete and carries no derivative."
        )
    cotangent = np.asarray(cotangent_vector["nodes"], dtype=np.float64)
    if cotangent.shape != positions.shape:
        raise ValueError(
            f"cotangent for 'nodes' is shaped {cotangent.shape} but node_positions is "
            f"{positions.shape}; the frozen map is the identity, so they must agree."
        )
    return {"node_positions": cotangent}
