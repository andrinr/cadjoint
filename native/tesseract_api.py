"""Tesseract packaging of the native (Rust) Tikhonov QEF vertex solve.

The differentiable stage of the native dual-contouring core
(``native/src/core.rs``) behind typed ``apply`` +
``vector_jacobian_product`` endpoints, mirroring the reference tesseracts
in ``cadjoint/fem/tesseracts/*``: Hermite data crosses the boundary as
plain arrays, the forward solve runs in the cdylib, and gradients are
served by the hand-derived linear-solve VJP implemented in Rust.

Inputs: per-edge Hermite ``points`` and unit ``normals`` (differentiable),
the frozen per-cell ``edge_ids`` slot table (12 slots, padded with -1) from
:func:`cadjoint.meshing.native.manifold_cell_incidence_native`, and the
Tikhonov ``regularization`` weight. Output: one *unclamped* vertex per
incidence row; the caller (``cadjoint.meshing.native.qef_vertices_native``)
applies the per-cell clamp in JAX so its subgradient semantics match the
reference pipeline exactly.

Runs locally without Docker via ``Tesseract.from_tesseract_api``, or
containerized with ``tesseract build`` for distribution (the build must
ship the compiled cdylib; see ``CADJOINT_NATIVE_MESHER``).
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float64, Int32, ShapeDType


class InputSchema(BaseModel):
    """Batched Tikhonov QEF: one vertex per incidence row."""

    points: Differentiable[Array[(None, 3), Float64]]
    normals: Differentiable[Array[(None, 3), Float64]]
    edge_ids: Array[(None, 12), Int32]
    regularization: Array[(), Float64]


class OutputSchema(BaseModel):
    """Unclamped QEF vertex per incidence row."""

    vertices: Differentiable[Array[(None, 3), Float64]]


def apply(inputs: InputSchema) -> OutputSchema:
    """Forward solve in the Rust core (opaque to JAX tracing)."""
    from cadjoint.meshing.native import qef_forward_arrays

    vertices = qef_forward_arrays(
        np.asarray(inputs.points),
        np.asarray(inputs.normals),
        np.asarray(inputs.edge_ids),
        float(np.asarray(inputs.regularization)),
    )
    return OutputSchema(vertices=vertices)


def abstract_eval(abstract_inputs):
    """Output shapes from input shapes (lets tesseract-jax trace the call)."""
    cell_count = abstract_inputs.edge_ids.shape[0]
    return {"vertices": ShapeDType(shape=(cell_count, 3), dtype="float64")}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict,
):
    """Hand-derived linear-solve VJP, computed in the Rust core."""
    from cadjoint.meshing.native import qef_vjp_arrays

    assert vjp_outputs == {"vertices"}, f"unexpected vjp_outputs: {vjp_outputs}"
    unsupported = set(vjp_inputs) - {"points", "normals"}
    if unsupported:
        raise ValueError(
            f"Non-differentiable inputs requested in vjp: {sorted(unsupported)}; "
            "the differentiable inputs are points and normals."
        )
    points_bar, normals_bar = qef_vjp_arrays(
        np.asarray(inputs.points),
        np.asarray(inputs.normals),
        np.asarray(inputs.edge_ids),
        float(np.asarray(inputs.regularization)),
        np.asarray(cotangent_vector["vertices"]),
    )
    gradients = {"points": points_bar, "normals": normals_bar}
    return {name: gradients[name] for name in vjp_inputs}
