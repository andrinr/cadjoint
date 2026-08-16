"""Tesseract packaging of the jax-fem elastic solve (reference plugin).

Mirrors the thermal reference tesseract: mesh + boundary conditions cross
the boundary as plain typed arrays, the forward solve runs as an opaque
``apply`` endpoint (jax-fem's PETSc-assembled Newton is not jax-traceable),
and gradients are served by a hand-written ``vector_jacobian_product``
endpoint backed by jax-fem's adjoint.

Boundary-condition encoding (variable patch counts over fixed-rank
arrays): ``fixed_nodes`` is the union of all fully-clamped patches (all
displacement components pinned to zero, so patch identity is irrelevant);
traction patches keep their identity via ``traction_nodes`` (all patch
vertex sets concatenated), ``traction_offsets`` (``P + 1`` prefix offsets;
patch ``p`` spans ``traction_nodes[offsets[p]:offsets[p+1]]``), and
``traction_vectors`` (one force-per-area vector per patch).

Differentiable inputs: ``points`` only — matching the direct backend,
which bakes material constants and traction vectors into weak-form
closures.  Dirichlet displacements are identically zero (clamps).

Runs locally without Docker via ``Tesseract.from_tesseract_api`` (used by
``jaxcad.fem.backends.TesseractBackend``), or containerized with
``tesseract build`` for distribution.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float64, Int32, ShapeDType


class InputSchema(BaseModel):
    """Small-strain linear elasticity on a HEX8 mesh."""

    points: Differentiable[Array[(None, 3), Float64]]
    cells: Array[(None, 8), Int32]
    fixed_nodes: Array[(None,), Int32]
    traction_nodes: Array[(None,), Int32]
    traction_offsets: Array[(None,), Int32]
    traction_vectors: Array[(None, 3), Float64]
    youngs: Array[(), Float64]
    poisson: Array[(), Float64]


class OutputSchema(BaseModel):
    """Per-node displacement."""

    displacement: Differentiable[Array[(None, 3), Float64]]


def _solve(points, inputs, base_points):
    """Run the jax-fem elastic solve; differentiable via its adjoint VJP."""
    from jaxcad.fem.backends import ElasticBCs, JaxFemBackend

    offsets = np.asarray(inputs.traction_offsets, dtype=np.int64)
    traction_nodes = np.asarray(inputs.traction_nodes, dtype=np.int32)
    traction_vectors = np.asarray(inputs.traction_vectors, dtype=np.float64)
    bcs = ElasticBCs(
        fixed_nodes=[np.asarray(inputs.fixed_nodes, dtype=np.int32)],
        traction_nodes=[
            traction_nodes[start:stop] for start, stop in zip(offsets[:-1], offsets[1:])
        ],
        traction_vectors=list(traction_vectors),
    )
    return JaxFemBackend().elastic(
        points,
        np.asarray(inputs.cells),
        bcs,
        youngs=float(np.asarray(inputs.youngs)),
        poisson=float(np.asarray(inputs.poisson)),
        base_points=base_points,
    )


def apply(inputs: InputSchema) -> OutputSchema:
    """Forward solve (opaque to JAX tracing; runs concretely)."""
    displacement = _solve(
        inputs.points, inputs, base_points=np.asarray(inputs.points, dtype=np.float64)
    )
    return OutputSchema(displacement=np.asarray(displacement))


def abstract_eval(abstract_inputs):
    """Output shapes from input shapes (lets tesseract-jax trace the call)."""
    num_nodes = abstract_inputs.points.shape[0]
    return {"displacement": ShapeDType(shape=(num_nodes, 3), dtype="float64")}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict,
):
    """Adjoint gradients for the differentiable inputs.

    Composes jax.vjp over the solve closure; inside, jax-fem's ``ad_wrapper``
    (custom_vjp) solves the adjoint system, so no tracing of the forward
    solver is required.
    """
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    assert vjp_outputs == {"displacement"}, f"unexpected vjp_outputs: {vjp_outputs}"
    unsupported = set(vjp_inputs) - {"points"}
    if unsupported:
        raise ValueError(
            f"Non-differentiable inputs requested in vjp: {sorted(unsupported)}; "
            "the only differentiable input is points."
        )

    base_points = np.asarray(inputs.points, dtype=np.float64)

    def fun(points):
        return _solve(points, inputs, base_points=base_points)

    _, vjp_fun = jax.vjp(fun, jnp.asarray(inputs.points, dtype=jnp.float64))
    (gradient,) = vjp_fun(jnp.asarray(cotangent_vector["displacement"], dtype=jnp.float64))
    return {"points": np.asarray(gradient)}
