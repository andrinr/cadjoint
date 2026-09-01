"""Tesseract packaging of the jax-fem thermal solve (reference plugin).

Proves the cadjoint FEM interop ABI: mesh + boundary conditions cross the
boundary as plain typed arrays, the forward solve runs as an opaque
``apply`` endpoint (jax-fem's PETSc-assembled Newton is not jax-traceable),
and gradients are served by a hand-written ``vector_jacobian_product``
endpoint backed by jax-fem's adjoint.  A third-party — even non-JAX —
solver plugs into cadjoint by shipping exactly this file shape and pointing
``cadjoint.fem.backends.TesseractBackend(api_path=...)`` at it.

Runs locally without Docker via ``Tesseract.from_tesseract_api`` (used by
``TesseractBackend``), or containerized with ``tesseract build`` for
distribution.

Differentiable inputs: ``points``, ``conductivity``, ``source``.  Dirichlet
values are static (jax-fem bakes them into the DOF elimination, outside the
adjoint's parameter path).
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float64, Int32, ShapeDType


class InputSchema(BaseModel):
    """Thermal problem: -div(k grad T) = q on a HEX8 mesh."""

    points: Differentiable[Array[(None, 3), Float64]]
    cells: Array[(None, 8), Int32]
    dirichlet_nodes: Array[(None,), Int32]
    dirichlet_values: Array[(None,), Float64]
    conductivity: Differentiable[Array[(), Float64]]
    source: Differentiable[Array[(), Float64]]


class OutputSchema(BaseModel):
    """Per-node steady-state temperature."""

    temperature: Differentiable[Array[(None,), Float64]]


def _solve(points, cells, dirichlet_nodes, dirichlet_values, conductivity, source, base_points):
    """Run the jax-fem thermal solve; differentiable via its adjoint VJP."""
    from cadjoint.fem.backends import JaxFemBackend, ThermalBCs

    nodes = np.asarray(dirichlet_nodes, dtype=np.int32)
    values = np.asarray(dirichlet_values, dtype=np.float64)
    patches = [(nodes[values == value], float(value)) for value in np.unique(values)]
    bcs = ThermalBCs(
        dirichlet_nodes=[patch_nodes for patch_nodes, _ in patches],
        dirichlet_values=[value for _, value in patches],
    )
    return JaxFemBackend().thermal(
        points,
        np.asarray(cells),
        bcs,
        conductivity=conductivity,
        source=source,
        base_points=base_points,
    )


def apply(inputs: InputSchema) -> OutputSchema:
    """Forward solve (opaque to JAX tracing; runs concretely)."""
    temperature = _solve(
        inputs.points,
        inputs.cells,
        inputs.dirichlet_nodes,
        inputs.dirichlet_values,
        inputs.conductivity,
        inputs.source,
        base_points=np.asarray(inputs.points, dtype=np.float64),
    )
    return OutputSchema(temperature=np.asarray(temperature))


def abstract_eval(abstract_inputs):
    """Output shapes from input shapes (lets tesseract-jax trace the call)."""
    num_nodes = abstract_inputs.points.shape[0]
    return {"temperature": ShapeDType(shape=(num_nodes,), dtype="float64")}


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
    assert vjp_outputs == {"temperature"}, f"unexpected vjp_outputs: {vjp_outputs}"

    def fun(params: dict):
        return _solve(
            params.get("points", inputs.points),
            inputs.cells,
            inputs.dirichlet_nodes,
            inputs.dirichlet_values,
            params.get("conductivity", np.asarray(inputs.conductivity)[()]),
            params.get("source", np.asarray(inputs.source)[()]),
            base_points=np.asarray(inputs.points, dtype=np.float64),
        )

    params = {}
    if "points" in vjp_inputs:
        params["points"] = jnp.asarray(inputs.points, dtype=jnp.float64)
    if "conductivity" in vjp_inputs:
        params["conductivity"] = jnp.asarray(inputs.conductivity, dtype=jnp.float64)[()]
    if "source" in vjp_inputs:
        params["source"] = jnp.asarray(inputs.source, dtype=jnp.float64)[()]
    unsupported = set(vjp_inputs) - set(params)
    if unsupported:
        raise ValueError(
            f"Non-differentiable inputs requested in vjp: {sorted(unsupported)}; "
            "differentiable inputs are points, conductivity, source."
        )

    _, vjp_fun = jax.vjp(fun, params)
    (gradients,) = vjp_fun(jnp.asarray(cotangent_vector["temperature"], dtype=jnp.float64))
    return {key: np.asarray(value) for key, value in gradients.items()}
