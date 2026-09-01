"""Tesseract packaging of the jax-fem thermal solve (reference plugin).

Proves the cadjoint FEM interop ABI: mesh + boundary conditions cross the
boundary as plain typed arrays, the forward solve runs as an opaque
``apply`` endpoint (jax-fem's PETSc-assembled Newton is not jax-traceable),
and gradients are served by a hand-written ``vector_jacobian_product``
endpoint backed by jax-fem's adjoint.  A third-party — even non-JAX —
solver plugs into cadjoint by shipping exactly this file shape and pointing
``cadjoint.fem.backends.TesseractBackend(api_path=...)`` at it.

Element types: the schema is element-agnostic — ``cells`` is ``(T, K)``
and ``K`` picks the element (4 = TET4, 8 = HEX8, 10 = TET10; meshio node
order).  HEX8 runs the direct backend's lifted solve verbatim; TET4/TET10
reuse :func:`cadjoint.fem.jaxfem.tet_thermal_solve` (direct sparse
solver, identical to the in-process tet path).  For TET10, boundary node
sets must include the patches' midside nodes.

Heat-flux (Neumann) patches mirror the direct backend's surface-map path:
``flux_nodes`` concatenates the vertex sets spanning each patch,
``flux_offsets`` are the ``P + 1`` prefix offsets (patch ``p`` spans
``flux_nodes[offsets[p]:offsets[p+1]]``), and ``flux_values`` prescribe
the inflow per area (positive heats the body).  A boundary face carries
the flux when all of its nodes are in the patch set; on tet meshes that
node-membership rule over-selects (interior faces whose corners all lie
on the patch), so ``flux_faces`` + ``flux_face_offsets`` optionally carry
the exact boundary corner triangles per patch (empty = pure membership,
the HEX8 behavior).

Runs locally without Docker via ``Tesseract.from_tesseract_api`` (used by
``TesseractBackend``), or containerized with ``tesseract build`` for
distribution.

Differentiable inputs: ``points``, ``conductivity``, ``source``.  Dirichlet
values are static (jax-fem bakes them into the DOF elimination, outside the
adjoint's parameter path); flux values are static like the direct backend's
(baked into weak-form closures).
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float64, Int32, ShapeDType

_ELE_TYPES = {4: "TET4", 8: "HEX8", 10: "TET10"}


class InputSchema(BaseModel):
    """Thermal problem: -div(k grad T) = q on a HEX8 / TET4 / TET10 mesh."""

    points: Differentiable[Array[(None, 3), Float64]]
    cells: Array[(None, None), Int32]
    dirichlet_nodes: Array[(None,), Int32]
    dirichlet_values: Array[(None,), Float64]
    flux_nodes: Array[(None,), Int32]
    flux_offsets: Array[(None,), Int32]
    flux_values: Array[(None,), Float64]
    flux_faces: Array[(None, 3), Int32]
    flux_face_offsets: Array[(None,), Int32]
    conductivity: Differentiable[Array[(), Float64]]
    source: Differentiable[Array[(), Float64]]


class OutputSchema(BaseModel):
    """Per-node steady-state temperature."""

    temperature: Differentiable[Array[(None,), Float64]]


def _split(flat: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    """Slice a concatenated per-patch array by its prefix offsets."""
    offsets = np.asarray(offsets, dtype=np.int64)
    return [flat[start:stop] for start, stop in zip(offsets[:-1], offsets[1:])]


def _solve(points, inputs, conductivity, source, base_points):
    """Run the jax-fem thermal solve; differentiable via its adjoint VJP."""
    from cadjoint.fem.backends import ThermalBCs
    from cadjoint.fem.jaxfem import JaxFemBackend, tet_thermal_solve

    cells = np.asarray(inputs.cells)
    try:
        ele_type = _ELE_TYPES[cells.shape[1]]
    except KeyError:
        raise ValueError(
            f"cells must be (T, 4), (T, 8), or (T, 10); got shape {cells.shape}."
        ) from None
    nodes = np.asarray(inputs.dirichlet_nodes, dtype=np.int32)
    values = np.asarray(inputs.dirichlet_values, dtype=np.float64)
    patches = [(nodes[values == value], float(value)) for value in np.unique(values)]
    bcs = ThermalBCs(
        dirichlet_nodes=[patch_nodes for patch_nodes, _ in patches],
        dirichlet_values=[value for _, value in patches],
        flux_nodes=_split(np.asarray(inputs.flux_nodes, dtype=np.int32), inputs.flux_offsets),
        flux_values=[float(value) for value in np.asarray(inputs.flux_values)],
    )
    face_offsets = np.asarray(inputs.flux_face_offsets, dtype=np.int64)
    faces = None
    if face_offsets.size:
        faces = _split(np.asarray(inputs.flux_faces, dtype=np.int64), face_offsets)
    if ele_type == "HEX8":
        if faces is not None:
            raise ValueError("flux_faces targeting is a tet feature; HEX8 uses node sets.")
        return JaxFemBackend().thermal(
            points,
            cells,
            bcs,
            conductivity=conductivity,
            source=source,
            base_points=base_points,
        )
    return tet_thermal_solve(
        points,
        cells,
        bcs,
        conductivity=conductivity,
        source=source,
        ele_type=ele_type,
        base_points=base_points,
        flux_faces=faces,
    )


def apply(inputs: InputSchema) -> OutputSchema:
    """Forward solve (opaque to JAX tracing; runs concretely)."""
    temperature = _solve(
        inputs.points,
        inputs,
        np.asarray(inputs.conductivity)[()],
        np.asarray(inputs.source)[()],
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
    solver is required.  The contract is identical across HEX8/TET4/TET10.
    """
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    assert vjp_outputs == {"temperature"}, f"unexpected vjp_outputs: {vjp_outputs}"

    def fun(params: dict):
        return _solve(
            params.get("points", inputs.points),
            inputs,
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
