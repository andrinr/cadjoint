"""Tesseract packaging of the CalculiX (ccx) elastic solve.

Mirrors the ``elastic_jaxfem`` reference schema (mesh + boundary
conditions as plain typed arrays) but runs the solve in a CalculiX
subprocess: ``apply`` writes a C3D8 ``.inp`` deck, invokes ``ccx`` and
parses the ``.dat``/``.frd`` results; ``vector_jacobian_product`` runs
ccx's native discrete adjoint (``*SENSITIVITY`` with the STRAINENERGY
design response over all boundary nodes) plus the volume-term correction
documented in :mod:`cadjoint.fem.calculix`.

Differentiability is objective-valued: only the ``strain_energy`` output
carries a VJP w.r.t. ``points`` (per-design-node normal sensitivities
chained through ccx's own outward node normals; tangential components
and interior nodes are zero — exact in the continuum limit, where
in-surface and interior mesh motion do not change the shape).  ccx has
no general displacement adjoint, so cotangents on ``displacement`` raise
``NotImplementedError``.  At traction-loaded nodes the VJP holds the
consistent nodal loads fixed (it neglects the load-area derivative);
restrict design freedom to unloaded boundary regions when that term
matters.

GPL note: CalculiX is GPL-2 and stays behind the subprocess boundary.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float64, Int32, ShapeDType


class InputSchema(BaseModel):
    """Small-strain linear elasticity on a HEX8 mesh (see elastic_jaxfem).

    ``traction_faces``/``traction_face_offsets`` exist for schema parity
    with ``elastic_jaxfem`` (whose tet modes use them for exact face
    targeting); the ccx path is HEX8-only and requires them empty.
    """

    points: Differentiable[Array[(None, 3), Float64]]
    cells: Array[(None, 8), Int32]
    fixed_nodes: Array[(None,), Int32]
    traction_nodes: Array[(None,), Int32]
    traction_offsets: Array[(None,), Int32]
    traction_vectors: Array[(None, 3), Float64]
    traction_faces: Array[(None, 3), Int32]
    traction_face_offsets: Array[(None,), Int32]
    youngs: Array[(), Float64]
    poisson: Array[(), Float64]


class OutputSchema(BaseModel):
    """Per-node displacement and the total strain energy of the solution."""

    displacement: Differentiable[Array[(None, 3), Float64]]
    strain_energy: Differentiable[Array[(), Float64]]


def _solve(inputs: InputSchema, *, sensitivities: bool):
    """Run ccx on the deck encoded by ``inputs``."""
    from cadjoint.fem.calculix import _unpack_elastic_bcs, elastic_ccx_solve

    if np.asarray(inputs.traction_face_offsets).size:
        raise ValueError(
            "traction_faces targeting is a tet feature of elastic_jaxfem; "
            "the CalculiX tesseract is HEX8-only and uses node sets."
        )
    return elastic_ccx_solve(
        np.asarray(inputs.points, dtype=np.float64),
        np.asarray(inputs.cells, dtype=np.int64),
        _unpack_elastic_bcs(
            inputs.fixed_nodes,
            inputs.traction_nodes,
            inputs.traction_offsets,
            inputs.traction_vectors,
        ),
        youngs=float(np.asarray(inputs.youngs)),
        poisson=float(np.asarray(inputs.poisson)),
        sensitivities=sensitivities,
    )


def apply(inputs: InputSchema) -> OutputSchema:
    """Forward solve (opaque to JAX tracing; subprocess ccx)."""
    solution = _solve(inputs, sensitivities=False)
    return OutputSchema(
        displacement=solution.displacement,
        strain_energy=np.asarray(solution.strain_energy, dtype=np.float64),
    )


def abstract_eval(abstract_inputs):
    """Output shapes from input shapes (lets tesseract-jax trace the call)."""
    num_nodes = abstract_inputs.points.shape[0]
    return {
        "displacement": ShapeDType(shape=(num_nodes, 3), dtype="float64"),
        "strain_energy": ShapeDType(shape=(), dtype="float64"),
    }


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict,
):
    """ccx adjoint gradients for the strain-energy objective.

    Supports cotangents on ``strain_energy`` only; a nonzero cotangent on
    ``displacement`` raises ``NotImplementedError`` (ccx exposes adjoints
    for its built-in design responses, not for the raw field).
    """
    del vjp_outputs  # which outputs carry cotangents is read from cotangent_vector
    unsupported = set(vjp_inputs) - {"points"}
    if unsupported:
        raise ValueError(
            f"Non-differentiable inputs requested in vjp: {sorted(unsupported)}; "
            "the only differentiable input is points."
        )
    displacement_cotangent = cotangent_vector.get("displacement")
    if displacement_cotangent is not None and np.any(
        np.asarray(displacement_cotangent, dtype=np.float64) != 0.0
    ):
        raise NotImplementedError(
            "The CalculiX tesseract only differentiates the strain_energy output "
            "(ccx has no general displacement adjoint); build objectives from "
            "strain_energy, or use the jax-fem backends for displacement-valued "
            "objectives."
        )
    scale = 0.0
    if "strain_energy" in cotangent_vector:
        scale = float(np.asarray(cotangent_vector["strain_energy"]))
    if scale == 0.0:
        return {"points": np.zeros_like(np.asarray(inputs.points, dtype=np.float64))}
    solution = _solve(inputs, sensitivities=True)
    return {"points": scale * solution.strain_energy_gradient}
