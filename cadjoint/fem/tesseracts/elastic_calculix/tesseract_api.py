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

Heterogeneous materials cross as the optional ``cell_youngs`` /
``cell_poisson`` per-element arrays (empty = "use the scalars").  A ccx
deck names materials rather than carrying per-element arrays, so a
continuously blended interface cannot be represented exactly; the deck
writer groups, caps and (past the cap) snaps the field onto named
materials, and warns about what that costs — see
:func:`cadjoint.fem.calculix.write_elastic_deck`.

GPL note: CalculiX is GPL-2 and stays behind the subprocess boundary.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float64, Int32, ShapeDType


def _empty(shape: tuple[int, ...]):
    """Default factory for an optional (absent) array input."""
    return lambda: np.zeros(shape, dtype=np.float64)


class InputSchema(BaseModel):
    """Small-strain linear elasticity on a HEX8 mesh (see elastic_jaxfem).

    ``traction_faces``/``traction_face_offsets`` exist for schema parity
    with ``elastic_jaxfem`` (whose tet modes use them for exact face
    targeting); the ccx path is HEX8-only and requires them empty.

    Materials: ``youngs``/``poisson`` are the single-material scalars.  A
    heterogeneous domain instead fills the optional ``cell_youngs`` /
    ``cell_poisson`` arrays with one value per element; both default to
    empty, which means "use the scalars", so every existing caller sends
    exactly the payload it always did (an HTTP caller, which cannot encode
    a zero-size array, simply omits the fields).  Since a ccx deck names materials
    and cannot carry a per-element array, a blended field is discretized
    onto named materials by ``cadjoint.fem.calculix.write_elastic_deck``
    (grouped by property, snapped onto reference materials past the group
    cap), which warns with ``CalculixQuantizationWarning`` when the deck
    cannot reproduce the requested field exactly.
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
    cell_youngs: Array[(None,), Float64] = Field(default_factory=_empty((0,)))
    cell_poisson: Array[(None,), Float64] = Field(default_factory=_empty((0,)))


class OutputSchema(BaseModel):
    """Per-node displacement and the total strain energy of the solution."""

    displacement: Differentiable[Array[(None, 3), Float64]]
    strain_energy: Differentiable[Array[(), Float64]]


def _solve(inputs: InputSchema, *, sensitivities: bool):
    """Run ccx on the deck encoded by ``inputs``."""
    from cadjoint.fem.calculix import _unpack_elastic_bcs, elastic_ccx_solve

    # A *populated* face-patch set is the tet feature; an empty one is just the
    # HEX8 caller saying "no face targeting".  Both the zero-length array and a
    # degenerate offsets array whose patches are all empty mean that -- the
    # latter because tesseract-core 1.11 cannot carry a zero-size array over the
    # HTTP boundary (polymorphic dimensions validate as PositiveInt), so a served
    # image can only be told "no faces" by an all-equal offsets array.
    face_offsets = np.asarray(inputs.traction_face_offsets)
    if face_offsets.size and int(face_offsets[-1]) > int(face_offsets[0]):
        raise ValueError(
            "traction_faces targeting is a tet feature of elastic_jaxfem; "
            "the CalculiX tesseract is HEX8-only and uses node sets."
        )
    cell_youngs = np.asarray(inputs.cell_youngs, dtype=np.float64)
    cell_poisson = np.asarray(inputs.cell_poisson, dtype=np.float64)
    return elastic_ccx_solve(
        np.asarray(inputs.points, dtype=np.float64),
        np.asarray(inputs.cells, dtype=np.int64),
        _unpack_elastic_bcs(
            inputs.fixed_nodes,
            inputs.traction_nodes,
            inputs.traction_offsets,
            inputs.traction_vectors,
        ),
        youngs=cell_youngs if cell_youngs.size else float(np.asarray(inputs.youngs)),
        poisson=cell_poisson if cell_poisson.size else float(np.asarray(inputs.poisson)),
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
