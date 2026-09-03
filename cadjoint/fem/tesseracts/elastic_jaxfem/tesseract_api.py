"""Tesseract packaging of the jax-fem elastic solve (reference plugin).

Mirrors the thermal reference tesseract: mesh + boundary conditions cross
the boundary as plain typed arrays, the forward solve runs as an opaque
``apply`` endpoint (jax-fem's PETSc-assembled Newton is not jax-traceable),
and gradients are served by a hand-written ``vector_jacobian_product``
endpoint backed by jax-fem's adjoint.

Element types: the schema is element-agnostic — ``cells`` is ``(T, K)``
and ``K`` picks the element (4 = TET4, 8 = HEX8, 10 = TET10; meshio
``tetra``/``hexahedron``/``tetra10`` node order).  HEX8 runs the direct
backend's solve verbatim; TET4/TET10 reuse
:func:`cadjoint.fem.jaxfem.tet_elastic_solve` (direct sparse solver,
identical to the in-process tet path).  For TET10, boundary-condition node
sets must include the patches' midside nodes
(:func:`cadjoint.fem.boundary.tet10_complete_nodes` /
:func:`~cadjoint.fem.boundary.tet10_face_midsides`).

Boundary-condition encoding (variable patch counts over fixed-rank
arrays): ``fixed_nodes`` is the union of all fully-clamped patches (all
displacement components pinned to zero, so patch identity is irrelevant);
traction patches keep their identity via ``traction_nodes`` (all patch
vertex sets concatenated), ``traction_offsets`` (``P + 1`` prefix offsets;
patch ``p`` spans ``traction_nodes[offsets[p]:offsets[p+1]]``), and
``traction_vectors`` (one force-per-area vector per patch).  On tet
meshes, node-membership face selection over-selects (interior faces whose
corners all lie on the loaded patch), so ``traction_faces`` +
``traction_face_offsets`` optionally carry the exact boundary corner
triangles per patch (same prefix-offset encoding; empty = pure node
membership, the HEX8 behavior).

Material heterogeneity and body loads: ``youngs`` / ``poisson`` are the
single-material scalars, and three optional arrays extend them —
``cell_youngs`` and ``cell_poisson`` carry one modulus per element
(``(C,)``), and ``body_force`` a body force density in N/m^3 (``(C, 3)``;
``density * gravity`` for self-weight).  All three default to **empty**,
which is what every scalar-path caller sends and means "the scalars apply
to the whole domain, no body force" — the payload and the solve are then
byte-identical to what they were before per-element properties existed.
A non-empty cell array supersedes its scalar (callers send ``0.0`` as the
placeholder); a non-empty ``body_force`` adds the mass-map term so the
strong form becomes ``div(sigma) + b = 0``.

Differentiable inputs: ``points``, ``cell_youngs``, ``cell_poisson``,
``body_force``.  The scalars ``youngs`` / ``poisson`` stay non-differentiable,
matching the direct backend, which bakes homogeneous material constants (and
the traction vectors) into weak-form closures where no adjoint reaches them.
The per-element arrays are a different matter: on the heterogeneous path
jax-fem receives the Lame constants and the body force through ``set_params``
as internal variables, so ``d(displacement)/d(E_e)``, ``d/d(nu_e)`` and
``d/d(b_e)`` fall out of the same adjoint solve that already serves
``points`` — no extra assembly, and the ``points`` VJP is untouched.  An
empty array never enters the traced closure, so its gradient is the
correctly-shaped zero and the single-material path is unchanged.  Dirichlet
displacements are identically zero (clamps).

Runs locally without Docker via ``Tesseract.from_tesseract_api`` (used by
``cadjoint.fem.backends.TesseractBackend``), or containerized with
``tesseract build`` for distribution.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float64, Int32, ShapeDType

_ELE_TYPES = {4: "TET4", 8: "HEX8", 10: "TET10"}


class InputSchema(BaseModel):
    """Small-strain linear elasticity on a HEX8 / TET4 / TET10 mesh."""

    points: Differentiable[Array[(None, 3), Float64]]
    cells: Array[(None, None), Int32]
    fixed_nodes: Array[(None,), Int32]
    traction_nodes: Array[(None,), Int32]
    traction_offsets: Array[(None,), Int32]
    traction_vectors: Array[(None, 3), Float64]
    traction_faces: Array[(None, 3), Int32]
    traction_face_offsets: Array[(None,), Int32]
    youngs: Array[(), Float64]
    poisson: Array[(), Float64]
    #: Optional per-element Young's modulus, ``(C,)``.  Empty (the default)
    #: selects the scalar ``youngs`` for the whole domain.
    cell_youngs: Differentiable[Array[(None,), Float64]] = Field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    #: Optional per-element Poisson ratio, ``(C,)``.  Empty (the default)
    #: selects the scalar ``poisson`` for the whole domain.
    cell_poisson: Differentiable[Array[(None,), Float64]] = Field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    #: Optional per-element body force density in N/m^3, ``(C, 3)``.  Empty
    #: (the default) means no body force.
    body_force: Differentiable[Array[(None, 3), Float64]] = Field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float64)
    )


class OutputSchema(BaseModel):
    """Per-node displacement."""

    displacement: Differentiable[Array[(None, 3), Float64]]


def _split(flat: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    """Slice a concatenated per-patch array by its prefix offsets."""
    offsets = np.asarray(offsets, dtype=np.int64)
    return [flat[start:stop] for start, stop in zip(offsets[:-1], offsets[1:])]


def _material(scalar, cell_values):
    """The modulus jax-fem should see: the ``(C,)`` array, else the scalar.

    An empty ``cell_values`` is the schema default and means "single
    material", so the scalar path is reproduced exactly (a python ``float``,
    as the pre-existing calls passed).
    """
    array = np.asarray(cell_values)
    return array if array.size else float(np.asarray(scalar))


def _body_force(values):
    """The body force jax-fem should see: the ``(C, 3)`` array, else ``None``."""
    array = np.asarray(values)
    return array if array.size else None


def _solve(points, inputs, youngs, poisson, body_force, base_points):
    """Run the jax-fem elastic solve; differentiable via its adjoint VJP.

    ``youngs`` / ``poisson`` are scalars (single material) or ``(C,)``
    per-element arrays, and ``body_force`` is ``None`` or a ``(C, 3)``
    density; every jax-fem entry point below accepts all three forms.
    """
    from cadjoint.fem.backends import ElasticBCs
    from cadjoint.fem.jaxfem import JaxFemBackend, tet_elastic_solve

    cells = np.asarray(inputs.cells)
    try:
        ele_type = _ELE_TYPES[cells.shape[1]]
    except KeyError:
        raise ValueError(
            f"cells must be (T, 4), (T, 8), or (T, 10); got shape {cells.shape}."
        ) from None
    traction_nodes = np.asarray(inputs.traction_nodes, dtype=np.int32)
    bcs = ElasticBCs(
        fixed_nodes=[np.asarray(inputs.fixed_nodes, dtype=np.int32)],
        traction_nodes=_split(traction_nodes, inputs.traction_offsets),
        traction_vectors=list(np.asarray(inputs.traction_vectors, dtype=np.float64)),
    )
    face_offsets = np.asarray(inputs.traction_face_offsets, dtype=np.int64)
    faces = None
    if face_offsets.size:
        faces = _split(np.asarray(inputs.traction_faces, dtype=np.int64), face_offsets)
    if ele_type == "HEX8":
        if faces is not None:
            raise ValueError("traction_faces targeting is a tet feature; HEX8 uses node sets.")
        return JaxFemBackend().elastic(
            points,
            cells,
            bcs,
            youngs=youngs,
            poisson=poisson,
            base_points=base_points,
            body_force=body_force,
        )
    return tet_elastic_solve(
        points,
        cells,
        bcs,
        youngs=youngs,
        poisson=poisson,
        ele_type=ele_type,
        base_points=base_points,
        traction_faces=faces,
        body_force=body_force,
    )


def apply(inputs: InputSchema) -> OutputSchema:
    """Forward solve (opaque to JAX tracing; runs concretely)."""
    displacement = _solve(
        inputs.points,
        inputs,
        _material(inputs.youngs, inputs.cell_youngs),
        _material(inputs.poisson, inputs.cell_poisson),
        _body_force(inputs.body_force),
        base_points=np.asarray(inputs.points, dtype=np.float64),
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
    solver is required.  The contract is identical across HEX8/TET4/TET10:
    ``points`` plus the per-element ``cell_youngs`` / ``cell_poisson`` /
    ``body_force`` arrays, which the heterogeneous formulation carries as
    internal variables.  An array left empty (the single-material default)
    never enters the traced closure, so jax hands back its zero of shape
    ``(0,)`` / ``(0, 3)``; the scalars ``youngs`` / ``poisson`` are not
    differentiable and are still rejected here.
    """
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    assert vjp_outputs == {"displacement"}, f"unexpected vjp_outputs: {vjp_outputs}"

    base_points = np.asarray(inputs.points, dtype=np.float64)
    cell_youngs = np.asarray(inputs.cell_youngs, dtype=np.float64)
    cell_poisson = np.asarray(inputs.cell_poisson, dtype=np.float64)
    force = np.asarray(inputs.body_force, dtype=np.float64)

    # Which branch the forward took is decided by the *concrete* payload, so
    # a traced parameter never has to be inspected for emptiness.
    per_element_youngs = bool(cell_youngs.size)
    per_element_poisson = bool(cell_poisson.size)
    has_body_force = bool(force.size)

    def fun(params: dict):
        youngs = (
            params.get("cell_youngs", cell_youngs)
            if per_element_youngs
            else float(np.asarray(inputs.youngs))
        )
        poisson = (
            params.get("cell_poisson", cell_poisson)
            if per_element_poisson
            else float(np.asarray(inputs.poisson))
        )
        body_force = params.get("body_force", force) if has_body_force else None
        return _solve(
            params.get("points", inputs.points),
            inputs,
            youngs,
            poisson,
            body_force,
            base_points=base_points,
        )

    params = {}
    if "points" in vjp_inputs:
        params["points"] = jnp.asarray(inputs.points, dtype=jnp.float64)
    if "cell_youngs" in vjp_inputs:
        params["cell_youngs"] = jnp.asarray(cell_youngs, dtype=jnp.float64)
    if "cell_poisson" in vjp_inputs:
        params["cell_poisson"] = jnp.asarray(cell_poisson, dtype=jnp.float64)
    if "body_force" in vjp_inputs:
        params["body_force"] = jnp.asarray(force, dtype=jnp.float64)
    unsupported = set(vjp_inputs) - set(params)
    if unsupported:
        raise ValueError(
            f"Non-differentiable inputs requested in vjp: {sorted(unsupported)}; "
            "differentiable inputs are points, cell_youngs, cell_poisson, body_force."
        )

    _, vjp_fun = jax.vjp(fun, params)
    (gradients,) = vjp_fun(jnp.asarray(cotangent_vector["displacement"], dtype=jnp.float64))
    return {key: np.asarray(value) for key, value in gradients.items()}
