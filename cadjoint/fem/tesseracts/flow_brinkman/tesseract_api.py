"""Tesseract packaging of the Brinkman-penalised steady flow solve.

The flow plugin differs from the solver plugins beside it in what crosses
the boundary.  ``thermal_jaxfem`` and ``elastic_jaxfem`` send a *mesh* --
points, cells, node sets -- because their solvers need one.  This one sends
a single array: ``chi``, the solid fraction on a fixed lattice.  There is no
mesh to ship because there is no mesh at all, and that is the whole
argument for the approach (see :mod:`cadjoint.flow.domain`).

That makes the wire contract unusually small and unusually stable.  The grid
is fixed by ``chi.shape``, so refining the study changes an array bound and
nothing else; no connectivity, no boundary node sets, no element type.

**Differentiable inputs.**  ``chi`` and ``inlet_velocity``.  ``chi`` is the
one that matters: it is where the design enters, and cadjoint's side of the
boundary produces it by sampling the scene SDF
(:func:`cadjoint.flow.sample_solid_fraction`), so ``d(pressure drop)/d(fin
tip)`` is this endpoint's ``d/d(chi)`` composed with a derivative the CAD
kernel already knows how to take.  Everything else -- Reynolds number,
penalisation strength, convergence tolerances -- is static configuration,
carried as plain scalars and baked into the traced closure.

**Why the VJP is cheap.**  ``apply`` marches thousands of pseudo-time steps
to a fixed point.  ``vector_jacobian_product`` does *not* differentiate that
march.  :func:`cadjoint.flow.steady_populations` carries a
:func:`jax.custom_vjp` whose backward pass is one matrix-free linear solve
against the converged state, so the reverse pass costs a few hundred
operator applications and holds one state rather than a trajectory of them.
The endpoint is therefore usable at grid sizes where a taped Tesseract would
simply run the host out of memory -- measured at 3456 cells, the tape needs
about 2.6 MB per step and 21 GB by the time the march has converged, against
a flat 460 MB for the adjoint.

Endpoint shape follows ``thermal_jaxfem`` exactly: ``apply`` runs
concretely, ``abstract_eval`` publishes output shapes so ``tesseract-jax``
can trace a call, and ``vector_jacobian_product`` composes :func:`jax.vjp`
over the same closure ``apply`` uses -- so the two cannot drift apart.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType


class InputSchema(BaseModel):
    """A solid fraction on a fixed lattice, plus the regime to solve it in."""

    #: Solid volume fraction in ``[0, 1]``, ``(NX, NY, NZ)``.  The duct axis
    #: is ``+Y``: inlet at ``y = 0``, outlet at ``y = NY - 1``.  This is the
    #: design.
    chi: Differentiable[Array[(None, None, None), Float64]]
    #: Inlet velocity in lattice units, ``(3,)``.  Differentiable so an
    #: operating point can be optimized alongside the geometry.
    inlet_velocity: Differentiable[Array[(3,), Float64]]
    #: Reynolds number, against ``characteristic_cells``.
    reynolds: float = 100.0
    #: The Reynolds number's length scale in cells; 0 means "use ``NZ``".
    characteristic_cells: int = 0
    #: Brinkman drag at ``chi = 1``.  See
    #: :func:`cadjoint.flow.recommended_alpha_max`.
    alpha_max: float = 200.0
    #: World volume of one cell, to make the heat-transfer integral
    #: extensive.  Purely a scale on one output.
    cell_volume: float = 1.0
    #: Cap on pseudo-time steps.
    max_steps: int = 20000
    #: Relative per-step change below which the march stops.
    tol: float = 1e-9
    #: ``"gmres"`` or ``"fixed_point"``.
    adjoint_solver: str = "gmres"
    #: Convergence tolerance for the adjoint solve.
    adjoint_tol: float = 1e-10
    #: Iteration cap for the adjoint solve.
    adjoint_max_steps: int = 2000
    #: Krylov subspace size for ``"gmres"``.
    adjoint_restart: int = Field(default=40)


class OutputSchema(BaseModel):
    """The two scalars a cooling study reads off the converged flow."""

    #: Inlet-to-outlet pressure difference, lattice units.  What the fan pays.
    pressure_drop: Differentiable[Array[(), Float64]]
    #: ``int chi |u|`` -- air in motion where the metal is.  What it buys.
    heat_transfer: Differentiable[Array[(), Float64]]


def _config(inputs: InputSchema, shape: tuple[int, int, int]):
    """Build the :class:`~cadjoint.flow.FlowConfig` these inputs describe.

    Args:
        inputs: The validated input schema.
        shape: ``(NX, NY, NZ)``, taken from ``chi``.

    Returns:
        The frozen configuration.
    """
    from cadjoint.flow import FlowConfig, SteadyOptions

    return FlowConfig(
        shape=shape,
        inlet_speed=float(np.linalg.norm(np.asarray(inputs.inlet_velocity, dtype=np.float64))),
        reynolds=float(inputs.reynolds),
        characteristic_cells=int(inputs.characteristic_cells) or None,
        alpha_max=float(inputs.alpha_max),
        steady=SteadyOptions(
            max_steps=int(inputs.max_steps),
            tol=float(inputs.tol),
            adjoint_solver=str(inputs.adjoint_solver),
            adjoint_tol=float(inputs.adjoint_tol),
            adjoint_max_steps=int(inputs.adjoint_max_steps),
            adjoint_restart=int(inputs.adjoint_restart),
        ),
    )


def _solve(chi, inlet_velocity, inputs: InputSchema):
    """Converge the flow and read off both objectives.

    Traceable end to end: the fixed-point adjoint lives inside
    :func:`cadjoint.flow.solve`, so :func:`jax.vjp` over this closure is
    the implicit-function-theorem gradient rather than a taped march.

    Args:
        chi: Solid fraction, ``(NX, NY, NZ)``.
        inlet_velocity: Inlet velocity, ``(3,)``.
        inputs: The validated input schema, for the static settings.

    Returns:
        ``(pressure_drop, heat_transfer)`` as scalars.
    """
    from cadjoint.flow import solve

    shape = tuple(int(n) for n in np.shape(chi))
    result = solve(
        chi,
        _config(inputs, shape),
        inlet_velocity=inlet_velocity,
        cell_volume=float(inputs.cell_volume),
    )
    return result.pressure_drop, result.heat_transfer


def apply(inputs: InputSchema) -> OutputSchema:
    """Forward solve: march to the fixed point, report the two scalars."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    drop, transfer = _solve(
        jnp.asarray(inputs.chi, dtype=jnp.float64),
        jnp.asarray(inputs.inlet_velocity, dtype=jnp.float64),
        inputs,
    )
    return OutputSchema(
        pressure_drop=np.asarray(drop, dtype=np.float64),
        heat_transfer=np.asarray(transfer, dtype=np.float64),
    )


def abstract_eval(abstract_inputs):
    """Output shapes from input shapes (lets tesseract-jax trace the call).

    Both outputs are scalars whatever the grid, so this does not consult
    ``chi``'s shape at all.

    Args:
        abstract_inputs: The shape/dtype stand-in for :class:`InputSchema`.

    Returns:
        The output shape/dtype mapping.
    """
    del abstract_inputs
    return {
        "pressure_drop": ShapeDType(shape=(), dtype="float64"),
        "heat_transfer": ShapeDType(shape=(), dtype="float64"),
    }


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict,
):
    """Adjoint gradients for the differentiable inputs.

    One :func:`jax.vjp` over the same closure :func:`apply` calls.  The
    fixed-point rule inside it means the reverse pass is a single linear
    solve at the converged state, so the cost here does not grow with how
    many pseudo-time steps the forward march happened to take.

    Args:
        inputs: The validated input schema.
        vjp_inputs: Which inputs to differentiate; a subset of ``chi`` and
            ``inlet_velocity``.
        vjp_outputs: Which outputs the cotangents belong to.
        cotangent_vector: Cotangents keyed by output name; a missing output
            contributes nothing.

    Returns:
        Gradients keyed by input name, as NumPy arrays.

    Raises:
        ValueError: On an unknown differentiable input or output name.
    """
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    known_outputs = {"pressure_drop", "heat_transfer"}
    unknown = set(vjp_outputs) - known_outputs
    if unknown:
        raise ValueError(
            f"unknown vjp_outputs {sorted(unknown)}; this Tesseract outputs "
            f"{sorted(known_outputs)}."
        )

    chi = jnp.asarray(inputs.chi, dtype=jnp.float64)
    inlet = jnp.asarray(inputs.inlet_velocity, dtype=jnp.float64)

    def fun(params: dict):
        drop, transfer = _solve(params.get("chi", chi), params.get("inlet_velocity", inlet), inputs)
        return {"pressure_drop": drop, "heat_transfer": transfer}

    params = {}
    if "chi" in vjp_inputs:
        params["chi"] = chi
    if "inlet_velocity" in vjp_inputs:
        params["inlet_velocity"] = inlet
    unsupported = set(vjp_inputs) - set(params)
    if unsupported:
        raise ValueError(
            f"Non-differentiable inputs requested in vjp: {sorted(unsupported)}; "
            "differentiable inputs are chi, inlet_velocity."
        )

    primal, vjp_fun = jax.vjp(fun, params)
    cotangents = {
        name: jnp.asarray(cotangent_vector[name], dtype=jnp.float64)
        if name in cotangent_vector
        else jnp.zeros_like(value)
        for name, value in primal.items()
    }
    (gradients,) = vjp_fun(cotangents)
    return {key: np.asarray(value, dtype=np.float64) for key, value in gradients.items()}
