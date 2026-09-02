"""Differentiable steady flow on a fixed grid, penalised by an SDF.

A prototype answering one question: can a flow solve join cadjoint's
gradient path on the same terms the thermal solve already does -- design
parameters in, a scalar out, a reverse-mode derivative between them -- with
no mesh generation anywhere in the loop?

The answer this package implements is Brinkman-penalised lattice Boltzmann.
The design never becomes a mesh; it becomes a *field*.  The scene's SDF is
sampled on a fixed lattice and squashed to a solid fraction
``chi = sigmoid(-f(x)/eps)`` (:mod:`~cadjoint.flow.domain`), which enters
the momentum equation as a drag ``-alpha_max chi u``
(:mod:`~cadjoint.flow.lbm`).  Because ``chi`` is smooth in the design and
the grid never moves, the whole chain from a sketch point to a pressure
drop is one differentiable JAX expression -- which is precisely what a
body-fitted mesh cannot be.

The solve is a fixed point, and its gradient is taken by the implicit
function theorem rather than by taping the iteration
(:mod:`~cadjoint.flow.steady`): one adjoint solve at the converged state,
memory independent of how many pseudo-time steps convergence took.

    >>> from cadjoint.flow import FlowConfig, FlowGrid, sample_solid_fraction, solve
    >>> grid = FlowGrid(shape=(32, 64, 32), origin=(-1.0, -0.8, -0.1), size=(2.0, 1.6, 1.0))
    >>> chi = sample_solid_fraction(sdf, grid)              # doctest: +SKIP
    >>> result = solve(chi, FlowConfig(shape=grid.shape))   # doctest: +SKIP
    >>> result.pressure_drop                                # doctest: +SKIP

Packaged for the plugin registry as the ``flow_solver`` kind by
``cadjoint/fem/tesseracts/flow_brinkman``.  What it is not yet: coupled to
the thermal study.  ``research/flow-solver.md`` sets out what that would
take and what it would cost.
"""

from __future__ import annotations

from cadjoint.flow.domain import PROFILES, FlowGrid, sample_solid_fraction, solid_fraction
from cadjoint.flow.lattice import CS2, OPP, C, Q, W, omega_from_viscosity, viscosity_from_omega
from cadjoint.flow.lbm import (
    bounce_back_mask,
    duct_walls,
    equilibrium,
    initial_populations,
    macroscopic,
    step,
    stream,
)
from cadjoint.flow.objectives import fields, heat_transfer_proxy, pressure, pressure_drop
from cadjoint.flow.solver import (
    DEFAULT_ALPHA_MAX,
    OMEGA_CEILING,
    FlowConfig,
    FlowResult,
    convergence,
    recommended_alpha_max,
    solve,
    step_for,
)
from cadjoint.flow.steady import (
    SteadyOptions,
    iterate_to_steady,
    residual_history,
    steady_populations,
    unrolled_populations,
)

__all__ = [
    "C",
    "DEFAULT_ALPHA_MAX",
    "CS2",
    "OMEGA_CEILING",
    "OPP",
    "PROFILES",
    "Q",
    "W",
    "FlowConfig",
    "FlowGrid",
    "FlowResult",
    "SteadyOptions",
    "bounce_back_mask",
    "convergence",
    "duct_walls",
    "equilibrium",
    "fields",
    "heat_transfer_proxy",
    "initial_populations",
    "iterate_to_steady",
    "macroscopic",
    "omega_from_viscosity",
    "pressure",
    "pressure_drop",
    "recommended_alpha_max",
    "residual_history",
    "sample_solid_fraction",
    "solid_fraction",
    "solve",
    "steady_populations",
    "step",
    "step_for",
    "stream",
    "unrolled_populations",
    "viscosity_from_omega",
]
