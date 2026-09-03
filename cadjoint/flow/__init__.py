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

Since the coupling landed, the package also carries the *thermal* half:
:mod:`~cadjoint.flow.energy` solves the steady energy equation on the same
lattice -- conduction in the metal and advection-diffusion in the air as
one equation with a variable coefficient -- and
:class:`~cadjoint.flow.FlowStudy` declares the pair in a scene the way
``ThermalStudy`` declares a conduction solve:

    >>> from cadjoint.fem import Nodes
    >>> from cadjoint.flow import FlowStudy, HeatSource, Inlet, Outlet, Walls
    >>> study = FlowStudy(                                  # doctest: +SKIP
    ...     name="sink-cooling",
    ...     resolution=(16, 32, 16),
    ...     bcs=[Inlet(velocity=0.02), Outlet(), Walls(),
    ...          HeatSource(Nodes.sphere([0, 0, -0.2], 0.2), power=1.0)],
    ... )
    >>> study.solve(sdf).peak_temperature                    # doctest: +SKIP

Packaged for the plugin registry as the ``flow_solver`` kind by
``cadjoint/fem/tesseracts/flow_brinkman``.  ``research/flow-solver.md``
carries the verification numbers, and what is still not true.
"""

from __future__ import annotations

from cadjoint.flow.domain import PROFILES, FlowGrid, sample_solid_fraction, solid_fraction
from cadjoint.flow.energy import (
    AIR_PRANDTL,
    SCHEMES,
    EnergyConfig,
    bulk_outlet_temperature,
    conductivity,
    energy_imbalance,
    mean_temperature,
    peak_temperature,
    solve_temperature,
    thermal_resistance,
)
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
from cadjoint.flow.regions import REGION_KINDS, region_mask
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
from cadjoint.flow.study import (
    BC_KINDS,
    FLOW_STUDY_KIND,
    FORCED_CONVECTION_RICHARDSON,
    FlowStudy,
    FlowStudyResult,
    HeatSource,
    HeldTemperature,
    Inlet,
    Outlet,
    Walls,
)

__all__ = [
    "AIR_PRANDTL",
    "BC_KINDS",
    "C",
    "CS2",
    "DEFAULT_ALPHA_MAX",
    "EnergyConfig",
    "FLOW_STUDY_KIND",
    "FORCED_CONVECTION_RICHARDSON",
    "FlowConfig",
    "FlowGrid",
    "FlowResult",
    "FlowStudy",
    "FlowStudyResult",
    "HeatSource",
    "HeldTemperature",
    "Inlet",
    "OMEGA_CEILING",
    "OPP",
    "Outlet",
    "PROFILES",
    "Q",
    "REGION_KINDS",
    "SCHEMES",
    "SteadyOptions",
    "W",
    "Walls",
    "bounce_back_mask",
    "bulk_outlet_temperature",
    "conductivity",
    "convergence",
    "duct_walls",
    "energy_imbalance",
    "equilibrium",
    "fields",
    "heat_transfer_proxy",
    "initial_populations",
    "iterate_to_steady",
    "macroscopic",
    "mean_temperature",
    "omega_from_viscosity",
    "peak_temperature",
    "pressure",
    "pressure_drop",
    "recommended_alpha_max",
    "region_mask",
    "residual_history",
    "sample_solid_fraction",
    "solid_fraction",
    "solve",
    "solve_temperature",
    "steady_populations",
    "step",
    "step_for",
    "stream",
    "thermal_resistance",
    "unrolled_populations",
    "viscosity_from_omega",
]
