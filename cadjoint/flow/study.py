"""A conjugate heat-transfer study a scene declares, like ``ThermalStudy``.

``ThermalStudy`` conducts heat through metal and calls whatever is outside
it a boundary condition.  For a heat sink that is the whole difficulty:
``scenes/starter.py`` holds the fin field at ambient with a Dirichlet patch,
an infinitely capable sink that makes taller fins look free and downstream
fins look as good as upstream ones.  A :class:`FlowStudy` replaces that
fiction with air that has to be pushed through the fins, warms up as it
goes, and stops cooling once it has.

**What it solves.**  Two problems on one fixed lattice, in this order:

1. **Momentum** — the Brinkman-penalised D3Q19 lattice Boltzmann solve of
   :mod:`cadjoint.flow.solver`, whose only design input is the solid
   fraction ``chi`` sampled from the scene's SDF.  Out comes a velocity
   field and the pressure drop the fan pays.
2. **Energy** — the single-domain finite-volume solve of
   :mod:`cadjoint.flow.energy`, which conducts through the metal and
   advects through the air *in one equation with a variable coefficient*,
   so temperature and heat flux are continuous at the interface by
   construction rather than by an interface condition.

**The coupling is one-way, and in this model that is exact rather than an
approximation.**  The momentum solve's inputs are ``chi``, the inlet
velocity, and a viscosity; not one of them is a function of temperature.
Nothing computed by step 2 can therefore change step 1, and iterating
between them to a fixed point would converge in one pass by definition.
What makes conjugate heat transfer genuinely two-way in other codes is
physics this model does not carry: buoyancy (density varying with
temperature, which drives natural convection) and temperature-dependent
viscosity and conductivity.  The condition for ignoring buoyancy is a small
Richardson number ``Ri = g beta dT L / U^2``; :attr:`FlowStudy.richardson`
computes it from the study's own numbers so the assumption can be checked
rather than assumed, and :meth:`FlowStudyResult.warnings` reports it when it
is not small.  ``research/flow-solver.md`` carries the measurement.

**What the study reads out.**  ``peak_temperature`` and
``mean_temperature`` over the solid, ``bulk_outlet_temperature`` (the
mixing-cup temperature of the air leaving), ``thermal_resistance``
(``(T_peak - T_inlet) / power``, the number a data sheet quotes), and
``pressure_drop``.  The last one is not decoration: a sink that maximises
contact with moving air by filling the duct with metal strangles the fan
driving it, so an optimiser given only a temperature walks into a solid
block.

**All of it is differentiable in the geometry.**  The momentum solve
carries the fixed-point adjoint of :mod:`cadjoint.flow.steady` and the
energy solve carries :func:`jax.lax.custom_linear_solve`'s transposed
solve, so a scalar read off a :class:`FlowStudyResult` differentiates back
through both, through ``chi``, and into the scene's parameters — without
either solver knowing the other exists.  ``tests/flow/test_study.py``
checks that composition against a finite difference.

Example::

    duct = FlowStudy(
        name="sink-cooling",
        resolution=(20, 40, 20),
        bounds=(-1.0, -1.4, -0.3),
        size=(2.0, 2.8, 1.4),
        reynolds=60.0,
        bcs=[
            Inlet(velocity=0.02, temperature=0.0),
            Outlet(),
            Walls(),
            HeatSource(Nodes.box([-0.1, -0.1, -0.25], [0.1, 0.1, -0.1]), power=1.0),
        ],
    )
    result = duct.solve(scene_sdf)
    result.peak_temperature, result.pressure_drop

**Selections are volumetric here.**  A lattice has no boundary surface, so
``Nodes.box`` and friends select every *cell centre* satisfying the
criterion rather than the boundary nodes a mesh study would take; see
:mod:`cadjoint.flow.regions`, which also refuses ``Nodes.side`` and
``Nodes.predicate`` with the alternative named.
"""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from cadjoint.flow.domain import PROFILES, FlowGrid, sample_solid_fraction
from cadjoint.flow.energy import (
    AIR_PRANDTL,
    SCHEMES,
    EnergyConfig,
    bulk_outlet_temperature,
    energy_imbalance,
    mean_temperature,
    peak_temperature,
    solve_temperature,
)
from cadjoint.flow.lbm import duct_walls
from cadjoint.flow.precision import double_precision
from cadjoint.flow.regions import region_mask
from cadjoint.flow.solver import DEFAULT_ALPHA_MAX, FlowConfig, solve
from cadjoint.flow.steady import SteadyOptions

__all__ = [
    "FLOW_STUDY_KIND",
    "FlowStudy",
    "FlowStudyResult",
    "HeatSource",
    "HeldTemperature",
    "Inlet",
    "Outlet",
    "Walls",
]

#: The ``kind`` a :meth:`FlowStudy.describe` payload carries.
#:
#: A module constant rather than a :class:`~cadjoint.enums.StudyKind`
#: member, and the distinction is the point.  ``StudyKind`` is the
#: vocabulary of studies the viewer can *create and edit* through its patch
#: endpoints -- its members drive four tables in
#: ``cadjoint/viewer/patch/studies.py`` that say how to write a study's
#: constructor and which boundary conditions it takes -- and none of that
#: exists for a flow study.  Adding a member would make ``add_study``
#: advertise ``flow`` as a kind it can write and then refuse to write it.
#:
#: What a scene *declares* is a wider vocabulary than what the viewer can
#: author, so ``StudyPayload.kind`` carries this string and the enum does
#: not.  That is what lets a scene declaring a flow study compile, render
#: and serialize its study today, while the Studies window's own editing
#: affordances stay honest about covering two kinds.
FLOW_STUDY_KIND = "flow"

#: Boundary conditions a flow study accepts.
BC_KINDS = ("inlet", "outlet", "walls", "heat_source", "held_temperature")

#: Richardson number above which buoyancy stops being negligible and the
#: one-way coupling stops being exact.  The conventional forced-convection
#: threshold: below 0.1 buoyancy contributes under about 10% of the
#: momentum, above 10 the flow is buoyancy-driven and this solver is the
#: wrong tool.
FORCED_CONVECTION_RICHARDSON = 0.1


def _triplet(value: Any, label: str) -> tuple[float, float, float]:
    """Three finite floats, or a message naming the argument."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must contain three finite numbers, got {value!r}.")
    return (float(array[0]), float(array[1]), float(array[2]))


def _selection(nodes: Any, bc_kind: str) -> Any:
    """Check a boundary condition was given a node selection."""
    from cadjoint.fem.selection import NodeSelection

    if not isinstance(nodes, NodeSelection):
        raise ValueError(
            f"{bc_kind} takes a node selection, got {type(nodes).__name__}. "
            "Build one via Nodes.box/sphere/halfspace/cylinder (from cadjoint.fem "
            "import Nodes); on a lattice it selects cell centres volumetrically."
        )
    return nodes


@dataclass(frozen=True)
class Inlet:
    """Air entering the duct at the ``y = 0`` plane.

    Attributes:
        velocity: Inlet velocity in lattice units (cells per step) -- a
            scalar, meaning that speed along ``+Y``, or a ``(3,)`` vector.
            Keep the magnitude well under the lattice sound speed
            ``1/sqrt(3)``; 0.05 or below keeps the compressibility error
            under a percent.  Exactly zero is legal and means *no flow*:
            the momentum solve is skipped and the study becomes pure
            conduction, which is how the conjugate path is checked against
            :class:`~cadjoint.fem.study.ThermalStudy`.
        temperature: Temperature of the air arriving.  Conventionally 0,
            because only differences from it mean anything -- every
            temperature this study reports is a rise above the inlet.
    """

    velocity: Any = 0.02
    temperature: float = 0.0

    def __post_init__(self) -> None:
        vector = (
            (0.0, float(self.velocity), 0.0)
            if np.ndim(self.velocity) == 0
            else _triplet(self.velocity, "Inlet velocity")
        )
        speed = float(np.linalg.norm(vector))
        if not np.isfinite(speed):
            raise ValueError(f"Inlet velocity must be finite, got {self.velocity!r}.")
        if speed >= 3.0**-0.5:
            raise ValueError(
                f"Inlet velocity {vector} has magnitude {speed:.4g}, at or above the "
                "lattice sound speed 1/sqrt(3) = 0.5774. Lattice Boltzmann is a "
                "low-Mach method and this is not a small correction -- it is outside "
                "the model. Use 0.05 or below (compressibility error under a percent) "
                "and scale the Reynolds number instead."
            )
        object.__setattr__(self, "velocity", vector)
        object.__setattr__(self, "temperature", float(self.temperature))

    @property
    def speed(self) -> float:
        """Magnitude of the inlet velocity, in cells per step."""
        return float(np.linalg.norm(self.velocity))

    @property
    def serializable(self) -> bool:
        """Always: an inlet is a plane and three numbers, not a selection."""
        return True

    def describe(self) -> dict[str, Any]:
        """JSON-ready description."""
        return {
            "type": "inlet",
            "velocity": list(self.velocity),
            "temperature": self.temperature,
        }


@dataclass(frozen=True)
class Outlet:
    """Air leaving at the ``y = NY-1`` plane, against a fixed pressure.

    Takes no arguments, and that is a statement rather than an omission.
    The momentum outlet anchors density at 1 and extrapolates velocity,
    which is what removes the undamped mean-density mode a velocity inlet
    with a zero-gradient outlet would leave free (see
    :func:`cadjoint.flow.lbm._apply_inlet_outlet`).  The energy outlet
    carries the cell's own temperature out with the flow and imposes no
    diffusive gradient.  Neither has a number to set: a duct that exhausts
    to the room is fully specified by the fact that it does.
    """

    @property
    def serializable(self) -> bool:
        """Always: the outlet carries no arguments to fail to serialize."""
        return True

    def describe(self) -> dict[str, Any]:
        """JSON-ready description."""
        return {"type": "outlet"}


@dataclass(frozen=True)
class Walls:
    """The duct's lateral walls -- the ``x`` and ``z`` extremes of the lattice.

    No-slip always: the momentum solve reflects populations off them with
    halfway bounce-back whether or not this condition is declared, because
    a duct without walls is not a duct.  What this condition sets is what
    they do *thermally*.

    Attributes:
        temperature: ``None`` (the default) for adiabatic walls -- the
            right choice for a duct that is carrying air past a part rather
            than acting as a heat exchanger itself, and the only choice
            under which the energy balance closes against the injected
            power alone.  A number holds them at that temperature, which
            gives the walls their own path for heat to leave by and is how
            a Nusselt-number check against a duct-flow correlation is set
            up.
    """

    temperature: float | None = None

    def __post_init__(self) -> None:
        if self.temperature is not None:
            value = float(self.temperature)
            if not np.isfinite(value):
                raise ValueError(f"Walls temperature must be finite, got {self.temperature!r}.")
            object.__setattr__(self, "temperature", value)

    @property
    def serializable(self) -> bool:
        """Always: a wall condition is one optional number."""
        return True

    def describe(self) -> dict[str, Any]:
        """JSON-ready description."""
        return {"type": "walls", "temperature": self.temperature}


@dataclass(frozen=True)
class HeatSource:
    """Total power injected over the cells a selection covers.

    ``power`` is the *total*, not a density: it is spread evenly over
    however many cells the region resolves to, so refining the lattice does
    not change how much heat the die puts out.  That is the opposite
    convention to :class:`~cadjoint.fem.study.HeatFlux`, which is a flux per
    area, and it is the right one here because the region is a volume and
    because ``thermal_resistance`` divides by exactly this number.

    Attributes:
        nodes: The region, as a :class:`~cadjoint.fem.selection.NodeSelection`.
            Resolved volumetrically against cell centres.
        power: Total power in lattice units.  May be negative (a cold plate).
    """

    nodes: Any
    power: float = 1.0

    def __post_init__(self) -> None:
        _selection(self.nodes, "HeatSource")
        value = float(self.power)
        if not np.isfinite(value):
            raise ValueError(f"HeatSource power must be finite, got {self.power!r}.")
        object.__setattr__(self, "power", value)

    @property
    def serializable(self) -> bool:
        """Whether :meth:`describe` round-trips (false for predicates)."""
        return self.nodes.serializable

    def describe(self) -> dict[str, Any]:
        """JSON-ready description."""
        return {"type": "heat_source", "nodes": self.nodes.describe(), "power": self.power}


@dataclass(frozen=True)
class HeldTemperature:
    """Cells pinned at a temperature, whatever the flow does around them.

    The lattice counterpart of :class:`~cadjoint.fem.study.Dirichlet`, and
    like it a modelling shortcut with teeth: a held region is an infinite
    heat source or sink, so a study that pins the fins is back to the
    fiction a flow study exists to remove.  Its honest uses are a cold plate
    with a real chiller behind it, and verification -- the analytic
    advection-diffusion column in ``tests/flow/test_energy.py`` is set up
    with one.

    Attributes:
        nodes: The region, resolved volumetrically against cell centres.
        value: The temperature to hold, relative to the inlet.
    """

    nodes: Any
    value: float = 0.0

    def __post_init__(self) -> None:
        _selection(self.nodes, "HeldTemperature")
        number = float(self.value)
        if not np.isfinite(number):
            raise ValueError(f"HeldTemperature value must be finite, got {self.value!r}.")
        object.__setattr__(self, "value", number)

    @property
    def serializable(self) -> bool:
        """Whether :meth:`describe` round-trips (false for predicates)."""
        return self.nodes.serializable

    def describe(self) -> dict[str, Any]:
        """JSON-ready description."""
        return {
            "type": "held_temperature",
            "nodes": self.nodes.describe(),
            "value": self.value,
        }


_BC_TYPES = (Inlet, Outlet, Walls, HeatSource, HeldTemperature)


@dataclass(frozen=True)
class FlowStudyResult:
    """Everything one conjugate solve produced, fields and scalars alike.

    The scalars are JAX arrays rather than floats on purpose: they are what
    an objective is built from, and converting them to Python floats here
    would sever the gradient path the whole study exists to provide.

    Attributes:
        name: The study's name.
        kind: :data:`FLOW_STUDY_KIND`.
        grid: The lattice the solve ran on.
        chi: Solid fraction, ``(NX, NY, NZ)``.
        velocity: Converged velocity in lattice units, ``(3, NX, NY, NZ)``;
            exactly zero when the inlet speed is zero.
        density: Converged density, ``(NX, NY, NZ)``; exactly 1 with no flow.
        temperature: Temperature rise above the inlet, ``(NX, NY, NZ)``.
        pressure_drop: Inlet-to-outlet pressure difference, lattice units.
        peak_temperature: Hottest cell in the solid.
        mean_temperature: ``chi``-weighted mean over the solid.
        bulk_outlet_temperature: Mixing-cup temperature of the air leaving.
        thermal_resistance: ``(T_peak - T_inlet) / power``.
        power: Total injected power, the sum over every heat source.
        energy_imbalance: What the solve failed to conserve, as a fraction
            of ``power``; ``None`` when no power was injected or when a
            held temperature or an isothermal wall gives heat another way
            out, in which case the balance is not the study's to close.
        reynolds: The Reynolds number solved at.
        peclet_cell: Largest cell Peclet number in the fluid -- how far the
            energy discretisation is from the diffusive regime.
        richardson: ``g beta dT L / U^2``, the buoyancy the one-way
            coupling neglects; see this module's docstring.
    """

    name: str
    kind: str
    grid: FlowGrid
    chi: jax.Array
    velocity: jax.Array
    density: jax.Array
    temperature: jax.Array
    pressure_drop: jax.Array
    peak_temperature: jax.Array
    mean_temperature: jax.Array
    bulk_outlet_temperature: jax.Array
    thermal_resistance: jax.Array
    power: float
    energy_imbalance: jax.Array | None
    reynolds: float
    peclet_cell: float
    richardson: float

    def warnings(self) -> list[str]:
        """Everything about this solve a reader should not have to derive.

        Returned rather than logged so a test can assert on them and a
        caller can decide how loud to be.  Empty means every assumption the
        study makes held on the numbers it actually ran.

        Returns:
            Human-readable strings, one per assumption that did not hold.
        """
        notes: list[str] = []
        if not bool(np.isfinite(np.asarray(self.peak_temperature))):
            notes.append(
                "The temperature field is not finite, which means the momentum march "
                "diverged rather than converging. The BGK relaxation rate is the usual "
                "cause and its stable ceiling falls with the lattice size, so a coarse "
                "duct needs more margin than OMEGA_CEILING allows: lower the Reynolds "
                "number, lower the inlet speed, or refine the lattice."
            )
        if self.richardson > FORCED_CONVECTION_RICHARDSON:
            notes.append(
                f"Richardson number {self.richardson:.3g} exceeds "
                f"{FORCED_CONVECTION_RICHARDSON}: buoyancy is not negligible at this "
                "temperature rise and inlet speed, so the one-way flow -> thermal "
                "coupling is no longer exact. Raise the inlet speed, lower the power, "
                "or treat the answer as a lower bound on the temperature."
            )
        if self.reynolds > 2300.0:
            notes.append(
                f"Reynolds number {self.reynolds:.0f} is above the ~2300 duct "
                "transition: the flow would be turbulent and this solver models no "
                "turbulence, so it reports the laminar answer, which understates "
                "mixing and therefore overstates the temperature."
            )
        if self.peclet_cell > 2.0:
            notes.append(
                f"Largest cell Peclet number is {self.peclet_cell:.2g}. The "
                "exponential scheme stays nodally exact in one dimension at any "
                "Peclet number, but the thermal boundary layer is thinner than a "
                "cell here, so its gradient -- and any Nusselt number read from it "
                "-- is under-resolved. Refine the lattice to check."
            )
        if self.energy_imbalance is not None and abs(float(self.energy_imbalance)) > 1e-6:
            notes.append(
                f"Energy balance closes to {float(self.energy_imbalance):.3g} of the "
                "injected power rather than to round-off. Summing the discrete "
                "equations makes this an exact identity of the assembled system, so "
                "a residual here is the linear solve, not the physics: restarted "
                "GMRES stagnates when energy_restart is too small for the "
                "conductivity ratio, and returns a badly wrong temperature without "
                "failing. Raise energy_restart (and energy_max_steps) and check that "
                "this number drops."
            )
        return notes


@dataclass
class FlowStudy:
    """Declarative conjugate heat transfer in forced convection.

    Declared in a scene exactly like :class:`~cadjoint.fem.study.ThermalStudy`
    -- constructing one inside :func:`~cadjoint.fem.study.capture_studies`
    registers it -- but it meshes nothing.  The lattice is fixed and the
    design enters as a field on it, which is why the whole solve stays one
    differentiable JAX expression.

    Attributes:
        name: Study identifier, unique within a scene program.
        resolution: Lattice size, an int or an ``(NX, NY, NZ)`` triplet.
            The duct axis is ``+Y``.
        bounds: World coordinates of the lattice's minimum corner.
        size: World extent of the lattice along each axis.
        bcs: :class:`Inlet`, :class:`Outlet`, :class:`Walls`,
            :class:`HeatSource` and :class:`HeldTemperature` conditions.
            An :class:`Inlet` is required; the rest have defaults.
        reynolds: Reynolds number against ``characteristic_cells``, which
            together with the inlet speed fixes the lattice viscosity.
        characteristic_cells: The Reynolds length scale in cells; ``None``
            uses the duct's ``NZ``.
        viscosity: Kinematic viscosity in lattice units, overriding
            ``reynolds``.  Required when the inlet speed is zero, because
            a Reynolds number does not define a viscosity without a speed.
        prandtl: Ratio of momentum to thermal diffusivity; the default is
            air at 300 K.
        conductivity_ratio: ``k_solid / k_fluid``.  Aluminium in air is
            about 8000; see :class:`~cadjoint.flow.energy.EnergyConfig` on
            why a few hundred is usually the better working point.
        alpha_max: Brinkman drag at ``chi = 1``.
        epsilon: Interface half-width in world units; ``None`` uses
            :meth:`~cadjoint.flow.FlowGrid.suggested_epsilon`.
        profile: How signed distance is squashed to a solid fraction; see
            :func:`~cadjoint.flow.solid_fraction`.
        scheme: Convection-diffusion blend, ``"exponential"`` or
            ``"upwind"``.
        expansion: Thermal expansion coefficient ``beta`` in 1/K, used only
            to report the :attr:`richardson` number the one-way coupling
            neglects.  The default is an ideal gas at 300 K.
        steady: Convergence settings for the momentum solve and its adjoint.
        energy_tol: Relative residual the energy solve stops at.
        energy_max_steps: Iteration cap for the energy solve.
        energy_restart: Krylov subspace size for the energy solve.  Raise
            it with ``conductivity_ratio``; too small a subspace makes
            restarted GMRES stagnate and report a badly wrong temperature
            without failing.  See
            :class:`~cadjoint.flow.energy.EnergyConfig`.
        domain: Optional SDF restricting which part of the scene becomes
            solid; ``None`` penalises the whole scene.
        last_result: The most recent :class:`FlowStudyResult`.

    Raises:
        ValueError: On a missing or duplicated boundary condition, a
            resolution the lattice cannot use, a viscosity that cannot be
            derived, or an unknown profile or scheme.
    """

    name: str
    resolution: Any = None
    _: KW_ONLY
    bounds: Any = None
    size: Any = None
    bcs: list[Any] = field(default_factory=list)
    reynolds: float = 100.0
    characteristic_cells: int | None = None
    viscosity: float | None = None
    prandtl: float = AIR_PRANDTL
    conductivity_ratio: float = 200.0
    alpha_max: float = DEFAULT_ALPHA_MAX
    epsilon: float | None = None
    profile: str = "smootherstep"
    scheme: str = "exponential"
    expansion: float = 1.0 / 300.0
    steady: SteadyOptions = field(default_factory=SteadyOptions)
    energy_tol: float = 1e-10
    energy_max_steps: int = 4000
    energy_restart: int = 60
    domain: Any = None
    last_result: FlowStudyResult | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        from cadjoint.fem.study import register_study

        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Flow study needs a non-empty name.")
        self._validate_resolution()
        self._validate_bcs()
        if self.profile not in PROFILES:
            raise ValueError(f"profile must be one of {PROFILES}; got {self.profile!r}.")
        if self.scheme not in SCHEMES:
            raise ValueError(f"scheme must be one of {SCHEMES}; got {self.scheme!r}.")
        if not self.prandtl > 0.0:
            raise ValueError(f"prandtl must be positive, got {self.prandtl}.")
        if self.domain is not None and not callable(self.domain):
            raise TypeError(
                f"domain must be an SDF object or a callable field, "
                f"got {type(self.domain).__name__}."
            )
        # Touch the derived viscosity now so an under-specified study fails
        # at its declaration rather than a minute into a march.
        self._viscosity()
        register_study(self)

    # ── validation ──────────────────────────────────────────────────────

    def _validate_resolution(self) -> None:
        """Resolve ``resolution``/``bounds``/``size`` into a usable lattice."""
        if self.resolution is None:
            raise ValueError(
                "Flow study needs a resolution: an int or an (NX, NY, NZ) triplet of "
                "lattice cells, with the duct axis along +Y."
            )
        counts = (
            (int(self.resolution),) * 3
            if np.ndim(self.resolution) == 0
            else tuple(int(count) for count in self.resolution)
        )
        if len(counts) != 3 or any(count < 4 for count in counts):
            raise ValueError(
                f"resolution must be three cell counts of at least 4, got {self.resolution!r}. "
                "The x and z extremes are duct walls and the y extremes are the inlet and "
                "outlet planes, so anything smaller has no interior left to solve in."
            )
        self.resolution = counts
        self.bounds = _triplet(
            self.bounds if self.bounds is not None else (-1.0, -1.0, -1.0), "bounds"
        )
        self.size = _triplet(self.size if self.size is not None else (2.0, 2.0, 2.0), "size")
        if any(extent <= 0.0 for extent in self.size):
            raise ValueError(f"size must be positive along every axis, got {self.size}.")

    def _validate_bcs(self) -> None:
        """Check the boundary conditions are a complete, non-contradictory set."""
        for bc in self.bcs:
            if not isinstance(bc, _BC_TYPES):
                names = ", ".join(cls.__name__ for cls in _BC_TYPES)
                raise ValueError(
                    f"Flow study accepts boundary conditions of type {names}; got "
                    f"{type(bc).__name__}. Mesh-study conditions (Dirichlet, HeatFlux, "
                    "Fixed, Traction) act on boundary faces of a mesh, and a flow "
                    "lattice has none -- HeldTemperature and HeatSource are their "
                    "volumetric counterparts."
                )
        for cls in (Inlet, Outlet, Walls):
            found = [bc for bc in self.bcs if isinstance(bc, cls)]
            if len(found) > 1:
                raise ValueError(
                    f"Flow study declares {len(found)} {cls.__name__} conditions; the duct "
                    "has one inlet plane, one outlet plane and one set of lateral walls, "
                    "so at most one of each is meaningful."
                )
        if not any(isinstance(bc, Inlet) for bc in self.bcs):
            raise ValueError(
                "Flow study needs an Inlet: it is what drives the flow and what sets the "
                "temperature everything else is measured against. Inlet(velocity=0.0) is "
                "legal and means no flow -- the pure-conduction case."
            )

    # ── derived quantities ──────────────────────────────────────────────

    @property
    def inlet(self) -> Inlet:
        """The declared inlet condition."""
        return next(bc for bc in self.bcs if isinstance(bc, Inlet))

    @property
    def outlet(self) -> Outlet:
        """The declared outlet condition, or the default one."""
        return next((bc for bc in self.bcs if isinstance(bc, Outlet)), Outlet())

    @property
    def walls(self) -> Walls:
        """The declared wall condition, or adiabatic walls."""
        return next((bc for bc in self.bcs if isinstance(bc, Walls)), Walls())

    @property
    def grid(self) -> FlowGrid:
        """The lattice this study samples the scene on."""
        return FlowGrid(shape=self.resolution, origin=self.bounds, size=self.size)

    @property
    def length_scale(self) -> int:
        """The Reynolds number's length scale, in cells."""
        return self.characteristic_cells or self.resolution[2]

    def _viscosity(self) -> float:
        """Kinematic viscosity in lattice units, however it is specified.

        Returns:
            The viscosity.

        Raises:
            ValueError: When the inlet does not move and no viscosity was
                stated, so ``U L / Re`` is zero and the fluid would be a
                perfect conductor of momentum and nothing else.
        """
        if self.viscosity is not None:
            value = float(self.viscosity)
            if not value > 0.0:
                raise ValueError(f"viscosity must be positive, got {self.viscosity!r}.")
            return value
        speed = self.inlet.speed
        if speed == 0.0:
            raise ValueError(
                f"Flow study {self.name!r} has Inlet(velocity=0) and no viscosity=. A "
                "Reynolds number is U L / nu, so with no speed it fixes no viscosity, "
                "and the energy solve still needs the air's thermal diffusivity "
                "(nu / Pr). State viscosity= explicitly for a no-flow study."
            )
        if not self.reynolds > 0.0:
            raise ValueError(f"reynolds must be positive, got {self.reynolds}.")
        return speed * self.length_scale / self.reynolds

    @property
    def fluid_diffusivity(self) -> float:
        """The air's thermal diffusivity in lattice units, ``nu / Pr``."""
        return self._viscosity() / self.prandtl

    @property
    def richardson(self) -> float:
        """Buoyancy over inertia, the number the one-way coupling neglects.

        ``Ri = g beta dT L / U^2`` with the temperature scale taken from
        the declared power and the duct's own enthalpy flow, so it is
        computable before the solve.  Gravity in lattice units is the one
        genuinely arbitrary piece: it is taken as ``1e-5`` cells per step
        squared, the conventional working value at which a lattice
        Boltzmann buoyancy force is stable, which makes this a *scale*
        rather than a measurement.  Ri below
        :data:`FORCED_CONVECTION_RICHARDSON` is the condition under which
        ignoring buoyancy is defensible.

        Returns:
            The Richardson number, or 0.0 when nothing drives the flow (in
            which case buoyancy would be the *only* driver and the number
            is not the right diagnostic -- see :meth:`FlowStudyResult.warnings`).
        """
        speed = self.inlet.speed
        power = sum(bc.power for bc in self.bcs if isinstance(bc, HeatSource))
        if speed == 0.0 or power == 0.0:
            return 0.0
        # Temperature scale: the bulk rise the injected power causes in the
        # air that passes, dT = P / (rho cp U A), with (rho cp) = 1.
        area = float(self.resolution[0] * self.resolution[2])
        rise = abs(power) / (speed * area)
        gravity = 1e-5
        return gravity * self.expansion * rise * self.length_scale / speed**2

    @property
    def peclet_cell(self) -> float:
        """Inlet speed over fluid diffusivity: the cell Peclet number.

        How far the energy discretisation sits from the diffusive regime,
        at one cell of spacing.  Above about 2 the thermal boundary layer
        is thinner than a cell.
        """
        return self.inlet.speed / self.fluid_diffusivity

    def flow_config(self) -> FlowConfig:
        """The momentum configuration this study implies.

        Returns:
            A frozen :class:`~cadjoint.flow.FlowConfig`.

        Raises:
            ValueError: If the inlet does not move, in which case there is
                no momentum problem to configure.
        """
        if self.inlet.speed == 0.0:
            raise ValueError(
                f"Flow study {self.name!r} has Inlet(velocity=0): there is no momentum "
                "problem to solve, and solve() skips it rather than configuring one."
            )
        return FlowConfig(
            shape=self.resolution,
            inlet_speed=self.inlet.speed,
            reynolds=self.inlet.speed * self.length_scale / self._viscosity(),
            characteristic_cells=self.length_scale,
            alpha_max=self.alpha_max,
            steady=self.steady,
        )

    def energy_config(self) -> EnergyConfig:
        """The energy configuration this study implies.

        Returns:
            A frozen :class:`~cadjoint.flow.energy.EnergyConfig`.
        """
        return EnergyConfig(
            shape=self.resolution,
            fluid_diffusivity=self.fluid_diffusivity,
            conductivity_ratio=self.conductivity_ratio,
            inlet_temperature=self.inlet.temperature,
            wall_temperature=self.walls.temperature,
            scheme=self.scheme,
            tol=self.energy_tol,
            max_steps=self.energy_max_steps,
            restart=self.energy_restart,
        )

    # ── regions ─────────────────────────────────────────────────────────

    def _regions(self) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """Resolve every selection against the lattice, once.

        Returns:
            ``(source, power, fixed_mask, fixed_value)`` -- the per-cell
            volumetric power, the total, the held-temperature mask and the
            values on it.  All concrete NumPy: the lattice never moves, so
            nothing here depends on the design.

        Raises:
            ValueError: If a selection covers no cell, or two held regions
                overlap with different temperatures.
        """
        centers = np.asarray(self.grid.centers(), dtype=np.float64)
        wall = duct_walls(self.resolution)
        shape = self.resolution

        source = np.zeros(shape, dtype=np.float64)
        power = 0.0
        for bc in self.bcs:
            if not isinstance(bc, HeatSource):
                continue
            mask = region_mask(bc.nodes, centers) & ~wall
            count = int(mask.sum())
            if count == 0:
                raise ValueError(
                    f"HeatSource selection {bc.nodes.describe()} covers no cell of the "
                    f"{shape} lattice spanning {self.bounds} + {self.size}. A lattice "
                    "region is volumetric and coarse: a selection thinner than one cell "
                    "of the duct falls between centres. Widen it, refine the lattice, or "
                    "check it against the study's bounds."
                )
            source += mask * (bc.power / count)
            power += bc.power

        fixed = np.zeros(shape, dtype=bool)
        value = np.zeros(shape, dtype=np.float64)
        for bc in self.bcs:
            if not isinstance(bc, HeldTemperature):
                continue
            mask = region_mask(bc.nodes, centers) & ~wall
            if not mask.any():
                raise ValueError(
                    f"HeldTemperature selection {bc.nodes.describe()} covers no cell of "
                    f"the {shape} lattice spanning {self.bounds} + {self.size}."
                )
            clash = fixed & mask & (value != bc.value)
            if clash.any():
                raise ValueError(
                    f"HeldTemperature selection {bc.nodes.describe()} overlaps an earlier "
                    f"one on {int(clash.sum())} cells with a different temperature. A cell "
                    "can only be held at one value; make the selections disjoint."
                )
            fixed |= mask
            value = np.where(mask, bc.value, value)
        return source, power, fixed, value

    # ── serialization ───────────────────────────────────────────────────

    def describe(self) -> dict[str, Any]:
        """JSON-ready payload: everything a viewer would need to display it.

        Returns:
            A dictionary whose ``kind`` is :data:`FLOW_STUDY_KIND`.
        """
        return {
            "name": self.name,
            "kind": FLOW_STUDY_KIND,
            "resolution": list(self.resolution),
            "bounds": list(self.bounds),
            "size": list(self.size),
            "mesh": None,
            "domain": {
                "name": getattr(self.domain, "name", None),
                "type": type(self.domain).__name__ if self.domain is not None else "none",
            },
            "material": {
                "conductivity_ratio": self.conductivity_ratio,
                "prandtl": self.prandtl,
            },
            "fluid": {
                "reynolds": self.reynolds if self.viscosity is None else None,
                "viscosity": self._viscosity(),
                "inlet_speed": self.inlet.speed,
                "alpha_max": self.alpha_max,
                "scheme": self.scheme,
                "peclet_cell": self.peclet_cell,
                "richardson": self.richardson,
            },
            "bcs": [bc.describe() for bc in self.bcs],
        }

    # ── the solve ───────────────────────────────────────────────────────

    def solve(self, sdf: Any = None, *, chi: Any = None) -> FlowStudyResult:
        """Sample the scene, converge the flow, then the temperature.

        Differentiable end to end in ``sdf``'s parameters: nothing here
        contours, meshes, or decides membership, and both solves carry
        implicit adjoints.

        Runs under :func:`~cadjoint.flow.precision.double_precision` and
        restores the caller's setting afterwards, so a scene that declares a
        flow study still compiles to a float32 shader.  The design
        parameters a scene builds at import are float32 when the process is
        in its default mode; ``chi`` is promoted explicitly below rather
        than relying on the flag having been on when they were made.

        **A caller who differentiates this must enable x64 process-wide.**
        The scope above covers the forward solve only -- ``jax.grad`` runs
        its transposed pass after this returns, and cannot then materialise
        the float64 intermediates. Same rule as the FEM solvers; see
        :mod:`cadjoint.flow.precision`.

        Args:
            sdf: A callable on ``(..., 3)`` points, as
                :func:`cadjoint.functionalize` produces.  Optional when
                ``chi`` is given directly.
            chi: A precomputed solid fraction, ``(NX, NY, NZ)``, bypassing
                the sampling step.  For tests and for reusing one sampling
                across several operating points.

        Returns:
            The :class:`FlowStudyResult`, also stored as ``last_result``.

        Raises:
            ValueError: If neither ``sdf`` nor ``chi`` is given, or ``chi``
                does not match the lattice.
        """
        with double_precision():
            return self._solve(sdf, chi)

    def _solve(self, sdf: Any, chi: Any) -> FlowStudyResult:
        """The solve proper, with double precision already in force."""
        grid = self.grid
        if chi is None:
            if sdf is None:
                raise ValueError(
                    f"Flow study {self.name!r} needs a scene to sample: pass the "
                    "functionalized SDF to solve(sdf), or a precomputed chi=."
                )
            field = self.domain if self.domain is not None else sdf
            chi = sample_solid_fraction(field, grid, self.epsilon, self.profile)
        # Explicit, not incidental: a scene's parameters are float32 when
        # they were built outside this scope, and every tolerance below is
        # quoted in a range float32 does not have.
        chi = jnp.asarray(chi, dtype=jnp.float64)
        if tuple(chi.shape) != tuple(self.resolution):
            raise ValueError(
                f"chi has shape {tuple(chi.shape)}, expected {tuple(self.resolution)}."
            )

        source, power, fixed, fixed_value = self._regions()
        wall = duct_walls(self.resolution)
        energy = self.energy_config()

        if self.inlet.speed == 0.0:
            # No momentum problem: with a zero inlet velocity and the outlet
            # anchoring density at 1, the lattice Boltzmann fixed point *is*
            # rest, so marching to it would return exactly this at the cost
            # of a few thousand steps.  Skipping it is not an approximation,
            # and it is what makes the pure-conduction comparison against
            # ThermalStudy an equality rather than a limit.
            velocity = jnp.zeros((3, *self.resolution), dtype=chi.dtype)
            density = jnp.ones(self.resolution, dtype=chi.dtype)
            drop = jnp.zeros((), dtype=chi.dtype)
        else:
            flow = solve(
                chi,
                self.flow_config(),
                inlet_velocity=jnp.asarray(self.inlet.velocity, dtype=chi.dtype),
                cell_volume=grid.cell_volume,
            )
            velocity, density, drop = flow.velocity, flow.density, flow.pressure_drop

        # The convective flux is the mass flux, not the velocity: it is
        # what the momentum solve conserves exactly, and passing the
        # velocity alone loses a few percent of the energy balance to the
        # lattice's compressibility error (see cadjoint.flow.energy).
        temperature = solve_temperature(
            chi,
            velocity,
            energy,
            density=density,
            wall=wall,
            source=jnp.asarray(source, dtype=chi.dtype),
            fixed_mask=fixed,
            fixed_value=jnp.asarray(fixed_value, dtype=chi.dtype),
        )

        peak = peak_temperature(temperature, chi)
        # The balance only means something when the injected power is the
        # only way heat enters or leaves.  A held region or an isothermal
        # wall is an uncounted port, and reporting a "failure to conserve"
        # for one would be reporting the boundary condition.
        closed = power != 0.0 and not fixed.any() and self.walls.temperature is None
        imbalance = (
            energy_imbalance(temperature, chi, velocity, power, energy, wall=wall, density=density)
            if closed
            else None
        )
        result = FlowStudyResult(
            name=self.name,
            kind=FLOW_STUDY_KIND,
            grid=grid,
            chi=chi,
            velocity=velocity,
            density=density,
            temperature=temperature,
            pressure_drop=drop,
            peak_temperature=peak,
            mean_temperature=mean_temperature(temperature, chi),
            bulk_outlet_temperature=bulk_outlet_temperature(
                temperature, velocity, wall, density=density
            ),
            thermal_resistance=(
                (peak - self.inlet.temperature) / power
                if power != 0.0
                else jnp.zeros((), dtype=chi.dtype)
            ),
            power=power,
            energy_imbalance=imbalance,
            reynolds=self.inlet.speed * self.length_scale / self._viscosity(),
            peclet_cell=self.peclet_cell,
            richardson=self.richardson,
        )
        self.last_result = result
        return result
