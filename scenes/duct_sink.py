"""A finned sink in a duct, cooled by air that has to be pushed past it.

The smallest scene that exercises the whole conjugate path: geometry from
sketchable parameters, air driven along the duct by a
:class:`~cadjoint.flow.FlowStudy`, heat conducted out of a die into the
fins and carried away by the air, and a derivative of the result with
respect to the fin box.

**What makes it different from ``scenes/starter.py``.**  That scene's
thermal study holds the fin field at ambient with a Dirichlet patch -- an
infinitely capable heat sink, under which taller fins are free and every
fin is as good as every other.  Here nothing is held.  The air arrives at
the inlet temperature, warms as it passes, and the last fin sees hotter air
than the first, which is the effect that decides how many fins are worth
building.  Nothing in the loop meshes: the design reaches the solver as a
solid fraction sampled from this file's SDF on a fixed lattice.

**Deliberately coarse, and necessarily cubic.**  The lattice below is
20 x 26 x 12 over a 1.40 x 1.82 x 0.84 duct, which makes every cell 0.07 on
a side.  The three spacings being equal is not tidiness: the solver works in
lattice units -- streaming moves one cell per axis, the energy stencil is one
cell wide -- and the world ``size`` reaches it only by deciding where the SDF
is sampled.  A lattice whose cells are not cubes therefore hands the solver
the duct *stretched* by the ratio of its spacings, and nothing downstream
knows.  This file used to do exactly that: 14 x 26 x 14 over a
1.40 x 1.80 x 0.85 box gave cells of 0.100 x 0.069 x 0.061, so the solver saw
a duct 1.6x too wide and read a fin 0.16 thick and 0.22 tall as 1.6 by 3.6
cells -- an aspect ratio of 0.44 where the drawing says 0.73.
:meth:`~cadjoint.flow.FlowStudyResult.warnings` now reports a non-cubic
lattice rather than letting it pass.

What coarseness remains is on purpose: the whole thing -- flow, temperature
and a gradient -- runs in well under a minute and can live in a test.  That
is not a resolution anyone should quote a thermal resistance from: 10 cells
span the open duct, where ``research/flow-solver.md``'s resolution study puts
a Nusselt number about 7% low, and the fin channels are four cells wide.
That file carries the study and the expensive version.

Run it directly::

    python scenes/duct_sink.py

**Precision.**  This file sets no jax flags, deliberately.  The flow solve
needs float64 and turns it on for itself, scoped and restored
(:func:`cadjoint.flow.precision.double_precision`); a scene that flipped
``jax_enable_x64`` at module scope would make the whole process float64,
and the WGSL backend cannot emit an ``f64`` -- so merely *declaring* a flow
study would stop the scene from opening in the viewer.

**In the viewer.**  The scene compiles, renders and drags like any other,
and the flow study reaches the compile payload with its boundary conditions
(``kind: "flow"``).  What it does not have yet is the Studies window's
*editing* affordances -- dragging a boundary condition, adding one from the
UI -- because those are driven by :class:`~cadjoint.enums.StudyKind`, which
deliberately still enumerates the two kinds the patch endpoints can write.
The declaration therefore reports ``editable: false``, which is true.
"""

import jax

from cadjoint import extract_parameters, functionalize
from cadjoint.construction import Solid
from cadjoint.flow import FlowStudy, HeatSource, Inlet, Outlet, SteadyOptions, Walls
from cadjoint.flow.precision import double_precision
from cadjoint.geometry import Vector
from cadjoint.render import Material
from cadjoint.sdf.boolean import Union
from cadjoint.studies import Nodes

# ── design parameters ────────────────────────────────────────────────────────
# The fin box is shared by every fin, so one parameter sets the whole comb.
# Its z component is a half-extent, so the box is twice it tall; the fin is
# centred on the deck's top face, which buries the lower half and leaves the
# z component itself as the height standing in the airflow.  Growing the
# fins reaches further into the free stream and costs pressure drop, and the
# flow solve is what makes that trade two-sided instead of one.
# Primitive dimensions are HALF-extents: Solid.box(size=[a, b, c]) spans
# +-a, +-b, +-c about its position.  Everything below is written that way.
DECK_TOP = -0.18
DECK_HALF_THICKNESS = 0.12
# Three thick fins on a wide pitch, not a fine comb: the channels between
# them have to be several cells across for the lattice to resolve a flow in
# them at all, and a demonstration that runs in seconds does not have cells
# to spare.  The 0.28 gap is four cells and a fin is two, which is the least
# that resolves either.  A real design has finer fins and needs the
# resolution in research/flow-solver.md.
# Half-extents: each fin is 0.16 thick, 0.90 long, and 0.44 tall overall.
# It is centred on the deck's top face, so its lower half is buried in the
# deck and the height standing in the airflow is the z half-extent.
fin_size = Vector([0.08, 0.45, 0.22], free=True, name="fin_size")
FIN_COUNT = 3
FIN_PITCH = 0.44

aluminum = Material(
    name="aluminum",
    color=[0.80, 0.82, 0.85],
    roughness=0.3,
    metallic=0.9,
    density=2700.0,
    conductivity=167.0,
    specific_heat=896.0,
)

# ── the sink: a base deck carrying a comb of fins ────────────────────────────
# The duct runs along +Y, so the fins run along the flow and the channels
# between them are the passages the air has to take.
deck = Solid.box(
    size=[0.52, 0.45, DECK_HALF_THICKNESS],
    position=[0.0, 0.0, DECK_TOP - DECK_HALF_THICKNESS],
    material=aluminum,
    name="deck",
)
fins = [
    Solid.box(
        size=fin_size,
        position=[(index - (FIN_COUNT - 1) / 2) * FIN_PITCH, 0.0, DECK_TOP],
        material=aluminum,
        name=f"fin_{index}",
    )
    for index in range(FIN_COUNT)
]

# A blend rather than a hard union: the fin roots get a fillet, and a smooth
# field is what keeps the sampled solid fraction differentiable across the
# join.  ``smoothness`` is the smooth minimum's parameter and not the fillet
# radius -- smooth_min scales it by four -- so 0.01 rounds each root by about
# 0.034, half a cell here and nearly invisible to the sampling.  The
# differentiability is the reason for it; the fillet is a side effect.
scene = Union(deck, *fins, smoothness=0.01)

# ── the conjugate study ──────────────────────────────────────────────────────
# The duct is a box around the sink with room above the fins for the air to
# go; the lateral walls are the lattice's own x and z extremes.  The die is
# a region inside the deck, selected volumetrically the way a mesh study
# selects nodes.
cooling = FlowStudy(
    name="duct-cooling",
    # Cubic cells, 0.07 on a side: 1.40/20, 1.82/26 and 0.84/12 are the same
    # number, and the module docstring says why they have to be.  What that
    # buys geometrically is two cells across a fin, four across a channel,
    # three of fin standing above the deck and three of clear air above that,
    # with the sink's 0.90 of length sitting on 12 cells between seven of
    # lead-in and seven of run-out.
    resolution=(20, 26, 12),
    bounds=(-0.70, -0.91, -0.50),
    size=(1.40, 1.82, 0.84),
    # Re = 25 against the duct's 12 cells of height is a lattice viscosity of
    # 0.0096 and a BGK relaxation rate of 1.891, which leaves real margin
    # under the 1.95 ceiling -- and a coarse duct needs it, because the
    # ceiling falls with the lattice size (see FlowStudy).
    reynolds=25.0,
    conductivity_ratio=200.0,
    # 1e-9 rather than 1e-11: on a duct this coarse the per-step relative
    # change floors out around 3e-9 in float64 and a tighter target only
    # buys the iteration cap.  The residual history is how that was found
    # (cadjoint.flow.convergence), and it is worth re-checking on any new
    # geometry rather than assuming.
    steady=SteadyOptions(
        tol=1e-9,
        max_steps=30000,
        adjoint_solver="fixed_point",
        adjoint_tol=1e-10,
        adjoint_max_steps=6000,
    ),
    energy_tol=1e-12,
    energy_max_steps=12000,
    # 60 rather than the ratio-blind 30: at conductivity_ratio=200 a
    # 30-vector subspace stagnates and reports a peak temperature 42x too
    # low without failing.  The energy imbalance printed below is the check.
    energy_restart=60,
    bcs=[
        # 0.02 cells per step is Mach 0.035 -- low enough that the lattice's
        # compressibility error stays under a tenth of a percent.
        Inlet(velocity=0.02, temperature=0.0),
        Outlet(),
        # Adiabatic: the duct carries air past the part, it is not itself a
        # heat exchanger, and it is the only wall condition under which the
        # energy balance closes against the die's power alone.
        Walls(),
        # The die's footprint, injected *through* the deck rather than
        # under it, and the lattice leaves no choice.  The deck's underside
        # is z = -0.42; the only cell centres below that belong to the duct
        # floor, which is wall, so there is nowhere under the deck for a
        # separate die to sit -- and a source landing in a fluid cell would
        # heat the air directly instead of conducting out through the metal,
        # which is the whole thing this scene demonstrates.  The box takes
        # the deck's own z extent, so it selects exactly the deck cells the
        # wall does not claim, under a 0.28 x 0.36 footprint.
        HeatSource(Nodes.box([-0.14, -0.18, -0.42], [0.14, 0.18, -0.18]), power=1.0),
    ],
)


def main():
    """Solve the coupled problem and take one derivative through both solves.

    The whole body runs in double precision because it takes a *gradient*:
    ``FlowStudy.solve`` scopes x64 around its own forward pass, but
    ``jax.grad`` runs the backward pass after that scope has closed and
    needs the flag still set.  Doing it here rather than at module scope is
    the point -- importing this file leaves the process in float32, so the
    scene still compiles to a shader.
    """
    with double_precision():
        free, fixed, _ = extract_parameters(scene)
        evaluate = functionalize(scene)

        result = cooling.solve(evaluate(free, fixed))
        print(f"peak temperature     {float(result.peak_temperature):.5f}")
        print(f"mean temperature     {float(result.mean_temperature):.5f}")
        print(f"outlet bulk air      {float(result.bulk_outlet_temperature):.5f}")
        print(f"thermal resistance   {float(result.thermal_resistance):.5f}")
        print(f"pressure drop        {float(result.pressure_drop):.6f}")
        print(f"energy imbalance     {float(result.energy_imbalance):.3e}")
        print(
            f"Re {result.reynolds:.0f}   cell Peclet {result.peclet_cell:.2f}   "
            f"Ri {result.richardson:.2e}"
        )
        for note in result.warnings():
            print(f"  ! {note}")

        def objective(parameters):
            return cooling.solve(evaluate(parameters, fixed)).peak_temperature

        gradient = jax.grad(objective)(free)["fin_size"]
        print(f"d(peak)/d(fin thickness, length, height) {gradient}")
        return result


if __name__ == "__main__":
    main()
