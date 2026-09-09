"""A blunt body in a duct, streamlined by gradient descent at constant volume.

The smallest scene that is *immersed shape optimization* and nothing else:
one obstacle in a channel, one scalar off the flow solve, and a declared
:class:`~cadjoint.optimize.Optimization` that walks the shape downhill.
``scenes/duct_sink.py`` is the conjugate scene -- heat conducted into fins
and carried away by air -- and it takes derivatives but does not descend.
This one descends, and there is no temperature anywhere in it.

**Immersed, not meshed.**  The body never becomes a surface.  Its SDF is
sampled on a fixed lattice into a solid fraction ``chi`` and enters the
momentum equation as a Brinkman drag ``-alpha_max chi u``
(:mod:`cadjoint.flow.domain`, :mod:`cadjoint.flow.lbm`), so the design
moves a *field* on a grid that never moves.  That is what makes the whole
chain -- half-extent to pressure drop -- one differentiable JAX expression,
and the reason the loop needs no remeshing step at all: contrast
``Optimization``'s study form, which refreezes a mesh topology every
``remesh_every`` steps because a body-fitted mesh cannot follow a design
that far.

**The objective, and why it needs two terms.**  Pressure drop alone has a
trivial minimum: delete the obstacle.  So the volume is held --

    J = dp / dp_ref + 6 (V / V_ref - 1)^2

-- and what is left to optimise is the *shape* at fixed material.  In a
duct the pressure drop is dominated by blockage, so at fixed volume the
descent should trade frontal area for length, which is streamlining.  It
does.  Sixteen steps of Adam, from the slab this file declares:

===================  ===========  ===========
quantity             start        finished
===================  ===========  ===========
half-extents         0.200 0.150  0.182 0.184
aspect ratio (y/x)   0.750        1.011
frontal area         0.1600       0.1328
volume               0.04800      0.04892
objective ``J``      1.000        0.588
pressure drop        4.597e-3     2.684e-3
===================  ===========  ===========

**And the drag fell because the shape changed, not because the body shrank.**
The volume drifts 1.9% over the descent, which is enough to muddy a 42%
claim, so :func:`main` ends by solving one more design: the *start* box
scaled isotropically to the volume the descent finished with -- same
material, same proportions, 0.6% bigger on every side.  It costs 4.706e-3,
against 2.684e-3 for the optimised shape at the identical volume.  Measured
that way the reduction is 43%, *larger* than the 42% against the start,
because the volume the optimiser finished with is the more expensive one to
carry.  The shape accounts for all of the gain and then some; none of it is
shrinkage.

**The trade is real, and it is in the gradient before any descent runs.**
At the declared starting box the adjoint gives

    d(dp)/d(half-width)  = +5.709e-2      (across the flow)
    d(dp)/d(half-length) = +1.242e-2      (along it)

Both are positive -- any growth costs pressure -- but growing across the
flow costs 4.6x what growing along it does.  Constant volume converts that
ratio into a direction, and the optimiser follows it.

**Verified against a central difference before it was believed.**  A
falling loss curve is not evidence of a correct gradient; a wrong one
often descends too, just to the wrong place.  :func:`main` re-runs the
check every time it is invoked, and asserts what makes it meaningful: the
relative error against the adjoint falls by about 100x when the step size
falls by 10x, which is the second-order convergence a central difference
has on a smooth objective, and which the ``"smootherstep"`` solid-fraction
profile exists to preserve (:mod:`cadjoint.flow.domain` carries that
measurement).  Agreement at a single step size proves only that the
truncation error happened to be small.

**Cubic cells, deliberately.**  16 x 32 x 16 over 1.00 x 2.00 x 1.00 makes
every cell 0.0625 on a side.  The solver works in lattice units and the
world ``size`` only decides where the SDF is sampled, so a lattice whose
cells are not cubes hands the solver the duct stretched by the ratio of its
spacings and nothing downstream knows -- the mistake
``scenes/duct_sink.py`` documents having made.
:meth:`~cadjoint.flow.FlowStudyResult.warnings` reports it now; this study
reports nothing.

**Coarse, deliberately.**  16 cells span the duct and the body is about six
across, which resolves a blockage and a wake but not a boundary layer.  The
whole descent -- 16 flow solves and 16 adjoint solves -- runs in about two
and a half minutes on a laptop CPU, which is what makes it a scene rather
than a cluster job.  ``research/flow-solver.md`` carries the resolution
study, and none of the numbers above should be quoted as a drag
coefficient: what is demonstrated here is that the derivative is right and
the descent is real, not what the right answer is at infinite resolution.

**Precision.**  This file sets no jax flags at module scope, deliberately:
``jax_enable_x64`` is process-global and the WGSL backend cannot emit an
``f64``, so a scene that flipped it could not be opened in the viewer.  The
flow solve scopes double precision around its own forward pass, and the
descent asks for it around the whole loop by declaring
``precision="double"`` on the optimization -- which is the piece a
*gradient* needs, because :func:`jax.grad` runs its transposed pass after
the forward scope has closed.

Run it directly::

    python scenes/duct_fairing.py
"""

import jax
import jax.numpy as jnp
import numpy as np

from cadjoint import extract_parameters, functionalize
from cadjoint.construction import Solid
from cadjoint.flow import FlowStudy, Inlet, Outlet, SteadyOptions, Walls
from cadjoint.flow.precision import double_precision
from cadjoint.geometry import Vector
from cadjoint.optimize import Optimization
from cadjoint.render import Material

# ── the design ───────────────────────────────────────────────────────────────
# Half-extents, like every primitive in cadjoint: the body spans twice these.
# x and z are across the flow (the frontal area the duct sees), y is along it.
# The start is deliberately the wrong way round -- a slab 0.40 wide, 0.40 tall
# and 0.30 long, blunter than it is long -- so the descent has somewhere to go.
fairing_size = Vector([0.20, 0.15, 0.20], free=True, name="fairing_size")

# The centre is pinned.  Solid.box would make the position three more free
# parameters, and a body free to *move* in a duct with a symmetric objective
# has a flat direction and a wall to drift into; the shape is the design here.
fairing_position = Vector([0.0, 0.0, 0.0], free=False, name="fairing_position")

steel = Material(
    name="steel",
    color=[0.55, 0.57, 0.60],
    roughness=0.4,
    metallic=0.9,
    density=7850.0,
    conductivity=45.0,
    specific_heat=470.0,
)

fairing = Solid.box(
    size=fairing_size,
    position=fairing_position,
    material=steel,
    name="fairing",
)
scene = fairing

# ── the duct ─────────────────────────────────────────────────────────────────
# Flow along +Y, walls at the x and z extremes, 1.00 x 2.00 x 1.00 of world on
# a cubic 16 x 32 x 16 lattice.  The body's 0.40 x 0.40 frontal face blocks 16%
# of the 1.00 x 1.00 section, which accelerates the free stream to about
# max|u| = 0.046 in lattice units -- Mach 0.08, where the lattice's
# compressibility error is still under a tenth of a percent.  Twice the
# blockage would not be: an earlier 12-cell duct with the same body reached
# max|u| = 0.154, which is not a flow this solver should be asked about.
drag = FlowStudy(
    name="duct-drag",
    resolution=(16, 32, 16),
    bounds=(-0.50, -1.00, -0.50),
    size=(1.00, 2.00, 1.00),
    # Re = 25 against the duct's 16 cells of height is a lattice viscosity of
    # 0.0128 and a BGK relaxation rate of 1.857, which leaves margin under the
    # 1.95 ceiling.  Laminar and steady, which is the only regime this solver
    # models -- it carries no turbulence closure.
    reynolds=25.0,
    bcs=[
        # No temperature anywhere in this scene: no HeatSource, so the energy
        # solve's right-hand side is zero and its answer is exactly zero.  The
        # inlet temperature is the reference for a field nothing drives.
        Inlet(velocity=0.02),
        Outlet(),
        Walls(),
    ],
    # Measured rather than assumed: loosening the adjoint from 1e-10 to 1e-8
    # left an 18-step descent trajectory identical to five digits and took
    # 9.4 s a step instead of 14.  The gradient here is not tolerance-limited,
    # so the loop does not pay for a tolerance it cannot use.  main() tightens
    # both for its finite-difference check, where it does matter.
    steady=SteadyOptions(
        tol=1e-9,
        max_steps=40000,
        adjoint_solver="fixed_point",
        adjoint_tol=1e-8,
        adjoint_max_steps=4000,
    ),
)

# ── the objective ────────────────────────────────────────────────────────────
free_start, fixed, _ = extract_parameters(fairing)
evaluate = functionalize(fairing)

#: Volume of the starting box, 8 a b c.  The constraint is written against
#: this rather than against a sampled ``sum(chi)`` on purpose: the box's
#: volume is exact and analytic, so the constraint carries no discretisation
#: error of its own and cannot be gamed by moving a face onto a cell centre.
REFERENCE_VOLUME = 8 * 0.20 * 0.15 * 0.20

#: Pressure drop of the starting box, in lattice units, measured once at the
#: tight tolerances :func:`main` uses for its gradient check and pasted here
#: (the descent's looser study reports 4.596540e-03, two in the last digit
#: away, which is the size of the convergence tolerance and not of anything
#: physical).  It is only a unit: Adam rescales each
#: coordinate by its own gradient history, so this number cannot change where
#: the descent goes.  What it fixes is the *ratio* between the two terms of
#: ``J``, and therefore how hard the volume is held -- which is why it is a
#: named constant and not an incidental 1.0.
REFERENCE_DROP = 4.596538e-03

#: Weight on the volume constraint.  6 rather than 20, and the difference is
#: visible in the trajectory rather than in the answer: both reach the same
#: shape and the same 2.684e-3 drop, but at 20 the penalty is stiff enough that
#: Adam's fixed step size overshoots it and the loss enters a limit cycle of
#: +-2.5% around the minimum instead of settling.
VOLUME_WEIGHT = 6.0


def streamlining_cost(parameters):
    """Pressure drop at (nearly) constant volume -- the descended objective.

    Args:
        parameters: The free-parameter dict, ``{"fairing_size": (3,)}``.

    Returns:
        Scalar ``dp / REFERENCE_DROP + VOLUME_WEIGHT (V/V0 - 1)^2``.
    """
    # Promoted explicitly rather than inherited: the scene is built at import
    # in float32 so it can still become a shader, and sampling chi from a
    # float32 SDF puts a 1e-7 ripple on a pressure drop the objective reads to
    # far better than that.  Under ``precision="double"`` this is a real cast;
    # in a float32 process it is a no-op and a warning, which is the honest
    # behaviour for a scene someone merely opened.
    parameters = {name: jnp.asarray(value, jnp.float64) for name, value in parameters.items()}
    half = parameters["fairing_size"]
    volume = 8 * half[0] * half[1] * half[2]
    drop = drag.solve(evaluate(parameters, fixed)).pressure_drop
    return drop / REFERENCE_DROP + VOLUME_WEIGHT * (volume / REFERENCE_VOLUME - 1.0) ** 2


# ── the optimization ─────────────────────────────────────────────────────────
# 16 steps at 0.005: the loss knee is around step 9, but the *volume* is what
# takes longest to settle -- Adam's momentum carries it down to 0.82x and back
# -- and stopping at 14 leaves it 3.2% high where 16 leaves it 1.9%.  About
# 9 s a step, so under three minutes, which fits the playground's per-run
# budget with room for the compile.
# precision="double" is not decoration -- without it this run dies in the
# backward pass with "lax.dynamic_update_slice requires arguments to have the
# same dtypes, got float32, float64", because the flow solve's own x64 scope
# covers its forward pass and has closed by the time jax.grad transposes it.
streamline = Optimization(
    name="streamline",
    objective=streamlining_cost,
    of=fairing,
    steps=16,
    learning_rate=0.005,
    method="adam",
    precision="double",
)


def main():
    """Solve, verify the gradient against a central difference, then descend.

    The verification is not optional scaffolding.  A plausible descent on a
    wrong gradient is the failure mode this scene is most exposed to, and it
    is invisible in the loss curve, so the check runs first and its
    second-order convergence is asserted rather than eyeballed.
    """
    with double_precision():
        free = {name: jnp.asarray(value, jnp.float64) for name, value in free_start.items()}

        result = drag.solve(evaluate(free, fixed))
        print(f"pressure drop        {float(result.pressure_drop):.6e}")
        print(f"max |u| (lattice)    {float(jnp.max(jnp.abs(result.velocity))):.4f}")
        print(f"solid cells (sum chi){float(jnp.sum(result.chi)):9.2f} of {result.grid.cells}")
        print(f"Re {result.reynolds:.0f}")
        for note in result.warnings():
            print(f"  ! {note}")

        # ── the gradient, against a central difference ───────────────────────
        # Tighter than the descent runs.  A central difference at h = 1e-4
        # moves the pressure drop by 2 h dJ/dx -- about five parts in ten
        # thousand on the least sensitive axis -- so the forward solve has to
        # be converged far past that or the check measures its own residual
        # rather than the derivative.  A second study rather than a tighter
        # `drag`, so the descent is not made to pay for the audit: the forward
        # tolerance moves the drop in the sixth significant figure (1e-9 gives
        # 4.59654038e-03 where 1e-12 gives 4.59653807e-03) and only the
        # difference of two nearby solves is sensitive to that.
        checked = FlowStudy(
            name="duct-drag-checked",
            resolution=drag.resolution,
            bounds=drag.bounds,
            size=drag.size,
            reynolds=drag.reynolds,
            bcs=drag.bcs,
            steady=SteadyOptions(
                tol=1e-12,
                max_steps=80000,
                adjoint_solver="fixed_point",
                adjoint_tol=1e-12,
                adjoint_max_steps=20000,
            ),
        )

        def pressure_drop(parameters):
            return checked.solve(evaluate(parameters, fixed)).pressure_drop

        gradient = np.asarray(jax.grad(pressure_drop)(free)["fairing_size"])
        print(f"\nd(dp)/d(half-extents) {gradient}")

        def difference(axis, step):
            plus = dict(free, fairing_size=free["fairing_size"].at[axis].add(step))
            minus = dict(free, fairing_size=free["fairing_size"].at[axis].add(-step))
            return float((pressure_drop(plus) - pressure_drop(minus)) / (2 * step))

        for axis, label in enumerate("xyz"):
            coarse = abs(difference(axis, 1e-3) - gradient[axis]) / abs(gradient[axis])
            fine = abs(difference(axis, 1e-4) - gradient[axis]) / abs(gradient[axis])
            print(
                f"  {label}: adjoint {gradient[axis]:+.6e}   relative error "
                f"{coarse:.2e} -> {fine:.2e}   fell {coarse / fine:.0f}x for 10x in h"
            )
            assert fine < 1e-3, f"adjoint and central difference disagree on {label}"
            assert coarse / fine > 30.0, f"the {label} difference is not second order"

        # ── the descent ──────────────────────────────────────────────────────
        print(f"\ndescending {streamline.name} ({streamline.steps} steps)")
        run = streamline.run(
            callback=lambda record: print(
                f"  step {record['step']:2d}  J {record['objective']:.6f}  "
                f"|grad| {record['grad_norm']:.3f}"
            )
        )
        start = np.asarray(run.initial["fairing_size"])
        final = np.asarray(run.parameters["fairing_size"])
        print(f"\nhalf-extents  {start} -> {final}")
        print(f"aspect (y/x)  {start[1] / start[0]:.3f} -> {final[1] / final[0]:.3f}")
        print(f"frontal area  {4 * start[0] * start[2]:.5f} -> {4 * final[0] * final[2]:.5f}")
        volume = 8 * final[0] * final[1] * final[2]
        ratio = volume / REFERENCE_VOLUME
        print(f"volume        {REFERENCE_VOLUME:.5f} -> {volume:.5f}  ({ratio:.4f}x)")
        print(f"objective     {run.history[0]['objective']:.6f} -> {run.objective:.6f}")

        # ── shape, or size?  the counterfactual that decides it ──────────────
        # The volume drifts by a couple of percent, so "the drag fell" is not
        # yet "the shape improved".  Scale the START box isotropically to the
        # volume the descent finished with and solve that: same material, same
        # proportions, only the size changed.  Whatever separates it from the
        # optimised box is shape and nothing else.
        scaled = start * ratio ** (1 / 3)

        def drop_of(values):
            params = {"fairing_size": jnp.asarray(values, jnp.float64)}
            return float(drag.solve(evaluate(params, fixed)).pressure_drop)

        as_run, as_scaled = drop_of(final), drop_of(scaled)
        print(f"\nsame volume, start proportions   dp {as_scaled:.6e}")
        print(f"same volume, optimised shape     dp {as_run:.6e}")
        print(f"attributable to shape alone      {100 * (as_run / as_scaled - 1):+.1f}%")
        return run


if __name__ == "__main__":
    main()
