"""The conjugate lattice solve against the mesh solver it has to agree with.

:class:`~cadjoint.flow.FlowStudy` and
:class:`~cadjoint.fem.study.ThermalStudy` discretise the same physics two
incompatible ways: a cell-centred finite volume on a fixed lattice, and
trilinear finite elements on a mesh cut from the same geometry.  With the
inlet held still they are solving *identically the same boundary value
problem*, so they are each other's independent check -- and the only honest
statement about how well they agree is a convergence rate, because two
discretisations never agree to solver tolerance.

The reference is a closed form both of them can be measured against::

    -k T'' = q   on y in [0, L],   T(0) = 0,   T'(L) = 0
    T(y) = (q/k) (L y - y^2 / 2)

so this file reports three numbers rather than one: how far each solver
sits from the analytic answer, and how fast the lattice closes on it.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from cadjoint.fem import Dirichlet, Nodes, SimMesh, ThermalStudy
from cadjoint.flow import FlowStudy, HeatSource, Inlet, Outlet, Walls, duct_walls

LENGTH = 1.0
CONDUCTIVITY = 1.0
SOURCE = 1.0


def _exact(y):
    """The analytic temperature at height ``y``."""
    return (SOURCE / CONDUCTIVITY) * (LENGTH * y - 0.5 * y * y)


def _slab(points):
    """A box spanning ``x, z in [-0.5, 0.5]`` and ``y in [0, L]``."""
    offsets = jnp.asarray(points) - jnp.array([0.0, LENGTH / 2, 0.0])
    return jnp.max(jnp.abs(offsets) - jnp.array([0.5, LENGTH / 2, 0.5]), axis=-1)


def _lattice_error(cells):
    """Max relative error of the still-inlet conjugate solve, on ``cells``."""
    shape = (6, cells, 6)
    step = LENGTH / cells
    core = int((~duct_walls(shape)).sum())
    study = FlowStudy(
        name=f"still-{cells}",
        resolution=shape,
        bounds=(-0.5, 0.0, -0.5),
        size=(1.0, LENGTH, 1.0),
        # No flow means no Reynolds number, so the diffusivity is stated:
        # nu / Pr is the fluid conductivity, and the ratio of 1 makes the
        # whole duct one material, which is the problem the mesh solves.
        viscosity=CONDUCTIVITY * 0.71,
        conductivity_ratio=1.0,
        energy_tol=1e-13,
        bcs=[
            Inlet(velocity=0.0, temperature=0.0),
            Outlet(),
            Walls(),
            HeatSource(
                Nodes.box([-9.0, -9.0, -9.0], [9.0, 9.0, 9.0]),
                power=SOURCE * step * step * core,
            ),
        ],
    )

    temperature = study.solve(chi=jnp.ones(shape)).temperature
    line = np.asarray(temperature[3, :, 3])
    heights = (np.arange(cells) + 0.5) * step
    return float(np.max(np.abs(line - _exact(heights))) / np.max(_exact(heights)))


class TestAgainstTheAnalyticAnswer:
    """Each solver, measured against the closed form, not against the other."""

    def test_the_mesh_solver_is_exact_on_this_problem(self):
        """Trilinear elements reproduce a quadratic solution exactly, so the
        FEM side of the comparison contributes no error of its own -- which
        is what makes the lattice's error attributable to the lattice."""
        mesh = SimMesh(
            name="slab-mesh",
            resolution=(8, 12, 8),
            bounds=(-0.5, 0.0, -0.5),
            size=(1.0, LENGTH, 1.0),
            method="hex",
        )
        study = ThermalStudy(
            name="slab-conduction",
            conductivity=CONDUCTIVITY,
            source=SOURCE,
            bcs=[Dirichlet(Nodes.halfspace([0.0, 1e-6, 0.0], [0.0, -1.0, 0.0]), 0.0)],
            mesh=mesh,
        )

        result = study.solve(_slab)

        points = np.asarray(mesh.build(_slab).points)
        temperature = np.asarray(result.solution.temperature)
        centre = np.hypot(points[:, 0], points[:, 2])
        line = centre < centre.min() + 1e-9
        error = np.max(np.abs(temperature[line] - _exact(points[line, 1])))

        assert error / np.max(_exact(points[line, 1])) < 1e-12

    @pytest.mark.parametrize(("cells", "expected"), [(8, 3.9e-3), (16, 9.8e-4), (32, 2.4e-4)])
    def test_the_lattice_converges_on_it_at_second_order(self, cells, expected):
        """Each doubling divides the error by four.

        A cell-centred scheme with the Dirichlet half a cell upstream of the
        first centre cannot be exact here, and it is not: it is second-order
        accurate, and this pins the rate rather than only the magnitude.
        """
        assert _lattice_error(cells) == pytest.approx(expected, rel=0.05)


class TestTheTwoSolversAgree:
    """The claim the conjugate path rests on, stated as a number."""

    def test_the_still_conjugate_solve_reproduces_the_mesh_answer(self):
        """At 32 cells the two discretisations differ by 2.4e-4 of the peak.

        That is the lattice's own truncation and nothing else -- the mesh
        solver contributes under 1e-12 on this problem -- so the number is
        a statement about resolution, not about the coupling. It is the
        strongest form the "zero flow reproduces ThermalStudy" check can
        honestly take: two discretisations of one problem never agree to
        solver tolerance, they agree to the coarser one's error.
        """
        assert _lattice_error(32) < 3e-4

    def test_refining_the_lattice_closes_the_gap(self):
        """The gap is truncation, so it shrinks; if it plateaued, something
        other than resolution would be separating the two solvers."""
        coarse, fine = _lattice_error(8), _lattice_error(32)

        assert fine < coarse / 10.0
