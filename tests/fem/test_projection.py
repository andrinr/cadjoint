"""Reverse-mode conditioning of the Newton projection (``project_points``).

The projection's step is ``-f(x) grad f / |grad f|^2``.  Guarding that
denominator with ``jnp.maximum(squared, 1e-12)`` keeps the *forward* pass
finite but freezes the denominator as a **constant**, so the step's
Jacobian becomes ``value * Hessian / 1e-12`` — a 1e12 amplification per
iteration that compounds over the eight iterations even where the forward
step is exactly zero.  ``cadjoint.fem.hexmesh`` now suppresses the step in
both passes instead (the repo's double-``where`` idiom, cf.
``cadjoint.meshing.edge_detection``), so a dead subgradient contributes
neither a displacement nor a derivative.

Measured on the starter heat sink (``scenes/starter.py``, ``fin_depth``)
before and after, on its own declared 18x13x11 grid and its 860-vertex DC
surface:

===========  ===================  ==================  ==============
fin_depth    adjoint (floored)    adjoint (guarded)   central FD
===========  ===================  ==================  ==============
1.10         -3.867e+68           +1.186e+02          +118.574
1.15         -3.532e+68           +1.239e+02          +123.851
1.1873       +2.648e+69           +1.278e+02          +127.787
1.19         +5.438e+69           +1.281e+02          +128.072
===========  ===================  ==================  ==============

(objective ``sum(points**2)`` through ``recompute_tet_points``).  The
per-node boundary Jacobian ``|dx/d fin_depth|`` went from a maximum of
4.2e+68 over the 860 surface nodes to a uniform 0.500.  The forward pass
improved too: the floored step used to fling 14-35 nodes off the zero set
(max ``|sdf|`` residual 1.62e-2); guarded, the same nodes stay on it
(9.0e-5).

The defect was invisible at the freeze design itself, which is why every
shipped test missed it: there the dead-subgradient nodes carry ``value ==
0.0`` exactly, so ``value * Hessian / 1e-12`` is zero and the measured
maximum amplification is the healthy 0.503.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cadjoint.fem.hexmesh import GridSpec, project_points, recompute_points, sdf_to_hex_mesh
from cadjoint.geometry.parameters import Scalar, Vector
from cadjoint.sdf.primitives import Box, Sphere

_REPO = Path(__file__).resolve().parents[2]
_SCENE_PATH = _REPO / "scenes" / "starter.py"

#: Every well-posed design in these sweeps must keep the boundary point
#: Jacobian bounded by this.  The healthy measured value is ~0.5.
_SANE_JACOBIAN = 1e3


def _dead_subgradient_field(tilt):
    """A wedge whose crease has a bit-exact dead subgradient.

    ``jnp.maximum`` splits its subgradient evenly across a tie, exactly as
    ``ExtrudedPolygon.sdf``'s ``jnp.maximum(d2, dz)`` does at a profile
    corner.  At ``tilt = 1`` the two half-space normals here are exactly
    opposite, so on the crease the two halves cancel to a bit-exact zero
    gradient while the field still carries a nonzero value and a nonzero
    *parameter* sensitivity — the configuration the starter's fin-tip and
    slug-bottom nodes land on.

    Args:
        tilt: Design parameter rotating one face; ``1.0`` is the crease.

    Returns:
        A scalar SDF callable on ``(3,)`` points.
    """

    def sdf(p):
        rotated = p[1] + (tilt - 1.0) * p[0]
        return jnp.maximum(rotated, -p[1]) - 1e-8

    return sdf


def _crease_points():
    """Start points sitting exactly on the wedge crease."""
    return jnp.asarray(
        [[x, 0.0, z] for x in (-0.4, -0.1, 0.2, 0.5) for z in (-0.3, 0.0, 0.3, 0.6)],
        dtype=jnp.float64,
    )


class TestDeadSubgradientGuard:
    """The guard's contract, on a field built to trip it."""

    def test_the_start_points_really_have_a_dead_subgradient(self):
        """The premise: |grad| is a bit-exact zero while the value is not."""
        sdf = _dead_subgradient_field(jnp.asarray(1.0))
        value, gradient = jax.vmap(jax.value_and_grad(lambda p: jnp.asarray(sdf(p)).reshape(())))(
            _crease_points()
        )
        assert np.abs(np.asarray(gradient)).max() == 0.0
        assert np.abs(np.asarray(value)).min() > 0.0

    def test_the_step_is_suppressed_in_the_forward_pass(self):
        """A point with no usable gradient direction must not move."""
        start = _crease_points()
        moved = project_points(_dead_subgradient_field(jnp.asarray(1.0)), start, 0.1)
        assert np.abs(np.asarray(moved) - np.asarray(start)).max() == 0.0

    def test_the_adjoint_stays_bounded_through_the_dead_subgradient(self):
        """The regression fence.

        With the old floored denominator this Jacobian measured
        **4.0e+04** — a purely numerical derivative for a point that never
        moves.  Guarded, it is exactly zero.
        """
        jacobian = jax.jacrev(
            lambda t: project_points(_dead_subgradient_field(t), _crease_points(), 0.1)
        )(jnp.asarray(1.0))
        jacobian = np.asarray(jacobian)
        assert np.isfinite(jacobian).all()
        magnitude = np.linalg.norm(jacobian, axis=-1)
        assert magnitude.max() < _SANE_JACOBIAN
        assert magnitude.max() == 0.0

    def test_no_nan_reaches_the_cotangent(self):
        """The inner ``where`` keeps the suppressed division finite."""
        cotangent = jax.grad(
            lambda t: jnp.sum(project_points(_dead_subgradient_field(t), _crease_points(), 0.1))
        )(jnp.asarray(1.0))
        assert np.isfinite(np.asarray(cotangent)).all()


class TestWellConditionedProjectionIsUntouched:
    """The guard must not cost the projection its job or its gradient."""

    def test_sphere_vertices_still_land_on_the_zero_set(self):
        """The tolerance the shipped hex test pins; measured 1.1e-16."""
        sphere = Sphere(Scalar(value=0.6, free=True, name="radius"))
        grid = GridSpec.from_bounds((-0.75, -0.75, -0.75), (1.5, 1.5, 1.5), 12)
        mesh = sdf_to_hex_mesh(sphere, grid)
        snapped = mesh.points[mesh.snap_mask]
        assert np.abs(np.asarray(sphere(jnp.asarray(snapped)))).max() < 1e-3

    def test_box_recompute_adjoint_is_bounded_and_matches_finite_differences(self):
        """A sweep of designs, adjoint vs central FD, on the hex path."""
        grid = GridSpec.from_bounds((-0.79, -0.77, -0.81), (1.6, 1.6, 1.6), 13)
        mesh = sdf_to_hex_mesh(Box(Vector([0.5, 0.5, 0.5], free=True, name="size")), grid)

        def spread(half):
            def sdf(p):
                return Box.sdf(p, jnp.stack([half, half, half]))

            return jnp.sum(recompute_points(sdf, mesh) ** 2)

        for half in (0.46, 0.48, 0.5, 0.52, 0.54):
            adjoint = float(jax.grad(spread)(jnp.asarray(half)))
            eps = 1e-5
            central = float(
                (spread(jnp.asarray(half + eps)) - spread(jnp.asarray(half - eps))) / (2 * eps)
            )
            assert np.isfinite(adjoint)
            assert abs(adjoint) < _SANE_JACOBIAN
            assert np.isclose(adjoint, central, rtol=1e-3), (half, adjoint, central)


@pytest.fixture(scope="module")
def starter_surface():
    """The starter's frozen DC surface and its traced design field.

    Returns:
        ``(mesh, base_points, field_at_depth)`` at the scene's own bounds
        and 18x13x11 resolution — the exact configuration the blow-up was
        measured on.
    """
    pytest.importorskip("tetgen")
    from cadjoint import extract_parameters, functionalize
    from cadjoint.fem import capture_sim_meshes, capture_studies
    from cadjoint.fem.tetmesh import sdf_to_tet_mesh

    namespace: dict = {}
    with capture_sim_meshes(), capture_studies():
        exec(compile(_SCENE_PATH.read_text(), str(_SCENE_PATH), "exec"), namespace, namespace)

    # The body the declared mesh discretizes (its ``domain=``), not the
    # rendered scene: the starter also draws board-level context the physics
    # never sees.
    scene = namespace["thermal_body"]
    free0, fixed, _ = extract_parameters(scene)
    scene_fn = functionalize(scene)

    def field_at_depth(fin_depth):
        free = dict(free0)
        free["fin_depth"] = fin_depth
        inner = scene_fn(free, fixed)
        return lambda p: jnp.asarray(inner(p))

    declared = namespace["sink_mesh"]
    grid = GridSpec.from_bounds(declared.bounds, declared.size, declared.resolution)
    fin0 = jnp.asarray(np.asarray(free0["fin_depth"]), dtype=jnp.float64).reshape(())
    mesh = sdf_to_tet_mesh(field_at_depth(fin0), grid)
    return mesh, jnp.asarray(mesh.base_points[: mesh.num_surface]), field_at_depth


#: The designs the blow-up was measured at, plus the freeze design.
_MEASURED_DEPTHS = (1.10, 1.15, 1.1873, 1.19, 1.20, 1.25)


class TestStarterSurfaceProjection:
    """The measured defect, pinned on the scene that exposed it."""

    def test_boundary_point_jacobian_stays_order_one(self, starter_surface):
        """Before: up to 4.2e+68 over the 860 nodes.  After: 0.500.

        The old code exceeded ``_SANE_JACOBIAN`` at every design below the
        freeze design; only at 1.20 itself did it look healthy.
        """
        mesh, base, field_at_depth = starter_surface
        for depth in _MEASURED_DEPTHS:
            jacobian = np.asarray(
                jax.jacrev(lambda t: project_points(field_at_depth(t), base, mesh.max_step))(
                    jnp.asarray(depth)
                )
            )
            assert np.isfinite(jacobian).all(), depth
            magnitude = np.linalg.norm(jacobian, axis=-1)
            assert magnitude.max() < _SANE_JACOBIAN, (depth, magnitude.max())

    def test_projected_vertices_stay_on_the_zero_set(self, starter_surface):
        """The forward half: the floored step used to fling nodes off it.

        Measured max ``|sdf|`` over the 860 projected vertices: 1.62e-2
        floored, 9.0e-5 guarded.  Asserted at 1e-3, which the old code
        fails by 16x.
        """
        mesh, base, field_at_depth = starter_surface
        for depth in _MEASURED_DEPTHS:
            sdf = field_at_depth(jnp.asarray(depth))
            projected = project_points(sdf, base, mesh.max_step)
            residual = np.abs(np.asarray(sdf(projected)))
            assert residual.max() < 1e-3, (depth, residual.max())

    def test_adjoint_agrees_with_finite_differences_off_the_kink(self, starter_surface):
        """Adjoint vs central FD on the smooth branch of the design range.

        ``fin_depth = 1.2`` is a kink of the frozen-topology map (the
        extrusion's cap branch switches there; see ``test_starter_tet``),
        so the smooth probes sit clear of it.  Measured relative errors are
        below 3e-5.
        """
        mesh, base, field_at_depth = starter_surface

        def objective(depth):
            return jnp.sum(project_points(field_at_depth(depth), base, mesh.max_step) ** 2)

        for depth in (1.12, 1.15, 1.18, 1.23):
            adjoint = float(jax.grad(objective)(jnp.asarray(depth)))
            eps = 1e-4
            central = float(
                (objective(jnp.asarray(depth + eps)) - objective(jnp.asarray(depth - eps)))
                / (2 * eps)
            )
            assert abs(adjoint) < _SANE_JACOBIAN, (depth, adjoint)
            assert np.isclose(adjoint, central, rtol=1e-3), (depth, adjoint, central)
