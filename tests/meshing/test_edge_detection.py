"""Tests for jaxcad.meshing.edge_detection (crossing detection + Hermite data)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxcad import extract_parameters, functionalize
from jaxcad.geometry.parameters import Scalar, Vector
from jaxcad.meshing.edge_detection import (
    GridSpec,
    detect_edges,
    edge_hermite_data,
    find_crossing_edges,
    sample_grid,
)
from jaxcad.sdf.primitives import Box, Sphere


def sphere_sdf(radius):
    """Unit-style sphere field ``||p|| - radius`` as a point callable."""

    def sdf(p):
        return jnp.sqrt(jnp.sum(p * p)) - radius

    return sdf


def sphere_grid() -> GridSpec:
    """Verified baseline grid whose lattice contains vertices exactly on the unit sphere."""
    return GridSpec.from_bounds((-1.3, -1.3, -1.3), (2.6, 2.6, 2.6), 26)


def edge_endpoints(grid: GridSpec, edges) -> tuple[np.ndarray, np.ndarray]:
    """Recompute world-space start/end vertices of each edge from the GridSpec alone."""
    origin = np.asarray(grid.origin, dtype=np.float64)
    spacing = np.asarray(grid.spacing, dtype=np.float64)
    start_index = np.asarray(edges.index, dtype=np.float64)
    end_index = start_index + np.eye(3)[np.asarray(edges.axis, dtype=np.int64)]
    return origin + start_index * spacing, origin + end_index * spacing


class TestPlane:
    """Axis-aligned plane f(p) = p[2] - 0.1 on an 8^3 grid."""

    GRID = GridSpec.from_bounds((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0), 8)

    @staticmethod
    def plane(h):
        def sdf(p):
            return p[2] - h

        return sdf

    def test_edge_count_and_axes(self):
        edges, _ = detect_edges(self.plane(0.1), self.GRID)
        assert edges.count == 81  # one z-edge per (x, y) lattice column: 9 * 9
        np.testing.assert_array_equal(np.asarray(edges.axis), np.full((81,), 2))

    def test_root_position(self):
        _, hermite = detect_edges(self.plane(0.1), self.GRID)
        np.testing.assert_allclose(np.asarray(hermite.points[:, 2]), 0.1, atol=1e-6)

    def test_gradient_of_summed_z_wrt_offset(self):
        edges, _ = detect_edges(self.plane(0.1), self.GRID)

        def summed_z(h):
            hermite = edge_hermite_data(self.plane(h), self.GRID, edges)
            return jnp.sum(hermite.points[:, 2])

        gradient = jax.grad(summed_z)(jnp.float32(0.1))
        # Every root moves 1:1 with the plane offset, so d/dh sum(z) == edge count.
        np.testing.assert_allclose(float(gradient), 81.0, rtol=1e-6)


class TestSphereBaseline:
    def test_residuals_with_lattice_vertices_exactly_on_surface(self):
        # Regression test for the frozen start_inside bisection: this grid has
        # lattice vertices lying EXACTLY on the unit sphere (e.g. (0, -0.8, 0.6),
        # since 0.8^2 + 0.6^2 == 1). Detection classifies value == 0 as outside,
        # but re-evaluating the field at such a vertex in float32 gives sign
        # noise; bisection must bracket against the frozen detection-time
        # orientation, not the live sign, or these edges lose their root.
        grid = sphere_grid()
        edges, hermite = detect_edges(sphere_sdf(1.0), grid)
        assert edges.count == 1830
        radii = np.linalg.norm(np.asarray(hermite.points), axis=-1)
        np.testing.assert_allclose(radii, 1.0, atol=1e-5)
        assert np.max(np.abs(np.asarray(hermite.values))) < 1e-5

    def test_detection_independence(self):
        # The reported edges must be reconstructible from the GridSpec alone and
        # must straddle the level under the 'exact zero counts as outside'
        # convention when the field is re-evaluated at the recomputed endpoints.
        grid = sphere_grid()
        sdf = sphere_sdf(1.0)
        values = sample_grid(sdf, grid)
        edges = find_crossing_edges(values)

        starts, ends = edge_endpoints(grid, edges)
        f_start = np.asarray(jax.vmap(sdf)(jnp.asarray(starts)), dtype=np.float64)
        f_end = np.asarray(jax.vmap(sdf)(jnp.asarray(ends)), dtype=np.float64)
        start_inside = f_start < 0  # value - level == 0 counts as OUTSIDE
        end_inside = f_end < 0
        assert bool(np.all(start_inside != end_inside))
        np.testing.assert_array_equal(start_inside, np.asarray(edges.start_inside))

        # Brute-force recount from the sampled lattice.
        inside = values < 0
        recount = 0
        for axis in range(3):
            leading = tuple(slice(None, -1) if a == axis else slice(None) for a in range(3))
            trailing = tuple(slice(1, None) if a == axis else slice(None) for a in range(3))
            recount += int(np.sum(inside[leading] != inside[trailing]))
        assert recount == edges.count

    def test_implicit_gradient_matches_analytic(self):
        grid = sphere_grid()
        edges, hermite = detect_edges(sphere_sdf(1.0), grid)

        def t_of_radius(radius):
            return edge_hermite_data(sphere_sdf(radius), grid, edges).t

        jacobian = np.asarray(jax.jacobian(t_of_radius)(jnp.float32(1.0)))
        # f(start + t d) = ||p|| - r = 0 differentiated in r: dt/dr = r / (p . d),
        # with d the edge vector (grid spacing along the edge axis).
        spacing = np.asarray(grid.spacing)
        directions = np.eye(3)[np.asarray(edges.axis, dtype=np.int64)] * spacing
        points = np.asarray(hermite.points, dtype=np.float64)
        analytic = 1.0 / np.sum(points * directions, axis=-1)
        np.testing.assert_allclose(jacobian, analytic, rtol=1e-4)


class TestFunctionalizePath:
    def test_sphere_radius_gradient_vs_finite_differences(self):
        shape = Sphere(Scalar(value=1.0, free=True, name="radius"))
        compiled = functionalize(shape)
        free, fixed, _ = extract_parameters(shape)
        grid = sphere_grid()
        edges, _ = detect_edges(compiled(free, fixed), grid)

        def loss(free_params):
            sdf = compiled(free_params, fixed)
            hermite = edge_hermite_data(sdf, grid, edges)
            return jnp.sum(hermite.points**2)

        gradient = jax.grad(loss)(free)["radius"]
        eps = 1e-3
        upper = float(loss({"radius": jnp.asarray(free["radius"]) + eps}))
        lower = float(loss({"radius": jnp.asarray(free["radius"]) - eps}))
        np.testing.assert_allclose(float(gradient), (upper - lower) / (2 * eps), rtol=1e-2)

    def test_box_size_gradient_vs_finite_differences(self):
        # Regression test: parameter gradients through edge_hermite_data were
        # identically zero for an axis-aligned Box.  The Newton step converges
        # bit-exactly onto the face plane, and Box.sdf's old epsilon-smoothed
        # sqrt had an exact stationary point there, zeroing both the residual
        # and slope derivatives.  Box.sdf now branches on inside/outside so
        # on-surface points get a valid one-sided subgradient.
        shape = Box(Vector([0.4, 0.5, 0.6], free=True, name="size"))
        compiled = functionalize(shape)
        free, fixed, _ = extract_parameters(shape)
        grid = GridSpec.from_bounds((-0.853, -0.853, -0.853), (1.7, 1.7, 1.7), 17)
        edges, _ = detect_edges(compiled(free, fixed), grid)
        assert edges.count > 0

        def loss(free_params):
            sdf = compiled(free_params, fixed)
            hermite = edge_hermite_data(sdf, grid, edges)
            return jnp.sum(hermite.points**2)

        gradient = np.asarray(jax.grad(loss)(free)["size"])
        eps = 1e-3
        size = np.asarray(free["size"], dtype=np.float64)
        for component in range(3):
            delta = np.zeros(3)
            delta[component] = eps
            upper = float(loss({"size": jnp.asarray(size + delta, dtype=jnp.float32)}))
            lower = float(loss({"size": jnp.asarray(size - delta, dtype=jnp.float32)}))
            np.testing.assert_allclose(gradient[component], (upper - lower) / (2 * eps), rtol=1e-2)


class TestHardCsg:
    # Lattice vertices sit at odd multiples of 0.05, so none lies exactly on
    # either sphere (three odd squares sum to 3 mod 8, never to 400): a vertex
    # exactly on a sphere reads as float noise of either sign and can turn a
    # tangency into a spurious crossing edge with a near-zero Newton slope.
    GRID = GridSpec.from_bounds((-1.25, -1.25, -1.25), (2.9, 2.5, 2.5), (29, 25, 25))
    SHIFT = jnp.asarray([0.5, 0.0, 0.0])

    @classmethod
    def csg(cls, combine, radius):
        def sdf(p):
            return combine(
                jnp.sqrt(jnp.sum(p * p)) - radius,
                jnp.sqrt(jnp.sum((p - cls.SHIFT) ** 2)) - 1.0,
            )

        return sdf

    @pytest.mark.parametrize("combine", [jnp.minimum, jnp.maximum], ids=["union", "intersection"])
    def test_residuals_and_finite_gradients(self, combine):
        edges, hermite = detect_edges(self.csg(combine, 1.0), self.GRID)
        assert edges.count > 0
        assert np.max(np.abs(np.asarray(hermite.values))) < 1e-5
        for array in hermite:
            assert np.isfinite(np.asarray(array)).all()

        def loss(radius):
            data = edge_hermite_data(self.csg(combine, radius), self.GRID, edges)
            return jnp.sum(data.points**2)

        gradient = jax.grad(loss)(jnp.float32(1.0))
        assert np.isfinite(float(gradient))


class TestTransformationsAndReuse:
    def test_jit_equivalence_and_grad_under_jit(self):
        grid = sphere_grid()
        edges, _ = detect_edges(sphere_sdf(1.0), grid)

        def points_of(radius):
            return edge_hermite_data(sphere_sdf(radius), grid, edges).points

        eager = np.asarray(points_of(jnp.float32(1.0)))
        jitted = np.asarray(jax.jit(points_of)(jnp.float32(1.0)))
        np.testing.assert_allclose(jitted, eager, atol=1e-6)

        def loss(radius):
            return jnp.sum(points_of(radius) ** 2)

        gradient = jax.jit(jax.grad(loss))(jnp.float32(1.0))
        assert np.isfinite(float(gradient))
        # Loss is edge_count * radius^2 on a sphere, so d/dr at 1 is 2 * count.
        np.testing.assert_allclose(float(gradient), 2.0 * edges.count, rtol=1e-4)

    def test_frozen_edge_reuse_at_perturbed_radius(self):
        grid = sphere_grid()
        edges, _ = detect_edges(sphere_sdf(1.0), grid)
        grown = sphere_sdf(1.05)
        hermite = edge_hermite_data(grown, grid, edges)

        starts, ends = edge_endpoints(grid, edges)
        f_start = np.linalg.norm(starts, axis=-1) - 1.05
        f_end = np.linalg.norm(ends, axis=-1) - 1.05
        still_straddling = (f_start < 0) != (f_end < 0)
        assert int(np.sum(still_straddling)) > 0
        residuals = np.abs(np.asarray(hermite.values))
        assert np.max(residuals[still_straddling]) < 1e-5

    def test_newton_steps_zero_gives_zero_parameter_gradient(self):
        # Documented boundary: with newton_steps=0 the output is built purely
        # from stop_gradient bisection values, so parameter gradients vanish.
        grid = sphere_grid()
        edges, _ = detect_edges(sphere_sdf(1.0), grid)

        def loss(radius):
            data = edge_hermite_data(sphere_sdf(radius), grid, edges, newton_steps=0)
            return jnp.sum(data.points**2)

        gradient = jax.grad(loss)(jnp.float32(1.0))
        assert float(gradient) == 0.0

    def test_empty_edge_set(self):
        grid = sphere_grid()

        def far_sphere(p):
            return jnp.sqrt(jnp.sum((p - jnp.asarray([10.0, 0.0, 0.0])) ** 2)) - 1.0

        edges, hermite = detect_edges(far_sphere, grid)
        assert edges.count == 0
        assert hermite.points.shape == (0, 3)
        assert hermite.gradients.shape == (0, 3)
        assert hermite.t.shape == (0,)
        assert hermite.values.shape == (0,)

    def test_nonzero_level(self):
        grid = sphere_grid()
        edges, hermite = detect_edges(sphere_sdf(1.0), grid, level=0.2)
        assert edges.count > 0
        radii = np.linalg.norm(np.asarray(hermite.points), axis=-1)
        np.testing.assert_allclose(radii, 1.2, atol=1e-5)


class TestValidation:
    def test_grid_spec_rejects_bad_resolution(self):
        with pytest.raises(ValueError):
            GridSpec.from_bounds((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0), 0)
        with pytest.raises(ValueError):
            GridSpec.from_bounds((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0), (4, 4))
        with pytest.raises(ValueError):
            GridSpec.from_bounds((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0), (4, 4, -1))

    def test_grid_spec_rejects_bad_size(self):
        with pytest.raises(ValueError):
            GridSpec.from_bounds((-1.0, -1.0, -1.0), (2.0, 0.0, 2.0), 4)
        with pytest.raises(ValueError):
            GridSpec.from_bounds((-1.0, -1.0, -1.0), (2.0, -1.0, 2.0), 4)
        with pytest.raises(ValueError):
            GridSpec.from_bounds((-1.0, -1.0, -1.0), (2.0, np.nan, 2.0), 4)

    def test_grid_spec_rejects_bad_bounds(self):
        with pytest.raises(ValueError):
            GridSpec.from_bounds((np.inf, -1.0, -1.0), (2.0, 2.0, 2.0), 4)

    def test_find_crossing_edges_rejects_bad_values(self):
        with pytest.raises(ValueError):
            find_crossing_edges(np.zeros((4, 4)))
        with pytest.raises(ValueError):
            find_crossing_edges(np.zeros((1, 4, 4)))
        with pytest.raises(ValueError):
            find_crossing_edges(np.zeros((4, 4, 4)), level=np.nan)

    def test_edge_hermite_data_rejects_bad_arguments(self):
        grid = GridSpec.from_bounds((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0), 4)
        sdf = sphere_sdf(1.0)
        edges = find_crossing_edges(sample_grid(sdf, grid))
        with pytest.raises(ValueError):
            edge_hermite_data(sdf, grid, edges, bisection_iterations=-1)
        with pytest.raises(ValueError):
            edge_hermite_data(sdf, grid, edges, newton_steps=-1)
        with pytest.raises(ValueError):
            edge_hermite_data(sdf, grid, edges, level=np.inf)

    def test_sample_grid_rejects_bad_field_output(self):
        grid = GridSpec.from_bounds((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0), 4)
        with pytest.raises(ValueError):
            sample_grid(lambda p: p, grid)  # vector output, not one scalar per point
        with pytest.raises(ValueError):
            sample_grid(lambda p: p[0] / 0.0, grid)  # non-finite values


class TestGradientFallback:
    def test_polygon_walls_yield_valid_normals(self):
        # Regression: piecewise-linear walls let the secant land bit-exactly
        # on the surface, where polygon SDFs used to have a dead subgradient
        # (epsilon-smoothed sqrt at zero).  Degenerate root gradients now
        # fall back to a nudged evaluation on the same smooth branch.
        from jaxcad.sdf.primitives.polygon import ExtrudedPolygon

        profile = [
            jnp.array([-1.1, -0.7]),
            jnp.array([1.1, -0.7]),
            jnp.array([1.1, 0.3]),
            jnp.array([0.0, 1.0]),
            jnp.array([-1.1, 0.3]),
        ]

        def sdf(p):
            return ExtrudedPolygon.sdf(
                p, jnp.float32(1.0), **{f"v{i}": v for i, v in enumerate(profile)}
            )

        grid = GridSpec.from_bounds((-1.45, -1.05, -0.85), (2.9, 2.4, 1.7), (29, 24, 17))
        edges, hermite = detect_edges(sdf, grid)
        assert edges.count > 0
        unit = np.asarray(hermite.unit_normals())
        degenerate = np.sum(unit**2, axis=1) < 0.25
        # Every sample carries a usable normal; allow a tiny residue for
        # roots landing exactly on profile corners.
        assert degenerate.mean() < 0.005
