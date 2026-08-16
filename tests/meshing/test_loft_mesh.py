"""Meshing tests for lofted solids (LoftedPolygon through extract_mesh)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxcad.construction import PolygonProfile, loft
from jaxcad.meshing import GridSpec, extract_mesh
from tests.meshing.test_dual_contouring import (
    euler_characteristic,
    signed_volume,
    undirected_edge_counts,
)

# Bottom: axis-aligned square. Top: smaller square rotated 45 degrees, with
# vertex i of the top matched to vertex i of the bottom (same winding).
BOTTOM = [[-0.6, -0.6], [0.6, -0.6], [0.6, 0.6], [-0.6, 0.6]]
TOP = [[0.0, -0.35], [0.35, 0.0], [0.0, 0.35], [-0.35, 0.0]]
HEIGHT = 1.2

# Spacing 0.1 with lattice planes at odd multiples of 0.05, so no lattice
# vertex lies exactly on the caps (z = +/-0.6) or the bottom walls (+/-0.6).
GRID = GridSpec.from_bounds((-0.85, -0.85, -0.85), (1.7, 1.7, 1.7), 17)


def loft_solid():
    return loft(
        PolygonProfile(BOTTOM, name="bottom"),
        PolygonProfile(TOP, name="top"),
        height=HEIGHT,
    )


def analytic_volume() -> float:
    """Exact volume: slice area is quadratic in z, so Simpson is exact."""

    def area(t: float) -> float:
        v = (1.0 - t) * np.asarray(BOTTOM) + t * np.asarray(TOP)
        rolled = np.roll(v, -1, axis=0)
        return 0.5 * abs(float(np.sum(v[:, 0] * rolled[:, 1] - rolled[:, 0] * v[:, 1])))

    return HEIGHT / 6.0 * (area(0.0) + 4.0 * area(0.5) + area(1.0))


class TestLoftMesh:
    def extract(self):
        return extract_mesh(loft_solid(), GRID)

    def test_watertight_manifold(self):
        mesh = self.extract()
        counts = undirected_edge_counts(mesh.faces)
        # Closed manifold: every undirected edge is shared by exactly two triangles.
        np.testing.assert_array_equal(np.unique(counts), [2])
        assert euler_characteristic(mesh.vertices.shape[0], mesh.faces) == 2

    def test_outward_orientation_and_volume(self):
        mesh = self.extract()
        volume = signed_volume(np.asarray(mesh.vertices, dtype=np.float64), mesh.faces)
        assert volume > 0  # counterclockwise from outside
        np.testing.assert_allclose(volume, analytic_volume(), rtol=5e-2)

    def test_caps_are_at_half_height(self):
        mesh = self.extract()
        z = np.asarray(mesh.vertices, dtype=np.float64)[:, 2]
        np.testing.assert_allclose(z.min(), -HEIGHT / 2.0, atol=5e-3)
        np.testing.assert_allclose(z.max(), HEIGHT / 2.0, atol=5e-3)

    def test_vertices_near_surface(self):
        mesh = self.extract()
        solid = loft_solid()
        residuals = np.abs(np.asarray(jax.vmap(solid)(jnp.asarray(mesh.vertices))))
        # The lofted field is a distance bound with mildly curved ruled walls;
        # QEF vertices still land within a fraction of the cell size.
        assert residuals.max() < 1e-2
