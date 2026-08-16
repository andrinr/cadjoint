"""CSG stress scenes for the full meshing pipeline.

Each scene is a hard CSG combination (``jnp.minimum``/``jnp.maximum`` over
primitive SDFs) meshed with :func:`jaxcad.meshing.extract_mesh` and checked
for watertightness, topology (Euler characteristic), signed volume against
an analytic or numerically integrated reference, vertex residuals, and
extraction determinism.  Selected scenes additionally verify that octree
pruning (``lipschitz=1.0``) reproduces the dense extraction exactly.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxcad.meshing import GridSpec, Mesh, extract_mesh
from jaxcad.sdf.primitives import Box, Cylinder, Sphere


def undirected_edge_counts(faces: np.ndarray) -> np.ndarray:
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, counts = np.unique(np.sort(directed, axis=1), axis=0, return_counts=True)
    return counts


def euler_characteristic(vertex_count: int, faces: np.ndarray) -> int:
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edge_count = np.unique(np.sort(directed, axis=1), axis=0).shape[0]
    return vertex_count - edge_count + faces.shape[0]


def signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    return float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0)


def integrated_volume(sdf: Callable, grid: GridSpec, resolution: int = 128) -> float:
    """Numerically integrate the inside volume by cell-center sampling."""
    origin = np.asarray(grid.origin)
    extent = np.asarray(grid.spacing) * np.asarray(grid.cells)
    axes = [origin[a] + (np.arange(resolution) + 0.5) * extent[a] / resolution for a in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    values = np.asarray(jax.vmap(sdf)(jnp.asarray(points, dtype=jnp.float32)))
    voxel = float(np.prod(extent / resolution))
    return float(np.count_nonzero(values < 0.0) * voxel)


# --- Primitive helpers (per-point SDF lambdas over hard min/max CSG) -----

SPHERE_R = 1.0
SPHERE_OFFSET = jnp.array([0.8, 0.0, 0.0])


def two_sphere_union(p):
    return jnp.minimum(
        Sphere.sdf(p, SPHERE_R),
        Sphere.sdf(p - SPHERE_OFFSET, SPHERE_R),
    )


TRI_R = 0.7
TRI_CENTERS = jnp.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.45, 0.9 * np.sqrt(3.0) / 2.0, 0.0]])


def three_sphere_union(p):
    d = Sphere.sdf(p - TRI_CENTERS[0], TRI_R)
    d = jnp.minimum(d, Sphere.sdf(p - TRI_CENTERS[1], TRI_R))
    return jnp.minimum(d, Sphere.sdf(p - TRI_CENTERS[2], TRI_R))


def lens_intersection(p):
    return jnp.maximum(
        Sphere.sdf(p, SPHERE_R),
        Sphere.sdf(p - SPHERE_OFFSET, SPHERE_R),
    )


BOX_HALF = jnp.array([0.5, 0.5, 0.5])
BALL_R = 0.65


def box_sphere_intersection(p):
    return jnp.maximum(Box.sdf(p, BOX_HALF), Sphere.sdf(p, BALL_R))


HOLE_R = 0.25


def box_minus_cylinder(p):
    return jnp.maximum(Box.sdf(p, BOX_HALF), -Cylinder.sdf(p, HOLE_R, 0.7))


INNER_HALF = jnp.array([0.25, 0.25, 0.6])


def box_minus_box(p):
    return jnp.maximum(Box.sdf(p, BOX_HALF), -Box.sdf(p, INNER_HALF))


GRAZE_HALF = jnp.array([0.4, 0.4, 0.4])
GRAZE_R = 0.3
GRAZE_CENTER = jnp.array([0.68, 0.0, 0.0])


def grazing_union(p):
    return jnp.minimum(Box.sdf(p, GRAZE_HALF), Sphere.sdf(p - GRAZE_CENTER, GRAZE_R))


# --- Analytic volume references ------------------------------------------


def lens_volume(r: float, d: float) -> float:
    """Volume of the intersection of two radius-``r`` spheres ``d`` apart."""
    return float(np.pi * (4.0 * r + d) * (2.0 * r - d) ** 2 / 12.0)


def cap_volume(r: float, h: float) -> float:
    """Volume of a spherical cap of height ``h`` on a radius-``r`` sphere."""
    return float(np.pi * h * h * (3.0 * r - h) / 3.0)


SPHERE_VOL = 4.0 / 3.0 * np.pi * SPHERE_R**3
TWO_SPHERE_UNION_VOL = 2.0 * SPHERE_VOL - lens_volume(SPHERE_R, 0.8)
LENS_VOL = lens_volume(SPHERE_R, 0.8)
# Sphere r=0.65 clipped by the unit box: six non-overlapping caps of height
# r - 0.5 removed (edge distance sqrt(0.5) > r, so caps never meet).
BOX_SPHERE_VOL = 4.0 / 3.0 * np.pi * BALL_R**3 - 6.0 * cap_volume(BALL_R, BALL_R - 0.5)
BOX_MINUS_CYL_VOL = 1.0 - np.pi * HOLE_R**2 * 1.0
BOX_MINUS_BOX_VOL = 1.0 - 0.5 * 0.5 * 1.0
# Grazing union: box + sphere - the height-0.02 cap that dips into the box
# (contact circle radius ~0.11 < 0.4, entirely within the face).
GRAZING_VOL = 0.8**3 + 4.0 / 3.0 * np.pi * GRAZE_R**3 - cap_volume(GRAZE_R, 0.02)


class Scene(NamedTuple):
    """One CSG stress scene and its expected mesh invariants."""

    name: str
    sdf: Callable
    grid: GridSpec
    euler: int
    volume: float | None  # None: integrate numerically over the scene grid.
    residual_tol: float = 2e-2


# Grids use offbeat bounds so primitive faces never land on lattice planes,
# except scene 6 where the lattice-aligned coincident planes are the point.
SCENES = [
    Scene(
        "two_sphere_union",
        two_sphere_union,
        GridSpec.from_bounds((-1.29, -1.31, -1.28), (3.42, 2.62, 2.61), (32, 26, 26)),
        euler=2,
        volume=TWO_SPHERE_UNION_VOL,
    ),
    Scene(
        "three_sphere_union",
        three_sphere_union,
        GridSpec.from_bounds((-0.94, -0.93, -0.95), (2.77, 2.66, 1.9), (30, 28, 22)),
        euler=2,
        volume=None,
    ),
    Scene(
        "lens_intersection",
        lens_intersection,
        GridSpec.from_bounds((-0.44, -1.14, -1.13), (1.69, 2.29, 2.28), (20, 26, 26)),
        euler=2,
        volume=LENS_VOL,
    ),
    Scene(
        "box_sphere_intersection",
        box_sphere_intersection,
        GridSpec.from_bounds((-0.77, -0.76, -0.78), (1.54, 1.53, 1.55), (24, 24, 24)),
        euler=2,
        volume=BOX_SPHERE_VOL,
    ),
    Scene(
        "box_minus_cylinder",
        box_minus_cylinder,
        GridSpec.from_bounds((-0.77, -0.76, -0.78), (1.54, 1.53, 1.55), (24, 24, 24)),
        euler=0,  # through-hole: torus topology
        volume=BOX_MINUS_CYL_VOL,
    ),
    Scene(
        "box_minus_box_flush",
        box_minus_box,
        # Spacing 0.0625 puts every box face plane (x, y = +/-0.25, +/-0.5 and
        # z = +/-0.5) exactly on lattice planes: coincident-face stress test.
        GridSpec.from_bounds((-0.75, -0.75, -0.75), (1.5, 1.5, 1.5), 24),
        euler=0,  # rectangular through-hole: torus topology
        volume=BOX_MINUS_BOX_VOL,
    ),
    Scene(
        "grazing_union",
        grazing_union,
        GridSpec.from_bounds((-0.61, -0.61, -0.61), (1.7, 1.22, 1.22), (34, 24, 24)),
        euler=2,
        volume=GRAZING_VOL,
    ),
]
SCENE_IDS = [scene.name for scene in SCENES]
SPARSE_SCENES = [
    s
    for s in SCENES
    if s.name in ("two_sphere_union", "box_sphere_intersection", "box_minus_box_flush")
]

_MESH_CACHE: dict[str, Mesh] = {}


def get_mesh(scene: Scene) -> Mesh:
    if scene.name not in _MESH_CACHE:
        _MESH_CACHE[scene.name] = extract_mesh(scene.sdf, scene.grid)
    return _MESH_CACHE[scene.name]


@pytest.mark.parametrize("scene", SCENES, ids=SCENE_IDS)
class TestScene:
    def test_watertight_manifold(self, scene: Scene):
        mesh = get_mesh(scene)
        assert mesh.faces.shape[0] > 0
        counts = undirected_edge_counts(mesh.faces)
        np.testing.assert_array_equal(np.unique(counts), [2])

    def test_euler_characteristic(self, scene: Scene):
        mesh = get_mesh(scene)
        assert euler_characteristic(mesh.vertices.shape[0], mesh.faces) == scene.euler

    def test_signed_volume(self, scene: Scene):
        mesh = get_mesh(scene)
        volume = signed_volume(np.asarray(mesh.vertices, dtype=np.float64), mesh.faces)
        assert volume > 0
        reference = (
            scene.volume if scene.volume is not None else integrated_volume(scene.sdf, scene.grid)
        )
        np.testing.assert_allclose(volume, reference, rtol=0.1)

    def test_vertices_near_surface(self, scene: Scene):
        mesh = get_mesh(scene)
        residuals = np.abs(np.asarray(jax.vmap(scene.sdf)(mesh.vertices)))
        assert residuals.max() < scene.residual_tol

    def test_extraction_is_deterministic(self, scene: Scene):
        first = get_mesh(scene)
        second = extract_mesh(scene.sdf, scene.grid)
        np.testing.assert_array_equal(first.faces, second.faces)
        np.testing.assert_array_equal(first.quads, second.quads)
        np.testing.assert_array_equal(np.asarray(first.vertices), np.asarray(second.vertices))


# Regression test for the manifold-DC cell split: when a CSG crease runs
# nearly tangent to a lattice plane (here the seam circles of the
# three-sphere union graze z = 0.568), two surface sheets cross the same
# cell.  Uniform DC's single vertex per cell fused them into edges shared by
# FOUR triangles with vertices ~0.033 off the surface; the per-component
# rows of manifold_cell_incidence give each sheet its own QEF vertex, so the
# mesh stays manifold and the vertices stay on the surface.
def test_grazing_crease_stays_manifold():
    grid = GridSpec.from_bounds((-0.94, -0.93, -0.92), (2.77, 2.66, 1.86), (30, 28, 20))
    mesh = extract_mesh(three_sphere_union, grid)
    counts = undirected_edge_counts(mesh.faces)
    np.testing.assert_array_equal(np.unique(counts), [2])
    residuals = np.abs(np.asarray(jax.vmap(three_sphere_union)(mesh.vertices)))
    assert residuals.max() < 2e-2


@pytest.mark.parametrize("scene", SPARSE_SCENES, ids=[s.name for s in SPARSE_SCENES])
def test_sparse_detection_matches_dense(scene: Scene):
    dense = get_mesh(scene)
    sparse = extract_mesh(scene.sdf, scene.grid, lipschitz=1.0)
    np.testing.assert_array_equal(dense.faces, sparse.faces)
    np.testing.assert_array_equal(dense.quads, sparse.quads)
    np.testing.assert_allclose(np.asarray(dense.vertices), np.asarray(sparse.vertices), atol=1e-6)
