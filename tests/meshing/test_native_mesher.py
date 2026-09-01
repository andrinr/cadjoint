"""Parity tests for the native (Rust) dual-contouring core.

Every test compares the native pipeline (`cadjoint.meshing.native`) against
the Python reference on the same inputs: discrete topology must be
bit-identical, continuous quantities must agree to 1e-6, and gradients
through the tesseract-backed QEF must match the reference JAX autodiff and
finite differences. The whole module skips cleanly when the cdylib has not
been built.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cadjoint.meshing import native
from cadjoint.meshing.dual_contouring import (
    dual_faces,
    extract_mesh,
    qef_vertices,
    sharp_qef_vertices,
)
from cadjoint.meshing.edge_detection import (
    GridSpec,
    edge_hermite_data,
    find_crossing_edges,
    sample_grid,
)
from cadjoint.meshing.features import manifold_cell_incidence
from cadjoint.sdf.boolean.smooth import smooth_min
from cadjoint.sdf.primitives.box import Box
from cadjoint.sdf.primitives.cylinder import Cylinder
from cadjoint.sdf.primitives.polygon import ExtrudedPolygon

pytestmark = pytest.mark.skipif(
    not native.native_available(),
    reason=(
        "native mesher cdylib not built (cargo build --release --manifest-path native/Cargo.toml)"
    ),
)


@pytest.fixture
def enable_x64():
    """Enable jax x64 for one test, restoring the caller's setting."""
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def box_sdf(p):
    return Box.sdf(p, jnp.asarray([0.4, 0.5, 0.6], dtype=jnp.asarray(p).dtype))


def box_union_sphere_sdf(p):
    p = jnp.asarray(p)
    sphere = jnp.sqrt(jnp.sum((p - jnp.asarray([0.5, 0.4, 0.3], dtype=p.dtype)) ** 2)) - 0.7
    return jnp.minimum(box_sdf(p), sphere)


def cylinder_sdf(p):
    return Cylinder.sdf(jnp.asarray(p), 0.5, 0.6)


def bracket_sdf(p):
    """The L-bracket of ``examples/fem_bracket_optimization.py`` (nominal)."""
    p = jnp.asarray(p)
    half_plate = 0.1
    plate = Box.sdf(
        p - jnp.array([0.0, 0.0, 1.0], dtype=p.dtype) * half_plate,
        jnp.asarray([1.2, 0.8, half_plate], dtype=p.dtype),
    )
    q_web = jnp.stack([p[..., 0], p[..., 2], p[..., 1] + 0.7], axis=-1)
    web = ExtrudedPolygon.sdf(
        q_web,
        depth=0.16,
        v0=jnp.array([-1.1, 0.0]),
        v1=jnp.array([1.1, 0.0]),
        v2=jnp.array([0.85, 1.2]),
        v3=jnp.array([-0.85, 1.2]),
    )
    q_rib = jnp.stack([p[..., 1], p[..., 2], p[..., 0]], axis=-1)
    rib = ExtrudedPolygon.sdf(
        q_rib,
        depth=0.12,
        v0=jnp.array([0.55, 0.02]),
        v1=jnp.array([-0.62, 0.02]),
        v2=jnp.array([-0.62, 0.88]),
    )
    body = smooth_min(smooth_min(plate, web, 0.05), rib, 0.05)
    for bolt_x in (-0.7, 0.7):
        hole = Cylinder.sdf(p - jnp.array([bolt_x, 0.35, half_plate], dtype=p.dtype), 0.16, 0.2)
        body = jnp.maximum(body, -hole)
    return body


# Offbeat bounds so primitive faces never land exactly on lattice planes.
SCENES = {
    "box": (box_sdf, (-0.85, -0.95, -1.05), (1.7, 1.9, 2.1)),
    "box_union_sphere": (box_union_sphere_sdf, (-0.53, -0.61, -0.72), (1.86, 1.83, 1.84)),
    "cylinder": (cylinder_sdf, (-0.63, -0.61, -0.77), (1.26, 1.22, 1.54)),
    "bracket": (bracket_sdf, (-1.3, -0.95, -0.06), (2.6, 1.9, 1.42)),
}


def scene_grid(name: str, resolution: int) -> GridSpec:
    _sdf, bounds, size = SCENES[name]
    return GridSpec.from_bounds(bounds, size, resolution)


def assert_faces_equal_up_to_tied_diagonals(mesh, mesh_native) -> None:
    """Faces must match bit for bit except on exactly-tied quad diagonals.

    The shorter-diagonal triangulation is a frozen discrete choice, but on
    a near-symmetric quad (degenerate rectangle, common on the bracket's
    coplanar regions) the diagonal gap sits below the float32 quantization
    noise of the vertex coordinates, and a one-ulp difference between
    LAPACK SVD and the native Jacobi solver flips it. Both triangulations
    of such a quad are valid; any flip on a quad whose diagonal gap exceeds
    the f32 noise floor is a real topology error and still fails.
    """
    if np.array_equal(mesh.faces, mesh_native.faces):
        return
    differing = np.flatnonzero(np.any(mesh.faces != mesh_native.faces, axis=1))
    positions = np.asarray(mesh.vertices, dtype=np.float64)
    f32_epsilon = float(np.finfo(np.float32).eps)
    for quad_index in np.unique(differing // 2):
        a, b, c, d = (int(value) for value in mesh.quads[quad_index])
        pair = slice(2 * quad_index, 2 * quad_index + 2)
        triangulations = [
            {(a, b, c), (a, c, d)},
            {(a, b, d), (b, c, d)},
        ]
        for faces in (mesh.faces[pair], mesh_native.faces[pair]):
            received = {tuple(int(v) for v in triangle) for triangle in faces}
            assert received in triangulations, f"invalid triangulation on quad {quad_index}"
        diagonal_ac = float(np.sum((positions[a] - positions[c]) ** 2))
        diagonal_bd = float(np.sum((positions[b] - positions[d]) ** 2))
        coordinate_scale = max(1.0, float(np.max(np.abs(positions[[a, b, c, d]]))))
        noise_floor = 32.0 * f32_epsilon * coordinate_scale**2
        assert (
            abs(diagonal_ac - diagonal_bd) <= noise_floor
        ), f"diagonal flip on non-tied quad {quad_index}"


@pytest.mark.parametrize("name", sorted(SCENES))
def test_discrete_stages_bit_identical(name):
    """Edges, incidence, and faces match the reference bit for bit."""
    sdf = SCENES[name][0]
    grid = scene_grid(name, 16)
    values = sample_grid(sdf, grid)
    edges = find_crossing_edges(values)
    edges_native = native.find_crossing_edges_native(values)
    np.testing.assert_array_equal(edges.axis, edges_native.axis)
    np.testing.assert_array_equal(edges.index, edges_native.index)
    np.testing.assert_array_equal(edges.start_inside, edges_native.start_inside)
    assert edges.count > 0

    inside = values < 0.0
    incidence = manifold_cell_incidence(edges, grid, inside)
    incidence_native = native.manifold_cell_incidence_native(edges_native, grid, inside)
    np.testing.assert_array_equal(incidence.cells, incidence_native.cells)
    np.testing.assert_array_equal(incidence.edge_ids, incidence_native.edge_ids)
    np.testing.assert_array_equal(incidence.counts, incidence_native.counts)

    hermite = edge_hermite_data(sdf, grid, edges)
    vertices = sharp_qef_vertices(hermite, incidence, grid)
    quads, triangles, skipped = dual_faces(edges, incidence, grid, vertices)
    quads_native, triangles_native, skipped_native = native.dual_faces_native(
        edges_native, incidence_native, grid, vertices
    )
    np.testing.assert_array_equal(quads, quads_native)
    np.testing.assert_array_equal(triangles, triangles_native)
    assert skipped == skipped_native


@pytest.mark.parametrize("resolution", [16, 32])
@pytest.mark.parametrize("name", sorted(SCENES))
def test_extract_mesh_parity(name, resolution):
    """Full pipeline: identical topology, vertices within 1e-6."""
    sdf = SCENES[name][0]
    grid = scene_grid(name, resolution)
    mesh = extract_mesh(sdf, grid)
    mesh_native = native.extract_mesh_native(sdf, grid)
    assert_faces_equal_up_to_tied_diagonals(mesh, mesh_native)
    np.testing.assert_array_equal(mesh.quads, mesh_native.quads)
    np.testing.assert_array_equal(mesh.cells, mesh_native.cells)
    np.testing.assert_allclose(
        np.asarray(mesh.vertices), np.asarray(mesh_native.vertices), atol=1e-6, rtol=0
    )
    np.testing.assert_allclose(
        np.asarray(mesh.normals), np.asarray(mesh_native.normals), atol=1e-6, rtol=0
    )


def test_sharp_qef_positions_match(enable_x64):
    """Sharp QEF vertices agree with the NumPy SVD reference to 1e-6 (f64)."""
    sdf = SCENES["box"][0]
    grid = scene_grid("box", 16)
    values = sample_grid(sdf, grid)
    edges = find_crossing_edges(values)
    incidence = manifold_cell_incidence(edges, grid, values < 0.0)
    hermite = edge_hermite_data(sdf, grid, edges)
    reference = sharp_qef_vertices(hermite, incidence, grid)
    placed = native.sharp_qef_vertices_native(hermite, incidence, grid)
    np.testing.assert_allclose(placed, reference, atol=1e-6, rtol=0)


def test_smooth_qef_forward_matches(enable_x64):
    """Tesseract-backed Tikhonov QEF equals the reference solve (f64)."""
    sdf = SCENES["box_union_sphere"][0]
    grid = scene_grid("box_union_sphere", 12)
    values = sample_grid(sdf, grid)
    edges = find_crossing_edges(values)
    incidence = manifold_cell_incidence(edges, grid, values < 0.0)
    hermite = edge_hermite_data(sdf, grid, edges)
    vertices, normals = qef_vertices(hermite, incidence, grid)
    vertices_native, normals_native = native.qef_vertices_native(hermite, incidence, grid)
    np.testing.assert_allclose(np.asarray(vertices_native), np.asarray(vertices), atol=1e-9, rtol=0)
    np.testing.assert_array_equal(np.asarray(normals_native), np.asarray(normals))


def test_qef_vjp_matches_reference_autodiff(enable_x64):
    """Gradients w.r.t. Hermite points and gradients match JAX autodiff."""
    sdf = SCENES["box_union_sphere"][0]
    grid = scene_grid("box_union_sphere", 12)
    values = sample_grid(sdf, grid)
    edges = find_crossing_edges(values)
    incidence = manifold_cell_incidence(edges, grid, values < 0.0)
    hermite = edge_hermite_data(sdf, grid, edges)

    def objective(qef):
        def loss(points, gradients):
            refreshed = hermite._replace(points=points, gradients=gradients)
            vertices, _ = qef(refreshed, incidence, grid)
            return jnp.sum(vertices)

        return jax.grad(loss, argnums=(0, 1))(hermite.points, hermite.gradients)

    reference = objective(qef_vertices)
    result = objective(native.qef_vertices_native)
    for expected, received in zip(reference, result):
        scale = max(1.0, float(jnp.max(jnp.abs(expected))))
        np.testing.assert_allclose(
            np.asarray(received), np.asarray(expected), atol=1e-6 * scale, rtol=0
        )


def test_end_to_end_parameter_gradient(enable_x64):
    """Design-parameter gradient through the JAX refinement layer + native QEF
    matches the reference chain and central finite differences."""

    def sdf_of(scale):
        return lambda p: jnp.sqrt(jnp.sum(p * p)) - scale

    grid = GridSpec.from_bounds((-1.31, -1.29, -1.3), (2.6, 2.6, 2.6), 12)
    values = sample_grid(sdf_of(1.0), grid)
    edges = native.find_crossing_edges_native(values)
    incidence = native.manifold_cell_incidence_native(edges, grid, values < 0.0)

    def loss_with(qef):
        def loss(scale):
            hermite = edge_hermite_data(sdf_of(scale), grid, edges)
            vertices, _ = qef(hermite, incidence, grid)
            return jnp.sum(vertices**2)

        return loss

    native_loss = loss_with(native.qef_vertices_native)
    gradient = float(jax.grad(native_loss)(jnp.float64(1.0)))
    reference = float(jax.grad(loss_with(qef_vertices))(jnp.float64(1.0)))
    step = 1e-4
    finite = float(native_loss(jnp.float64(1.0 + step)) - native_loss(jnp.float64(1.0 - step)))
    finite /= 2 * step
    assert gradient == pytest.approx(reference, rel=1e-9)
    assert gradient == pytest.approx(finite, rel=5e-4)


def test_sparse_detection_matches_dense():
    """The octree-pruned path returns the identical native mesh."""

    def sphere(p):
        return jnp.sqrt(jnp.sum(p * p)) - 1.0

    grid = GridSpec.from_bounds((-1.31, -1.29, -1.3), (2.6, 2.6, 2.6), 16)
    dense = native.extract_mesh_native(sphere, grid)
    sparse = native.extract_mesh_native(sphere, grid, lipschitz=1.0)
    np.testing.assert_array_equal(dense.faces, sparse.faces)
    np.testing.assert_allclose(
        np.asarray(dense.vertices), np.asarray(sparse.vertices), atol=1e-6, rtol=0
    )


def test_empty_surface_yields_empty_mesh():
    def far_away(p):
        return jnp.sqrt(jnp.sum(p * p)) + 1.0

    grid = GridSpec.from_bounds((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0), 8)
    mesh = native.extract_mesh_native(far_away, grid)
    assert mesh.faces.shape == (0, 3)
    assert mesh.vertices.shape == (0, 3)
    assert mesh.quads.shape == (0, 4)


def test_inconsistent_inside_raises():
    def sphere(p):
        return jnp.sqrt(jnp.sum(p * p)) - 1.0

    grid = GridSpec.from_bounds((-1.31, -1.29, -1.3), (2.6, 2.6, 2.6), 8)
    values = sample_grid(sphere, grid)
    edges = native.find_crossing_edges_native(values)
    wrong = np.zeros(grid.lattice_shape, dtype=bool)
    with pytest.raises(ValueError, match="inconsistent"):
        native.manifold_cell_incidence_native(edges, grid, wrong)


def test_qef_vertices_native_requires_x64():
    def sphere(p):
        return jnp.sqrt(jnp.sum(p * p)) - 1.0

    grid = GridSpec.from_bounds((-1.31, -1.29, -1.3), (2.6, 2.6, 2.6), 8)
    values = sample_grid(sphere, grid)
    edges = native.find_crossing_edges_native(values)
    incidence = native.manifold_cell_incidence_native(edges, grid, values < 0.0)
    hermite = edge_hermite_data(sphere, grid, edges)
    if jax.config.jax_enable_x64:
        pytest.skip("x64 already enabled process-wide")
    with pytest.raises(RuntimeError, match="x64"):
        native.qef_vertices_native(hermite, incidence, grid)
