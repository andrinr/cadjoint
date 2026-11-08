"""The meshing surface an integrator builds on (docs/integrating.qmd).

Two millimetre parts carry the tests, the ones text-to-cad's cadgen meshes:
the clevis bracket (plate, lugs, ribs, holes; hard CSG seams beside a
blended root) and the space-frame node (six blended tubes shelled to a 2 mm
wall, opened by a sphere).  Hand-made fields check what has an exact answer.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

import cadjoint.meshing as meshing
from cadjoint.meshing import (
    MIN_WALL_CELLS,
    SurfaceOptions,
    extract_mesh,
    extract_surface,
    mesh_topology,
    surface_grid,
    wall_thickness,
)
from tests.meshing.fixtures import clevis_bracket, space_frame_node

#: The names docs/integrating.qmd documents as the stable surface.
INTEGRATOR_API = (
    "surface_grid",
    "SurfaceOptions",
    "extract_surface",
    "Surface",
    "RenderMesh",
    "CREASE_ANGLE",
    "fit_bounds",
    "boundary_inside",
    "mesh_topology",
    "MeshTopology",
    "wall_thickness",
    "WallThickness",
    "MIN_WALL_CELLS",
    "WALL_CELLS",
    "simplify_mesh",
    "remove_slivers",
    "lattice_feature_edges",
    "save_stl",
    "save_obj",
    "Mesh",
    "GridSpec",
)

HALF = np.array([0.7, 0.5, 0.3])


def box(p):
    return jnp.max(jnp.abs(p) - jnp.asarray(HALF), axis=-1)


def sphere(p):
    return jnp.linalg.norm(p, axis=-1) - 1.0


@pytest.fixture(scope="module")
def clevis():
    scene = clevis_bracket.scene
    surface = extract_surface(
        scene, SurfaceOptions(feature_edges=True), grid=surface_grid(scene, cell=1.5)
    )
    return scene, surface


@pytest.fixture(scope="module")
def node():
    scene = space_frame_node.scene
    return scene, extract_surface(scene, grid=surface_grid(scene, cell=1.0))


def _facet_normals(mesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    a, b, c = (vertices[faces[:, k]] for k in range(3))
    normal = np.cross(b - a, c - a)
    return normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)


def _corner_angles(corner_normals: np.ndarray, facets: np.ndarray) -> np.ndarray:
    cosine = np.einsum("fki,fi->fk", corner_normals, facets)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


# ── the surface ─────────────────────────────────────────────────────────────


def test_the_documented_names_are_public():
    for name in INTEGRATOR_API:
        assert name in meshing.__all__, name
        assert getattr(meshing, name) is not None


def test_a_cell_grid_keeps_the_box_and_never_exceeds_the_cell():
    grid = surface_grid(box, cell=0.13, bounds=((-1, -1, -1), (1, 0.5, 0.2)))
    assert grid.origin == (-1.0, -1.0, -1.0)
    extent = np.asarray(grid.spacing) * np.asarray(grid.cells)
    assert np.allclose(extent, [2.0, 1.5, 1.2])
    assert max(grid.spacing) <= 0.13
    assert grid.cells == (16, 12, 10)


def test_a_cell_grid_fits_the_part_when_no_bounds_are_given():
    grid = surface_grid(box, cell=0.1)
    low = np.asarray(grid.origin)
    high = low + np.asarray(grid.spacing) * np.asarray(grid.cells)
    assert (low < -HALF).all() and (high > HALF).all()
    assert max(grid.spacing) <= 0.1


@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        ({}, "exactly one"),
        ({"resolution": 32, "cell": 0.1}, "exactly one"),
        ({"cell": 0.0}, "cell"),
        ({"cell": -1.0}, "cell"),
        ({"cell": math.inf}, "cell"),
        ({"cell": True}, "cell"),
        ({"cell": 0.1, "bounds": ((0, 0, 0), (1, 1))}, "bounds"),
        ({"cell": 0.1, "bounds": ((0, 0, 0), (1, -1, 1))}, "bounds"),
    ],
)
def test_bad_grids_are_refused(arguments, match):
    with pytest.raises(ValueError, match=match):
        surface_grid(box, **arguments)


def test_the_clevis_is_one_closed_body_with_its_eight_holes(clevis):
    _, surface = clevis
    topology = mesh_topology(surface.mesh)
    assert topology.watertight
    assert topology.bodies == 1
    # Genus 8: the pin hole through two lugs, four bolt holes, two cutouts.
    assert topology.euler_characteristic == 2 - 2 * 8
    assert not surface.clipped


def test_bounds_that_cut_the_part_are_reported_as_clipped():
    grid = surface_grid(box, cell=0.1, bounds=((-0.5, -1, -1), (1, 1, 1)))
    surface = extract_surface(box, grid=grid)
    assert surface.clipped
    assert not mesh_topology(surface.mesh).watertight


def test_bounds_wholly_inside_the_part_hold_no_surface_and_say_so():
    surface = extract_surface(
        box, grid=surface_grid(box, cell=0.05, bounds=((-0.2,) * 3, (0.2,) * 3))
    )
    assert surface.clipped
    assert surface.mesh.faces.shape[0] == 0


def test_the_surface_is_the_dual_contour_welded():
    """On a part with nothing to repair or weld, the front door is plain DC."""
    grid = surface_grid(sphere, cell=0.1)
    surface = extract_surface(sphere, grid=grid)
    reference = extract_mesh(sphere, grid)
    assert np.array_equal(np.asarray(surface.mesh.vertices), np.asarray(reference.vertices))
    assert np.array_equal(surface.mesh.faces, reference.faces)


def test_mesh_topology_counts_bodies_and_open_edges():
    def two(p):
        return jnp.minimum(sphere(p - jnp.array([1.5, 0, 0])), sphere(p + jnp.array([1.5, 0, 0])))

    mesh = extract_surface(two, grid=surface_grid(two, cell=0.15)).mesh
    topology = mesh_topology(mesh)
    assert (topology.bodies, topology.open_edges, topology.euler_characteristic) == (2, 0, 4)
    opened = mesh._replace(faces=np.asarray(mesh.faces)[1:])
    assert mesh_topology(opened).open_edges == 3
    assert not mesh_topology(opened).watertight


# ── crease-split normals ────────────────────────────────────────────────────


def test_a_smooth_part_keeps_its_welded_vertices_and_field_normals():
    surface = extract_surface(sphere, grid=surface_grid(sphere, cell=0.1))
    rendered = surface.render_mesh()
    assert rendered.vertices.shape == np.asarray(surface.mesh.vertices).shape
    assert np.allclose(rendered.normals, np.asarray(surface.mesh.normals)[rendered.rows])
    assert np.array_equal(
        rendered.vertices[rendered.faces], np.asarray(surface.mesh.vertices)[surface.mesh.faces]
    )


def test_a_box_splits_into_one_vertex_per_face_at_every_edge_and_corner():
    surface = extract_surface(box, grid=surface_grid(box, cell=0.1))
    rendered = surface.render_mesh()
    copies = np.bincount(rendered.rows)
    vertices = np.asarray(surface.mesh.vertices, dtype=np.float64)
    on_faces = (np.abs(np.abs(vertices) - HALF) < 1e-4).sum(axis=1)
    # A corner lies on three faces, an edge vertex on two, the rest on one.
    assert np.array_equal(copies, np.maximum(on_faces, 1))
    # Every copy's normal is its face's, to a degree.
    axis_normal = np.zeros_like(rendered.normals)
    dominant = np.abs(rendered.normals).argmax(axis=1)
    axis_normal[np.arange(len(dominant)), dominant] = np.sign(
        rendered.normals[np.arange(len(dominant)), dominant]
    )
    angle = np.degrees(np.arccos(np.clip((rendered.normals * axis_normal).sum(axis=1), -1, 1)))
    assert angle.max() < 1.0


def test_the_clevis_rims_shade_as_their_faces(clevis):
    """The grainy rim: field normals average both sides; split ones do not."""
    _, surface = clevis
    rendered = surface.render_mesh()
    split = np.bincount(rendered.rows, minlength=surface.mesh.vertices.shape[0]) > 1
    faces = np.asarray(surface.mesh.faces)
    at_split = split[faces]
    facets = _facet_normals(surface.mesh)
    field = _corner_angles(np.asarray(surface.mesh.normals, dtype=np.float64)[faces], facets)
    corners = _corner_angles(surface.corner_normals, facets)
    assert split.sum() > 500
    assert np.median(field[at_split]) > 30.0
    assert np.median(corners[at_split]) < 2.0
    assert np.percentile(corners[at_split], 90) < 10.0
    # The render mesh is the same triangles at the same points.
    assert np.array_equal(
        rendered.vertices[rendered.faces], np.asarray(surface.mesh.vertices)[faces]
    )
    assert np.allclose(np.linalg.norm(rendered.normals, axis=1), 1.0, atol=1e-6)


def test_simplified_surfaces_carry_split_normals_too():
    surface = extract_surface(box, SurfaceOptions(simplify=0.05), grid=surface_grid(box, cell=0.1))
    rendered = surface.render_mesh()
    assert surface.corner_normals.shape == (surface.mesh.faces.shape[0], 3, 3)
    assert rendered.vertices.shape[0] > surface.mesh.vertices.shape[0]
    assert mesh_topology(surface.mesh).watertight


# ── feature edges ───────────────────────────────────────────────────────────


def test_feature_edges_leave_the_mesh_as_it_is():
    grid = surface_grid(clevis_bracket.scene, cell=3.0)
    plain = extract_surface(clevis_bracket.scene, grid=grid)
    featured = extract_surface(clevis_bracket.scene, SurfaceOptions(feature_edges=True), grid=grid)
    assert plain.feature_edges.shape == (0, 2, 3)
    assert featured.feature_edges.shape[0] > 0
    assert np.array_equal(np.asarray(plain.mesh.vertices), np.asarray(featured.mesh.vertices))
    assert np.array_equal(plain.mesh.faces, featured.mesh.faces)


def test_a_box_draws_its_twelve_edges():
    surface = extract_surface(
        box, SurfaceOptions(feature_edges=True), grid=surface_grid(box, cell=0.05)
    )
    ends = surface.feature_edges.reshape(-1, 3)
    # Every end lies on an edge: on two of the box's faces at once.
    on_faces = (np.abs(np.abs(ends) - HALF) < 1e-3).sum(axis=1)
    assert (on_faces >= 2).all()
    drawn = np.linalg.norm(surface.feature_edges[:, 1] - surface.feature_edges[:, 0], axis=1).sum()
    assert drawn == pytest.approx(8.0 * HALF.sum(), rel=0.05)


def test_the_clevis_edges_lie_on_its_creases_and_skip_its_flush_seams(clevis):
    scene, surface = clevis
    chords = surface.feature_edges
    ends = chords.reshape(-1, 3)
    assert np.abs(np.asarray(scene(jnp.asarray(ends, jnp.float32)))).max() < 0.1 * surface.cell
    assert (
        np.linalg.norm(chords[:, 1] - chords[:, 0], axis=1).max()
        < 2.0 * math.sqrt(3) * surface.cell
    )
    # Each bolt hole's top rim (7 mm across at z = 10) is drawn ...
    for x, y in ((-45.0, -20.0), (-45.0, 20.0), (45.0, -20.0), (45.0, 20.0)):
        radial = np.linalg.norm(ends[:, :2] - (x, y), axis=1)
        rim = (np.abs(radial - 3.5) < 0.5 * surface.cell) & (np.abs(ends[:, 2] - 10.0) < 0.5)
        assert rim.sum() >= 8, (x, y)
    # ... and the line where a lug's stem meets its crown, flush on the
    # lug's flat faces at z = 34 outside the pin hole, is not.
    flat = (np.abs(np.abs(ends[:, 1]) - np.array([8.0, 26.0])[:, None]) < 0.3).any(axis=0)
    seam = (
        flat
        & (np.abs(ends[:, 2] - 34.0) < 1.0)
        & (np.abs(ends[:, 0]) > 8.5)
        & (np.abs(ends[:, 0]) < 17.0)
    )
    assert not seam.any()


# ── walls ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("thickness", [0.2, 0.5])
def test_a_slab_measures_its_thickness(thickness):
    def slab(p):
        return jnp.max(jnp.abs(p) - jnp.array([1.0, 1.0, thickness / 2]), axis=-1)

    surface = extract_surface(slab, grid=surface_grid(slab, cell=0.04))
    walls = wall_thickness(slab, surface.mesh, surface.cell)
    assert walls.thinnest == pytest.approx(thickness, rel=0.02)
    assert walls.cells == pytest.approx(thickness / surface.cell, rel=0.02)
    assert walls.resolved
    assert walls.per_vertex.shape == (surface.mesh.vertices.shape[0],)


def test_a_rib_beside_a_convex_edge_is_not_mistaken_for_a_wall(clevis):
    """The clevis's thinnest wall is its 6 mm rib, not a ray off a rim."""
    scene, surface = clevis
    walls = wall_thickness(scene, surface.mesh, surface.cell)
    assert 5.5 < walls.thinnest < 6.2
    assert walls.resolved


def test_the_node_wall_is_two_millimetres(node):
    scene, surface = node
    walls = wall_thickness(scene, surface.mesh, surface.cell)
    assert walls.thinnest == pytest.approx(2.0, rel=0.05)
    assert walls.cells == pytest.approx(2.0, rel=0.05)


def test_a_wall_under_two_cells_is_not_resolved():
    scene = space_frame_node.scene
    surface = extract_surface(scene, grid=surface_grid(scene, cell=1.5))
    walls = wall_thickness(scene, surface.mesh, surface.cell)
    assert walls.cells < MIN_WALL_CELLS
    assert not walls.resolved


def test_a_mesh_with_no_inside_measures_nothing():
    surface = extract_surface(sphere, grid=surface_grid(sphere, cell=0.1))
    flipped = surface.mesh._replace(normals=-np.asarray(surface.mesh.normals))
    walls = wall_thickness(sphere, flipped, surface.cell)
    assert walls.thinnest is None and walls.cells is None and walls.resolved
    assert np.isnan(walls.per_vertex).all()
    with pytest.raises(ValueError, match="cell"):
        wall_thickness(sphere, surface.mesh, 0.0)
