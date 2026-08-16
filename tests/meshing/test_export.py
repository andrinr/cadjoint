"""Tests for jaxcad.meshing.export (planar merging and OBJ/STL/STEP writers)."""

from __future__ import annotations

import re

import jax.numpy as jnp
import numpy as np
import pytest

from jaxcad.meshing.dual_contouring import Mesh, extract_mesh
from jaxcad.meshing.edge_detection import GridSpec
from jaxcad.meshing.export import merge_planar_faces, save_obj, save_step, save_stl
from jaxcad.sdf.primitives import Box

BOX_SIZE = jnp.array([0.4, 0.5, 0.6])
# Spacing 0.11 from -0.77 puts no lattice plane on a box face at +-0.4/0.5/0.6.
BOX_GRID = GridSpec.from_bounds((-0.77, -0.77, -0.77), (1.54, 1.54, 1.54), 14)
SPHERE_GRID = GridSpec.from_bounds((-1.3, -1.3, -1.3), (2.6, 2.6, 2.6), 26)
COARSE_SPHERE_GRID = GridSpec.from_bounds((-1.3, -1.3, -1.3), (2.6, 2.6, 2.6), 8)


def sphere_sdf(p):
    return jnp.sqrt(jnp.sum(p * p)) - 1.0


@pytest.fixture(scope="module")
def box_mesh() -> Mesh:
    return extract_mesh(lambda p: Box.sdf(p, BOX_SIZE), BOX_GRID)


@pytest.fixture(scope="module")
def sphere_mesh() -> Mesh:
    return extract_mesh(sphere_sdf, SPHERE_GRID)


@pytest.fixture(scope="module")
def coarse_sphere_mesh() -> Mesh:
    return extract_mesh(sphere_sdf, COARSE_SPHERE_GRID)


def loop_normal(vertices: np.ndarray, loop: list[int]) -> np.ndarray:
    points = vertices[loop]
    rolled = np.roll(points, -1, axis=0)
    return 0.5 * np.sum(np.cross(points, rolled), axis=0)


class TestMergePlanarFaces:
    def test_box_collapses_to_six_polygons(self, box_mesh):
        polygons, leftover = merge_planar_faces(box_mesh)
        assert len(polygons) == 6
        assert leftover.shape == (0, 3)

    def test_box_polygons_are_valid_simple_loops(self, box_mesh):
        polygons, _ = merge_planar_faces(box_mesh)
        vertex_count = box_mesh.vertices.shape[0]
        for loop in polygons:
            assert len(loop) >= 4
            assert len(set(loop)) == len(loop)
            assert all(0 <= index < vertex_count for index in loop)

    def test_box_polygons_planar_and_ccw_outward(self, box_mesh):
        polygons, _ = merge_planar_faces(box_mesh)
        vertices = np.asarray(box_mesh.vertices, dtype=np.float64)
        seen_normals = []
        for loop in polygons:
            normal = loop_normal(vertices, loop)
            normal /= np.linalg.norm(normal)
            points = vertices[loop]
            offsets = (points - points.mean(axis=0)) @ normal
            np.testing.assert_allclose(offsets, 0.0, atol=1e-5)
            # CCW around the outward normal: for a box centered at the
            # origin the polygon normal must point away from the origin.
            assert float(normal @ points.mean(axis=0)) > 0
            seen_normals.append(normal)
        # One face per box side: the six normals are the signed axes.
        stacked = np.sort(np.round(np.asarray(seen_normals), 5), axis=0)
        expected = np.sort(np.concatenate([np.eye(3), -np.eye(3)]), axis=0)
        np.testing.assert_allclose(stacked, expected, atol=1e-5)

    def test_box_polygon_area_matches_faces(self, box_mesh):
        polygons, _ = merge_planar_faces(box_mesh)
        vertices = np.asarray(box_mesh.vertices, dtype=np.float64)
        areas = sorted(float(np.linalg.norm(loop_normal(vertices, loop))) for loop in polygons)
        # Half-extents (0.4, 0.5, 0.6): face areas 4*(0.5*0.6, 0.4*0.6, 0.4*0.5).
        expected = sorted([1.2, 1.2, 0.96, 0.96, 0.8, 0.8])
        np.testing.assert_allclose(areas, expected, rtol=1e-4)

    def test_sphere_yields_zero_polygons(self, sphere_mesh):
        polygons, leftover = merge_planar_faces(sphere_mesh)
        assert polygons == []
        np.testing.assert_array_equal(leftover, sphere_mesh.faces)


def parse_obj(path):
    obj_vertices = []
    obj_faces = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            obj_vertices.append([float(value) for value in parts[1:]])
        elif parts[0] == "f":
            obj_faces.append([int(value) for value in parts[1:]])
    return np.asarray(obj_vertices, dtype=np.float64), obj_faces


class TestSaveObj:
    def test_box_roundtrip(self, box_mesh, tmp_path):
        path = tmp_path / "box.obj"
        save_obj(box_mesh, path)
        obj_vertices, obj_faces = parse_obj(path)
        assert obj_vertices.shape[0] == box_mesh.vertices.shape[0]
        assert len(obj_faces) == 6  # merged n-gons only, no leftover triangles
        for face in obj_faces:
            assert len(face) >= 4
            assert len(set(face)) == len(face)  # closed simple loop
            assert all(1 <= index <= obj_vertices.shape[0] for index in face)
        np.testing.assert_allclose(
            obj_vertices, np.asarray(box_mesh.vertices, dtype=np.float64), atol=1e-5
        )

    def test_sphere_unmerged_triangles(self, sphere_mesh, tmp_path):
        path = tmp_path / "sphere.obj"
        save_obj(sphere_mesh, path)
        obj_vertices, obj_faces = parse_obj(path)
        assert obj_vertices.shape[0] == sphere_mesh.vertices.shape[0]
        assert len(obj_faces) == sphere_mesh.faces.shape[0]
        assert all(len(face) == 3 for face in obj_faces)
        np.testing.assert_array_equal(np.asarray(obj_faces, dtype=np.int32) - 1, sphere_mesh.faces)

    def test_merge_planar_false_writes_all_triangles(self, box_mesh, tmp_path):
        path = tmp_path / "box_tris.obj"
        save_obj(box_mesh, path, merge_planar=False)
        _, obj_faces = parse_obj(path)
        assert len(obj_faces) == box_mesh.faces.shape[0]

    def test_deterministic(self, box_mesh, tmp_path):
        first, second = tmp_path / "a.obj", tmp_path / "b.obj"
        save_obj(box_mesh, first)
        save_obj(box_mesh, second)
        assert first.read_bytes() == second.read_bytes()


class TestSaveStl:
    def test_binary_roundtrip(self, sphere_mesh, tmp_path):
        path = tmp_path / "sphere.stl"
        save_stl(sphere_mesh, path)
        raw = path.read_bytes()
        triangle_count = int(np.frombuffer(raw[80:84], dtype="<u4")[0])
        assert triangle_count == sphere_mesh.faces.shape[0]
        assert len(raw) == 84 + 50 * triangle_count
        records = np.frombuffer(
            raw[84:],
            dtype=[("normal", "<f4", (3,)), ("corners", "<f4", (3, 3)), ("attribute", "<u2")],
        )
        vertices = np.asarray(sphere_mesh.vertices, dtype=np.float64)
        np.testing.assert_allclose(records["corners"], vertices[sphere_mesh.faces], atol=1e-5)
        # Normals follow the CCW winding: unit length, outward for a sphere.
        lengths = np.linalg.norm(records["normal"], axis=1)
        np.testing.assert_allclose(lengths, 1.0, atol=1e-4)
        centers = vertices[sphere_mesh.faces].mean(axis=1)
        assert float(np.min(np.einsum("ij,ij->i", records["normal"], centers))) > 0
        assert np.all(records["attribute"] == 0)

    def test_ascii_fallback(self, coarse_sphere_mesh, tmp_path):
        path = tmp_path / "sphere_ascii.stl"
        save_stl(coarse_sphere_mesh, path, binary=False)
        text = path.read_text()
        assert text.startswith("solid ")
        assert text.rstrip().endswith("endsolid jaxcad")
        assert text.count("facet normal") == coarse_sphere_mesh.faces.shape[0]
        assert text.count("vertex") == 3 * coarse_sphere_mesh.faces.shape[0]

    def test_deterministic(self, coarse_sphere_mesh, tmp_path):
        first, second = tmp_path / "a.stl", tmp_path / "b.stl"
        save_stl(coarse_sphere_mesh, first)
        save_stl(coarse_sphere_mesh, second)
        assert first.read_bytes() == second.read_bytes()


def step_ids(text):
    defined = re.findall(r"#(\d+)\s*=", text)
    referenced = re.findall(r"#(\d+)", text)
    return defined, referenced


def parse_step_entities(text: str) -> dict[int, str]:
    """Parse a STEP data section into ``{entity id: body}``, one per line.

    Asserts that every entity id is defined exactly once.  Test helper only;
    it understands exactly the single-line ``#n = BODY;`` records that
    :func:`~jaxcad.meshing.export.save_step` writes.
    """
    data = text.split("DATA;", 1)[1].split("ENDSEC;", 1)[0]
    entities: dict[int, str] = {}
    for match in re.finditer(r"^#(\d+)\s*=\s*(.+);\s*$", data, re.MULTILINE):
        entity_id = int(match.group(1))
        assert entity_id not in entities, f"entity #{entity_id} defined twice"
        entities[entity_id] = match.group(2).strip()
    return entities


def step_references(body: str) -> list[int]:
    return [int(number) for number in re.findall(r"#(\d+)", body)]


def check_step_graph(mesh: Mesh, text: str) -> None:
    """Validate the structural entity graph of a save_step output."""
    entities = parse_step_entities(text)

    # Every reference resolves to a defined entity.
    for entity_id, body in entities.items():
        for reference in step_references(body):
            assert reference in entities, f"#{entity_id} references undefined #{reference}"

    def of_type(name: str) -> dict[int, str]:
        return {i: body for i, body in entities.items() if body.startswith(name + "(")}

    # Exactly one shell, referencing only ADVANCED_FACE entities — all of them.
    shells = of_type("CLOSED_SHELL")
    assert len(shells) == 1
    (shell_id,) = shells
    shell_faces = step_references(shells[shell_id])
    faces = of_type("ADVANCED_FACE")
    assert sorted(shell_faces) == sorted(faces)
    assert len(shell_faces) == len(set(shell_faces))

    # Exactly one solid, referencing the shell.
    breps = of_type("MANIFOLD_SOLID_BREP")
    assert len(breps) == 1
    assert step_references(next(iter(breps.values()))) == [shell_id]

    # Every EDGE_CURVE runs between two VERTEX_POINT entities.
    vertex_points = of_type("VERTEX_POINT")
    edge_curves = of_type("EDGE_CURVE")
    for body in edge_curves.values():
        start, end = step_references(body)[:2]
        assert start in vertex_points
        assert end in vertex_points

    # Entity counts match the merged polygon mesh the writer serializes.
    polygons, triangles = merge_planar_faces(mesh)
    loops = [list(loop) for loop in polygons]
    loops.extend([int(a), int(b), int(c)] for a, b, c in triangles)
    used_vertices = {index for loop in loops for index in loop}
    undirected_edges = {
        (a, b) if a < b else (b, a) for loop in loops for a, b in zip(loop, loop[1:] + loop[:1])
    }
    assert len(faces) == len(loops)
    assert len(vertex_points) == len(used_vertices)
    assert len(edge_curves) == len(undirected_edges)
    assert len(of_type("ORIENTED_EDGE")) == sum(len(loop) for loop in loops)


class TestSaveStep:
    def test_box_structure(self, box_mesh, tmp_path):
        path = tmp_path / "box.step"
        save_step(box_mesh, path)
        text = path.read_text()
        assert text.startswith("ISO-10303-21;")
        assert text.rstrip().endswith("END-ISO-10303-21;")
        defined, referenced = step_ids(text)
        assert len(defined) == len(set(defined))  # every '#n =' id unique
        assert set(referenced) <= set(defined)  # every reference defined
        polygons, leftover = merge_planar_faces(box_mesh)
        face_count = len(polygons) + leftover.shape[0]
        assert face_count == 6
        assert text.count("ADVANCED_FACE(") == face_count
        assert text.count("FACE_OUTER_BOUND(") == face_count
        assert text.count("EDGE_LOOP(") == face_count
        assert text.count("PLANE(") == face_count
        assert text.count("CLOSED_SHELL(") == 1
        assert text.count("MANIFOLD_SOLID_BREP(") == 1
        used = {index for loop in polygons for index in loop}
        assert text.count("VERTEX_POINT(") == len(used)
        edges = {
            (a, b) if a < b else (b, a)
            for loop in polygons
            for a, b in zip(loop, loop[1:] + loop[:1])
        }
        assert text.count("EDGE_CURVE(") == len(edges)
        assert text.count("ORIENTED_EDGE(") == sum(len(loop) for loop in polygons)

    def test_triangle_mesh_counts(self, coarse_sphere_mesh, tmp_path):
        path = tmp_path / "sphere.step"
        save_step(coarse_sphere_mesh, path)
        text = path.read_text()
        defined, referenced = step_ids(text)
        assert len(defined) == len(set(defined))
        assert set(referenced) <= set(defined)
        triangle_count = coarse_sphere_mesh.faces.shape[0]
        assert text.count("ADVANCED_FACE(") == triangle_count
        assert text.count("ORIENTED_EDGE(") == 3 * triangle_count
        assert text.count("CLOSED_SHELL(") == 1

    def test_deterministic(self, box_mesh, tmp_path):
        first, second = tmp_path / "a.step", tmp_path / "b.step"
        save_step(box_mesh, first)
        save_step(box_mesh, second)
        assert first.read_bytes() == second.read_bytes()

    def test_box_structural_graph(self, box_mesh, tmp_path):
        path = tmp_path / "box_graph.step"
        save_step(box_mesh, path)
        check_step_graph(box_mesh, path.read_text())

    def test_sphere_structural_graph(self, coarse_sphere_mesh, tmp_path):
        path = tmp_path / "sphere_graph.step"
        save_step(coarse_sphere_mesh, path)
        check_step_graph(coarse_sphere_mesh, path.read_text())
