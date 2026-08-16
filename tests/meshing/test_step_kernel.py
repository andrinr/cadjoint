"""Validate STEP export against a real CAD kernel (OCCT via cadquery-ocp).

Round-trips :func:`jaxcad.meshing.export.save_step` output through the OCCT
STEP reader and asserts what a downstream CAD system would care about: the
file parses, transfers to a single ``BRepCheck``-valid closed solid, keeps
exactly the faces :func:`merge_planar_faces` produced (after degenerate-edge
welding), and encloses the mesh's signed volume.

Requires the ``stepcheck`` extra (``uv pip install -e '.[stepcheck]'``);
skipped when ``OCP`` is not importable.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaxcad.meshing.dual_contouring import Mesh, extract_mesh
from jaxcad.meshing.edge_detection import GridSpec
from jaxcad.meshing.export import _weld_degenerate_edges, merge_planar_faces, save_step
from jaxcad.sdf.primitives import Box

pytest.importorskip("OCP")

from OCP.BRepCheck import BRepCheck_Analyzer  # noqa: E402
from OCP.BRepGProp import BRepGProp  # noqa: E402
from OCP.GProp import GProp_GProps  # noqa: E402
from OCP.IFSelect import IFSelect_RetDone  # noqa: E402
from OCP.Interface import Interface_Static  # noqa: E402
from OCP.STEPControl import STEPControl_Reader  # noqa: E402
from OCP.TopAbs import TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID  # noqa: E402
from OCP.TopExp import TopExp_Explorer  # noqa: E402

BOX_SIZE = jnp.array([0.4, 0.5, 0.6])
# Spacing 0.11 from -0.77 puts no lattice plane on a box face at +-0.4/0.5/0.6.
BOX_GRID = GridSpec.from_bounds((-0.77, -0.77, -0.77), (1.54, 1.54, 1.54), 14)
SPHERE_GRID = GridSpec.from_bounds((-1.3, -1.3, -1.3), (2.6, 2.6, 2.6), 12)
# The union sphere reaches x = 0.85, so the union grid must extend past it.
UNION_GRID = GridSpec.from_bounds((-1.03, -1.03, -1.03), (2.06, 2.06, 2.06), 16)


def sphere_sdf(p):
    return jnp.sqrt(jnp.sum(p * p)) - 1.0


def union_sdf(p):
    box = Box.sdf(p, BOX_SIZE)
    ball = jnp.sqrt(jnp.sum((p - jnp.array([0.4, 0.0, 0.0])) ** 2)) - 0.45
    return jnp.minimum(box, ball)


@pytest.fixture(scope="module", autouse=True)
def metre_reader_unit():
    """Read STEP files in metres so kernel volumes match mesh volumes directly.

    ``save_step`` declares ``SI_UNIT($,.METRE.)``; OCCT's default working
    unit is millimetres, which would scale every volume by ``1000 ** 3``.
    Constructing a reader first initializes the Interface statics.
    """
    STEPControl_Reader()
    previous = Interface_Static.CVal_s("xstep.cascade.unit")
    assert Interface_Static.SetCVal_s("xstep.cascade.unit", "M")
    yield
    Interface_Static.SetCVal_s("xstep.cascade.unit", previous or "MM")


def make_mesh(name: str) -> Mesh:
    if name == "box":
        return extract_mesh(lambda p: Box.sdf(p, BOX_SIZE), BOX_GRID)
    if name == "sphere":
        return extract_mesh(sphere_sdf, SPHERE_GRID)
    return extract_mesh(union_sdf, UNION_GRID)


@pytest.fixture(scope="module", params=["box", "sphere", "union"])
def case(request, tmp_path_factory):
    """(name, mesh, transferred OCCT shape) for one exported test mesh."""
    name = request.param
    mesh = make_mesh(name)
    path = tmp_path_factory.mktemp("step") / f"{name}.step"
    save_step(mesh, path)
    reader = STEPControl_Reader()
    assert reader.ReadFile(str(path)) == IFSelect_RetDone, "STEP file must parse"
    assert reader.TransferRoots() == 1, "exactly one root must transfer"
    return name, mesh, reader.OneShape()


def count_subshapes(shape, kind) -> int:
    explorer = TopExp_Explorer(shape, kind)
    total = 0
    while explorer.More():
        total += 1
        explorer.Next()
    return total


def expected_face_count(mesh: Mesh) -> int:
    """Faces ``save_step`` writes: merged loops after degenerate-edge welding."""
    polygons, triangles = merge_planar_faces(mesh)
    loops = [list(polygon) for polygon in polygons]
    loops.extend([int(a), int(b), int(c)] for a, b, c in triangles)
    return len(_weld_degenerate_edges(np.asarray(mesh.vertices, dtype=np.float64), loops))


def mesh_signed_volume(mesh: Mesh) -> float:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64).reshape((-1, 3))
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    return float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0)


class TestStepKernelRoundTrip:
    def test_single_closed_solid(self, case):
        _, _, shape = case
        assert count_subshapes(shape, TopAbs_SOLID) == 1
        assert count_subshapes(shape, TopAbs_SHELL) == 1
        shell_explorer = TopExp_Explorer(shape, TopAbs_SHELL)
        assert shell_explorer.Current().Closed()

    def test_shape_is_brepcheck_valid(self, case):
        _, _, shape = case
        assert BRepCheck_Analyzer(shape).IsValid()

    def test_face_count_matches_merge_planar_faces(self, case):
        name, mesh, shape = case
        expected = expected_face_count(mesh)
        assert count_subshapes(shape, TopAbs_FACE) == expected
        if name == "box":
            assert expected == 6  # A box must collapse to its six faces.

    def test_volume_matches_mesh_signed_volume(self, case):
        _, mesh, shape = case
        properties = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape, properties)
        mesh_volume = mesh_signed_volume(mesh)
        assert mesh_volume > 0
        assert properties.Mass() == pytest.approx(mesh_volume, rel=0.01)
