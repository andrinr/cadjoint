"""Shared scenes for the B-rep suite.

Two fixtures cover the two things a derived B-rep has to get right: a hard
CSG solid whose whole graph is exact (a plate with a bore — six planes, one
cylinder, twelve straight edges, two rim circles, eight corners), and the
playground's own starter, whose smooth unions produce blend faces that lie
on no patch at all.  Both are module-scoped because the extraction is the
expensive part.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cadjoint.geometry import Scalar, Vector
from cadjoint.meshing.edge_detection import GridSpec
from cadjoint.sdf.boolean import Difference
from cadjoint.sdf.primitives import Box, Cylinder
from cadjoint.sdf.transforms import Translate

_REPO = Path(__file__).resolve().parents[2]
_SCENE_PATH = _REPO / "scenes" / "starter.py"

#: Half-extents of the test plate and the radius of the bore through it.
PLATE_SIZE = (0.6, 0.6, 0.4)
BORE_RADIUS = 0.25

#: Spacing 0.083 from -0.83 puts no lattice plane on a plate face, so the
#: extraction never has to place a vertex on a bit-exact zero.
PLATE_GRID = GridSpec.from_bounds((-0.83, -0.83, -0.63), (1.66, 1.66, 1.26), 20)


def plate_scene():
    """A plate with a through bore: a hard Difference with sharp edges."""
    box = Box(size=Vector(list(PLATE_SIZE)))
    bore = Translate(
        Cylinder(radius=Scalar(BORE_RADIUS), height=Scalar(0.9)),
        Vector([0.0, 0.0, 0.0]),
    )
    return Difference((box, bore), smoothness=0.0)


def plate_volume() -> float:
    """The plate's exact volume, box minus cylinder."""
    import numpy as np

    return float(
        8.0 * PLATE_SIZE[0] * PLATE_SIZE[1] * PLATE_SIZE[2]
        - np.pi * BORE_RADIUS**2 * 2.0 * PLATE_SIZE[2]
    )


@pytest.fixture(scope="session")
def plate():
    """The plate scene (rebuilt per session; parameters are mutable state)."""
    return plate_scene()


@pytest.fixture(scope="session")
def plate_brep(plate):
    """The plate's ownership graph on :data:`PLATE_GRID`."""
    from cadjoint.brep import extract_brep

    return extract_brep(plate, PLATE_GRID)


@pytest.fixture(scope="session")
def starter_namespace():
    """Execute ``scenes/starter.py`` the way the compile worker does."""
    from cadjoint.fem import capture_sim_meshes, capture_studies

    namespace: dict = {}
    with capture_sim_meshes(), capture_studies():
        exec(compile(_SCENE_PATH.read_text(), str(_SCENE_PATH), "exec"), namespace, namespace)
    return namespace


@pytest.fixture(scope="session")
def starter_grid(starter_namespace):
    """The grid the starter's own ``SimMesh`` declares."""
    declared = starter_namespace["sink_mesh"]
    return GridSpec.from_bounds(
        tuple(declared.bounds), tuple(declared.size), tuple(declared.resolution)
    )


@pytest.fixture(scope="session")
def thermal_brep(starter_namespace, starter_grid):
    """The starter thermal body's graph: planar caps, bushings, revolve, blends."""
    from cadjoint.brep import extract_brep

    return extract_brep(starter_namespace["thermal_body"], starter_grid)
