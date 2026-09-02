"""Validate the graph's STEP output against the OCCT kernel.

The mesh writer's own kernel test (``tests/meshing/test_step_kernel.py``)
asks the same questions of a faceted file.  The interesting claim here is
stronger: with analytic surfaces the imported solid's volume is not close to
the model's, it *is* the model's, because a ``CYLINDRICAL_SURFACE`` bounded
by two shared ``CIRCLE`` edges carries no discretization error at all.

Requires the ``stepcheck`` extra (``uv pip install -e '.[stepcheck]'``).
"""

from __future__ import annotations

import pytest

from cadjoint.brep import save_brep_step
from tests.brep.conftest import plate_volume

pytest.importorskip("OCP")

from OCP.BRepCheck import BRepCheck_Analyzer  # noqa: E402
from OCP.BRepGProp import BRepGProp  # noqa: E402
from OCP.GProp import GProp_GProps  # noqa: E402
from OCP.IFSelect import IFSelect_RetDone  # noqa: E402
from OCP.Interface import Interface_Static  # noqa: E402
from OCP.STEPControl import STEPControl_Reader  # noqa: E402
from OCP.TopAbs import TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID  # noqa: E402
from OCP.TopExp import TopExp_Explorer  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def metre_reader_unit():
    """Read STEP files in metres so kernel volumes match model volumes."""
    STEPControl_Reader()
    previous = Interface_Static.CVal_s("xstep.cascade.unit")
    Interface_Static.SetCVal_s("xstep.cascade.unit", "M")
    yield
    Interface_Static.SetCVal_s("xstep.cascade.unit", previous or "MM")


def _read(path):
    reader = STEPControl_Reader()
    assert reader.ReadFile(str(path)) == IFSelect_RetDone
    reader.TransferRoots()
    return reader.OneShape()


def _count(shape, kind) -> int:
    explorer = TopExp_Explorer(shape, kind)
    total = 0
    while explorer.More():
        total += 1
        explorer.Next()
    return total


def _volume(shape) -> float:
    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, properties)
    return float(properties.Mass())


def test_the_plate_reads_back_as_one_valid_solid(plate_brep, tmp_path):
    path = tmp_path / "plate.step"
    report = save_brep_step(plate_brep, path)
    shape = _read(path)
    assert BRepCheck_Analyzer(shape).IsValid()
    assert _count(shape, TopAbs_SOLID) == 1
    assert _count(shape, TopAbs_SHELL) == 1
    assert _count(shape, TopAbs_FACE) == report["step_faces"] == 7


def test_the_analytic_solid_has_the_exact_volume(plate_brep, tmp_path):
    """No discretization error survives: the bore is a real cylinder."""
    path = tmp_path / "plate.step"
    save_brep_step(plate_brep, path)
    assert _volume(_read(path)) == pytest.approx(plate_volume(), rel=1e-6)


def test_the_faceted_alternative_is_only_approximate(plate_brep, tmp_path):
    """The same graph written without analytic surfaces loses the bore."""
    path = tmp_path / "faceted.step"
    save_brep_step(plate_brep, path, analytic=False)
    faceted = _volume(_read(path))
    assert faceted == pytest.approx(plate_volume(), rel=5e-3)
    assert abs(faceted - plate_volume()) > 1e-5, "a polygonal bore over-fills the hole"


def test_the_thermal_body_still_sews_into_one_shell(thermal_brep, tmp_path):
    """Analytic faces and blend facets must agree on their shared curves."""
    path = tmp_path / "sink.step"
    report = save_brep_step(thermal_brep, path)
    shape = _read(path)
    assert BRepCheck_Analyzer(shape).IsValid()
    assert _count(shape, TopAbs_SHELL) == 1
    assert _count(shape, TopAbs_FACE) == report["step_faces"]
