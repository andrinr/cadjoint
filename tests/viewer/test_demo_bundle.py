"""The static demo's bundle: current, well formed, and honest about what it may move.

``frontend/src/static/bundle`` is written by :mod:`cadjoint.viewer.demo` from
the shipped scenes and committed, so the demo builds without Python.  These
tests keep the committed files the build of the committed scenes, and pin
the parts of the export a reader of one file could not check: which
parameters a constraint holds, and that every range holds its value.
"""

from __future__ import annotations

import json

import pytest

from cadjoint.viewer import demo
from cadjoint.viewer.schema.api import CapabilitiesResponse
from cadjoint.viewer.schema.demo import DemoScene

SCENE_FILES = sorted(demo.BUNDLE.glob("*.demo.json"))


@pytest.fixture(scope="module")
def bracket() -> DemoScene:
    scene = demo.demo_scene(demo.SCENES / "bracket.py")
    assert scene is not None
    return scene


def test_the_committed_bundle_is_the_build_of_the_scenes():
    assert demo.main(["--check"]) == 0, "run `python -m cadjoint.viewer.demo` and commit the bundle"


@pytest.mark.parametrize("path", SCENE_FILES, ids=lambda p: p.stem)
def test_every_scene_file_is_a_demo_scene(path):
    scene = DemoScene.model_validate_json(path.read_text())
    assert scene.name == path.name.removesuffix(".demo.json")
    assert scene.title and scene.description
    assert all(node.faces == [] for node in scene.construction)
    for parameter in scene.parameters:
        assert (
            len(parameter.value)
            == len(parameter.lower)
            == len(parameter.upper)
            == parameter.components
        )
        assert all(
            lo <= v <= hi for lo, v, hi in zip(parameter.lower, parameter.value, parameter.upper)
        )


def test_the_public_tier_has_no_private_kind():
    tier = CapabilitiesResponse.model_validate_json((demo.BUNDLE / demo.TIER_FILE).read_text())
    assert not tier.installed
    assert all(not kind.available and kind.message for kind in tier.kinds.values())
    assert tier.kinds["brep"].label == "Dual contouring · public tier"


def test_constrained_parameters_are_driven_and_the_rest_free(bracket):
    driven = {p.name: p.drivenBy for p in bracket.parameters if p.drivenBy}
    assert set(driven) == {"rib_tip", "rib_web_corner", "bolt_left", "bolt_right"}
    assert driven["bolt_left"].startswith("FixedConstraint(bolt_left)")
    assert "DistanceConstraint(bolt_left, bolt_right)" in driven["bolt_right"]


def test_parameters_name_the_handle_that_moves_them(bracket):
    owners = {p.name: p.owner for p in bracket.parameters}
    nodes = {node.id: node for node in bracket.construction}
    assert nodes[owners["plate_tr"]].name == "plate"
    assert nodes[owners["hole_left_radius"]].name == "hole_left"
    assert owners["plate_thickness"] is None


def test_ranges_around_a_length_and_a_coordinate(bracket):
    by_name = {p.name: p for p in bracket.parameters}
    thickness = by_name["plate_thickness"]
    assert not thickness.bounded
    assert thickness.lower[0] == pytest.approx(0.05) and thickness.upper[0] == pytest.approx(0.4)
    corner = by_name["plate_tr"]
    assert corner.lower[0] < corner.value[0] - 1 and corner.upper[0] > corner.value[0] + 1


def test_declared_bounds_are_the_range(tmp_path):
    source = """\
from cadjoint.construction import Solid
from cadjoint.geometry import Scalar
radius = Scalar(0.5, free=True, name="radius", bounds=(0.1, 0.9))
scene = Solid.sphere(radius=radius, name="ball")
"""
    path = tmp_path / "ball.py"
    path.write_text(f'"""A ball.\n\nOne bounded radius."""\n{source}')
    scene = demo.demo_scene(path)
    assert scene is not None
    (radius,) = [p for p in scene.parameters if p.name == "radius"]
    assert radius.bounded and (radius.lower, radius.upper) == ([0.1], [0.9])
    assert json.loads(scene.model_dump_json())["format"] == "cadjoint-demo"
