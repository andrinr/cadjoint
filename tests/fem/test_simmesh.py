"""Tests for cadjoint.fem.simmesh (named mesh objects, quality inspection)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from cadjoint.fem import SimMesh, aspect_ratios, capture_sim_meshes, scaled_jacobians
from cadjoint.geometry.parameters import Vector
from cadjoint.sdf.primitives import Box

_BOUNDS = (-1.1, -0.25, -0.25)
_SIZE = (2.2, 0.5, 0.5)
_RESOLUTION = (22, 5, 5)


def _bar():
    return Box(Vector([1.0, 0.15, 0.15], free=True, name="size"))


def _bar_mesh(**overrides):
    settings = {
        "name": "bar-mesh",
        "resolution": _RESOLUTION,
        "bounds": _BOUNDS,
        "size": _SIZE,
    }
    settings.update(overrides)
    return SimMesh(**settings)


class TestValidation:
    def test_empty_name(self):
        with pytest.raises(ValueError, match="non-empty name"):
            _bar_mesh(name="  ")

    def test_bad_resolution(self):
        with pytest.raises(ValueError, match="resolution"):
            _bar_mesh(resolution=(22, 0, 5))

    def test_bounds_and_size_come_together(self):
        with pytest.raises(ValueError, match="together"):
            SimMesh(name="m", resolution=8, bounds=_BOUNDS)
        with pytest.raises(ValueError, match="together"):
            SimMesh(name="m", resolution=8, size=_SIZE)

    def test_negative_size(self):
        with pytest.raises(ValueError, match="size"):
            _bar_mesh(size=(1.0, -0.5, 1.0))

    def test_bad_padding(self):
        with pytest.raises(ValueError, match="padding"):
            _bar_mesh(padding=-0.1)

    def test_domain_must_be_callable(self):
        with pytest.raises(TypeError, match="domain"):
            _bar_mesh(domain="the bracket")

    def test_build_needs_a_field(self):
        with pytest.raises(ValueError, match="declares no domain"):
            _bar_mesh().build()


class TestCapture:
    def test_capture_collects_in_declaration_order(self):
        with capture_sim_meshes() as meshes:
            _bar_mesh(name="first")
            _bar_mesh(name="second")
        assert [mesh.name for mesh in meshes] == ["first", "second"]

    def test_no_capture_outside_context(self):
        with capture_sim_meshes() as meshes:
            pass
        _bar_mesh()
        assert meshes == []

    def test_nested_contexts_isolate(self):
        with capture_sim_meshes() as outer:
            _bar_mesh(name="outer-mesh")
            with capture_sim_meshes() as inner:
                _bar_mesh(name="inner-mesh")
        assert [mesh.name for mesh in inner] == ["inner-mesh"]
        assert [mesh.name for mesh in outer] == ["outer-mesh"]


class TestDescribe:
    def test_describe_is_json_ready(self):
        payload = _bar_mesh().describe()
        assert json.loads(json.dumps(payload)) == payload
        assert payload == {
            "kind": "mesh",
            "name": "bar-mesh",
            "resolution": [22, 5, 5],
            "bounds": list(_BOUNDS),
            "size": list(_SIZE),
            "padding": 0.1,
            "domain": None,
        }

    def test_domain_recorded_by_name_and_type(self):
        domain = _bar()
        domain.name = "bar"
        payload = _bar_mesh(domain=domain).describe()
        assert payload["domain"] == {"name": "bar", "type": "Box"}

    def test_unnamed_domain_records_type_only(self):
        payload = _bar_mesh(domain=lambda p: p[..., 0]).describe()
        assert payload["domain"]["name"] is None

    def test_auto_bounds_describe_as_null(self):
        payload = SimMesh(name="auto", resolution=8, domain=_bar()).describe()
        assert payload["bounds"] is None
        assert payload["size"] is None


class TestBuild:
    def test_build_meshes_the_scene_sdf(self):
        mesh = _bar_mesh().build(_bar())
        assert mesh.num_cells > 0
        assert mesh.snap_mask.any()

    def test_build_caches_per_field_object(self):
        sim_mesh = _bar_mesh()
        bar = _bar()
        first = sim_mesh.build(bar)
        assert sim_mesh.build(bar) is first
        assert sim_mesh.build(_bar()) is not first  # new field object -> rebuild

    def test_rebuild_flag_forces_extraction(self):
        sim_mesh = _bar_mesh()
        bar = _bar()
        first = sim_mesh.build(bar)
        assert sim_mesh.build(bar, rebuild=True) is not first

    def test_parameter_change_invalidates_the_cache(self):
        sim_mesh = _bar_mesh()
        bar = _bar()
        first = sim_mesh.build(bar)
        sim_mesh.resolution = (11, 5, 5)
        second = sim_mesh.build(bar)
        assert second is not first
        assert second.num_cells < first.num_cells

    def test_domain_wins_over_the_passed_sdf(self):
        small = Box(Vector([0.3, 0.15, 0.15], free=True, name="small"))
        sim_mesh = _bar_mesh(domain=small)
        mesh = sim_mesh.build(_bar())  # scene sdf is the long bar; domain is short
        assert mesh.points[:, 0].max() < 0.5
        assert mesh.points[:, 0].min() > -0.5

    def test_auto_bounds_cover_the_domain_with_padding(self):
        sim_mesh = SimMesh(name="auto", resolution=12, domain=_bar(), padding=0.2)
        grid = sim_mesh.grid()
        origin = np.asarray(grid.origin)
        extent = np.asarray(grid.spacing) * np.asarray(grid.cells)
        assert np.all(origin < np.array([-1.0, -0.15, -0.15]))
        assert np.all(origin + extent > np.array([1.0, 0.15, 0.15]))
        mesh = sim_mesh.build()
        assert mesh.num_cells > 0

    def test_auto_bounds_reject_an_empty_field(self):
        nowhere = SimMesh(name="empty", resolution=8, domain=lambda p: p[..., 0] * 0.0 + 1.0)
        with pytest.raises(ValueError, match="bounds"):
            nowhere.grid()


class TestQuality:
    def test_perfect_lattice_scores_one(self):
        # Cube aligned with the grid: snapped vertices land on lattice
        # positions, so every hex stays a perfect cube.
        cube = Box(Vector([0.5, 0.5, 0.5], free=True, name="size"))
        sim_mesh = SimMesh(
            name="cube", resolution=4, bounds=(-0.5, -0.5, -0.5), size=(1.0, 1.0, 1.0)
        )
        metrics = sim_mesh.quality(cube)
        assert metrics["scaled_jacobian"].shape == (sim_mesh.build(cube).num_cells,)
        assert np.allclose(metrics["scaled_jacobian"], 1.0, atol=1e-9)
        assert np.allclose(metrics["aspect_ratio"], 1.0, atol=1e-9)

    def test_anisotropic_cells_report_their_aspect_ratio(self):
        cube = Box(Vector([0.5, 0.5, 0.5], free=True, name="size"))
        sim_mesh = SimMesh(
            name="flat", resolution=(2, 2, 4), bounds=(-0.5, -0.5, -0.5), size=(1.0, 1.0, 1.0)
        )
        metrics = sim_mesh.quality(cube)
        assert np.allclose(metrics["aspect_ratio"], 2.0, atol=1e-9)  # 0.5 / 0.25
        assert np.allclose(metrics["scaled_jacobian"], 1.0, atol=1e-9)

    def test_snapped_boundary_degrades_quality_but_stays_valid(self):
        # Resolution 20 puts lattice planes off the bar's faces (spacing
        # 0.11), so boundary snapping genuinely moves vertices.
        metrics = _bar_mesh(resolution=(20, 5, 5)).quality(_bar())
        jacobian = metrics["scaled_jacobian"]
        assert jacobian.min() > 0.0  # inversion guard held
        assert jacobian.min() < 1.0 - 1e-6  # snapping actually distorted cells
        assert metrics["aspect_ratio"].min() >= 1.0

    def test_metric_functions_flag_distorted_and_inverted_hexes(self):
        unit = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=np.float64,
        )
        cells = np.arange(8, dtype=np.int32).reshape(1, 8)
        assert scaled_jacobians(unit, cells) == pytest.approx([1.0])
        assert aspect_ratios(unit, cells) == pytest.approx([1.0])
        sheared = unit.copy()
        sheared[unit[:, 2] == 1.0, 0] += 0.5  # shear the top face along +x
        assert scaled_jacobians(sheared, cells)[0] < 0.95
        assert aspect_ratios(sheared, cells)[0] > 1.0
        inverted = unit.copy()
        inverted[0] = [2.0, 2.0, 2.0]  # push a corner through the cell
        assert scaled_jacobians(inverted, cells)[0] < 0.0


class TestInspect:
    def test_inspect_is_json_ready_and_complete(self):
        sim_mesh = _bar_mesh()
        report = sim_mesh.inspect(_bar())
        assert json.loads(json.dumps(report)) == report
        mesh = sim_mesh.build(_bar())
        assert report["name"] == "bar-mesh"
        assert report["nodes"] == mesh.num_points
        assert report["elements"] == mesh.num_cells
        assert report["grid"]["cells"] == list(_RESOLUTION)
        low, high = report["bounds"]["min"], report["bounds"]["max"]
        assert low[0] == pytest.approx(-1.0, abs=1e-3)
        assert high[0] == pytest.approx(1.0, abs=1e-3)
        for metric in ("scaled_jacobian", "aspect_ratio"):
            summary = report["quality"][metric]
            assert summary["min"] <= summary["mean"] <= summary["max"]
