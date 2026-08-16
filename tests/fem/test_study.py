"""Tests for jaxcad.fem.study (declarative studies, code parity)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from jaxcad.fem import (
    Dirichlet,
    ElasticStudy,
    Fixed,
    GridSpec,
    HeatFlux,
    Nodes,
    ThermalStudy,
    Traction,
    capture_studies,
    sdf_to_hex_mesh,
)
from jaxcad.geometry.parameters import Vector
from jaxcad.sdf.primitives import Box

_BOUNDS = (-1.1, -0.25, -0.25)
_SIZE = (2.2, 0.5, 0.5)
_RESOLUTION = (22, 5, 5)


def _bar():
    return Box(Vector([1.0, 0.15, 0.15], free=True, name="size"))


def _bar_mesh():
    return sdf_to_hex_mesh(_bar(), GridSpec.from_bounds(_BOUNDS, _SIZE, _RESOLUTION))


def _thermal_study(**overrides):
    settings = {
        "name": "bar-conduction",
        "resolution": _RESOLUTION,
        "conductivity": 2.0,
        "bcs": [
            Dirichlet(Nodes.side("-x"), 1.0),
            Dirichlet(Nodes.side("+x"), 0.0),
        ],
        "bounds": _BOUNDS,
        "size": _SIZE,
    }
    settings.update(overrides)
    return ThermalStudy(**settings)


def _elastic_study(**overrides):
    settings = {
        "name": "cantilever",
        "resolution": _RESOLUTION,
        "youngs": 1000.0,
        "poisson": 0.3,
        "bcs": [
            Fixed(Nodes.side("-x")),
            Traction(Nodes.side("+x"), (0.0, 0.0, -1.0)),
        ],
        "bounds": _BOUNDS,
        "size": _SIZE,
    }
    settings.update(overrides)
    return ElasticStudy(**settings)


class TestValidation:
    def test_bc_requires_a_node_selection(self):
        with pytest.raises(ValueError, match="node selection"):
            Dirichlet(lambda center: center[0] > 0.99, 1.0)
        with pytest.raises(ValueError, match="node selection"):
            Fixed("-x")
        with pytest.raises(ValueError, match="node selection"):
            Traction(None, (0.0, 0.0, -1.0))
        with pytest.raises(ValueError, match="node selection"):
            HeatFlux([1, 2, 3], 0.5)

    def test_traction_vector_shape(self):
        with pytest.raises(ValueError, match="vector"):
            Traction(Nodes.side("+x"), (0.0, 1.0))

    def test_empty_name(self):
        with pytest.raises(ValueError, match="non-empty name"):
            _thermal_study(name="  ")

    def test_bad_resolution(self):
        with pytest.raises(ValueError, match="resolution"):
            _thermal_study(resolution=(22, 0, 5))

    def test_bad_conductivity(self):
        with pytest.raises(ValueError, match="conductivity"):
            _thermal_study(conductivity=0.0)

    def test_bad_poisson(self):
        with pytest.raises(ValueError, match="poisson"):
            _elastic_study(poisson=0.5)

    def test_wrong_bc_type_for_study(self):
        with pytest.raises(ValueError, match="accepts boundary conditions"):
            _thermal_study(bcs=[Fixed(Nodes.side("-x"))])
        with pytest.raises(ValueError, match="accepts boundary conditions"):
            _elastic_study(bcs=[Dirichlet(Nodes.side("-x"), 1.0)])

    def test_thermal_solve_requires_a_dirichlet(self):
        study = _thermal_study(bcs=[])
        with pytest.raises(ValueError, match="at least one Dirichlet"):
            study.solve(_bar())

    def test_elastic_solve_requires_a_fixed(self):
        study = _elastic_study(bcs=[Traction(Nodes.side("+x"), (0.0, 0.0, -1.0))])
        with pytest.raises(ValueError, match="at least one Fixed"):
            study.solve(_bar())

    def test_unmatched_selection_fails_before_solving(self):
        study = _thermal_study(
            bcs=[
                Dirichlet(Nodes.side("-x"), 1.0),
                Dirichlet(Nodes.sphere([50.0, 0.0, 0.0], 0.1), 0.0),
            ]
        )
        with pytest.raises(ValueError, match="matched no boundary nodes"):
            study.solve(_bar(), mesh=_bar_mesh())

    def test_faceless_flux_selection_fails_before_solving(self):
        # A single corner node spans no complete boundary quad, so an
        # area-integrated BC cannot act on it.
        study = _thermal_study(
            bcs=[
                Dirichlet(Nodes.side("-x"), 1.0),
                HeatFlux(Nodes.sphere([1.0, 0.15, 0.15], 0.01), 0.5),
            ]
        )
        with pytest.raises(ValueError, match="spans no complete boundary face"):
            study.solve(_bar(), mesh=_bar_mesh())


class TestDescribe:
    def test_thermal_describe_is_json_ready(self):
        study = _thermal_study()
        payload = study.describe()
        round_trip = json.loads(json.dumps(payload))
        assert round_trip == payload
        assert payload["kind"] == "thermal"
        assert payload["name"] == "bar-conduction"
        assert payload["resolution"] == [22, 5, 5]
        assert payload["bounds"] == list(_BOUNDS)
        assert payload["material"] == {"conductivity": 2.0}
        assert payload["bcs"][0] == {
            "type": "dirichlet",
            "nodes": {"kind": "side", "side": "-x", "tol": None},
            "value": 1.0,
        }

    def test_elastic_describe_is_json_ready(self):
        payload = _elastic_study().describe()
        assert json.loads(json.dumps(payload)) == payload
        assert payload["kind"] == "elastic"
        assert payload["material"] == {"youngs": 1000.0, "poisson": 0.3}
        assert payload["bcs"][0] == {
            "type": "fixed",
            "nodes": {"kind": "side", "side": "-x", "tol": None},
        }
        assert payload["bcs"][1]["vector"] == [0.0, 0.0, -1.0]

    def test_heat_flux_describe(self):
        bc = HeatFlux(Nodes.side("+x"), 0.5)
        assert bc.describe() == {
            "type": "heat_flux",
            "nodes": {"kind": "side", "side": "+x", "tol": None},
            "flux": 0.5,
        }

    def test_composed_selection_describes_through_the_bc(self):
        clamp = Fixed(Nodes.sphere([0.0, 0.0, 0.0], 0.5) | Nodes.box([0, 0, 0], [1, 1, 1]))
        payload = clamp.describe()
        assert payload["nodes"]["kind"] == "or"
        assert json.loads(json.dumps(payload)) == payload


class TestCapture:
    def test_capture_collects_exec_declared_studies(self):
        source = "\n".join(
            [
                "from jaxcad.fem import Dirichlet, ElasticStudy, Fixed, Nodes, ThermalStudy",
                "ThermalStudy(name='captured-thermal', resolution=8, conductivity=1.0,",
                "             bcs=[Dirichlet(Nodes.side('-x'), 1.0)])",
                "ElasticStudy(name='captured-elastic', resolution=8, youngs=10.0, poisson=0.2,",
                "             bcs=[Fixed(Nodes.side('-x'))])",
            ]
        )
        with capture_studies() as studies:
            exec(source, {})  # noqa: S102 - deliberate: user scene programs are exec'd
        assert [study.name for study in studies] == ["captured-thermal", "captured-elastic"]
        assert isinstance(studies[0], ThermalStudy)
        assert isinstance(studies[1], ElasticStudy)

    def test_no_capture_outside_context(self):
        with capture_studies() as studies:
            pass
        _thermal_study()
        assert studies == []

    def test_nested_contexts_isolate(self):
        with capture_studies() as outer:
            _thermal_study(name="outer-study")
            with capture_studies() as inner:
                _thermal_study(name="inner-study")
            _thermal_study(name="outer-again")
        assert [study.name for study in inner] == ["inner-study"]
        assert [study.name for study in outer] == ["outer-study", "outer-again"]


class TestSolve:
    def test_thermal_study_reproduces_direct_solve(self):
        pytest.importorskip("jax_fem")
        result = _thermal_study().solve(_bar())
        temperature = np.asarray(result.temperature)
        x = result.mesh.points[:, 0]
        assert np.abs(temperature - (1.0 - x) / 2.0).max() < 1e-6

    def test_elastic_study_reproduces_cantilever(self):
        pytest.importorskip("jax_fem")
        result = _elastic_study().solve(_bar())
        displacement = np.asarray(result.displacement)
        tip = np.isclose(result.mesh.points[:, 0], 1.0)
        tip_uz = displacement[tip, 2].mean()
        force = 1.0 * 0.3 * 0.3
        inertia = 0.3 * 0.3**3 / 12.0
        euler = -force * 2.0**3 / (3.0 * 1000.0 * inertia)
        assert tip_uz < 0.0
        assert 0.7 < tip_uz / euler < 1.3

    def test_heat_flux_study_solves_on_the_direct_backend(self):
        pytest.importorskip("jax_fem")
        # -k T'' = 0 with T(-1) = 0 and k T'(1) = q: T(x) = (q/k)(x + 1).
        study = _thermal_study(
            bcs=[
                Dirichlet(Nodes.side("-x"), 0.0),
                HeatFlux(Nodes.side("+x"), 1.0),
            ]
        )
        result = study.solve(_bar())
        temperature = np.asarray(result.temperature)
        expected = (result.mesh.points[:, 0] + 1.0) * (1.0 / 2.0)
        assert np.abs(temperature - expected).max() < 1e-6

    def test_solve_accepts_pre_extracted_mesh(self):
        pytest.importorskip("jax_fem")
        mesh = _bar_mesh()
        result = _thermal_study().solve(_bar(), mesh=mesh)
        assert result.mesh is mesh
