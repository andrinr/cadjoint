"""``describe()`` shapes (:mod:`cadjoint.descriptions`) agree with the wire models.

Core describes itself in TypedDicts; the viewer's schema models extend
those with source locations.  Two checks keep them in step: every key a
TypedDict declares is a field of its model, and what the shipped scenes'
declarations actually return has exactly the TypedDict's keys and types.
"""

from __future__ import annotations

import typing
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from cadjoint import descriptions as d
from cadjoint.viewer.schema import api, payloads
from cadjoint.viewer.worker.scene import _execute_scene

SCENES = Path(__file__).resolve().parents[2] / "scenes"

PAIRS = [
    (d.BoxSelectionDescription, payloads.BoxSelection),
    (d.SphereSelectionDescription, payloads.SphereSelection),
    (d.HalfspaceSelectionDescription, payloads.HalfspaceSelection),
    (d.CylinderSelectionDescription, payloads.CylinderSelection),
    (d.SideSelectionDescription, payloads.SideSelection),
    (d.PredicateSelectionDescription, payloads.PredicateSelection),
    (d.CombinedSelectionDescription, payloads.CombinedSelection),
    (d.ComplementSelectionDescription, payloads.ComplementSelection),
    *[(bc, payloads.StudyBcDescription) for bc in typing.get_args(d.BoundaryConditionDescription)],
    *[(study, payloads.StudyDescription) for study in typing.get_args(d.StudyDescription)],
    (d.DomainDescription, payloads.DomainEntry),
    (d.FluidDescription, payloads.FluidDescription),
    (d.SimMeshDescription, payloads.SimMeshPayload),
    (d.OptimizationDescription, payloads.OptimizationPayload),
    (d.ResultDescription, api.ResultSummary),
    (d.MaterialDescription, payloads.MaterialDefinition),
    (d.FaceDescription, payloads.ConstructionFace),
    (d.PlaneDescription, payloads.ConstructionPlane),
]


def _keys(shape: type) -> set[str]:
    return set(typing.get_type_hints(shape))


@pytest.mark.parametrize(("shape", "model"), PAIRS, ids=lambda item: item.__name__)
def test_every_described_key_is_a_field_of_the_wire_model(shape, model):
    missing = _keys(shape) - set(model.model_fields)
    assert not missing, f"{model.__name__} does not declare {sorted(missing)}"


def _conforms(value: dict, shape: type) -> None:
    assert set(value) == _keys(shape), f"{shape.__name__}: {sorted(set(value) ^ _keys(shape))}"
    TypeAdapter(shape).validate_python(value)


def _by_kind(value: dict, union: typing.Any, tag: str) -> type:
    for shape in typing.get_args(union):
        if value[tag] in typing.get_args(typing.get_type_hints(shape)[tag]):
            return shape
    raise AssertionError(f"no {union} member for {tag}={value[tag]!r}")


@pytest.mark.parametrize("scene", sorted(path.name for path in SCENES.glob("*.py")))
def test_the_shipped_scenes_describe_themselves_as_declared(scene):
    namespace = _execute_scene((SCENES / scene).read_text())
    for study in namespace["__studies__"]:
        described = study.describe()
        _conforms(described, _by_kind(described, d.StudyDescription, "kind"))
        for bc in described["bcs"]:
            _conforms(bc, _by_kind(bc, d.BoundaryConditionDescription, "type"))
            if "nodes" in bc:
                _conforms(bc["nodes"], _by_kind(bc["nodes"], d.SelectionDescription, "kind"))
    for mesh in namespace["__sim_meshes__"]:
        _conforms(mesh.describe(), d.SimMeshDescription)
    for optimization in namespace["__optimizations__"]:
        _conforms(optimization.describe(namespace["scene"]), d.OptimizationDescription)
