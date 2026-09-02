"""Editing a material's own properties — the physical ones especially.

A material's optical properties are always stated, because they have
defaults; its physical ones are stated only when the scene cares about
physics.  So the interesting half of ``set_material_property`` is the half
``set_value`` never had to do: *add* a keyword that is not there, and take
one back out again.  These tests pin that, the brackets a value must fall
inside, and the one shape the operation refuses — a material built by a
catalogue factory, which has no keyword to edit at all.

The starter program is the fixture throughout: it is the text every user
meets first, its ``copper`` states all seven physical properties, and its
``fr4`` states none of them.
"""

from __future__ import annotations

import ast

import pytest

from cadjoint.viewer._example_scene import EXAMPLE_SOURCE
from cadjoint.viewer._patch_requests import patch_source
from cadjoint.viewer._source_map import PLAYGROUND_FILENAME, build_material_payload
from cadjoint.viewer._worker_scene import _execute_scene
from cadjoint.viewer.patch import PatchError, apply_operation
from cadjoint.viewer.patch.materials import (
    EDITABLE_PROPERTIES,
    PROPERTY_BOUNDS,
    set_material_property,
)
from cadjoint.viewer.source_map.capture import capture_profiles

PHYSICAL = (
    "density",
    "conductivity",
    "specific_heat",
    "youngs_modulus",
    "poisson_ratio",
    "thermal_expansion",
    "yield_strength",
)

#: A value comfortably inside every property's bracket, per property.
INSIDE = {
    "roughness": 0.62,
    "metallic": 0.25,
    "opacity": 0.75,
    "ior": 1.9,
    "reflectivity": 0.15,
    "density": 8950.5,
    "conductivity": 401.25,
    "specific_heat": 390.5,
    "youngs_modulus": 118500000000.0,
    "poisson_ratio": 0.345,
    "thermal_expansion": 1.72e-05,
    "yield_strength": 71500000.0,
}

CATALOGUE_SOURCE = '''"""A scene whose material comes from the catalogue."""

from cadjoint.construction import Solid
from cadjoint.materials import aluminium_6061

alu = aluminium_6061()
slug = Solid.sphere(radius=0.5, material=alu, name="slug")
scene = slug
'''


def keyword_value(source: str, variable: str, keyword: str):
    """The literal a material's keyword carries, or None when it is absent."""
    tree = ast.parse(source)
    for statement in tree.body:
        if not (
            isinstance(statement, ast.Assign)
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == variable
            and isinstance(statement.value, ast.Call)
        ):
            continue
        for item in statement.value.keywords:
            if item.arg == keyword:
                return ast.literal_eval(item.value)
        return None
    raise AssertionError(f"no assignment to {variable!r}")


def patched(**request) -> str:
    """Apply one request to the starter program, asserting it was accepted."""
    result = patch_source({"source": EXAMPLE_SOURCE, **request})
    assert result["ok"] is True, result.get("error")
    return result["source"]


class TestSettingAStatedProperty:
    """``copper`` states all seven, so every one of them is a literal rewrite."""

    @pytest.mark.parametrize("key", PHYSICAL)
    def test_the_value_is_replaced_in_place(self, key: str):
        source = patched(
            op="set_material_property", material="copper", property=key, value=INSIDE[key]
        )
        assert keyword_value(source, "copper", key) == INSIDE[key]

    @pytest.mark.parametrize("key", PHYSICAL)
    def test_no_line_moves_and_only_that_line_changes(self, key: str):
        source = patched(
            op="set_material_property", material="copper", property=key, value=INSIDE[key]
        )
        before, after = EXAMPLE_SOURCE.splitlines(), source.splitlines()
        assert len(before) == len(after)
        assert sum(1 for a, b in zip(before, after) if a != b) == 1

    def test_the_other_materials_are_untouched(self):
        source = patched(
            op="set_material_property", material="copper", property="density", value=8950.5
        )
        assert keyword_value(source, "aluminum", "density") == 2700.0
        assert keyword_value(source, "steel", "density") == 7870.0

    def test_an_optical_property_goes_through_the_same_door(self):
        source = patched(
            op="set_material_property", material="copper", property="roughness", value=0.62
        )
        assert keyword_value(source, "copper", "roughness") == 0.62


class TestAddingAnUnstatedProperty:
    """``fr4`` states no physics at all, which is the case that matters."""

    @pytest.mark.parametrize("key", PHYSICAL)
    def test_the_keyword_is_added(self, key: str):
        assert keyword_value(EXAMPLE_SOURCE, "fr4", key) is None
        source = patched(
            op="set_material_property", material="fr4", property=key, value=INSIDE[key]
        )
        assert keyword_value(source, "fr4", key) == INSIDE[key]

    def test_every_other_argument_survives(self):
        source = patched(
            op="set_material_property", material="fr4", property="density", value=1850.0
        )
        assert keyword_value(source, "fr4", "color") == [0.10, 0.36, 0.22]
        assert keyword_value(source, "fr4", "roughness") == 0.85
        assert keyword_value(source, "fr4", "metallic") == 0.0

    def test_a_keyword_that_fits_does_not_move_a_line(self):
        source = patched(
            op="set_material_property", material="fr4", property="density", value=1850.0
        )
        assert len(source.splitlines()) == len(EXAMPLE_SOURCE.splitlines())

    def test_a_keyword_that_does_not_fit_wraps_onto_its_own_line(self):
        source = (
            "from cadjoint.render import Material\n"
            'brass = Material(name="brass", color=[0.7, 0.6, 0.2], '
            "roughness=0.3, metallic=0.9, ior=1.5)\n"
        )
        result = set_material_property(source, "density", 8500.0, material="brass")
        lines = result.splitlines()
        assert len(lines) == len(source.splitlines()) + 1
        assert lines[1].endswith("ior=1.5,")
        assert lines[2] == "    density=8500.0)"
        assert all(len(line) <= 100 for line in lines)
        assert keyword_value(result, "brass", "ior") == 1.5

    def test_adding_to_a_call_with_no_arguments_is_refused(self):
        source = "from cadjoint.render import Material\nbare = Material()\n"
        with pytest.raises(PatchError, match="no arguments"):
            set_material_property(source, "density", 1000.0, material="bare")


class TestRemovingAProperty:
    @pytest.mark.parametrize("key", PHYSICAL)
    def test_null_takes_the_keyword_out(self, key: str):
        source = patched(op="set_material_property", material="copper", property=key, value=None)
        assert keyword_value(source, "copper", key) is None
        assert len(source.splitlines()) == len(EXAMPLE_SOURCE.splitlines()) - 1

    @pytest.mark.parametrize("key", PHYSICAL)
    def test_the_six_others_stay_exactly_as_they_were(self, key: str):
        source = patched(op="set_material_property", material="copper", property=key, value=None)
        for other in PHYSICAL:
            if other != key:
                assert keyword_value(source, "copper", other) == keyword_value(
                    EXAMPLE_SOURCE, "copper", other
                )

    @pytest.mark.parametrize("key", PHYSICAL)
    def test_removing_then_adding_it_back_restores_the_value(self, key: str):
        original = keyword_value(EXAMPLE_SOURCE, "copper", key)
        removed = patched(op="set_material_property", material="copper", property=key, value=None)
        result = patch_source(
            {
                "source": removed,
                "op": "set_material_property",
                "material": "copper",
                "property": key,
                "value": original,
            }
        )
        assert result["ok"] is True, result.get("error")
        assert keyword_value(result["source"], "copper", key) == original

    def test_removing_what_is_already_absent_changes_nothing(self):
        source = patched(op="set_material_property", material="fr4", property="density", value=None)
        assert source == EXAMPLE_SOURCE

    def test_a_keyword_sharing_its_line_is_cut_out_in_place(self):
        source = (
            "from cadjoint.render import Material\n"
            'fr4 = Material(name="fr4", color=[0.1, 0.4, 0.2], roughness=0.85, metallic=0.0)\n'
        )
        result = set_material_property(source, "roughness", None, material="fr4")
        assert len(result.splitlines()) == len(source.splitlines())
        assert result.splitlines()[1] == (
            'fr4 = Material(name="fr4", color=[0.1, 0.4, 0.2], metallic=0.0)'
        )

    def test_the_last_keyword_takes_its_comma_with_it(self):
        source = (
            "from cadjoint.render import Material\n"
            'fr4 = Material(name="fr4", roughness=0.85, metallic=0.0)\n'
        )
        result = set_material_property(source, "metallic", None, material="fr4")
        assert result.splitlines()[1] == 'fr4 = Material(name="fr4", roughness=0.85)'


class TestBounds:
    """The brackets ``Material`` enforces when ``free=True``, and no others."""

    @pytest.mark.parametrize(
        ("key", "value", "message"),
        [
            ("density", 0.5, "`density` must be a number from 1 to 25000 kg/m^3."),
            ("density", 25001.0, "`density` must be a number from 1 to 25000 kg/m^3."),
            (
                "conductivity",
                0.0,
                "`conductivity` must be a number from 0.001 to 3000 W/(m*K).",
            ),
            (
                "conductivity",
                3001.0,
                "`conductivity` must be a number from 0.001 to 3000 W/(m*K).",
            ),
            (
                "specific_heat",
                0.5,
                "`specific_heat` must be a number from 1 to 10000 J/(kg*K).",
            ),
            (
                "youngs_modulus",
                999.0,
                "`youngs_modulus` must be a number from 1000 to 1e12 Pa.",
            ),
            (
                "poisson_ratio",
                0.5,
                "`poisson_ratio` must be a number from 0 to 0.499 (dimensionless).",
            ),
            (
                "thermal_expansion",
                0.01,
                "`thermal_expansion` must be a number from 0 to 0.001 1/K.",
            ),
            (
                "yield_strength",
                1e12,
                "`yield_strength` must be a number from 1000 to 1e11 Pa.",
            ),
            ("roughness", 1.5, "`roughness` must be a number from 0 to 1 (dimensionless)."),
            ("ior", 0.5, "`ior` must be a number from 1 to 3 (dimensionless)."),
        ],
    )
    def test_a_value_outside_the_bracket_is_refused_by_name_and_unit(self, key, value, message):
        result = patch_source(
            {
                "source": EXAMPLE_SOURCE,
                "op": "set_material_property",
                "material": "copper",
                "property": key,
                "value": value,
            }
        )
        assert result == {"ok": False, "error": message}

    @pytest.mark.parametrize("key", EDITABLE_PROPERTIES)
    def test_both_ends_of_every_bracket_are_accepted(self, key: str):
        low, high = PROPERTY_BOUNDS[key]
        for value in (low, high):
            result = patch_source(
                {
                    "source": EXAMPLE_SOURCE,
                    "op": "set_material_property",
                    "material": "copper",
                    "property": key,
                    "value": value,
                }
            )
            assert result["ok"] is True, result.get("error")

    def test_a_value_that_is_not_a_number_is_refused(self):
        result = patch_source(
            {
                "source": EXAMPLE_SOURCE,
                "op": "set_material_property",
                "material": "copper",
                "property": "density",
                "value": "8940",
            }
        )
        assert result["ok"] is False
        assert result["error"].startswith("`density` must be a number")

    def test_a_property_the_operation_does_not_edit_is_refused(self):
        result = patch_source(
            {
                "source": EXAMPLE_SOURCE,
                "op": "set_material_property",
                "material": "copper",
                "property": "color",
                "value": 0.5,
            }
        )
        assert result["ok"] is False
        assert result["error"].startswith("Material `property` must be one of: roughness,")


class TestAddressingTheMaterial:
    def test_a_stable_id_and_a_name_reach_the_same_definition(self):
        by_id = patch_source(
            {
                "source": EXAMPLE_SOURCE,
                "op": "set_material_property",
                "id": "assign:copper",
                "property": "density",
                "value": 8950.5,
            }
        )
        by_name = patch_source(
            {
                "source": EXAMPLE_SOURCE,
                "op": "set_material_property",
                "material": "copper",
                "property": "density",
                "value": 8950.5,
            }
        )
        assert by_id == by_name
        assert by_id["ok"] is True

    def test_the_payload_index_reaches_the_same_definition(self):
        by_index = patched(op="set_material_property", material=1, property="density", value=8950.5)
        by_name = patched(
            op="set_material_property", material="copper", property="density", value=8950.5
        )
        assert by_index == by_name

    def test_an_index_past_the_end_is_refused(self):
        result = patch_source(
            {
                "source": EXAMPLE_SOURCE,
                "op": "set_material_property",
                "material": 99,
                "property": "density",
                "value": 8950.5,
            }
        )
        assert result == {
            "ok": False,
            "error": "Material index 99 is out of range; the program declares 7.",
        }

    def test_an_unknown_name_lists_what_the_program_declares(self):
        result = patch_source(
            {
                "source": EXAMPLE_SOURCE,
                "op": "set_material_property",
                "material": "brass",
                "property": "density",
                "value": 8950.5,
            }
        )
        assert result["ok"] is False
        assert result["error"].startswith("No single material named 'brass'; the program declares:")
        assert "'copper'" in result["error"]

    def test_a_material_is_not_addressed_by_nothing(self):
        result = patch_source(
            {
                "source": EXAMPLE_SOURCE,
                "op": "set_material_property",
                "property": "density",
                "value": 8950.5,
            }
        )
        assert result == {
            "ok": False,
            "error": "The patch request needs `material` as a name or a non-negative index.",
        }

    def test_an_id_naming_something_that_is_not_a_material_is_refused(self):
        result = patch_source(
            {
                "source": EXAMPLE_SOURCE,
                "op": "set_material_property",
                "id": "assign:slug",
                "property": "density",
                "value": 8950.5,
            }
        )
        assert result["ok"] is False
        assert "which `set_material_property` cannot address" in result["error"]


class TestACatalogueMaterial:
    """``alu = aluminium_6061()`` has no keyword to edit; say so, or expand it."""

    def test_the_edit_is_refused_and_says_what_to_do(self):
        result = patch_source(
            {
                "source": CATALOGUE_SOURCE,
                "op": "set_material_property",
                "material": "alu",
                "property": "density",
                "value": 2705.0,
            }
        )
        assert result == {
            "ok": False,
            "error": (
                "`alu` is built by the catalogue factory `aluminium_6061()`, which has no "
                "property keyword to edit. Convert it to a literal `Material(...)` first — "
                "send this request again with `expand: true` to have that done for you."
            ),
        }

    def test_expand_converts_it_and_then_applies_the_edit(self):
        result = patch_source(
            {
                "source": CATALOGUE_SOURCE,
                "op": "set_material_property",
                "material": "alu",
                "property": "density",
                "value": 2705.0,
                "expand": True,
            }
        )
        assert result["ok"] is True, result.get("error")
        source = result["source"]
        assert "alu = Material(\n" in source
        assert keyword_value(source, "alu", "density") == 2705.0
        assert keyword_value(source, "alu", "conductivity") == 167.0
        assert keyword_value(source, "alu", "name") == "aluminium_6061"
        assert "from cadjoint.render import Material" in source
        # Everything else in the program is left where it was.
        assert 'slug = Solid.sphere(radius=0.5, material=alu, name="slug")' in source

    def test_the_expanded_material_is_still_the_material_it_was(self):
        source = patch_source(
            {
                "source": CATALOGUE_SOURCE,
                "op": "set_material_property",
                "material": "alu",
                "property": "density",
                "value": 2705.0,
                "expand": True,
            }
        )["source"]
        from cadjoint.materials import aluminium_6061

        original = aluminium_6061().describe()["physical"]
        for key, value in original.items():
            if key == "density":
                continue
            assert keyword_value(source, "alu", key) == pytest.approx(value)

    def test_expanding_twice_is_only_the_property_edit_the_second_time(self):
        once = patch_source(
            {
                "source": CATALOGUE_SOURCE,
                "op": "set_material_property",
                "material": "alu",
                "property": "density",
                "value": 2705.0,
                "expand": True,
            }
        )["source"]
        twice = patch_source(
            {
                "source": once,
                "op": "set_material_property",
                "material": "alu",
                "property": "density",
                "value": 2705.0,
                "expand": True,
            }
        )
        assert twice["ok"] is True, twice.get("error")
        assert twice["source"] == once

    def test_a_factory_called_with_arguments_is_not_expanded(self):
        source = CATALOGUE_SOURCE.replace("aluminium_6061()", "aluminium_6061(0.5)")
        with pytest.raises(PatchError, match="called with arguments"):
            set_material_property(source, "density", 2705.0, material="alu", expand=True)


class TestThePayloadAndTheProgram:
    def test_the_payload_publishes_a_span_for_every_stated_physical_key(self):
        namespace = _execute_scene(
            EXAMPLE_SOURCE,
            capture=(("__profiles__", lambda: capture_profiles(PLAYGROUND_FILENAME)),),
        )
        entries = {item["name"]: item for item in build_material_payload(namespace, EXAMPLE_SOURCE)}
        copper = entries["copper"]
        for key in PHYSICAL:
            assert copper["physical"][key] is not None
            assert key in copper["spans"], key
            start, end = copper["spans"][key]
            assert (
                EXAMPLE_SOURCE[start:end]
                == {
                    "density": "8940.0",
                    "conductivity": "391.0",
                    "specific_heat": "385.0",
                    "youngs_modulus": "117e9",
                    "poisson_ratio": "0.34",
                    "thermal_expansion": "17.0e-6",
                    "yield_strength": "69e6",
                }[key]
            )
        # An unstated property has no span, which is what tells the inspector
        # to offer a "state it" action instead of a draggable number.
        assert all(entries["fr4"]["physical"][key] is None for key in PHYSICAL)
        assert not any(key in entries["fr4"]["spans"] for key in PHYSICAL)

    def test_the_patched_starter_still_compiles_and_reports_the_new_value(self):
        source = patched(
            op="set_material_property", material="fr4", property="density", value=1850.0
        )
        source = patch_source(
            {
                "source": source,
                "op": "set_material_property",
                "material": "copper",
                "property": "youngs_modulus",
                "value": 118500000000.0,
            }
        )["source"]
        namespace = _execute_scene(
            source, capture=(("__profiles__", lambda: capture_profiles(PLAYGROUND_FILENAME)),)
        )
        entries = {item["name"]: item for item in build_material_payload(namespace, source)}
        assert entries["fr4"]["physical"]["density"] == pytest.approx(1850.0)
        assert entries["copper"]["physical"]["youngs_modulus"] == pytest.approx(1.185e11)
        assert "density" in entries["fr4"]["spans"]


class TestTheRegistry:
    def test_the_registry_path_matches_the_direct_call(self):
        direct = set_material_property(
            EXAMPLE_SOURCE, "density", 8950.5, material="copper", line=None, expand=False
        )
        through = apply_operation(
            EXAMPLE_SOURCE,
            "set_material_property",
            property="density",
            value=8950.5,
            material="copper",
            expand=False,
        )
        assert direct == through

    def test_the_bounds_are_the_ones_material_itself_enforces(self):
        from cadjoint.render.material import _BOUNDS

        for key, bracket in PROPERTY_BOUNDS.items():
            assert bracket == pytest.approx(_BOUNDS[key])
        assert set(PROPERTY_BOUNDS) == set(EDITABLE_PROPERTIES)
