"""Deleting an object drops it from whichever Union assignment lists it."""

from __future__ import annotations

import textwrap

import pytest

from cadjoint.viewer.patch import PatchError, apply_operation
from cadjoint.viewer.patch.geometry import delete_object

SOURCE = textwrap.dedent(
    """
    from cadjoint.construction import Solid
    from cadjoint.sdf.boolean import Union

    a = Solid.box(size=[1.0, 1.0, 1.0], name="a")
    b = Solid.sphere(radius=0.5, position=[2.0, 0.0, 0.0], name="b")
    body = Union(a, b, smoothness=0.03)
    c = Solid.cylinder(radius=0.2, height=0.5, position=[0.0, 2.0, 0.0], name="c")
    scene = Union(body, c, smoothness=0.0)
    """
).lstrip()


def _line_of(text: str, needle: str) -> int:
    return text[: text.index(needle)].count("\n") + 1


def test_delete_drops_an_operand_of_a_named_sub_assembly():
    patched = delete_object(SOURCE, _line_of(SOURCE, "b = Solid.sphere"))
    assert "b = Solid.sphere" not in patched
    assert "body = Union(a, smoothness=0.03)" in patched
    # The scene, which never referenced `b` directly, is untouched.
    assert "scene = Union(body, c, smoothness=0.0)" in patched


def test_delete_still_drops_a_direct_scene_operand():
    patched = delete_object(SOURCE, _line_of(SOURCE, "c = Solid.cylinder"))
    assert "c = Solid.cylinder" not in patched
    assert "scene = Union(body, smoothness=0.0)" in patched


def test_delete_still_refuses_a_use_outside_any_union():
    source = SOURCE + "\nprobe = a\n"
    with pytest.raises(PatchError, match="used elsewhere"):
        delete_object(source, _line_of(source, "a = Solid.box"))


def test_registry_path_matches_the_direct_call():
    line = _line_of(SOURCE, "b = Solid.sphere")
    assert apply_operation(SOURCE, "delete_object", line=line) == delete_object(SOURCE, line)


def test_delete_drops_an_operand_of_a_qualified_union():
    source = SOURCE.replace(
        "from cadjoint.sdf.boolean import Union", "from cadjoint.sdf import boolean"
    ).replace("Union(", "boolean.Union(")
    patched = delete_object(source, _line_of(source, "c = Solid.cylinder"))
    assert "c = Solid.cylinder" not in patched
    assert "scene = boolean.Union(body, smoothness=0.0)" in patched
