"""The construction *elements* of a program, as the properties window sees them.

The construction payload (:mod:`cadjoint.viewer.source_map.payload`) is built
from the objects a program *ran* — profiles and primitives, with the geometry
the viewport draws.  The properties window asks a different question: for
the thing the user selected, which arguments did the source write, which of
them are numbers it may rewrite, and which are expressions it may only show?
That is a static question, so this module answers it from the AST alone,
one entry per construction call the program contains:

- a **sketch** (``PolygonProfile``), and beside it its **plane** — stated
  ``SketchPlane(origin=..., normal=...)``, the default world plane, or a
  plane derived from other geometry (``SketchPlane.on(body.cap("+"))``);
- a **primitive** (``Solid.box``/``sphere``/``cylinder``);
- a **feature** (``extrude``/``revolve``/``loft``);
- a **boolean** (``Union``/``Difference``/``Intersection``).

Every argument carries the literal it resolved to (through a named
``Scalar``/``Vector`` where the program used one), the source text, and —
when the patch layer may rewrite it — the exact ``set_value`` /
``assign_material`` address to send.  The rule is the one the rest of the
source map enforces: a field is only editable when its literal was located
unambiguously and the contract table (:data:`EDITABLE_CALLS`) lists it.
"""

from __future__ import annotations

import ast
from typing import Any

from cadjoint.viewer.source_map.features import (
    FEATURE_CALL_KINDS,
    PLANE_CONSTRUCTORS,
    PRIMITIVE_CALL_KINDS,
)
from cadjoint.viewer.source_map.identity import Identity, build_identities, identity_at
from cadjoint.viewer.source_map.nodes import (
    _assignment_value,
    _call_namespace,
    _called_name,
    _editable_value_node,
    _is_construction_call,
    _is_profile_call,
    _line_offsets,
    _node_span,
    _resolved_container,
    parse_module,
)

#: Boolean combinators, keyed by call name.
BOOLEAN_CALL_KINDS = frozenset({"Union", "Difference", "Intersection"})

#: Positional parameter names of every call an element is built from, so a
#: positional argument is reported under the name the keyword would carry.
POSITIONAL_PARAMETERS: dict[str, tuple[str, ...]] = {
    "PolygonProfile": ("vertices", "plane", "name", "free"),
    "SketchPlane": ("origin", "normal", "x_axis"),
    "box": ("position", "rotation", "material", "name"),
    "sphere": ("position", "rotation", "material", "name"),
    "cylinder": ("position", "rotation", "material", "name"),
    "extrude": ("profile", "depth", "material", "draft", "twist"),
    "revolve": ("profile", "offset", "material"),
    "loft": ("profile_a", "profile_b", "height", "material"),
}

#: The world plane a ``PolygonProfile`` takes when it states none.
DEFAULT_PLANE_ORIGIN = [0.0, 0.0, 0.0]
DEFAULT_PLANE_NORMAL = [0.0, 0.0, 1.0]


def _editable_calls() -> dict[str, dict[str, int]]:
    # Imported lazily: the patch package imports the source map, not the
    # other way round, and this module must not close that loop at import.
    from cadjoint.viewer.patch.geometry import EDITABLE_CALLS

    return EDITABLE_CALLS


def _literal(node: ast.AST) -> Any:
    """The Python value of a numeric literal or a list/tuple of them."""
    value = ast.literal_eval(node)
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [float(item) for item in value]
    return float(value)


def _segment(source: str, node: ast.AST) -> str:
    return ast.get_source_segment(source, node) or ""


def _call_arguments(call: ast.Call, name: str) -> list[tuple[str, ast.AST]]:
    """``(argument name, value node)`` pairs, positionals named by the signature."""
    names = POSITIONAL_PARAMETERS.get(name, ())
    pairs: list[tuple[str, ast.AST]] = []
    for position, value in enumerate(call.args):
        if isinstance(value, ast.Starred):
            continue
        label = names[position] if position < len(names) else f"arg{position}"
        pairs.append((label, value))
    for keyword in call.keywords:
        if keyword.arg is not None:
            pairs.append((keyword.arg, keyword.value))
    return pairs


def _patch_address(
    op: str, name: str, argument: str, line: int, identifier: str | None
) -> dict[str, Any]:
    return {"op": op, "name": name, "argument": argument, "line": line, "id": identifier}


def _argument_entry(
    source: str,
    tree: ast.Module,
    offsets: list[int],
    name: str,
    value: ast.AST,
    *,
    call: str,
    patch_call: str,
    patch_argument: str | None,
    patch_line: int,
    patch_id: str | None,
) -> dict[str, Any]:
    """Classify one written argument and say how (whether) it can be rewritten.

    ``patch_call``/``patch_argument``/``patch_line``/``patch_id`` describe the
    ``set_value`` request that rewrites it — for a plane component that is the
    owning sketch's ``planeOrigin``, not the ``SketchPlane`` call itself, so the
    request survives the sketch having no plane yet.
    """
    text = _segment(source, value)
    entry: dict[str, Any] = {
        "name": name,
        "kind": "expression",
        "value": None,
        "text": text,
        "span": None,
        "parameter": None,
        "patch": None,
    }
    editable = _editable_calls()
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        entry.update(kind="string", value=value.value)
        return entry
    if isinstance(value, ast.Constant) and isinstance(value.value, bool):
        entry.update(kind="string", value=str(value.value))
        return entry
    if name == "material":
        # A material is a name or nothing the window can offer a choice for:
        # ``material=None`` and inline ``Material(...)`` calls stay expressions.
        if isinstance(value, ast.Name):
            entry.update(kind="reference", value=value.id)
            if call in FEATURE_CALL_KINDS | PRIMITIVE_CALL_KINDS:
                entry["patch"] = _patch_address(
                    "assign_material", call, "material", patch_line, patch_id
                )
        return entry
    literal = _editable_value_node(value, tree)
    if literal is not None:
        span = _node_span(source, offsets, literal)
        number = _literal(literal)
        entry.update(
            kind="vector" if isinstance(number, list) else "number",
            value=number,
            span=list(span) if span is not None else None,
            parameter=value.id if isinstance(value, ast.Name) else None,
        )
        argument = patch_argument or name
        if span is not None and argument in editable.get(patch_call, {}):
            entry["patch"] = _patch_address("set_value", patch_call, argument, patch_line, patch_id)
        return entry
    if isinstance(value, ast.Name):
        entry.update(kind="reference", value=value.id)
    return entry


def _plane_argument(call: ast.Call) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == "plane":
            return keyword.value
    return call.args[1] if len(call.args) > 1 else None


def _plain_sketch_plane(node: ast.AST, tree: ast.Module) -> ast.Call | None:
    """A literal ``SketchPlane(...)`` call, reached directly or through one name."""
    if isinstance(node, ast.Name):
        value = _assignment_value(tree, node.id, node.lineno)
        return _plain_sketch_plane(value, tree) if value is not None else None
    if (
        isinstance(node, ast.Call)
        and _called_name(node) == "SketchPlane"
        and _call_namespace(node) is None
    ):
        return node
    return None


def _sketch_plane_method(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return None
    owner = node.func.value
    if isinstance(owner, ast.Name) and owner.id == "SketchPlane":
        return node.func.attr if node.func.attr in PLANE_CONSTRUCTORS else None
    return None


def _default_argument(name: str, value: list[float], patch: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "kind": "default",
        "value": value,
        "text": "",
        "span": None,
        "parameter": None,
        "patch": patch,
    }


def _plane_element(
    source: str,
    tree: ast.Module,
    offsets: list[int],
    profile: ast.Call,
    sketch: dict[str, Any],
    identities: list[Identity],
) -> dict[str, Any]:
    """The plane of one sketch: stated, defaulted, or derived."""
    sketch_line = profile.lineno
    sketch_id = sketch["stableId"]
    plane_identity = identity_at(identities, sketch_line, {"plane"}) if sketch_id else None
    element: dict[str, Any] = {
        "id": plane_identity.id if plane_identity else f"plane@{sketch_line}",
        "stableId": plane_identity.id if plane_identity else None,
        "kind": "plane",
        "call": "SketchPlane",
        "line": sketch_line,
        "span": sketch["span"],
        "name": sketch["name"],
        "variable": None,
        "owner": sketch["id"],
        "arguments": [],
    }
    address = {
        "call": "PolygonProfile",
        "patch_call": "PolygonProfile",
        "patch_line": sketch_line,
        "patch_id": sketch_id,
    }
    argument = _plane_argument(profile)
    if argument is None:
        element["arguments"] = [
            _default_argument(
                "origin",
                DEFAULT_PLANE_ORIGIN,
                _patch_address(
                    "set_value", "PolygonProfile", "planeOrigin", sketch_line, sketch_id
                ),
            ),
            _default_argument(
                "normal",
                DEFAULT_PLANE_NORMAL,
                _patch_address(
                    "set_value", "PolygonProfile", "planeNormal", sketch_line, sketch_id
                ),
            ),
        ]
        return element

    plain = _plain_sketch_plane(argument, tree)
    if plain is not None:
        span = _node_span(source, offsets, plain)
        element["line"] = plain.lineno
        element["span"] = list(span) if span is not None else sketch["span"]
        element["variable"] = argument.id if isinstance(argument, ast.Name) else None
        written = dict(_call_arguments(plain, "SketchPlane"))
        arguments: list[dict[str, Any]] = []
        for component, default in (
            ("origin", DEFAULT_PLANE_ORIGIN),
            ("normal", DEFAULT_PLANE_NORMAL),
        ):
            patch_argument = "planeOrigin" if component == "origin" else "planeNormal"
            if component in written:
                arguments.append(
                    _argument_entry(
                        source,
                        tree,
                        offsets,
                        component,
                        written[component],
                        patch_argument=patch_argument,
                        **address,
                    )
                )
            else:
                arguments.append(
                    _default_argument(
                        component,
                        default,
                        _patch_address(
                            "set_value", "PolygonProfile", patch_argument, sketch_line, sketch_id
                        ),
                    )
                )
        for name, value in written.items():
            if name in {"origin", "normal"}:
                continue
            arguments.append(
                _argument_entry(source, tree, offsets, name, value, patch_argument=None, **address)
            )
        element["arguments"] = arguments
        return element

    method = _sketch_plane_method(argument)
    element["call"] = f"SketchPlane.{method}" if method else "expression"
    element["line"] = getattr(argument, "lineno", sketch_line)
    element["arguments"] = [
        {
            "name": "reference",
            "kind": "expression",
            "value": None,
            "text": _segment(source, argument),
            "span": None,
            "parameter": None,
            "patch": None,
        }
    ]
    return element


def _vertices_argument(source: str, tree: ast.Module, value: ast.AST) -> dict[str, Any]:
    container = _resolved_container(value, tree)
    count = len(container.elts) if container is not None else None
    return {
        "name": "vertices",
        "kind": "expression",
        "value": None if count is None else f"{count} points",
        "text": _segment(source, value),
        "span": None,
        "parameter": value.id if isinstance(value, ast.Name) else None,
        "patch": None,
    }


def _element_kind(name: str) -> str | None:
    if name == "PolygonProfile":
        return "sketch"
    if name in PRIMITIVE_CALL_KINDS:
        return "primitive"
    if name in FEATURE_CALL_KINDS:
        return "feature"
    if name in BOOLEAN_CALL_KINDS:
        return "boolean"
    return None


def _literal_name(call: ast.Call) -> str | None:
    for keyword in call.keywords:
        if (
            keyword.arg == "name"
            and isinstance(keyword.value, ast.Constant)
            and isinstance(keyword.value.value, str)
        ):
            return keyword.value.value
    return None


def build_element_payload(source: str) -> list[dict[str, Any]]:
    """Serialize every construction element of *source* for the properties window.

    Args:
        source: The program text.

    Returns:
        One dict per element, in source order, each with its ``id`` (the
        stable identity where one exists, else a synthetic one that is still
        stable across edits elsewhere in the file), the call it was built by,
        where it sits, and its written arguments with their patch addresses.
        An unparseable program yields an empty list; nothing here runs the
        program.
    """
    try:
        tree = parse_module(source)
    except SyntaxError:
        return []
    offsets = _line_offsets(source)
    identities = build_identities(source)
    identity_kinds = {
        "sketch": {"sketch"},
        "primitive": {"primitive"},
        "feature": {"feature"},
    }
    elements: list[dict[str, Any]] = []
    for statement in tree.body:
        bound = (
            statement.targets[0].id
            if isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            else None
        )
        for node in ast.walk(statement):
            name = _called_name(node)
            if not isinstance(node, ast.Call) or name is None:
                continue
            kind = _element_kind(name)
            if kind is None or (kind == "primitive" and not _is_construction_call(node)):
                continue
            variable = bound if getattr(statement, "value", None) is node else None
            span = _node_span(source, offsets, node)
            identity = (
                identity_at(identities, node.lineno, identity_kinds[kind])
                if kind in identity_kinds
                else None
            )
            if identity is not None:
                identifier = identity.id
            elif kind == "boolean" and variable is not None:
                identifier = f"boolean:{variable}"
            else:
                identifier = f"{kind}@{node.lineno}"
            element: dict[str, Any] = {
                "id": identifier,
                "stableId": identity.id if identity is not None else None,
                "kind": kind,
                "call": name,
                "line": node.lineno,
                "span": list(span) if span is not None else [0, 0],
                "name": _literal_name(node),
                "variable": variable,
                "owner": None,
                "arguments": [],
            }
            address = {
                "call": name,
                "patch_call": name,
                "patch_argument": None,
                "patch_line": node.lineno,
                "patch_id": identity.id if identity is not None else None,
            }
            arguments: list[dict[str, Any]] = []
            if kind == "boolean":
                operands = [
                    _segment(source, item)
                    for item in node.args
                    if not isinstance(item, ast.Starred)
                ]
                arguments.append(
                    {
                        "name": "operands",
                        "kind": "reference",
                        "value": ", ".join(operands),
                        "text": ", ".join(operands),
                        "span": None,
                        "parameter": None,
                        "patch": None,
                    }
                )
                for keyword in node.keywords:
                    if keyword.arg is None:
                        continue
                    arguments.append(
                        _argument_entry(
                            source, tree, offsets, keyword.arg, keyword.value, **address
                        )
                    )
            else:
                for label, value in _call_arguments(node, name):
                    if kind == "sketch" and label == "vertices":
                        arguments.append(_vertices_argument(source, tree, value))
                        continue
                    if kind == "sketch" and label == "plane":
                        # Reported as its own element right after the sketch.
                        continue
                    if label == "name" and isinstance(value, ast.Constant):
                        continue
                    arguments.append(
                        _argument_entry(source, tree, offsets, label, value, **address)
                    )
            element["arguments"] = arguments
            elements.append(element)
            if kind == "sketch" and _is_profile_call(node):
                elements.append(_plane_element(source, tree, offsets, node, element, identities))
    return elements


__all__ = ["BOOLEAN_CALL_KINDS", "POSITIONAL_PARAMETERS", "build_element_payload"]
