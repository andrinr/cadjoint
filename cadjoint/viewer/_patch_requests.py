"""Validating ``/patch`` requests, and applying the ones that check out.

``/patch`` never executes the user's program — it only rewrites literals in
it — so what the endpoint needs is a complete description of what the
frontend is allowed to ask for.  That description is :data:`PATCH_VALIDATORS`:
one validator per operation, each checking that operation's fields and
returning the keyword arguments :func:`cadjoint.viewer._patch.apply_operation`
will run with, or the rejection to send back instead.

Every rejection message the endpoint can produce lives in this module, and
the tests pin them, so treat the strings as the API they are.  A validator
returns ``(error, arguments)``: exactly one of the two is meaningful, and
the checks inside one validator run in the order a caller would hit them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cadjoint.viewer._limits import OVERSIZED_SOURCE_ERROR, exceeds_source_limit
from cadjoint.viewer._patch import OPERATIONS, PatchError, apply_operation
from cadjoint.viewer.patch.materials import (
    EDITABLE_PROPERTIES,
    PROPERTY_BOUNDS,
    property_range_error,
)
from cadjoint.viewer.source_map.identity import Identity, identity_index

# ``(error, arguments)``: a rejection to return, or the keyword arguments the
# operation runs with.  ``error`` is None exactly when the request is good.
Checked = tuple[dict[str, Any] | None, dict[str, Any]]
Validator = Callable[[dict[str, Any]], Checked]


def _error(message: str) -> dict[str, Any]:
    """One rejected request, in the shape every endpoint answers with."""
    return {"ok": False, "error": message}


# ── Stable identities ───────────────────────────────────────────────────────
#
# A request may address its target by the stable id the payload published
# instead of by the line the payload happened to report.  An id is resolved
# against the *current* text and then written into the legacy fields, so every
# validator below keeps checking exactly what it always checked — and a
# frontend that still sends lines keeps working unchanged.

_UNRESOLVED_ID = "No statement in this program has the id {identifier!r}."
_WRONG_ID_KIND = "The id {identifier!r} names a {kind}, which `{operation}` cannot address."
_NO_ID = "The patch operation `{operation}` creates a new object, so it takes no `id`."
_LOFT_IDS = "`add_loft` names its two sketches with `id_a` and `id_b`, not `id`."

#: Identity kinds whose id resolves to a ``line`` and nothing else.
_LINE_KINDS = frozenset({"sketch", "primitive", "feature", "material", "plane"})

#: Identity kinds that name a top-level declaration a request refers to by
#: name or index; the id supplies the index under this request key.
_DECLARATION_KEYS = {"study": "study", "mesh": "mesh", "optimization": "optimization"}

#: What each operation's ``id`` is allowed to name.  One entry per operation in
#: :data:`PATCH_VALIDATORS`, so a mis-aimed id is reported as the mis-aim it is
#: rather than as a missing field three checks later.  An empty set means the
#: operation creates something and has no existing target to address.
_ID_TARGETS: dict[str, frozenset[str]] = {
    "set_vertex": frozenset({"sketch", "vertex"}),
    "insert_vertex": frozenset({"sketch", "vertex"}),
    "delete_vertex": frozenset({"sketch", "vertex"}),
    "set_value": frozenset({"sketch", "primitive", "feature", "material", "plane"}),
    "add_primitive": frozenset(),
    "add_material": frozenset(),
    "assign_material": frozenset({"sketch", "primitive", "feature"}),
    "set_material_property": frozenset({"material"}),
    "add_sketch": frozenset(),
    "set_sketch_plane": frozenset({"sketch", "plane"}),
    "add_extrusion": frozenset({"sketch"}),
    "add_revolution": frozenset({"sketch"}),
    "add_loft": frozenset(),
    "add_constraint": frozenset({"sketch"}),
    "delete_constraint": frozenset({"sketch", "constraint"}),
    "set_constraint_value": frozenset({"sketch", "constraint"}),
    "solve_sketch": frozenset({"sketch"}),
    "delete_object": frozenset({"sketch", "primitive", "feature"}),
    "add_study": frozenset(),
    "delete_study": frozenset({"study"}),
    "add_study_bc": frozenset({"study"}),
    "delete_study_bc": frozenset({"study", "bc"}),
    "set_study_value": frozenset({"study", "bc"}),
    "add_mesh": frozenset(),
    "delete_mesh": frozenset({"mesh"}),
    "set_mesh_value": frozenset({"mesh"}),
    "delete_optimization": frozenset({"optimization"}),
    "set_optimization_value": frozenset({"optimization"}),
}

#: ``add_loft`` joins two sketches, and a face reference names a feature.
_LOFT_TARGETS = frozenset({"sketch"})
_OWNER_TARGETS = frozenset({"feature", "primitive"})


def _derived_fields(identity: Identity, index: dict[str, Identity]) -> dict[str, Any] | None:
    """The legacy request fields an id stands in for, or None for a kind it cannot fill."""
    if identity.kind in _LINE_KINDS:
        return {"line": identity.line}
    if identity.kind == "vertex":
        return {"line": identity.line, "index": identity.index}
    if identity.kind == "constraint":
        owner = index.get(identity.owner or "")
        return {"line": owner.line, "index": identity.index} if owner is not None else None
    if identity.kind == "bc":
        owner = index.get(identity.owner or "")
        return {"study": owner.index, "bc": identity.index} if owner is not None else None
    key = _DECLARATION_KEYS.get(identity.kind)
    return {key: identity.index} if key is not None else None


def _resolve_identifier(
    identifier: Any,
    index: dict[str, Identity],
    allowed: frozenset[str],
    operation: str,
    key: str = "id",
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """``(error, fields)`` for one id, checked against the current source."""
    if not isinstance(identifier, str) or not identifier.strip():
        return _error(f"The patch request needs `{key}` as a non-empty string."), {}
    identity = index.get(identifier)
    if identity is None:
        return _error(_UNRESOLVED_ID.format(identifier=identifier)), {}
    if identity.kind not in allowed:
        return _error(
            _WRONG_ID_KIND.format(identifier=identifier, kind=identity.kind, operation=operation)
        ), {}
    fields = _derived_fields(identity, index)
    if fields is None:
        return _error(_UNRESOLVED_ID.format(identifier=identifier)), {}
    return None, fields


def _resolved_request(request: dict[str, Any], source: str, operation: str) -> Checked:
    """Rewrite a request's stable ids into the line/index fields validators read.

    ``id`` names the target of the operation; ``id_a``/``id_b`` the two
    sketches a loft joins; a string ``reference.owner`` the feature whose face
    a sketch plane is being planted on.  A request carrying none of those is
    returned untouched, so the common path pays nothing.

    Args:
        request: The raw request object.
        source: The program text the ids are resolved against.
        operation: The requested operation, which decides what an id may name.

    Returns:
        ``(error, request)``: a rejection, or the request with the ids
        replaced by the fields they stand for.
    """
    reference = request.get("reference")
    owner_is_id = isinstance(reference, dict) and isinstance(reference.get("owner"), str)
    keys = [key for key in ("id", "id_a", "id_b") if key in request]
    if not keys and not owner_is_id:
        return None, request

    index = identity_index(source)
    resolved = dict(request)
    for key in keys:
        allowed = _LOFT_TARGETS if key != "id" else _ID_TARGETS.get(operation, frozenset())
        if not allowed:
            return _error(
                _LOFT_IDS if operation == "add_loft" else _NO_ID.format(operation=operation)
            ), {}
        error, fields = _resolve_identifier(request[key], index, allowed, operation, key)
        if error is not None:
            return error, {}
        if key == "id":
            resolved.update(fields)
        else:
            resolved["line_a" if key == "id_a" else "line_b"] = fields["line"]
    if owner_is_id:
        error, fields = _resolve_identifier(
            reference["owner"], index, _OWNER_TARGETS, operation, "reference.owner"
        )
        if error is not None:
            return error, {}
        resolved["reference"] = {**reference, "owner": fields["line"]}
    return None, resolved


def _integer(value: Any) -> bool:
    """True for a plain integer — a ``bool`` does not count as one here."""
    return isinstance(value, int) and not isinstance(value, bool)


def _number(value: Any) -> bool:
    """True for a plain number — a ``bool`` does not count as one here."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _numbers(value: Any, count: int | None = None) -> list[float] | None:
    """Validate a list of plain numbers, optionally of a fixed length."""
    if not isinstance(value, (list, tuple)):
        return None
    if count is not None and len(value) != count:
        return None
    if not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
        return None
    return [float(item) for item in value]


def _scalar_or_numbers(value: Any) -> float | list[float] | None:
    """A plain number or a list of them — whichever the request carries.

    None means neither, which every caller reports with its own message.
    """
    if _number(value):
        return float(value)
    return _numbers(value)


# ── Geometry ────────────────────────────────────────────────────────────────


def _validate_add_sketch(request: dict[str, Any]) -> Checked:
    origin = _numbers(request.get("origin"), 3)
    if origin is None:
        return _error("The patch request needs `origin` as three numbers."), {}
    return None, {"origin": origin}


def _validate_add_primitive(request: dict[str, Any]) -> Checked:
    kind = request.get("kind")
    if not isinstance(kind, str):
        return _error("The patch request needs a string `kind`."), {}
    position = _numbers(request.get("position"), 3)
    if position is None:
        return _error("The patch request needs `position` as three numbers."), {}
    raw = request.get("dimensions")
    if not isinstance(raw, dict):
        return _error("The patch request needs a `dimensions` object."), {}
    dimensions: dict[str, Any] = {}
    for key, value in raw.items():
        if _number(value):
            dimensions[key] = float(value)
            continue
        vector = _numbers(value)
        if vector is None:
            return _error(f"Dimension `{key}` must be a number or numbers."), {}
        dimensions[key] = vector
    return None, {"kind": kind, "position": position, "dimensions": dimensions}


# Material properties the request may set, each with its ``(low, high, default)``.
_MATERIAL_RANGES = {
    "roughness": (0.0, 1.0, 0.4),
    "metallic": (0.0, 1.0, 0.0),
    "opacity": (0.0, 1.0, 1.0),
    "ior": (1.0, 3.0, 1.45),
    "reflectivity": (0.0, 1.0, 0.0),
}


def _validate_add_material(request: dict[str, Any]) -> Checked:
    color = _numbers(request.get("color"), 3)
    if color is None or any(value < 0.0 or value > 1.0 for value in color):
        return _error("The patch request needs `color` as three numbers from 0 to 1."), {}
    properties: dict[str, float] = {}
    for key, (low, high, default) in _MATERIAL_RANGES.items():
        raw = request.get(key, default)
        if not _number(raw) or not low <= float(raw) <= high:
            return _error(f"The patch request needs `{key}` from {low:g} to {high:g}."), {}
        properties[key] = float(raw)
    return None, {"color": color, **properties}


def _validate_set_material_property(request: dict[str, Any]) -> Checked:
    """One property of one material: which material, which property, what value.

    The material is named the way declarations are — by the payload's index or
    by a name — with a stable ``id`` resolving to the definition's ``line``
    instead.  ``value`` is a number inside that property's bracket, or ``null``
    to take the keyword back out of the call.
    """
    arguments: dict[str, Any] = {}
    line = request.get("line")
    material = request.get("material")
    if line is not None:
        if not _integer(line):
            return _error("The patch request needs an integer `line`."), {}
        arguments["line"] = line
    elif (_integer(material) and material >= 0) or (isinstance(material, str) and material.strip()):
        arguments["material"] = material
    else:
        return _error("The patch request needs `material` as a name or a non-negative index."), {}
    name = request.get("property")
    if not isinstance(name, str) or name not in EDITABLE_PROPERTIES:
        allowed = ", ".join(EDITABLE_PROPERTIES)
        return _error(f"Material `property` must be one of: {allowed}."), {}
    arguments["property"] = name
    raw_value = request.get("value")
    if raw_value is None:
        arguments["value"] = None
    else:
        low, high = PROPERTY_BOUNDS[name]
        if not _number(raw_value) or not low <= float(raw_value) <= high:
            return _error(property_range_error(name)), {}
        arguments["value"] = float(raw_value)
    expand = request.get("expand", False)
    if not isinstance(expand, bool):
        return _error("The patch request needs `expand` as a boolean."), {}
    arguments["expand"] = expand
    return None, arguments


def _validate_assign_material(request: dict[str, Any]) -> Checked:
    line = request.get("line")
    material = request.get("material")
    if not _integer(line):
        return _error("The patch request needs an integer `line`."), {}
    if not isinstance(material, str) or not material.isidentifier():
        return _error("The patch request needs `material` as a Python identifier."), {}
    return None, {"line": line, "material": material}


# Each reference kind and the extra field it carries beside ``owner``.
_PLANE_REFERENCE_FIELDS = {"cap": "sign", "side": "edge", "face": "key", "tangent": "near"}


def _validate_plane_reference(raw: Any) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Check the reference a sketch's plane is being planted on.

    A face reference names a feature by the line the payload reported for it,
    plus the one field that picks the face out of that feature.  ``world`` is
    the way back: an explicit origin and normal, no reference at all.
    """
    if not isinstance(raw, dict):
        return _error("The patch request needs `reference` as an object."), {}
    kind = raw.get("kind")
    if kind == "world":
        origin = _numbers(raw.get("origin"), 3)
        normal = _numbers(raw.get("normal"), 3)
        if origin is None or normal is None:
            return _error("A `world` plane needs `origin` and `normal` as three numbers."), {}
        if not any(abs(component) > 1e-9 for component in normal):
            return _error("A sketch-plane normal must not be zero."), {}
        return None, {"kind": "world", "origin": origin, "normal": normal}
    if kind not in _PLANE_REFERENCE_FIELDS:
        allowed = ", ".join(sorted({*_PLANE_REFERENCE_FIELDS, "world"}))
        return _error(f"Plane `reference.kind` must be one of: {allowed}."), {}
    owner = raw.get("owner")
    if not _integer(owner):
        return _error("The plane reference needs an integer `owner` line."), {}
    reference: dict[str, Any] = {"kind": kind, "owner": owner}
    field = _PLANE_REFERENCE_FIELDS[kind]
    value = raw.get(field)
    if kind == "cap":
        if not isinstance(value, str) or value not in {"+", "-"}:
            return _error("A cap reference needs `sign` as `+` or `-`."), {}
        reference["sign"] = value
    elif kind == "side":
        if not _integer(value) or value < 0:
            return _error("A side reference needs a non-negative `edge` index."), {}
        reference["edge"] = value
    elif kind == "face":
        if not isinstance(value, str) or not value.strip():
            return _error("A face reference needs a non-empty `key`."), {}
        reference["key"] = value
    else:
        near = _numbers(value, 3)
        if near is None:
            return _error("A tangent reference needs `near` as three numbers."), {}
        reference["near"] = near
    return None, reference


def _validate_set_sketch_plane(request: dict[str, Any]) -> Checked:
    line = request.get("line")
    if not _integer(line):
        return _error("The patch request needs an integer `line`."), {}
    error, reference = _validate_plane_reference(request.get("reference"))
    if error is not None:
        return error, {}
    arguments: dict[str, Any] = {"line": line, "reference": reference}
    x_axis = request.get("x_axis")
    if x_axis is not None:
        vector = _numbers(x_axis, 3)
        if vector is None or not any(abs(component) > 1e-9 for component in vector):
            return _error("`x_axis` must be three numbers and must not be zero."), {}
        arguments["x_axis"] = vector
    flip = request.get("flip", False)
    if not isinstance(flip, bool):
        return _error("The patch request needs `flip` as a boolean."), {}
    arguments["flip"] = flip
    offset = request.get("offset")
    if offset is not None:
        if not _number(offset):
            return _error("The patch request needs a numeric `offset`."), {}
        arguments["offset"] = float(offset)
    return None, arguments


def _validate_add_extrusion(request: dict[str, Any]) -> Checked:
    line = request.get("line")
    depth = request.get("depth", 0.5)
    if not _integer(line):
        return _error("The patch request needs an integer `line`."), {}
    if not _number(depth):
        return _error("The patch request needs a numeric `depth`."), {}
    return None, {"line": line, "depth": float(depth)}


def _validate_add_revolution(request: dict[str, Any]) -> Checked:
    line = request.get("line")
    offset = request.get("offset", 0.0)
    if not _integer(line):
        return _error("The patch request needs an integer `line`."), {}
    if not _number(offset):
        return _error("The patch request needs a numeric `offset`."), {}
    return None, {"line": line, "offset": float(offset)}


def _validate_add_loft(request: dict[str, Any]) -> Checked:
    line_a = request.get("line_a")
    line_b = request.get("line_b")
    height = request.get("height", 1.0)
    if not all(_integer(value) for value in (line_a, line_b)):
        return _error("The patch request needs integer `line_a` and `line_b`."), {}
    if not _number(height):
        return _error("The patch request needs a numeric `height`."), {}
    return None, {"line_a": line_a, "line_b": line_b, "height": float(height)}


def _validate_delete_object(request: dict[str, Any]) -> Checked:
    line = request.get("line")
    if not _integer(line):
        return _error("The patch request needs an integer `line`."), {}
    return None, {"line": line}


def _validate_set_value(request: dict[str, Any]) -> Checked:
    arguments: dict[str, Any] = {}
    for key in ("name", "argument"):
        value = request.get(key)
        if not isinstance(value, str):
            return _error(f"The patch request needs a string `{key}`."), {}
        arguments[key] = value
    line = request.get("line")
    if not _integer(line):
        return _error("The patch request needs an integer `line`."), {}
    arguments["line"] = line
    raw_value = request.get("value")
    scalar = float(raw_value) if _number(raw_value) else None
    vector = _numbers(raw_value)
    if scalar is None and vector is None:
        return _error("The patch request needs `value` as a number or numbers."), {}
    if arguments["argument"] in {"planeOrigin", "planeNormal"}:
        if vector is None or len(vector) != 3:
            return _error("A sketch-plane edit needs `value` as three numbers."), {}
        if arguments["argument"] == "planeNormal" and not any(
            abs(component) > 1e-9 for component in vector
        ):
            return _error("A sketch-plane normal must not be zero."), {}
    arguments["value"] = scalar if scalar is not None else vector
    return None, arguments


# ── Sketch vertices ─────────────────────────────────────────────────────────


def _validate_vertex(request: dict[str, Any]) -> Checked:
    """``line`` + ``index``: the vertex an edit acts on.

    Also the default contract — an operation this module has no entry for
    gets these two fields checked and nothing else.
    """
    arguments: dict[str, Any] = {}
    for key in ("line", "index"):
        value = request.get(key)
        if not _integer(value):
            return _error(f"The patch request needs an integer `{key}`."), {}
        arguments[key] = value
    return None, arguments


def _validate_placed_vertex(request: dict[str, Any]) -> Checked:
    """A vertex edit that also places the vertex somewhere (``xy``)."""
    error, arguments = _validate_vertex(request)
    if error is not None:
        return error, {}
    xy = _numbers(request.get("xy"), 2)
    if xy is None:
        return _error("The patch request needs `xy` as two numbers."), {}
    arguments["xy"] = (xy[0], xy[1])
    return None, arguments


# ── Constraints ─────────────────────────────────────────────────────────────

# Constraint kinds and how many sketch points each one takes.
_VALUED_CONSTRAINTS = {"fixed": 1, "distance": 2}
_RELATIONAL_CONSTRAINTS = {
    "horizontal": 2,
    "vertical": 2,
    "coincident": 2,
    "parallel": 4,
    "perpendicular": 4,
}


def _validate_add_constraint(request: dict[str, Any]) -> Checked:
    line = request.get("line")
    kind = request.get("kind")
    indices = request.get("indices")
    if not _integer(line):
        return _error("The patch request needs an integer `line`."), {}
    if kind not in _VALUED_CONSTRAINTS and kind not in _RELATIONAL_CONSTRAINTS:
        allowed = ", ".join(sorted({**_VALUED_CONSTRAINTS, **_RELATIONAL_CONSTRAINTS}))
        return _error(f"Constraint `kind` must be one of: {allowed}."), {}
    arity = _VALUED_CONSTRAINTS.get(kind) or _RELATIONAL_CONSTRAINTS[kind]
    if not (
        isinstance(indices, list)
        and len(indices) == arity
        and all(_integer(index) for index in indices)
    ):
        return _error(f"`{kind}` takes exactly {arity} integer `indices`."), {}
    value = None
    if kind in _VALUED_CONSTRAINTS:
        value = _scalar_or_numbers(request.get("value"))
        if value is None:
            return _error("The constraint needs a numeric `value`."), {}
    return None, {"line": line, "kind": kind, "indices": indices, "value": value}


def _validate_constraint_target(request: dict[str, Any]) -> Checked:
    """The constraint an edit acts on: its sketch ``line`` and its ``index``."""
    line = request.get("line")
    index = request.get("index")
    if not _integer(line):
        return _error("The patch request needs an integer `line`."), {}
    if not _integer(index) or index < 0:
        return _error("The patch request needs a non-negative `index`."), {}
    return None, {"line": line, "index": index}


def _validate_set_constraint_value(request: dict[str, Any]) -> Checked:
    error, arguments = _validate_constraint_target(request)
    if error is not None:
        return error, {}
    value = _scalar_or_numbers(request.get("value"))
    if value is None:
        return _error("The constraint needs a numeric `value`."), {}
    arguments["value"] = value
    return None, arguments


def _validate_solve_sketch(request: dict[str, Any]) -> Checked:
    line = request.get("line")
    if not _integer(line):
        return _error("The patch request needs an integer `line`."), {}
    method = request.get("method", "newton")
    iterations = request.get("iterations", 8)
    if not isinstance(method, str) or method not in {"newton", "adam", "sgd"}:
        return _error("Solver `method` must be `newton`, `adam`, or `sgd`."), {}
    if not _integer(iterations) or not 1 <= iterations <= 512:
        return _error("Solver `iterations` must be an integer from 1 to 512."), {}
    return None, {"line": line, "method": method, "iterations": iterations}


# ── Studies ─────────────────────────────────────────────────────────────────


def _validate_add_study(request: dict[str, Any]) -> Checked:
    kind = request.get("kind")
    if not isinstance(kind, str) or kind not in {"thermal", "elastic"}:
        return _error("Study `kind` must be `thermal` or `elastic`."), {}
    name = request.get("name")
    if name is not None and (not isinstance(name, str) or not name.strip()):
        return _error("Study `name` must be a non-empty string."), {}
    return None, {"kind": kind, "name": name}


def _validate_study_target(request: dict[str, Any]) -> Checked:
    """The study an edit acts on, named or indexed."""
    study = request.get("study")
    if not ((_integer(study) and study >= 0) or (isinstance(study, str) and study.strip())):
        return _error("The patch request needs `study` as a name or a non-negative index."), {}
    return None, {"study": study}


def _validate_add_study_bc(request: dict[str, Any]) -> Checked:
    error, arguments = _validate_study_target(request)
    if error is not None:
        return error, {}
    bc_type = request.get("bc_type")
    if not isinstance(bc_type, str) or bc_type not in {
        "dirichlet",
        "heat_flux",
        "fixed",
        "traction",
    }:
        return _error("`bc_type` must be one of: dirichlet, heat_flux, fixed, traction."), {}
    selection = request.get("selection")
    if not isinstance(selection, dict):
        return _error("The patch request needs `selection` as a description object."), {}
    arguments.update(bc_type=bc_type, selection=selection)
    raw_value = request.get("value")
    if bc_type == "fixed":
        if raw_value is not None:
            return _error("A `fixed` boundary condition takes no value."), {}
    elif bc_type == "traction":
        vector = _numbers(raw_value, 3)
        if vector is None:
            return _error("A `traction` boundary condition needs `value` as three numbers."), {}
        arguments["value"] = vector
    else:
        if not _number(raw_value):
            return _error(f"A `{bc_type}` boundary condition needs a numeric `value`."), {}
        arguments["value"] = float(raw_value)
    return None, arguments


def _validate_delete_study_bc(request: dict[str, Any]) -> Checked:
    error, arguments = _validate_study_target(request)
    if error is not None:
        return error, {}
    bc = request.get("bc")
    if not _integer(bc) or bc < 0:
        return _error("The patch request needs a non-negative `bc` index."), {}
    arguments["bc"] = bc
    return None, arguments


def _validate_set_study_value(request: dict[str, Any]) -> Checked:
    error, arguments = _validate_study_target(request)
    if error is not None:
        return error, {}
    bc = request.get("bc")
    argument = request.get("argument")
    if (bc is None) == (argument is None):
        return _error("The patch request needs exactly one of `bc` or `argument`."), {}
    if bc is not None:
        if not _integer(bc) or bc < 0:
            return _error("The patch request needs a non-negative `bc` index."), {}
        arguments["bc"] = bc
    else:
        if not isinstance(argument, str):
            return _error("The patch request needs a string `argument`."), {}
        arguments["argument"] = argument
    raw_value = request.get("value")
    if argument in {"mesh", "domain"}:
        # These reference declared objects by name, not numbers.
        if not isinstance(raw_value, str) or not raw_value.strip():
            return _error(f"The patch request needs `value` as a `{argument}` name."), {}
        arguments["value"] = raw_value
    else:
        value = _scalar_or_numbers(raw_value)
        if value is None:
            return _error("The patch request needs `value` as a number or numbers."), {}
        arguments["value"] = value
    return None, arguments


# ── Simulation meshes ───────────────────────────────────────────────────────


def _validate_add_mesh(request: dict[str, Any]) -> Checked:
    name = request.get("name")
    if name is not None and (not isinstance(name, str) or not name.strip()):
        return _error("Mesh `name` must be a non-empty string."), {}
    return None, {"name": name}


def _validate_mesh_target(request: dict[str, Any]) -> Checked:
    """The simulation mesh an edit acts on, named or indexed."""
    mesh = request.get("mesh")
    if not ((_integer(mesh) and mesh >= 0) or (isinstance(mesh, str) and mesh.strip())):
        return _error("The patch request needs `mesh` as a name or a non-negative index."), {}
    return None, {"mesh": mesh}


def _validate_set_mesh_value(request: dict[str, Any]) -> Checked:
    error, arguments = _validate_mesh_target(request)
    if error is not None:
        return error, {}
    argument = request.get("argument")
    if not isinstance(argument, str):
        return _error("The patch request needs a string `argument`."), {}
    arguments["argument"] = argument
    raw_value = request.get("value")
    if argument == "domain":
        if not isinstance(raw_value, str) or not raw_value.strip():
            return _error("The patch request needs `value` as a `domain` name."), {}
        arguments["value"] = raw_value
    elif argument == "method":
        if not isinstance(raw_value, str) or raw_value not in {"hex", "tet4", "tet10"}:
            return _error("Mesh `method` must be one of: hex, tet4, tet10."), {}
        arguments["value"] = raw_value
    else:
        value = _scalar_or_numbers(raw_value)
        if value is None:
            return _error("The patch request needs `value` as a number or numbers."), {}
        arguments["value"] = value
    return None, arguments


# ── Optimizations ───────────────────────────────────────────────────────────


def _validate_optimization_target(request: dict[str, Any]) -> Checked:
    """The optimization an edit acts on, named or indexed."""
    optimization = request.get("optimization")
    valid_index = _integer(optimization) and optimization >= 0
    if not (valid_index or (isinstance(optimization, str) and optimization.strip())):
        return _error(
            "The patch request needs `optimization` as a name or a non-negative index."
        ), {}
    return None, {"optimization": optimization}


def _validate_set_optimization_value(request: dict[str, Any]) -> Checked:
    error, arguments = _validate_optimization_target(request)
    if error is not None:
        return error, {}
    argument = request.get("argument")
    if not isinstance(argument, str) or argument not in {"steps", "learning_rate"}:
        return _error("Optimization `argument` must be `steps` or `learning_rate`."), {}
    raw_value = request.get("value")
    if not _number(raw_value):
        return _error("The patch request needs a numeric `value`."), {}
    arguments.update(argument=argument, value=raw_value)
    return None, arguments


# One entry per operation in ``cadjoint.viewer._patch.OPERATIONS``: this table
# is the whole contract ``/patch`` requests must satisfy.  Operations that
# share a shape (the four study edits all name their study the same way)
# share the helper that checks it, one validator deep.
PATCH_VALIDATORS: dict[str, Validator] = {
    "set_vertex": _validate_placed_vertex,
    "insert_vertex": _validate_placed_vertex,
    "delete_vertex": _validate_vertex,
    "set_value": _validate_set_value,
    "add_primitive": _validate_add_primitive,
    "add_material": _validate_add_material,
    "assign_material": _validate_assign_material,
    "set_material_property": _validate_set_material_property,
    "add_sketch": _validate_add_sketch,
    "set_sketch_plane": _validate_set_sketch_plane,
    "add_extrusion": _validate_add_extrusion,
    "add_revolution": _validate_add_revolution,
    "add_loft": _validate_add_loft,
    "add_constraint": _validate_add_constraint,
    "delete_constraint": _validate_constraint_target,
    "set_constraint_value": _validate_set_constraint_value,
    "solve_sketch": _validate_solve_sketch,
    "delete_object": _validate_delete_object,
    "add_study": _validate_add_study,
    "delete_study": _validate_study_target,
    "add_study_bc": _validate_add_study_bc,
    "delete_study_bc": _validate_delete_study_bc,
    "set_study_value": _validate_set_study_value,
    "add_mesh": _validate_add_mesh,
    "delete_mesh": _validate_mesh_target,
    "set_mesh_value": _validate_set_mesh_value,
    "delete_optimization": _validate_optimization_target,
    "set_optimization_value": _validate_set_optimization_value,
}


def patch_source(request: dict[str, Any]) -> dict[str, Any]:
    """Apply one viewer edit to the user's program text.

    Every operation addresses its target either by the stable ``id`` the last
    compile published for it, or by the ``line``/``index`` that payload
    reported.  The id is the durable one: it is resolved against the text in
    this very request, so an edit made in the editor since the compile cannot
    send the patch to the wrong statement.

    Args:
        request: ``{"source", "op"}`` plus the operation's own fields — an
            ``"id"`` naming the target, or the legacy ``"line"``/``"index"``,
            and e.g. ``"xy"`` for the operations that place a vertex.

    Returns:
        ``{"ok": True, "source": ...}`` or ``{"ok": False, "error": ...}``.
    """
    source = request.get("source")
    operation = request.get("op")
    if not isinstance(source, str):
        return _error("The patch request must contain a string `source` field.")
    if exceeds_source_limit(source):
        return _error(OVERSIZED_SOURCE_ERROR)
    if not isinstance(operation, str):
        return _error("The patch request must contain a string `op` field.")
    if operation not in OPERATIONS:
        # Reject up front: otherwise an operation this server does not know
        # falls through to the vertex-edit checks and complains about a missing
        # `line`, which points nowhere near the real problem — usually a browser
        # running newer assets than the server process.
        return _error(
            f"This server does not support the patch operation {operation!r}. "
            "If you updated cadjoint, restart the playground server."
        )

    error, request = _resolved_request(request, source, operation)
    if error is not None:
        return error

    error, arguments = PATCH_VALIDATORS.get(operation, _validate_vertex)(request)
    if error is not None:
        return error
    try:
        return {"ok": True, "source": apply_operation(source, operation, **arguments)}
    except PatchError as failure:
        return _error(str(failure))
