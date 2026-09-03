"""Stable identities for everything the viewer can address in the source.

Line numbers are a *position*, not an identity.  A payload built by one
compile carries the lines the statements sat on at that moment, and any
edit in the editor — a blank line, an import, an ``add_sketch`` that
inserts three lines above the profile — moves them silently.  Every patch
request that still carries the old line then addresses the wrong
statement, or nothing at all.

This module gives each addressable thing a name derived from the *AST
path* and the name the program assigned it, so it survives every edit that
does not touch the statement itself:

``assign:comb_profile``
    The call that is the value of the module-level assignment
    ``comb_profile = PolygonProfile(...)``.  Profiles, primitives,
    features, materials, studies, meshes and optimizations all take this
    form when they are bound to a plain name — which is the normal case,
    and the one the viewer's own edits always produce.

``call:extrude@comb_profile``
    A feature call that is *not* bound to a name, keyed by the sketch it
    consumes.  ``sketch:comb``, ``box:block`` and friends do the same for
    an unbound construction call carrying a literal ``name=``.

``bc:sink-conduction[1]``
    An ordinal inside its owner: boundary conditions inside a study,
    constraints and vertices inside a sketch.  The owner is named by its
    *token* — the variable it is bound to, else its literal ``name=``,
    else ``#<n>`` — and the bracket is the same ordinal the payload
    serializes, so a chip's identity and the statement it deletes agree.

What is *not* promised: an id containing ``#<n>`` is an ordinal among
anonymous objects of the same kind, so inserting another anonymous object
before it does move it.  Naming an object — which every viewer-authored
statement does — takes it out of that class.

Everything here is static: it reads the program text and never runs it, so
``/patch`` can resolve an id without spawning the compile worker.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache

from cadjoint.viewer.source_map.calls import locate_profile_call
from cadjoint.viewer.source_map.constraints import locate_constraint_statements
from cadjoint.viewer.source_map.declarations import (
    locate_mesh_statements,
    locate_optimization_statements,
    locate_study_statements,
)
from cadjoint.viewer.source_map.features import (
    FEATURE_CALL_KINDS,
    PRIMITIVE_CALL_KINDS,
)
from cadjoint.viewer.source_map.nodes import (
    _called_name,
    _is_construction_call,
    _is_profile_call,
    parse_module,
)

#: Every ``kind`` an :class:`Identity` can carry, and what the id addresses.
IDENTITY_KINDS = frozenset(
    {
        "sketch",
        "primitive",
        "feature",
        "material",
        "study",
        "mesh",
        "optimization",
        "constraint",
        "bc",
        "vertex",
        "plane",
    }
)


@dataclass(frozen=True)
class Identity:
    """One addressable thing, named independently of where it sits.

    Attributes:
        id: The stable identity, e.g. ``assign:comb_profile``.
        kind: One of :data:`IDENTITY_KINDS`.
        token: The short name this thing is known by inside its owner's ids
            — the bound variable, the literal ``name=``, or ``#<n>``.
        call: The constructor or generator called, e.g. ``PolygonProfile``,
            ``extrude``, ``cylinder``; None for things that are not a call
            of their own (a vertex, a plane, a boundary condition).
        line: 1-based line the id resolved to *in this text* — a derived
            hint, never the identity.
        index: The ordinal a patch request addresses this thing by, when it
            has one: a declaration's source-order index, a constraint's or
            boundary condition's position in its owner.
        owner: Id of the thing this one lives inside, or None.
        name: The literal ``name=`` argument, when the call carries one.
        variable: The module-level variable the call is bound to, or None.
    """

    id: str
    kind: str
    token: str
    call: str | None
    line: int | None
    index: int | None = None
    owner: str | None = None
    name: str | None = None
    variable: str | None = None


def _literal_name(call: ast.Call) -> str | None:
    """The literal ``name="..."`` a constructor call carries, if any."""
    for keyword in call.keywords:
        if (
            keyword.arg == "name"
            and isinstance(keyword.value, ast.Constant)
            and isinstance(keyword.value.value, str)
            and keyword.value.value.strip()
        ):
            return keyword.value.value
    return None


def _bound_variable(statement: ast.stmt) -> str | None:
    """The plain name a module-level statement binds, or None."""
    if (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    ):
        return statement.targets[0].id
    if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
        return statement.target.id
    return None


def _token(variable: str | None, name: str | None, ordinal: int) -> str:
    """The short name an owner is known by inside the ids of its children."""
    if variable is not None:
        return variable
    if name is not None:
        return name
    return f"#{ordinal}"


def _identify(
    *,
    variable: str | None,
    name: str | None,
    ordinal: int,
    fallback_prefix: str,
) -> tuple[str, str]:
    """``(id, token)`` for one construction call.

    A call bound to a module-level variable is ``assign:<variable>`` — the
    AST path plus the assigned name, which is what makes it survive edits
    elsewhere in the file.  Everything else falls back to a kind-specific
    form that is still independent of the line it sits on.
    """
    token = _token(variable, name, ordinal)
    if variable is not None:
        return f"assign:{variable}", token
    return f"{fallback_prefix}:{token}", token


def _statements_by_node(tree: ast.Module) -> dict[int, ast.stmt]:
    """Every node in the module, mapped to the top-level statement holding it.

    Built in one walk rather than re-walking the tree once per call: the
    identity table asks this question for every construction call in the
    program, and on the starter scene that was the bulk of its cost.
    """
    owners: dict[int, ast.stmt] = {}
    for statement in tree.body:
        for node in ast.walk(statement):
            owners[id(node)] = statement
    return owners


def _construction_identities(tree: ast.Module) -> list[Identity]:
    """Sketches, primitives, and the features generated from them."""
    identities: list[Identity] = []
    profile_tokens: dict[int, str] = {}
    owners = _statements_by_node(tree)

    def binding(call: ast.Call) -> str | None:
        """The variable *call* is bound to, when it is the assigned value."""
        statement = owners.get(id(call))
        if statement is None or getattr(statement, "value", None) is not call:
            return None
        return _bound_variable(statement)

    sketches = [node for node in ast.walk(tree) if _is_profile_call(node)]
    sketches.sort(key=lambda node: (node.lineno, node.col_offset))
    for ordinal, call in enumerate(sketches):
        variable = binding(call)
        name = _literal_name(call)
        identifier, token = _identify(
            variable=variable, name=name, ordinal=ordinal, fallback_prefix="sketch"
        )
        profile_tokens[id(call)] = token
        identities.append(
            Identity(
                id=identifier,
                kind="sketch",
                token=token,
                call="PolygonProfile",
                line=call.lineno,
                index=ordinal,
                name=name,
                variable=variable,
            )
        )

    primitives = [
        node
        for node in ast.walk(tree)
        if _called_name(node) in PRIMITIVE_CALL_KINDS and _is_construction_call(node)
    ]
    primitives.sort(key=lambda node: (node.lineno, node.col_offset))
    for ordinal, call in enumerate(primitives):
        variable = binding(call)
        name = _literal_name(call)
        kind_name = _called_name(call) or ""
        identifier, token = _identify(
            variable=variable, name=name, ordinal=ordinal, fallback_prefix=kind_name
        )
        identities.append(
            Identity(
                id=identifier,
                kind="primitive",
                token=token,
                call=kind_name,
                line=call.lineno,
                index=ordinal,
                name=name,
                variable=variable,
            )
        )

    # A feature keyed by the sketch it consumes needs the sketch's token, and
    # a sketch consumed by two features needs the pair disambiguated.
    variables = {
        profile_tokens[id(profile)] for profile in sketches if binding(profile) is not None
    }
    features = [
        node
        for node in ast.walk(tree)
        if _called_name(node) in FEATURE_CALL_KINDS and _is_construction_call(node)
    ]
    features.sort(key=lambda node: (node.lineno, node.col_offset))
    consumed: dict[tuple[str, str], int] = {}
    for ordinal, call in enumerate(features):
        variable = binding(call)
        kind_name = _called_name(call) or ""
        source_token = next(
            (
                argument.id
                for argument in call.args
                if isinstance(argument, ast.Name) and argument.id in variables
            ),
            None,
        )
        if source_token is not None:
            seen = consumed.get((kind_name, source_token), 0)
            consumed[(kind_name, source_token)] = seen + 1
            source_token = source_token if seen == 0 else f"{source_token}[{seen}]"
        token = _token(variable, _literal_name(call), ordinal)
        if variable is not None:
            identifier = f"assign:{variable}"
        elif source_token is not None:
            # ``call:extrude@comb_profile`` — the sketch it consumes names it.
            identifier = f"call:{kind_name}@{source_token}"
        else:
            identifier = f"call:{kind_name}:{token}"
        identities.append(
            Identity(
                id=identifier,
                kind="feature",
                token=token,
                call=kind_name,
                line=call.lineno,
                index=ordinal,
                name=_literal_name(call),
                variable=variable,
            )
        )
    return identities


def _material_identities(tree: ast.Module) -> list[Identity]:
    """Named ``Material`` definitions — always a module-level assignment."""
    identities: list[Identity] = []
    ordinal = 0
    for statement in tree.body:
        variable = _bound_variable(statement)
        value = getattr(statement, "value", None)
        if variable is None or not isinstance(value, ast.Call):
            continue
        if _called_name(value) != "Material":
            continue
        identities.append(
            Identity(
                id=f"assign:{variable}",
                kind="material",
                token=variable,
                call="Material",
                line=value.lineno,
                index=ordinal,
                name=_literal_name(value),
                variable=variable,
            )
        )
        ordinal += 1
    return identities


def _declaration_identities(source: str) -> list[Identity]:
    """Studies, simulation meshes, optimizations, and the BCs inside studies."""
    identities: list[Identity] = []
    for kind, locate, call_name in (
        ("study", locate_study_statements, None),
        ("mesh", locate_mesh_statements, "SimMesh"),
        ("optimization", locate_optimization_statements, "Optimization"),
    ):
        for statement in locate(source) or []:
            identifier, token = _identify(
                variable=statement.variable,
                name=statement.name,
                ordinal=statement.index,
                fallback_prefix=kind,
            )
            identities.append(
                Identity(
                    id=identifier,
                    kind=kind,
                    token=token,
                    call=call_name or _called_name(statement.call),
                    line=statement.statement.lineno,
                    index=statement.index,
                    name=statement.name,
                    variable=statement.variable,
                )
            )
            if kind != "study" or statement.bcs is None:
                continue
            for position, element in enumerate(statement.bcs.elts):
                identities.append(
                    Identity(
                        id=f"bc:{token}[{position}]",
                        kind="bc",
                        token=f"{token}[{position}]",
                        call=_called_name(element),
                        line=element.lineno,
                        index=position,
                        owner=identifier,
                    )
                )
    return identities


def _sketch_child_identities(source: str, sketches: list[Identity]) -> list[Identity]:
    """A sketch's own plane, its vertices, and its constraint statements."""
    identities: list[Identity] = []
    for sketch in sketches:
        identities.append(
            Identity(
                id=f"plane:{sketch.token}",
                kind="plane",
                token=sketch.token,
                call=None,
                line=sketch.line,
                owner=sketch.id,
            )
        )
        call = locate_profile_call(source, sketch.line) if sketch.line is not None else None
        for position in range(len(call.element_spans) if call is not None else 0):
            identities.append(
                Identity(
                    id=f"vertex:{sketch.token}[{position}]",
                    kind="vertex",
                    token=f"{sketch.token}[{position}]",
                    call=None,
                    line=sketch.line,
                    index=position,
                    owner=sketch.id,
                )
            )
        statements = (
            locate_constraint_statements(source, sketch.line) if sketch.line is not None else None
        )
        for position, statement in enumerate(statements or []):
            identities.append(
                Identity(
                    id=f"constraint:{sketch.token}[{position}]",
                    kind="constraint",
                    token=f"{sketch.token}[{position}]",
                    call=_called_name(statement.call),
                    line=statement.statement.lineno,
                    index=position,
                    owner=sketch.id,
                )
            )
    return identities


@lru_cache(maxsize=8)
def _identities(source: str) -> tuple[Identity, ...]:
    """The identity table for one text, computed once per text.

    One compile asks for it many times over — once for the construction
    payload, once per declaration entry — and one ``/patch`` request asks
    once. Caching by text is exact: any edit produces a different string
    and so a different entry.
    """
    try:
        tree = parse_module(source)
    except SyntaxError:
        return ()
    identities = _construction_identities(tree)
    sketches = [item for item in identities if item.kind == "sketch"]
    identities.extend(_material_identities(tree))
    identities.extend(_declaration_identities(source))
    identities.extend(_sketch_child_identities(source, sketches))
    return tuple(identities)


def build_identities(source: str) -> list[Identity]:
    """Every stable identity the program text declares, in a fixed order.

    Args:
        source: The full program text.

    Returns:
        One :class:`Identity` per addressable thing. Empty when the source
        cannot be parsed — an id can only name a statement that exists.
    """
    return list(_identities(source))


def describe_identities(source: str) -> list[dict]:
    """The whole identity table, JSON-ready, for the compile payload.

    Publishing the table — not just an id per entry — lets the viewer name
    anything the payload mentions only by line (an operator chip, a face's
    owner) without the payload having to grow a field in every pinned shape.

    Args:
        source: The full program text.

    Returns:
        One dict per identity: ``id``, ``kind``, ``token``, ``call``,
        ``line``, ``index``, ``owner``, ``name``, ``variable``.
    """
    return [
        {
            "id": identity.id,
            "kind": identity.kind,
            "token": identity.token,
            "call": identity.call,
            "line": identity.line,
            "index": identity.index,
            "owner": identity.owner,
            "name": identity.name,
            "variable": identity.variable,
        }
        for identity in build_identities(source)
    ]


@lru_cache(maxsize=8)
def _identity_index(source: str) -> dict[str, Identity]:
    """Map every stable id in *source* to what it names.

    A duplicate id — two module-level assignments to the same variable —
    keeps neither entry, because resolving it would be a guess. That
    mirrors the resolver layer's rule that ambiguity is refused.

    Args:
        source: The full program text.

    Returns:
        ``{id: Identity}`` for every unambiguous identity.
    """
    index: dict[str, Identity] = {}
    duplicates: set[str] = set()
    for identity in _identities(source):
        if identity.id in index:
            duplicates.add(identity.id)
            continue
        index[identity.id] = identity
    for identifier in duplicates:
        index.pop(identifier, None)
    return index


def identity_index(source: str) -> dict[str, Identity]:
    """Map every stable id in *source* to what it names.

    Args:
        source: The full program text.

    Returns:
        A fresh ``{id: Identity}`` the caller owns; the table behind it is
        cached, so this costs a dict copy on a repeated text.
    """
    return dict(_identity_index(source))


def identity_for(source: str, identifier: str) -> Identity | None:
    """The one thing *identifier* names in *source*, or None."""
    return identity_index(source).get(identifier)


def identity_at(
    identities: list[Identity], line: int | None, kinds: frozenset[str] | set[str]
) -> Identity | None:
    """The identity of one of *kinds* sitting at *line*, or None if ambiguous.

    Used the other way round from :func:`identity_for`: the payload knows the
    line an object was built on and wants the stable id to publish with it.

    Args:
        identities: The result of :func:`build_identities` for this source.
        line: 1-based line the object was constructed on.
        kinds: Which identity kinds may answer.

    Returns:
        The single matching identity, or None when the line holds none or
        several.
    """
    if line is None:
        return None
    matches = [item for item in identities if item.kind in kinds and item.line == line]
    return matches[0] if len(matches) == 1 else None
