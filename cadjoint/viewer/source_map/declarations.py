"""Locate the top-level simulation declarations: studies, meshes, optimizations.

These three locators share one contract.  Only *unambiguous top-level*
statements — an assignment or a bare expression containing exactly one
constructor call — are located, and their position in source order doubles as
a stable index that the compile payload and the patch operations both address.
Declarations built in loops or helper functions are deliberately not located:
the compile payload notices the count mismatch against the captured objects and
marks every entry non-editable rather than editing the wrong line.

Add a locator here when a new simulation concept becomes a top-level
declaration the viewer edits by index or name.  Each one returns a frozen
statement dataclass carrying the spans of the keyword values that stay
editable, so the patch layer never has to re-walk the tree.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from cadjoint.viewer.source_map.nodes import (
    Span,
    _called_name,
    _line_offsets,
    _node_span,
    _resolved_container,
    parse_module,
)

STUDY_CALL_KINDS = {"ThermalStudy": "thermal", "ElasticStudy": "elastic"}
"""Study constructor names mapped to their viewer payload ``kind``."""


@dataclass(frozen=True)
class StudyStatement:
    """One top-level study constructor statement, in source order.

    Statement order equals construction order for top-level declarations, so
    the position doubles as a stable index: the compile payload and the study
    patch operations agree on which statement an index refers to.  ``bcs`` is
    the resolved literal boundary-condition list (None when the argument is
    absent or not a literal container), and ``bc_spans`` are the character
    spans of its elements so single conditions can be edited or deleted.
    ``mesh_span`` and ``domain_span`` are the character spans of the
    ``mesh=`` / ``domain=`` keyword *values* (None when absent) so those
    references stay editable from the viewer.
    """

    index: int
    kind: str
    name: str | None
    variable: str | None
    statement: ast.stmt
    call: ast.Call
    call_span: Span
    bcs: ast.List | ast.Tuple | None
    bcs_span: Span | None
    bc_spans: tuple[Span, ...]
    mesh_span: Span | None
    domain_span: Span | None


def locate_study_statements(source: str) -> list[StudyStatement] | None:
    """Top-level ``ThermalStudy``/``ElasticStudy`` constructors, in source order.

    Mirrors :func:`locate_constraint_statements`: only unambiguous top-level
    statements (an assignment or a bare expression containing exactly one
    study constructor) are located.  Studies built in loops or functions are
    not — the compile payload detects the count mismatch against the captured
    studies and marks every entry non-editable.

    Args:
        source: The full program text.

    Returns:
        The study statements in source order, or None when the source cannot
        be parsed.
    """
    try:
        tree = parse_module(source)
    except SyntaxError:
        return None
    offsets = _line_offsets(source)
    statements: list[StudyStatement] = []
    for item in tree.body:
        if not isinstance(item, (ast.Assign, ast.AnnAssign, ast.Expr)):
            continue
        calls = [node for node in ast.walk(item) if _called_name(node) in STUDY_CALL_KINDS]
        if len(calls) != 1:
            continue
        call = calls[0]
        call_span = _node_span(source, offsets, call)
        if call_span is None:
            continue
        name = next(
            (
                keyword.value.value
                for keyword in call.keywords
                if keyword.arg == "name"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            ),
            None,
        )
        variable = None
        if (
            isinstance(item, ast.Assign)
            and len(item.targets) == 1
            and isinstance(item.targets[0], ast.Name)
        ):
            variable = item.targets[0].id
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            variable = item.target.id
        bcs_argument = next(
            (keyword.value for keyword in call.keywords if keyword.arg == "bcs"),
            None,
        )
        bcs = _resolved_container(bcs_argument, tree) if bcs_argument is not None else None
        bcs_span = _node_span(source, offsets, bcs) if bcs is not None else None
        bc_spans: list[Span] = []
        for element in bcs.elts if bcs is not None else []:
            span = _node_span(source, offsets, element)
            if span is None:  # pragma: no cover - constructors always carry spans
                bcs = bcs_span = None
                bc_spans = []
                break
            bc_spans.append(span)

        def keyword_span(argument: str, keywords=call.keywords) -> Span | None:
            value = next(
                (keyword.value for keyword in keywords if keyword.arg == argument),
                None,
            )
            return _node_span(source, offsets, value) if value is not None else None

        statements.append(
            StudyStatement(
                index=len(statements),
                kind=STUDY_CALL_KINDS[_called_name(call) or ""],
                name=name,
                variable=variable,
                statement=item,
                call=call,
                call_span=call_span,
                bcs=bcs,
                bcs_span=bcs_span,
                bc_spans=tuple(bc_spans),
                mesh_span=keyword_span("mesh"),
                domain_span=keyword_span("domain"),
            )
        )
    return statements


MESH_CALL_NAME = "SimMesh"


@dataclass(frozen=True)
class MeshStatement:
    """One top-level ``SimMesh`` constructor statement, in source order.

    Mirrors :class:`StudyStatement`: statement order equals construction
    order for top-level declarations, so the position doubles as a stable
    index shared by the compile payload and the mesh patch operations.
    ``name`` is the literal ``name=`` argument (or first positional string),
    ``variable`` the assignment target when there is exactly one.
    """

    index: int
    name: str | None
    variable: str | None
    statement: ast.stmt
    call: ast.Call
    call_span: Span


def locate_mesh_statements(source: str) -> list[MeshStatement] | None:
    """Top-level ``SimMesh`` constructors, in source order.

    Mirrors :func:`locate_study_statements`: only unambiguous top-level
    statements (an assignment or a bare expression containing exactly one
    ``SimMesh`` constructor) are located.  Meshes built in loops or helper
    functions are not — the compile payload detects the count mismatch
    against the captured meshes and marks every entry non-editable.

    Args:
        source: The full program text.

    Returns:
        The mesh statements in source order, or None when the source cannot
        be parsed.
    """
    try:
        tree = parse_module(source)
    except SyntaxError:
        return None
    offsets = _line_offsets(source)
    statements: list[MeshStatement] = []
    for item in tree.body:
        if not isinstance(item, (ast.Assign, ast.AnnAssign, ast.Expr)):
            continue
        calls = [node for node in ast.walk(item) if _called_name(node) == MESH_CALL_NAME]
        if len(calls) != 1:
            continue
        call = calls[0]
        call_span = _node_span(source, offsets, call)
        if call_span is None:
            continue
        name_node = next(
            (keyword.value for keyword in call.keywords if keyword.arg == "name"),
            call.args[0] if call.args else None,
        )
        name = (
            name_node.value
            if isinstance(name_node, ast.Constant) and isinstance(name_node.value, str)
            else None
        )
        variable = None
        if (
            isinstance(item, ast.Assign)
            and len(item.targets) == 1
            and isinstance(item.targets[0], ast.Name)
        ):
            variable = item.targets[0].id
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            variable = item.target.id
        statements.append(
            MeshStatement(
                index=len(statements),
                name=name,
                variable=variable,
                statement=item,
                call=call,
                call_span=call_span,
            )
        )
    return statements


OPTIMIZATION_CALL_NAME = "Optimization"


@dataclass(frozen=True)
class OptimizationStatement:
    """One top-level ``Optimization`` constructor statement, in source order.

    Mirrors :class:`MeshStatement`: statement order equals construction
    order for top-level declarations, so the position doubles as a stable
    index shared by the compile payload and the optimization patch
    operations.  ``name`` is the literal ``name=`` argument (or first
    positional string), ``variable`` the assignment target when there is
    exactly one.  ``steps_span`` and ``learning_rate_span`` are the
    character spans of those keyword *values* (None when absent) so the
    viewer can edit them in place.
    """

    index: int
    name: str | None
    variable: str | None
    statement: ast.stmt
    call: ast.Call
    call_span: Span
    steps_span: Span | None
    learning_rate_span: Span | None


def locate_optimization_statements(source: str) -> list[OptimizationStatement] | None:
    """Top-level ``Optimization`` constructors, in source order.

    Mirrors :func:`locate_mesh_statements`: only unambiguous top-level
    statements (an assignment or a bare expression containing exactly one
    ``Optimization`` constructor) are located.  Optimizations built in
    loops or helper functions are not — the compile payload detects the
    count mismatch against the captured optimizations and marks every
    entry non-editable.

    Args:
        source: The full program text.

    Returns:
        The optimization statements in source order, or None when the
        source cannot be parsed.
    """
    try:
        tree = parse_module(source)
    except SyntaxError:
        return None
    offsets = _line_offsets(source)
    statements: list[OptimizationStatement] = []
    for item in tree.body:
        if not isinstance(item, (ast.Assign, ast.AnnAssign, ast.Expr)):
            continue
        calls = [node for node in ast.walk(item) if _called_name(node) == OPTIMIZATION_CALL_NAME]
        if len(calls) != 1:
            continue
        call = calls[0]
        call_span = _node_span(source, offsets, call)
        if call_span is None:
            continue
        name_node = next(
            (keyword.value for keyword in call.keywords if keyword.arg == "name"),
            call.args[0] if call.args else None,
        )
        name = (
            name_node.value
            if isinstance(name_node, ast.Constant) and isinstance(name_node.value, str)
            else None
        )
        variable = None
        if (
            isinstance(item, ast.Assign)
            and len(item.targets) == 1
            and isinstance(item.targets[0], ast.Name)
        ):
            variable = item.targets[0].id
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            variable = item.target.id

        def keyword_span(argument: str, keywords=call.keywords) -> Span | None:
            value = next(
                (keyword.value for keyword in keywords if keyword.arg == argument),
                None,
            )
            return _node_span(source, offsets, value) if value is not None else None

        statements.append(
            OptimizationStatement(
                index=len(statements),
                name=name,
                variable=variable,
                statement=item,
                call=call,
                call_span=call_span,
                steps_span=keyword_span("steps"),
                learning_rate_span=keyword_span("learning_rate"),
            )
        )
    return statements
