"""Span surgery: the primitives every patch operation edits source with.

An operation's job is to decide *what* to change; this module is *how* the
change is made to the text.  Everything here works on absolute character
offsets so an edit touches only the characters it must — formatting, comments
and the rest of the file stay byte for byte identical.

Two invariants are worth knowing before adding to this module:

- **Line numbers must not shift.** Patch requests address their targets by line
  number, so :func:`_ensure_import` extends an existing ``from x import y``
  line in place, or inserts beside the edited statement, rather than pushing a
  new line in above the target.
- **Every result goes through :func:`_validate`,** so a patch can never emit a
  file that no longer parses.

Add a helper here when it manipulates a span, an import, or a call argument in
a way that does not depend on which cadjoint concept is being edited.  Anything
that knows about ``scene``, sketches, studies or meshes belongs one layer up.
"""

from __future__ import annotations

import ast

from cadjoint.viewer.patch.errors import PatchError
from cadjoint.viewer.source_map.nodes import _editable_value_node, _line_offsets, _node_span


def _validate(source: str) -> str:
    """Guard against emitting a file that no longer parses."""
    try:
        ast.parse(source)
    except SyntaxError as error:  # pragma: no cover - defensive
        raise PatchError(f"Patched source is not valid Python: {error}") from error
    return source


def _module_names(tree: ast.Module) -> set[str]:
    """Every name bound at module level, for choosing a fresh variable."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.alias):
            names.add(node.asname or node.name.split(".")[0])
    return names


def _ensure_import(
    source: str,
    tree: ast.Module,
    module: str,
    symbol: str,
    *,
    prefer_offset: int | None = None,
) -> str:
    """Add ``from module import symbol`` when the symbol is not already bound.

    An existing single-line ``from module import ...`` is extended in place
    rather than adding a new line: line-addressed patch operations identify
    their targets by line number, so imports must not shift the file.  When
    no such line exists and ``prefer_offset`` is given, the new import is
    inserted there (module-level imports are legal anywhere) so callers can
    keep everything above their target line untouched.
    """
    if symbol in _module_names(tree):
        return source
    offsets = _line_offsets(source)
    for node in tree.body:
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == module
            and node.level == 0
            and node.lineno == node.end_lineno
            and not any(alias.asname for alias in node.names)
        ):
            _, end = _node_span(source, offsets, node)
            return source[:end] + f", {symbol}" + source[end:]
    if prefer_offset is not None:
        return source[:prefer_offset] + f"from {module} import {symbol}\n" + source[prefer_offset:]
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    insert_line = (imports[-1].end_lineno if imports else 0) or 0
    offset = offsets[insert_line] if insert_line < len(offsets) else len(source)
    return source[:offset] + f"from {module} import {symbol}\n" + source[offset:]


def _after_statement(source: str, statement: ast.stmt) -> int:
    offsets = _line_offsets(source)
    return offsets[min(statement.end_lineno or statement.lineno, len(offsets) - 1)]


def _set_keyword_expression(source: str, call: ast.Call, keyword: str, expression: str) -> str:
    """Replace or append a call keyword with a trusted Python expression."""
    offsets = _line_offsets(source)
    existing = next((item for item in call.keywords if item.arg == keyword), None)
    if existing is not None:
        span = _node_span(source, offsets, existing.value)
        if span is None:  # pragma: no cover - defensive
            raise PatchError(f"Could not locate `{keyword}` in the source.")
        start, end = span
        return _validate(source[:start] + expression + source[end:])

    raw_spans = [
        span
        for argument in [*call.args, *(item.value for item in call.keywords)]
        if (span := _node_span(source, offsets, argument)) is not None
    ]
    if not raw_spans:
        raise PatchError("Cannot add a material to a call with no arguments.")
    insert = max(end for _, end in raw_spans)
    return _validate(source[:insert] + f", {keyword}={expression}" + source[insert:])


def _rewrite_call_argument(
    source: str, call: ast.Call, fields: tuple[str, ...], argument: str, expression: str, noun: str
) -> str:
    """Rewrite one call argument in place (or append it as a keyword).

    The argument is found as a keyword, or positionally through its slot in
    *fields*; a name bound to a literal follows the assignment indirection
    of :func:`_editable_value_node`.
    """
    target = next(
        (keyword.value for keyword in call.keywords if keyword.arg == argument),
        None,
    )
    if target is None and argument in fields:
        position = fields.index(argument)
        if position < len(call.args):
            target = call.args[position]
    if target is None:
        return _set_keyword_expression(source, call, argument, expression)
    tree = ast.parse(source)
    literal = _editable_value_node(target, tree)
    if literal is None:
        raise PatchError(f"The {noun}'s `{argument}` value is not an editable literal.")
    offsets = _line_offsets(source)
    start, end = _node_span(source, offsets, literal)
    return _validate(source[:start] + expression + source[end:])


def _statement_containing(tree: ast.Module, node: ast.AST) -> ast.stmt | None:
    """The module-level statement whose subtree holds *node*."""
    for statement in tree.body:
        if any(candidate is node for candidate in ast.walk(statement)):
            return statement
    return None


def _name_references(tree: ast.Module, name: str, exclude: ast.AST) -> list[ast.Name]:
    """Every load of *name* outside the statement that defines it."""
    return [
        node
        for statement in tree.body
        if statement is not exclude
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load)
    ]


def _argument_span(source: str, offsets, node) -> tuple[int, int]:
    """Span of one call argument, including the comma that follows it."""
    span = _node_span(source, offsets, node)
    if span is None:  # pragma: no cover - defensive
        raise PatchError("Could not locate the argument to remove.")
    start, end = span
    while end < len(source) and source[end] in ", ":
        end += 1
    return start, end


def _delete_statement(source: str, statement: ast.stmt) -> str:
    """Remove a whole statement's lines from *source*.

    The slice runs from the statement's first line to the start of the line
    after its last, so the following statement keeps its own line.
    """
    offsets = _line_offsets(source)
    start = offsets[statement.lineno - 1]
    end = offsets[min(statement.end_lineno or statement.lineno, len(offsets) - 1)]
    return source[:start] + source[end:]
