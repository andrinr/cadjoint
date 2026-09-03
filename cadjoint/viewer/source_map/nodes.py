"""AST primitives shared by every locator and payload builder.

This is the leaf of the source-map package: character-span arithmetic over
``ast`` nodes plus the conservative name-following rules that decide whether a
value is *editable* (a literal the viewer may rewrite) or must be left alone.

Put a helper here when it answers a question about a bare AST node and needs
no knowledge of the calls cadjoint's DSL happens to define — spans, call
names, numeric literals, and resolving a name or a ``Scalar(value=...)``
wrapper down to the literal underneath.  Anything that knows about
``PolygonProfile``, ``SimMesh``, materials, or the viewer payload belongs in
one of the sibling modules instead.

Nothing here imports from the rest of the package, so it can never take part
in an import cycle.
"""

from __future__ import annotations

import ast
from functools import lru_cache

Span = tuple[int, int]


@lru_cache(maxsize=8)
def parse_module(source: str) -> ast.Module:
    """Parse *source*, reusing the result across the locators.

    Every locator in this package starts by parsing the program, and a
    single viewer action runs a dozen of them over the same text: building
    the identity table alone parses once per locator and once more per
    sketch. Nothing in the package or in :mod:`cadjoint.viewer.patch`
    mutates a tree — the edits are span surgery on the *text* — so one
    parse can safely serve them all.

    The cache is small on purpose: the playground works on one program at
    a time, and a patch immediately invalidates its own entry by producing
    different text.

    Args:
        source: The program text.

    Returns:
        The parsed module. Treat it as read-only; it is shared.

    Raises:
        SyntaxError: If the text does not parse. Failures are not cached.
    """
    return ast.parse(source)


def _line_offsets(source: str) -> list[int]:
    """Absolute character offset of the start of each line (0-indexed lines)."""
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _node_span(source: str, offsets: list[int], node: ast.AST) -> Span | None:
    """Absolute character span of an AST node.

    ``col_offset`` is a UTF-8 byte offset within its line, so the prefix is
    decoded rather than sliced directly.
    """
    if getattr(node, "end_lineno", None) is None:
        return None
    lines = source.splitlines(keepends=True)

    def absolute(lineno: int, col: int) -> int:
        line = lines[lineno - 1]
        prefix = line.encode()[:col].decode()
        return offsets[lineno - 1] + len(prefix)

    return absolute(node.lineno, node.col_offset), absolute(node.end_lineno, node.end_col_offset)


def _called_name(node: ast.AST) -> str | None:
    """Name of the function a Call node invokes, bare or attribute-qualified."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _call_namespace(node: ast.AST) -> str | None:
    """Name the call is qualified by: ``Solid.box(...)`` answers ``"Solid"``."""
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return None
    owner = node.func.value
    return owner.id if isinstance(owner, ast.Name) else None


# Namespaces whose methods share a name with a construction primitive but build
# something else entirely — ``Nodes.sphere(...)`` selects mesh nodes, it does
# not create a solid.
NON_CONSTRUCTION_NAMESPACES = frozenset({"Nodes"})


def _is_construction_call(node: ast.AST) -> bool:
    """True when a call named like a primitive really does build geometry."""
    return _call_namespace(node) not in NON_CONSTRUCTION_NAMESPACES


def _is_profile_call(node: ast.AST) -> bool:
    return _called_name(node) == "PolygonProfile"


_PARAMETER_CALLS = {"Parameter", "Scalar", "Vector", "Vector2"}
_ARRAY_CALLS = {"array", "asarray"}


def _assignment_value(tree: ast.Module, name: str, before_line: int) -> ast.AST | None:
    """Value of one unambiguous module-level assignment visible at a use site.

    Following names is intentionally conservative: rebinding, tuple unpacking,
    loop targets, and function-local values are rejected rather than risking a
    source edit at the wrong definition.
    """
    values: list[ast.AST] = []
    for statement in tree.body:
        if statement.lineno >= before_line:
            continue
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == name
        ) or (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == name
            and statement.value is not None
        ):
            values.append(statement.value)
    return values[0] if len(values) == 1 else None


def _call_value_argument(call: ast.Call) -> ast.AST | None:
    """The numeric payload of a Parameter/array wrapper call."""
    for keyword in call.keywords:
        if keyword.arg == "value":
            return keyword.value
    return call.args[0] if call.args else None


def _editable_value_node(
    node: ast.AST,
    tree: ast.Module,
    seen: set[str] | None = None,
) -> ast.AST | None:
    """Resolve a numeric value through named parameters to its source literal."""
    seen = set() if seen is None else seen
    if _is_number(node):
        return node
    if isinstance(node, (ast.List, ast.Tuple)) and all(_is_number(item) for item in node.elts):
        return node
    if isinstance(node, ast.Name):
        if node.id in seen:
            return None
        value = _assignment_value(tree, node.id, node.lineno)
        return _editable_value_node(value, tree, seen | {node.id}) if value is not None else None
    if isinstance(node, ast.Call) and _called_name(node) in _PARAMETER_CALLS | _ARRAY_CALLS:
        value = _call_value_argument(node)
        return _editable_value_node(value, tree, seen) if value is not None else None
    return None


def _resolved_container(
    node: ast.AST,
    tree: ast.Module,
    seen: set[str] | None = None,
) -> ast.List | ast.Tuple | None:
    """Resolve a named vertex collection to its literal list or tuple."""
    seen = set() if seen is None else seen
    if isinstance(node, (ast.List, ast.Tuple)):
        return node
    if isinstance(node, ast.Name):
        if node.id in seen:
            return None
        value = _assignment_value(tree, node.id, node.lineno)
        return _resolved_container(value, tree, seen | {node.id}) if value is not None else None
    if isinstance(node, ast.Call) and _called_name(node) in _ARRAY_CALLS:
        value = _call_value_argument(node)
        return _resolved_container(value, tree, seen) if value is not None else None
    return None


def _resolved_call(
    node: ast.AST,
    tree: ast.Module,
    expected: str,
    seen: set[str] | None = None,
) -> ast.Call | None:
    """Resolve a named construction object to the call that creates it."""
    seen = set() if seen is None else seen
    if isinstance(node, ast.Call):
        return node if _called_name(node) == expected else None
    if isinstance(node, ast.Name):
        if node.id in seen:
            return None
        value = _assignment_value(tree, node.id, node.lineno)
        return (
            _resolved_call(value, tree, expected, seen | {node.id}) if value is not None else None
        )
    return None


def _is_number(node: ast.AST) -> bool:
    """True for numeric literals, including negated ones like ``-1.5``."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _is_number(node.operand)
    return False


def _contains(outer: ast.AST, inner: ast.AST) -> bool:
    return any(node is inner for node in ast.walk(outer))


def statement_span(source: str, line: int | None) -> Span | None:
    """Character span of the statement that declares whatever sits on *line*.

    The viewer addresses an object by the line its constructor call starts on,
    which is enough to patch it but not enough to *show* it: pointing an editor
    at ``Solid.box(`` reveals a call, and pointing it at the position literal —
    which is what the payload used to offer — reveals three numbers. What a
    reader wants when they select an object is the statement that declares it,
    ``board = Solid.box(...)``, however many lines it runs to.

    Innermost wins. A construction call inside a function or a comprehension
    resolves to the smallest statement containing it rather than to the
    enclosing ``def``, so the reveal lands on the line that built the object
    even when the program is not a flat script.

    Args:
        source: The program text.
        line: 1-based line of the constructor call, or ``None`` for an object
            the capture could not place.

    Returns:
        ``(start, end)`` character offsets of the statement, or ``None`` when
        the line is unknown or the text does not parse.
    """
    if line is None:
        return None
    try:
        tree = parse_module(source)
    except SyntaxError:
        return None
    offsets = _line_offsets(source)
    best: ast.stmt | None = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.stmt):
            continue
        end = getattr(node, "end_lineno", None)
        if end is None or not (node.lineno <= line <= end):
            continue
        # Innermost: the latest-starting statement that still contains the
        # line, so a nested one beats the block it lives in.
        if (
            best is None
            or node.lineno > best.lineno
            or (node.lineno == best.lineno and (best.end_lineno or 0) > end)
        ):
            best = node
    return _node_span(source, offsets, best) if best is not None else None
