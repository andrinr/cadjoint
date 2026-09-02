"""Patch operations for sketch constraints and the solve step that applies them.

Constraints are bare top-level statements, so their identity is positional: the
ordinal :func:`~cadjoint.viewer.source_map.constraints.locate_constraint_statements`
assigns is what a viewer chip's ``index`` addresses.  Two consequences shape
this module:

- a new constraint is appended **after the profile's last existing constraint
  statement**, never prepended — source order is creation order, and
  prepending would renumber every chip already on screen;
- its import is inserted **beside the statement** (``prefer_offset``) so
  nothing above the profile moves and further line-addressed calls stay valid
  without recompiling in between.

:func:`set_constraint_value` follows named-parameter indirection: when the
target is a name bound to ``Scalar(value=...)``, the Scalar's literal is
rewritten so every use of the name stays consistent.  It writes exact
``repr`` values — the compact ``%.4g`` form is for drag-generated coordinates,
not numbers the user chose.

:func:`solve_sketch` belongs here because the solve call has to land after the
whole constraint block, imports included.

Add an operation here when it edits a constraint statement or the solve step.
"""

from __future__ import annotations

import ast

from cadjoint.enums import ConstraintSolveMethod, either, values
from cadjoint.viewer.patch.edits import (
    _after_statement,
    _delete_statement,
    _ensure_import,
    _set_keyword_expression,
    _validate,
)
from cadjoint.viewer.patch.errors import PatchError
from cadjoint.viewer.patch.format import _format_value
from cadjoint.viewer.patch.resolvers import _located_constraint, _profile_binding
from cadjoint.viewer.source_map import locate_constraint_statements
from cadjoint.viewer.source_map.nodes import (
    _called_name,
    _editable_value_node,
    _line_offsets,
    _node_span,
)

_RELATIONAL_CONSTRAINT_SYMBOLS = {
    "horizontal": "HorizontalConstraint",
    "vertical": "VerticalConstraint",
    "coincident": "CoincidentConstraint",
    "parallel": "ParallelEdgesConstraint",
    "perpendicular": "PerpendicularEdgesConstraint",
}


def add_constraint(
    source: str,
    line: int,
    kind: str,
    indices: list[int],
    value=None,
) -> str:
    """Attach a constraint statement to sketch vertices.

    ``fixed`` and ``distance`` carry a numeric ``value``; the relational kinds
    (``horizontal``, ``vertical``, ``coincident`` on two vertices, ``parallel``
    and ``perpendicular`` on two edges given as four vertices) take none.
    """
    _, _, statement, profile = _profile_binding(source, line)

    def vertex(index: int) -> str:
        return f"{profile}.vertices[{index}]"

    if kind == "fixed" and len(indices) == 1:
        constraint = f"FixedConstraint({vertex(indices[0])}, {_format_value(value)})\n"
        symbol = "FixedConstraint"
    elif kind == "distance" and len(indices) == 2:
        constraint = (
            f"DistanceConstraint({vertex(indices[0])}, {vertex(indices[1])}, "
            f"{_format_value(value)})\n"
        )
        symbol = "DistanceConstraint"
    elif kind in {"horizontal", "vertical", "coincident"} and len(indices) == 2:
        symbol = _RELATIONAL_CONSTRAINT_SYMBOLS[kind]
        constraint = f"{symbol}({vertex(indices[0])}, {vertex(indices[1])})\n"
    elif kind in {"parallel", "perpendicular"} and len(indices) == 4:
        symbol = _RELATIONAL_CONSTRAINT_SYMBOLS[kind]
        constraint = f"{symbol}({', '.join(vertex(index) for index in indices)})\n"
    else:
        raise PatchError(
            "Constraints take one vertex (`fixed`), two (`distance`, `horizontal`, "
            "`vertical`, `coincident`), or four (`parallel`, `perpendicular`)."
        )

    # Append after the profile's LAST existing constraint statement so source
    # order equals creation order — the serialized payload indices the viewer
    # chips address are statement ordinals, and prepending would reverse them.
    existing = locate_constraint_statements(source, line)
    anchor = existing[-1].statement if existing else statement
    insert = _after_statement(source, anchor)
    patched = source[:insert] + constraint + source[insert:]
    # Keep everything above the profile untouched: a first-time constraint
    # import lands next to the constraint statement, so repeated
    # line-addressed calls stay valid without recompiling in between.
    patched = _ensure_import(
        patched, ast.parse(patched), "cadjoint.constraints", symbol, prefer_offset=insert
    )
    return _validate(patched)


def delete_constraint(source: str, line: int, index: int) -> str:
    """Remove one constraint statement, identified by its payload index.

    The index is the ordinal :func:`locate_constraint_statements` assigns —
    the same ordering the viewer payload carries — so a chip's ``index``
    deletes exactly the statement it displays.
    """
    located = _located_constraint(source, line, index)
    return _validate(_delete_statement(source, located.statement))


def set_constraint_value(source: str, line: int, index: int, value) -> str:
    """Rewrite the numeric target of a fixed or distance constraint.

    Follows named-parameter indirection: when the target is a name bound to
    ``Scalar(value=...)`` (the starter program's ``wall_width`` pattern), the
    Scalar's literal is rewritten so every use of the name stays consistent.
    """
    located = _located_constraint(source, line, index)
    if located.kind == "distance":
        target = located.call.args[2] if len(located.call.args) > 2 else None
    elif located.kind == "fixed":
        target = located.call.args[1] if len(located.call.args) > 1 else None
    else:
        raise PatchError("Only `fixed` and `distance` constraints carry an editable value.")
    if target is None:
        raise PatchError("The constraint statement has no value argument to rewrite.")
    tree = ast.parse(source)
    literal = _editable_value_node(target, tree)
    if literal is None:
        raise PatchError("The constraint value is not an editable literal.")
    offsets = _line_offsets(source)
    start, end = _node_span(source, offsets, literal)

    def exact(component) -> str:
        # Typed-in values must round-trip exactly; the %.4g compaction is for
        # drag-generated coordinates, not numbers the user chose.
        return repr(float(component))

    formatted = (
        exact(value)
        if isinstance(value, (int, float))
        else "[" + ", ".join(exact(component) for component in value) + "]"
    )
    return _validate(source[:start] + formatted + source[end:])


def solve_sketch(
    source: str,
    line: int,
    method: str = "newton",
    iterations: int = 8,
) -> str:
    """Add or update an in-program solve step for a sketch's constraints."""
    if method not in values(ConstraintSolveMethod):
        raise PatchError(f"Solver method must be {either(ConstraintSolveMethod)}.")
    if (
        not isinstance(iterations, int)
        or isinstance(iterations, bool)
        or not 1 <= iterations <= 512
    ):
        raise PatchError("Solver iterations must be an integer from 1 to 512.")

    tree, _, profile_statement, profile = _profile_binding(source, line)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and _called_name(node) == "satisfy_constraints"
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == profile
        ):
            patched = _set_keyword_expression(source, node, "method", repr(method))
            updated_tree = ast.parse(patched)
            updated_call = next(
                candidate
                for candidate in ast.walk(updated_tree)
                if isinstance(candidate, ast.Call)
                and _called_name(candidate) == "satisfy_constraints"
                and candidate.args
                and isinstance(candidate.args[0], ast.Name)
                and candidate.args[0].id == profile
            )
            return _set_keyword_expression(
                patched,
                updated_call,
                "steps",
                str(iterations),
            )

    statements = list(tree.body)
    index = statements.index(profile_statement)
    anchor = profile_statement
    constraint_names = {
        "FixedConstraint",
        "DistanceConstraint",
        "HorizontalConstraint",
        "VerticalConstraint",
        "CoincidentConstraint",
        "EqualLengthConstraint",
        "PointOnLineConstraint",
        "ParallelEdgesConstraint",
        "PerpendicularEdgesConstraint",
    }
    for candidate in statements[index + 1 :]:
        if (
            isinstance(candidate, ast.Expr)
            and isinstance(candidate.value, ast.Call)
            and _called_name(candidate.value) in constraint_names
        ):
            anchor = candidate
            continue
        if isinstance(candidate, (ast.Import, ast.ImportFrom)):
            # Constraint imports are placed beside their statements; they are
            # part of the block the solve step must follow.
            anchor = candidate
            continue
        break

    insert = _after_statement(source, anchor)
    patched = (
        source[:insert]
        + f"satisfy_constraints({profile}, method={method!r}, steps={iterations})\n"
        + source[insert:]
    )
    patched = _ensure_import(
        patched, ast.parse(patched), "cadjoint.constraints", "satisfy_constraints"
    )
    return _validate(patched)
