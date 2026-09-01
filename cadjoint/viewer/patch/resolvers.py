"""Turn a viewer reference into the source construct it names.

A patch request never carries an AST node — it carries a line number, a
payload index, a study name, a parameter name.  Every ``_located_*`` function
here performs that one job: resolve such a reference against the program text
and either return the located construct or raise a :class:`PatchError` whose
message tells the user why the edit cannot be made.

Two rules keep the layer honest:

- **Ambiguity is refused, never guessed.** Two candidate calls on a line, two
  studies with the same name, a parameter declared twice — all become errors
  rather than an edit at the wrong place.
- **The index is the payload's index.** ``_located_constraint``,
  ``_located_study`` and friends address the same ordinals the compile payload
  serialized, which is what makes a chip's index delete exactly the statement
  it displays.

Add a resolver here whenever a new kind of viewer reference appears.  The
operations then read as "resolve, then edit", and none of them re-walks the
tree looking for its own target.
"""

from __future__ import annotations

import ast

from cadjoint.viewer.patch.edits import _statement_containing
from cadjoint.viewer.patch.errors import PatchError
from cadjoint.viewer.source_map import (
    MeshStatement,
    OptimizationStatement,
    Span,
    StudyStatement,
    locate_constraint_statements,
    locate_mesh_statements,
    locate_optimization_statements,
    locate_profile_call,
    locate_study_statements,
)
from cadjoint.viewer.source_map.nodes import _called_name

CONSTRUCTION_CALLS = {"PolygonProfile", "box", "sphere", "cylinder"}

_PARAMETER_CALL_NAMES = ("Scalar", "Vector", "Vector2")


def _require_call(source: str, line: int):
    call = locate_profile_call(source, line)
    if call is None:
        raise PatchError(
            f"No editable PolygonProfile literal found at line {line}. "
            "Sketches built in a loop or from variables cannot be edited from the viewer."
        )
    return call


def _profile_binding(source: str, line: int):
    """Return ``(tree, call, statement, variable)`` for a named profile."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise PatchError(f"Source is not valid Python: {error}") from error
    calls = [
        node
        for node in ast.walk(tree)
        if _called_name(node) == "PolygonProfile"
        and node.lineno <= line <= (node.end_lineno or node.lineno)
    ]
    if len(calls) != 1:
        raise PatchError(f"No single PolygonProfile() call found at line {line}.")
    call = calls[0]
    statement = _statement_containing(tree, call)
    if not (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    ):
        raise PatchError("Name the sketch before adding constraints or operators from the viewer.")
    return tree, call, statement, statement.targets[0].id


def _located_constraint(source: str, line: int, index: int):
    """The profile's ``index``-th constraint statement, payload-order."""
    statements = locate_constraint_statements(source, line)
    if statements is None:
        raise PatchError("Name the sketch before editing constraints from the viewer.")
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(statements):
        raise PatchError(
            f"Constraint index {index} is out of range; the sketch has {len(statements)}."
        )
    return statements[index]


def _located_study(source: str, study) -> StudyStatement:
    """Resolve a study reference — payload index or name — to its statement."""
    statements = locate_study_statements(source)
    if statements is None:
        raise PatchError("Source is not valid Python.")
    if isinstance(study, bool) or not isinstance(study, (int, str)):
        raise PatchError("A study is referenced by its name or its non-negative index.")
    if isinstance(study, int):
        if not 0 <= study < len(statements):
            raise PatchError(
                f"Study index {study} is out of range; the program declares {len(statements)}."
            )
        return statements[study]
    matches = [
        statement for statement in statements if study in (statement.name, statement.variable)
    ]
    if len(matches) != 1:
        declared = ", ".join(
            repr(statement.name or statement.variable or f"#{statement.index}")
            for statement in statements
        )
        raise PatchError(
            f"No single study named {study!r}; the program declares: {declared or 'none'}."
        )
    return matches[0]


def _reject_predicate_bc(element: ast.AST) -> None:
    """Refuse edits to a BC built on a non-serializable predicate selection."""
    if any(_called_name(node) == "predicate" for node in ast.walk(element)):
        raise PatchError(
            "This boundary condition uses a `Nodes.predicate` selection, which is not "
            "serializable; edit it directly in the code."
        )


def _located_study_bc(located: StudyStatement, bc) -> tuple[ast.AST, Span]:
    """The ``bc``-th element of a study's literal BC list, with its span."""
    if located.bcs is None:
        raise PatchError("The study's `bcs` argument is not an editable literal list.")
    count = len(located.bc_spans)
    if not isinstance(bc, int) or isinstance(bc, bool) or not 0 <= bc < count:
        raise PatchError(f"Boundary-condition index {bc} is out of range; the study has {count}.")
    element = located.bcs.elts[bc]
    _reject_predicate_bc(element)
    return element, located.bc_spans[bc]


def _located_mesh(source: str, mesh) -> MeshStatement:
    """Resolve a mesh reference — payload index, name, or variable."""
    statements = locate_mesh_statements(source)
    if statements is None:
        raise PatchError("Source is not valid Python.")
    if isinstance(mesh, bool) or not isinstance(mesh, (int, str)):
        raise PatchError("A mesh is referenced by its name or its non-negative index.")
    if isinstance(mesh, int):
        if not 0 <= mesh < len(statements):
            raise PatchError(
                f"Mesh index {mesh} is out of range; the program declares {len(statements)}."
            )
        return statements[mesh]
    matches = [
        statement for statement in statements if mesh in (statement.name, statement.variable)
    ]
    if len(matches) != 1:
        declared = ", ".join(
            repr(statement.name or statement.variable or f"#{statement.index}")
            for statement in statements
        )
        raise PatchError(
            f"No single mesh named {mesh!r}; the program declares: {declared or 'none'}."
        )
    return matches[0]


def _located_optimization(source: str, optimization) -> OptimizationStatement:
    """Resolve an optimization reference — payload index, name, or variable."""
    statements = locate_optimization_statements(source)
    if statements is None:
        raise PatchError("Source is not valid Python.")
    if isinstance(optimization, bool) or not isinstance(optimization, (int, str)):
        raise PatchError("An optimization is referenced by its name or its non-negative index.")
    if isinstance(optimization, int):
        if not 0 <= optimization < len(statements):
            raise PatchError(
                f"Optimization index {optimization} is out of range; the program declares "
                f"{len(statements)}."
            )
        return statements[optimization]
    matches = [
        statement
        for statement in statements
        if optimization in (statement.name, statement.variable)
    ]
    if len(matches) != 1:
        declared = ", ".join(
            repr(statement.name or statement.variable or f"#{statement.index}")
            for statement in statements
        )
        raise PatchError(
            f"No single optimization named {optimization!r}; the program declares: "
            f"{declared or 'none'}."
        )
    return matches[0]


def _located_parameter_call(source: str, name: str) -> ast.Call:
    """The one top-level parameter constructor declaring ``name=<name>``."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise PatchError(f"Source is not valid Python: {error}") from error
    matches = [
        node
        for item in tree.body
        for node in ast.walk(item)
        if _called_name(node) in _PARAMETER_CALL_NAMES
        and any(
            keyword.arg == "name"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == name
            for keyword in node.keywords
        )
    ]
    if len(matches) != 1:
        raise PatchError(
            f"The optimized parameter {name!r} maps to {len(matches)} "
            "Scalar/Vector/Vector2 declarations in the program; every free parameter "
            "needs exactly one top-level declaration to write its value back."
        )
    return matches[0]


def _located_derived_argument(source: str, parameter: str) -> tuple[ast.Call, str] | None:
    """A construction call declaring *parameter* as ``<name>_<argument>``.

    Solid constructors wrap raw dimension literals into free parameters
    named after the solid — ``Solid.cylinder(radius=0.07, name="bush_a")``
    declares ``bush_a_radius`` — so the literal keyword in the call is that
    parameter's one source declaration.  Returns the call and the argument
    name when exactly one top-level call matches, None otherwise.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None  # the caller re-raises its clearer PatchError
    matches: list[tuple[ast.Call, str]] = []
    for item in tree.body:
        for node in ast.walk(item):
            if not isinstance(node, ast.Call):
                continue
            base = next(
                (
                    keyword.value.value
                    for keyword in node.keywords
                    if keyword.arg == "name"
                    and isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, str)
                ),
                None,
            )
            if base is None or not parameter.startswith(f"{base}_"):
                continue
            argument = parameter[len(base) + 1 :]
            if any(keyword.arg == argument for keyword in node.keywords):
                matches.append((node, argument))
    return matches[0] if len(matches) == 1 else None
