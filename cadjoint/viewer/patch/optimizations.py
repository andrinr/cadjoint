"""Patch operations for ``Optimization`` declarations.

Optimization declarations are patched like studies and meshes: located by
stable index or name
(:func:`~cadjoint.viewer.source_map.declarations.locate_optimization_statements`),
edited by span surgery on their constructor call.

There is deliberately **no add operation**: an objective is code, so the panel
points users at the editor to declare one.  Only the two solver controls are
editable from the viewer — ``steps`` (written integral) and ``learning_rate``
(exact float ``repr``) — because everything else in the constructor is the
objective itself.

Writing an optimizer's *result* back into the program is a different concern
and lives in :mod:`cadjoint.viewer.patch.parameters`.
"""

from __future__ import annotations

import ast

from cadjoint.viewer.patch.edits import (
    _delete_statement,
    _name_references,
    _rewrite_call_argument,
    _validate,
)
from cadjoint.viewer.patch.errors import PatchError
from cadjoint.viewer.patch.format import _exact_number
from cadjoint.viewer.patch.resolvers import _located_optimization

# Constructor field order, for resolving positionally written arguments;
# `steps`/`learning_rate`/`method` are keyword-only in the constructor.
_OPTIMIZATION_FIELDS = ("name", "objective", "of")
_OPTIMIZATION_ARGUMENTS = ("steps", "learning_rate")


def delete_optimization(source: str, optimization) -> str:
    """Remove one optimization declaration, identified by index or name."""
    located = _located_optimization(source, optimization)
    if located.variable is not None:
        tree = ast.parse(source)
        uses = _name_references(tree, located.variable, located.statement)
        if uses:
            raise PatchError(
                f"`{located.variable}` is used elsewhere in the program, so it cannot be "
                "deleted from the viewer. Remove those uses first."
            )
    return _validate(_delete_statement(source, located.statement))


def set_optimization_value(source: str, optimization, argument, value) -> str:
    """Edit an optimization's ``steps`` or ``learning_rate`` in place.

    Args:
        source: The program text.
        optimization: Optimization reference — payload index, name, or
            variable.
        argument: ``steps`` (positive whole number, written integral) or
            ``learning_rate`` (positive number, written with exact float
            ``repr``).
        value: The new number.

    Returns:
        The patched source.

    Raises:
        PatchError: On an unknown argument, an invalid value, or a target
            that is not an editable literal.
    """
    located = _located_optimization(source, optimization)
    if not isinstance(argument, str) or argument not in _OPTIMIZATION_ARGUMENTS:
        allowed = ", ".join(_OPTIMIZATION_ARGUMENTS)
        raise PatchError(f"An optimization's editable arguments are: {allowed}.")
    if argument == "steps":
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or float(value) != int(value)
            or int(value) < 1
        ):
            raise PatchError("`steps` must be a positive whole number.")
        expression = str(int(value))
    else:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not float(value) > 0:
            raise PatchError("`learning_rate` must be a positive number.")
        expression = _exact_number(value)
    return _rewrite_call_argument(
        source, located.call, _OPTIMIZATION_FIELDS, argument, expression, "optimization"
    )
