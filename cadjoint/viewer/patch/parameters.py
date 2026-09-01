"""Write optimized parameter values back into the program.

``/api/optimize`` turns a finished run into a source patch: each optimized free
parameter's declaration literal is rewritten with the exact-repr conventions of
``set_constraint_value``, so the optimizer stays a patch layer over the code
rather than a separate store of results.

A parameter reaches its declaration by one of two routes, tried in that order:
an explicit ``Scalar``/``Vector``/``Vector2`` call naming it, or — for the
parameters a solid derives from its own name, where
``Solid.cylinder(radius=0.07, name="bush_a")`` declares ``bush_a_radius`` —
the matching keyword literal inside that construction call.

:func:`set_parameter_values` re-locates each declaration as it goes, so an
earlier rewrite can never invalidate a later span, and sorts by name so the
output is deterministic.

These operations are not in the viewer's ``OPERATIONS`` registry: they are
called by the optimize endpoint, not by a user gesture.
"""

from __future__ import annotations

from cadjoint.viewer.patch.edits import _rewrite_call_argument
from cadjoint.viewer.patch.errors import PatchError
from cadjoint.viewer.patch.format import _exact_value
from cadjoint.viewer.patch.resolvers import _located_derived_argument, _located_parameter_call


def set_parameter_value(source: str, name: str, value) -> str:
    """Rewrite one named free parameter's value literal (exact repr).

    Explicit ``Scalar``/``Vector``/``Vector2`` declarations rewrite their
    ``value`` literal; parameters a construction call derives from its own
    ``name`` (``Solid.cylinder(radius=..., name="bush_a")`` declares
    ``bush_a_radius``) rewrite the corresponding keyword literal instead.
    """
    try:
        call = _located_parameter_call(source, name)
    except PatchError:
        derived = _located_derived_argument(source, name)
        if derived is None:
            raise
        derived_call, argument = derived
        return _rewrite_call_argument(
            source, derived_call, (argument,), argument, _exact_value(value), "parameter"
        )
    return _rewrite_call_argument(
        source, call, ("value",), "value", _exact_value(value), "parameter"
    )


def set_parameter_values(source: str, values: dict) -> str:
    """Write a dict of optimized parameter values back into the program.

    Each rewrite re-locates its declaration, so earlier edits cannot
    invalidate later spans.  Parameter order is name-sorted for
    deterministic output.
    """
    for name in sorted(values):
        source = set_parameter_value(source, name, values[name])
    return source
