"""The cut-cell solver, as the research scripts import it.

The solver itself moved into the package (:mod:`cadjoint.fem.cutfem`),
where a ``SimMesh(method="cutfem")`` builds it and a ``ThermalStudy``
solves on it.  This module re-exports it — private helpers included, the
tables lean on them — and keeps one convenience of the prototype: a model
in the proto JSON form of ``research/zero-set.proto`` is accepted anywhere
a field is, through :mod:`research.cutfem.model`.
"""

from __future__ import annotations

from typing import Any

import jax

from cadjoint.fem import cutfem as _package
from cadjoint.fem.cutfem import Field, Grid, Structure
from research.cutfem.model import field as _table_field

globals().update(
    {name: value for name, value in vars(_package).items() if not name.startswith("__")}
)

# The prototype's scripts and tables assume float64 throughout; the package
# module switches it on only inside a study's solve.
jax.config.update("jax_enable_x64", True)


def as_field(model: Any) -> Field:
    """A :class:`Field`, also from a proto-JSON node table."""
    if isinstance(model, dict):
        return Field(_table_field(model))
    return _package.as_field(model)


def classify(model: Any, theta, grid: Grid, *args: Any, **kwargs: Any) -> Structure:
    """:func:`cadjoint.fem.cutfem.classify`, accepting a proto-JSON table as the model."""
    return _package.classify(as_field(model), theta, grid, *args, **kwargs)


__all__ = [name for name in vars(_package) if not name.startswith("_")] + ["as_field", "classify"]
