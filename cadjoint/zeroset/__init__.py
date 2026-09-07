"""The zero-set interface: cadjoint as a frontend of ``research/zero-set.proto``.

A model here is a flat node table — expressions, and shapes in three forms
(patch, warp, combine) — plus a parameter vector.  :func:`lower` turns a
cadjoint SDF graph into one; :mod:`.evaluate` evaluates a table with JAX,
which is the host's oracle for derivatives; :mod:`.wgsl` emits a table as
WGSL, a fold with one case per form.
"""

from __future__ import annotations

from cadjoint.zeroset.table import Model, Table, lower

__all__ = ["Model", "Table", "lower"]
