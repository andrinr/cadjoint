"""Rewrite the user's Python source in response to viewer interactions.

Viewer interactions are applied to the program text itself, so the editor stays
the single source of truth. Edits are pure span surgery guided by
:mod:`cadjoint.viewer.source_map`: only the characters of the targeted vertex
literal change, leaving formatting, comments, and the rest of the file byte for
byte identical.

No user code is executed here — the server can patch without spawning the
compile worker.

The registry
------------

:data:`OPERATIONS` is the whole vocabulary the server will accept, and
:func:`apply_operation` is the only entry point it uses.  Adding an operation
means writing it in the module for its domain and adding one line to that
table; the operation's own name is what the request carries, and a name that
is not in the table is rejected with a :class:`~.errors.PatchError`.

Where to add code
-----------------

The package is layered, and imports only ever point *down* this list:

- :mod:`.errors` — :class:`PatchError`, the one exception type (leaf).
- :mod:`.format` — values to literal source text: compact for viewer-generated
  numbers, exact ``repr`` for user-typed ones.
- :mod:`.edits` — span surgery: validation, import insertion, keyword rewrite.
- :mod:`.scene` — the ``scene = ...`` assignment, and names in scope at a line.
- :mod:`.resolvers` — every ``_located_*``: a viewer reference (line, index,
  name) to the source construct it names.
- The operation modules, one per domain, which use all of the above:
  :mod:`.sketch` (vertices, the sketch's work plane, extrude/revolve/loft),
  :mod:`.geometry` (solids:
  placement, creation, deletion), :mod:`.materials`, :mod:`.constraints`
  (constraint statements and the solve step), :mod:`.studies`, :mod:`.meshes`,
  :mod:`.optimizations`, and :mod:`.parameters` (optimizer writeback).
- This module — the registry and dispatch, which imports the operations and
  nothing imports back.
"""

from __future__ import annotations

from cadjoint.viewer.patch.constraints import (
    add_constraint,
    delete_constraint,
    set_constraint_value,
    solve_sketch,
)
from cadjoint.viewer.patch.errors import PatchError
from cadjoint.viewer.patch.geometry import add_primitive, delete_object, set_value
from cadjoint.viewer.patch.materials import add_material, assign_material
from cadjoint.viewer.patch.meshes import add_mesh, delete_mesh, set_mesh_value
from cadjoint.viewer.patch.optimizations import delete_optimization, set_optimization_value
from cadjoint.viewer.patch.parameters import set_parameter_value, set_parameter_values
from cadjoint.viewer.patch.resolvers import CONSTRUCTION_CALLS
from cadjoint.viewer.patch.sketch import (
    add_extrusion,
    add_loft,
    add_revolution,
    add_sketch,
    delete_vertex,
    insert_vertex,
    set_sketch_plane,
    set_vertex,
)
from cadjoint.viewer.patch.studies import (
    add_study,
    add_study_bc,
    delete_study,
    delete_study_bc,
    set_study_value,
)

OPERATIONS = {
    "set_vertex": set_vertex,
    "insert_vertex": insert_vertex,
    "delete_vertex": delete_vertex,
    "set_value": set_value,
    "add_primitive": add_primitive,
    "add_material": add_material,
    "assign_material": assign_material,
    "add_sketch": add_sketch,
    "set_sketch_plane": set_sketch_plane,
    "add_extrusion": add_extrusion,
    "add_revolution": add_revolution,
    "add_loft": add_loft,
    "add_constraint": add_constraint,
    "delete_constraint": delete_constraint,
    "set_constraint_value": set_constraint_value,
    "solve_sketch": solve_sketch,
    "delete_object": delete_object,
    "add_study": add_study,
    "delete_study": delete_study,
    "add_study_bc": add_study_bc,
    "delete_study_bc": delete_study_bc,
    "set_study_value": set_study_value,
    "add_mesh": add_mesh,
    "delete_mesh": delete_mesh,
    "set_mesh_value": set_mesh_value,
    "delete_optimization": delete_optimization,
    "set_optimization_value": set_optimization_value,
}


def apply_operation(source: str, operation: str, **kwargs) -> str:
    """Dispatch a named patch operation.

    Args:
        source: The program text.
        operation: One of ``set_vertex``, ``insert_vertex``, ``delete_vertex``.
        **kwargs: Arguments for that operation (``line``, ``index``, ``xy``).

    Returns:
        The patched source.

    Raises:
        PatchError: On an unknown operation or a failed edit.
    """
    handler = OPERATIONS.get(operation)
    if handler is None:
        raise PatchError(f"Unknown patch operation {operation!r}.")
    try:
        return handler(source, **kwargs)
    except TypeError as error:
        raise PatchError(f"Invalid arguments for {operation!r}: {error}") from error


__all__ = [
    "CONSTRUCTION_CALLS",
    "OPERATIONS",
    "PatchError",
    "add_constraint",
    "add_extrusion",
    "add_loft",
    "add_material",
    "add_mesh",
    "add_primitive",
    "add_revolution",
    "add_sketch",
    "add_study",
    "add_study_bc",
    "apply_operation",
    "assign_material",
    "delete_constraint",
    "delete_mesh",
    "delete_object",
    "delete_optimization",
    "delete_study",
    "delete_study_bc",
    "delete_vertex",
    "insert_vertex",
    "set_constraint_value",
    "set_mesh_value",
    "set_optimization_value",
    "set_parameter_value",
    "set_parameter_values",
    "set_sketch_plane",
    "set_study_value",
    "set_value",
    "set_vertex",
    "solve_sketch",
]
