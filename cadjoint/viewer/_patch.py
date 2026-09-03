"""Backwards-compatible alias for :mod:`cadjoint.viewer.patch`.

The patch layer outgrew a single module and now lives in the package
:mod:`cadjoint.viewer.patch`, split into formatting, span surgery, reference
resolvers, and one operation module per domain.  This module stays because
several callers — including :mod:`cadjoint.viewer.playground`,
:mod:`cadjoint.viewer._compile_worker` and the test suite — import
``cadjoint.viewer._patch`` by name; it re-exports that package's surface
unchanged and holds no logic of its own.

New code should import from :mod:`cadjoint.viewer.patch` directly.
"""

from __future__ import annotations

from cadjoint.viewer.patch import (  # noqa: F401
    CONSTRUCTION_CALLS,
    OPERATIONS,
    PatchError,
    add_constraint,
    add_extrusion,
    add_loft,
    add_material,
    add_mesh,
    add_primitive,
    add_revolution,
    add_sketch,
    add_study,
    add_study_bc,
    apply_operation,
    assign_material,
    delete_constraint,
    delete_mesh,
    delete_object,
    delete_optimization,
    delete_study,
    delete_study_bc,
    delete_vertex,
    insert_vertex,
    set_constraint_value,
    set_material_property,
    set_mesh_value,
    set_optimization_value,
    set_parameter_value,
    set_parameter_values,
    set_sketch_plane,
    set_study_value,
    set_value,
    set_vertex,
    solve_sketch,
)
