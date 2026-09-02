"""Backwards-compatible alias for :mod:`cadjoint.viewer.source_map`.

The source map outgrew a single module and now lives in the package
:mod:`cadjoint.viewer.source_map`, split into node primitives, runtime capture,
locators, and payload builders.  This module stays because several callers —
including :mod:`cadjoint.viewer._compile_worker` and the test suite — import
``cadjoint.viewer._source_map`` by name; it re-exports that package's surface
unchanged, private helpers included, and holds no logic of its own.

New code should import from :mod:`cadjoint.viewer.source_map` directly.
"""

from __future__ import annotations

from cadjoint.viewer.source_map import (  # noqa: F401
    CONSTRAINT_CLASS_KINDS,
    FEATURE_CALL_KINDS,
    IDENTITY_KINDS,
    MESH_CALL_NAME,
    OPTIMIZATION_CALL_NAME,
    PLANE_CONSTRUCTORS,
    PLAYGROUND_FILENAME,
    PRIMITIVE_CALL_KINDS,
    STUDY_CALL_KINDS,
    CallSite,
    ConstraintStatement,
    FeatureCall,
    Identity,
    MeshStatement,
    OptimizationStatement,
    PlaneReference,
    ProfileCall,
    Span,
    StudyStatement,
    build_construction_payload,
    build_construction_relations,
    build_identities,
    build_material_payload,
    capture_profiles,
    describe_identities,
    identity_at,
    identity_for,
    identity_index,
    locate_call,
    locate_constraint_statements,
    locate_feature_call,
    locate_feature_calls,
    locate_mesh_statements,
    locate_optimization_statements,
    locate_plane_reference,
    locate_profile_call,
    locate_study_statements,
)
from cadjoint.viewer.source_map.calls import _vertices_argument  # noqa: F401
from cadjoint.viewer.source_map.capture import _caller_line  # noqa: F401
from cadjoint.viewer.source_map.nodes import (  # noqa: F401
    _assignment_value,
    _call_value_argument,
    _called_name,
    _contains,
    _editable_value_node,
    _is_number,
    _is_profile_call,
    _line_offsets,
    _node_span,
    _resolved_call,
    _resolved_container,
)
