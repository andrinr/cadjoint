"""Map construction-tree objects back to the source text that created them.

The playground edits Python source, not a scene graph, so viewer interactions
(select a sketch vertex, drag it, add one) have to be expressed as edits to the
user's program. That needs two things:

1. **Which line created this profile** — captured at construction time by
   wrapping ``PolygonProfile.__init__`` while the user program executes and
   walking the stack for the playground frame.
2. **Where each vertex literal sits in the text** — recovered afterwards by
   parsing the source and locating the ``PolygonProfile(...)`` call on that
   line, then reading the character spans of its vertex list elements.

Profiles the mapper cannot pin down unambiguously (built in a loop, vertices
passed as a variable) still render in the viewer; they are just marked
non-editable.

Where to add code
-----------------

The package is layered, and imports only ever point *down* this list:

- :mod:`.nodes` — AST span/name/literal-resolution primitives (imports nothing
  from the package).
- :mod:`.capture` — runtime capture of construction lines (independent leaf).
- :mod:`.calls` — line-addressed locators for one construction call
  (``locate_call``, ``locate_profile_call``).
- :mod:`.features` — the calls that generate features, the variables they bind,
  and the plane reference a sketch was drawn on.
- :mod:`.constraints` — the profile's constraint statements and the ordinal
  that gives each constraint its viewer identity.
- :mod:`.declarations` — top-level study / mesh / optimization declarations,
  located by stable source-order index.
- :mod:`.identity` — the stable id every addressable thing is named by,
  derived from the AST path and the assigned name rather than from a line.
- :mod:`.materials` — the material browser payload and material lookups.
- :mod:`.parameters` — which named free parameter backs each value a handle
  can drag, so the frontend can move it through the shader's uniform buffer
  instead of through a source rewrite.
- :mod:`.payload` — joins captured objects with located spans into the viewer
  JSON (the only module that knows what the viewer draws).

This module re-exports the whole public surface, so callers keep importing
``from cadjoint.viewer.source_map import ...`` regardless of which layer a name
lives in.
"""

from __future__ import annotations

from cadjoint.viewer.source_map.calls import (
    CallSite,
    ProfileCall,
    locate_call,
    locate_profile_call,
)
from cadjoint.viewer.source_map.capture import PLAYGROUND_FILENAME, capture_profiles
from cadjoint.viewer.source_map.constraints import (
    CONSTRAINT_CLASS_KINDS,
    ConstraintStatement,
    locate_constraint_statements,
)
from cadjoint.viewer.source_map.declarations import (
    MESH_CALL_NAME,
    OPTIMIZATION_CALL_NAME,
    STUDY_CALL_KINDS,
    MeshStatement,
    OptimizationStatement,
    StudyStatement,
    locate_mesh_statements,
    locate_optimization_statements,
    locate_study_statements,
)
from cadjoint.viewer.source_map.features import (
    FEATURE_CALL_KINDS,
    PLANE_CONSTRUCTORS,
    PRIMITIVE_CALL_KINDS,
    FeatureCall,
    PlaneReference,
    locate_feature_call,
    locate_feature_calls,
    locate_plane_reference,
)
from cadjoint.viewer.source_map.identity import (
    IDENTITY_KINDS,
    Identity,
    build_identities,
    describe_identities,
    identity_at,
    identity_for,
    identity_index,
)
from cadjoint.viewer.source_map.materials import build_material_payload
from cadjoint.viewer.source_map.nodes import Span, statement_span
from cadjoint.viewer.source_map.payload import (
    build_construction_payload,
    build_construction_relations,
)

__all__ = [
    "CONSTRAINT_CLASS_KINDS",
    "IDENTITY_KINDS",
    "FEATURE_CALL_KINDS",
    "MESH_CALL_NAME",
    "OPTIMIZATION_CALL_NAME",
    "PLANE_CONSTRUCTORS",
    "PLAYGROUND_FILENAME",
    "PRIMITIVE_CALL_KINDS",
    "STUDY_CALL_KINDS",
    "CallSite",
    "ConstraintStatement",
    "FeatureCall",
    "Identity",
    "MeshStatement",
    "OptimizationStatement",
    "PlaneReference",
    "ProfileCall",
    "Span",
    "StudyStatement",
    "build_construction_payload",
    "build_construction_relations",
    "build_identities",
    "build_material_payload",
    "capture_profiles",
    "describe_identities",
    "identity_at",
    "identity_for",
    "identity_index",
    "locate_call",
    "locate_constraint_statements",
    "locate_feature_call",
    "locate_feature_calls",
    "locate_mesh_statements",
    "locate_optimization_statements",
    "locate_plane_reference",
    "locate_profile_call",
    "locate_study_statements",
    "statement_span",
]
