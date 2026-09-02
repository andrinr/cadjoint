"""The typed contract between the compile worker and the viewer.

For most of the playground's life the payload had no schema: the worker
built nested dicts, the frontend declared what it hoped to find in
``frontend/src/types.ts``, and the only thing keeping the two honest was
that both were edited in the same commit. A field renamed on one side
became ``undefined`` on the other, silently.

This package is the single definition instead. :mod:`.payloads` models the
compile response and everything inside it; :mod:`.requests` models every
``/patch`` request the server accepts, as a union discriminated on ``op``.
:mod:`.emit` turns those models into TypeScript, and
``payloads.d.ts`` — regenerated and diffed by a test — is what the frontend
imports, so the two sides cannot drift without something going red.

The models are *descriptive*: the worker builds the payload as it always
did and validates it at the boundary. Nothing here changes what a payload
says, only what it is allowed to say.
"""

from __future__ import annotations

from cadjoint.viewer.schema.emit import TYPESCRIPT_PATH, typescript_source
from cadjoint.viewer.schema.payloads import (
    CompilePayload,
    ConstraintSolverRun,
    ConstructionConstraint,
    ConstructionFace,
    ConstructionNode,
    ConstructionOperator,
    ConstructionPlane,
    ConstructionRelation,
    ConstructionTransform,
    ConstructionVertex,
    DomainEntry,
    FaceOwner,
    IdentityEntry,
    MaterialDefinition,
    MeshEdgePayload,
    OptimizationPayload,
    PlaneReference,
    SimMeshPayload,
    StudyBc,
    StudyPayload,
    StudySelection,
)
from cadjoint.viewer.schema.requests import (
    PATCH_REQUEST_MODELS,
    PatchRequest,
    PatchResponse,
    validate_patch_request,
)

__all__ = [
    "PATCH_REQUEST_MODELS",
    "TYPESCRIPT_PATH",
    "CompilePayload",
    "ConstraintSolverRun",
    "ConstructionConstraint",
    "ConstructionFace",
    "ConstructionNode",
    "ConstructionOperator",
    "ConstructionPlane",
    "ConstructionRelation",
    "ConstructionTransform",
    "ConstructionVertex",
    "DomainEntry",
    "FaceOwner",
    "IdentityEntry",
    "MaterialDefinition",
    "MeshEdgePayload",
    "OptimizationPayload",
    "PatchRequest",
    "PatchResponse",
    "PlaneReference",
    "SimMeshPayload",
    "StudyBc",
    "StudyPayload",
    "StudySelection",
    "typescript_source",
    "validate_patch_request",
]
