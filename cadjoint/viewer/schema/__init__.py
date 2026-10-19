"""The typed contract between the compile worker and the viewer.

This package is the single definition of the payloads both sides exchange:

- :mod:`.payloads` models the compile response and everything inside it;
- :mod:`.requests` models every ``/api/patch`` request the server accepts,
  as a union discriminated on ``op``, and the export request;
- :mod:`.api` models every other route's request and response, lists every
  route in :data:`~.api.ROUTES`, and carries :data:`~.api.API_VERSION`;
- :mod:`.worker` models the server↔worker pipe, one request and one reply
  per mode, and is the boundary check on everything a worker sends.

:mod:`.emit` turns those models into TypeScript, and ``payloads.d.ts`` —
regenerated and diffed by a test — is what the frontend imports, so the two
sides cannot drift without something going red.

The models are *descriptive*: the worker and the endpoints build their
answers as plain dicts, and the answers are checked at the boundary.
Nothing here changes what a payload says, only what it is allowed to say.
"""

from __future__ import annotations

from cadjoint.viewer.schema.api import API_VERSION, API_VERSION_HEADER, ROUTES, Failure, Route
from cadjoint.viewer.schema.emit import TYPESCRIPT_PATH, typescript_source
from cadjoint.viewer.schema.payloads import (
    CompilePayload,
    ConstraintSolverRun,
    ConstructionArgument,
    ConstructionConstraint,
    ConstructionElement,
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
    PatchAddress,
    PlaneReference,
    SimMeshPayload,
    StudyBc,
    StudyBcDescription,
    StudyPayload,
    StudySelection,
)
from cadjoint.viewer.schema.requests import (
    PATCH_REQUEST_MODELS,
    ExportRequest,
    PatchRequest,
    PatchResponse,
    validate_patch_request,
)

__all__ = [
    "API_VERSION",
    "API_VERSION_HEADER",
    "PATCH_REQUEST_MODELS",
    "ROUTES",
    "TYPESCRIPT_PATH",
    "CompilePayload",
    "ConstraintSolverRun",
    "ConstructionArgument",
    "ConstructionElement",
    "PatchAddress",
    "ConstructionConstraint",
    "ConstructionFace",
    "ConstructionNode",
    "ConstructionOperator",
    "ConstructionPlane",
    "ConstructionRelation",
    "ConstructionTransform",
    "ConstructionVertex",
    "DomainEntry",
    "ExportRequest",
    "FaceOwner",
    "Failure",
    "IdentityEntry",
    "MaterialDefinition",
    "MeshEdgePayload",
    "OptimizationPayload",
    "PatchRequest",
    "PatchResponse",
    "PlaneReference",
    "Route",
    "SimMeshPayload",
    "StudyBc",
    "StudyBcDescription",
    "StudyPayload",
    "StudySelection",
    "typescript_source",
    "validate_patch_request",
]
