"""Serializing the objects a playground program *declares* for the viewer.

One entry per declared study, simulation mesh, and optimization: the
object's own ``describe()`` payload plus where it was written — statement
line, constructor-call span, argument spans — and whether the viewer may
edit it there.  Declaration only: nothing here meshes, solves, or
descends; that is what the simulate/optimize/mesh-inspect stages do.

The source locations come from :mod:`cadjoint.viewer._source_map`; these
entries are what the compile payload carries under ``studies``,
``sim_meshes``, and ``optimizations``.
"""

from __future__ import annotations

from typing import Any

from cadjoint.viewer._source_map import (
    locate_mesh_statements,
    locate_optimization_statements,
    locate_study_statements,
)
from cadjoint.viewer.source_map.identity import build_identities


def _declaration_ids(source: str, kind: str) -> dict[int, str]:
    """Source-order index to stable id, for one kind of declaration.

    The entries below address their statement by ``index``, which is a
    position and moves the moment a declaration is added above.  The stable
    id does not, so every entry carries both.
    """
    return {
        identity.index: identity.id
        for identity in build_identities(source)
        if identity.kind == kind and identity.index is not None
    }


def _bc_ids(source: str, study_index: int) -> dict[int, str]:
    """Position in a study's ``bcs`` list to the stable id naming it."""
    owner = next(
        (
            identity.id
            for identity in build_identities(source)
            if identity.kind == "study" and identity.index == study_index
        ),
        None,
    )
    if owner is None:
        return {}
    return {
        identity.index: identity.id
        for identity in build_identities(source)
        if identity.kind == "bc" and identity.owner == owner and identity.index is not None
    }


def _study_entries(studies: list[Any], source: str) -> list[dict[str, Any]]:
    """Serialize declared studies for the viewer, with their source locations.

    Mirrors how constraints flow into the payload: each entry is the study's
    ``describe()`` dict plus a stable ``index``, the statement's ``line`` and
    the constructor call's character ``span``, an ``editable`` flag, and a
    per-BC ``serializable`` flag (false only for predicate selections) with
    the BC argument's character ``span``.

    Studies are matched to source statements positionally: top-level
    declarations execute in source order, so the alignment holds exactly when
    the counts and kinds agree.  Anything else (studies built in loops, from
    helper functions) still renders but is marked non-editable.
    """
    statements = locate_study_statements(source) or []
    stable_ids = _declaration_ids(source, "study")
    aligned = len(statements) == len(studies) and all(
        statement.kind == study.describe()["kind"] for statement, study in zip(statements, studies)
    )
    entries: list[dict[str, Any]] = []
    for index, study in enumerate(studies):
        described = study.describe()
        statement = statements[index] if aligned else None
        bc_spans: tuple[Any, ...] = ()
        if statement is not None and len(statement.bc_spans) == len(study.bcs):
            bc_spans = statement.bc_spans
        bc_ids = _bc_ids(source, index) if statement is not None else {}
        entries.append(
            {
                **described,
                "index": index,
                "stableId": stable_ids.get(index) if statement is not None else None,
                "line": statement.statement.lineno if statement is not None else None,
                "span": list(statement.call_span) if statement is not None else None,
                "editable": statement is not None,
                "mesh_span": list(statement.mesh_span)
                if statement is not None and statement.mesh_span is not None
                else None,
                "domain_span": list(statement.domain_span)
                if statement is not None and statement.domain_span is not None
                else None,
                "bcs": [
                    {
                        **bc.describe(),
                        "stableId": bc_ids.get(position),
                        "serializable": bc.nodes.serializable,
                        "span": list(bc_spans[position]) if bc_spans else None,
                    }
                    for position, bc in enumerate(study.bcs)
                ],
            }
        )
    return entries


def _mesh_entries(sim_meshes: list[Any], source: str) -> list[dict[str, Any]]:
    """Serialize declared simulation meshes for the viewer, with locations.

    Mirrors :func:`_study_entries`: each entry is the mesh's ``describe()``
    dict plus a stable ``index``, the statement's ``line`` and the
    constructor call's character ``span``, and an ``editable`` flag.  Meshes
    are matched to source statements positionally; a count mismatch (meshes
    built in loops or helpers) or a literal-name mismatch marks every entry
    non-editable.  Declaration only: nothing is built here.
    """
    statements = locate_mesh_statements(source) or []
    stable_ids = _declaration_ids(source, "mesh")
    aligned = len(statements) == len(sim_meshes) and all(
        statement.name is None or statement.name == mesh.name
        for statement, mesh in zip(statements, sim_meshes)
    )
    entries: list[dict[str, Any]] = []
    for index, mesh in enumerate(sim_meshes):
        statement = statements[index] if aligned else None
        entries.append(
            {
                **mesh.describe(),
                "index": index,
                "stableId": stable_ids.get(index) if statement is not None else None,
                "line": statement.statement.lineno if statement is not None else None,
                "span": list(statement.call_span) if statement is not None else None,
                "editable": statement is not None,
            }
        )
    return entries


def _optimization_entries(
    optimizations: list[Any], source: str, scene: Any = None
) -> list[dict[str, Any]]:
    """Serialize declared optimizations for the viewer, with locations.

    Mirrors :func:`_mesh_entries`: each entry is the optimization's
    ``describe()`` dict plus a stable ``index``, the statement's ``line``
    and the constructor call's character ``span``, an ``editable`` flag,
    and the ``steps``/``learning_rate`` argument-value spans.
    Optimizations are matched to source statements positionally; a count
    mismatch (declarations built in loops or helpers) or a literal-name
    mismatch marks every entry non-editable.  ``scene`` lets a study-backed
    optimization whose study meshes the whole scene report the scene's
    free parameters.  Declaration only: nothing is optimized here.
    """
    statements = locate_optimization_statements(source) or []
    stable_ids = _declaration_ids(source, "optimization")
    aligned = len(statements) == len(optimizations) and all(
        statement.name is None or statement.name == optimization.name
        for statement, optimization in zip(statements, optimizations)
    )
    entries: list[dict[str, Any]] = []
    for index, optimization in enumerate(optimizations):
        statement = statements[index] if aligned else None

        def span(value) -> list[int] | None:
            return list(value) if value is not None else None

        entries.append(
            {
                **optimization.describe(scene),
                "index": index,
                "stableId": stable_ids.get(index) if statement is not None else None,
                "line": statement.statement.lineno if statement is not None else None,
                "span": span(statement.call_span) if statement is not None else None,
                "editable": statement is not None,
                "steps_span": span(statement.steps_span) if statement is not None else None,
                "learning_rate_span": span(statement.learning_rate_span)
                if statement is not None
                else None,
            }
        )
    return entries
