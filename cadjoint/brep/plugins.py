"""The five private kinds, filled from this tree — temporary.

While the derived B-rep still lives in ``cadjoint/brep``, this module is
what the registry's ``python`` specs point at
(:data:`cadjoint.plugins.registry.BUILTIN_PYTHON`), so the public tier's
seams — the viewer's overlay, the exporter, the optimizer's node map — are
exercised against the real components before anything moves
(``research/two-tier.md`` D12).  When the B-rep moves to ``diff-brep`` this
module goes with it as ``diff_brep.plugins`` and the same objects are
registered through the ``cadjoint.plugins`` entry-point group; nothing
public changes.

Each object satisfies one Protocol of :mod:`cadjoint.plugins.contracts`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

import numpy as np

from cadjoint.plugins.contracts import CONTRACT_VERSION, DragOutcome, EdgeSet, OwnedNodes

__all__ = ["VERSION", "brep", "drag", "feature_edges", "node_map", "step_export"]

#: What the in-tree providers report as their version.
VERSION = "0.1.0"


class _NodeMap:
    """:class:`~cadjoint.plugins.contracts.NodeMap` over :func:`cadjoint.brep.mesh_gmsh.node_positions`."""

    version = VERSION
    contract_version = CONTRACT_VERSION

    def positions(
        self,
        scene: Any,
        params: Mapping[str, Any],
        owned: OwnedNodes,
        *,
        smooth_passes: int = 0,
    ):
        from cadjoint.brep.mesh_gmsh import node_positions

        return node_positions(scene, params, owned, smooth_passes=smooth_passes)


class _FeatureEdges:
    """:class:`~cadjoint.plugins.contracts.FeatureEdges` over :func:`cadjoint.brep.edges.feature_edges`."""

    version = VERSION
    contract_version = CONTRACT_VERSION

    def feature_edges(
        self,
        scene: Any,
        grid: Any,
        *,
        design_leaves: np.ndarray | None = None,
        blend_tolerance: float | None = None,
        steps: int = 4,
    ) -> EdgeSet:
        from cadjoint.brep.edges import feature_edges

        return feature_edges(
            scene, grid, design_leaves=design_leaves, blend_tolerance=blend_tolerance, steps=steps
        )


class _BRep:
    """:class:`~cadjoint.plugins.contracts.BRepExtractor` over :func:`cadjoint.brep.extract_brep`."""

    version = VERSION
    contract_version = CONTRACT_VERSION

    def extract(self, scene: Any, grid: Any, **options: Any) -> Any:
        from cadjoint.brep.graph import extract_brep

        return extract_brep(scene, grid, **options)


class _StepExport:
    """:class:`~cadjoint.plugins.contracts.StepExporter` over :func:`cadjoint.brep.step.save_brep_step`."""

    version = VERSION
    contract_version = CONTRACT_VERSION

    def step_export(self, scene: Any, grid: Any, path: Any, **options: Any) -> dict[str, Any]:
        from cadjoint.brep.graph import extract_brep
        from cadjoint.brep.step import save_brep_step

        extracted = extract_brep(scene, grid)
        return save_brep_step(extracted, path, **options)


class _Drag:
    """:class:`~cadjoint.plugins.contracts.Drag` over :func:`cadjoint.brep.drag.drag_handle`."""

    version = VERSION
    contract_version = CONTRACT_VERSION

    def drag(
        self, scene: Any, brep: Any, handle: str, index: int, target: Any, **options: Any
    ) -> DragOutcome:
        from cadjoint.brep.drag import drag_handle

        result = drag_handle(scene, brep, handle, index, target, **options)
        return DragOutcome(**asdict(result))


node_map = _NodeMap()
feature_edges = _FeatureEdges()
brep = _BRep()
step_export = _StepExport()
drag = _Drag()
