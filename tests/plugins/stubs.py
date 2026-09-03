"""A stand-in provider for the private tier's five kinds.

cadjoint never installs ``diff-brep``, so the seam has to be testable from
the public side with nothing private present.  These five objects are the
smallest thing that satisfies each Protocol in
:mod:`cadjoint.plugins.contracts` — they compute nothing worth having, and
that is the point: if :class:`~cadjoint.plugins.PythonPlugin` can bind one,
resolve the contract's method and report its capabilities, then the seam is
the *interface* and not the implementation.

They are addressed the way a real provider is, by dotted path
(``tests.plugins.stubs:NODE_MAP``), so the registry's import path is
exercised too rather than short-circuited by handing it a live object.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cadjoint.plugins.contracts import CONTRACT_VERSION, DragOutcome, EdgeSet, OwnedNodes

__all__ = ["VERSION", "BREP", "DRAG", "FEATURE_EDGES", "NODE_MAP", "STEP_EXPORT"]

#: What the stand-in reports as its own version.
VERSION = "0.0.0-stub"


class _NodeMap:
    """:class:`~cadjoint.plugins.contracts.NodeMap`: the seeds, unmoved."""

    version = VERSION
    contract_version = CONTRACT_VERSION

    def positions(
        self,
        scene: Any,
        params: Any,
        owned: OwnedNodes,
        *,
        smooth_passes: int = 0,
    ):
        import jax.numpy as jnp

        return jnp.asarray(owned.seeds)


class _FeatureEdges:
    """:class:`~cadjoint.plugins.contracts.FeatureEdges`: one straight segment."""

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
        return EdgeSet(
            polylines=(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),),
            closed=np.array([False]),
            patches=np.array([[0, 1]], dtype=np.int32),
            kind=("traced",),
            residual=np.array([0.0]),
            vertices=np.array([[-1, -1]], dtype=np.int32),
            stats={"curves": 1, "stub": True},
        )


class _BRep:
    """:class:`~cadjoint.plugins.contracts.BRepExtractor`: an opaque nothing."""

    version = VERSION
    contract_version = CONTRACT_VERSION

    def extract(self, scene: Any, grid: Any, **options: Any) -> Any:
        return {"scene": scene, "grid": grid, "options": options, "stub": True}


class _StepExport:
    """:class:`~cadjoint.plugins.contracts.StepExporter`: an empty file."""

    version = VERSION
    contract_version = CONTRACT_VERSION

    def step_export(self, scene: Any, grid: Any, path: Any, **options: Any) -> dict[str, Any]:
        from pathlib import Path

        Path(path).write_text("ISO-10303-21;\nEND-ISO-10303-21;\n", encoding="ascii")
        return {"path": str(path), "faces": {}, "stub": True}


class _Drag:
    """:class:`~cadjoint.plugins.contracts.Drag`: a drag that solved nothing."""

    version = VERSION
    contract_version = CONTRACT_VERSION

    def drag(
        self, scene: Any, brep: Any, handle: str, index: int, target: Any, **options: Any
    ) -> DragOutcome:
        wanted = np.asarray(target, dtype=np.float64).reshape(3)
        return DragOutcome(
            handle=f"{handle}:{index}",
            target=wanted,
            achieved=wanted,
            error=0.0,
            parameters={},
            delta={},
            moved=[],
            constraint_residual=0.0,
            topology_changed=False,
            applied=False,
            reason="stub provider: nothing is solved here",
        )


NODE_MAP = _NodeMap()
FEATURE_EDGES = _FeatureEdges()
BREP = _BRep()
STEP_EXPORT = _StepExport()
DRAG = _Drag()

#: kind -> ``module:attribute``, the shape ``plugins.toml`` and the
#: ``cadjoint.plugins`` entry-point group both accept.
TARGETS = {
    "node_map": "tests.plugins.stubs:NODE_MAP",
    "feature_edges": "tests.plugins.stubs:FEATURE_EDGES",
    "brep": "tests.plugins.stubs:BREP",
    "step_export": "tests.plugins.stubs:STEP_EXPORT",
    "drag": "tests.plugins.stubs:DRAG",
}
