"""The overlay store: a mesh answer's seed becomes a refreshable overlay."""

from __future__ import annotations

import numpy as np
import pytest

from cadjoint.viewer._overlay_refresh import OverlayStore

pytest.importorskip("wgpu")

SOURCE = """
from cadjoint.geometry.parameters import Vector
from cadjoint.sdf.primitives import Box
scene = Box(size=Vector([0.5, 0.4, 0.3], free=True, name="s"))
"""


def _seed():
    from cadjoint.viewer._edge_overlay import _mesh_edge_layers
    from cadjoint.viewer._overlay_refresh import refresh_seed
    from cadjoint.viewer.worker.scene import _execute_scene

    scene = _execute_scene(SOURCE)["scene"]
    vertices, wire, sharp, edges = _mesh_edge_layers(scene)
    return refresh_seed(scene, vertices, wire, sharp, 64, edges)


def test_absorb_keeps_the_seed_out_of_the_answer_and_refreshes_from_it():
    store = OverlayStore()
    seed = _seed()
    answer = store.absorb(SOURCE, {"ok": True, "mesh_edges": {"wire": []}, "refresh": seed})
    assert "refresh" not in answer and answer["ok"]
    assert store.has(SOURCE)
    moved = store.refresh({"source": SOURCE, "values": {"s": [0.7, 0.5, 0.45]}})
    assert moved["ok"], moved
    assert moved["stale"] < 0.02
    wire = np.asarray(moved["mesh_edges"]["wire"], float).reshape(-1, 3)
    assert len(wire) == 2 * len(seed["wire"])
    # the box grew: its overlay now reaches the new half-extents and no further
    np.testing.assert_allclose(np.abs(wire).max(axis=0), [0.7, 0.5, 0.45], atol=2e-3)
    assert moved["mesh_edges"]["resolution"] == 64


def test_refresh_without_an_overlay_says_so():
    store = OverlayStore()
    assert store.refresh({"source": "x = 1", "values": {}})["ok"] is False
    assert store.refresh({"source": 3, "values": {}})["ok"] is False
    assert store.absorb("x = 1", {"ok": False, "error": "boom"}) == {"ok": False, "error": "boom"}


def test_the_store_keeps_only_the_newest_programs():
    store = OverlayStore(keep=1)
    seed = _seed()
    store.absorb("a", {"refresh": dict(seed)})
    store.absorb("b", {"refresh": dict(seed)})
    assert not store.has("a") and store.has("b")
