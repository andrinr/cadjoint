"""Refreshing the mesh-edge overlay at fixed topology, between extractions.

A ``/api/mesh`` extraction is the expensive half of the overlay: dual
contouring, classification, chains.  Its result is a set of boundary points
and how they join, and for a values-only edit — a handle drag, a slider —
that topology is exactly what the protocol's refresh rule keeps
(``research/zero-set.proto``): the same surfaces, re-solved at the new θ,
seeded from where the points were.  :class:`cadjoint.zeroset.refresh.Overlay`
does the solving on the GPU; this module is the plumbing that gets an
extraction's points into one and a refreshed payload back out.

Two halves, on the two sides of the worker boundary:

* :func:`refresh_seed` runs in the worker next to the extraction and lowers
  the scene to its node table, so the server never executes user code.
* :class:`OverlayStore` lives in the server.  It absorbs the seed from a
  ``/api/mesh`` answer, builds the overlay on a background thread (the
  pipeline compile is seconds on a big scene), and answers
  ``/api/mesh_refresh`` with re-solved segments plus how many points the
  certificate refused — above a few in a thousand the client should extract
  again rather than trust the picture.

Optional throughout: no wgpu, no GPU, or a scene the table cannot express,
and every refresh answers ``ok: false`` so the client falls back to a full
extraction.  The overlay is a faster path, never the only one.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

import numpy as np

__all__ = ["OverlayStore", "OVERLAYS", "refresh_seed"]

#: Overlays kept per server: the current program and the one before it,
#: which is what an undo lands on.  Each holds a compiled GPU pipeline.
KEEP = 2

#: How long a refresh waits for a build still in progress before giving up.
BUILD_WAIT_SECONDS = 30.0


def refresh_seed(
    scene: Any,
    vertices: np.ndarray,
    wire_edges: np.ndarray,
    sharp: np.ndarray,
    resolution: int,
    edges: str,
) -> dict[str, Any]:
    """Everything a server needs to refresh this extraction: worker side.

    Args:
        scene: Root SDF node, already executed.
        vertices: (N, 3) re-solved dual vertices.
        wire_edges: (E, 2) index pairs into ``vertices``.
        sharp: (M, 2, 3) feature-curve chords.
        resolution: The overlay grid's resolution, echoed into payloads.
        edges: Which layer produced ``sharp`` (``"graph"`` or ``"lattice"``).
    """
    from cadjoint.zeroset import lower

    model = lower(scene)
    return {
        "model": model.to_dict(),
        "names": list(model.names),
        "vertices": np.round(np.asarray(vertices, float), 6).tolist(),
        "wire": np.asarray(wire_edges, int).tolist(),
        "sharp": np.round(np.asarray(sharp, float).reshape(-1, 2, 3), 6).tolist(),
        "resolution": int(resolution),
        "edges": edges,
    }


def _segments(pairs: np.ndarray) -> list[list[list[float]]]:
    return [[[round(float(v), 3) for v in point] for point in pair] for pair in pairs]


class _Entry:
    """One program's overlay: the seed, then the built overlay or why not."""

    def __init__(self, seed: dict[str, Any]) -> None:
        self.seed = seed
        self.ready = threading.Event()
        self.lock = threading.Lock()
        self.overlay: Any = None
        self.error: str | None = None
        self.vertex_count = len(seed["vertices"])
        self.wire = np.asarray(seed["wire"], int).reshape(-1, 2)

    def build(self) -> None:
        try:
            from cadjoint.zeroset.refresh import Overlay
            from cadjoint.zeroset.table import Model

            m = self.seed["model"]
            model = Model(
                nodes=m["node"], root=m["root"], theta=m["theta"], names=self.seed["names"]
            )
            points = np.concatenate(
                [
                    np.asarray(self.seed["vertices"], float).reshape(-1, 3),
                    np.asarray(self.seed["sharp"], float).reshape(-1, 3),
                ]
            )
            self.overlay = Overlay(model, points)
        except Exception as error:  # noqa: BLE001 - an overlay that cannot be built is reported, not raised
            self.error = f"{type(error).__name__}: {error}"
        finally:
            self.ready.set()

    def refresh(self, values: dict[str, Any], certify: bool) -> dict[str, Any]:
        from cadjoint.zeroset.refresh import theta_from_values

        if not self.ready.wait(BUILD_WAIT_SECONDS):
            return {"ok": False, "error": "The overlay is still being built."}
        if self.overlay is None:
            return {"ok": False, "error": f"No refreshable overlay: {self.error}"}
        with self.lock:
            moved = self.overlay.refresh(
                theta_from_values(self.overlay.model, values), certify=certify
            )
        vertices = moved.points[: self.vertex_count]
        sharp = moved.points[self.vertex_count :].reshape(-1, 2, 3)
        return {
            "ok": True,
            "mesh_edges": {
                "wire": _segments(vertices[self.wire]),
                "sharp": _segments(sharp),
                "resolution": self.seed["resolution"],
                "edges": self.seed["edges"],
            },
            "stale": moved.stale,
        }


class OverlayStore:
    """The server's overlays, keyed by the program they were extracted from."""

    def __init__(self, keep: int = KEEP) -> None:
        self._keep = keep
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, _Entry] = OrderedDict()

    def absorb(self, source: Any, result: dict[str, Any]) -> dict[str, Any]:
        """Take the seed out of a ``/api/mesh`` answer and start building on it.

        Returns the answer without the seed, which is what the client gets:
        the seed is server state, not payload.
        """
        seed = result.pop("refresh", None) if isinstance(result, dict) else None
        if seed is None or not isinstance(source, str):
            return result
        entry = _Entry(seed)
        with self._lock:
            self._entries.pop(source, None)
            self._entries[source] = entry
            while len(self._entries) > self._keep:
                self._entries.popitem(last=False)
        threading.Thread(target=entry.build, name="overlay-build", daemon=True).start()
        return result

    def refresh(self, payload: dict[str, Any]) -> dict[str, Any]:
        """``/api/mesh_refresh``: ``{source, values, certify?}`` → refreshed segments.

        ``source`` is the program the overlay was extracted from — the base
        of a values-only edit, not the edit — and ``values`` maps
        free-parameter names to their new values, whole parameters at a time.
        """
        source = payload.get("source")
        values = payload.get("values")
        if not isinstance(source, str) or not isinstance(values, dict):
            return {"ok": False, "error": "A refresh needs the base `source` and a `values` map."}
        with self._lock:
            entry = self._entries.get(source)
        if entry is None:
            return {"ok": False, "error": "No overlay has been extracted for this program."}
        try:
            return entry.refresh(values, bool(payload.get("certify", True)))
        except Exception as error:  # noqa: BLE001 - the client falls back to extraction
            return {"ok": False, "error": f"{type(error).__name__}: {error}"}

    def has(self, source: str) -> bool:
        with self._lock:
            return source in self._entries


#: The server's store.  Module-level because the routing table is a module-
#: level function and a playground is one process.
OVERLAYS = OverlayStore()
