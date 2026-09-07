"""A refresh: the same topology, re-solved for a new design.

The protocol's rule (``research/zero-set.proto``): a topology found once
can be solved again at another θ, seeded with the previous solution, and
the result is certified the same way — residual within tolerance, incidence
unchanged.  An :class:`Overlay` holds exactly that for an extraction's
points: which surfaces each lies on, and where it last was.  Asked for a
new θ it returns the moved points, and how many it could not vouch for —
the caller's cue to extract again rather than trust the refresh.

Any extraction can seed one: the points just have to lie on the boundary.
The classification (:meth:`Projector.classify`) names their surfaces by
ownership, which is all a solve needs to hold them where they are.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cadjoint.zeroset.gpu import Projector
from cadjoint.zeroset.table import Model

__all__ = ["Overlay", "Refreshed", "theta_from_values"]


def theta_from_values(model: Model, values: dict[str, list[float] | float]) -> np.ndarray:
    """The table's θ with ``values`` (by free-parameter name) written over it.

    A name is a whole parameter — ``"o"`` sets ``o[0..2]`` — and a name the
    table does not know is ignored, so a caller can pass every slot a shader
    program has without first asking which ones the table reads.
    """
    theta = np.asarray(model.theta, np.float64).copy()
    slots: dict[str, list[int]] = {}
    for i, name in enumerate(model.names):
        slots.setdefault(name.split("[", 1)[0], []).append(i)
    for name, value in values.items():
        indices = slots.get(name)
        if indices is None:
            continue
        flat = np.atleast_1d(np.asarray(value, np.float64)).ravel()
        if flat.size < len(indices):
            raise ValueError(f"{name}: {flat.size} components for {len(indices)} entries")
        theta[indices] = flat[: len(indices)]
    return theta


@dataclass(frozen=True)
class Refreshed:
    points: np.ndarray  #: (N, 3) — a point the solve could not certify keeps its last position
    certified: np.ndarray  #: (N,) bool
    theta: np.ndarray

    @property
    def stale(self) -> float:
        """The fraction of points not certified: 0 means the refresh is the truth."""
        return float(1.0 - self.certified.mean()) if len(self.certified) else 0.0


class Overlay:
    """Boundary points with their surfaces, movable to any θ at fixed topology.

    Args:
        model: The table the points were extracted from.
        points: (N, 3) points on the boundary at ``model.theta``.
        tolerance: Field units: how far off a surface a point may sit and
            still be classified onto it, and the residual a solve must
            reach to be certified.  A lattice extraction's vertices are
            re-solved to well under a thousandth of a cell.
    """

    def __init__(self, model: Model, points: np.ndarray, *, tolerance: float = 2e-3) -> None:
        self.model = model
        self.tolerance = tolerance
        self.projector = Projector(model)
        self.theta = np.asarray(model.theta, np.float64)
        pts = np.asarray(points, np.float64).reshape(-1, 3)
        self.incidence = self._classify(self.theta, pts, real=None)
        # snap: the seeds land on their surfaces exactly, so a later refresh
        # measures the design's motion and not the extraction's slack
        snapped = self.projector.solve(self.theta, pts, self.incidence, tolerance=1e-6)
        good = np.isfinite(snapped.points).all(axis=1) & (snapped.residual <= tolerance)
        self.points = np.where(good[:, None], snapped.points, pts)
        self.placed = good | np.array([len(i) == 0 for i in self.incidence])

    @property
    def surfaces(self) -> int:
        return len(self.projector.surfaces)

    def _classify(
        self, theta: np.ndarray, points: np.ndarray, real: set[int] | None
    ) -> list[list[int]]:
        """Ownership on the GPU, then the census's global rule.

        A band that owns no point of the extraction has no boundary of its
        own — an exact box's extent blend is one: its transition is positive
        only off the surface — so its rim and folds are not surfaces here,
        whatever their values say at a point.  The first classification
        decides which bands are real; a refresh reuses the decision, so the
        certificate compares like with like.
        """
        kinds = [k for k, _w, _x in self.projector.surfaces]
        found = self.projector.classify(theta, points, tolerance=self.tolerance)
        if real is None:
            real = {s for row in found for s in row if kinds[s] == "band"}
            self.real_bands = real
        band_of = {}
        band = None
        for s, kind in enumerate(kinds):
            if kind == "band":
                band = s
            band_of[s] = band
        return [
            [s for s in row if kinds[s] in ("patch", "band") or band_of[s] in real][:3]
            for row in found
        ]

    def refresh(self, theta: np.ndarray, *, certify: bool = True) -> Refreshed:
        """Solve every point at ``theta`` from where it last was.

        Args:
            theta: The new design.
            certify: Re-classify at the solved points and require the
                incidence to be unchanged — the protocol's certificate.
                A live drag can skip it and let the release pay.
        """
        theta = np.asarray(theta, np.float64)
        solved = self.projector.solve(theta, self.points, self.incidence, tolerance=1e-6)
        ok = np.isfinite(solved.points).all(axis=1) & (solved.residual <= self.tolerance)
        if certify:
            named = self._classify(theta, solved.points, real=self.real_bands)
            same = np.array(
                [sorted(a) == sorted(b) for a, b in zip(named, self.incidence)], dtype=bool
            )
            ok &= same
        ok &= self.placed
        points = np.where(ok[:, None], solved.points, self.points)
        self.points, self.theta = points, theta
        return Refreshed(points=points, certified=ok, theta=theta)
