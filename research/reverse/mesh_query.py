"""Axis-aligned ray casting on a triangle soup, with no optional dependencies.

Everything downstream — the inside/outside test the overlay scores against,
and the material intervals the probe measures thicknesses from — is the same
question: *where along this line does the surface sit?*  This module answers
it for lines parallel to a coordinate axis, which is all either caller needs
and is far cheaper than the general case.

Why not ``trimesh.ray``: it wants ``rtree`` for its bounding tree and
``pyembree`` to be fast, and neither is in this repository's environment.  A
uniform bucket grid over the two coordinates the ray does *not* travel along
does the same job in thirty lines of numpy, and a tessellated casting is
exactly the input it suits — a hundred and sixty thousand small triangles
spread thinly over a bounded box.

The parity rule is the usual one and so is its failure mode: a ray that grazes
an edge can be counted twice.  Callers get a lattice, not a single point, and
:func:`inside_lattice` jitters the ray origins off the lattice by an
irrational fraction of a cell so that a plane of the model never lines up with
a plane of rays.  That is a mitigation, not a proof; an odd crossing count is
reported rather than silently rounded, so a caller can see how often it
happened.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["RayGrid", "load_mesh", "save_mesh"]


def load_mesh(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and triangles from an ``.npz`` written by :func:`save_mesh`."""
    data = np.load(path)
    return np.asarray(data["vertices"], dtype=np.float64), np.asarray(data["faces"], dtype=np.int64)


def save_mesh(path: str, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Cache a tessellation so the STEP only has to be read once."""
    np.savez_compressed(
        path, vertices=np.asarray(vertices, np.float32), faces=np.asarray(faces, np.int32)
    )


@dataclass
class RayGrid:
    """A triangle soup bucketed for rays parallel to one coordinate axis.

    Args:
        vertices: ``(n, 3)`` positions.
        faces: ``(m, 3)`` vertex indices.
        axis: The axis rays travel along — 0 for x, 1 for y, 2 for z.
        cell: Bucket size in millimetres, in the two other coordinates.  Four
            is a good default for a casting tessellated at a few tenths: small
            enough that a bucket holds a couple of dozen triangles, large
            enough that a triangle rarely spans more than one.
    """

    vertices: np.ndarray
    faces: np.ndarray
    axis: int = 2
    cell: float = 4.0

    def __post_init__(self) -> None:
        self.u, self.v = (a for a in range(3) if a != self.axis)
        tri = self.vertices[self.faces]  # (m, 3, 3)
        self._tri = tri
        lo = tri[:, :, [self.u, self.v]].min(axis=1)
        hi = tri[:, :, [self.u, self.v]].max(axis=1)
        self.origin = lo.min(axis=0) - self.cell
        self.shape = np.maximum(
            np.ceil((hi.max(axis=0) + self.cell - self.origin) / self.cell).astype(int), 1
        )

        # Every (triangle, bucket) pair its 2D bounding box touches, flattened
        # into one CSR-shaped table: `starts` indexes `members` by bucket.
        ilo = np.floor((lo - self.origin) / self.cell).astype(int)
        ihi = np.floor((hi - self.origin) / self.cell).astype(int)
        spans = (ihi - ilo + 1).prod(axis=1)
        which = np.repeat(np.arange(len(self.faces)), spans)
        offset = np.arange(spans.sum()) - np.repeat(np.cumsum(spans) - spans, spans)
        width = (ihi - ilo + 1)[:, 1]
        cu = ilo[which, 0] + offset // np.repeat(width, spans)
        cv = ilo[which, 1] + offset % np.repeat(width, spans)
        flat = cu * self.shape[1] + cv
        order = np.argsort(flat, kind="stable")
        self._members = which[order]
        self._starts = np.searchsorted(flat[order], np.arange(self.shape.prod() + 1))

    def _bucket(self, p: float, q: float) -> np.ndarray:
        cu = int((p - self.origin[0]) // self.cell)
        cv = int((q - self.origin[1]) // self.cell)
        if not (0 <= cu < self.shape[0] and 0 <= cv < self.shape[1]):
            return np.empty(0, dtype=np.int64)
        flat = cu * self.shape[1] + cv
        return self._members[self._starts[flat] : self._starts[flat + 1]]

    def crossings(self, p: float, q: float) -> np.ndarray:
        """Sorted positions along ``axis`` where the surface crosses this line.

        Args:
            p: Coordinate along the first of the two axes the ray does not
                travel along.
            q: Coordinate along the second.

        Returns:
            The crossing positions, ascending.  An even count means the line
            enters and leaves the solid the same number of times.
        """
        candidates = self._bucket(p, q)
        if not len(candidates):
            return np.empty(0)
        tri = self._tri[candidates]
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        au, av = a[:, self.u] - p, a[:, self.v] - q
        bu, bv = b[:, self.u] - p, b[:, self.v] - q
        cu_, cv_ = c[:, self.u] - p, c[:, self.v] - q
        # Barycentric via the three signed sub-areas of the projected triangle.
        w0 = bu * cv_ - bv * cu_
        w1 = cu_ * av - cv_ * au
        w2 = au * bv - av * bu
        total = w0 + w1 + w2
        keep = (np.abs(total) > 1e-12) & (
            ((w0 >= 0) & (w1 >= 0) & (w2 >= 0)) | ((w0 <= 0) & (w1 <= 0) & (w2 <= 0))
        )
        if not keep.any():
            return np.empty(0)
        w0, w1, w2, total = w0[keep], w1[keep], w2[keep], total[keep]
        za = a[keep][:, self.axis]
        zb = b[keep][:, self.axis]
        zc = c[keep][:, self.axis]
        return np.sort((w0 * za + w1 * zb + w2 * zc) / total)

    def runs(self, p: float, q: float) -> list[tuple[float, float]]:
        """The solid intervals along the line, as ``(enter, leave)`` pairs."""
        t = self.crossings(p, q)
        return [(float(t[i]), float(t[i + 1])) for i in range(0, len(t) - 1, 2)]

    def inside_lattice(
        self, us: np.ndarray, vs: np.ndarray, ws: np.ndarray, jitter: float = 0.31830988
    ) -> tuple[np.ndarray, int]:
        """Inside/outside for the lattice ``us x vs x ws``, one ray per column.

        ``us`` and ``vs`` run along the two axes the rays do not travel; ``ws``
        runs along :attr:`axis`.  One ray serves a whole column, which is what
        makes a lattice of a few hundred thousand points affordable.

        Args:
            us: Sample coordinates on the first cross axis.
            vs: Sample coordinates on the second cross axis.
            ws: Sample coordinates along the ray axis.
            jitter: Fraction of a lattice step to offset the ray origins by, so
                that a face of the model coplanar with a plane of samples does
                not decide every ray on that plane at once.  The default is
                1/pi, which lines up with nothing.

        Returns:
            A boolean array shaped ``(len(us), len(vs), len(ws))``, and the
            number of columns whose crossing count came back odd — surface
            hits the parity rule could not pair, which is the honest measure of
            how much to trust the rest.
        """
        du = float(us[1] - us[0]) if len(us) > 1 else 0.0
        dv = float(vs[1] - vs[0]) if len(vs) > 1 else 0.0
        out = np.zeros((len(us), len(vs), len(ws)), dtype=bool)
        odd = 0
        for i, u in enumerate(us):
            for j, v in enumerate(vs):
                t = self.crossings(float(u) + jitter * du, float(v) + jitter * dv)
                if len(t) % 2:
                    odd += 1
                    t = t[: len(t) - 1]
                if len(t):
                    out[i, j] = np.searchsorted(t, ws) % 2 == 1
        return out, odd
