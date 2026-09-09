"""Measure a machined casting from its STEP file.

The premise is that a production STEP is already a measurement, not a picture:
CATIA wrote its bores as ``B_SPLINE_SURFACE`` only where they had to be, and
everything a machinist actually dimensioned — a bore, a drilling, a face — is
still an analytic surface with an exact radius and an exact axis.  Walking the
faces and sorting them by surface type therefore recovers the drawing, and no
part of that answer is a guess.

What the walk gives directly:

* ``GeomAbs_Cylinder`` → an axis line and a radius.  Cluster by the line and
  the bores, the head bolts, the oil gallery and every drilling separate
  themselves out; the *spacing between the bore axes* is the bore pitch, with
  no inference at all.
* ``GeomAbs_Plane`` → a normal and an offset.  The extreme plane normal to the
  bore axis is the deck; the extreme one the other way is the pan rail.

What it does not give is anything the casting only *implies*: a wall thickness,
the height of a jacket floor, whether a saddle was line-bored or left as cast.
Those come from the tessellation, by casting axis-aligned rays through it and
reading the solid intervals off the line (:mod:`research.reverse.mesh_query`).
Both halves are reported, and which half a number came from is marked, because
a radius read off a B-rep face is exact and a thickness read off a 0.35 mm
tessellation is not.

Usage::

    python -m research.reverse.step_probe "/path/to/part.stp" [--json out.json]
        [--mesh-cache cache.npz] [--deflection 0.35]

The frame it reports is derived, never assumed: the bore axis is the direction
carrying the most large-radius cylindrical area, the crank axis is fitted to
the saddle arches, and the origin is put on the crank axis under the middle of
the bore row.  A CAD part parked at an arbitrary place in its own file — which
is the normal case — comes out in a frame a modeller can actually use.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# The surface-type enum, in OpenCascade's order.
SURFACE_KINDS = (
    "plane",
    "cylinder",
    "cone",
    "sphere",
    "torus",
    "bezier",
    "bspline",
    "revolution",
    "extrusion",
    "offset",
    "other",
)


@dataclass
class Face:
    """One face of the solid, reduced to what a measurement needs."""

    kind: str
    area: float
    bounds: tuple[float, float, float, float, float, float]
    radius: float | None = None
    location: tuple[float, float, float] | None = None
    direction: tuple[float, float, float] | None = None


@dataclass
class Part:
    """A STEP file's single solid, read once."""

    faces: list[Face]
    bounds: np.ndarray
    volume: float
    centre_of_mass: tuple[float, float, float]
    vertices: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    triangles: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.int64))


# ── reading ──────────────────────────────────────────────────────────────────


def read_step(path: str, deflection: float = 0.35, mesh_cache: str | None = None) -> Part:
    """Read a STEP file into faces and a tessellation.

    Args:
        path: The ``.stp`` / ``.step`` file.
        deflection: Chordal tolerance for the tessellation, in model units.
        mesh_cache: Optional ``.npz`` to read the tessellation from, or write
            it to.  Reading a 30 MB STEP takes about four seconds and
            tessellating it about one; caching makes an iterate-and-compare
            loop pleasant.

    Returns:
        The :class:`Part`.

    Raises:
        ImportError: If OCP is not installed.  It is a heavy optional
            dependency and nothing in ``cadjoint/`` needs it, which is why
            this module lives under ``research/``.
    """
    try:
        from OCP.Bnd import Bnd_Box
        from OCP.BRep import BRep_Tool
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.BRepBndLib import BRepBndLib
        from OCP.BRepGProp import BRepGProp
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.GProp import GProp_GProps
        from OCP.STEPControl import STEPControl_Reader
        from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopLoc import TopLoc_Location
        from OCP.TopoDS import TopoDS
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "research/reverse needs OpenCascade's Python bindings: pip install cadquery-ocp"
        ) from exc

    reader = STEPControl_Reader()
    reader.ReadFile(path)
    reader.TransferRoots()
    shape = reader.OneShape()

    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    bounds = np.array(box.Get(), dtype=float).reshape(2, 3)

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    com = props.CentreOfMass()

    cached = mesh_cache is not None and os.path.exists(mesh_cache)
    if not cached:
        BRepMesh_IncrementalMesh(shape, deflection, False, 0.5, True)

    faces: list[Face] = []
    verts: list[np.ndarray] = []
    tris: list[np.ndarray] = []
    base = 0
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        topo = TopoDS.Face_s(explorer.Current())
        adaptor = BRepAdaptor_Surface(topo)
        index = int(adaptor.GetType())
        kind = SURFACE_KINDS[index] if index < len(SURFACE_KINDS) else "other"

        area = GProp_GProps()
        BRepGProp.SurfaceProperties_s(topo, area)
        fbox = Bnd_Box()
        BRepBndLib.Add_s(topo, fbox)

        entry = Face(kind=kind, area=area.Mass(), bounds=tuple(fbox.Get()))
        analytic = None
        if kind == "cylinder":
            analytic = adaptor.Cylinder()
            entry.radius = analytic.Radius()
        elif kind == "cone":
            analytic = adaptor.Cone()
            entry.radius = analytic.RefRadius()
        elif kind == "plane":
            analytic = adaptor.Plane()
        if analytic is not None:
            axis = analytic.Axis()
            loc, direction = axis.Location(), axis.Direction()
            entry.location = (loc.X(), loc.Y(), loc.Z())
            entry.direction = (direction.X(), direction.Y(), direction.Z())
        faces.append(entry)

        if not cached:
            location = TopLoc_Location()
            tri = BRep_Tool.Triangulation_s(topo, location)
            if tri is not None:
                transform = location.Transformation()
                n = tri.NbNodes()
                pts = np.empty((n, 3))
                for i in range(1, n + 1):
                    p = tri.Node(i).Transformed(transform)
                    pts[i - 1] = (p.X(), p.Y(), p.Z())
                idx = np.empty((tri.NbTriangles(), 3), dtype=np.int64)
                for i in range(1, tri.NbTriangles() + 1):
                    a, b, c = tri.Triangle(i).Get()
                    idx[i - 1] = (a - 1, b - 1, c - 1)
                if topo.Orientation() == TopAbs_REVERSED:
                    idx = idx[:, ::-1]
                verts.append(pts)
                tris.append(idx + base)
                base += n
        explorer.Next()

    if cached:
        from research.reverse.mesh_query import load_mesh

        vertices, triangles = load_mesh(mesh_cache)
    else:
        vertices = np.concatenate(verts) if verts else np.empty((0, 3))
        triangles = np.concatenate(tris) if tris else np.empty((0, 3), dtype=np.int64)
        if mesh_cache:
            from research.reverse.mesh_query import save_mesh

            save_mesh(mesh_cache, vertices, triangles)

    return Part(
        faces=faces,
        bounds=bounds,
        volume=props.Mass(),
        centre_of_mass=(com.X(), com.Y(), com.Z()),
        vertices=vertices,
        triangles=triangles,
    )


# ── geometry helpers ─────────────────────────────────────────────────────────


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def canonical(direction) -> np.ndarray:
    """A line's direction with a sign convention, so +Z and -Z are one axis."""
    d = unit(direction)
    return d if d[int(np.argmax(np.abs(d)))] > 0 else -d


def axis_identity(location, direction, decimals: int = 1):
    """A hashable identity for the infinite line through ``location``."""
    d = canonical(direction)
    p = np.asarray(location, dtype=float)
    foot = p - float(np.dot(p, d)) * d
    return tuple(np.round(d, 3)), tuple(np.round(foot, decimals))


def cylinder_axes(faces: list[Face], decimals: int = 1) -> list[dict[str, Any]]:
    """Cylindrical faces grouped onto the lines they turn about."""
    groups: dict[Any, list[Face]] = collections.defaultdict(list)
    for f in faces:
        if f.kind == "cylinder":
            groups[axis_identity(f.location, f.direction, decimals)].append(f)
    out = []
    for (direction, foot), members in groups.items():
        bb = np.array([m.bounds for m in members])
        out.append(
            {
                "direction": np.asarray(direction, dtype=float),
                "foot": np.asarray(foot, dtype=float),
                "count": len(members),
                "radii": sorted({round(m.radius, 3) for m in members}),
                "area": float(sum(m.area for m in members)),
                "dominant_radius": max(
                    {round(m.radius, 3) for m in members},
                    key=lambda r: sum(f.area for f in members if round(f.radius, 3) == r),
                ),
                "radius_spans": {
                    round(r, 3): [
                        float(min(f.bounds[i] for f in members if round(f.radius, 3) == r))
                        for i in range(3)
                    ]
                    + [
                        float(max(f.bounds[3 + i] for f in members if round(f.radius, 3) == r))
                        for i in range(3)
                    ]
                    for r in {round(m.radius, 3) for m in members}
                },
                "low": bb[:, :3].min(axis=0),
                "high": bb[:, 3:].max(axis=0),
            }
        )
    return sorted(out, key=lambda g: -g["area"])


def fit_circle(points: np.ndarray) -> tuple[float, float, float]:
    """Least-squares circle through 2D points; returns ``(cx, cy, radius)``."""
    x, y = points[:, 0], points[:, 1]
    a = np.stack([x, y, np.ones_like(x)], axis=1)
    sol, *_ = np.linalg.lstsq(a, x**2 + y**2, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    return float(cx), float(cy), float(math.sqrt(sol[2] + cx**2 + cy**2))


# ── the census ───────────────────────────────────────────────────────────────


def census(part: Part) -> dict[str, Any]:
    """How many faces of each surface type, and how much area each carries."""
    counts: collections.Counter[str] = collections.Counter()
    areas: collections.Counter[str] = collections.Counter()
    for f in part.faces:
        counts[f.kind] += 1
        areas[f.kind] += f.area
    return {
        "faces": len(part.faces),
        "by_kind": {
            k: {"count": counts[k], "area_cm2": areas[k] / 100.0} for k, _ in counts.most_common()
        },
    }


def radius_census(part: Part, tol: float = 0.05) -> list[dict[str, Any]]:
    """Distinct cylinder radii, with how many faces and which axes carry them."""
    cyls = [f for f in part.faces if f.kind == "cylinder"]
    if not cyls:
        return []
    radii = np.array([f.radius for f in cyls])
    order = np.argsort(radii)
    groups: list[list[int]] = []
    current = [int(order[0])]
    for i in order[1:]:
        if radii[i] - radii[current[-1]] < tol:
            current.append(int(i))
        else:
            groups.append(current)
            current = [int(i)]
    groups.append(current)
    out = []
    for g in groups:
        dirs: collections.Counter = collections.Counter()
        for i in g:
            dirs[tuple(np.round(canonical(cyls[i].direction), 3))] += 1
        out.append(
            {
                "radius": float(radii[g].mean()),
                "count": len(g),
                "area_cm2": float(sum(cyls[i].area for i in g) / 100.0),
                "directions": [{"dir": list(d), "count": n} for d, n in dirs.most_common()],
            }
        )
    return sorted(out, key=lambda r: -r["count"])


# ── frame and dimensions ─────────────────────────────────────────────────────


def find_bore_row(part: Part, min_radius: float = 15.0) -> dict[str, Any] | None:
    """The bores: the largest family of equal-radius, parallel, collinear axes.

    A cylinder block's bores are the only feature that is at once big, repeated
    on one straight line, and cut to the same radius; nothing else in a casting
    is all three.  Finding them is what fixes the frame, because their common
    direction is the bore axis and the line their centres lie on is parallel to
    the crank.
    """
    axes = [g for g in cylinder_axes(part.faces) if max(g["radii"]) >= min_radius]
    families: dict[Any, list[dict]] = collections.defaultdict(list)
    for g in axes:
        # The bore is the radius that carries the *area* on this axis, not the
        # largest one: a bore's counterbore and its liner seat are coaxial with
        # it and wider, but they are millimetres tall and the bore is not.
        families[(tuple(np.round(g["direction"], 3)), round(g["dominant_radius"], 2))].append(g)
    best = None
    for (direction, radius), members in families.items():
        if len(members) < 2:
            continue
        d = np.asarray(direction, dtype=float)
        feet = np.array([m["foot"] for m in members])
        centred = feet - feet.mean(axis=0)
        # Collinear if one singular value dominates the other two.
        _, sv, vh = np.linalg.svd(centred)
        if sv[0] < 1e-6 or (len(sv) > 1 and sv[1] > 0.02 * sv[0]):
            continue
        along = vh[0]
        t = np.sort(centred @ along)
        spacing = np.diff(t)
        score = len(members) * radius * sum(m["area"] for m in members)
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "axis": d,
                "row_direction": canonical(along),
                "radius": float(radius),
                "count": len(members),
                "centres": feet[np.argsort(feet @ canonical(along))],
                "spans": [m["radius_spans"] for m in members],
                "pitch": float(np.median(spacing)) if len(spacing) else 0.0,
                "pitch_spread": float(spacing.max() - spacing.min()) if len(spacing) else 0.0,
                "radii": sorted({r for m in members for r in m["radii"]}),
            }
    return best


def dominant_planes(part: Part, normal, min_area: float = 2000.0) -> list[dict[str, Any]]:
    """Planes facing along ``normal``, largest first, with their offsets."""
    n = canonical(normal)
    out = []
    for f in part.faces:
        if f.kind != "plane" or f.area < min_area:
            continue
        if abs(float(np.dot(canonical(f.direction), n))) < 0.999:
            continue
        out.append(
            {
                "offset": float(np.dot(np.asarray(f.location, dtype=float), n)),
                "area_cm2": f.area / 100.0,
                "bounds": [round(v, 2) for v in f.bounds],
            }
        )
    return sorted(out, key=lambda p: -p["area_cm2"])


def probe(path: str, deflection: float = 0.35, mesh_cache: str | None = None) -> dict[str, Any]:
    """Read a STEP and report everything this module knows how to measure."""
    from research.reverse.mesh_query import RayGrid

    part = read_step(path, deflection=deflection, mesh_cache=mesh_cache)
    size = part.bounds[1] - part.bounds[0]
    report: dict[str, Any] = {
        "file": path,
        "bounds": {"low": part.bounds[0].tolist(), "high": part.bounds[1].tolist()},
        "size_mm": size.tolist(),
        "volume_cm3": part.volume / 1000.0,
        "mass_kg_at_7200": part.volume * 7.2e-6,
        "centre_of_mass": list(part.centre_of_mass),
        "census": census(part),
        "radii": radius_census(part),
    }

    row = find_bore_row(part)
    if row is None:
        report["frame"] = None
        return report

    bore_axis = canonical(row["axis"])
    row_axis = canonical(row["row_direction"])
    cross = canonical(np.cross(bore_axis, row_axis))
    ia = int(np.argmax(np.abs(bore_axis)))
    ir = int(np.argmax(np.abs(row_axis)))
    ic = int(np.argmax(np.abs(cross)))

    centres = row["centres"]
    row_positions = centres @ row_axis
    across = float(np.mean(centres @ cross))

    grid_bore = RayGrid(part.vertices, part.triangles, axis=ia)
    grid_row = RayGrid(part.vertices, part.triangles, axis=ir)
    grid_cross = RayGrid(part.vertices, part.triangles, axis=ic)

    def bore_runs(row_pos: float) -> list[tuple[float, float]]:
        p, q = (across, row_pos) if ic < ir else (row_pos, across)
        return grid_bore.runs(p, q)

    # The deck: the outermost plane normal to the bore axis on the side the
    # bores open to. A bore is open at the deck, so the ray down a bore axis
    # finds no material at all; the deck is the extreme large plane.
    planes = dominant_planes(part, bore_axis)
    deck = max(planes, key=lambda p: p["offset"])["offset"]
    rail = min(planes, key=lambda p: p["offset"])["offset"]

    # How far each coaxial radius runs along the bore axis, read straight off
    # the B-rep faces rather than off the tessellation: this is what says which
    # of the radii sharing a bore axis is the bore and which are its
    # counterbore and its liner seat.
    coaxial: dict[float, list[float]] = {}
    for spans in row["spans"]:
        for radius, box in spans.items():
            lo, hi = float(np.asarray(box[:3]) @ bore_axis), float(np.asarray(box[3:]) @ bore_axis)
            lo, hi = min(lo, hi), max(lo, hi)
            if radius in coaxial:
                coaxial[radius] = [min(coaxial[radius][0], lo), max(coaxial[radius][1], hi)]
            else:
                coaxial[radius] = [lo, hi]
    bore_span = coaxial[round(row["radius"], 3)]
    bore_top, bore_bottom = [bore_span[1]], [bore_span[0]]

    # The main bearing saddle. At a bulkhead station, a ray across the block
    # at a given height meets the web, then the saddle opening, then the web
    # again; the half-width of that opening as a function of height traces the
    # saddle, and a circle through it has the crank axis for a centre.
    #
    # Two things make this the reliable way to find a crank axis rather than
    # looking for a cylindrical face. A cast saddle is not a cylinder in the
    # B-rep at all — this block's is spline surfaces — and a line-bored one is
    # a half cylinder whose axis a face walk finds but cannot tell apart from
    # any other drilling. The opening's own shape is unambiguous.
    saddle = None
    if len(row_positions) >= 2:
        stations = (row_positions[:-1] + row_positions[1:]) / 2.0
        step = 0.5
        levels = np.arange(rail + 2.0, (deck + rail) / 2.0, step)
        widths: list[tuple[float, float]] = []
        for level in levels:
            halves = []
            for station in stations:
                pa, qa = (
                    (float(station), float(level)) if ir < ia else (float(level), float(station))
                )
                runs = grid_cross.runs(pa, qa)
                gaps = [(runs[k][1], runs[k + 1][0]) for k in range(len(runs) - 1)]
                inner = [g for g in gaps if g[0] < across < g[1]]
                if inner:
                    halves.append(min(across - inner[0][0], inner[0][1] - across))
            if len(halves) == len(stations):
                widths.append((float(level), float(np.median(halves))))
        arr = np.array(widths)
        if len(arr) >= 12:
            # Several arches close as the height rises — the saddle, and above
            # it whatever windows the web carries. Each is found the same way:
            # start at a local closing point and grow the fit downward while a
            # circle still explains the samples. The saddle is the largest of
            # them, which is also the one a crank has to fit through.
            candidates = []
            blocks: list[list[int]] = []
            for k in range(len(arr)):
                if blocks and abs(arr[k, 0] - arr[k - 1, 0] - step) < 1e-6:
                    blocks[-1].append(k)
                else:
                    blocks.append([k])
            for block in blocks:
                if len(block) < 8:
                    continue
                heights = arr[block, 0]
                halves = arr[block, 1]
                tip = int(np.argmin(halves))
                best = None
                for lo in range(tip - 7, -1, -1):
                    z = heights[lo : tip + 1]
                    h = halves[lo : tip + 1]
                    if not np.all(np.diff(h) <= 1e-9):
                        break
                    design = np.stack([2.0 * z, np.ones_like(z)], axis=1)
                    sol, *_ = np.linalg.lstsq(design, h**2 + z**2, rcond=None)
                    zc = float(sol[0])
                    radius = float(math.sqrt(max(sol[1] + zc**2, 0.0)))
                    model = np.sqrt(np.maximum(radius**2 - (z - zc) ** 2, 0.0))
                    residual = float(np.abs(model - h).max())
                    if residual > 0.4:
                        break
                    best = {
                        "radius": radius,
                        "centre_across": across,
                        "centre_height": zc,
                        "fit_residual_mm": residual,
                        "samples": int(len(z)),
                        "height_range": [float(z.min()), float(z.max())],
                    }
                if best is not None:
                    candidates.append(best)
            if candidates:
                saddle = max(candidates, key=lambda c: c["radius"])

    crank_height = saddle["centre_height"] if saddle else rail

    # Bulkheads: material along the row direction on the crank axis, a little
    # above it — each run is one web, and the run length is its thickness.
    webs = []
    if saddle:
        # Just above the arch apex: high enough to be in solid web, low enough
        # to be under whatever the web is windowed with, so each run along the
        # crank is one bulkhead and its length is that bulkhead's thickness.
        height = crank_height + saddle["radius"] + 3.0
        pp, qq = (saddle["centre_across"], height) if ic < ia else (height, saddle["centre_across"])
        webs = grid_row.runs(pp, qq)

    report["frame"] = {
        "bore_axis": bore_axis.tolist(),
        "crank_axis_direction": row_axis.tolist(),
        "cross_axis": cross.tolist(),
        "note": "the derived frame: origin on the crank axis under the middle of the bore row",
    }
    report["bores"] = {
        "count": int(row["count"]),
        "radius_mm": row["radius"],
        "pitch_mm": row["pitch"],
        "pitch_spread_mm": row["pitch_spread"],
        "coaxial_radii_mm": row["radii"],
        "coaxial_spans": {
            str(r): [round(v, 2) for v in span] for r, span in sorted(coaxial.items())
        },
        "row_positions": row_positions.tolist(),
        "across_position": across,
        "top": float(np.median(bore_top)) if bore_top else None,
        "bottom": float(np.median(bore_bottom)) if bore_bottom else None,
        "open_on_the_axis": [len(bore_runs(float(p))) == 0 for p in row_positions],
    }
    report["planes"] = {
        "deck_offset": deck,
        "pan_rail_offset": rail,
        "normal_to_bore_axis": planes[:8],
    }
    report["saddle"] = saddle
    report["bulkheads"] = {
        "runs": [[round(a, 2), round(b, 2)] for a, b in webs],
        "thickness_mm": [round(b - a, 2) for a, b in webs],
        "centres": [round((a + b) / 2.0, 2) for a, b in webs],
    }
    report["derived_mm"] = {
        "bore_diameter": 2.0 * row["radius"],
        "bore_pitch": row["pitch"],
        "deck_height_above_crank_axis": deck - crank_height,
        "pan_rail_below_crank_axis": crank_height - rail,
        "block_height_rail_to_deck": deck - rail,
        "bore_depth": (float(np.median(bore_top)) - float(np.median(bore_bottom)))
        if bore_top
        else None,
        "saddle_radius": saddle["radius"] if saddle else None,
        "bank_length_over_end_bores": float(row_positions[-1] - row_positions[0]),
    }

    # Envelope: how far the casting reaches at a ladder of heights. Taken from
    # the tessellation's own vertices rather than from a ray, because a ray
    # down the middle of a hollow crankcase finds nothing and would report the
    # part as absent exactly where it is widest.
    heights = part.vertices @ bore_axis
    along = part.vertices @ row_axis
    across_all = part.vertices @ cross
    levels = np.linspace(rail, deck, 12)
    envelope = []
    for lo, hi in zip(levels[:-1], levels[1:]):
        band = (heights >= lo) & (heights <= hi)
        if not band.any():
            continue
        envelope.append(
            {
                "height": round(float((lo + hi) / 2.0), 1),
                "along_crank": [
                    round(float(along[band].min()), 1),
                    round(float(along[band].max()), 1),
                ],
                "across": [
                    round(float(across_all[band].min()), 1),
                    round(float(across_all[band].max()), 1),
                ],
            }
        )
    report["envelope"] = envelope

    # Drillings parallel to the bore axis that break the deck: the head bolts,
    # plus whatever else is tapped into the top face.
    deck_holes = []
    for g in cylinder_axes(part.faces):
        if abs(float(np.dot(g["direction"], bore_axis))) < 0.999:
            continue
        if g["dominant_radius"] >= row["radius"] * 0.6:
            continue
        top = float(g["high"] @ bore_axis)
        if top < deck - 1.0:
            continue
        deck_holes.append(
            {
                "radius": g["dominant_radius"],
                "along_crank": round(float(g["foot"] @ row_axis), 2),
                "across": round(float(g["foot"] @ cross), 2),
                "depth": round(top - float(g["low"] @ bore_axis), 2),
            }
        )
    report["deck_drillings"] = sorted(
        deck_holes, key=lambda h: (h["radius"], h["along_crank"], h["across"])
    )
    return report


# ── report ───────────────────────────────────────────────────────────────────


def format_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    size = report["size_mm"]
    add(f"file            {report['file']}")
    add(f"bounding box    {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} mm")
    add(f"volume          {report['volume_cm3']:.1f} cm^3")
    add(f"mass at 7.2     {report['mass_kg_at_7200']:.2f} kg")
    add("")
    add("face census")
    for kind, entry in report["census"]["by_kind"].items():
        add(f"  {kind:12} {entry['count']:6}  {entry['area_cm2']:10.1f} cm^2")
    add(f"  {'TOTAL':12} {report['census']['faces']:6}")
    add("")
    add(f"distinct cylinder radii: {len(report['radii'])}")
    add(f"  {'radius':>9} {'faces':>6} {'area cm2':>10}  principal axes")
    for entry in report["radii"]:
        axes = ", ".join(f"{tuple(d['dir'])}x{d['count']}" for d in entry["directions"][:2])
        add(f"  {entry['radius']:9.3f} {entry['count']:6} {entry['area_cm2']:10.1f}  {axes}")

    if report.get("frame") is None:
        add("\nno bore row found — the frame could not be derived")
        return "\n".join(lines)

    add("")
    add("frame (derived, not assumed)")
    add(f"  bore axis         {report['frame']['bore_axis']}")
    add(f"  crank axis along  {report['frame']['crank_axis_direction']}")
    bores = report["bores"]
    add("")
    add("bores")
    add(f"  count             {bores['count']}")
    add(f"  radius            {bores['radius_mm']:.3f} mm   diameter {2 * bores['radius_mm']:.2f}")
    add(f"  pitch             {bores['pitch_mm']:.3f} mm  (spread {bores['pitch_spread_mm']:.3f})")
    add("  coaxial with the bore axis (radius: span along the bore axis)")
    for radius, span in bores["coaxial_spans"].items():
        add(
            f"    r {float(radius):8.3f}   {span[0]:8.2f} .. {span[1]:8.2f}   ({span[1] - span[0]:.2f} long)"
        )
    add(f"  open on the axis  {bores['open_on_the_axis']}")
    add(f"  top / bottom      {bores['top']:.2f} / {bores['bottom']:.2f}")
    saddle = report.get("saddle")
    if saddle:
        add("")
        add("main bearing saddle (fitted to the arch, not read off a face)")
        add(f"  radius            {saddle['radius']:.2f} mm")
        add(
            f"  crank axis at     across {saddle['centre_across']:.2f}, height {saddle['centre_height']:.2f}"
        )
        add(f"  fit residual      {saddle['fit_residual_mm']:.2f} mm")
    webs = report["bulkheads"]
    if webs["thickness_mm"]:
        add("")
        add("bulkheads (runs along the crank axis, just above the saddle)")
        add(f"  count             {len(webs['thickness_mm'])}")
        add(f"  thickness         {webs['thickness_mm']}")
        add(f"  centres           {webs['centres']}")
    add("")
    add("derived dimensions, mm")
    for key, value in report["derived_mm"].items():
        if value is not None:
            add(f"  {key:34} {value:10.2f}")
    holes = report.get("deck_drillings") or []
    if holes:
        add("")
        add(f"drillings that break the deck: {len(holes)}")
        by_radius: dict[float, list] = {}
        for h in holes:
            by_radius.setdefault(h["radius"], []).append(h)
        for radius, group in sorted(by_radius.items()):
            offsets = sorted({h["across"] for h in group})
            add(
                f"  r {radius:6.3f}  x{len(group):3}  across {offsets}  "
                f"depth {sorted({h['depth'] for h in group})}"
            )
    add("")
    add("envelope by height")
    add(f"  {'height':>8} {'along crank':>22} {'across':>22}")
    for entry in report["envelope"]:
        add(f"  {entry['height']:8.1f} {str(entry['along_crank']):>22} {str(entry['across']):>22}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", help="path to the STEP file")
    parser.add_argument("--json", dest="json_path", help="also write the report as JSON")
    parser.add_argument("--mesh-cache", help="npz to cache the tessellation in")
    parser.add_argument("--deflection", type=float, default=0.35)
    args = parser.parse_args(argv)

    report = probe(args.step, deflection=args.deflection, mesh_cache=args.mesh_cache)
    print(format_report(report))
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=float)
        print(f"\nwrote {args.json_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
