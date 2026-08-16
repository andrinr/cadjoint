"""Quantitative artifact metrics for the playground's sharp feature-edge links.

Drives ``_mesh_edge_payload`` directly over a battery of scenes and measures
three failure modes of the sharp-link set:

* ``crossing``: pairs of non-endpoint-sharing links that pass within 0.02 of
  each other at interior parameters — the X-lattice detector.
* ``debris``: open connected components with under three cells of total
  length — orphan "tick" fragments left hovering near tangential contact.
* ``coverage``: for scenes whose feature curves are analytically known, the
  fraction of each curve lying within 1.5 cells of some link — chain gaps
  show up here as dashed real edges (the house-eave regression).

The suite prints a per-configuration metric table; run with ``-s`` to see it
on success.
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxcad.geometry.parameters import Vector
from jaxcad.sdf import Box, Cylinder, Sphere, Translate, Union
from jaxcad.viewer._compile_worker import (
    _MESH_EDGE_RESOLUTION,
    _MESH_EDGE_SIZE,
    _execute_scene,
    _mesh_edge_payload,
)
from jaxcad.viewer.playground import EXAMPLE_SOURCE

CELL = max(_MESH_EDGE_SIZE) / _MESH_EDGE_RESOLUTION

CROSSING_DISTANCE = 0.02
CROSSING_INTERIOR = 0.05
DEBRIS_LENGTH = 3.0 * CELL
COVERAGE_REACH = 1.5 * CELL
COVERAGE_MINIMUM = 0.9

RING_ORIGIN = "origin=[0.0, 1.65, 0.15]"
RING_YS = (0.95, 1.05, 1.12, 1.2, 1.35, 1.65)


# --------------------------------------------------------------------------
# Analytic curve construction
# --------------------------------------------------------------------------


def _edge(start, end) -> np.ndarray:
    """Densely sample a straight segment at roughly quarter-cell spacing."""
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    count = max(int(np.ceil(np.linalg.norm(end - start) / (CELL / 4))), 2)
    return start[None, :] + np.linspace(0.0, 1.0, count)[:, None] * (end - start)[None, :]


def _circle(center, radius, axis) -> np.ndarray:
    """Densely sample a circle around a world axis ('y' or 'z')."""
    center = np.asarray(center, dtype=np.float64)
    count = max(int(np.ceil(2 * np.pi * radius / (CELL / 4))), 64)
    theta = np.linspace(0.0, 2 * np.pi, count, endpoint=False)
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 0.0, 1.0]) if axis == "y" else np.array([0.0, 1.0, 0.0])
    return center[None, :] + radius * (np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v)


def _box_edges(half, center=(0.0, 0.0, 0.0)) -> list[tuple[str, np.ndarray]]:
    """The 12 edges of an axis-aligned box with the given half-extents."""
    hx, hy, hz = half
    cx, cy, cz = center
    edges = []
    for sy in (-1, 1):
        for sz in (-1, 1):
            edges.append((cx - hx, cy + sy * hy, cz + sz * hz, cx + hx, cy + sy * hy, cz + sz * hz))
    for sx in (-1, 1):
        for sz in (-1, 1):
            edges.append((cx + sx * hx, cy - hy, cz + sz * hz, cx + sx * hx, cy + hy, cz + sz * hz))
    for sx in (-1, 1):
        for sy in (-1, 1):
            edges.append((cx + sx * hx, cy + sy * hy, cz - hz, cx + sx * hx, cy + sy * hy, cz + hz))
    return [(f"edge[{index}]", _edge(values[:3], values[3:])) for index, values in enumerate(edges)]


# --------------------------------------------------------------------------
# Scene configurations
# --------------------------------------------------------------------------


def _example_config(ring_y: float):
    """The playground example with the copper ring moved to ``ring_y``.

    The house body is an extruded polygon whose edges are exactly computable:
    the profile outline at both extrusion caps plus the lateral edges at the
    profile vertices.  For ring heights clear of the roof the ring's four
    revolved-corner circles are included as well; at and below the grazing
    height they intersect (or trade surface ownership with) the roof, so only
    the house edges stay analytically complete.
    """
    source = EXAMPLE_SOURCE.replace(RING_ORIGIN, f"origin=[0.0, {ring_y}, 0.15]")
    assert ring_y == 1.65 or source != EXAMPLE_SOURCE, "ring origin not found in example"
    namespace = _execute_scene(source)
    profile = [
        np.asarray(namespace[name].value, dtype=np.float64)
        for name in ("base_left", "base_right", "eave_right", "roof_peak", "eave_left")
    ]
    half_depth = float(namespace["body_depth"].value) / 2.0
    ring = namespace["ring"]
    body = namespace["body"]
    curves = []
    for index in range(len(profile)):
        (ax, ay), (bx, by) = profile[index], profile[(index + 1) % len(profile)]
        for z in (-half_depth, half_depth):
            curves.append((f"house[{index}]@z={z:+.2f}", _edge([ax, ay, z], [bx, by, z]), [ring]))
        curves.append(
            (
                f"house-lateral[{index}]",
                _edge([ax, ay, -half_depth], [ax, ay, half_depth]),
                [ring],
            )
        )
    if ring_y >= 1.3:
        for name in ("ring_inner_low", "ring_outer_low", "ring_outer_high", "ring_inner_high"):
            radius, offset = np.asarray(namespace[name].value, dtype=np.float64)
            curves.append((name, _circle([0.0, ring_y + offset, 0.15], radius, axis="y"), [body]))
    return namespace["scene"], curves


def _box_config():
    return Box(size=Vector([0.8, 0.6, 0.5])), _box_edges((0.8, 0.6, 0.5))


def _box_sphere_config():
    """A sphere poking out of the box's +x face: 12 edges plus a seam circle."""
    scene = Union(
        Box(size=Vector([0.8, 0.6, 0.5])),
        Translate(Sphere(0.4), offset=Vector([1.0, 0.0, 0.0])),
        smoothness=0.0,
    )
    curves = _box_edges((0.8, 0.6, 0.5))
    seam_radius = float(np.sqrt(0.4**2 - 0.2**2))
    center = np.array([0.8, 0.0, 0.0])
    count = max(int(np.ceil(2 * np.pi * seam_radius / (CELL / 4))), 64)
    theta = np.linspace(0.0, 2 * np.pi, count, endpoint=False)
    circle = center[None, :] + seam_radius * (
        np.cos(theta)[:, None] * np.array([0.0, 1.0, 0.0])
        + np.sin(theta)[:, None] * np.array([0.0, 0.0, 1.0])
    )
    curves.append(("seam-circle", circle))
    return scene, curves


def _cylinder_config():
    scene = Cylinder(radius=0.7, height=0.5)
    curves = [
        ("rim@z=-0.5", _circle([0.0, 0.0, -0.5], 0.7, axis="z")),
        ("rim@z=+0.5", _circle([0.0, 0.0, 0.5], 0.7, axis="z")),
    ]
    return scene, curves


def _thin_slab_config():
    """A slab thinner than one cell: its rim collapses to the mid-outline."""
    scene = Box(size=Vector([0.9, 0.7, 0.03]))
    rectangle = [(-0.9, -0.7), (0.9, -0.7), (0.9, 0.7), (-0.9, 0.7)]
    curves = [
        (
            f"outline[{index}]",
            _edge([*rectangle[index], 0.0], [*rectangle[(index + 1) % 4], 0.0]),
        )
        for index in range(4)
    ]
    return scene, curves


def _staggered_slabs_config():
    """Two sub-cell slabs whose long rims run parallel within a cell.

    This is the double-rail regression scene: each slab is thinner than a
    cell (its own rims are ~0.6 cells apart), the second slab's top rim runs
    ~0.3 cells under the first slab's bottom rim over the overlap region, and
    the slabs never intersect — so ownership flips must not suppress the
    rails as fake seams, and cross-links must not weave an X-band.  The
    centers keep a lattice plane inside each slab (planes sit at multiples
    of 0.09375): z in [-0.01, 0.05] contains 0, [-0.10, -0.04] contains
    -0.09375.
    """
    half = (0.9, 0.7, 0.03)
    scene = Union(
        Translate(Box(size=Vector(list(half))), offset=Vector([0.0, 0.0, 0.02])),
        Translate(Box(size=Vector(list(half))), offset=Vector([0.35, 0.0, -0.07])),
        smoothness=0.0,
    )
    curves = []
    for label, (cx, cz) in (("a", (0.0, 0.02)), ("b", (0.35, -0.07))):
        rectangle = [
            (cx - 0.9, -0.7),
            (cx + 0.9, -0.7),
            (cx + 0.9, 0.7),
            (cx - 0.9, 0.7),
        ]
        curves.extend(
            (
                f"{label}-outline[{index}]",
                _edge([*rectangle[index], cz], [*rectangle[(index + 1) % 4], cz]),
            )
            for index in range(4)
        )
    return scene, curves


CONFIGS: dict[str, object] = {
    **{f"example-ring-y={y}": (lambda y=y: _example_config(y)) for y in RING_YS},
    "box": _box_config,
    "box-sphere": _box_sphere_config,
    "cylinder": _cylinder_config,
    "thin-slab": _thin_slab_config,
    "staggered-slabs": _staggered_slabs_config,
}


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def _segment_closest(a0, a1, b0, b1):
    """Clamped closest parameters and distance between segment batches."""
    d1 = a1 - a0
    d2 = b1 - b0
    r = a0 - b0
    a = np.einsum("...i,...i->...", d1, d1)
    e = np.einsum("...i,...i->...", d2, d2)
    f = np.einsum("...i,...i->...", d2, r)
    c = np.einsum("...i,...i->...", d1, r)
    b = np.einsum("...i,...i->...", d1, d2)
    denom = a * e - b * b
    s = np.where(
        denom > 1e-12, np.clip((b * f - c * e) / np.where(denom > 1e-12, denom, 1.0), 0.0, 1.0), 0.0
    )
    t = (b * s + f) / np.where(e > 1e-12, e, 1.0)
    clamped = np.clip(t, 0.0, 1.0)
    s = np.where(
        t != clamped,
        np.clip((b * clamped - c) / np.where(a > 1e-12, a, 1.0), 0.0, 1.0),
        s,
    )
    t = clamped
    closest_a = a0 + s[..., None] * d1
    closest_b = b0 + t[..., None] * d2
    return s, t, np.linalg.norm(closest_a - closest_b, axis=-1)


def count_crossings(segments: np.ndarray) -> int:
    """Pairs of non-endpoint-sharing links passing within 0.02 mid-segment.

    Only pairs belonging to one local link complex (graph-connected within
    four hops) count: an X-lattice weaves between chains that its own
    cross-links connect, while two INDEPENDENT feature curves — a ring rim
    passing just above a roof edge — may legitimately run arbitrarily close
    in 3D and share no links at all.
    """
    n = segments.shape[0]
    if n < 2:
        return 0
    a0, a1 = segments[:, 0], segments[:, 1]
    candidates: list[tuple[int, int]] = []
    for start in range(0, n, 128):
        stop = min(start + 128, n)
        s, t, dist = _segment_closest(
            a0[start:stop, None], a1[start:stop, None], a0[None, :], a1[None, :]
        )
        block = segments[start:stop][:, None, :, None, :]
        other = segments[None, :, None, :, :]
        endpoint_gap = np.linalg.norm(block - other, axis=-1).min(axis=(2, 3))
        mask = (
            (dist < CROSSING_DISTANCE)
            & (s > CROSSING_INTERIOR)
            & (s < 1.0 - CROSSING_INTERIOR)
            & (t > CROSSING_INTERIOR)
            & (t < 1.0 - CROSSING_INTERIOR)
            & (endpoint_gap > 1e-6)
            & (np.arange(start, stop)[:, None] < np.arange(n)[None, :])
        )
        candidates.extend((int(i) + start, int(j)) for i, j in np.argwhere(mask))
    if not candidates:
        return 0
    nodes: dict[tuple[float, float, float], int] = {}
    graph: dict[int, set[int]] = {}
    ends = np.empty((n, 2), dtype=np.int64)
    for index, pair in enumerate(segments):
        for end in (0, 1):
            key = tuple(float(value) for value in pair[end])
            ends[index, end] = nodes.setdefault(key, len(nodes))
        graph.setdefault(ends[index, 0], set()).add(ends[index, 1])
        graph.setdefault(ends[index, 1], set()).add(ends[index, 0])
    total = 0
    for i, j in candidates:
        frontier = set(ends[i])
        seen = set(frontier)
        targets = set(ends[j])
        for _hop in range(4):
            if frontier & targets:
                break
            frontier = {step for node in frontier for step in graph[node]} - seen
            seen |= frontier
        if seen & targets:
            total += 1
    return total


def count_debris(segments: np.ndarray) -> int:
    """Open link components with total length under three cells."""
    if segments.shape[0] == 0:
        return 0
    nodes: dict[tuple[float, float, float], int] = {}
    links = []
    for pair in segments:
        row = []
        for point in pair:
            key = (float(point[0]), float(point[1]), float(point[2]))
            row.append(nodes.setdefault(key, len(nodes)))
        links.append(row)
    parent = list(range(len(nodes)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    degree = np.zeros(len(nodes), dtype=np.int64)
    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    for (u, v), _length in zip(links, lengths):
        degree[u] += 1
        degree[v] += 1
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
    component_length: dict[int, float] = {}
    component_open: dict[int, bool] = {}
    for (u, _v), length in zip(links, lengths):
        root = find(u)
        component_length[root] = component_length.get(root, 0.0) + float(length)
    for node, count in enumerate(degree):
        root = find(node)
        if count == 1:
            component_open[root] = True
    return sum(
        1
        for root, length in component_length.items()
        if component_open.get(root, False) and length < DEBRIS_LENGTH
    )


def _sample_link_distances(segments: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Distance from every curve sample to every link, shaped (samples, links)."""
    a0 = segments[:, 0]
    direction = segments[:, 1] - segments[:, 0]
    squared = np.maximum(np.einsum("ij,ij->i", direction, direction), 1e-12)
    blocks = []
    for start in range(0, samples.shape[0], 512):
        block = samples[start : start + 512]
        t = np.clip(
            np.einsum("pi,si->ps", block, direction) - np.einsum("si,si->s", a0, direction),
            0.0,
            squared,
        )
        closest = a0[None, :, :] + (t / squared)[:, :, None] * direction[None, :, :]
        blocks.append(np.linalg.norm(block[:, None, :] - closest, axis=-1))
    return np.concatenate(blocks, axis=0)


def curve_coverage(segments: np.ndarray, samples: np.ndarray) -> tuple[float, float]:
    """Coverage of a curve by any link, and by the best single link chain.

    The first number is the fraction of curve samples within 1.5 cells of
    some link.  It misses dashing: when every other chain link is dropped,
    the survivors still blanket the curve.  The second number therefore
    covers the curve with only the largest CONNECTED component of the links
    hugging it (both endpoints within 0.75 cells) — a dashed edge falls
    apart into short components and scores low.
    """
    if segments.shape[0] == 0:
        return 0.0, 0.0
    distances = _sample_link_distances(segments, samples)
    any_coverage = float((distances.min(axis=1) <= COVERAGE_REACH).mean())

    endpoint_distance = np.minimum(
        np.linalg.norm(segments[:, None, 0, :] - samples[None, :, :], axis=-1).min(axis=1),
        np.linalg.norm(segments[:, None, 1, :] - samples[None, :, :], axis=-1).min(axis=1),
    )
    hugging = np.flatnonzero(endpoint_distance <= 0.75 * CELL)
    if hugging.size == 0:
        return any_coverage, 0.0
    keys: dict[tuple[float, float, float], int] = {}
    parent = list(range(2 * hugging.size))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    members: dict[int, list[int]] = {}
    for position, link in enumerate(hugging):
        for end in (0, 1):
            key = tuple(float(value) for value in segments[link, end])
            node = keys.setdefault(key, 2 * position + end)
            root_a, root_b = find(2 * position + end), find(node)
            if root_a != root_b:
                parent[root_a] = root_b
        root = find(2 * position)
        parent[find(2 * position + 1)] = root
    for position, link in enumerate(hugging):
        members.setdefault(find(2 * position), []).append(int(link))
    chain_coverage = 0.0
    for component in members.values():
        fraction = float((distances[:, component].min(axis=1) <= COVERAGE_REACH).mean())
        chain_coverage = max(chain_coverage, fraction)
    return any_coverage, chain_coverage


# --------------------------------------------------------------------------
# Shared computation and the metric table
# --------------------------------------------------------------------------


def _visible_runs(scene, label, samples: np.ndarray, others=()) -> list[tuple[str, np.ndarray]]:
    """Split an analytic curve into the parts that stay cleanly drawable.

    Two honest reasons an edge stops being a feature curve: another solid
    swallows it (the copper ring at low heights eats a stretch of the roof
    cap edges — there is genuinely no edge inside), and another surface
    passes within a fraction of a cell (tangential contact: the region is
    seam-contested and nothing sharp exists to draw).  Interior samples and
    samples within 0.75 cells of a listed other operand are dropped, a
    further 1.5 cells around each dropped stretch is trimmed (seam
    junctions replace the edge there), and each remaining run at least 3
    cells long becomes its own curve.
    """
    import jax
    import jax.numpy as jnp

    points = jnp.asarray(samples, dtype=jnp.float32)
    values = np.asarray(jax.vmap(lambda p: jnp.asarray(scene(p)))(points), dtype=np.float64)
    hidden = values < -1e-3
    for field in others:
        magnitude = np.abs(
            np.asarray(jax.vmap(lambda p, f=field: jnp.asarray(f(p)))(points), dtype=np.float64)
        )
        hidden |= magnitude < 0.75 * CELL
    if not hidden.any():
        return [(label, samples)]
    margin = int(np.ceil(1.5 * CELL / (CELL / 4)))
    padded = np.convolve(hidden.astype(int), np.ones(2 * margin + 1, dtype=int), "same") > 0
    runs: list[tuple[str, np.ndarray]] = []
    start = None
    minimum = int(np.ceil(3.0 * CELL / (CELL / 4)))
    for index, hidden in enumerate([*padded, True]):
        if not hidden and start is None:
            start = index
        elif hidden and start is not None:
            if index - start >= minimum:
                runs.append((f"{label}#{len(runs)}", samples[start:index]))
            start = None
    return runs


@pytest.fixture(scope="module")
def results():
    computed = {}
    for name, builder in CONFIGS.items():
        built = builder()
        scene, curves = built if isinstance(built, tuple) else (built, [])
        payload = _mesh_edge_payload(scene)
        assert payload is not None, f"{name}: mesh edge payload unavailable"
        segments = np.asarray(payload["sharp"], dtype=np.float64)
        assert segments.size, f"{name}: no sharp links at all"
        visible = []
        for entry in curves:
            label, samples, *rest = entry
            visible.extend(_visible_runs(scene, label, samples, rest[0] if rest else ()))
        coverages = [(label, curve_coverage(segments, samples)) for label, samples in visible]
        computed[name] = {
            "links": segments.shape[0],
            "crossings": count_crossings(segments),
            "debris": count_debris(segments),
            "coverages": coverages,
        }
    header = (
        f"{'config':<22} {'links':>5} {'cross':>5} {'debris':>6} "
        f"{'worst coverage (any / chain)':<44}"
    )
    print("\n" + header)
    print("-" * len(header))
    for name, row in computed.items():
        if row["coverages"]:
            label, (any_cov, chain_cov) = min(
                row["coverages"], key=lambda item: (item[1][1], item[1][0])
            )
            worst = f"{any_cov:6.3f} / {chain_cov:6.3f}  {label}"
        else:
            worst = "     -"
        print(f"{name:<22} {row['links']:>5} {row['crossings']:>5} {row['debris']:>6} {worst:<44}")
    return computed


@pytest.mark.parametrize("name", list(CONFIGS))
def test_no_crossing_pairs(results, name):
    """No two independent links may pass through each other mid-segment."""
    assert (
        results[name]["crossings"] == 0
    ), f"{name}: {results[name]['crossings']} X-crossing link pairs"


@pytest.mark.parametrize("name", list(CONFIGS))
def test_no_debris_fragments(results, name):
    """No orphan sub-three-cell open fragments may survive."""
    assert (
        results[name]["debris"] == 0
    ), f"{name}: {results[name]['debris']} short open link fragments"


@pytest.mark.parametrize("name", list(CONFIGS))
def test_analytic_curve_coverage(results, name):
    """Every analytic feature curve must be chorded along >90% of its length.

    Both by the link set as a whole (spec metric) and by a single connected
    chain of links (continuity: dashed edges pass the former, not this).
    """
    coverages = results[name]["coverages"]
    if not coverages:
        pytest.skip(f"{name}: no analytic curves")
    failing = [
        (label, any_cov, chain_cov)
        for label, (any_cov, chain_cov) in coverages
        if any_cov <= COVERAGE_MINIMUM or chain_cov <= COVERAGE_MINIMUM
    ]
    assert not failing, f"{name}: undercovered curves (label, any, chain): {failing}"
