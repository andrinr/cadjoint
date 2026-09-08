"""The interval fold is sound: its bound really does contain the field.

Soundness is the only property worth asserting hard here, because it is the
one the mesher's pruning relies on — a bound that is too *wide* costs work,
a bound that is too *narrow* silently drops surface.  So every check below
compares the claimed interval against the field's own values sampled inside
the box, and asserts containment, never tightness.

The reference is :mod:`cadjoint.zeroset.evaluate` in float32, so containment
is asserted with a small slack: a bound that misses by an ulp of the thing
it is being checked against is a float32 artefact in the *reference*, not an
interval error.  Slack is only ever granted in that direction.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from cadjoint.meshing.edge_detection import GridSpec
from cadjoint.zeroset import lower
from cadjoint.zeroset.bounds import Interval, bound_field, surface_cells
from cadjoint.zeroset.evaluate import field
from tests.zeroset.test_lowering import NODES

SCENES = sorted(Path(__file__).resolve().parents[2].joinpath("scenes").glob("*.py"))

# Float32 slack, granted to the reference rather than to the bound.
SLACK = 2e-4


def _boxes(rng, count, extent, width):
    """Random boxes of the given width, centred in a cube of the given extent."""
    centre = rng.uniform(-extent, extent, size=(count, 3))
    half = rng.uniform(0.05, width, size=(count, 3))
    return centre - half, centre + half


def _sampled_range(model, theta, lo, hi, per_box=512, seed=0):
    """The field's observed min and max inside each box, by dense sampling.

    A sampled range is always *inside* the true range, so a sound bound must
    contain it.  That makes this a one-sided test which cannot produce a
    false failure from undersampling.
    """
    rng = np.random.default_rng(seed)
    unit = rng.uniform(0.0, 1.0, size=(per_box, lo.shape[0], 3))
    points = lo[None] + unit * (hi - lo)[None]
    values = np.asarray(field(model)(theta, jnp.asarray(points.reshape(-1, 3))))
    values = values.reshape(per_box, lo.shape[0])
    # The corners too: extrema of a piecewise-linear-ish field often sit there.
    corners = np.array(list(np.ndindex(2, 2, 2)))
    stacked = np.where(corners[:, None, :] == 0, lo[None], hi[None])
    at_corners = np.asarray(field(model)(theta, jnp.asarray(stacked.reshape(-1, 3))))
    at_corners = at_corners.reshape(len(corners), lo.shape[0])
    both = np.concatenate([values, at_corners])
    return both.min(axis=0), both.max(axis=0)


def _assert_contains(interval, low, high, label):
    """The interval contains the sampled range, up to reference slack."""
    tolerance = SLACK * (1.0 + np.maximum(np.abs(low), np.abs(high)))
    under = interval.lo - low
    over = high - interval.hi
    assert np.all(under <= tolerance), (
        f"{label}: bound's floor rose above the field by {under.max():.3e} "
        f"in {int((under > tolerance).sum())} of {low.size} boxes"
    )
    assert np.all(over <= tolerance), (
        f"{label}: bound's ceiling fell below the field by {over.max():.3e} "
        f"in {int((over > tolerance).sum())} of {low.size} boxes"
    )
    assert np.all(interval.lo <= interval.hi), f"{label}: inverted interval"


class TestSoundness:
    """Every node kind's bound contains its field, at three box scales."""

    @pytest.mark.parametrize("label", list(NODES))
    @pytest.mark.parametrize("width", [0.05, 0.4, 1.5], ids=["tight", "medium", "coarse"])
    def test_node_bound_contains_field(self, label, width):
        model = lower(NODES[label]())
        theta = jnp.asarray(model.theta)
        rng = np.random.default_rng(abs(hash(label)) % 2**31)
        lo, hi = _boxes(rng, 64, extent=1.2, width=width)
        interval = bound_field(model, theta, lo, hi)
        low, high = _sampled_range(model, theta, lo, hi)
        _assert_contains(interval, low, high, f"{label} @ {width}")

    def test_degenerate_box_is_point_evaluation(self):
        """A box of zero width brackets the field's value at that point."""
        model = lower(NODES["smooth union"]())
        theta = jnp.asarray(model.theta)
        points = np.random.default_rng(3).uniform(-1.2, 1.2, size=(48, 3))
        interval = bound_field(model, theta, points, points)
        values = np.asarray(field(model)(theta, jnp.asarray(points)))
        assert np.all(interval.lo <= values + SLACK)
        assert np.all(values - SLACK <= interval.hi)

    def test_nested_box_bound_is_no_wider(self):
        """Shrinking a box can only shrink its bound: the fold is monotone."""
        model = lower(NODES["smooth difference"]())
        theta = jnp.asarray(model.theta)
        rng = np.random.default_rng(11)
        lo, hi = _boxes(rng, 48, extent=1.0, width=1.2)
        centre = 0.5 * (lo + hi)
        inner_lo, inner_hi = centre + 0.25 * (lo - centre), centre + 0.25 * (hi - centre)
        outer = bound_field(model, theta, lo, hi)
        inner = bound_field(model, theta, inner_lo, inner_hi)
        assert np.all(outer.lo <= inner.lo + SLACK)
        assert np.all(inner.hi <= outer.hi + SLACK)


class TestShippedScenes:
    """The scenes are where compositions no single node exercises would show."""

    @pytest.mark.slow
    @pytest.mark.parametrize("path", SCENES, ids=[p.stem for p in SCENES])
    def test_scene_bound_contains_field(self, path):
        from cadjoint.viewer.worker.scene import _execute_scene

        model = lower(_execute_scene(path.read_text())["scene"])
        theta = jnp.asarray(model.theta)
        rng = np.random.default_rng(7)
        lo, hi = _boxes(rng, 32, extent=1.0, width=0.6)
        interval = bound_field(model, theta, lo, hi)
        low, high = _sampled_range(model, theta, lo, hi, per_box=256)
        _assert_contains(interval, low, high, path.stem)


class TestSurfaceCells:
    """The descent keeps every cell the surface actually crosses.

    "Crosses" means the corners straddle the level by more than float32 can
    be wrong about.  Cells that only *touch* it are left optional on purpose:
    an axis-aligned box's own face can land exactly on a lattice plane, and
    there the float64 fold reads the half-extent as its float32 value and
    calls the corner -6e-9, while the float32 reference calls it 0.  Both are
    right at their own precision, and no bound can be required to resolve a
    tie the reference itself cannot state.
    """

    #: Corner values inside this band are float32 noise, not a side.
    NOISE = 1e-6

    @classmethod
    def _crossing_cells(cls, model, theta, grid):
        """Cells whose corners reach clearly past the level in both directions."""
        counts = np.asarray(grid.cells, dtype=np.int64)
        axes = [grid.origin[a] + np.arange(counts[a] + 1) * grid.spacing[a] for a in range(3)]
        mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        values = np.asarray(field(model)(theta, jnp.asarray(mesh.reshape(-1, 3))))
        values = values.reshape(mesh.shape[:3])
        corners = [
            values[i : i + counts[0], j : j + counts[1], k : k + counts[2]]
            for i, j, k in np.ndindex(2, 2, 2)
        ]
        stack = np.stack(corners)
        crosses = (stack.min(axis=0) <= -cls.NOISE) & (stack.max(axis=0) >= cls.NOISE)
        return set(map(tuple, np.argwhere(crosses)))

    @pytest.mark.parametrize("label", ["sphere", "box", "hard union", "shell"])
    def test_is_a_superset_of_the_crossing_cells(self, label):
        model = lower(NODES[label]())
        theta = jnp.asarray(model.theta)
        grid = GridSpec.from_bounds((-1.6, -1.6, -1.6), (3.2, 3.2, 3.2), 16)
        kept = set(map(tuple, surface_cells(model, theta, grid)))
        truth = self._crossing_cells(model, theta, grid)
        missing = truth - kept
        assert (
            not missing
        ), f"{label}: dropped {len(missing)} crossing cells, e.g. {sorted(missing)[:4]}"

    def test_stats_report_the_descent(self):
        model = lower(NODES["sphere"]())
        theta = jnp.asarray(model.theta)
        grid = GridSpec.from_bounds((-1.6, -1.6, -1.6), (3.2, 3.2, 3.2), 16)
        stats: dict = {}
        cells = surface_cells(model, theta, grid, stats=stats)
        assert stats["levels"] == 5, "16 cells is four halvings plus the root"
        assert 0 < stats["bounds"], "the descent bounded no boxes at all"
        assert len(cells) < np.prod(grid.cells), "pruning kept the whole lattice"

    def test_a_level_outside_the_field_keeps_nothing(self):
        model = lower(NODES["sphere"]())
        theta = jnp.asarray(model.theta)
        grid = GridSpec.from_bounds((-1.6, -1.6, -1.6), (3.2, 3.2, 3.2), 8)
        assert len(surface_cells(model, theta, grid, level=1e6)) == 0


class TestIntervalRules:
    """The rules that are easy to get subtly wrong, stated as their own cases."""

    def test_abs_across_zero_has_floor_zero(self):
        from cadjoint.zeroset.bounds import _abs

        out = _abs(Interval(np.array([-2.0, 1.0, -5.0]), np.array([3.0, 4.0, -1.0])))
        np.testing.assert_allclose(out.lo, [0.0, 1.0, 1.0])
        np.testing.assert_allclose(out.hi, [3.0, 4.0, 5.0])

    def test_sqrt_of_a_negative_interval_is_clamped(self):
        from cadjoint.zeroset.bounds import _sqrt

        out = _sqrt(Interval(np.array([-4.0, -1.0]), np.array([9.0, -0.5])))
        np.testing.assert_allclose(out.lo, [0.0, 0.0])
        np.testing.assert_allclose(out.hi, [3.0, 0.0])

    def test_a_sine_spanning_its_peak_reaches_one(self):
        """On [1, 2] the peak is interior, so the ceiling is 1 and not an endpoint."""
        from cadjoint.zeroset.bounds import _periodic

        out = _periodic(Interval(np.array([1.0]), np.array([2.0])), np.sin, np.pi / 2.0)
        assert out.hi[0] == pytest.approx(1.0, abs=1e-12)
        assert out.lo[0] == pytest.approx(np.sin(1.0), abs=1e-12)

    def test_an_interval_wider_than_a_period_takes_the_whole_range(self):
        from cadjoint.zeroset.bounds import _periodic

        out = _periodic(Interval(np.array([-10.0]), np.array([10.0])), np.sin, np.pi / 2.0)
        np.testing.assert_allclose([out.lo[0], out.hi[0]], [-1.0, 1.0])

    def test_contains_is_inclusive_at_both_ends(self):
        interval = Interval(np.array([-1.0, 0.0, 0.5]), np.array([1.0, 0.0, 2.0]))
        np.testing.assert_array_equal(interval.contains(0.0), [True, True, False])

    def test_width_is_never_negative(self):
        interval = Interval(np.array([-1.0, 2.0]), np.array([1.0, 2.0]))
        np.testing.assert_allclose(interval.width, [2.0, 0.0])
