"""The rung ladder, and the claim that padding a tet mesh onto it changes nothing.

Two kinds of test, and the second is the one that matters.

The first kind is arithmetic on the ladder and on the ghost body it builds:
a rung covers its count and does not overshoot by more than the growth
factor; every ghost node is reachable from some ghost cell (an isolated
node owns an empty row of the tangent, which PETSc refuses to eliminate —
the failure that made the first draft of this unusable); every ghost cell is
element 0 again, so it has element 0's volume and orientation.

The second is that the padded solve answers the unpadded question.  The
padding exists to make shapes recur, so it is worth nothing if it perturbs
the physics: ``TestPaddingIsInert`` runs the same problem both ways and
compares the temperature field, the displacement field, and the reverse-mode
gradient the optimizer actually descends through.  The tolerance is not
zero — a larger sparse factorization rounds differently — and it is stated
as a number rather than hidden behind ``allclose``'s defaults.

``TestShapesRecur`` is the point of the exercise: two meshes with different
node and element counts have to come out with *the same* padded shapes, or
the compile is not being reused and the rest is decoration.
"""
# Imports follow pytest.importorskip by design.
# ruff: noqa: E402

from __future__ import annotations

import numpy as np
import pytest

from cadjoint.fem.rungs import enabled, pad_cell_field, pad_indices, pad_tets, rung

pytest.importorskip("jax_fem")

import jax
import jax.numpy as jnp

from cadjoint.fem.backends import ElasticBCs, ThermalBCs
from cadjoint.fem.jaxfem import tet_elastic_solve, tet_thermal_solve

_GROWTH = 1.5


def _lattice_bar(nx: int = 6, ny: int = 3, nz: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """A structured tet bar over ``[0, 1] x [0, .5] x [0, .5]``, six tets per cell.

    Built by hand rather than meshed so the tests do not depend on TetGen
    and so the node count is a knob: ``nx`` moves it by whole layers, which
    is what a design edit does to a real mesh.
    """
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 0.5, ny)
    zs = np.linspace(0.0, 0.5, nz)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)
    points = grid.reshape(-1, 3)

    def node(i, j, k):
        return (i * ny + j) * nz + k

    # The six-tet split of a cube that leaves every tet positively oriented.
    split = ((0, 1, 3, 7), (0, 1, 7, 5), (0, 5, 7, 4), (0, 3, 2, 7), (0, 6, 4, 7), (0, 2, 6, 7))
    corners = (
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    )
    cells = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                ids = [node(i + di, j + dj, k + dk) for di, dj, dk in corners]
                cells.extend([[ids[a] for a in tet] for tet in split])
    return points, np.asarray(cells, dtype=np.int32)


def _tet_volumes(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    corners = points[cells[:, :4]]
    return np.linalg.det(corners[:, 1:, :] - corners[:, :1, :]) / 6.0


def _thermal_case(points, cells):
    """Cold at ``x = 0``, unit inflow over the nodes at ``x = 1``."""
    cold = np.flatnonzero(points[:, 0] <= 1e-12).astype(np.int32)
    hot = np.flatnonzero(points[:, 0] >= 1.0 - 1e-12).astype(np.int32)
    return ThermalBCs(
        dirichlet_nodes=[cold],
        dirichlet_values=[0.0],
        flux_nodes=[hot],
        flux_values=[1.0],
    )


def _elastic_case(points, cells):
    """Clamped at ``x = 0``, pulled down over the nodes at ``x = 1``."""
    clamp = np.flatnonzero(points[:, 0] <= 1e-12).astype(np.int32)
    tip = np.flatnonzero(points[:, 0] >= 1.0 - 1e-12).astype(np.int32)
    return ElasticBCs(
        fixed_nodes=[clamp],
        traction_nodes=[tip],
        traction_vectors=[np.array([0.0, 0.0, -1.0])],
    )


class TestTheLadder:
    def test_the_ladder_covers_and_bounds(self):
        """Every rung holds its count, and overshoots by at most the growth."""
        assert rung(0) == 0
        for count in [1, 2, 63, 64, 65, 100, 511, 900, 3135, 5726, 60_000]:
            size = rung(count)
            assert size >= count
            assert size < max(count * _GROWTH, 65)

    def test_the_ladder_is_monotone_and_deterministic(self):
        """A ladder that wobbled would defeat the whole point of having one."""
        sizes = [rung(n) for n in range(0, 4000)]
        assert sizes == sorted(sizes)
        assert sizes == [rung(n) for n in range(0, 4000)]
        assert len(set(sizes)) < 30  # four thousand counts collapse onto a couple of dozen shapes

    def test_the_ladder_takes_no_negative_count(self):
        with pytest.raises(ValueError, match="non-negative"):
            rung(-1)

    def test_the_switch_is_on_by_default(self, monkeypatch):
        monkeypatch.delenv("CADJOINT_FEM_RUNGS", raising=False)
        assert enabled()
        for off in ("0", "off", "false", "NO", " Off "):
            monkeypatch.setenv("CADJOINT_FEM_RUNGS", off)
            assert not enabled()
        monkeypatch.setenv("CADJOINT_FEM_RUNGS", "1")
        assert enabled()


class TestIndexPadding:
    def test_index_padding_keeps_the_set(self):
        """Padding repeats an entry, so membership — all anyone asks — is unchanged."""
        indices = np.arange(70, dtype=np.int32) * 3
        padded = pad_indices(indices)
        assert padded.shape == (rung(70),)
        assert np.array_equal(padded[:70], indices)
        assert set(padded.tolist()) == set(indices.tolist())

    def test_index_padding_can_be_told_what_to_repeat(self):
        padded = pad_indices(np.arange(70, dtype=np.int32), filler=999)
        assert set(padded[70:].tolist()) == {999}

    def test_an_empty_set_is_left_alone(self):
        assert pad_indices(np.zeros(0, dtype=np.int32)).shape == (0,)

    def test_a_set_already_on_a_rung_is_returned_unchanged(self):
        indices = np.arange(64, dtype=np.int32)
        assert np.array_equal(pad_indices(indices), indices)


class TestTheGhostBody:
    def test_the_caller_s_mesh_survives_verbatim(self):
        points, cells = _lattice_bar()
        padded = pad_tets(points, points, cells)
        assert np.array_equal(padded.points[: len(points)], points)
        assert np.array_equal(padded.cells[: len(cells)], cells)
        assert padded.node_count == len(points)
        assert padded.cell_count == len(cells)

    def test_the_counts_land_on_rungs(self):
        points, cells = _lattice_bar()
        padded = pad_tets(points, points, cells)
        assert len(padded.points) == rung(len(points) + cells.shape[1])
        assert len(padded.cells) == rung(len(padded.cells))

    def test_no_ghost_cell_reaches_a_real_node(self):
        """The ghost body shares no degree of freedom with the mesh."""
        points, cells = _lattice_bar()
        padded = pad_tets(points, points, cells)
        assert padded.cells[len(cells) :].min() >= len(points)

    def test_every_ghost_node_is_named_by_a_ghost_cell(self):
        """An isolated node has no diagonal, and PETSc will not eliminate it."""
        points, cells = _lattice_bar()
        padded = pad_tets(points, points, cells)
        named = set(np.unique(padded.cells[len(cells) :]).tolist())
        assert set(padded.ghost_nodes.tolist()) <= named

    def test_every_ghost_cell_is_element_zero_again(self):
        """Same volume, same sign — a copy, not a degenerate filler."""
        points, cells = _lattice_bar()
        padded = pad_tets(points, points, cells)
        volumes = _tet_volumes(padded.points, padded.cells)
        ghosts = volumes[len(cells) :]
        assert ghosts.size > 0
        assert np.max(np.abs(ghosts - volumes[0])) == 0.0

    def test_every_ghost_cell_names_distinct_nodes(self):
        points, cells = _lattice_bar()
        padded = pad_tets(points, points, cells)
        for row in padded.cells[len(cells) :]:
            assert len(set(row.tolist())) == cells.shape[1]

    def test_the_ghost_cell_handle_is_a_ghost_cell(self):
        points, cells = _lattice_bar()
        padded = pad_tets(points, points, cells)
        assert padded.ghost_cell >= len(cells)
        assert padded.ghost_cell < len(padded.cells)

    def test_padding_stays_traced(self):
        """The ghost rows are a gather of real rows, so a design keeps its gradient."""
        points, cells = _lattice_bar()

        def height(scale):
            moved = jnp.asarray(points) * scale
            return pad_tets(moved, points, cells).points.sum()

        assert np.isfinite(float(jax.grad(height)(1.0)))

    def test_the_switch_turns_it_off(self, monkeypatch):
        monkeypatch.setenv("CADJOINT_FEM_RUNGS", "off")
        points, cells = _lattice_bar()
        assert pad_tets(points, points, cells) is None

    def test_a_mesh_with_no_cells_is_refused(self):
        points, _ = _lattice_bar()
        with pytest.raises(ValueError, match="at least one cell"):
            pad_tets(points, points, np.zeros((0, 4), dtype=np.int32))


class TestShapesRecur:
    def test_two_designs_of_different_size_get_one_pair_of_shapes(self):
        """The whole point: a run of similar designs shares one compiled program."""
        first_points, first_cells = _lattice_bar(nx=11)
        second_points, second_cells = _lattice_bar(nx=13)
        assert len(first_points) != len(second_points)
        assert len(first_cells) != len(second_cells)
        a = pad_tets(first_points, first_points, first_cells)
        b = pad_tets(second_points, second_points, second_cells)
        assert a.points.shape == b.points.shape
        assert a.cells.shape == b.cells.shape

    def test_a_design_far_enough_away_gets_its_own_rung(self):
        """The ladder is a ladder, not one bucket; padding stays bounded."""
        small = _lattice_bar(nx=6)
        large = _lattice_bar(nx=19)
        a = pad_tets(small[0], small[0], small[1])
        b = pad_tets(large[0], large[0], large[1])
        assert a.cells.shape != b.cells.shape
        assert len(b.cells) < len(large[1]) * _GROWTH + 1


class TestPerElementFields:
    """A heterogeneous solve carries one value per element; padding must too."""

    def test_a_scalar_per_element_grows_to_the_cell_rung(self):
        points, cells = _lattice_bar()
        padded = pad_tets(points, points, cells)
        field = np.linspace(1.0, 2.0, len(cells))
        grown = pad_cell_field(field, padded, 1)
        assert grown.shape == (len(padded.cells),)
        assert np.array_equal(grown[: len(cells)], field)
        assert np.all(grown[len(cells) :] == field[0])

    def test_a_vector_per_element_grows_too(self):
        points, cells = _lattice_bar()
        padded = pad_tets(points, points, cells)
        field = np.tile(np.array([0.0, 0.0, -9.81]), (len(cells), 1))
        grown = pad_cell_field(field, padded, 2)
        assert grown.shape == (len(padded.cells), 3)

    def test_scalars_and_a_single_body_force_pass_through(self):
        """``(3,)`` is one force for the whole body, not one per element."""
        points, cells = _lattice_bar()
        padded = pad_tets(points, points, cells)
        assert pad_cell_field(2.5, padded, 1) == 2.5
        assert pad_cell_field(None, padded, 2) is None
        single = np.array([0.0, 0.0, -9.81])
        assert np.array_equal(pad_cell_field(single, padded, 2), single)

    def test_a_traced_field_stays_traced(self):
        points, cells = _lattice_bar()
        padded = pad_tets(points, points, cells)

        def total(scale):
            field = jnp.full(len(cells), scale)
            return pad_cell_field(field, padded, 1).sum()

        assert float(jax.grad(total)(1.0)) == float(len(padded.cells))


@pytest.mark.slow
class TestPaddingIsInert:
    """The padded solve answers the unpadded question, to stated tolerances."""

    def test_the_temperature_field_is_unchanged(self, monkeypatch):
        points, cells = _lattice_bar()
        bcs = _thermal_case(points, cells)
        monkeypatch.setenv("CADJOINT_FEM_RUNGS", "off")
        plain = np.asarray(tet_thermal_solve(points, cells, bcs, conductivity=1.0))
        monkeypatch.setenv("CADJOINT_FEM_RUNGS", "1")
        padded = np.asarray(tet_thermal_solve(points, cells, bcs, conductivity=1.0))
        assert padded.shape == plain.shape == (len(points),)
        assert np.max(np.abs(padded - plain)) < 1e-12 * max(1.0, np.abs(plain).max())

    def test_the_displacement_field_is_unchanged(self, monkeypatch):
        points, cells = _lattice_bar()
        bcs = _elastic_case(points, cells)
        monkeypatch.setenv("CADJOINT_FEM_RUNGS", "off")
        plain = np.asarray(tet_elastic_solve(points, cells, bcs, youngs=1000.0, poisson=0.3))
        monkeypatch.setenv("CADJOINT_FEM_RUNGS", "1")
        padded = np.asarray(tet_elastic_solve(points, cells, bcs, youngs=1000.0, poisson=0.3))
        assert padded.shape == plain.shape == (len(points), 3)
        assert np.max(np.abs(padded - plain)) < 1e-12 * max(1.0, np.abs(plain).max())

    def test_the_gradient_the_optimizer_descends_is_unchanged(self, monkeypatch):
        """The check that matters: the adjoint runs through the padded problem."""
        points, cells = _lattice_bar()
        bcs = _thermal_case(points, cells)

        def peak(stretch):
            moved = jnp.asarray(points).at[:, 0].multiply(stretch)
            field = tet_thermal_solve(moved, cells, bcs, conductivity=1.0, base_points=points)
            return jnp.sum(field**2)

        monkeypatch.setenv("CADJOINT_FEM_RUNGS", "off")
        plain = float(jax.grad(peak)(1.0))
        monkeypatch.setenv("CADJOINT_FEM_RUNGS", "1")
        padded = float(jax.grad(peak)(1.0))
        assert abs(padded - plain) < 1e-8 * max(1.0, abs(plain))
