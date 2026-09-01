"""Tests for the CalculiX (ccx) integration.

Unit tests (deck writer, parsers, the volume-term correction) run
everywhere; live tests require a ``ccx`` binary (CADJOINT_CCX / CCX env
var or PATH — e.g. ``micromamba create -p ./ccx-env -c conda-forge
calculix``) and skip cleanly without one.
"""
# Guarded imports follow pytest.importorskip by design.
# ruff: noqa: E402

from __future__ import annotations

import numpy as np
import pytest

from cadjoint.fem.backends import ElasticBCs
from cadjoint.fem.calculix import (
    consistent_nodal_forces,
    energy_volume_gradient,
    find_ccx,
    parse_dat_displacements,
    parse_dat_stresses,
    parse_frd_fields,
    von_mises,
    write_elastic_deck,
)

_CCX = find_ccx()
needs_ccx = pytest.mark.skipif(_CCX is None, reason="ccx binary not found (CADJOINT_CCX/CCX/PATH)")

# A single unit-cube element (VTK corner order).
_CUBE_POINTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
)
_CUBE_CELLS = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
_CUBE_BCS = ElasticBCs(
    fixed_nodes=[np.array([0, 1, 2, 3])],
    traction_nodes=[np.array([4, 5, 6, 7])],
    traction_vectors=[np.array([0.0, 0.0, 1.0])],
)


class TestConsistentNodalForces:
    def test_uniform_traction_on_unit_quad_splits_evenly(self):
        faces = np.array([[4, 5, 6, 7]])
        forces = consistent_nodal_forces(_CUBE_POINTS, faces, np.array([0.0, 0.0, 2.0]))
        assert np.allclose(forces[4:], [[0.0, 0.0, 0.5]] * 4)
        assert np.allclose(forces[:4], 0.0)

    def test_total_force_equals_traction_times_area(self):
        # Two quads spanning a 2x1 strip, shared edge accumulates.
        points = np.array(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0]], dtype=float
        )
        faces = np.array([[0, 1, 4, 3], [1, 2, 5, 4]])
        traction = np.array([0.5, -1.0, 2.0])
        forces = consistent_nodal_forces(points, faces, traction)
        assert np.allclose(forces.sum(axis=0), 2.0 * traction)
        assert np.allclose(forces[1], forces[4])  # shared edge nodes
        assert np.allclose(forces[1], 2.0 * forces[0])  # twice a corner's share


class TestDeckWriter:
    def test_golden_deck(self):
        deck = write_elastic_deck(_CUBE_POINTS, _CUBE_CELLS, _CUBE_BCS, youngs=1000.0, poisson=0.3)
        expected = """*NODE, NSET=NALL
1, 0, 0, 0
2, 1, 0, 0
3, 1, 1, 0
4, 0, 1, 0
5, 0, 0, 1
6, 1, 0, 1
7, 1, 1, 1
8, 0, 1, 1
*ELEMENT, TYPE=C3D8, ELSET=EALL
1, 1, 2, 3, 4, 5, 6, 7, 8
*NSET, NSET=FIXED
1, 2, 3, 4
*MATERIAL, NAME=MAT0
*ELASTIC
1000, 0.29999999999999999
*SOLID SECTION, ELSET=EALL, MATERIAL=MAT0
*STEP
*STATIC
*BOUNDARY
FIXED, 1, 3, 0.0
*CLOAD
5, 3, 0.24999999999999997
6, 3, 0.24999999999999997
7, 3, 0.25
8, 3, 0.24999999999999997
*NODE PRINT, NSET=NALL
U
*EL PRINT, ELSET=EALL
S
*NODE FILE
U
*END STEP
"""
        assert deck.text == expected
        assert deck.num_nodes == 8
        assert deck.num_cells == 1
        assert np.allclose(deck.nodal_forces.sum(axis=0), [0.0, 0.0, 1.0])

    def test_sensitivity_deck_declares_design_variables(self):
        deck = write_elastic_deck(
            _CUBE_POINTS,
            _CUBE_CELLS,
            _CUBE_BCS,
            youngs=1000.0,
            poisson=0.3,
            design_nodes=np.array([4, 5, 6, 7]),
        )
        assert "*NSET, NSET=DESIGN\n5, 6, 7, 8" in deck.text
        assert "*DESIGN VARIABLES, TYPE=COORDINATE\nDESIGN" in deck.text
        assert "*SENSITIVITY" in deck.text
        assert "STRAINENERGY" in deck.text
        assert deck.text.index("*DESIGN VARIABLES") < deck.text.index("*MATERIAL")

    def test_traction_only_on_fully_spanned_faces(self):
        # A patch covering three of four top corners spans no complete face.
        bcs = ElasticBCs(
            fixed_nodes=[np.array([0, 1, 2, 3])],
            traction_nodes=[np.array([4, 5, 6])],
            traction_vectors=[np.array([0.0, 0.0, 1.0])],
        )
        deck = write_elastic_deck(_CUBE_POINTS, _CUBE_CELLS, bcs, youngs=1.0, poisson=0.0)
        assert "*CLOAD" not in deck.text
        assert np.allclose(deck.nodal_forces, 0.0)


class TestParsers:
    _DAT = """
                        S T E P       1

                                INCREMENT     1

 displacements (vx,vy,vz) for set NALL and time  0.1000000E+01

         1  0.000000E+00  0.000000E+00  0.000000E+00
         5  1.950000E-04  1.950000E-04  9.100000E-04
         6 -1.950000E-04  1.950000E-04  9.100000E-04

 stresses (elem, integ.pnt.,sxx,syy,szz,sxy,sxz,syz) for set EALL and time  0.1000000E+01

          1   1  1.000000E+00  2.000000E+00  3.000000E+00  0.000000E+00  0.000000E+00  0.000000E+00
          1   2  3.000000E+00  2.000000E+00  1.000000E+00  0.000000E+00  0.000000E+00  0.000000E+00
"""

    def test_dat_displacements(self):
        displacement = parse_dat_displacements(self._DAT, 8)
        assert displacement.shape == (8, 3)
        assert np.allclose(displacement[4], [1.95e-4, 1.95e-4, 9.1e-4])
        assert np.allclose(displacement[5], [-1.95e-4, 1.95e-4, 9.1e-4])
        assert np.allclose(displacement[7], 0.0)

    def test_dat_displacements_missing_block_raises(self):
        with pytest.raises(ValueError, match="displacement block"):
            parse_dat_displacements("no data here", 8)

    def test_dat_stresses_average_integration_points(self):
        stress = parse_dat_stresses(self._DAT, 1)
        assert stress.shape == (1, 6)
        assert np.allclose(stress[0], [2.0, 2.0, 2.0, 0.0, 0.0, 0.0])

    def test_von_mises(self):
        assert np.isclose(von_mises(np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])), 0.0)
        assert np.isclose(von_mises(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])), 1.0)
        assert np.isclose(von_mises(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])), np.sqrt(3.0))

    _FRD = """    1C
    1UVERSION           Version 2.23
    1PSTEP                         1           1           1
  100CL  101 1.000000000           8                     0    1           1
 -4  DISP        4    1
 -5  D1          1    2    1    0
 -5  D2          1    2    2    0
 -5  D3          1    2    3    0
 -5  ALL         1    2    0    0    1ALL
 -1         1 0.00000E+00 0.00000E+00 0.00000E+00
 -1         5 1.95000E-04 1.95000E-04 9.10000E-04
 -1         6-1.95000E-04 1.95000E-04 9.10000E-04
 -3
 -4  NORM        4    1
 -5  NORMX       1    2    1    0
 -5  NORMY       1    2    2    0
 -5  NORMZ       1    2    3    0
 -5  ALL         1    2    0    0    1ALL
 -1         5-5.77350E-01-5.77350E-01 5.77350E-01
 -3
 -4  SENENER     2    1
 -5  DFDN        1    1    1    0
 -5  DFDNFIL     1    1    2    0
 -1         5-2.45807E-04-5.00000E-01
 -3
 9999
"""

    def test_frd_fields(self):
        fields = parse_frd_fields(self._FRD, 8)
        assert set(fields) == {"DISP", "NORM", "SENENER"}
        assert fields["DISP"].shape == (8, 3)
        # Fixed 12-char columns: values may run together without spaces.
        assert np.allclose(fields["DISP"][5], [-1.95e-4, 1.95e-4, 9.1e-4])
        assert np.allclose(fields["NORM"][4], [-0.57735, -0.57735, 0.57735])
        assert fields["SENENER"].shape == (8, 2)
        assert np.isclose(fields["SENENER"][4, 0], -2.45807e-4)
        assert np.isclose(fields["SENENER"][4, 1], -0.5)
        assert np.allclose(fields["SENENER"][0], 0.0)


class TestEnergyVolumeGradient:
    def test_matches_finite_differences_at_frozen_strain(self):
        """g[i] . d == d/dh of the frozen-strain energy volume integral."""
        rng = np.random.default_rng(7)
        displacement = 1e-3 * rng.standard_normal(_CUBE_POINTS.shape)
        youngs, poisson = 1000.0, 0.3
        lame_lambda = youngs * poisson / ((1 + poisson) * (1 - 2 * poisson))
        lame_mu = youngs / (2 * (1 + poisson))

        gradient = energy_volume_gradient(
            _CUBE_POINTS, _CUBE_CELLS, displacement, youngs=youngs, poisson=poisson
        )
        h = 1e-6
        for node in (0, 4, 6):
            for axis in range(3):
                plus = _CUBE_POINTS.copy()
                minus = _CUBE_POINTS.copy()
                plus[node, axis] += h
                minus[node, axis] -= h
                # The analytic formula freezes the energy density w at the
                # base geometry and differentiates only detJ; the FD
                # reference does the same.
                fd = (
                    _frozen_density_energy(plus, displacement, lame_lambda, lame_mu)
                    - _frozen_density_energy(minus, displacement, lame_lambda, lame_mu)
                ) / (2 * h)
                assert np.isclose(gradient[node, axis], fd, rtol=1e-5, atol=1e-12)


def _frozen_density_energy(points, displacement, lame_lambda, lame_mu):
    """sum_q w_q(base geometry) * detJ_q(points) for the cube element."""
    from cadjoint.fem.calculix import _HEX_GAUSS_GRADS

    base = _CUBE_POINTS[_CUBE_CELLS][0]
    corners = points[_CUBE_CELLS][0]
    disp = displacement[_CUBE_CELLS][0]
    total = 0.0
    for q in range(8):
        d = _HEX_GAUSS_GRADS[q]
        jac0 = base.T @ d
        grad0 = d @ np.linalg.inv(jac0)
        u_grad = disp.T @ grad0
        eps = 0.5 * (u_grad + u_grad.T)
        w = 0.5 * lame_lambda * np.trace(eps) ** 2 + lame_mu * np.sum(eps * eps)
        total += w * np.linalg.det(corners.T @ d)
    return total


# ---------------------------------------------------------------------------
# Live tests (require the ccx binary).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bar_mesh():
    pytest.importorskip("jax")
    from cadjoint.fem.hexmesh import GridSpec, sdf_to_hex_mesh
    from cadjoint.geometry.parameters import Vector
    from cadjoint.sdf.primitives import Box

    bar = Box(Vector([1.0, 0.15, 0.15], free=True, name="size"))
    grid = GridSpec.from_bounds((-1.1, -0.25, -0.25), (2.2, 0.5, 0.5), (22, 5, 5))
    return sdf_to_hex_mesh(bar, grid)


def _clamp(center):
    return center[0] < -0.999


def _tip(center):
    return center[0] > 0.999


_FIXED = [_clamp]
_TRACTIONS = [(_tip, [0.0, 0.0, -1.0])]


@needs_ccx
class TestLiveForward:
    def test_cube_matches_axial_theory(self, tmp_path):
        from cadjoint.fem.calculix import elastic_ccx_solve

        solution = elastic_ccx_solve(
            _CUBE_POINTS,
            _CUBE_CELLS,
            _CUBE_BCS,
            youngs=1000.0,
            poisson=0.3,
            workdir=tmp_path,
        )
        assert np.allclose(solution.displacement[:4], 0.0)
        assert np.allclose(solution.displacement[4:, 2], 9.1e-4, rtol=1e-5)
        assert solution.strain_energy == pytest.approx(4.55e-4, rel=1e-5)
        # Mean axial stress is the applied traction.
        assert solution.cell_stress[0, 2] == pytest.approx(1.0, rel=1e-4)

    def test_forward_parity_with_jaxfem(self, bar_mesh):
        """Same mesh + BCs through ccx and jax-fem agree to output precision.

        Both solve the identical discrete system (fully integrated
        trilinear HEX8, consistent 2x2 Gauss surface loads); the ~1e-5
        relative tolerance is dominated by ccx's 6-significant-digit
        text output, not by the discretizations.
        """
        pytest.importorskip("jax_fem")
        from cadjoint.fem.simulate import elastic_solve

        direct = elastic_solve(
            bar_mesh,
            youngs=1000.0,
            poisson=0.3,
            dirichlet=_FIXED,
            tractions=_TRACTIONS,
        )
        ccx = elastic_solve(
            bar_mesh,
            youngs=1000.0,
            poisson=0.3,
            dirichlet=_FIXED,
            tractions=_TRACTIONS,
            backend="calculix",
        )
        reference = np.asarray(direct.displacement)
        difference = np.abs(np.asarray(ccx.displacement) - reference)
        scale = np.abs(reference).max()
        assert scale > 1e-4  # the load genuinely bends the bar
        assert difference.max() < 2e-5 * scale

    def test_von_mises_parity_with_jaxfem(self, bar_mesh):
        pytest.importorskip("jax_fem")
        from cadjoint.fem.backends import ElasticBCs as BCs
        from cadjoint.fem.calculix import elastic_ccx_solve, von_mises
        from cadjoint.fem.simulate import elastic_solve

        direct = elastic_solve(
            bar_mesh, youngs=1000.0, poisson=0.3, dirichlet=_FIXED, tractions=_TRACTIONS
        )
        reference = direct.von_mises()
        from cadjoint.fem.simulate import _face_patch, _node_patch

        bcs = BCs(
            fixed_nodes=[_node_patch(bar_mesh, _clamp)],
            traction_nodes=[_face_patch(bar_mesh, _tip)],
            traction_vectors=[np.array([0.0, 0.0, -1.0])],
        )
        solution = elastic_ccx_solve(
            bar_mesh.points, bar_mesh.cells, bcs, youngs=1000.0, poisson=0.3
        )
        mises = von_mises(solution.cell_stress)
        # Element-center evaluation vs mean of 2x2x2 integration points:
        # close but not identical; compare the fields loosely.
        assert mises.shape == reference.shape
        correlation = np.corrcoef(mises, reference)[0, 1]
        assert correlation > 0.99
        assert np.median(np.abs(mises - reference)) < 0.05 * reference.max()


@needs_ccx
class TestLiveAdjoint:
    def test_sensitivities_match_finite_differences(self, tmp_path):
        """Corrected SENENER DFDN vs central FD with ccx forward solves.

        This is the validation of the full adjoint chain: ccx's raw DFDN
        plus the volume-term correction equals dE/d(normal offset) at
        every checked design node.
        """
        from cadjoint.fem.calculix import elastic_ccx_solve

        # 4x2x2 bar of 0.5-sized cells (mixed face/edge/corner nodes).
        nx, ny, nz = 5, 3, 3
        points = np.array(
            [[i * 0.5, j * 0.5, k * 0.5] for i in range(nx) for j in range(ny) for k in range(nz)]
        )
        index = lambda i, j, k: i * ny * nz + j * nz + k  # noqa: E731
        cells = np.array(
            [
                [
                    index(i, j, k),
                    index(i + 1, j, k),
                    index(i + 1, j + 1, k),
                    index(i, j + 1, k),
                    index(i, j, k + 1),
                    index(i + 1, j, k + 1),
                    index(i + 1, j + 1, k + 1),
                    index(i, j + 1, k + 1),
                ]
                for i in range(nx - 1)
                for j in range(ny - 1)
                for k in range(nz - 1)
            ]
        )
        bcs = ElasticBCs(
            fixed_nodes=[np.array([index(0, j, k) for j in range(ny) for k in range(nz)])],
            traction_nodes=[np.array([index(nx - 1, j, k) for j in range(ny) for k in range(nz)])],
            traction_vectors=[np.array([0.0, 0.0, -1.0])],
        )
        adjoint = elastic_ccx_solve(
            points, cells, bcs, youngs=1000.0, poisson=0.3, sensitivities=True
        )
        # Mid-span ring nodes (i=2), away from clamp and load.
        ring = [index(2, j, k) for j in range(ny) for k in range(nz)]
        checked = 0
        h = 1e-2
        for node in ring:
            normal = adjoint.normals[node]
            if np.linalg.norm(normal) < 0.5:
                continue  # interior node, not a design variable
            predicted = float(adjoint.strain_energy_gradient[node] @ normal)
            plus, minus = points.copy(), points.copy()
            plus[node] += h * normal
            minus[node] -= h * normal
            e_plus = elastic_ccx_solve(plus, cells, bcs, youngs=1000.0, poisson=0.3).strain_energy
            e_minus = elastic_ccx_solve(minus, cells, bcs, youngs=1000.0, poisson=0.3).strain_energy
            fd = (e_plus - e_minus) / (2 * h)
            assert predicted == pytest.approx(fd, rel=2e-4), f"node {node}"
            checked += 1
        assert checked >= 8  # all boundary ring nodes exercised

    def test_three_gradient_paths_agree(self, bar_mesh):
        """ccx adjoint vs jax-fem adjoint vs FD on the strain energy.

        The jax-fem path differentiates E = f . u / 2 (f = the consistent
        nodal loads, held fixed) through its adjoint; the ccx path uses
        *SENSITIVITY + the volume correction.  Both are exact derivatives
        of the same discrete objective, so away from the loaded patch the
        normal-projected gradients agree to ccx output precision.
        """
        pytest.importorskip("jax_fem")
        import jax
        import jax.numpy as jnp

        from cadjoint.fem.calculix import elastic_ccx_solve
        from cadjoint.fem.simulate import _face_patch, _node_patch, elastic_solve

        bcs = ElasticBCs(
            fixed_nodes=[_node_patch(bar_mesh, _clamp)],
            traction_nodes=[_face_patch(bar_mesh, _tip)],
            traction_vectors=[np.array([0.0, 0.0, -1.0])],
        )
        adjoint = elastic_ccx_solve(
            bar_mesh.points, bar_mesh.cells, bcs, youngs=1000.0, poisson=0.3, sensitivities=True
        )
        loads = jnp.asarray(
            write_elastic_deck(
                bar_mesh.points, bar_mesh.cells, bcs, youngs=1000.0, poisson=0.3
            ).nodal_forces
        )

        def strain_energy(points):
            result = elastic_solve(
                bar_mesh,
                youngs=1000.0,
                poisson=0.3,
                dirichlet=_FIXED,
                tractions=_TRACTIONS,
                points=points,
            )
            return 0.5 * jnp.sum(loads * result.displacement)

        grad_jaxfem = np.asarray(
            jax.grad(strain_energy)(jnp.asarray(bar_mesh.points, dtype=jnp.float64))
        )

        # Compare normal-projected gradients at design nodes away from the
        # clamped and loaded patches (there the fixed-load convention and
        # jax-fem's load-area derivative coincide).
        excluded = set(np.concatenate([bcs.fixed_nodes[0], bcs.traction_nodes[0]]).tolist())
        nodes, ccx_values, jaxfem_values = [], [], []
        for node in range(bar_mesh.num_points):
            normal = adjoint.normals[node]
            if np.linalg.norm(normal) < 0.5 or node in excluded:
                continue
            nodes.append(node)
            ccx_values.append(float(adjoint.strain_energy_gradient[node] @ normal))
            jaxfem_values.append(float(grad_jaxfem[node] @ normal))
        ccx_values = np.asarray(ccx_values)
        jaxfem_values = np.asarray(jaxfem_values)
        assert len(nodes) >= 100
        scale = np.abs(jaxfem_values).max()
        assert scale > 1e-4
        difference = np.abs(ccx_values - jaxfem_values)
        # Achieved on this mesh: max scaled deviation ~4.5e-5, max relative
        # error ~3.2e-3 on entries above 1% of the gradient scale — set by
        # ccx's 5-6 significant digit text output, not by the adjoint.
        assert difference.max() < 2e-4 * scale
        significant = np.abs(jaxfem_values) > 1e-2 * scale
        assert significant.sum() >= 50
        relative = difference[significant] / np.abs(jaxfem_values[significant])
        assert relative.max() < 1e-2

        # FD spot-check on the three largest entries (two ccx solves each).
        for position in np.argsort(-np.abs(jaxfem_values))[:3]:
            node = nodes[position]
            normal = adjoint.normals[node]
            h = 1e-3
            plus, minus = np.array(bar_mesh.points), np.array(bar_mesh.points)
            plus[node] += h * normal
            minus[node] -= h * normal
            e_plus = elastic_ccx_solve(
                plus, bar_mesh.cells, bcs, youngs=1000.0, poisson=0.3
            ).strain_energy
            e_minus = elastic_ccx_solve(
                minus, bar_mesh.cells, bcs, youngs=1000.0, poisson=0.3
            ).strain_energy
            fd = (e_plus - e_minus) / (2 * h)
            assert ccx_values[position] == pytest.approx(fd, rel=5e-3), f"node {node} (FD)"


@needs_ccx
class TestLiveTesseract:
    def test_backend_registry_roundtrip(self):
        pytest.importorskip("tesseract_core")
        from cadjoint.fem.backends import available_backends, get_backend

        assert "calculix" in available_backends()
        backend = get_backend("calculix")
        assert backend.name == "calculix"
        with pytest.raises(NotImplementedError, match="elastic solves only"):
            backend.thermal(None, None, None, conductivity=1.0, source=0.0)

    def test_strain_energy_gradient_through_tesseract(self, bar_mesh):
        """jax.grad of the tesseract strain_energy equals the raw adjoint."""
        pytest.importorskip("tesseract_core")
        pytest.importorskip("tesseract_jax")
        import jax
        import jax.numpy as jnp

        from cadjoint.fem.calculix import CalculixBackend, elastic_ccx_solve
        from cadjoint.fem.simulate import _face_patch, _node_patch

        bcs = ElasticBCs(
            fixed_nodes=[_node_patch(bar_mesh, _clamp)],
            traction_nodes=[_face_patch(bar_mesh, _tip)],
            traction_vectors=[np.array([0.0, 0.0, -1.0])],
        )
        backend = CalculixBackend()

        def objective(points):
            return backend.elastic_strain_energy(
                points, bar_mesh.cells, bcs, youngs=1000.0, poisson=0.3
            )

        points = jnp.asarray(bar_mesh.points, dtype=jnp.float64)
        value = objective(points)
        gradient = np.asarray(jax.grad(objective)(points))

        reference = elastic_ccx_solve(
            bar_mesh.points, bar_mesh.cells, bcs, youngs=1000.0, poisson=0.3, sensitivities=True
        )
        assert float(value) == pytest.approx(reference.strain_energy, rel=1e-9)
        assert np.allclose(gradient, reference.strain_energy_gradient, rtol=1e-9, atol=1e-15)
        assert np.abs(gradient).max() > 1e-6

    def test_displacement_cotangent_raises(self, bar_mesh):
        pytest.importorskip("tesseract_core")
        pytest.importorskip("tesseract_jax")
        import jax
        import jax.numpy as jnp

        from cadjoint.fem.simulate import elastic_solve

        def objective(points):
            result = elastic_solve(
                bar_mesh,
                youngs=1000.0,
                poisson=0.3,
                dirichlet=_FIXED,
                tractions=_TRACTIONS,
                backend="calculix",
                points=points,
            )
            return jnp.sum(result.displacement**2)

        points = jnp.asarray(bar_mesh.points, dtype=jnp.float64)
        with pytest.raises(Exception, match="strain_energy"):
            jax.grad(objective)(points)

    def test_strain_energy_solve_helper(self, bar_mesh):
        pytest.importorskip("tesseract_core")
        pytest.importorskip("tesseract_jax")
        from cadjoint.fem.calculix import strain_energy_solve

        energy = strain_energy_solve(
            bar_mesh,
            youngs=1000.0,
            poisson=0.3,
            dirichlet=_FIXED,
            tractions=_TRACTIONS,
        )
        assert float(energy) > 0.0
