"""The three families answer the same questions: the discretization protocol.

Everything that is not a mesh — studies, the optimiser, the viewer's
payloads, selections — writes against :mod:`cadjoint.fem.discretization`,
so each family has to satisfy it in the same terms.  This pins that, on
small meshes of one ball, and the two things callers rely on most: a
placement is what ``moved`` returns, and a condition that selects nothing
is refused by name.
"""

from __future__ import annotations

import numpy as np
import pytest

from cadjoint.fem import Dirichlet, HeatFlux, Nodes, SimMesh
from cadjoint.fem.discretization import Discretization, Surface, ThermalProblem
from cadjoint.fem.render_payload import surface_render_payload
from cadjoint.geometry.parameters import Scalar
from cadjoint.sdf.primitives import Sphere

pytest.importorskip("jax_fem", reason="FEM simulation needs the 'fem' extra")

BALL = Sphere(radius=Scalar(0.6, free=True, name="R"))
EVERYWHERE = Nodes.sphere([0.0, 0.0, 0.0], 10.0)
NOWHERE = Nodes.sphere([9.0, 9.0, 9.0], 0.1)


def _mesh(method: str, resolution: int = 8):
    return SimMesh(
        name=f"ball-{method}",
        resolution=(resolution,) * 3,
        bounds=(-0.8, -0.8, -0.8),
        size=(1.6, 1.6, 1.6),
        method=method,
        domain=BALL,
    ).build(BALL)


@pytest.fixture(scope="module", params=["hex", "tet4", "cutfem"])
def mesh(request):
    return _mesh(request.param)


def test_each_family_satisfies_the_protocol(mesh):
    assert isinstance(mesh, Discretization)
    assert mesh.family in ("hex", "tet", "cut")
    assert mesh.num_points > 0 and mesh.num_cells > 0


def test_the_surface_is_what_the_payload_draws(mesh):
    surface = mesh.surface()
    assert isinstance(surface, Surface)
    faces = surface.faces
    assert faces.shape[1] in (3, 4) and faces.max() < surface.points.shape[0]
    payload = surface_render_payload(surface, np.zeros(surface.points.shape[0]))
    assert payload["vertex_count"] > 0 and len(payload["indices"]) % 3 == 0
    assert [group["id"] for group in payload["groups"]] == [
        group_id for group_id, _ in surface.groups
    ]


def test_a_condition_that_selects_nothing_is_named(mesh):
    assert mesh.unresolvable([Dirichlet(EVERYWHERE, 0.0)]) is None
    problem = mesh.unresolvable([Dirichlet(EVERYWHERE, 0.0), HeatFlux(NOWHERE, 1.0)])
    assert problem is not None and "HeatFlux" in problem


def test_quality_is_per_element_or_empty(mesh):
    quality = mesh.quality()
    if mesh.cells is None:
        assert quality == {}
    else:
        assert all(values.shape == (mesh.num_cells,) for values in quality.values())


def test_a_placement_is_what_moved_returns(mesh):
    field = lambda p: BALL(p)  # noqa: E731
    placement = mesh.moved(field)
    if mesh.family == "cut":
        assert placement is field
    else:
        assert np.asarray(placement).shape == np.asarray(mesh.points).shape
    result = mesh.thermal(
        ThermalProblem(conductivity=2.0, source=1.0, dirichlet=((EVERYWHERE, 0.0),)),
        placement=placement,
    )
    temperature = np.asarray(result.temperature)
    assert np.isfinite(temperature).all() and temperature.max() > 0
