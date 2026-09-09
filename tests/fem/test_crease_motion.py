"""A mesh that knows its table moves each vertex on every surface that meets there.

The head-to-head in ``research/cutfem`` found the single-field Newton
projection's derivative 11-13 % off its own finite differences on the
starter, at 136 of 860 surface vertices: a vertex sitting on a crease steps
between the two faces that meet there, and neither its position nor its
derivative is the intersection's.  With the scene lowered to its node table
(:func:`cadjoint.fem.hexmesh.with_table`) each vertex is classified onto the
census and solved onto *all* of its surfaces at once
(:func:`cadjoint.zeroset.project.project_table`), which is the same solve the
protocol's ``Solve`` does.

Finite differences are the check, so the scenes here are smooth: a polygon's
even-odd test is a ``sign`` whose flip a difference step reads as a jump
(``research/cutfem/README.md`` measures that), which says nothing about the
derivative.  The starter is checked structurally instead.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cadjoint import extract_parameters
from cadjoint.fem import SimMesh
from cadjoint.geometry.parameters import Scalar, Vector
from cadjoint.sdf.boolean import Union
from cadjoint.sdf.primitives import Box, Sphere
from cadjoint.sdf.transforms import Translate

pytest.importorskip("jax_fem", reason="FEM simulation needs the 'fem' extra")


def _body(smoothness: float):
    """A box and a sphere: hard, their intersection is a real crease curve."""
    return Union(
        Box(size=Vector([0.5, 0.4, 0.35], free=True, name="s")),
        Translate(
            Sphere(radius=Scalar(0.35, free=True, name="r")),
            Vector([0.45, 0.0, 0.0], free=True, name="o"),
        ),
        smoothness=smoothness,
    )


def _mesh(body, method: str = "tet4"):
    return SimMesh(
        name=f"crease-{method}-{id(body)}",
        resolution=(14, 12, 12),
        bounds=(-0.9, -0.7, -0.7),
        size=(1.9, 1.4, 1.4),
        method=method,
        domain=body,
    ).build(body)


@pytest.fixture(scope="module", autouse=True)
def _x64():
    """Double precision for this module only, restored on the way out.

    The finite-difference comparisons below need it.  ``jax_enable_x64`` is
    process-global, so leaving it on leaks into every module that runs
    afterwards: every array becomes float64, and the WGSL emitter — which
    has no 64-bit numeric type — then refuses scenes it should accept.
    """
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.fixture(scope="module")
def creased():
    body = _body(0.0)
    return body, _mesh(body)


def test_the_mesh_carries_its_table_and_its_creases(creased):
    body, mesh = creased
    assert mesh.table is not None
    assert len(mesh.incidence) == mesh.num_surface
    on_creases = sum(1 for row in mesh.incidence if len(row) >= 2)
    assert on_creases > 20, "the box's edges and its seam with the sphere"
    assert any(len(row) == 3 for row in mesh.incidence), "and its corners"


def test_moving_to_the_design_it_was_built_at_changes_nothing(creased):
    body, mesh = creased
    free, _fixed, _meta = extract_parameters(body)
    params = {name: jnp.asarray(value, dtype=float) for name, value in free.items()}
    np.testing.assert_allclose(
        np.asarray(mesh.moved(None, design=(body, params))), mesh.points, atol=1e-9
    )


@pytest.mark.parametrize("smoothness", [0.0, 0.06], ids=["hard", "smooth"])
@pytest.mark.parametrize("parameter", ["r", "s", "o"])
def test_the_derivative_of_the_motion_is_its_finite_difference(smoothness, parameter):
    body = _body(smoothness)
    mesh = _mesh(body)
    free, _fixed, _meta = extract_parameters(body)
    params = {name: jnp.asarray(value, dtype=float) for name, value in free.items()}
    moved = lambda p: mesh.moved(None, design=(body, p))  # noqa: E731

    value = np.asarray(params[parameter], dtype=float)
    direction = np.eye(value.size)[0].reshape(value.shape)
    step = 1e-6
    forward = np.asarray(moved(dict(params, **{parameter: jnp.asarray(value + step * direction)})))
    backward = np.asarray(moved(dict(params, **{parameter: jnp.asarray(value - step * direction)})))
    difference = (forward - backward) / (2 * step)
    tangent = {name: jnp.zeros_like(v) for name, v in params.items()}
    tangent[parameter] = jnp.asarray(direction)
    adjoint = np.asarray(jax.jvp(moved, (params,), (tangent,))[1])
    np.testing.assert_allclose(adjoint, difference, atol=1e-6)


def test_the_starter_classifies_its_creases_and_holds_its_points():
    """The scene the finding came from: polygons, extrusions, a smooth union."""
    from pathlib import Path

    from cadjoint.viewer.worker.scene import _execute_scene

    namespace = _execute_scene(
        Path(__file__).resolve().parents[2].joinpath("scenes", "starter.py").read_text()
    )
    body, mesh = namespace["thermal_body"], namespace["sink_mesh"].build(namespace["thermal_body"])
    assert mesh.table is not None
    assert sum(1 for row in mesh.incidence if len(row) >= 2) > 50, "the fin comb's edges"
    free, _fixed, _meta = extract_parameters(body)
    params = {name: jnp.asarray(value, dtype=float) for name, value in free.items()}
    moved = np.asarray(mesh.moved(None, design=(body, params)))
    np.testing.assert_allclose(moved, mesh.points, atol=1e-8)
    # every surface vertex is on the model, to well under a cell
    from cadjoint.functionalize import functionalize

    field = functionalize(body)(free, _fixed)
    values = np.abs(np.asarray(field(jnp.asarray(moved[: mesh.num_surface]))))
    assert values.max() < 1e-6
