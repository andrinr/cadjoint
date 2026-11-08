"""``SurfaceOptions`` and ``extract_surface``: the polygon mesh's knobs and front door."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from cadjoint.meshing import SurfaceOptions, extract_mesh, extract_surface, surface_grid
from cadjoint.meshing.diagnostics import mesh_report


def box(p):
    return jnp.max(jnp.abs(p) - jnp.array([0.7, 0.5, 0.3]), axis=-1)


def test_defaults_are_the_export_dialogs():
    assert SurfaceOptions() == SurfaceOptions(
        resolution=64, sharp=True, simplify=0.0, feature_edges=False
    )


@pytest.mark.parametrize(
    ("knob", "value"),
    [
        ("resolution", 7),
        ("resolution", 257),
        ("resolution", 64.0),
        ("resolution", True),
        ("simplify", -0.1),
        ("simplify", 1.5),
    ],
)
def test_out_of_range_knobs_are_refused(knob, value):
    with pytest.raises(ValueError, match=knob):
        SurfaceOptions(**{knob: value})


def test_resolution_counts_cells_on_the_longest_axis():
    grid = surface_grid(box, 32)
    assert max(grid.cells) == 32
    assert np.allclose(grid.spacing, grid.spacing[0], rtol=0.1)


def test_default_options_are_plain_dual_contouring():
    mesh = extract_surface(box, SurfaceOptions(resolution=24)).mesh
    reference = extract_mesh(box, surface_grid(box, 24))
    assert np.array_equal(np.asarray(mesh.vertices), np.asarray(reference.vertices))
    assert np.array_equal(mesh.faces, reference.faces)


def test_simplify_collapses_flat_faces_and_stays_closed():
    full = extract_surface(box, SurfaceOptions(resolution=24)).mesh
    simple = extract_surface(box, SurfaceOptions(resolution=24, simplify=0.05)).mesh
    assert simple.faces.shape[0] < 0.15 * full.faces.shape[0]
    report = mesh_report(box, simple)
    assert report["watertight"]
    assert report["surface_deviation"]["max_abs"] < 0.05 * min(surface_grid(box, 24).spacing) + 1e-9


def test_sharp_off_rounds_the_corners():
    sharp = extract_surface(box, SurfaceOptions(resolution=16)).mesh
    smooth = extract_surface(box, SurfaceOptions(resolution=16, sharp=False)).mesh
    corner = np.array([0.7, 0.5, 0.3])
    reach = lambda mesh: np.linalg.norm(np.asarray(mesh.vertices) - corner, axis=1).min()  # noqa: E731
    assert reach(sharp) < 1e-6 < reach(smooth)
