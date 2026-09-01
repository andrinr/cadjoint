"""Forward rendering utilities for cadjoint SDFs.

Two backends:
- :func:`render_scene` / :func:`raymarch` — early-exit sphere tracing with
  reconstructed silhouettes, finite-difference normals, soft shadows, and GGX
  shading.
- :func:`render_marching_cubes` — mesh extraction via marching cubes (requires
  scikit-image).
"""

from cadjoint.functionalize import functionalize_scene
from cadjoint.render.marching_cubes import render_marching_cubes
from cadjoint.render.material import Material
from cadjoint.render.overlay import draw_plane, draw_profile, project_points
from cadjoint.render.raymarch import (
    make_gradient_sky,
    raymarch,
    render_raymarched,
    render_scene,
)
from cadjoint.render.scene import Camera, Scene
from cadjoint.render.settings import RenderSettings

__all__ = [
    "raymarch",
    "render_scene",
    "render_raymarched",
    "render_marching_cubes",
    "Material",
    "Camera",
    "Scene",
    "RenderSettings",
    "functionalize_scene",
    "make_gradient_sky",
    "project_points",
    "draw_profile",
    "draw_plane",
]
