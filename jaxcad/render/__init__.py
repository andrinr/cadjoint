"""Forward rendering utilities for jaxcad SDFs.

Two backends:
- :func:`render_scene` / :func:`raymarch` — early-exit sphere tracing with
  finite-difference normals, geometric AO, soft shadows, and GGX shading.
- :func:`render_marching_cubes` — mesh extraction via marching cubes (requires
  scikit-image).
"""

from jaxcad.functionalize import functionalize_scene
from jaxcad.render.marching_cubes import render_marching_cubes
from jaxcad.render.material import Material
from jaxcad.render.raymarch import (
    make_gradient_sky,
    raymarch,
    render_raymarched,
    render_scene,
)
from jaxcad.render.scene import Camera, Scene
from jaxcad.render.settings import RenderSettings

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
]
