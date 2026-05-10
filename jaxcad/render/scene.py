"""Scene and Camera: top-level Fluent nodes for inverse rendering."""

from __future__ import annotations

import jax.numpy as jnp

from jaxcad.fluent import Fluent
from jaxcad.geometry.parameters import Scalar, Vector


class Camera(Fluent):
    """Perspective camera stored as Fluent parameters.

    All params can be marked ``free=True`` to include them in gradient-based
    optimisation (e.g. camera pose fitting)::

        cam = Camera(
            camera_pos=Vector([4.0, 2.0, 6.0], free=True, name="camera_pos"),
            look_at=Vector([0.0, 0.0, 0.0],    free=False, name="look_at"),
            fov=Scalar(0.55,                    free=False, name="fov"),
        )

    Args:
        camera_pos: World-space camera position.
        look_at: Point the camera looks toward.
        fov: Half-width field-of-view (world units at unit depth).
    """

    def __init__(
        self,
        camera_pos: Vector,
        look_at: Vector,
        fov: Scalar,
    ):
        self.params = {
            "camera_pos": camera_pos,
            "look_at": look_at,
            "fov": fov,
        }

    def children(self) -> list:
        return []


class Scene(Fluent):
    """Root Fluent node: geometry tree + camera + lighting.

    A single ``extract_parameters(scene)`` call yields every free parameter
    across geometry, materials, camera, and lights.

    Lights are fixed plain arrays by default.  To make them differentiable,
    pass ``light_dirs`` / ``light_colors`` as lists of ``Vector`` params::

        light_dirs = [
            Vector([0.45, 0.85, 0.25], free=True, name="light_dir_0"),
            Vector([-0.3, 0.5, -0.2],  free=True, name="light_dir_1"),
        ]
        light_colors = [
            Vector([0.95, 0.88, 0.75], free=True, name="light_color_0"),
            Vector([0.3,  0.38, 0.55], free=True, name="light_color_1"),
        ]

    Args:
        geometry: SDF tree (``Union``, ``Translate``, primitives, …).
        camera: :class:`Camera` instance holding pose and FOV.
        light_dirs: ``(N, 3)`` array of light directions (normalised internally),
            or a list of ``Vector`` params for differentiable lights.
        light_colors: ``(N, 3)`` array of light RGB colours,
            or a list of ``Vector`` params for differentiable lights.
        bg_color: Background / sky colour in logit space (sigmoid applied at
            render time so values remain unconstrained during optimisation).
            Pass a ``Vector`` param to make it differentiable.
    """

    def __init__(
        self,
        geometry,
        camera: Camera,
        light_dirs=None,
        light_colors=None,
        bg_color=None,
    ):
        self.params = {}
        self.geometry = geometry
        self.camera = camera

        # bg_color: free Vector param or fixed array
        if bg_color is None:
            bg_color = Vector([0.1, 0.25, 0.55], name="bg_color")
        self.params["bg_color"] = bg_color

        # Detect free Vector params vs fixed plain arrays for lights
        if (
            isinstance(light_dirs, (list, tuple))
            and light_dirs
            and isinstance(light_dirs[0], Vector)
        ):
            n = len(light_dirs)
            if light_colors is None:
                light_colors = [Vector([1.0, 1.0, 1.0]) for _ in range(n)]
            for i, ld in enumerate(light_dirs):
                self.params[f"light_dir_{i}"] = ld
            for i, lc in enumerate(light_colors):
                self.params[f"light_color_{i}"] = lc
            self.n_lights = n
            self.free_lights = True
        else:
            if light_dirs is None:
                light_dirs = jnp.array([[0.45, 0.85, 0.25]])
            ld = jnp.atleast_2d(jnp.asarray(light_dirs, dtype=jnp.float32))
            if light_colors is None:
                light_colors = jnp.ones_like(ld)
            self.light_dirs = ld / jnp.linalg.norm(ld, axis=1, keepdims=True)
            self.light_colors = jnp.atleast_2d(jnp.asarray(light_colors, dtype=jnp.float32))
            self.n_lights = ld.shape[0]
            self.free_lights = False

    def children(self) -> list:
        return [self.geometry, self.camera]
