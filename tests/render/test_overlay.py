"""Tests for the construction-tree overlay projection."""

import jax.numpy as jnp

from cadjoint.construction import PolygonProfile, SketchPlane
from cadjoint.render import Camera, project_points


class TestProjectPoints:
    def test_camera_target_hits_image_center(self):
        cam = Camera(position=(4.0, 3.0, 6.0), target=(0.3, -0.2, 0.1), fov=0.5)
        h, w = 240, 320
        pixels, valid = project_points(jnp.array([[0.3, -0.2, 0.1]]), cam, (h, w))
        assert bool(valid[0])
        assert abs(pixels[0, 0] - (w - 1) / 2) < 1e-3
        assert abs(pixels[0, 1] - (h - 1) / 2) < 1e-3

    def test_point_behind_camera_invalid(self):
        cam = Camera(position=(4.0, 3.0, 6.0), target=(0.0, 0.0, 0.0), fov=0.5)
        _, valid = project_points(jnp.array([[8.0, 6.0, 12.0]]), cam, (240, 320))
        assert not bool(valid[0])

    def test_offset_along_camera_right_moves_right(self):
        cam = Camera(position=(4.0, 3.0, 6.0), target=(0.0, 0.0, 0.0), fov=0.5)
        pos = jnp.array(cam.position)
        target = jnp.array(cam.target)
        fwd = (target - pos) / jnp.linalg.norm(target - pos)
        right = jnp.cross(fwd, jnp.array([0.0, 1.0, 0.0]))
        right = right / jnp.linalg.norm(right)
        pixels, _ = project_points(jnp.stack([target, target + right]), cam, (240, 320))
        assert pixels[1, 0] > pixels[0, 0]

    def test_profile_vertices_project_finitely(self):
        profile = PolygonProfile(
            [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
            plane=SketchPlane(origin=[0.0, 0.5, 0.0], normal=[0.0, 1.0, 0.0]),
            name="ov",
        )
        cam = Camera(position=(4.0, 3.0, 6.0), target=(0.0, 0.0, 0.0), fov=0.5)
        pixels, valid = project_points(profile.world_vertices(), cam, (240, 320))
        assert valid.all()
        assert jnp.isfinite(jnp.asarray(pixels)).all()
