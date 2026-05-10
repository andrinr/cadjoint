"""Tests for the sphere-tracing renderer.

Each test is designed around a specific, analytically verifiable property of the
renderer rather than trivial shape/range checks.
"""

import jax
import jax.numpy as jnp
import numpy as np

from jaxcad.render.material import Material
from jaxcad.render.raymarch import (
    _camera_rays,
    _cast_shadow,
    _normal_fd,
    _normalize,
    _render_pixel,
    _shade_surface,
    _sphere_trace,
    raymarch,
)
from jaxcad.sdf.primitives import Sphere


def _sphere_sdf(radius=1.0):
    def sdf(p):
        return jnp.linalg.norm(p) - radius

    return sdf


def _default_mat():
    return Material().as_dict()


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------


def test_normalize_known_vector():
    """3-4-5 triangle gives [0.6, 0.8, 0.0] exactly."""
    n = _normalize(jnp.array([3.0, 4.0, 0.0]))
    assert jnp.allclose(n, jnp.array([0.6, 0.8, 0.0]), atol=1e-6)


# ---------------------------------------------------------------------------
# _camera_rays
# ---------------------------------------------------------------------------


def test_camera_center_ray_aligns_with_lookat():
    """For an odd-sized image, the center pixel ray should equal normalize(look_at - cam)."""
    cam = jnp.array([0.0, 0.0, 5.0])
    look = jnp.array([0.0, 0.0, 0.0])
    h, w = 11, 11
    rays = _camera_rays(cam, look, (h, w), fov=0.6)
    center_ray = rays[h // 2 * w + w // 2]
    expected = _normalize(look - cam)
    assert jnp.allclose(center_ray, expected, atol=1e-5)


def test_camera_wider_fov_increases_angular_spread():
    """Wider FOV → larger angle between center and corner rays."""
    cam = jnp.array([0.0, 0.0, 5.0])
    look = jnp.array([0.0, 0.0, 0.0])
    h, w = 11, 11

    def corner_angle(fov):
        rays = _camera_rays(cam, look, (h, w), fov=fov)
        center = rays[h // 2 * w + w // 2]
        corner = rays[0]
        return float(jnp.arccos(jnp.clip(jnp.dot(center, corner), -1.0, 1.0)))

    assert corner_angle(0.9) > corner_angle(0.3)


def test_camera_rays_left_right_x_symmetry():
    """Left and right edge rays in the same row should have opposite x-components."""
    cam = jnp.array([0.0, 0.0, 5.0])
    look = jnp.array([0.0, 0.0, 0.0])
    h, w = 10, 10
    rays = _camera_rays(cam, look, (h, w), fov=0.6)
    mid_row = h // 2
    left = rays[mid_row * w + 0]
    right = rays[mid_row * w + (w - 1)]
    assert jnp.isclose(left[0], -right[0], atol=1e-5)
    assert jnp.isclose(left[1], right[1], atol=1e-5)  # same y
    assert jnp.isclose(left[2], right[2], atol=1e-5)  # same z depth


# ---------------------------------------------------------------------------
# _cast_shadow
# ---------------------------------------------------------------------------


def test_cast_shadow_fully_occluded():
    """Shadow ray that passes through the sphere should return ~0."""
    sdf = _sphere_sdf(radius=1.0)
    # pos directly below the sphere; light is directly above
    # shadow ray from [0,-2,0] toward [0,1,0] passes through the unit sphere
    pos = jnp.array([0.0, -2.0, 0.0])
    normal = jnp.array([0.0, -1.0, 0.0])
    light_dir = jnp.array([0.0, 1.0, 0.0])
    shadow = _cast_shadow(sdf, pos, normal, light_dir, steps=64, hardness=8.0)
    assert float(shadow) < 0.05


def test_cast_shadow_outer_penumbra_harder_means_lighter():
    """At the outer edge of the penumbra, higher hardness → larger shadow factor (more lit).

    IQ soft-shadow formula: shadow = min(k * h / t).
    At the closest approach (h_min = closest_dist - radius, t_min):
      shadow ≈ k * h_min / t_min   (clamped to [0,1]).
    For h_min / t_min << 1/k, larger k → larger (less-shadowed) value.
    This is what makes k large = hard shadows: the lit/shadowed boundary is sharp,
    and points just outside the umbra are fully lit.
    """
    sdf = _sphere_sdf(radius=1.0)
    # Shadow ray from [1.1, -3, 0] toward [0,1,0]:
    # closest approach to origin = 1.1, so h_min = 0.1 at t_min ≈ 3.
    # soft (k=2):  2 * 0.1 / 3 ≈ 0.067   → barely lit
    # hard (k=32): 32 * 0.1 / 3 ≈ 1.067 → clamps to 1 → fully lit
    pos = jnp.array([1.1, -3.0, 0.0])
    normal = jnp.array([0.0, -1.0, 0.0])
    light_dir = jnp.array([0.0, 1.0, 0.0])

    shadow_soft = _cast_shadow(sdf, pos, normal, light_dir, steps=64, hardness=2.0)
    shadow_hard = _cast_shadow(sdf, pos, normal, light_dir, steps=64, hardness=32.0)
    assert float(shadow_hard) > float(shadow_soft)


# ---------------------------------------------------------------------------
# _sphere_trace
# ---------------------------------------------------------------------------


def test_sphere_trace_t_value():
    """Ray from [5,0,0] toward [-1,0,0] hits unit sphere at t ≈ 4 (= 5 - radius)."""
    sdf = _sphere_sdf(radius=1.0)
    origin = jnp.array([5.0, 0.0, 0.0])
    direction = jnp.array([-1.0, 0.0, 0.0])
    t_hit, _ = _sphere_trace(sdf, origin, direction, steps=64)
    assert jnp.isclose(t_hit, 4.0, atol=0.05)


def test_sphere_trace_d_min_near_zero_on_hit():
    """d_min should converge to ~0 on a direct hit."""
    sdf = _sphere_sdf(radius=1.0)
    _, d_min = _sphere_trace(sdf, jnp.array([5.0, 0.0, 0.0]), jnp.array([-1.0, 0.0, 0.0]), steps=64)
    assert float(d_min) < 0.01


def test_sphere_trace_miss_d_min_matches_geometry():
    """Ray [0,5,0] → [0,0,1] stays at y=5; d_min should equal sdf(closest point) = norm([0,5,0])-1 = 4."""
    sdf = _sphere_sdf(radius=1.0)
    origin = jnp.array([0.0, 5.0, 0.0])
    direction = jnp.array([0.0, 0.0, 1.0])  # never approaches sphere
    _, d_min = _sphere_trace(sdf, origin, direction, steps=64)
    # The ray stays at y=5, so minimum sdf = sqrt(0^2 + 5^2 + z^2) - 1, minimised at z=0 → 5-1=4
    assert jnp.isclose(d_min, 4.0, atol=0.1)


# ---------------------------------------------------------------------------
# _shade_surface
# ---------------------------------------------------------------------------


def test_shade_surface_backlit_with_zero_ao_is_black():
    """When both diffuse and specular contributions are zero (and ao=0), output is black.

    Geometry: normal=[0,1,0], ray_dir=[0,0,-1], light_dir=[0,-1,0] (behind surface).
      diffuse = clip(dot([0,1,0],[0,-1,0]), 0,1) = clip(-1,0,1) = 0
      halfway = normalize([0,-1,0] - [0,0,-1]) = normalize([0,-1,1])
      specular = clip(dot([0,-1/√2,1/√2],[0,1,0]),0,1) = clip(-1/√2,0,1) = 0
      ao=0 → AO term also zero; ambient=0 → output is all-zero.
    """
    sdf = _sphere_sdf(radius=1.0)
    pos = jnp.array([0.0, 1.0, 0.0])
    normal = jnp.array([0.0, 1.0, 0.0])
    ray_dir = jnp.array([0.0, 0.0, -1.0])
    light_dirs = jnp.array([[0.0, -1.0, 0.0]])  # light behind the surface
    light_colors = jnp.ones((1, 3))
    rgb = _shade_surface(
        sdf,
        _default_mat(),
        pos,
        normal,
        ray_dir,
        jnp.array(0.0),  # ao=0 eliminates the AO term
        light_dirs,
        light_colors,
        shadow_steps=8,
        shadow_hardness=8.0,
        ambient=0.0,
    )
    assert jnp.allclose(rgb, 0.0, atol=1e-5)


def test_shade_surface_metallic_tints_specular():
    """metallic=1 with a pure-red base_color should zero out the green specular channel.

    Setup: normal=[1,0,0], ray_dir=[-1,0,0], ldir=[1,0,0].
      halfway = normalize([1,0,0] - [-1,0,0]) = [1,0,0] = normal → specular = 1.
      metallic=0: specular_color = [1,1,1] → green highlight present.
      metallic=1: specular_color = [1,0,0] → no green in specular.
    """
    sdf = _sphere_sdf(radius=1.0)
    pos = jnp.array([1.0, 0.0, 0.0])
    normal = jnp.array([1.0, 0.0, 0.0])
    ray_dir = jnp.array([-1.0, 0.0, 0.0])
    light_dirs = jnp.array([[1.0, 0.0, 0.0]])  # halfway = normal → max specular
    light_colors = jnp.ones((1, 3))
    red = jnp.array([1.0, 0.0, 0.0])

    def shade(metallic, roughness=0.1):
        mat = {
            **_default_mat(),
            "color": red,
            "metallic": jnp.array(float(metallic)),
            "roughness": jnp.array(float(roughness)),
        }
        return _shade_surface(
            sdf,
            mat,
            pos,
            normal,
            ray_dir,
            jnp.array(1.0),
            light_dirs,
            light_colors,
            shadow_steps=4,
            shadow_hardness=8.0,
            ambient=0.0,
        )

    rgb_dielectric = shade(metallic=0.0)  # white specular: green channel non-zero
    rgb_metallic = shade(metallic=1.0)  # red specular: green channel = 0

    # Dielectric: specular_color = [1,1,1] adds equally to all channels
    # Metallic:   specular_color = base_color = [1,0,0] adds nothing to green
    assert float(rgb_dielectric[1]) > float(rgb_metallic[1])


def test_shade_surface_ambient_lifts_backlit_surface():
    """ambient > 0 should produce non-zero output even when the light is behind the surface."""
    sdf = _sphere_sdf(radius=1.0)
    pos = jnp.array([0.0, 1.0, 0.0])
    normal = jnp.array([0.0, 1.0, 0.0])
    ray_dir = jnp.array([0.0, 0.0, -1.0])
    light_dirs = jnp.array([[0.0, -1.0, 0.0]])  # behind surface
    light_colors = jnp.ones((1, 3))
    rgb = _shade_surface(
        sdf,
        _default_mat(),
        pos,
        normal,
        ray_dir,
        jnp.array(0.0),
        light_dirs,
        light_colors,
        shadow_steps=8,
        shadow_hardness=8.0,
        ambient=0.1,
    )
    assert jnp.all(rgb > 0.0)


def test_shade_surface_direct_facing_brighter_than_side():
    """A surface facing directly toward a light should be brighter than one lit at 60°."""
    sdf = _sphere_sdf(radius=1.0)
    pos = jnp.array([1.0, 0.0, 0.0])
    normal = jnp.array([1.0, 0.0, 0.0])
    ray_dir = jnp.array([-1.0, 0.0, 0.0])
    ao = jnp.array(0.0)  # zero out AO so only diffuse contributes

    def shade(ldir):
        light_dirs = jnp.array([ldir])
        return _shade_surface(
            sdf,
            _default_mat(),
            pos,
            normal,
            ray_dir,
            ao,
            light_dirs,
            jnp.ones((1, 3)),
            shadow_steps=4,
            shadow_hardness=8.0,
            ambient=0.0,
        ).sum()

    ldir_direct = jnp.array([1.0, 0.0, 0.0])  # dot(normal, l) = 1.0
    ldir_oblique = _normalize(jnp.array([1.0, 1.73, 0.0]))  # dot ≈ 0.5 (60°)
    assert shade(ldir_direct) > shade(ldir_oblique)


# ---------------------------------------------------------------------------
# raymarch (end-to-end)
# ---------------------------------------------------------------------------


def test_raymarch_lit_side_brighter_than_dark_side():
    """With +x lighting, pixels on the right of the sphere should be brighter than on the left.

    Camera at [0,0,5], sphere at origin, radius 1, fov=0.6.
    Screen x ∈ [-0.6, 0.6]; sphere visible where |screen_x| < 0.2 ≈ ±5 pixels from center.
    Pixel at (mid, mid+4) lands on right side (lit), (mid, mid-4) on left (dark).
    """
    img = raymarch(
        _sphere_sdf(radius=1.0),
        camera_pos=jnp.array([0.0, 0.0, 5.0]),
        look_at=jnp.array([0.0, 0.0, 0.0]),
        light_dirs=jnp.array([1.0, 0.0, 0.0]),  # +x
        resolution=(32, 32),
        max_steps=64,
        ambient=0.0,
    )
    h, w = img.shape[:2]
    mid = h // 2
    lit_px = img[mid, mid + 4].mean()
    dark_px = img[mid, mid - 4].mean()
    assert lit_px > dark_px


def test_raymarch_ambient_lifts_minimum_on_hit_pixels():
    """ambient > 0 should raise the darkest hit pixel vs ambient=0."""
    sdf = _sphere_sdf(radius=1.0)
    cam = jnp.array([0.0, 0.0, 5.0])
    look = jnp.array([0.0, 0.0, 0.0])
    light = jnp.array([0.0, 1.0, 0.0])

    img_no_ambient = raymarch(
        sdf,
        camera_pos=cam,
        look_at=look,
        light_dirs=light,
        resolution=(32, 32),
        max_steps=64,
        ambient=0.0,
    )
    img_with_ambient = raymarch(
        sdf,
        camera_pos=cam,
        look_at=look,
        light_dirs=light,
        resolution=(32, 32),
        max_steps=64,
        ambient=0.3,
    )

    # Restrict to pixels that actually hit the sphere (brighter than pure background)
    bg = 0.0
    hit_mask = img_no_ambient.mean(axis=2) > bg + 1e-3
    assert img_with_ambient[hit_mask].min() > img_no_ambient[hit_mask].min()


def test_raymarch_shadow_hardness_changes_penumbra():
    """Changing shadow_hardness should produce meaningfully different images.

    Low hardness → wide soft penumbra; high hardness → sharp boundary.
    The renders must differ by more than floating-point noise.
    """
    sdf = _sphere_sdf(radius=1.0)
    cam = jnp.array([0.0, 0.0, 5.0])
    look = jnp.array([0.0, 0.0, 0.0])
    light = jnp.array([1.0, 0.0, 0.0])

    img_soft = raymarch(
        sdf,
        camera_pos=cam,
        look_at=look,
        light_dirs=light,
        resolution=(32, 32),
        max_steps=64,
        shadow_hardness=1.0,
    )
    img_hard = raymarch(
        sdf,
        camera_pos=cam,
        look_at=look,
        light_dirs=light,
        resolution=(32, 32),
        max_steps=64,
        shadow_hardness=64.0,
    )

    max_diff = float(np.abs(img_soft - img_hard).max())
    assert max_diff > 0.01


def test_raymarch_background_color_on_miss():
    """Corner pixels that miss the geometry must equal gamma(background_color)."""
    bg = jnp.array([0.2, 0.4, 0.6])
    gamma = 2.2
    img = raymarch(
        _sphere_sdf(radius=0.01),
        camera_pos=jnp.array([5.0, 5.0, 5.0]),
        resolution=(8, 8),
        max_steps=8,
        background_color=bg,
        gamma=gamma,
    )
    expected = np.array(bg, dtype=np.float32) ** (1.0 / gamma)
    np.testing.assert_allclose(img[0, 0], expected, atol=1e-4)


def test_raymarch_single_light_1d_vs_2d():
    """(3,) and (1,3) light_dirs must produce pixel-identical images."""
    ldir = jnp.array([0.5, 1.0, 0.3])
    img1 = raymarch(_sphere_sdf(), light_dirs=ldir, resolution=(16, 16), max_steps=32)
    img2 = raymarch(_sphere_sdf(), light_dirs=ldir[None], resolution=(16, 16), max_steps=32)
    np.testing.assert_allclose(img1, img2, atol=1e-5)


def test_raymarch_refraction_produces_finite_image():
    """refract_steps > 0 must run without error and produce all-finite values."""
    img = raymarch(
        _sphere_sdf(radius=1.0),
        resolution=(16, 16),
        max_steps=32,
        refract_steps=16,
    )
    assert np.isfinite(img).all()


def test_raymarch_with_sdf_primitive_material():
    """Sphere primitive (with material_at) should render and differ from no-material version."""
    plain = raymarch(_sphere_sdf(radius=1.0), resolution=(24, 24), max_steps=32)
    colored = raymarch(
        Sphere(radius=1.0, material=Material(color=[0.2, 0.8, 0.2])),
        resolution=(24, 24),
        max_steps=32,
    )
    # Colored sphere should differ from default-material plain sphere
    assert not np.allclose(plain, colored, atol=1e-3)


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------


def test_grad_light_colors_positive():
    """∂(total brightness)/∂light_colors should be non-negative: more light → brighter."""
    sdf = _sphere_sdf(radius=1.0)
    pos = jnp.array([1.0, 0.0, 0.0])
    normal = jnp.array([1.0, 0.0, 0.0])
    ray_dir = jnp.array([-1.0, 0.0, 0.0])
    # Light from the side so diffuse > 0
    light_dirs = jnp.array([[0.0, 1.0, 0.0]])
    ao = jnp.array(1.0)

    def f(lcolors):
        return _shade_surface(
            sdf,
            _default_mat(),
            pos,
            normal,
            ray_dir,
            ao,
            light_dirs,
            lcolors,
            8,
            8.0,
            0.0,
        ).sum()

    grad = jax.grad(f)(jnp.ones((1, 3)))
    # Each light-color channel linearly scales the output, so gradient ≥ 0
    assert jnp.all(grad >= 0.0)
    assert jnp.any(grad > 0.0)


def test_grad_light_dirs_finite_and_nonzero():
    """Gradient w.r.t. light_dirs should exist, be finite, and be non-trivially non-zero."""
    sdf = _sphere_sdf(radius=1.0)
    pos = jnp.array([1.0, 0.0, 0.0])
    normal = jnp.array([1.0, 0.0, 0.0])
    ray_dir = jnp.array([-1.0, 0.0, 0.0])
    light_colors = jnp.ones((1, 3))
    ao = jnp.array(1.0)

    def f(ldirs):
        return _shade_surface(
            sdf,
            _default_mat(),
            pos,
            normal,
            ray_dir,
            ao,
            ldirs,
            light_colors,
            8,
            8.0,
            0.0,
        ).sum()

    ldirs = jnp.array([[0.0, 1.0, 0.0]])
    grad = jax.grad(f)(ldirs)
    assert grad.shape == ldirs.shape
    assert jnp.all(jnp.isfinite(grad))
    assert jnp.any(grad != 0.0)


def test_grad_material_color_positive():
    """∂(total brightness)/∂color > 0: brighter material → brighter pixel."""
    sdf = _sphere_sdf(radius=1.0)
    pos = jnp.array([1.0, 0.0, 0.0])
    normal = jnp.array([1.0, 0.0, 0.0])
    ray_dir = jnp.array([-1.0, 0.0, 0.0])
    light_dirs = jnp.array([[0.0, 1.0, 0.0]])
    light_colors = jnp.ones((1, 3))
    ao = jnp.array(1.0)
    base_mat = _default_mat()

    def f(color):
        return _shade_surface(
            sdf,
            {**base_mat, "color": color},
            pos,
            normal,
            ray_dir,
            ao,
            light_dirs,
            light_colors,
            8,
            8.0,
            0.0,
        ).sum()

    grad = jax.grad(f)(jnp.array([0.5, 0.5, 0.5]))
    assert jnp.all(grad > 0.0)


def test_grad_roughness_finite_and_nonzero():
    """Gradient w.r.t. roughness should be finite and non-zero (roughness controls shininess).

    ldir = normalize([1,1,0]) gives halfway ≈ [0.924, 0.383, 0], so
    dot(halfway, normal) ≈ 0.924 ≠ 1 → d(0.924^shininess)/d(shininess) ≠ 0.
    """
    sdf = _sphere_sdf(radius=1.0)
    pos = jnp.array([1.0, 0.0, 0.0])
    normal = jnp.array([1.0, 0.0, 0.0])
    ray_dir = jnp.array([-1.0, 0.0, 0.0])
    # 45° light: dot(halfway, normal) ≈ 0.924, not 1, so roughness gradient is non-zero
    light_dirs = jnp.array([[1.0, 1.0, 0.0]]) / jnp.sqrt(2.0)
    light_colors = jnp.ones((1, 3))
    ao = jnp.array(1.0)
    base_mat = _default_mat()

    def f(roughness):
        return _shade_surface(
            sdf,
            {**base_mat, "roughness": roughness},
            pos,
            normal,
            ray_dir,
            ao,
            light_dirs,
            light_colors,
            4,
            8.0,
            0.0,
        ).sum()

    grad = jax.grad(f)(jnp.array(0.3))
    assert jnp.isfinite(grad)
    assert float(grad) != 0.0


# ---------------------------------------------------------------------------
# fd_normals vs AD normals
# ---------------------------------------------------------------------------


def test_fd_normals_image_close_to_ad_normals():
    """fd_normals=True and fd_normals=False should produce visually identical renders.

    Central FD at eps=1e-4 approximates the exact gradient to O(eps^2), so pixel
    values should agree to within a small tolerance.
    """
    sdf = _sphere_sdf(radius=1.0)
    common = {
        "camera_pos": jnp.array([0.0, 0.0, 5.0]),
        "look_at": jnp.array([0.0, 0.0, 0.0]),
        "light_dirs": jnp.array([0.5, 1.0, 0.3]),
        "resolution": (32, 32),
        "max_steps": 64,
        "ambient": 0.05,
    }
    img_ad = raymarch(sdf, fd_normals=False, **common)
    img_fd = raymarch(sdf, fd_normals=True, **common)

    max_diff = float(np.abs(img_ad - img_fd).max())
    assert max_diff < 0.02, f"max pixel diff between AD and FD normals: {max_diff:.4f}"


def test_fd_normals_grad_wrt_sdf_param_finite_nonzero():
    """jax.grad through _render_pixel with fd_normals=True should yield finite, non-zero gradients.

    Uses a ray that grazes the unit sphere (d_min ≈ 0.07), so the edge coverage
    term coverage = clip(1 - d_min/edge_width) is in its linear region.  The
    gradient flows: radius → d_min (d(d_min)/d(radius) = -1) → coverage → pixel.

    With fd_normals=True the pipeline contains only forward SDF calls, so the
    outer jax.grad only needs first-order AD.
    """
    light_dirs = jnp.array([[0.5, 1.0, 0.3]])
    light_dirs = light_dirs / jnp.linalg.norm(light_dirs, axis=1, keepdims=True)
    light_colors = jnp.ones((1, 3))
    ray_origin = jnp.array([0.0, 0.0, 5.0])
    # Ray offset by 1.1 in x: closest approach to unit sphere ≈ 1.073, d_min ≈ 0.073.
    # With edge_width=0.5, coverage = clip(1 - 0.073/0.5) ≈ 0.85 — in the linear region.
    _v = jnp.array([1.1, 0.0, -5.0])
    ray_dir = _v / jnp.linalg.norm(_v)

    def loss(radius):
        def sdf(p):
            return jnp.linalg.norm(p) - radius

        pixel = _render_pixel(
            sdf,
            lambda _p: {
                "color": jnp.ones(3) * 0.8,
                "roughness": jnp.array(0.5),
                "metallic": jnp.array(0.0),
                "opacity": jnp.array(1.0),
                "ior": jnp.array(1.5),
            },
            ray_origin,
            ray_dir,
            light_dirs,
            light_colors,
            max_steps=64,
            max_dist=20.0,
            shadow_steps=16,
            shadow_hardness=8.0,
            ambient=0.05,
            edge_width=0.5,
            background_color=jnp.zeros(3),
            refract_steps=0,
            fd_normals=True,
        )
        return pixel.sum()

    grad = jax.grad(loss)(jnp.array(1.0))
    assert jnp.isfinite(grad), f"gradient is not finite: {grad}"
    assert float(grad) != 0.0, "gradient is zero"


# ---------------------------------------------------------------------------
# Reflections
# ---------------------------------------------------------------------------


def test_reflect_steps_zero_unchanged():
    """reflect_steps=0 (default) produces the same image as not passing it."""
    sdf = _sphere_sdf(radius=1.0)
    common = {
        "camera_pos": jnp.array([0.0, 0.0, 5.0]),
        "look_at": jnp.array([0.0, 0.0, 0.0]),
        "light_dirs": jnp.array([0.5, 1.0, 0.3]),
        "resolution": (16, 16),
        "max_steps": 32,
    }
    img_default = raymarch(sdf, **common)
    img_zero = raymarch(sdf, reflect_steps=0, **common)
    np.testing.assert_allclose(img_default, img_zero, atol=1e-6)


def test_reflect_steps_changes_image():
    """A sphere with high reflectivity + reflect_steps>0 differs from reflectivity=0."""
    from jaxcad.sdf.primitives import Sphere

    mirror = Sphere(radius=1.0, material=Material(reflectivity=0.9))
    plain = Sphere(radius=1.0, material=Material(reflectivity=0.0))
    common = {
        "camera_pos": jnp.array([0.0, 0.0, 5.0]),
        "look_at": jnp.array([0.0, 0.0, 0.0]),
        "light_dirs": jnp.array([0.5, 1.0, 0.3]),
        "resolution": (16, 16),
        "max_steps": 32,
        "reflect_steps": 16,
    }
    img_mirror = raymarch(mirror, **common)
    img_plain = raymarch(plain, **common)
    assert not np.allclose(img_mirror, img_plain, atol=1e-3)


def test_reflect_steps_background_on_miss():
    """Reflected ray that misses all geometry uses background_color.

    Ray straight along -z toward the sphere top. Normal at top is [0,0,1].
    reflect_dir = [0,0,-1] - 2*dot([0,0,-1],[0,0,1])*[0,0,1] = [0,0,1].
    This reflected ray moves away from the sphere → miss → background_color.
    With reflectivity=1 the surface color is entirely the reflected color, so the
    final pixel should equal background_color (modulo edge coverage ≈ 1).
    """
    bg = jnp.array([0.5, 0.2, 0.8])

    def sdf(p):
        return jnp.linalg.norm(p) - 1.0

    def mat_fn(_p):
        return {
            "color": jnp.ones(3),
            "roughness": jnp.array(0.5),
            "metallic": jnp.array(0.0),
            "opacity": jnp.array(1.0),
            "ior": jnp.array(1.0),
            "reflectivity": jnp.array(1.0),
        }

    light_dirs = jnp.array([[0.0, 1.0, 0.0]])
    light_colors = jnp.ones((1, 3))
    ray_origin = jnp.array([0.0, 0.0, 5.0])
    ray_dir = jnp.array([0.0, 0.0, -1.0])

    pixel = _render_pixel(
        sdf,
        mat_fn,
        ray_origin,
        ray_dir,
        light_dirs,
        light_colors,
        max_steps=64,
        max_dist=20.0,
        shadow_steps=8,
        shadow_hardness=8.0,
        ambient=0.0,
        edge_width=0.05,
        background_color=bg,
        refract_steps=0,
        fd_normals=True,
        reflect_steps=32,
    )
    # With reflectivity=1, rgb_surface = rgb_reflected = bg (miss).
    # With opacity=1, rgb = bg.  Coverage ≈ 1 on a direct center hit.
    np.testing.assert_allclose(np.array(pixel), np.array(bg), atol=0.05)


def test_grad_reflectivity_finite_nonzero():
    """jax.grad through _render_pixel w.r.t. reflectivity is finite and non-zero."""
    light_dirs = jnp.array([[0.5, 1.0, 0.3]])
    light_dirs = light_dirs / jnp.linalg.norm(light_dirs, axis=1, keepdims=True)
    light_colors = jnp.ones((1, 3))
    ray_origin = jnp.array([0.0, 0.0, 5.0])
    ray_dir = jnp.array([0.0, 0.0, -1.0])  # hits unit sphere at top

    def loss(reflectivity):
        def sdf(p):
            return jnp.linalg.norm(p) - 1.0

        def mat_fn(_p):
            return {
                "color": jnp.ones(3) * 0.8,
                "roughness": jnp.array(0.5),
                "metallic": jnp.array(0.0),
                "opacity": jnp.array(1.0),
                "ior": jnp.array(1.0),
                "reflectivity": reflectivity,
            }

        pixel = _render_pixel(
            sdf,
            mat_fn,
            ray_origin,
            ray_dir,
            light_dirs,
            light_colors,
            max_steps=64,
            max_dist=20.0,
            shadow_steps=8,
            shadow_hardness=8.0,
            ambient=0.1,
            edge_width=0.1,
            background_color=jnp.array([0.0, 0.5, 1.0]),
            refract_steps=0,
            fd_normals=True,
            reflect_steps=16,
        )
        return pixel.sum()

    grad = jax.grad(loss)(jnp.array(0.5))
    assert jnp.isfinite(grad), f"gradient is not finite: {grad}"
    assert float(grad) != 0.0, "gradient is zero"


def test_normal_fd_matches_ad_on_sphere():
    """_normal_fd should agree with jax.grad(sdf) to O(eps^2) on a unit sphere.

    At pos = [1, 0, 0] on the unit sphere the exact outward normal is [1, 0, 0].
    Both methods should recover this to within FD truncation error.
    """

    def sdf(p):
        return jnp.linalg.norm(p) - 1.0

    pos = jnp.array([1.0, 0.0, 0.0])
    eps = 1e-4

    ad_raw = jax.grad(sdf)(pos)
    ad_normal = ad_raw / jnp.linalg.norm(ad_raw)

    fd_raw, fd_mag = _normal_fd(sdf, pos, eps)
    fd_normal = fd_raw / jnp.where(fd_mag > 1e-6, fd_mag, 1.0)

    assert jnp.allclose(
        fd_normal, ad_normal, atol=1e-3
    ), f"FD normal {fd_normal} differs from AD normal {ad_normal}"
