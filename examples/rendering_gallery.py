"""Generate the forward-renderer comparison images used by the docs and PR."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from cadjoint.render import (
    Camera,
    Material,
    RenderSettings,
    Scene,
    make_gradient_sky,
    render_scene,
)
from cadjoint.sdf.boolean import Union
from cadjoint.sdf.primitives import Plane, Sphere, Torus
from cadjoint.sdf.transforms import Rotate, Translate


def build_scene(*, glass: bool) -> Scene:
    """Build one small material-and-visibility showcase."""
    red = Material(color=[0.78, 0.06, 0.025], roughness=0.42)
    gold = Material(
        color=[1.0, 0.58, 0.08],
        roughness=0.28,
        metallic=0.8,
        reflectivity=0.35,
    )
    sphere_material = Material(
        color=[0.96, 0.98, 1.0] if glass else [0.035, 0.18, 0.72],
        roughness=0.08 if glass else 0.28,
        opacity=0.03 if glass else 1.0,
        ior=1.5,
    )
    ground = Material(color=[0.1, 0.12, 0.16], roughness=0.72)
    ground_height = -0.92
    contact_clearance = 0.002
    sphere_radius = 0.58 if glass else 0.78
    sphere_position = (
        jnp.array([-0.25, 0.5, 1.24])
        if glass
        else jnp.array([1.4, ground_height + sphere_radius + contact_clearance, 0.0])
    )
    geometry = Union(
        Translate(
            Sphere(0.82, material=red),
            jnp.array([-1.45, ground_height + 0.82 + contact_clearance, 0.0]),
        ),
        Translate(
            Rotate(Torus(0.62, 0.23, material=gold), axis="y", angle=0.45),
            jnp.array([0.0, ground_height + 0.62 + 0.23 + contact_clearance, 0.0]),
        ),
        Translate(
            Sphere(sphere_radius, material=sphere_material),
            sphere_position,
        ),
        Plane(ground_height, material=ground),
        smoothness=0.0,
    )
    return Scene(
        geometry,
        camera=Camera(
            position=(4.4, 3.5, 7.5),
            target=(0.0, -0.35, 0.0),
            fov=0.38,
        ),
        light_directions=((0.35, 1.0, 0.55), (-0.65, 0.45, -0.2), (0.15, 0.2, 1.0)),
        light_colors=((1.5, 1.35, 1.15), (0.22, 0.3, 0.5), (0.2, 0.23, 0.3)),
        background_color=(0.025, 0.04, 0.08),
        environment_map=make_gradient_sky(
            sky_color=(0.7, 0.76, 0.88),
            horizon_color=(0.18, 0.27, 0.44),
            ground_color=(0.16, 0.18, 0.24),
        ),
    )


def save_comparison(path: Path, images: list, labels: list[str], columns: int) -> None:
    """Save a tightly cropped, labeled comparison image."""
    rows = (len(images) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 3.7 * rows))
    axes = np.asarray(axes, dtype=object).reshape(-1)
    for axis, image, label in zip(axes, images, labels):
        axis.imshow(image)
        axis.set_title(label, color="white", fontsize=14, fontweight="bold", pad=9)
        axis.axis("off")
    for axis in axes[len(images) :]:
        axis.axis("off")
    figure.patch.set_facecolor("#0b0f18")
    figure.tight_layout(pad=1.4)
    figure.savefig(
        path,
        dpi=100,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    output = Path(__file__).parents[1] / "docs" / "assets"
    output.mkdir(parents=True, exist_ok=True)
    resolution = (300, 440)

    solid_scene = build_scene(glass=False)
    glass_scene = build_scene(glass=True)
    high_quality = replace(
        RenderSettings.high_quality(resolution),
        ambient=0.2,
        exposure=1.15,
    )
    mode_scenes = [solid_scene, solid_scene, solid_scene, glass_scene]
    mode_settings = [
        replace(high_quality, shadow_steps=0),
        high_quality,
        replace(high_quality, reflect_steps=64),
        replace(high_quality, reflect_steps=64, refract_steps=64),
    ]
    mode_images = [
        render_scene(scene, settings) for scene, settings in zip(mode_scenes, mode_settings)
    ]
    save_comparison(
        output / "rendering-modes.png",
        mode_images,
        ["Direct lighting", "Soft shadows", "Metal reflections", "Glass refraction"],
        columns=2,
    )

    quality_scene = solid_scene
    quality_settings = [
        replace(RenderSettings.draft(resolution), ambient=0.2, exposure=1.15),
        replace(RenderSettings.balanced(resolution), ambient=0.2, exposure=1.15),
        high_quality,
    ]
    quality_images = [render_scene(quality_scene, settings) for settings in quality_settings]
    save_comparison(
        output / "rendering-quality.png",
        quality_images,
        ["Draft", "Balanced", "High quality (3x3 SSAA)"],
        columns=3,
    )


if __name__ == "__main__":
    main()
