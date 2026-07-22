"""Generate the forward-renderer comparison images used by the docs and PR."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jaxcad.render import (
    Camera,
    Material,
    RenderSettings,
    Scene,
    make_gradient_sky,
    render_scene,
)
from jaxcad.sdf.boolean import Union
from jaxcad.sdf.primitives import Plane, RoundBox, Sphere
from jaxcad.sdf.transforms import Translate


def build_scene(*, glass: bool) -> Scene:
    """Build one small material-and-visibility showcase."""
    red = Material(color=[0.8, 0.08, 0.04], roughness=0.55)
    gold = Material(
        color=[0.95, 0.55, 0.08],
        roughness=0.18,
        metallic=0.9,
        reflectivity=0.45,
    )
    blue = Material(
        color=[0.45, 0.75, 1.0] if glass else [0.08, 0.3, 0.85],
        roughness=0.12 if glass else 0.35,
        opacity=0.12 if glass else 1.0,
        ior=1.5,
    )
    ground = Material(color=[0.18, 0.21, 0.26], roughness=0.82)
    ground_height = -0.92
    contact_clearance = 0.005
    geometry = Union(
        Translate(
            Sphere(0.82, material=red),
            jnp.array([-1.45, ground_height + 0.82 + contact_clearance, 0.0]),
        ),
        Translate(
            RoundBox([0.68, 0.68, 0.68], 0.16, material=gold),
            jnp.array([0.0, ground_height + 0.68 + 0.16 + contact_clearance, 0.0]),
        ),
        Translate(
            Sphere(0.78, material=blue),
            jnp.array([1.4, ground_height + 0.78 + contact_clearance, 0.0]),
        ),
        Plane(ground_height, material=ground),
        smoothness=0.0,
    )
    return Scene(
        geometry,
        camera=Camera(
            position=(4.8, 2.7, 7.6),
            target=(0.0, -0.18, 0.0),
            fov=0.56,
        ),
        light_directions=((0.55, 1.0, 0.45), (-0.55, 0.35, -0.25)),
        light_colors=((1.15, 0.96, 0.78), (0.24, 0.36, 0.7)),
        background_color=(0.06, 0.09, 0.16),
        environment_map=make_gradient_sky(
            sky_color=(0.08, 0.2, 0.52),
            horizon_color=(0.85, 0.58, 0.32),
            ground_color=(0.05, 0.04, 0.04),
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
    figure.patch.set_facecolor("#10141d")
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

    modes_scene = build_scene(glass=True)
    high_quality = RenderSettings.high_quality(resolution)
    mode_settings = [
        replace(high_quality, shadow_steps=0, ao_steps=0, ambient=0.1),
        replace(high_quality, ao_steps=0),
        high_quality,
        replace(high_quality, reflect_steps=64, refract_steps=64),
    ]
    mode_images = [render_scene(modes_scene, settings) for settings in mode_settings]
    save_comparison(
        output / "rendering-modes.png",
        mode_images,
        ["Direct lighting", "Soft shadows", "Shadows + AO", "Reflection + refraction"],
        columns=2,
    )

    quality_scene = build_scene(glass=False)
    quality_settings = [
        RenderSettings.draft(resolution),
        RenderSettings.balanced(resolution),
        RenderSettings.high_quality(resolution),
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
