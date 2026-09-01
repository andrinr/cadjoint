"""Small, forward-rendering scene descriptions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, eq=False)
class Camera:
    """Perspective camera used by :func:`cadjoint.render.render_scene`."""

    position: Any = (5.0, 5.0, 5.0)
    target: Any = (0.0, 0.0, 0.0)
    fov: float = 0.6

    def __post_init__(self) -> None:
        if self.fov <= 0.0:
            raise ValueError("camera fov must be positive")


@dataclass(frozen=True, eq=False)
class Scene:
    """Geometry, camera, lighting, and background for one forward render."""

    geometry: Any
    camera: Camera = field(default_factory=Camera)
    light_directions: Any = ((0.5, 1.0, 0.3),)
    light_colors: Any | None = None
    background_color: Any = (0.0, 0.0, 0.0)
    environment_map: Any | None = None
