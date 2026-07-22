"""Renderer configuration and named quality presets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ToneMapping = Literal["aces", "none"]


@dataclass(frozen=True)
class RenderSettings:
    """Quality and performance controls for the forward renderer.

    Use :meth:`draft`, :meth:`balanced`, or :meth:`high_quality` as clear
    starting points, then override individual fields with
    :func:`dataclasses.replace` when a scene needs different trade-offs.
    """

    resolution: tuple[int, int] = (200, 200)
    max_steps: int = 96
    max_distance: float = 20.0
    hit_epsilon: float = 1e-3
    step_scale: float = 0.9
    normal_epsilon: float = 1e-3
    shadow_steps: int = 32
    shadow_distance: float = 20.0
    shadow_hardness: float = 12.0
    ambient: float = 0.08
    ao_steps: int = 4
    ao_step_size: float = 0.08
    ao_strength: float = 0.5
    aa_samples: int = 1
    exposure: float = 1.0
    tone_mapping: ToneMapping = "aces"
    gamma: float = 2.2
    reflect_steps: int = 0
    refract_steps: int = 0

    def __post_init__(self) -> None:
        height, width = self.resolution
        if height <= 0 or width <= 0:
            raise ValueError("resolution dimensions must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.max_distance <= 0.0:
            raise ValueError("max_distance must be positive")
        if self.hit_epsilon <= 0.0 or self.normal_epsilon <= 0.0:
            raise ValueError("surface epsilons must be positive")
        if not 0.0 < self.step_scale <= 1.0:
            raise ValueError("step_scale must be in (0, 1]")
        if self.shadow_steps < 0 or self.ao_steps < 0:
            raise ValueError("shadow_steps and ao_steps cannot be negative")
        if self.shadow_distance <= 0.0 or self.ao_step_size <= 0.0:
            raise ValueError("shadow_distance and ao_step_size must be positive")
        if self.shadow_hardness <= 0.0:
            raise ValueError("shadow_hardness must be positive")
        if self.ambient < 0.0 or self.ao_strength < 0.0:
            raise ValueError("ambient and ao_strength cannot be negative")
        if self.aa_samples < 1:
            raise ValueError("aa_samples must be at least 1")
        if self.exposure <= 0.0:
            raise ValueError("exposure must be positive")
        if self.tone_mapping not in {"aces", "none"}:
            raise ValueError("tone_mapping must be 'aces' or 'none'")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if self.reflect_steps < 0 or self.refract_steps < 0:
            raise ValueError("secondary-ray step counts cannot be negative")

    @classmethod
    def draft(cls, resolution: tuple[int, int] = (200, 200)) -> RenderSettings:
        """Fast interactive preview with direct light and no secondary visibility."""
        return cls(
            resolution=resolution,
            max_steps=72,
            hit_epsilon=1.5e-3,
            shadow_steps=0,
            ambient=0.1,
            ao_steps=0,
        )

    @classmethod
    def balanced(cls, resolution: tuple[int, int] = (200, 200)) -> RenderSettings:
        """Balanced default for notebooks, tests, and documentation."""
        return cls(resolution=resolution)

    @classmethod
    def high_quality(cls, resolution: tuple[int, int] = (400, 400)) -> RenderSettings:
        """Higher precision, denser visibility sampling, and 3x3 SSAA."""
        return cls(
            resolution=resolution,
            max_steps=160,
            hit_epsilon=5e-4,
            step_scale=0.8,
            normal_epsilon=5e-4,
            shadow_steps=64,
            shadow_hardness=16.0,
            ao_steps=6,
            aa_samples=3,
        )
