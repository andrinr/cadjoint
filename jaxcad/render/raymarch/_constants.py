"""Shared numerical constants for the forward ray marcher."""

# Prevent approximate distance fields from stalling at a surface.
_MIN_MARCH_STEP: float = 1e-5

# Secondary rays must start outside the primary surface.
_SECONDARY_RAY_OFFSET: float = 4e-3
_GLASS_SURFACE_OFFSET: float = 2e-3

# Soft-shadow rays start away from zero to keep the penumbra ratio finite.
_SHADOW_T_START: float = 1e-2

# Degenerate finite-difference normals remain finite.
_NORMAL_ZERO_THRESHOLD: float = 1e-8
