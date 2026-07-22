"""Shared numerical constants for the ray-marching pipeline."""

# ---------------------------------------------------------------------------
# Core marching
# ---------------------------------------------------------------------------

# Sentinel distance returned / used when a ray misses all geometry.
# Also the initial d_min value before any SDF evaluation.
_SDF_INF: float = 1e9

# Minimum marching step size.  Prevents the ray from stalling when the SDF
# value is zero or negative (e.g. barely inside a surface).
_MIN_MARCH_STEP: float = 1e-5

# ---------------------------------------------------------------------------
# Hit detection
# ---------------------------------------------------------------------------

# d_min threshold below which a primary (or secondary) ray is considered a hit.
_HIT_THRESHOLD: float = 5e-3

# ---------------------------------------------------------------------------
# Secondary ray offsets
# ---------------------------------------------------------------------------

# Reflection ray origin is offset this far along the surface normal to avoid
# self-intersection.  Must exceed _HIT_THRESHOLD so the self-surface d_min
# stays above the hit threshold on the next trace.
_SECONDARY_RAY_OFFSET: float = 0.01

# ---------------------------------------------------------------------------
# Refraction / glass
# ---------------------------------------------------------------------------

# Offset applied along the ray direction when spawning a ray at a glass
# surface (both entry into and exit from the medium) to avoid immediately
# re-hitting the same surface.
_GLASS_SURFACE_OFFSET: float = 1e-3

# Minimum SDF step inside a glass volume.  Slightly larger than _MIN_MARCH_STEP
# because the interior SDF is negated (-sdf), so small positive values here
# mean we're close to the exit face — we want to resolve that boundary.
_GLASS_MIN_STEP: float = 1e-4

# ---------------------------------------------------------------------------
# Surface normals
# ---------------------------------------------------------------------------

# Epsilon added inside sqrt when computing normal magnitude; prevents the
# gradient of sqrt from blowing up at exactly zero.
_NORMAL_MAG_EPS: float = 1e-12

# Normals with magnitude below this are treated as degenerate (zero vector);
# we return the un-normalized raw gradient rather than dividing by ~0.
_NORMAL_ZERO_THRESHOLD: float = 1e-6

# ---------------------------------------------------------------------------
# Shadows
# ---------------------------------------------------------------------------

# Initial t for the shadow march.  Starting at exactly 0 would divide by t
# in the penumbra formula (hardness * h / t), so we begin a small step away.
_SHADOW_T_START: float = 1e-2

# ---------------------------------------------------------------------------
# Blinn-Phong shading
# ---------------------------------------------------------------------------

# Avoid division by zero when converting roughness → Blinn-Phong shininess.
_ROUGHNESS_EPS: float = 1e-4

# Blinn-Phong lighting weights.
_AO_WEIGHT: float = 0.2
_DIFFUSE_WEIGHT: float = 0.7
_SPECULAR_WEIGHT: float = 0.3
