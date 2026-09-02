"""Material properties for SDF primitives — optical *and* physical.

A material carries two families of properties in one object:

* **Optical** (``color``, ``roughness``, ``metallic``, ``opacity``, ``ior``,
  ``reflectivity``) — what the renderer shades with.  Every one has a
  sensible default, so a bare ``Material()`` shades as white matte plastic.
* **Physical** (``density``, ``conductivity``, ``specific_heat``,
  ``youngs_modulus``, ``poisson_ratio``, ``thermal_expansion``,
  ``yield_strength``) — what the simulation solves with, in SI units.  These
  default to ``None`` meaning *not specified*: a scene that only cares about
  looks never has to invent a Young's modulus, and a study that asks for a
  property the material does not carry gets a clear error rather than a
  plausible-looking wrong number.

Both families travel through the same :class:`~cadjoint.geometry.parameters.Parameter`
containers, so a physical property can be marked ``free`` and tuned by the
optimizer exactly like a color, and stays traceable through
:meth:`Material.as_dict` and :meth:`Material.blend`.  Because every SDF node
answers :meth:`~cadjoint.sdf.base.SDF.material_at` and the smooth booleans
blend the answers, a scene assembled from several materials already defines a
*field* of physical properties that the FEM layer samples per element (see
:mod:`cadjoint.fem.properties`).

Unspecified physical properties are represented as NaN inside the array
representation (``as_dict``) rather than dropped, so the dict keeps one static
pytree structure whatever the scene contains — blending an unspecified value
with a specified one yields NaN, i.e. "still unknown", which the sampling layer
reports as an explicit error instead of silently substituting a default.

Ready-made materials with cited real-world values live in
:mod:`cadjoint.materials`.
"""

from __future__ import annotations

import math
from typing import Any

from jax import Array

from cadjoint.fluent import Fluent
from cadjoint.geometry.parameters import Parameter, Scalar, Vector

#: Optical properties, in the order they are stored.
OPTICAL_PROPERTIES: tuple[str, ...] = (
    "color",
    "roughness",
    "metallic",
    "opacity",
    "ior",
    "reflectivity",
)

#: Physical (simulation) properties, in the order they are stored.
PHYSICAL_PROPERTIES: tuple[str, ...] = (
    "density",
    "conductivity",
    "specific_heat",
    "youngs_modulus",
    "poisson_ratio",
    "thermal_expansion",
    "yield_strength",
)

#: SI unit of every physical property (for display; values are always SI).
UNITS: dict[str, str] = {
    "density": "kg/m^3",
    "conductivity": "W/(m*K)",
    "specific_heat": "J/(kg*K)",
    "youngs_modulus": "Pa",
    "poisson_ratio": "-",
    "thermal_expansion": "1/K",
    "yield_strength": "Pa",
}

# Per-property bounds enforced when free=True.  The physical bounds are wide
# engineering brackets (aerogel-to-tungsten), not tight priors: they exist to
# keep an optimizer inside physically meaningful territory, not to encode a
# material choice.
_BOUNDS: dict[str, tuple] = {
    "color": (0.0, 1.0),
    "roughness": (0.0, 1.0),
    "metallic": (0.0, 1.0),
    "opacity": (0.0, 1.0),
    "ior": (1.0, 3.0),
    "reflectivity": (0.0, 1.0),
    "density": (1.0, 25000.0),
    "conductivity": (1e-3, 3000.0),
    "specific_heat": (1.0, 1e4),
    "youngs_modulus": (1e3, 1e12),
    "poisson_ratio": (0.0, 0.499),
    "thermal_expansion": (0.0, 1e-3),
    "yield_strength": (1e3, 1e11),
}

_NAN = float("nan")


def _display(value: float | None, digits: int = 6) -> float | None:
    """Round a property to ``digits`` significant figures for display.

    Parameters are stored as JAX scalars, which are float32 unless x64 is on,
    so a documented 68.9 GPa reads back as 68900003840.0.  That is a display
    artefact of the storage precision, not a different number — six
    significant figures is well past what any material datasheet claims, so
    rounding there makes the payload legible without losing anything real.
    Only :meth:`Material.describe` rounds; :meth:`Material.get` and
    :meth:`Material.as_dict` hand back the stored value untouched.
    """
    if value is None or value == 0.0 or not math.isfinite(value):
        return value
    exponent = math.floor(math.log10(abs(value)))
    return round(value, digits - 1 - exponent)


def _is_unspecified(value: Any) -> bool:
    """True when a physical property value reads as "not specified" (NaN)."""
    import numpy as np

    array = np.asarray(value, dtype=np.float64)
    return bool(np.all(np.isnan(array)))


def iter_materials(root: Any) -> list[Material]:
    """Every distinct Material in a Fluent tree, in walk order.

    Primitives list their material among their ``children()``, so one ordinary
    Fluent walk reaches every material a scene assigns.

    Args:
        root: Any Fluent node (an SDF, usually the scene root).

    Returns:
        The materials found, deduplicated by identity.  Empty when ``root`` is
        not a walkable Fluent tree.
    """
    found: list[Material] = []
    seen: set[int] = set()

    def walk(node: Any) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        if isinstance(node, Material):
            found.append(node)
            return
        children = getattr(node, "children", None)
        if children is None:
            return
        for child in children():
            walk(child)

    walk(root)
    return found


def unspecified_materials(root: Any, key: str) -> list[str]:
    """Names of the tree's materials that do not specify ``key``.

    A *structural* check, so it is valid under ``jax.grad`` where the sampled
    field itself cannot be inspected: an unspecified property is always a fixed
    NaN, never a traced value.  Callers use it to decide whether sampling is
    worth doing at all — which keeps a scene that never mentions density from
    paying for a density lookup on every solve.

    Args:
        root: Any Fluent node (an SDF, usually the scene root).
        key: Property name.

    Returns:
        One name (or ``"<unnamed material>"``) per material that leaves ``key``
        unspecified.  Empty when every material specifies it — *and* also empty
        when the tree contains no materials at all, so check
        :func:`iter_materials` too if that distinction matters.
    """
    return [
        material.name or "<unnamed material>"
        for material in iter_materials(root)
        if material.get(key) is None
    ]


def specifies_everywhere(root: Any, key: str) -> bool:
    """True when ``root``'s tree has materials and every one specifies ``key``."""
    materials = iter_materials(root)
    return bool(materials) and all(material.get(key) is not None for material in materials)


class Material(Fluent):
    """Material properties for physically-inspired shading and simulation.

    Properties use the same Parameter containers as SDF primitives so material
    assignment, CSG blending and optimization share one consistent
    representation.  Optical properties always have a value; physical ones
    default to ``None`` ("not specified") and are carried as NaN in the array
    representation.

    Args:
        name: Material name, used to derive parameter names when ``free=True``
            (e.g. ``"bumper_mat"`` → ``"bumper_mat_color"``, …).  Optional when
            all properties are already Parameter objects.
        color: RGB surface color, values in [0, 1].
        roughness: Surface roughness, 0 = mirror, 1 = fully diffuse.
        metallic: Metallic factor; 0 = dielectric, 1 = metallic specular.
        opacity: Opacity; 0 = fully transparent, 1 = fully opaque.
        ior: Index of refraction; 1.0 = air, 1.33 = water, 1.5 = glass.
        reflectivity: Mirror reflectivity; 0 = fully diffuse, 1 = perfect mirror.
        density: Mass density in kg/m^3 (mass reporting, self-weight loads,
            mass regularizers).
        conductivity: Thermal conductivity in W/(m*K).
        specific_heat: Specific heat capacity in J/(kg*K) (carried for
            transient studies; steady-state solves ignore it).
        youngs_modulus: Young's modulus in Pa.
        poisson_ratio: Poisson ratio, dimensionless, in [0, 0.5).
        thermal_expansion: Linear coefficient of thermal expansion in 1/K.
        yield_strength: Yield strength in Pa (drives the safety-factor
            post-process on elastic results).
        free: If True, wrap the *specified* raw values as free Parameters with
            sensible bounds.  Unspecified physical properties stay fixed (a
            free NaN is meaningless), and already-constructed Parameter objects
            are left unchanged.  Requires ``name``.

    Example::

        body_mat = Material(color=[0.5, 0.5, 0.5], roughness=0.4,
                            metallic=0.12, density=2700.0,
                            conductivity=167.0, youngs_modulus=68.9e9,
                            poisson_ratio=0.33)
    """

    def __init__(
        self,
        name: str | None = None,
        color=None,
        roughness: float = 0.5,
        metallic: float = 0.0,
        opacity: float = 1.0,
        ior: float = 1.0,
        reflectivity: float = 0.0,
        *,
        density: float | None = None,
        conductivity: float | None = None,
        specific_heat: float | None = None,
        youngs_modulus: float | None = None,
        poisson_ratio: float | None = None,
        thermal_expansion: float | None = None,
        yield_strength: float | None = None,
        free: bool = False,
    ):
        self.name = name
        self.params = {
            "color": color if color is not None else [1.0, 1.0, 1.0],
            "roughness": roughness,
            "metallic": metallic,
            "opacity": opacity,
            "ior": ior,
            "reflectivity": reflectivity,
            "density": density,
            "conductivity": conductivity,
            "specific_heat": specific_heat,
            "youngs_modulus": youngs_modulus,
            "poisson_ratio": poisson_ratio,
            "thermal_expansion": thermal_expansion,
            "yield_strength": yield_strength,
        }
        self._cast_params(name=name, free=free)

    def _cast_params(self, name: str | None = None, free: bool = False) -> None:  # type: ignore[override]
        """Convert raw values to Parameter objects, optionally marking them free.

        Extends the base implementation: when ``free=True``, raw values are
        wrapped as free Parameters with sensible bounds and names derived from
        ``name``.  Existing Parameter objects are always left unchanged, and an
        unspecified physical property (``None``) becomes a *fixed* NaN Scalar
        whatever ``free`` says — an optimizer cannot tune a value the material
        never claimed to have.
        """
        from cadjoint.geometry.parameters import as_parameter

        for key, val in self.params.items():
            if isinstance(val, Parameter):
                continue  # already a Parameter — never overwrite
            if val is None:
                self.params[key] = Scalar(_NAN, free=False, name=None)
                continue
            if free:
                if name is None:
                    raise ValueError(
                        f"Material requires a name when free=True "
                        f"(needed to name the '{key}' parameter)."
                    )
                param_name = f"{name}_{key}"
                bounds = _BOUNDS[key]
                if key == "color":
                    self.params[key] = Vector(list(val), free=True, name=param_name, bounds=bounds)
                else:
                    self.params[key] = Scalar(float(val), free=True, name=param_name, bounds=bounds)
            else:
                self.params[key] = as_parameter(val)

    def children(self) -> list:
        return []

    def as_dict(self) -> dict:
        """Return material properties as a JAX-pytree-compatible dict of arrays.

        Every optical *and* physical property is present, always with the same
        keys, so the dict has one static structure across a whole scene.
        Unspecified physical properties read as NaN.
        """
        return {k: (v.xyz if isinstance(v, Vector) else v.value) for k, v in self.params.items()}

    def get(self, key: str) -> float | None:
        """Concrete value of one property, or None when unspecified.

        Args:
            key: Property name (optical or physical).

        Returns:
            ``float`` for scalar properties, ``list[float]`` for ``color``, and
            ``None`` for a physical property this material does not specify.
        """
        param = self.params[key]
        if isinstance(param, Vector):
            return [float(component) for component in param.value]  # type: ignore[return-value]
        if key in PHYSICAL_PROPERTIES and _is_unspecified(param.value):
            return None
        return float(param.value)

    @staticmethod
    def blend(m1: dict, m2: dict, t: Array) -> dict:
        """Linearly interpolate between two material dicts.

        Physical properties blend exactly like optical ones, which is what
        makes a smooth CSG interface a smooth *property* interface (and hence
        differentiable w.r.t. the geometry that moves it).  An unspecified
        (NaN) property stays unspecified through the blend.

        Args:
            m1: First material dict.
            m2: Second material dict.
            t: Blend factor; t=1 returns m1, t=0 returns m2.
        """
        return {k: m2[k] * (1.0 - t) + m1[k] * t for k in m1}

    def describe(self) -> dict[str, Any]:
        """JSON-ready payload of every property, for the viewer's inspector.

        Returns:
            A dict with the optical properties inline (``color`` as a
            three-float list, the rest as floats), a ``physical`` map whose
            values are floats or ``None`` where the material does not specify
            them, a ``units`` map naming the SI unit of each physical
            property, and a ``free`` map flagging which properties are
            optimizer-controlled.
        """
        return {
            "color": self.get("color"),
            **{key: self.get(key) for key in OPTICAL_PROPERTIES if key != "color"},
            "physical": {key: _display(self.get(key)) for key in PHYSICAL_PROPERTIES},
            "units": dict(UNITS),
            "free": {key: bool(self.params[key].free) for key in self.params},
        }

    def __repr__(self) -> str:
        specified = [
            f"{key}={self.get(key)!r}" for key in PHYSICAL_PROPERTIES if self.get(key) is not None
        ]
        label = f"{self.name!r}, " if self.name else ""
        return (
            f"Material({label}{', '.join(specified)})" if specified else f"Material({label[:-2]})"
        )
