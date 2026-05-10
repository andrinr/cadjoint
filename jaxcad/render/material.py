"""Material properties for SDF primitives."""

from __future__ import annotations

from jax import Array

from jaxcad.fluent import Fluent
from jaxcad.geometry.parameters import Parameter, Scalar, Vector

# Per-property bounds enforced when free=True.
_BOUNDS: dict[str, tuple] = {
    "color": (0.0, 1.0),
    "roughness": (0.0, 1.0),
    "metallic": (0.0, 1.0),
    "opacity": (0.0, 1.0),
    "ior": (1.0, 3.0),
    "reflectivity": (0.0, 1.0),
}


class Material(Fluent):
    """Material properties for physically-inspired shading.

    Properties are stored as Parameter objects in ``self.params`` (fixed by
    default), consistent with how SDF primitives handle their parameters.
    ``extract_parameters`` can therefore discover and—if marked free—optimise
    material properties alongside geometry.

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
        free: If True, wrap all raw values as free Parameters with sensible
            bounds.  Already-constructed Parameter objects are left unchanged.
            Requires ``name`` when any property is still a raw value.

    Example — fully free material in one line::

        body_mat = Material("body_mat", color=[0.5, 0.5, 0.5], roughness=0.4,
                            metallic=0.12, free=True)
        # → free params: body_mat_color, body_mat_roughness, body_mat_metallic,
        #                body_mat_opacity, body_mat_ior, body_mat_reflectivity
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
        free: bool = False,
    ):
        self.params = {
            "color": color if color is not None else [1.0, 1.0, 1.0],
            "roughness": roughness,
            "metallic": metallic,
            "opacity": opacity,
            "ior": ior,
            "reflectivity": reflectivity,
        }
        self._cast_params(name=name, free=free)

    def _cast_params(self, name: str | None = None, free: bool = False) -> None:  # type: ignore[override]
        """Convert raw values to Parameter objects, optionally marking them free.

        Extends the base implementation: when ``free=True``, raw values are
        wrapped as free Parameters with sensible bounds and names derived from
        ``name``.  Existing Parameter objects are always left unchanged.
        """
        for key, val in self.params.items():
            if isinstance(val, Parameter):
                continue  # already a Parameter — never overwrite
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
                from jaxcad.geometry.parameters import as_parameter

                self.params[key] = as_parameter(val)

    def children(self) -> list:
        return []

    def as_dict(self) -> dict:
        """Return material properties as a JAX-pytree-compatible dict of arrays."""
        return {k: (v.xyz if isinstance(v, Vector) else v.value) for k, v in self.params.items()}

    @staticmethod
    def blend(m1: dict, m2: dict, t: Array) -> dict:
        """Linearly interpolate between two material dicts.

        Args:
            m1: First material dict.
            m2: Second material dict.
            t: Blend factor; t=1 returns m1, t=0 returns m2.
        """
        return {k: m2[k] * (1.0 - t) + m1[k] * t for k in m1}
