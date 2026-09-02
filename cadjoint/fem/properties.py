"""Per-element physical properties sampled from the scene's material field.

Every SDF node answers :meth:`~cadjoint.sdf.base.SDF.material_at`, and the
smooth booleans blend those answers (:meth:`~cadjoint.render.material.Material.blend`),
so a scene assembled from several materials already *is* a continuous field of
physical properties — a copper slug pressed into an aluminium sink has copper's
conductivity inside the slug, aluminium's outside, and a smooth transition
exactly as wide as the CSG blend that joins them.  This module turns that field
into the per-element arrays a solver consumes.

What belongs here: sampling the field on a mesh (element centroids), the
element volumes that turn a density field into a mass, and the quantization
rule that projects a blended field onto a finite set of named materials for
solvers that cannot express a per-element continuum (CalculiX).

What does *not* belong here: the finite-element formulations that *use* the
arrays (:mod:`cadjoint.fem.jaxfem`), patch resolution
(:mod:`cadjoint.fem.simulate`), or the declarative study layer
(:mod:`cadjoint.fem.study`).  Nothing here reads a mesh object — functions take
``(points, cells)`` arrays, so this module sits at the same layer as
:mod:`cadjoint.fem.quality` and the mesh modules stay free of it.

Differentiability: sampling is pure JAX.  ``points`` may be traced, so the
gradient of a solve flows back to the geometry two ways at once — through where
each centroid *is* (element motion) and through what the material field *says*
there (the smooth blend at an interface, whose position is a design parameter).
Material properties marked ``free`` are traced through
:meth:`~cadjoint.render.material.Material.as_dict` in the same pass, so an
optimizer can tune a conductivity and a shape in one gradient.

Sampling point: element **centroids**, one sample per element.  Quadrature-point
sampling would cost ``num_quads`` (8 for HEX8) SDF-tree evaluations per element
instead of one and would make the property field discontinuous *within* an
element, which the piecewise-constant-per-element assumption of
``internal_vars`` cannot represent anyway; the centroid value is the natural
piecewise-constant representative and keeps the material evaluation off the
solve's critical path.

Unspecified properties: a material that does not declare, say, a conductivity
carries NaN for it (see :mod:`cadjoint.render.material`), and a blend involving
it stays NaN.  Sampling therefore raises a specific error naming the property
and the number of offending elements rather than substituting a default.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from cadjoint.fem.elements import HEX_CORNER_OFFSETS

__all__ = [
    "FROM_MATERIAL",
    "cell_centroids",
    "cell_volumes",
    "maybe_sample_cell_property",
    "quantize_to_materials",
    "sample_cell_property",
    "sample_material_field",
    "total_mass",
]

#: Sentinel accepted wherever a study takes a material property scalar, asking
#: for the value to be sampled from the scene's material field per element.
FROM_MATERIAL = "material"

# 2-point Gauss-Legendre rule on [-1, 1]: exact for the trilinear hex Jacobian
# determinant (degree <= 3 per direction), so hex volumes below are exact for
# the same geometry map jax-fem integrates over.
_GAUSS_1D = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))

#: Natural coordinates of the eight HEX8 corners, in VTK/meshio order.
_HEX_CORNER_NATURAL = 2.0 * HEX_CORNER_OFFSETS.astype(np.float64) - 1.0


def _corner_slots(cells: Any) -> int:
    """How many leading connectivity slots are the element's straight corners."""
    width = int(np.asarray(cells).shape[1])
    if width == 8:
        return 8
    if width in (4, 10):
        return 4
    raise ValueError(f"Unsupported element width {width}; expected 8 (HEX8), 4 or 10 (TET).")


def cell_centroids(points: Any, cells: Any) -> Any:
    """Element centroids, ``(C, 3)``, differentiable in ``points``.

    Computed from the element's straight corners only: a TET10's midside nodes
    would bias the mean away from the centroid of the straight-sided tet the
    mesher actually produced.

    Args:
        points: Node positions ``(N, 3)`` (may be traced).
        cells: Connectivity ``(C, 8)`` (HEX8), ``(C, 4)`` or ``(C, 10)`` (tets).

    Returns:
        A JAX array of element centroids, ``(C, 3)``.
    """
    import jax.numpy as jnp

    slots = _corner_slots(cells)
    corners = jnp.asarray(points)[np.asarray(cells)[:, :slots]]
    return jnp.mean(corners, axis=1)


def cell_volumes(points: Any, cells: Any) -> Any:
    """Element volumes, ``(C,)``, differentiable in ``points``.

    HEX8 volumes integrate the trilinear geometry map's Jacobian determinant
    with the 2x2x2 Gauss rule, which is *exact* for that map — the same volume
    jax-fem's own quadrature sees.  Tet volumes are ``|det| / 6`` over the
    straight corners (TET10 meshes here are straight-sided).

    Args:
        points: Node positions ``(N, 3)`` (may be traced).
        cells: Connectivity ``(C, 8)``, ``(C, 4)`` or ``(C, 10)``.

    Returns:
        A JAX array of absolute element volumes, ``(C,)``.
    """
    import jax.numpy as jnp

    slots = _corner_slots(cells)
    corners = jnp.asarray(points)[np.asarray(cells)[:, :slots]]
    if slots == 4:
        edges = corners[:, 1:, :] - corners[:, :1, :]  # (C, 3, 3)
        return jnp.abs(jnp.linalg.det(edges)) / 6.0
    natural = jnp.asarray(_HEX_CORNER_NATURAL)  # (8, 3)
    quads = jnp.asarray(
        [(x, y, z) for x in _GAUSS_1D for y in _GAUSS_1D for z in _GAUSS_1D]
    )  # (8, 3)
    # dN_a/dxi_d at each quadrature point: (Q, 8, 3).
    plus = 1.0 + quads[:, None, :] * natural[None, :, :]  # (Q, 8, 3)
    grads = (
        jnp.stack(
            [
                natural[None, :, 0] * plus[:, :, 1] * plus[:, :, 2],
                natural[None, :, 1] * plus[:, :, 0] * plus[:, :, 2],
                natural[None, :, 2] * plus[:, :, 0] * plus[:, :, 1],
            ],
            axis=-1,
        )
        / 8.0
    )
    jacobians = jnp.einsum("qad,can->cqnd", grads, corners)  # (C, Q, 3, 3)
    return jnp.sum(jnp.abs(jnp.linalg.det(jacobians)), axis=1)


def _material_field(sdf: Any):
    """The scene's ``material_at`` callable, or a clear error explaining why not."""
    material_at = getattr(sdf, "material_at", None)
    if material_at is None:
        raise TypeError(
            "Sampling material properties needs an SDF object that answers "
            f"material_at(p); got {type(sdf).__name__}. Pass the scene SDF (not a "
            "bare callable field), or give the study an explicit scalar property."
        )
    return material_at


def sample_material_field(sdf: Any, points: Any, cells: Any) -> dict[str, Any]:
    """Sample the whole material field at every element centroid.

    Args:
        sdf: Scene SDF object answering ``material_at(p) -> dict``.
        points: Node positions ``(N, 3)`` (may be traced).
        cells: Connectivity ``(C, ...)``.

    Returns:
        The material dict with every value batched over elements: scalar
        properties shaped ``(C,)`` and ``color`` shaped ``(C, 3)``.
    """
    import jax

    return jax.vmap(_material_field(sdf))(cell_centroids(points, cells))


def _is_traced(values: Any) -> bool:
    """True when ``values`` is a JAX tracer (so concrete checks must be skipped)."""
    import jax

    return isinstance(values, jax.core.Tracer)


def sample_cell_property(sdf: Any, points: Any, cells: Any, key: str, *, label: str = "") -> Any:
    """Sample one physical property per element, validated.

    Args:
        sdf: Scene SDF object answering ``material_at(p) -> dict``.
        points: Node positions ``(N, 3)`` (may be traced).
        cells: Connectivity ``(C, ...)``.
        key: Property name, e.g. ``"conductivity"`` or ``"youngs_modulus"``.
        label: Optional context prefix for error messages (a study name).

    Returns:
        A JAX array of per-element values, ``(C,)``, differentiable in
        ``points`` and in any ``free`` material parameter it reads.

    Raises:
        ValueError: If the sampled field is NaN anywhere — i.e. some element
            sits in (or blends with) a material that does not specify the
            property.  Concrete samples only; a traced sample cannot be
            inspected and is passed through.
    """
    import jax
    import jax.numpy as jnp

    material_at = _material_field(sdf)
    centroids = cell_centroids(points, cells)
    values = jax.vmap(lambda point: jnp.asarray(material_at(point)[key]).reshape(()))(centroids)
    if not _is_traced(values):
        missing = int(np.count_nonzero(np.isnan(np.asarray(values))))
        if missing:
            where = f"{label}: " if label else ""
            raise ValueError(
                f"{where}the scene's material field does not specify {key!r} for "
                f"{missing} of {values.shape[0]} elements. Give every material in the "
                f"simulated domain a {key!r} value (see cadjoint.materials for a "
                "catalogue of real ones), or pass an explicit scalar to the study."
            )
    return values


def maybe_sample_cell_property(
    sdf: Any,
    points: Any,
    cells: Any,
    key: str,
    *,
    base_points: Any = None,
) -> Any | None:
    """Sample a property per element, or return None when the scene lacks it.

    The permissive counterpart of :func:`sample_cell_property`, for properties a
    study *reports on* rather than *solves with* — mass needs a density, but a
    study whose materials never mention one should still solve and simply not
    report a mass.

    The presence check always runs on concrete positions (``base_points``, else
    ``points`` when it is concrete), because a traced sample cannot be inspected
    for NaN; only once the property is known to be specified is it re-sampled at
    the possibly-traced ``points`` so the value stays differentiable.

    Args:
        sdf: Scene SDF object, or anything else (which yields None).
        points: Node positions ``(N, 3)`` (may be traced).
        cells: Connectivity ``(C, ...)``.
        key: Property name.
        base_points: Concrete node positions for the presence check; defaults
            to ``points``.

    Returns:
        A ``(C,)`` array of per-element values, or None when the scene has no
        material field or does not specify ``key`` everywhere.
    """
    from cadjoint.render.material import specifies_everywhere

    if getattr(sdf, "material_at", None) is None:
        return None
    if hasattr(sdf, "children") and not specifies_everywhere(sdf, key):
        # Structural, and the reason a scene that never mentions a density does
        # not pay for a density lookup on every solve.
        return None
    probe_points = points if base_points is None else base_points
    if _is_traced(probe_points):
        return None
    try:
        probe = sample_cell_property(sdf, probe_points, cells, key)
    except (TypeError, ValueError):
        return None
    if probe_points is points:
        return probe
    return sample_cell_property(sdf, points, cells, key)


def total_mass(points: Any, cells: Any, density: Any) -> Any:
    """Mass of the meshed domain, ``sum(rho_e * V_e)``.

    Args:
        points: Node positions ``(N, 3)`` (may be traced).
        cells: Connectivity ``(C, ...)``.
        density: Scalar density, or per-element densities ``(C,)``.

    Returns:
        A JAX scalar mass in kg (given SI geometry), differentiable in both
        ``points`` and ``density``.
    """
    import jax.numpy as jnp

    return jnp.sum(jnp.asarray(density) * cell_volumes(points, cells))


def quantize_to_materials(
    values: dict[str, Any],
    references: Sequence[Any],
    *,
    keys: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Snap a per-element property field onto a finite set of named materials.

    Solvers that describe materials by *name* rather than by per-element arrays
    — CalculiX being the case in hand, with its ``*MATERIAL`` / ``*ELSET`` /
    ``*SOLID SECTION`` triple — cannot represent a continuously blended
    interface at all.  This projects each element onto its nearest reference
    material in log-property space (properties span many decades, so a relative
    metric is the only sane one), which is exact wherever the blend is sharp
    and approximate exactly in the transition band.

    Args:
        values: Per-element property arrays keyed by property name, ``(C,)``
            each (concrete; quantization is not differentiable).
        references: Reference :class:`~cadjoint.render.material.Material`
            objects to snap to.
        keys: Which properties participate in the distance.

    Returns:
        ``(assignment, error)`` — ``assignment`` is a ``(C,)`` int array of
        indices into ``references``, and ``error`` is the ``(C,)`` maximum
        relative property error the snap introduced (0 where an element landed
        exactly on a reference material).

    Raises:
        ValueError: If ``references`` is empty or a reference does not specify
            one of ``keys``.
    """
    if not references:
        raise ValueError("quantize_to_materials needs at least one reference material.")
    table = np.empty((len(references), len(keys)), dtype=np.float64)
    for row, material in enumerate(references):
        for column, key in enumerate(keys):
            value = material.get(key) if hasattr(material, "get") else None
            if value is None:
                name = getattr(material, "name", None) or f"reference {row}"
                raise ValueError(f"Reference material {name!r} does not specify {key!r}.")
            table[row, column] = float(value)
    sampled = np.stack([np.asarray(values[key], dtype=np.float64) for key in keys], axis=1)
    # Log-space distance: 'twice as stiff' is the same error whether the
    # modulus is in MPa or GPa.  Poisson-like ratios sit near 1 and behave the
    # same either way, so one metric covers both.
    floor = 1e-30
    log_sampled = np.log(np.maximum(sampled, floor))
    log_table = np.log(np.maximum(table, floor))
    distances = np.linalg.norm(log_sampled[:, None, :] - log_table[None, :, :], axis=2)
    assignment = np.argmin(distances, axis=1).astype(np.int64)
    chosen = table[assignment]
    error = np.max(
        np.abs(chosen - sampled) / np.maximum(np.abs(sampled), floor),
        axis=1,
    )
    return assignment, error
