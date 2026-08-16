"""Per-primitive patch fields: exact surface ownership for feature edges.

Every hard-CSG primitive is internally a ``min``/``max`` composition over
smooth patch fields (a Box over six face half-spaces, a Cylinder over its
side and two caps, an ExtrudedPolygon over one half-plane per profile edge
plus two caps).  A surface point belongs to the patch whose field magnitude
is smallest, and the primitive's *exact* feature edges are where that
ownership switches — no angular threshold involved.

This module lifts the per-node :meth:`jaxcad.sdf.base.SDF.patch_fields`
protocol to whole scenes:

- :func:`world_frame_leaves` walks the Boolean structure into world-frame
  leaf subtrees (the same decomposition the viewer's seam detector uses).
- :func:`scene_patch_fields` collects each leaf's patch fields, falling back
  to the leaf's own SDF as a single opaque patch when the protocol is not
  implemented.
- :func:`patch_signatures` labels query points with ``(leaf_id, patch_id)``
  via ``argmin |field|``; the returned signature function is a pure JAX
  computation and composes with ``vmap``/``jit``.
- :func:`exact_feature_mask` marks mesh-vertex adjacencies whose endpoint
  signatures differ — exactly the edges that straddle an analytic feature
  curve (a primitive edge or a CSG seam).

Signatures are discrete labels; like edge sets and connectivity elsewhere in
the pipeline they are frozen per extraction, while the underlying fields stay
differentiable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


def world_frame_leaves(node: Any) -> list[Any]:
    """Maximal world-frame subtrees below the scene's Boolean structure.

    Hard CSG is built from ``min``/``max``, so the exact seam between two
    operands is where surface ownership switches between them.  Descend only
    through Boolean nodes: their children share the parent's coordinate
    frame and stay callable in world space, while anything else (a
    transformed subtree, a primitive) becomes one opaque leaf.

    Args:
        node: Root of an SDF scene tree.

    Returns:
        Leaf subtrees in depth-first order; their list index is the leaf id
        used throughout this module.
    """
    from jaxcad.sdf.boolean.base import BooleanOp

    if isinstance(node, BooleanOp):
        leaves: list[Any] = []
        for child in node.children():
            leaves.extend(world_frame_leaves(child))
        return leaves
    return [node]


class ScenePatchFields(NamedTuple):
    """Per-leaf patch decomposition of a scene.

    Attributes:
        leaves: World-frame leaf subtrees, in depth-first order.
        leaf_ids: Leaf id per leaf (its index; kept explicit for consumers).
        fields: One list of world-frame callables per leaf.  Leaves that
            implement :meth:`~jaxcad.sdf.base.SDF.patch_fields` contribute
            their exact decomposition; the rest fall back to their own SDF
            as a single patch (patch id 0).
        exact: Per leaf, whether ``fields`` came from the protocol (``True``)
            or from the single-patch fallback.
    """

    leaves: list[Any]
    leaf_ids: list[int]
    fields: list[list[Callable[[Array], Array]]]
    exact: list[bool]


def scene_patch_fields(scene: Any) -> ScenePatchFields:
    """Collect world-frame patch fields for every Boolean leaf of a scene.

    Args:
        scene: Root SDF node.

    Returns:
        :class:`ScenePatchFields` with one field list per leaf.
    """
    leaves = world_frame_leaves(scene)
    fields: list[list[Callable[[Array], Array]]] = []
    exact: list[bool] = []
    for leaf in leaves:
        patch = leaf.patch_fields() if hasattr(leaf, "patch_fields") else None
        if patch:
            fields.append(list(patch))
            exact.append(True)
        else:
            fields.append([lambda p, leaf=leaf: jnp.asarray(leaf(p))])
            exact.append(False)
    return ScenePatchFields(
        leaves=leaves, leaf_ids=list(range(len(leaves))), fields=fields, exact=exact
    )


def signature_function(scene: Any) -> Callable[[Array], tuple[Array, Array]]:
    """Build the pure ``(leaf_id, patch_id)`` signature function of a scene.

    The returned callable takes a single point shaped ``(3,)`` and returns
    two ``int32`` scalars: the owning leaf (``argmin |leaf sdf|``) and the
    owning patch within that leaf (``argmin |patch field|``).  It is a pure
    JAX computation — compose it with ``jax.vmap`` and ``jax.jit`` freely.

    Args:
        scene: Root SDF node, or a prebuilt :class:`ScenePatchFields`.

    Returns:
        Callable ``p -> (leaf_id, patch_id)``.
    """
    decomposition = scene if isinstance(scene, ScenePatchFields) else scene_patch_fields(scene)
    leaves = decomposition.leaves
    per_leaf_fields = decomposition.fields

    def signature(p: Array) -> tuple[Array, Array]:
        leaf_magnitudes = jnp.stack([jnp.abs(jnp.asarray(leaf(p))) for leaf in leaves])
        patch_ids = jnp.stack(
            [
                jnp.argmin(jnp.stack([jnp.abs(jnp.asarray(field(p))) for field in fields]))
                for fields in per_leaf_fields
            ]
        )
        leaf_id = jnp.argmin(leaf_magnitudes)
        return leaf_id.astype(jnp.int32), patch_ids[leaf_id].astype(jnp.int32)

    return signature


def patch_signatures(scene: Any, points: Array) -> tuple[np.ndarray, np.ndarray]:
    """Label points with their ``(leaf_id, patch_id)`` surface signature.

    Args:
        scene: Root SDF node, or a prebuilt :class:`ScenePatchFields`.
        points: Query points shaped ``(..., 3)`` (surface points for
            meaningful ownership; any points are accepted).

    Returns:
        Tuple of ``int32`` NumPy arrays ``(leaf_ids, patch_ids)``, each
        shaped like ``points`` without its last axis.
    """
    signature = signature_function(scene)
    array = jnp.asarray(points, dtype=jnp.float32)
    flat = array.reshape(-1, 3)
    leaf_ids, patch_ids = jax.vmap(signature)(flat)
    shape = array.shape[:-1]
    return (
        np.asarray(leaf_ids, dtype=np.int32).reshape(shape),
        np.asarray(patch_ids, dtype=np.int32).reshape(shape),
    )


def exact_feature_mask(
    leaf_ids: np.ndarray, patch_ids: np.ndarray, adjacency: np.ndarray
) -> np.ndarray:
    """Mark adjacencies whose endpoint signatures differ.

    A mesh edge whose two vertices carry different signatures straddles an
    exact feature: either a CSG seam (leaf change) or a primitive edge
    (patch change within one leaf).  Face interiors never trigger — both
    endpoints own the same smooth patch there.

    Args:
        leaf_ids: Per-vertex leaf ids, shaped ``(vertex_count,)``.
        patch_ids: Per-vertex patch ids, shaped ``(vertex_count,)``.
        adjacency: Vertex-index pairs shaped ``(edge_count, 2)`` (e.g. the
            quad edges of a dual-contour mesh).

    Returns:
        Boolean mask shaped ``(edge_count,)``: ``True`` where the signature
        changes across the edge.
    """
    leaf_ids = np.asarray(leaf_ids)
    patch_ids = np.asarray(patch_ids)
    pairs = np.asarray(adjacency, dtype=np.int64).reshape(-1, 2)
    first, second = pairs[:, 0], pairs[:, 1]
    return (leaf_ids[first] != leaf_ids[second]) | (patch_ids[first] != patch_ids[second])
