# `cadjoint-scene` graph format

A cadjoint SDF tree as JSON. The browser evaluates it on jax-js and builds its
WGSL from it, and both sides test against it. `scene.py` writes and reads it.

```json
{"format": "cadjoint-scene", "version": 2, "root": { ...node... }}
```

| version | written by | read by |
| --- | --- | --- |
| 1 | `scene_graph(sdf)` (the default) | `graph_sdf`, the jax-js runtime (`web-spike/src/runtime.ts`), `frontend/src/scene` |
| 2 | `scene_graph(sdf, version=2)` | `graph_sdf`, `frontend/src/scene` |

Version 2 only adds fields. Every version 2 graph is a valid version 1 graph
once its `version` is set to 1, which is how the static viewer hands one to
the jax-js runtime.

## Node (version 1)

| field | type | meaning |
| --- | --- | --- |
| `type` | string | A node type from `kernels.NODE_TYPES`: `sphere`, `box`, `round_box`, `cylinder`, `capsule`, `torus`, `plane`, `extruded_polygon`, `revolved_polygon`, `lofted_polygon`, `translate`, `rotate`, `scale`, `twist`, `mirror`, `offset`, `shell`, `union`, `difference`, `intersection`, `xor`, `linear_pattern`, `polar_pattern`. |
| `params` | object | The node type's parameters by kernel name. Scalars are numbers and vectors are lists. Pattern `count` and `skip_mask` are integers. |
| `children` | list | Child nodes in operand order: one for transforms and patterns, two or more for booleans, none for primitives. |

Parameters of the profile primitives:

| type | parameters |
| --- | --- |
| `extruded_polygon` | `depth`, `draft` (degrees), `twist` (degrees over the depth), `vertices` |
| `revolved_polygon` | `offset`, `vertices` |
| `lofted_polygon` | `height`, `vertices`, `vertices_b` (the same count as `vertices`) |

`vertices` is a list of `[x, y]` pairs, at most 64 of them.

## Node fields added in version 2

| field | type | present | meaning |
| --- | --- | --- | --- |
| `id` | string | always | `n0`, `n1`, … in first-visit (depth-first, operand) order. A subtree reached twice is written out at both places with the same ids, so a reader can emit it once. |
| `free` | object | when the node has free parameters | Graph parameter name → the parameter's declared name, for each free parameter. For `vertices` and `vertices_b` the value is a list with one entry per vertex, `null` for a fixed one. Two nodes giving the same declared name share one design variable. |
| `implicit` | list of strings | `extruded_polygon`, when `draft` or `twist` was never declared | Those parameters are written as `0.0` but are not part of the node. This matters to bounds: a declared draft, even a zero one, means the node gives no culling box. |
| `material` | object | every primitive | `color` (`[r, g, b]`), `roughness`, `metallic`, `opacity`, `ior`, `reflectivity`: the primitive's material at the origin, the two rows of a shader's material block. A non-finite value is `null`. |

A version 1 graph read as version 2 has no shared subtrees, no free
parameters, an `implicit` list holding each zero `draft` and `twist`, and the
default `Material()` on every primitive.

## Shaders from a graph

`frontend/src/scene/direct.ts` writes the viewer's scene module from a version
2 graph. In the uniform form it is the module `cadjoint.backends.wgsl.direct`
writes for the same scene, byte for byte. Its parameter slots have the same
names, and the same values except the culling boxes, which agree to float32
rounding. `tests/backends/test_wgsl_frontend.py` checks this on every shipped
scene and runs both modules on a device. The types the direct backend has no
kernel for (`round_box`, `plane`, `twist`, `xor`, `scale`) are refused on both
sides.
