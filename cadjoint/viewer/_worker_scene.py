"""Executing a playground program and resolving what it declared.

The one place the worker runs the user's Python: :func:`_execute_scene`
execs it inside the mesh/study/optimization capture registries and hands
back the resulting namespace (with the captured lists attached).  The
``_named_*`` helpers then pick one declared object out of those lists by
the name a request asked for, raising a message that lists the
alternatives when there is no single match.

``_FEM_UNAVAILABLE_MESSAGE`` lives here too: it is what every mode that
needs the optional solver extra reports when the import fails.
"""

from __future__ import annotations

from typing import Any

from cadjoint.viewer._source_map import PLAYGROUND_FILENAME


def _execute_scene(source: str) -> dict[str, Any]:
    """Run playground source and return its namespace (the scene lives inside).

    The exec always happens inside :func:`capture_sim_meshes` +
    :func:`capture_studies` + :func:`capture_optimizations` registries: a
    scene program that references a declared mesh by name (``mesh="..."``)
    can only resolve it through an active capture context, so every worker
    mode needs them, whether or not it looks at the captured lists
    afterwards.
    """
    from cadjoint.fem.simmesh import capture_sim_meshes
    from cadjoint.fem.study import capture_studies
    from cadjoint.optimize import capture_optimizations

    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "__name__": "__cadjoint_playground__",
    }
    with (
        capture_sim_meshes() as sim_meshes,
        capture_studies() as studies,
        capture_optimizations() as optimizations,
    ):
        exec(compile(source, PLAYGROUND_FILENAME, "exec"), namespace, namespace)
    if "scene" not in namespace:
        raise ValueError("Your program must assign the SDF to a variable named `scene`.")
    namespace["__sim_meshes__"] = sim_meshes
    namespace["__studies__"] = studies
    namespace["__optimizations__"] = optimizations
    return namespace


_FEM_UNAVAILABLE_MESSAGE = (
    "FEM simulation needs the 'fem' extra (jax-fem). Install it with: pip install cadjoint[fem]"
)


def _named_study(studies: list[Any], name: Any) -> Any:
    """The one declared study called *name* (or raise, listing the others)."""
    matches = [study for study in studies if study.name == name]
    if not matches:
        declared = ", ".join(repr(study.name) for study in studies) or "none"
        raise ValueError(f"The program declares no study named {name!r} (declared: {declared}).")
    if len(matches) > 1:
        raise ValueError(f"The program declares more than one study named {name!r}.")
    return matches[0]


def _named_optimization(optimizations: list[Any], name: Any) -> Any:
    """The one declared optimization called *name* (or raise, listing them)."""
    matches = [optimization for optimization in optimizations if optimization.name == name]
    if not matches:
        declared = ", ".join(repr(optimization.name) for optimization in optimizations) or "none"
        raise ValueError(
            f"The program declares no optimization named {name!r} (declared: {declared})."
        )
    if len(matches) > 1:
        raise ValueError(f"The program declares more than one optimization named {name!r}.")
    return matches[0]
