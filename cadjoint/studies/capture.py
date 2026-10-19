"""The registry a declared study lands in, whatever physics it solves.

A scene program declares studies by constructing them; the compile worker
wraps the program in :func:`capture_studies` and receives them in
construction order.  Nothing here knows what a study is beyond a name and a
``describe()``, which is why it sits below both physics packages:
:class:`cadjoint.fem.ThermalStudy` discretises a mesh,
:class:`cadjoint.flow.FlowStudy` fills a lattice, and both land in the same
list.

One :class:`cadjoint.capture.Capture`, like the mesh, optimization and
constraint-solve registries.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

from cadjoint.capture import Capture

__all__ = ["capture_studies", "declared_studies", "register_study"]

_STUDIES: Capture[Any] = Capture("cadjoint_captured_studies")


def capture_studies() -> AbstractContextManager[list[Any]]:
    """Collect every study constructed inside this context.

    Yields:
        The list the studies land in, in construction order.
    """
    return _STUDIES.collect()


def declared_studies() -> list[Any]:
    """The studies declared so far in the active capture context.

    How a declaration resolves a *name* to the study it refers to — an
    ``Optimization(study="sink-conduction")`` naming one declared above it.
    Empty outside a capture context.

    Returns:
        The captured studies in construction order.  A copy: appending to
        it does not register anything.
    """
    return _STUDIES.declared()


def register_study(study: Any) -> None:
    """Add a study to the active :func:`capture_studies` context, if any.

    The registration hook the study classes call from ``__post_init__``.
    Outside a capture context this does nothing, which is what makes a study
    usable from a plain script.

    Args:
        study: Any object with a ``name`` and a ``describe()``.
    """
    _STUDIES.record(study)
