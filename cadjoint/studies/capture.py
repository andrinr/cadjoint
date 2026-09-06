"""The registry a declared study lands in, whatever physics it solves.

A scene program declares studies by constructing them; the compile worker
wraps the program in :func:`capture_studies` and receives them in
construction order.  Nothing here knows what a study is beyond a name and a
``describe()``, which is why it sits below both physics packages:
:class:`cadjoint.fem.ThermalStudy` discretises a mesh,
:class:`cadjoint.flow.FlowStudy` fills a lattice, and both land in the same
list.

Mirrors ``capture_constraint_solves`` in :mod:`cadjoint.constraints.solve`
and ``capture_sim_meshes`` in :mod:`cadjoint.fem.simmesh`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

__all__ = ["capture_studies", "declared_studies", "register_study"]

_CAPTURED_STUDIES: ContextVar[list[Any] | None] = ContextVar(
    "cadjoint_captured_studies",
    default=None,
)


@contextmanager
def capture_studies() -> Iterator[list[Any]]:
    """Collect every study constructed inside this context.

    Yields:
        The list the studies land in, in construction order.
    """
    studies: list[Any] = []
    token = _CAPTURED_STUDIES.set(studies)
    try:
        yield studies
    finally:
        _CAPTURED_STUDIES.reset(token)


def declared_studies() -> list[Any]:
    """The studies declared so far in the active capture context.

    How a declaration resolves a *name* to the study it refers to — an
    ``Optimization(study="sink-conduction")`` naming one declared above it.
    Empty outside a capture context.

    Returns:
        The captured studies in construction order.  A copy: appending to
        it does not register anything.
    """
    return list(_CAPTURED_STUDIES.get() or ())


def register_study(study: Any) -> None:
    """Add a study to the active :func:`capture_studies` context, if any.

    The registration hook the study classes call from ``__post_init__``.
    Outside a capture context this does nothing, which is what makes a study
    usable from a plain script.

    Args:
        study: Any object with a ``name`` and a ``describe()``.
    """
    captured = _CAPTURED_STUDIES.get()
    if captured is not None:
        captured.append(study)
