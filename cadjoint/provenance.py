"""Where a program built its objects: the construction-capture hook.

An editor that shows a program's model next to its source has to pair each
object with the call that built it — including objects passed straight into
``extrude()`` or ``Union()`` and never bound to a name, which no reading of
the source alone can place.  So the classes worth pairing report their own
construction: :func:`records_construction` marks a class, and while a
:func:`capture_constructions` block is active every instance built inside it
is handed to the block's callback together with its position in one source
file.  :func:`record_placement` does the same for the placed SDF a
:class:`~cadjoint.construction.solid.ConstructionPrimitive` builds, so the
SDF graph node can be paired with the construction node it came from.

Outside a capture block a marked constructor costs one context-variable
read.  Mirrors ``capture_constraint_solves`` in
:mod:`cadjoint.constraints.solve` and ``capture_studies`` in
:mod:`cadjoint.studies.capture`: a :class:`~contextvars.ContextVar`, so a
capture belongs to the thread (and task) that opened it.

The marked classes are :class:`~cadjoint.construction.sketch.PolygonProfile`,
:class:`~cadjoint.construction.solid.ConstructionPrimitive` and the four
plain booleans (:class:`~cadjoint.sdf.boolean.Union`,
:class:`~cadjoint.sdf.boolean.Difference`,
:class:`~cadjoint.sdf.boolean.Intersection`,
:class:`~cadjoint.sdf.boolean.Xor`).
"""

from __future__ import annotations

import functools
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, TypeVar

__all__ = [
    "Site",
    "capture_constructions",
    "record_placement",
    "records_construction",
]

_T = TypeVar("_T", bound=type)


@dataclass(frozen=True)
class Site:
    """Where in the captured source file an object was constructed.

    Attributes:
        line: Line of the innermost frame executing the file — the program
            line that (directly or through library code) built the object —
            or None when the construction did not originate from it.
        direct: Whether that frame called the constructor itself, rather
            than through library code (``a | b``, a ``PolygonProfile.circle``
            helper, a subclass's ``super().__init__``).
    """

    line: int | None
    direct: bool


@dataclass(frozen=True)
class _Capture:
    filename: str
    constructed: Callable[[Any, Site], None] | None
    placed: Callable[[Any, Any], None] | None


_ACTIVE: ContextVar[tuple[_Capture, ...]] = ContextVar("cadjoint_construction_captures", default=())


@contextmanager
def capture_constructions(
    filename: str,
    *,
    constructed: Callable[[Any, Site], None] | None = None,
    placed: Callable[[Any, Any], None] | None = None,
) -> Iterator[None]:
    """Report every marked object constructed inside this block.

    Blocks nest: each active one receives every event.

    Args:
        filename: The source file positions are reported in — the
            ``co_filename`` the program was compiled under.
        constructed: Called as ``constructed(obj, site)`` after a marked
            class's ``__init__`` returns, in construction order.
        placed: Called as ``placed(primitive, sdf)`` each time a
            construction primitive builds its placed SDF.
    """
    token = _ACTIVE.set((*_ACTIVE.get(), _Capture(filename, constructed, placed)))
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def _site(caller: Any, filename: str) -> Site:
    frame = caller
    while frame is not None:
        if frame.f_code.co_filename == filename:
            return Site(frame.f_lineno, frame is caller)
        frame = frame.f_back
    return Site(None, False)


def records_construction(cls: _T) -> _T:
    """Class decorator: report each instance to the active captures.

    Wraps the class's ``__init__`` *as it stands when the decorator runs* —
    for an SDF that is the ``_cast_params`` wrapper ``SDF.__init_subclass__``
    installs, so the object is reported fully initialised.  The reported
    site is that of the constructor's caller.
    """
    init = cls.__init__

    @functools.wraps(init)
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        init(self, *args, **kwargs)
        captures = _ACTIVE.get()
        if not captures:
            return
        caller = sys._getframe(1)
        try:
            for capture in captures:
                if capture.constructed is not None:
                    capture.constructed(self, _site(caller, capture.filename))
        finally:
            del caller

    cls.__init__ = __init__  # type: ignore[misc]
    return cls


def record_placement(primitive: Any, sdf: Any) -> None:
    """Report that ``primitive`` built the placed SDF ``sdf``.

    Called by :meth:`~cadjoint.construction.solid.ConstructionPrimitive.sdf`;
    does nothing outside a capture block.
    """
    for capture in _ACTIVE.get():
        if capture.placed is not None:
            capture.placed(primitive, sdf)
