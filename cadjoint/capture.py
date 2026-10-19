"""Collecting what a scene program declares while it runs.

A scene program declares meshes, studies and optimizations by constructing
them, and reports constraint solves by running them.  The compile worker
wraps the program in a :meth:`Capture.collect` block for each kind and
receives them in construction order.  Outside a block, recording does
nothing, which is what keeps every declaration usable from a plain script.

Captures are per context (a :class:`~contextvars.ContextVar`): an inner
block shadows an outer one, and concurrent threads do not see each other's
declarations.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Generic, Protocol, TypeVar

__all__ = ["Capture", "Named", "one_named"]

T = TypeVar("T")


class Named(Protocol):
    name: str


N = TypeVar("N", bound=Named)


class Capture(Generic[T]):
    """One kind of declaration, collected per context.

    Args:
        name: The context variable's name, for debugging.
    """

    def __init__(self, name: str) -> None:
        self._active: ContextVar[list[T] | None] = ContextVar(name, default=None)

    @contextmanager
    def collect(self) -> Iterator[list[T]]:
        """Collect every item recorded inside this block, in order."""
        items: list[T] = []
        token = self._active.set(items)
        try:
            yield items
        finally:
            self._active.reset(token)

    @contextmanager
    def suspended(self) -> Iterator[None]:
        """Record nothing inside this block, even within a :meth:`collect`."""
        token = self._active.set(None)
        try:
            yield
        finally:
            self._active.reset(token)

    def record(self, item: T) -> None:
        """Add *item* to the innermost active :meth:`collect` block, if any."""
        items = self._active.get()
        if items is not None:
            items.append(item)

    def declared(self) -> list[T]:
        """A copy of what the active block has collected so far (empty outside one)."""
        return list(self._active.get() or ())


def one_named(items: Iterable[N], name: object, noun: str, *, hint: str = "") -> N:
    """The one item called *name*.

    Args:
        items: The declarations to search.
        name: The name asked for.
        noun: What the items are, for the error message.
        hint: Appended to the not-found message.

    Raises:
        ValueError: When no item or more than one is called *name*.
    """
    items = list(items)
    matches = [item for item in items if item.name == name]
    if len(matches) > 1:
        raise ValueError(f"The program declares more than one {noun} named {name!r}.")
    if not matches:
        declared = ", ".join(repr(item.name) for item in items) or "none"
        message = f"The program declares no {noun} named {name!r} (declared: {declared})."
        raise ValueError(f"{message} {hint}" if hint else message)
    return matches[0]
