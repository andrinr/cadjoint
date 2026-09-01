"""The one error type every patch operation raises.

Kept in its own module so that formatting, span surgery, resolvers and the
operations can all raise it without importing each other — this is the bottom
of the package's dependency order.

:class:`PatchError` messages are user-facing: the playground puts them straight
in front of whoever clicked, and the test suite asserts on their wording, so
edit a message only when the behaviour it describes changes.
"""

from __future__ import annotations


class PatchError(ValueError):
    """Raised when a source edit cannot be applied safely."""
