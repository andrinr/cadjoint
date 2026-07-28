"""Lazy matplotlib access for the optional display helpers."""

from __future__ import annotations

from types import ModuleType


def require_matplotlib() -> ModuleType:
    """Return ``matplotlib.pyplot``, with an actionable error when it is missing.

    Matplotlib is only needed by the display helpers, so it stays an optional
    dependency and is imported on first use instead of at import time.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError(
            "This helper requires matplotlib. Install with: pip install matplotlib"
        ) from err
    return plt
