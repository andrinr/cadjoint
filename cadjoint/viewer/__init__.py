from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .widget import SDFViewer

__all__ = ["SDFViewer"]


def __getattr__(name: str):
    if name == "SDFViewer":
        from .widget import SDFViewer

        return SDFViewer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
