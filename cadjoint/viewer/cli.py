"""The ``cadjoint-viewer`` console script.

The playground imports pydantic at module scope, and pydantic is in the
``viewer`` extra, not the core install. A bare ``pip install cadjoint``
still installs the script, so it names the extra instead of failing with a
traceback.
"""

from __future__ import annotations

#: Modules the playground needs that only the ``viewer`` extra provides.
_VIEWER_EXTRA = {"pydantic", "psutil", "anywidget"}


def main() -> None:
    try:
        from cadjoint.viewer.playground import main as serve
    except ModuleNotFoundError as error:
        if error.name not in _VIEWER_EXTRA:
            raise
        raise SystemExit(
            f"cadjoint-viewer needs the viewer extra ({error.name} is not installed):\n"
            "    pip install 'cadjoint[viewer]'"
        ) from None
    serve()
