"""Helpers several test packages share: x64 scope and a live playground."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.request import Request, urlopen

#: The repository's own scenes, copied into every live server's workspace.
SHIPPED_SCENES = Path(__file__).resolve().parents[1] / "scenes"


@contextmanager
def x64() -> Iterator[None]:
    """Double precision for the block, restored on the way out.

    ``jax_enable_x64`` is process-global: left on, every later module's
    arrays become float64, and the WGSL emitter — which has no 64-bit
    numeric type — refuses scenes it should accept.
    """
    import jax

    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@contextmanager
def running_server() -> Iterator[str]:
    """A live playground on a free port, writing scenes to a temp directory.

    Each server gets its own scenes directory, seeded with copies of the
    shipped scenes, so a request that saves a scene never writes into the
    checkout and the listing has something real in it.
    """
    from cadjoint.viewer._scenes import SCENES_DIR_ENV
    from cadjoint.viewer.playground import create_server

    with tempfile.TemporaryDirectory() as workspace:
        root = Path(workspace) / "scenes"
        root.mkdir()
        for path in sorted(SHIPPED_SCENES.glob("*.py")):
            shutil.copyfile(path, root / path.name)
        previous = os.environ.get(SCENES_DIR_ENV)
        os.environ[SCENES_DIR_ENV] = str(root)
        server = create_server(0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address
            yield f"http://{host}:{port}"
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
            if previous is None:
                os.environ.pop(SCENES_DIR_ENV, None)
            else:
                os.environ[SCENES_DIR_ENV] = previous


def session_token(base: str) -> str:
    """The token a live server hands out at ``/api/session``."""
    with urlopen(f"{base}/api/session") as response:
        return json.loads(response.read())["token"]


def post(base: str, path: str, payload: dict, token: str | None = None) -> Request:
    """A JSON ``POST`` to *path*, with the session token when one is given."""
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["X-Cadjoint-Token"] = token
    return Request(base + path, data=json.dumps(payload).encode(), headers=headers, method="POST")
