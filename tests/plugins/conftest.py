"""Shared fixtures: x64, registry isolation, and a really-served plugin.

The remote transport is tested against a Tesseract that is actually served
over HTTP (``tesseract-runtime serve`` on a loopback port), not against a
mock: the whole claim of the ``remote`` transport is that a component
reached over the wire answers with the same numbers as one in this process,
and only a real server can be evidence for that.  The same mechanism is
what a Kubernetes Service exposes, so this is the cluster path in
miniature.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("tesseract_core")

_REPO = Path(__file__).resolve().parents[2]
#: Ports reserved for this suite (see the repository's port allocation).
_PORT_RANGE = range(4800, 4900)
#: How long to wait for a served Tesseract to answer /health.
_STARTUP_TIMEOUT = 90.0


@pytest.fixture(autouse=True, scope="package")
def _plugins_x64():
    """Scope jax's x64 mode to this suite (see ``tests/fem/conftest.py``)."""
    import jax

    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.fixture(autouse=True)
def _isolated_registry():
    """Keep every test off the developer's own ``plugins.toml``.

    Clears the process-wide registry before and after each test and hides
    ``$CADJOINT_PLUGINS``, so a machine that happens to have a config file
    cannot change what these tests measure.
    """
    from cadjoint.plugins import set_registry

    saved = os.environ.pop("CADJOINT_PLUGINS", None)
    old = set_registry(None)
    yield
    registry = set_registry(old)
    if registry is not None:
        registry.close()
    if saved is not None:
        os.environ["CADJOINT_PLUGINS"] = saved


def _free_port() -> int:
    """A free port from the range this suite is allowed to bind."""
    for port in _PORT_RANGE:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"no free port in {_PORT_RANGE.start}-{_PORT_RANGE.stop - 1}")


class ServedTesseract:
    """A ``tesseract-runtime serve`` subprocess and its URL.

    Attributes:
        url: Where the component answers.
        name: The name the runtime reports in its OpenAPI ``info.title``.
        version: The version it reports.
    """

    def __init__(self, package: Path, name: str, version: str):
        from cadjoint.plugins import SERVED_OUTPUT_ENV, runtime_scratch

        self.package = package
        self.name = name
        self.version = version
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        # A served runtime writes run_<uuid>/logs/ under its output path,
        # which defaults to its working directory -- that is what fills a
        # checkout with run_* directories. Point it at the same per-process
        # scratch PluginSpec gives the in-process transport.
        self.output_path = runtime_scratch()
        environment = dict(os.environ)
        environment.update(
            {
                "TESSERACT_API_PATH": str(package / "tesseract_api.py"),
                # The runtime reports "unknown" unless told; feeding it the
                # package's own name/version is what makes Plugin.probe's
                # version check mean something for a served component.
                "TESSERACT_NAME": name,
                "TESSERACT_VERSION": version,
                SERVED_OUTPUT_ENV: str(self.output_path),
            }
        )
        self.process = subprocess.Popen(  # noqa: S603
            [
                sys.executable,
                "-m",
                "tesseract_core.runtime.cli",
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
            ],
            cwd=str(_REPO),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    def wait_until_healthy(self) -> None:
        """Block until the server answers ``/health``, or fail the test."""
        import requests

        deadline = time.monotonic() + _STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                output = (self.process.stdout.read() or b"").decode(errors="replace")
                pytest.fail(f"tesseract-runtime serve exited early:\n{output}")
            try:
                response = requests.get(f"{self.url}/health", timeout=2.0)
                if response.ok:
                    return
            except Exception:  # noqa: BLE001 - the server is simply not up yet
                time.sleep(0.25)
        self.stop()
        pytest.fail(f"served Tesseract at {self.url} did not become healthy")

    def stop(self) -> None:
        """Terminate the server (SIGKILL if it will not go)."""
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:  # pragma: no cover
                self.process.kill()
                self.process.wait(timeout=15)
        if self.process.stdout is not None:
            self.process.stdout.close()


@pytest.fixture(scope="session")
def served_mesher():
    """The ``mesher`` package, served over HTTP for the whole session.

    The mesher is the package this suite serves because its frozen-topology
    payload contains no zero-size arrays: tesseract-core 1.11 validates
    polymorphic array dimensions as ``PositiveInt``, so an empty ``(0, …)``
    input cannot cross the HTTP boundary at all (the limitation is recorded
    in ``docs/plugins.qmd``).  It is also the cheapest package to boot.
    """
    package = _REPO / "cadjoint" / "fem" / "tesseracts" / "mesher"
    server = ServedTesseract(package, name="cadjoint_mesher", version="0.1.0")
    try:
        server.wait_until_healthy()
        yield server
    finally:
        server.stop()
