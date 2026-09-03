"""The runtime's ``run_<uuid>/`` scratch stays out of the working directory.

Every Tesseract endpoint call opens a ``run_<uuid>/logs/`` directory under
the runtime's output path.  Unconfigured, that path is the *working
directory* of whatever process runs the endpoint, so a served Tesseract
started from a checkout fills it with run directories — measured: 140 of
them, 560 KB, in this repository before :func:`runtime_scratch` existed, and
four more per served test run.  In-process it is an unmanaged ``mkdtemp``
that nothing ever removes.

:meth:`PluginSpec.open` passes the per-process scratch to
``Tesseract.from_tesseract_api`` as its ``output_path``, and a served
Tesseract is started with ``TESSERACT_OUTPUT_PATH`` pointing at the same
place (``tests/plugins/conftest.py``).  These tests hold both ends of that
down from a clean temporary cwd.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tesseract_core")

from cadjoint.plugins import (  # noqa: E402
    SCRATCH_ENV,
    PluginSpec,
    TesseractPlugin,
    get_plugin,
    runtime_scratch,
    scratch_root,
)

_REPO = Path(__file__).resolve().parents[2]
_TETFILL = _REPO / "cadjoint" / "fem" / "tesseracts" / "tetfill" / "tesseract_api.py"


def _tetrahedron_payload():
    """The smallest well-formed tetfill payload (one closed surface)."""
    return {
        "points": np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        "triangles": np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], np.int32),
        "element": np.int32(0),
        "min_ratio": np.float64(1.5),
        "min_dihedral": np.float64(10.0),
        "interior_points": np.zeros((0, 3)),
        "node_ids": np.zeros(0, np.int32),
        "cell_template": np.zeros((0, 4), np.int32),
    }


def _run_dirs(directory: Path) -> list[Path]:
    return sorted(directory.glob("run_*"))


class TestScratchLocation:
    def test_the_root_is_outside_any_checkout(self):
        """``~/.cache/cadjoint/tesseract-runs``, beside the JAX cache."""
        root = scratch_root()
        assert root.is_absolute()
        assert _REPO not in root.parents and root != _REPO

    def test_the_root_is_overridable(self, monkeypatch, tmp_path):
        monkeypatch.setenv(SCRATCH_ENV, str(tmp_path / "elsewhere"))
        assert scratch_root() == tmp_path / "elsewhere"

    def test_the_process_scratch_is_one_directory_under_it(self):
        scratch = runtime_scratch()
        assert scratch.is_dir()
        assert scratch.parent == scratch_root()
        # Stable within the process, so the whole run shares one directory.
        assert runtime_scratch() is scratch


class TestNothingLandsInTheWorkingDirectory:
    def test_a_local_apply_leaves_the_cwd_clean(self, tmp_path, monkeypatch):
        """The regression this exists for, from an empty temporary cwd."""
        monkeypatch.chdir(tmp_path)
        plugin = TesseractPlugin(
            PluginSpec(name="tetfill", kind="tetfill", transport="local", api_path=_TETFILL)
        )
        try:
            before = _run_dirs(runtime_scratch())
            result = plugin.apply(_tetrahedron_payload())
            assert np.asarray(result["nodes"]).shape[1] == 3
        finally:
            plugin.close()
        assert os.listdir(tmp_path) == []
        assert _run_dirs(tmp_path) == []
        # And it did write its run somewhere -- in the scratch, not here.
        assert len(_run_dirs(runtime_scratch())) > len(before)

    def test_the_registry_path_leaves_the_cwd_clean(self, tmp_path, monkeypatch):
        """The way every caller actually reaches a plugin."""
        monkeypatch.chdir(tmp_path)
        get_plugin("tetfill").apply(_tetrahedron_payload())
        assert _run_dirs(tmp_path) == []

    def test_a_traced_call_leaves_the_cwd_clean(self, tmp_path, monkeypatch):
        """The frozen chains call plugins through JAX, not through apply."""
        pytest.importorskip("tesseract_jax")
        import jax
        import jax.numpy as jnp

        monkeypatch.chdir(tmp_path)
        payload = _tetrahedron_payload()
        plugin = get_plugin("tetfill")
        # A frozen-topology payload, so the traced call has real shapes.
        found = plugin.apply(payload)
        frozen = dict(
            payload,
            node_ids=np.arange(np.asarray(found["nodes"]).shape[0], dtype=np.int32),
            cell_template=np.asarray(found["cells"]).astype(np.int32),
            interior_points=np.asarray(found["nodes"])[4:],
        )
        call = plugin.as_jax()
        static = {k: v for k, v in frozen.items() if k != "points"}

        def objective(points):
            with jax.ensure_compile_time_eval():
                return jnp.sum(call(dict(points=points, **static))["nodes"] ** 2)

        gradient = jax.grad(objective)(jnp.asarray(frozen["points"]))
        assert np.isfinite(np.asarray(gradient)).all()
        assert _run_dirs(tmp_path) == []

    def test_a_served_apply_writes_into_the_scratch_not_the_checkout(self, served_mesher):
        """The actual source of the run_* directories that accumulated here.

        The server's working directory *is* the repository (the fixture
        starts it there on purpose, so this test can fail if the fix
        regresses); only ``TESSERACT_OUTPUT_PATH`` keeps its run
        directories out.
        """
        assert served_mesher.output_path == runtime_scratch()
        remote = TesseractPlugin(
            PluginSpec(name="mesher", kind="mesher", transport="remote", url=served_mesher.url)
        )
        local = get_plugin("mesher")
        try:
            # A frozen HEX8 payload: no zero-size array, so it can cross HTTP.
            n = 4
            origin = np.array([-1.0, -1.0, -1.0])
            spacing = np.array([2.0 / n] * 3)
            axis = origin[0] + spacing[0] * np.arange(n + 1)
            x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
            static = {
                "field_values": np.sqrt(x**2 + y**2 + z**2) - 0.8,
                "origin": origin,
                "spacing": spacing,
                "element": np.int32(1),
                "sharp": np.int32(0),
                "min_ratio": np.float64(1.5),
                "min_dihedral": np.float64(10.0),
            }
            found = local.apply(
                dict(
                    static,
                    point_ids=np.zeros(0, np.int32),
                    cell_template=np.zeros((0, 8), np.int32),
                    num_surface=np.int32(0),
                )
            )
            frozen = dict(
                static,
                point_ids=np.arange(np.asarray(found["points"]).shape[0], dtype=np.int32),
                cell_template=np.zeros(np.asarray(found["cells"]).shape, np.int32),
                num_surface=np.int32(int(np.asarray(found["surface_mask"]).sum())),
            )
            before = len(_run_dirs(runtime_scratch()))
            repo_before = _run_dirs(_REPO)
            assert np.asarray(remote.apply(frozen)["points"]).shape[1] == 3
        finally:
            remote.close()
        assert _run_dirs(_REPO) == repo_before
        assert len(_run_dirs(runtime_scratch())) > before
