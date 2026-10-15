"""A ``container`` spec naming a registry image pulls it when it is missing.

tesseract-core never pulls (``Tesseract.from_image`` inspects the local
store, and its fallback runs with ``--pull never``), so the prebuilt images
on GHCR would be unusable from a spec without a manual ``docker pull``.
These tests stub ``docker`` and check which commands :func:`_pull_if_missing`
issues; the real pull is exercised by the ``tesseract-images`` workflow.
"""

from __future__ import annotations

import subprocess

import pytest

from cadjoint.plugins import spec
from cadjoint.plugins.spec import PluginConfigError, _pull_if_missing


@pytest.fixture
def docker(monkeypatch):
    """Record docker invocations; ``present`` and ``pull_ok`` steer them."""
    state = {"calls": [], "present": False, "pull_ok": True}

    def run(argv, **_):
        state["calls"].append(argv[1:])
        if argv[1:3] == ["image", "inspect"]:
            code = 0 if state["present"] else 1
            return subprocess.CompletedProcess(argv, code, "", "")
        if argv[1] == "pull":
            code = 0 if state["pull_ok"] else 1
            return subprocess.CompletedProcess(argv, code, "", "denied: not found")
        raise AssertionError(f"unexpected docker call {argv}")

    monkeypatch.setattr(spec.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(subprocess, "run", run)
    return state


IMAGE = "ghcr.io/andrinr/cadjoint_mesher:latest"


def test_missing_registry_image_is_pulled(docker):
    _pull_if_missing(IMAGE)
    assert docker["calls"] == [["image", "inspect", IMAGE], ["pull", IMAGE]]


def test_present_image_is_not_pulled(docker):
    docker["present"] = True
    _pull_if_missing(IMAGE)
    assert docker["calls"] == [["image", "inspect", IMAGE]]


def test_bare_local_name_is_never_pulled(docker):
    # A `tesseract build` output; pulling it would ask Docker Hub for
    # library/cadjoint_mesher.
    _pull_if_missing("cadjoint_mesher:latest")
    assert docker["calls"] == []


def test_failed_pull_names_the_image_and_reason(docker):
    docker["pull_ok"] = False
    with pytest.raises(PluginConfigError, match="cadjoint_mesher.*denied"):
        _pull_if_missing(IMAGE)


def test_without_docker_it_defers_to_tesseract_core(docker, monkeypatch):
    monkeypatch.setattr(spec.shutil, "which", lambda name: None)
    _pull_if_missing(IMAGE)
    assert docker["calls"] == []
