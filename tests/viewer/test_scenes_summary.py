"""The static description of a saved scene, and where saved scenes live.

``GET /api/scenes`` carries a *description* of every file beside the bare
names, so a browser can show what a program contains before anyone opens it.
The whole point of these tests is the word *static*: the description comes
from one :mod:`ast` pass, so listing a directory must never import, execute,
or even open a Python module for its side effects.  A scene browser that ran
every program in the directory to describe it would be a browser that runs
arbitrary code on a directory listing.

The second half pins where the directory is.  It defaults to ``./scenes``
under the server's working directory — the repository's own scenes when the
playground is started from a checkout — and an automated run points
``CADJOINT_SCENES_DIR`` at a temporary copy so that nothing a test saves ever
lands in the repository.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from cadjoint.extraction import extract_parameters
from cadjoint.viewer._compile_worker import _compile_source
from cadjoint.viewer._scenes import (
    MAX_SUMMARY_CHARS,
    SCENES_DIR_ENV,
    describe_scene,
    docstring_summary,
    list_scenes,
    save_scene,
    scenes_root,
    summarize_scene,
)
from cadjoint.viewer._worker_scene import _execute_scene

SCENES = Path(__file__).resolve().parents[2] / "scenes"
SHIPPED = ["starter.py", "bracket.py", "end_cap.py"]


@pytest.fixture
def workspace(tmp_path, monkeypatch) -> Path:
    """A scenes directory of the shipped scenes, outside the repository."""
    root = tmp_path / "scenes"
    root.mkdir()
    for name in SHIPPED:
        (root / name).write_text((SCENES / name).read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv(SCENES_DIR_ENV, str(root))
    return root


class TestTheSummaryIsRead:
    """One ast pass, and nothing else."""

    @pytest.mark.parametrize("name", SHIPPED)
    def test_a_shipped_scene_is_described(self, name: str):
        entry = describe_scene(SCENES / name)

        assert entry["name"] == name
        assert entry["error"] is None
        # A sha256 of the bytes on disk: the same digest the job registry
        # stamps on a request, which is what makes it a cache key.
        assert entry["source_hash"] is not None
        assert len(entry["source_hash"]) == 64
        assert entry["bytes"] > 0
        assert entry["modified"] is not None and entry["modified"].endswith("Z")
        # Every shipped scene opens with a one-line title in its docstring.
        assert entry["summary"]
        assert "\n" not in entry["summary"]

    @pytest.mark.parametrize("name", SHIPPED)
    def test_the_free_count_is_checked_against_what_actually_runs(self, name: str):
        """Every free parameter counted statically is one the optimizer sees.

        This is the test a wrong counter fails.  A scene's design freedom
        lives mostly *inside* its sketches — the starter declares three
        top-level ``Scalar`` parameters and nineteen free ``Vector2`` points —
        so a counter that looked only at top-level ``Scalar`` assignments
        reported 3 for a program whose optimizer drives 19.  The expectation
        is therefore derived rather than written down: execute the scene the
        way the worker does and ask
        :func:`~cadjoint.extraction.extract_parameters` what it found.

        The relation is containment, not equality, and deliberately so.  The
        static count is of what the *source says*: a call carrying both
        ``name=`` and ``free=True``.  The runtime set is larger, because a
        named primitive hands its own placement and dimensions over as free
        parameters without the program writing ``free=True`` anywhere — the
        starter's ``Solid.box(..., name="board")`` contributes
        ``board_position`` and ``board_size``.  Guessing at that from the ast
        would mean encoding the library's defaults in a file that must never
        execute the library, so the card reports the honest, conservative
        number: what you can read in the program.
        """
        namespace = _execute_scene((SCENES / name).read_text(encoding="utf-8"))
        free, _fixed, _metadata = extract_parameters(namespace["scene"])
        counts = describe_scene(SCENES / name)["counts"]

        assert 0 < counts["free"] <= len(free)
        # Every free parameter is a named one; the named set is larger,
        # because a fixed `name=` is still a parameter you can point at.
        assert counts["parameters"] >= counts["free"]

    def test_the_starter_s_free_count_is_the_run_the_panel_offers(self):
        """For the starter the two senses coincide, so pin the number.

        `cool-sink` drives every parameter the program declares free, so the
        card's `FREE 19` and the Optimize panel's parameter list are the same
        nineteen names. That is the case the user actually looked at.
        """
        payload = _compile_source((SCENES / "starter.py").read_text(encoding="utf-8"))
        driven = len(payload["optimizations"][0]["parameters"])
        counts = describe_scene(SCENES / "starter.py")["counts"]

        assert driven == 19
        assert counts["free"] == driven

    def test_the_starter_counts_what_it_declares(self):
        counts = describe_scene(SCENES / "starter.py")["counts"]

        assert counts["free"] == 19
        assert counts["parameters"] == 25
        assert counts["studies"] == 1
        assert counts["meshes"] == 1
        assert counts["optimizations"] == 1

    def test_the_bracket_counts_what_it_declares(self):
        entry = describe_scene(SCENES / "bracket.py")

        assert entry["counts"]["free"] == 15
        assert entry["counts"]["studies"] == 1
        assert entry["counts"]["optimizations"] == 1
        assert "steel" in entry["materials"]

    def test_the_end_cap_is_the_deep_one(self):
        entry = describe_scene(SCENES / "end_cap.py")

        assert entry["summary"].startswith("Gearbox output end-cap")
        assert entry["counts"]["parameters"] >= 10
        assert entry["counts"]["studies"] == 1
        # It declares no optimization, which the browser must show as 0
        # rather than as "unknown".
        assert entry["counts"]["optimizations"] == 0
        assert "aluminum" in entry["materials"]

    def test_a_free_parameter_needs_both_a_name_and_free(self):
        summary = summarize_scene(
            "a = Scalar(1.0, name='a', free=True)\nb = Scalar(2.0, name='b')\nc = Scalar(3.0)\n"
        )

        assert summary["counts"]["parameters"] == 2
        assert summary["counts"]["free"] == 1

    def test_a_parameter_counts_wherever_it_is_written(self):
        # Nested inside a call, inside a list, inside a comprehension: a
        # sketch declares its points this way and they are most of the
        # design freedom in a real scene.
        summary = summarize_scene(
            "profile = PolygonProfile([\n"
            "    Vector2([0, 0], name='a', free=True),\n"
            "    Vector2([1, 0], name='b', free=True),\n"
            "])\n"
        )

        assert summary["counts"]["parameters"] == 2
        assert summary["counts"]["free"] == 2

    def test_materials_are_named_once_each(self):
        summary = summarize_scene(
            "x = Material(name='steel')\ny = Material(name='steel')\nz = Material(name='brass')\n"
        )

        assert summary["counts"]["materials"] == 3
        assert summary["materials"] == ["steel", "brass"]

    def test_only_the_first_paragraph_is_kept(self):
        module = "\n".join(['"""Title line', "spilling onto a second.", "", "Body.", '"""'])

        assert docstring_summary(__import__("ast").parse(module)) == (
            "Title line spilling onto a second."
        )

    def test_a_long_paragraph_is_cut(self):
        summary = summarize_scene('"""' + "word " * 200 + '"""')["summary"]

        assert len(summary) <= MAX_SUMMARY_CHARS
        assert summary.endswith("…")

    def test_a_broken_scene_is_still_listed(self):
        summary = summarize_scene("def (:\n")

        assert summary["error"] is not None
        assert summary["counts"]["studies"] == 0
        assert summary["summary"] == ""

    def test_describing_a_scene_never_imports_it(self, workspace: Path):
        # The proof, rather than the promise: a module that would blow up on
        # import describes fine, and never appears in sys.modules.
        (workspace / "landmine.py").write_text(
            '"""A scene that refuses to be imported."""\n'
            "raise SystemExit('this must never run')\n"
            "study = ThermalStudy(name='never')\n",
            encoding="utf-8",
        )

        listing = list_scenes()
        entry = next(item for item in listing["scenes"] if item["name"] == "landmine.py")

        assert entry["summary"] == "A scene that refuses to be imported."
        assert entry["counts"]["studies"] == 1
        assert "landmine" not in sys.modules


class TestWhereScenesLive:
    def test_the_listing_reads_the_configured_directory(self, workspace: Path):
        listing = list_scenes()

        assert scenes_root() == workspace
        assert listing["files"] == sorted(SHIPPED)
        assert [entry["name"] for entry in listing["scenes"]] == sorted(SHIPPED)

    def test_a_save_lands_there_and_nowhere_near_the_repository(self, workspace: Path):
        before = sorted(path.name for path in SCENES.glob("*.py"))

        assert save_scene({"name": "scene.py", "source": "scene = None\n"})["ok"] is True

        assert (workspace / "scene.py").read_text(encoding="utf-8") == "scene = None\n"
        # The repository's own scenes directory is untouched: this is the
        # regression a Playwright run left behind once already.
        assert sorted(path.name for path in SCENES.glob("*.py")) == before

    def test_without_the_override_it_is_the_working_directory(self, tmp_path, monkeypatch):
        monkeypatch.delenv(SCENES_DIR_ENV, raising=False)
        monkeypatch.chdir(tmp_path)

        assert scenes_root() == tmp_path / "scenes"
