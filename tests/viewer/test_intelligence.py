"""The playground's editor intelligence: ruff diagnostics, jedi completions.

These endpoints run in the server process on every keystroke, so the tests
here pin three things: the wire shape the CodeMirror client is written
against, that the analysers stay *static* (a scene whose top level would
delete a file must be analysable without the file moving), and that no
analyser failure escapes as an exception instead of an error response.
"""

from __future__ import annotations

import json
import sys
import time

import pytest

from cadjoint.viewer._example_scene import EXAMPLE_SOURCE
from cadjoint.viewer._intelligence import (
    MAX_COMPLETIONS,
    complete_source,
    lint_source,
    record_compile,
    signature_source,
)
from cadjoint.viewer._limits import MAX_SOURCE_BYTES

jedi = pytest.importorskip("jedi")


CONSTRUCTION_IMPORT = "from cadjoint.construction import PolygonProfile, SketchPlane, extrude\n"


def labels(result: dict) -> list[str]:
    """The completion labels of a successful ``/api/complete`` response."""
    assert result["ok"] is True, result.get("error")
    return [item["label"] for item in result["completions"]]


def codes(result: dict) -> list[str]:
    """The rule codes of a successful ``/api/lint`` response."""
    assert result["ok"] is True, result.get("error")
    return [item["code"] for item in result["diagnostics"]]


# ── lint ────────────────────────────────────────────────────────────────────


def test_the_starter_scene_lints_clean():
    # The program the playground opens with must not greet the user with a
    # gutter full of complaints.
    result = lint_source({"source": EXAMPLE_SOURCE})

    assert result["ok"] is True
    assert result["diagnostics"] == []


def test_lint_reports_a_syntax_error_as_an_error_severity():
    result = lint_source({"source": "def f( :\n    pass\n"})

    assert result["ok"] is True
    first = result["diagnostics"][0]
    assert first["severity"] == "error"
    assert first["code"] == "invalid-syntax"
    assert first["source"] == "ruff"
    assert first["from_line"] == 1


def test_lint_grades_severity_across_the_three_bands():
    source = "import sys\nprint(undefined_thing)\nl = 1\n"

    result = lint_source({"source": source})

    by_code = {item["code"]: item for item in result["diagnostics"]}
    # An undefined name cannot run: error.  An unused import is a probable
    # mistake: warning.  These three must be visually distinguishable.
    assert by_code["F821"]["severity"] == "error"
    assert by_code["F401"]["severity"] == "warning"
    assert by_code["E741"]["severity"] == "warning"


def test_lint_columns_are_zero_based_and_lines_are_one_based():
    # `sys` starts at 1-based column 8, so 0-based column 7.
    result = lint_source({"source": "import sys\n"})

    unused = next(item for item in result["diagnostics"] if item["code"] == "F401")
    assert (unused["from_line"], unused["from_col"]) == (1, 7)
    assert (unused["to_line"], unused["to_col"]) == (1, 10)


def test_lint_carries_ruffs_autofix_so_the_client_can_offer_it():
    result = lint_source({"source": "import sys\n"})

    fix = next(item for item in result["diagnostics"] if item["code"] == "F401")["fix"]
    assert fix["message"] == "Remove unused import: `sys`"
    assert fix["applicability"] == "safe"
    assert fix["edits"][0]["content"] == ""
    assert fix["edits"][0]["from_line"] == 1


def test_lint_links_the_rule_documentation():
    result = lint_source({"source": "import sys\n"})

    unused = next(item for item in result["diagnostics"] if item["code"] == "F401")
    assert unused["url"].startswith("https://docs.astral.sh/ruff/rules/")


def test_lint_ignores_the_rules_that_are_noise_in_a_scene():
    # Line length is a formatter's business, and a scene under edit often has
    # an import below other statements.
    source = "x = 1\nimport sys\nprint(sys, '" + "y" * 200 + "')\n"

    assert "E501" not in codes(lint_source({"source": source}))
    assert "E402" not in codes(lint_source({"source": source}))


def test_lint_does_not_inherit_the_repository_ruff_configuration():
    # `--isolated`: the answer must not depend on the server's directory.
    # The repo ignores B905, so a `zip` without `strict` proves isolation.
    result = lint_source({"source": "print(list(zip([1], [2])))\n"})

    assert "B905" in codes(result)


def test_lint_refuses_a_non_string_source():
    assert lint_source({"source": 42}) == {
        "ok": False,
        "error": "The request must contain a string `source` field.",
    }


def test_lint_refuses_an_oversized_program():
    result = lint_source({"source": "x = 1\n" * MAX_SOURCE_BYTES})

    assert result["ok"] is False
    assert "limit" in result["error"]


def test_lint_survives_a_broken_analyser(monkeypatch):
    monkeypatch.setattr(
        "cadjoint.viewer._intelligence._ruff_diagnostics",
        lambda *_: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    result = lint_source({"source": "x = 1\n"})

    assert result["ok"] is False
    assert "boom" in result["error"]


def test_lint_reports_a_missing_ruff_binary_instead_of_raising(monkeypatch):
    monkeypatch.setattr("cadjoint.viewer._intelligence._ruff_binary", lambda: None)

    result = lint_source({"source": "x = 1\n"})

    assert result == {
        "ok": False,
        "error": "Linting needs the `ruff` executable; install cadjoint[editor].",
    }


# ── the last runtime failure ────────────────────────────────────────────────


def playground_traceback(line: int) -> dict:
    """A compile worker failure whose traceback blames one scene line."""
    return {
        "ok": False,
        "error": (
            "Traceback (most recent call last):\n"
            '  File "/x/_compile_worker.py", line 97, in main\n'
            "    exec(compile(source, PLAYGROUND_FILENAME, 'exec'), namespace)\n"
            f'  File "<cadjoint-playground>", line {line}, in <module>\n'
            "ZeroDivisionError: division by zero\n"
        ),
    }


def test_a_failed_compile_becomes_the_first_diagnostic():
    source = "x = 1\ny = 2\nboom = 1 / 0\n"
    record_compile(source, playground_traceback(3))

    result = lint_source({"source": source})

    first = result["diagnostics"][0]
    assert result["runtime"] is True
    assert first["source"] == "runtime"
    assert first["severity"] == "error"
    assert first["code"] == "ZeroDivisionError"
    assert first["message"] == "ZeroDivisionError: division by zero"
    assert (first["from_line"], first["to_line"]) == (3, 3)
    assert first["to_col"] == len("boom = 1 / 0")


def test_a_runtime_diagnostic_dies_with_the_text_that_caused_it():
    source = "boom = 1 / 0\n"
    record_compile(source, playground_traceback(1))

    edited = lint_source({"source": "boom = 1 / 2\n"})

    assert edited["runtime"] is False
    assert all(item["source"] != "runtime" for item in edited["diagnostics"])


def test_a_successful_compile_clears_the_remembered_failure():
    source = "boom = 1 / 0\n"
    record_compile(source, playground_traceback(1))

    passed = record_compile(source, {"ok": True, "sdf": "fn sdf() {}"})

    assert passed["sdf"] == "fn sdf() {}"  # a tap, not a filter
    assert lint_source({"source": source})["runtime"] is False


def test_a_failure_that_never_reached_the_scene_is_not_a_diagnostic():
    source = "x = 1\n"
    record_compile(source, {"ok": False, "error": "Source must be a string."})

    assert lint_source({"source": source})["runtime"] is False


# ── completion ──────────────────────────────────────────────────────────────


def test_completion_knows_cadjoints_real_construction_api():
    source = CONSTRUCTION_IMPORT + "sk = Sk"

    found = labels(complete_source({"source": source, "line": 2, "column": 7}))

    assert "SketchPlane" in found


def test_completion_resolves_attributes_of_an_inferred_cadjoint_type():
    source = CONSTRUCTION_IMPORT + "plane = SketchPlane()\nplane."

    found = labels(complete_source({"source": source, "line": 3, "column": 6}))

    # These come from the installed class, not a guess: jedi inferred the
    # type of `plane` through the constructor call.
    assert {"origin", "normal", "to_world"} <= set(found)


def test_completion_offers_keyword_arguments_inside_a_call():
    source = CONSTRUCTION_IMPORT + "SketchPlane(o"

    result = complete_source({"source": source, "line": 2, "column": 13})

    origin = next(item for item in result["completions"] if item["label"] == "origin=")
    assert origin["detail"] == "param"
    assert origin["type"] == "property"


def test_completion_reports_where_the_typed_prefix_starts():
    source = CONSTRUCTION_IMPORT + "sk = Sket"

    result = complete_source({"source": source, "line": 2, "column": 9})

    # `Sket` begins at 0-based column 5; the client replaces 5..9.
    assert result["from_column"] == 5
    assert result["from_line"] == 2


def test_completion_teaches_the_api_through_the_info_field():
    source = CONSTRUCTION_IMPORT + "sk = SketchPlan"

    result = complete_source({"source": source, "line": 2, "column": 15})

    plane = next(item for item in result["completions"] if item["label"] == "SketchPlane")
    assert plane["type"] == "class"
    assert "work plane" in plane["info"].lower()


def test_completion_hides_private_names_until_the_caret_asks_for_them():
    source = CONSTRUCTION_IMPORT + "plane = SketchPlane()\nplane."

    public = labels(complete_source({"source": source, "line": 3, "column": 6}))
    private = labels(complete_source({"source": source + "_", "line": 3, "column": 7}))

    assert not any(label.startswith("_") for label in public)
    assert any(label.startswith("_") for label in private)


def test_completion_caps_the_list_it_returns():
    result = complete_source({"source": "import os\n", "line": 2, "column": 0})

    assert len(result["completions"]) <= MAX_COMPLETIONS


def test_completion_clamps_a_caret_that_ran_ahead_of_the_source():
    # The editor debounces, so a request can name a position the text has
    # not reached yet; that must answer, not fail.
    result = complete_source({"source": "x = 1\n", "line": 99, "column": 99})

    assert result["ok"] is True


def test_completion_refuses_a_non_integer_position():
    result = complete_source({"source": "x = 1\n", "line": "2", "column": 0})

    assert result == {"ok": False, "error": "The request needs an integer `line` (1-based)."}


def test_completion_survives_a_broken_analyser(monkeypatch):
    monkeypatch.setattr(
        "cadjoint.viewer._intelligence._script",
        lambda _: (_ for _ in ()).throw(RuntimeError("jedi fell over")),
    )

    result = complete_source({"source": "x = 1\n", "line": 1, "column": 5})

    assert result["ok"] is False
    assert "jedi fell over" in result["error"]


# ── signature help ──────────────────────────────────────────────────────────


def test_signature_describes_the_call_the_caret_sits_inside():
    source = CONSTRUCTION_IMPORT + "SketchPlane("

    result = signature_source({"source": source, "line": 2, "column": 12})

    assert result["ok"] is True
    signature = result["signatures"][0]
    assert signature["name"] == "SketchPlane"
    assert signature["label"].startswith("SketchPlane(origin=")
    names = [param["name"] for param in signature["parameters"]]
    assert names[:2] == ["origin", "normal"]
    assert signature["active_parameter"] == 0
    assert "work plane" in signature["documentation"].lower()


def test_signature_tracks_which_argument_is_being_typed():
    source = CONSTRUCTION_IMPORT + "SketchPlane(origin=[0, 0, 0], "

    result = signature_source({"source": source, "line": 2, "column": 30})

    assert result["signatures"][0]["active_parameter"] == 1


def test_signature_is_empty_outside_a_call():
    result = signature_source({"source": "x = 1\n", "line": 1, "column": 5})

    assert result == {"ok": True, "signatures": []}


# ── these analysers never run the program ───────────────────────────────────


def test_the_analysers_never_execute_the_program_they_read(tmp_path):
    """A scene whose top level would delete a file must leave it alone."""
    victim = tmp_path / "victim.txt"
    victim.write_text("still here", encoding="utf-8")
    marker = tmp_path / "ran.txt"
    source = (
        "import os\n"
        "import subprocess\n"
        f"os.remove({str(victim)!r})\n"
        f"open({str(marker)!r}, 'w').write('executed')\n"
        f"os.system('touch {marker}')\n"
        f"subprocess.run(['touch', {str(marker)!r}])\n"
        "SketchPlane()\n"
    )
    mid_call = source + "SketchPlane("

    lint = lint_source({"source": source})
    complete = complete_source({"source": source + "os.", "line": 8, "column": 3})
    signature = signature_source({"source": mid_call, "line": 8, "column": 12})

    assert lint["ok"] is True
    assert complete["ok"] is True
    assert signature["ok"] is True
    assert victim.read_text(encoding="utf-8") == "still here"
    assert not marker.exists()
    # And the static analyser still did its job on that program.
    assert "F821" in codes(lint)  # `SketchPlane` was never imported


# ── latency ─────────────────────────────────────────────────────────────────


def measure(call, repeats: int = 5) -> float:
    """Best-of wall time for one endpoint call, in milliseconds."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        result = call()
        best = min(best, (time.perf_counter() - start) * 1000.0)
        assert result["ok"] is True
    return best


@pytest.mark.skipif(sys.platform == "win32", reason="timings are measured on the CI platforms")
def test_warm_lint_and_completion_stay_inside_the_keystroke_budget():
    # The budget the editor is designed against: lint under 50 ms, completion
    # under 100 ms, both warm, on the starter scene.  Generous multiples of
    # what was measured (about 6 ms and 15 ms) so this catches a regression in
    # kind — a per-call subprocess, a lost cache — not machine noise.
    line = len(EXAMPLE_SOURCE.split("\n")) + 1
    typed = EXAMPLE_SOURCE + "\nSk"

    lint_ms = measure(lambda: lint_source({"source": EXAMPLE_SOURCE}))
    complete_ms = measure(lambda: complete_source({"source": typed, "line": line, "column": 2}))

    assert lint_ms < 200.0, f"warm lint took {lint_ms:.0f} ms"
    assert complete_ms < 400.0, f"warm completion took {complete_ms:.0f} ms"


# ── the wire, end to end ────────────────────────────────────────────────────


def test_every_intelligence_endpoint_answers_over_http():
    from urllib.request import urlopen

    from tests.test_playground import post, running_server

    source = CONSTRUCTION_IMPORT + "SketchPlane("
    with running_server() as base:
        with urlopen(f"{base}/api/session") as response:
            token = json.loads(response.read())["token"]

        with urlopen(post(base, "/api/lint", {"source": "import sys\n"}, token)) as response:
            lint = json.loads(response.read())
        position = {"source": source, "line": 2, "column": 12}
        with urlopen(post(base, "/api/complete", position, token)) as response:
            complete = json.loads(response.read())
        with urlopen(post(base, "/api/signature", position, token)) as response:
            signature = json.loads(response.read())

    assert [item["code"] for item in lint["diagnostics"]] == ["F401"]
    assert complete["ok"] is True
    assert signature["signatures"][0]["name"] == "SketchPlane"


def test_completion_leads_with_the_calls_keyword_arguments():
    # Jedi buries `origin=` in one alphabetical run with every builtin; at an
    # open paren the call's own arguments are the answer being asked for.
    source = CONSTRUCTION_IMPORT + "SketchPlane("

    result = complete_source({"source": source, "line": 2, "column": 12})

    kinds = [item["detail"] for item in result["completions"]]
    params = set(labels(result)[: kinds.count("param")])
    # Every keyword argument leads, then jedi's own order resumes.
    assert kinds.index("function") > kinds.count("param") - 1
    assert {"origin=", "normal="} <= params
    # Hoisted into the head of the list, so each one carries its default.
    origin = next(item for item in result["completions"] if item["label"] == "origin=")
    assert origin["info"] == "param origin=(0.0, 0.0, 0.0)"


def test_hoisting_keyword_arguments_never_duplicates_one():
    source = CONSTRUCTION_IMPORT + "SketchPlane(o"

    found = labels(complete_source({"source": source, "line": 2, "column": 13}))

    assert found[0] == "origin="
    assert found.count("origin=") == 1
