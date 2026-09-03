"""Malformed request values are refused with an error, never a crash."""

from __future__ import annotations

import pytest

from cadjoint.viewer._example_scene import EXAMPLE_SOURCE
from cadjoint.viewer._patch_requests import patch_source


@pytest.mark.parametrize(
    "request_body",
    [
        {"op": "add_study", "kind": ["thermal"]},
        {"op": "add_study", "kind": {"thermal": True}},
        {"op": "add_study_bc", "study": 0, "bc_type": ["dirichlet"], "selection": {}, "value": 1},
        {"op": "set_optimization_value", "optimization": 0, "argument": [1], "value": 2},
    ],
)
def test_unhashable_values_are_refused_not_raised(request_body):
    result = patch_source({"source": EXAMPLE_SOURCE, **request_body})
    assert result["ok"] is False
    assert "must be one of" in result["error"] or "must be" in result["error"]
