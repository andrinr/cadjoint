"""``CADJOINT_PUBLIC_TIER``: the public tier on a machine that has diff-brep.

The playground and its workers read the process-wide registry, so the switch
is what lets one machine show both tiers (and what the viewer's tier tags are
checked against).  Also pins what ``/api/capabilities`` serves per kind: the
tag and, without the kind, the sentence.
"""

from __future__ import annotations

import pytest

from cadjoint import tier
from cadjoint.plugins import registry as process_registry
from cadjoint.plugins import set_registry
from cadjoint.plugins.registry import PUBLIC_TIER_ENV, public_tier_forced
from cadjoint.viewer.schema.api import CapabilitiesResponse


@pytest.fixture
def rebuilt(monkeypatch):
    """Rebuild the process registry under the current environment, then restore it."""
    previous = set_registry(None)
    yield monkeypatch
    set_registry(previous)


@pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
def test_the_switch_reads_the_usual_truthy_spellings(monkeypatch, value):
    monkeypatch.setenv(PUBLIC_TIER_ENV, value)
    assert public_tier_forced()


@pytest.mark.parametrize("value", ["", "0", "false", "no"])
def test_anything_else_leaves_the_registry_alone(monkeypatch, value):
    monkeypatch.setenv(PUBLIC_TIER_ENV, value)
    assert not public_tier_forced()


def test_the_switch_hides_every_private_kind(rebuilt):
    rebuilt.setenv(PUBLIC_TIER_ENV, "1")
    status = tier.status(process_registry())
    assert status.installed is False
    assert not any(entry.available for entry in status.kinds.values())


def test_the_switch_keeps_the_public_plugins(rebuilt):
    rebuilt.setenv(PUBLIC_TIER_ENV, "1")
    current = process_registry()
    kinds = {current.spec(name).kind for name in current.names()}
    assert not kinds & {str(kind) for kind in tier.KINDS}
    # The public tier's own slots — the solvers — are still there.
    assert kinds


def test_capabilities_carry_a_tag_and_the_public_sentence():
    with tier.absent():
        body = {"ok": True, **tier.status().as_dict()}
    CapabilitiesResponse.model_validate(body)
    for kind, entry in body["kinds"].items():
        assert entry["label"] == tier.label(kind, False)
        assert entry["label"].endswith(("public tier", "lattice"))
        assert entry["message"] == tier.message(kind, entry["reason"])
    assert body["kinds"]["brep"]["label"] == "Dual contouring · public tier"
    assert body["kinds"]["step_export"]["label"] == "Faceted · public tier"
    assert body["kinds"]["feature_edges"]["label"] == "Approximate · lattice"


def test_a_filled_kind_is_tagged_exact_and_carries_no_sentence(stub_tier):
    body = {"ok": True, **tier.status().as_dict()}
    CapabilitiesResponse.model_validate(body)
    for kind in tier.KINDS:
        entry = body["kinds"][kind]
        assert entry["available"] is True
        assert entry["label"].endswith("private tier")
        assert entry["message"] is None
