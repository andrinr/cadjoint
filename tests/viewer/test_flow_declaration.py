"""A flow study declared beside a mesh study, in the compile payload.

`FlowStudy` is declared in a scene exactly like `ThermalStudy` and lands in
the same captured list, but it meshes nothing and the viewer cannot author
one.  Two things have to hold at once, and before this they did not: the
mesh study beside it must keep its editability, and the flow study must
report honestly that the GUI cannot write it.
"""

from __future__ import annotations

import pytest

from cadjoint.viewer._worker_declarations import _study_entries
from cadjoint.viewer._worker_scene import _execute_scene
from cadjoint.viewer.source_map import STUDY_CALL_KINDS, locate_study_statements

pytest.importorskip("jax", reason="declaring a study builds parameters")

MIXED = '''
from cadjoint.fem import Dirichlet, Nodes, ThermalStudy
from cadjoint.flow import FlowStudy, Inlet
from cadjoint.sdf.primitives import Sphere

scene = Sphere(1.0)
heat = ThermalStudy(
    name="conduction", resolution=8, conductivity=1.0,
    bcs=[Dirichlet(Nodes.side("-x"), 1.0)],
)
cool = FlowStudy(
    name="cooling", resolution=8, bounds=(-1.0, -1.0, -1.0), size=(2.0, 2.0, 2.0),
    bcs=[Inlet(velocity=0.02)],
)
'''

ONE_FLOW = '''
from cadjoint.flow import FlowStudy, Inlet
from cadjoint.sdf.primitives import Sphere

scene = Sphere(1.0)
cool = FlowStudy(
    name="cooling", resolution=8, bounds=(-1.0, -1.0, -1.0), size=(2.0, 2.0, 2.0),
    bcs=[Inlet(velocity=0.02)],
)
'''


def entries(source: str) -> list[dict]:
    return _study_entries(_execute_scene(source)["__studies__"], source)


class TestTheSourceMapSeesBoth:
    def test_a_flow_constructor_is_a_located_study(self):
        assert STUDY_CALL_KINDS["FlowStudy"] == "flow"
        located = locate_study_statements(ONE_FLOW)
        assert [statement.kind for statement in located] == ["flow"]
        assert [statement.name for statement in located] == ["cooling"]

    def test_a_flow_study_does_not_cost_its_neighbour_its_editability(self):
        """The alignment is positional, so an unknown constructor broke both.

        `_study_entries` matches statements to captured studies by count and
        kind.  With `FlowStudy` unknown to the source map a mixed scene had
        two studies and one statement, so *neither* aligned and the thermal
        study — perfectly ordinary, perfectly writable — reported itself as
        defined dynamically and lost every control on its card.
        """
        by_kind = {entry["kind"]: entry for entry in entries(MIXED)}
        assert by_kind["thermal"]["editable"] is True
        assert by_kind["thermal"]["line"] is not None
        assert by_kind["flow"]["line"] is not None


class TestAuthorabilityIsNarrowerThanLocatability:
    def test_a_flow_study_is_located_but_not_authorable(self):
        """Locating it is what keeps the payload honest; authoring it is not
        offered, because no patch operation writes a `FlowStudy`."""
        entry = entries(ONE_FLOW)[0]
        assert entry["kind"] == "flow"
        assert entry["line"] is not None, "it is found in the source"
        assert entry["editable"] is False, "but the GUI cannot write one"

    def test_its_conditions_are_reported_with_their_own_shape(self):
        entry = entries(ONE_FLOW)[0]
        inlet = entry["bcs"][0]
        assert inlet["type"] == "inlet"
        assert inlet["serializable"] is True
        # The shape a mesh row assumes, and this one does not have.
        assert inlet.get("nodes") is None
        assert inlet["velocity"] == [0.0, 0.02, 0.0]
