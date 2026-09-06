"""A flow study declared beside a mesh study, in the compile payload.

`FlowStudy` is declared in a scene exactly like `ThermalStudy` and lands in
the same captured list, but it meshes nothing and the viewer cannot author
one.  Two things have to hold at once, and before this they did not: the
mesh study beside it must keep its editability, and the flow study must
report honestly that the GUI cannot write it.
"""

from __future__ import annotations

import pytest

from cadjoint.viewer.source_map import STUDY_CALL_KINDS, locate_study_statements
from cadjoint.viewer.worker.declarations import _study_entries
from cadjoint.viewer.worker.scene import _execute_scene

pytest.importorskip("jax", reason="declaring a study builds parameters")

MIXED = """
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
"""

ONE_FLOW = """
from cadjoint.flow import FlowStudy, Inlet
from cadjoint.sdf.primitives import Sphere

scene = Sphere(1.0)
cool = FlowStudy(
    name="cooling", resolution=8, bounds=(-1.0, -1.0, -1.0), size=(2.0, 2.0, 2.0),
    bcs=[Inlet(velocity=0.02)],
)
"""


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


class TestAuthorability:
    def test_a_flow_study_is_located_and_authorable(self):
        """Both, and they are separate claims.

        Locating it keeps the payload honest and its neighbours aligned;
        authoring it is offered only because the patch layer writes a
        `FlowStudy`, its conditions and its arguments.  A kind that were
        located but unwritable would report `editable: false` and send the
        user to the code.
        """
        entry = entries(ONE_FLOW)[0]
        assert entry["kind"] == "flow"
        assert entry["line"] is not None, "it is found in the source"
        assert entry["editable"] is True, "and the GUI can write one"

    def test_its_conditions_are_reported_with_their_own_shape(self):
        entry = entries(ONE_FLOW)[0]
        inlet = entry["bcs"][0]
        assert inlet["type"] == "inlet"
        assert inlet["serializable"] is True
        # The shape a mesh row assumes, and this one does not have.
        assert inlet.get("nodes") is None
        assert inlet["velocity"] == [0.0, 0.02, 0.0]


class TestThePatchLayerWritesOne:
    """`add_study` and `add_study_bc` produce a program that runs.

    A flow study is the case where "write the source" is not a formality:
    its constructor refuses to build without an inlet, its classes come from
    a different module than a mesh study's, and three of its conditions take
    no region — so a writer that assumed the mesh shape would emit a scene
    that raises on the next compile.
    """

    def run(self, source: str) -> dict:
        namespace: dict = {}
        exec(compile(source, "<patched>", "exec"), namespace, namespace)
        return namespace

    def test_a_new_flow_study_declares_a_runnable_one(self):
        from cadjoint.viewer.patch import add_study

        patched = add_study(
            "from cadjoint.sdf.primitives import Sphere\nscene = Sphere(1.0)\n",
            "flow",
            name="cooling",
        )
        assert "from cadjoint.flow import" in patched
        # Empty `bcs` would not construct: the inlet is what drives the flow.
        study = self.run(patched)["study1"]
        assert study.name == "cooling"
        assert [bc.describe()["type"] for bc in study.bcs] == ["inlet", "outlet", "walls"]

    def test_a_placed_condition_carries_its_region_and_an_unplaced_one_does_not(self):
        from cadjoint.viewer.patch import add_study, add_study_bc

        patched = add_study(
            "from cadjoint.sdf.primitives import Sphere\nscene = Sphere(1.0)\n",
            "flow",
            name="cooling",
        )
        patched = add_study_bc(
            patched,
            "cooling",
            "heat_source",
            {"kind": "box", "min_corner": [0.0, 0.0, 0.0], "max_corner": [1.0, 1.0, 1.0]},
            2.5,
        )
        assert "HeatSource(Nodes.box(" in patched
        assert "from cadjoint.studies import Nodes" in patched
        types = [bc.describe()["type"] for bc in self.run(patched)["study1"].bcs]
        assert types == ["inlet", "outlet", "walls", "heat_source"]

    def test_an_unplaced_condition_refuses_a_region(self):
        import pytest as _pytest

        from cadjoint.viewer.patch import PatchError, add_study, add_study_bc

        patched = add_study(
            "from cadjoint.sdf.primitives import Sphere\nscene = Sphere(1.0)\n",
            "flow",
            name="cooling",
        )
        with _pytest.raises(PatchError, match="places no region"):
            add_study_bc(
                patched,
                "cooling",
                "inlet",
                {"kind": "sphere", "center": [0.0, 0.0, 0.0], "radius": 1.0},
                0.02,
            )

    def test_a_study_refuses_a_condition_of_the_other_family(self):
        import pytest as _pytest

        from cadjoint.viewer.patch import PatchError, add_study, add_study_bc

        patched = add_study(
            "from cadjoint.sdf.primitives import Sphere\nscene = Sphere(1.0)\n",
            "flow",
            name="cooling",
        )
        with _pytest.raises(PatchError, match="accepts"):
            add_study_bc(
                patched, "cooling", "dirichlet", {"kind": "side", "side": "-x", "tol": None}, 1.0
            )
