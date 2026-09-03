"""The public tier alone: what still works, what degrades, what refuses.

Nothing in this repository fills the five private kinds, so *this* is the
state the public repository ships in and the state its CI runs in: an
unmodified checkout is already degraded.  Two fixtures give the file both
halves of the matrix anyway.  ``stub_tier`` registers
``tests/plugins/stubs.py`` — five objects that satisfy the Protocols and
compute nothing — to see what a filled kind changes; and
:func:`cadjoint.tier.absent` installs a copy of the process registry with
those kinds removed, so the degraded assertions hold even on a developer's
machine that *does* have diff-brep installed alongside.

The matrix this file pins (``research/two-tier.md`` §2.5):

=========================  ==================================================
compile / mesh / solve     unchanged — nothing in a solve moves a node
inspect / VTK              unchanged; a Gmsh mesh reports ``frozen_geometry``
feature edges              fall back to the lattice classifier, ``"lattice"``
STEP export                falls back to the faceted writer, ``report["tier"]``
``Optimization``           refuses at validation with one sentence
=========================  ==================================================

The refusal is deliberate (D6): the public tier *could* slide every
boundary node onto the scene's zero set with an arity-1 projection and call
that a gradient, but it would take crease nodes off their creases and put
midsides back on chords, silently.  A refusal that names the tier is better
than a derivative that is quietly wrong.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cadjoint import tier
from cadjoint.enums import PluginKind
from cadjoint.fem.gmsh import gmsh_available
from cadjoint.meshing import GridSpec
from cadjoint.sdf.boolean import Difference
from cadjoint.sdf.primitives import Box, Cylinder

BOX_SOURCE = """
from cadjoint.geometry import Vector
from cadjoint.sdf.primitives import Box

scene = Box(Vector([0.6, 0.4, 0.3]))
"""

#: A plate with a bore: sharp creases the lattice classifier has to find,
#: and a curved face the faceted STEP writer has to give up on.
GRID = GridSpec.from_bounds((-0.83, -0.83, -0.63), (1.66, 1.66, 1.26), 20)


def plate():
    from cadjoint.geometry import Scalar, Vector

    return Difference(
        (Box(size=Vector([0.6, 0.6, 0.4])), Cylinder(radius=Scalar(0.25), height=Scalar(0.9))),
        smoothness=0.0,
    )


# ── the switch itself ────────────────────────────────────────────────────────


class TestTheFixtureBlanksTheRegistry:
    """``tier.absent`` has to be the real thing, or nothing below means much."""

    def test_no_provider_is_shipped_so_the_default_state_is_absent(self):
        """This is the whole point of the split, asserted once.

        Nothing in this repository fills the five kinds, so an unmodified
        checkout is already the degraded state — ``tier.absent()`` is what
        the rest of this file uses to reach it deliberately even when a
        developer *has* installed diff-brep alongside.
        """
        status = tier.status()
        assert not status.installed
        assert not any(status.flags().values())
        for kind in tier.KINDS:
            assert status[kind].reason == "not registered"

    def test_the_switch_still_blanks_a_registry_that_has_a_provider(self, stub_tier):
        """With a provider registered, ``absent`` takes it out and puts it back."""
        assert tier.status().installed, "the stand-in should fill all five kinds"
        with tier.absent():
            inside = tier.status()
            assert not inside.installed
            assert not any(inside.flags().values())
            for kind in tier.KINDS:
                assert inside[kind].reason == "not registered"
        assert tier.status().installed, "the registry must come back"

    def test_the_public_kinds_are_untouched(self):
        """Blanking the private kinds must not take the meshers with it."""
        with tier.absent():
            from cadjoint.plugins import registry

            assert PluginKind.TET_MESHER.value in registry().kinds()
            assert PluginKind.MESHER.value in registry().kinds()

    def test_component_degrades_and_require_refuses(self):
        with tier.absent():
            assert tier.component(PluginKind.BREP.value) is None
            assert tier.component(PluginKind.BREP.value, default="fallback") == "fallback"
            with pytest.raises(tier.TierUnavailable) as caught:
                tier.require(PluginKind.BREP.value)
            assert caught.value.kind == PluginKind.BREP.value
            assert "diff-brep" in str(caught.value)

    def test_the_report_is_json_ready_and_names_the_provider(self):
        with tier.absent():
            report = tier.report()
        assert report["versions"]["cadjoint"] is not None
        assert report["tier"]["provider"] == "diff-brep"
        assert report["tier"]["installed"] is False
        assert set(report["tier"]["kinds"]) == set(tier.KINDS)

    def test_one_message_per_kind_and_they_all_name_the_way_out(self):
        for kind in tier.KINDS:
            sentence = tier.message(kind)
            assert "diff-brep" in sentence or "private tier" in sentence
            assert sentence == sentence.strip() and sentence


# ── feature edges: the lattice layer ─────────────────────────────────────────


class TestFeatureEdgesFallBackToTheLattice:
    """The overlay is always drawn; only its *source* changes."""

    def test_the_payload_says_which_layer_drew_the_sharp_chords(self):
        from cadjoint.viewer._edge_overlay import _mesh_edge_payload

        scene = plate()
        with tier.absent():
            degraded = _mesh_edge_payload(scene)
        assert degraded is not None, "the overlay must survive without the tier"
        assert degraded["edges"] == "lattice"
        assert degraded["sharp"], "the lattice classifier still finds the creases"
        assert degraded["wire"], "and the wire layer is the public DC pass"

    def test_the_graph_layer_is_what_fills_it_when_the_kind_is_there(self, stub_tier):
        """A registered ``feature_edges`` provider is what the overlay draws.

        The stand-in returns one segment, which is not geometry anyone would
        want — what is being pinned is that the overlay *asks the kind* and
        labels the answer ``"graph"``, so an installed diff-brep is reached
        without the viewer knowing its name.
        """
        from cadjoint.viewer._edge_overlay import _mesh_edge_payload

        payload = _mesh_edge_payload(plate())
        assert payload["edges"] == "graph"
        assert payload["sharp"], "the kind's curves are what the sharp layer carries"

    def test_the_lattice_layer_is_geometry_not_an_empty_list(self):
        """A silently empty sharp layer would look like a working fallback."""
        from cadjoint.viewer._edge_overlay import _lattice_layers, _overlay_grid

        vertices, quad_edges, sharp = _lattice_layers(plate(), _overlay_grid())
        assert vertices.shape[1] == 3 and quad_edges.shape[1] == 2
        assert sharp.shape[1:] == (2, 3)
        assert sharp.shape[0] > 8, "a plate with a bore has creases to draw"
        lengths = np.linalg.norm(sharp[:, 1] - sharp[:, 0], axis=1)
        assert (lengths > 0).all(), "a chord of zero length is not an edge"


# ── STEP export: the faceted writer ──────────────────────────────────────────


class TestStepExportFallsBackToFaceted:
    """A STEP file is *always* produced; the report says by which writer."""

    def _export(self, tmp_path: Path, **fields) -> dict:
        from cadjoint.viewer._export import export_scene

        tmp_path.mkdir(parents=True, exist_ok=True)
        path = tmp_path / "out.step"
        return export_scene(
            {
                "source": BOX_SOURCE,
                "mode": "export",
                "path": str(path),
                "format": "step",
                "resolution": 12,
                **fields,
            }
        )

    def test_without_the_tier_the_file_is_faceted_and_the_report_says_why(self, tmp_path):
        result = self._export(tmp_path)
        assert result["ok"] is True, result
        assert result["report"]["path"] == "mesh"
        assert result["report"]["tier"] == tier.message(PluginKind.STEP_EXPORT.value)
        assert (tmp_path / "out.step").read_text().startswith("ISO-10303-21;")

    def test_with_the_tier_the_kind_takes_it_and_the_report_says_nothing(self, tmp_path, stub_tier):
        """A registered ``step_export`` provider writes the file instead.

        The exporter hands the kind the scene and a path and reports which
        writer produced the file.  What the private writer *puts* in it —
        analytic ``PLANE`` and ``CYLINDRICAL_SURFACE`` entities rather than
        thousands of facets — is diff-brep's own test; this is the handover.
        """
        result = self._export(tmp_path)
        assert result["ok"] is True, result
        assert result["report"]["path"] == "brep"
        assert "tier" not in result["report"]
        assert (tmp_path / "out.step").read_text().startswith("ISO-10303-21;")

    def test_the_faceted_request_never_mentions_the_tier(self, tmp_path):
        """``analytic=False`` asks for the public writer; nothing is missing."""
        result = self._export(tmp_path, analytic=False)
        assert result["report"]["path"] == "mesh"
        assert "tier" not in result["report"]


# ── the compile payload ──────────────────────────────────────────────────────


class TestTheCompilePayloadCarriesTheTier:
    """The viewer is told, rather than left to guess from an empty layer."""

    def test_the_flags_are_one_boolean_per_kind(self):
        from cadjoint.viewer._compile_worker import _tier_flags

        assert _tier_flags() == dict.fromkeys(tier.KINDS, False)

    def test_the_flags_go_true_when_a_provider_is_registered(self, stub_tier):
        from cadjoint.viewer._compile_worker import _tier_flags

        assert _tier_flags() == dict.fromkeys(tier.KINDS, True)
        with tier.absent():
            assert _tier_flags() == dict.fromkeys(tier.KINDS, False)

    def test_the_schema_field_is_optional_so_an_old_client_is_unaffected(self):
        from cadjoint.viewer.schema.payloads import CompilePayload

        field = CompilePayload.model_fields["tier"]
        assert not field.is_required() and field.default is None

    def test_the_http_capabilities_endpoint_is_the_status(self):
        """``GET /api/capabilities`` is what the Processes window reads."""
        with tier.absent():
            body = {"ok": True, **tier.status().as_dict()}
        assert body["ok"] and body["installed"] is False
        assert body["contract_version"] == tier.CONTRACT_VERSION
        assert body["kinds"][PluginKind.NODE_MAP.value]["available"] is False


# ── the FEM path ─────────────────────────────────────────────────────────────


gmsh_only = pytest.mark.skipif(
    not gmsh_available(), reason="the optional 'gmsh' extra is not installed"
)


class TestATetGenMeshIsUnaffected:
    """The default route has a public derivative and never degrades."""

    def test_a_tetgen_simmesh_is_never_frozen(self):
        from cadjoint.fem.simmesh import SimMesh

        mesh = SimMesh(name="m", resolution=8, method="tet4")
        with tier.absent():
            assert mesh.frozen_geometry is False
            assert mesh.describe()["frozen_geometry"] is False

    def test_a_hex_simmesh_is_never_frozen(self):
        from cadjoint.fem.simmesh import SimMesh

        with tier.absent():
            assert SimMesh(name="m", resolution=8).frozen_geometry is False


@gmsh_only
class TestAGmshMeshIsFrozenGeometry:
    """Everything that does not move a node still works."""

    @pytest.fixture(scope="class")
    def gmsh_mesh(self):
        from cadjoint.fem.simmesh import SimMesh

        declared = SimMesh(
            name="plate",
            resolution=16,
            method="tet10",
            mesher="gmsh",
            bounds=GRID.origin,
            size=[spacing * count for spacing, count in zip(GRID.spacing, GRID.cells)],
        )
        return declared, declared.build(plate())

    def test_meshing_needs_nothing_private(self, gmsh_mesh):
        _declared, built = gmsh_mesh
        with tier.absent():
            from cadjoint.fem.quality import tet_volumes

            assert built.ele_type == "TET10"
            assert (tet_volumes(built.points, built.cells) > 0).all()
            assert built.owned is not None and built.owned.count == built.num_points

    def test_inspection_reports_the_frozen_geometry(self, gmsh_mesh):
        declared, _built = gmsh_mesh
        report = declared.inspect(plate())
        assert report["mesher"] == "gmsh"
        assert report["frozen_geometry"] is True
        assert declared.describe()["frozen_geometry"] is True

    def test_the_node_map_kind_is_what_unfreezes_it(self, gmsh_mesh, stub_tier):
        """``frozen_geometry`` reads the registry, not the mesher."""
        declared, _built = gmsh_mesh
        assert declared.frozen_geometry is False
        with tier.absent():
            assert declared.frozen_geometry is True

    def test_vtk_export_works_on_a_frozen_mesh(self, gmsh_mesh, tmp_path):
        import meshio

        _declared, built = gmsh_mesh
        with tier.absent():
            path = tmp_path / "frozen.vtu"
            meshio.Mesh(built.points, [("tetra10", built.cells)]).write(path)
            assert path.stat().st_size > 0

    def test_optimization_refuses_at_declaration_with_the_one_sentence(self, gmsh_mesh):
        from cadjoint.fem import Dirichlet, Nodes, ThermalStudy
        from cadjoint.optimize import Optimization

        declared, _built = gmsh_mesh
        study = ThermalStudy(
            name="frozen",
            mesh=declared,
            conductivity=1.0,
            bcs=[Dirichlet(Nodes.side("-x"), 0.0), Dirichlet(Nodes.side("+x"), 100.0)],
        )
        with pytest.raises(tier.TierUnavailable) as caught:
            Optimization(name="nope", study=study, metric="max")
        message = str(caught.value)
        assert "frozen geometry" in message
        assert "mesher='tetgen'" in message
        assert "install diff-brep" in message

    def test_the_same_declaration_is_accepted_once_the_kind_is_filled(self, gmsh_mesh, stub_tier):
        """The refusal is about the missing kind, not about Gmsh."""
        from cadjoint.fem import Dirichlet, Nodes, ThermalStudy
        from cadjoint.optimize import Optimization

        declared, _built = gmsh_mesh
        study = ThermalStudy(
            name="thawed",
            mesh=declared,
            conductivity=1.0,
            bcs=[Dirichlet(Nodes.side("-x"), 0.0), Dirichlet(Nodes.side("+x"), 100.0)],
        )
        Optimization(name="fine", study=study, metric="max")
