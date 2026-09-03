"""Tests for cadjoint.enums: the fixed option sets, and every boundary that
normalises to them.

Three properties are what make the enums safe to adopt:

1. **Parity** — the enum member and its plain string build the same object
   and the same ``describe()`` payload, so scene programs never have to
   change and the wire is untouched.
2. **Messages** — a rejected value still names the accepted spellings, in
   declaration order, in the wording the API already promised.
3. **One source of truth** — the viewer's validators, the pydantic request
   models and the generated TypeScript all read the same member list, so a
   member added to the enum cannot reach one of them and miss the others.
"""

from __future__ import annotations

import json
import typing
from typing import Literal

import pytest

from cadjoint import enums
from cadjoint.enums import (
    BoundaryConditionType,
    ConstraintKind,
    ConstraintSolveMethod,
    ExportFormat,
    FemBackend,
    GradientPath,
    MeshMethod,
    ObjectiveMetric,
    OptimizationArgument,
    OptimizerMethod,
    PluginKind,
    PluginTransport,
    Side,
    StudyKind,
    either,
    listed,
    parse,
    values,
)

#: Every option set the module publishes, for the shape tests.
OPTION_SETS = (
    MeshMethod,
    StudyKind,
    BoundaryConditionType,
    Side,
    GradientPath,
    ObjectiveMetric,
    OptimizerMethod,
    OptimizationArgument,
    ConstraintSolveMethod,
    ConstraintKind,
    ExportFormat,
    FemBackend,
    PluginKind,
    PluginTransport,
)


class TestOptionSets:
    @pytest.mark.parametrize("option", OPTION_SETS, ids=lambda option: option.__name__)
    def test_members_are_plain_strings(self, option):
        for member in option:
            assert isinstance(member, str)
            assert member == member.value
            assert f"{member}" == member.value
            assert json.loads(json.dumps({"m": member}))["m"] == member.value

    @pytest.mark.parametrize("option", OPTION_SETS, ids=lambda option: option.__name__)
    def test_members_repr_as_the_value(self, option):
        """Messages interpolating ``{value!r}`` must read as they always did."""
        for member in option:
            assert repr(member) == repr(member.value)
            assert f"got {member!r}." == f"got {member.value!r}."
            assert f"{ {'k': member} }" == f"{ {'k': member.value} }"

    @pytest.mark.parametrize("option", OPTION_SETS, ids=lambda option: option.__name__)
    def test_values_are_declaration_ordered(self, option):
        assert values(option) == tuple(member.value for member in option)
        assert all(type(value) is str for value in values(option))

    @pytest.mark.parametrize("option", OPTION_SETS, ids=lambda option: option.__name__)
    def test_every_option_set_is_exported(self, option):
        assert option.__name__ in enums.__all__

    @pytest.mark.parametrize("option", OPTION_SETS, ids=lambda option: option.__name__)
    def test_the_like_alias_spells_out_every_member(self, option):
        """``MeshMethodLike``'s ``Literal`` arm must not drift from the enum.

        The alias is what public signatures annotate, so a member added to
        the enum and forgotten in the alias would type-check as an error
        for the very spelling the constructor accepts.
        """
        alias = getattr(enums, f"{option.__name__}Like", None)
        if alias is None:
            # FemBackend is open: there is no closed alias to check.
            assert option is FemBackend
            return
        literals = [
            argument
            for arm in typing.get_args(alias)
            if typing.get_origin(arm) is Literal
            for argument in typing.get_args(arm)
        ]
        if not literals:
            # PluginKind's alias is `PluginKind | str`: any kind registers.
            assert option is PluginKind
            assert str in typing.get_args(alias)
            return
        assert tuple(literals) == values(option)
        assert option in typing.get_args(alias)

    def test_listed_joins_in_declaration_order(self):
        assert listed(MeshMethod) == "hex, tet4, tet10"
        assert listed(BoundaryConditionType) == "dirichlet, heat_flux, fixed, traction"

    def test_either_reads_as_prose(self):
        assert either(StudyKind) == "`thermal` or `elastic`"
        assert either(ConstraintSolveMethod) == "`newton`, `adam`, or `sgd`"
        assert either(StudyKind, quote="") == "thermal or elastic"

    def test_parse_accepts_both_spellings(self):
        assert parse(MeshMethod, "tet4", "nope") is MeshMethod.TET4
        assert parse(MeshMethod, MeshMethod.TET4, "nope") is MeshMethod.TET4

    def test_parse_raises_the_given_message(self):
        with pytest.raises(ValueError, match="^say this$"):
            parse(MeshMethod, "voxel", "say this")


class TestSimMesh:
    def test_the_enum_and_the_literal_build_the_same_mesh(self):
        from cadjoint.fem import SimMesh

        literal = SimMesh(name="m", resolution=4, method="tet10")
        member = SimMesh(name="m", resolution=4, method=MeshMethod.TET10)

        assert literal == member
        assert literal.describe() == member.describe()
        assert literal.method is MeshMethod.TET10

    def test_the_payload_stays_a_plain_string(self):
        from cadjoint.fem import SimMesh

        payload = SimMesh(name="m", resolution=4, method=MeshMethod.TET4).describe()

        assert type(payload["method"]) is str
        assert json.loads(json.dumps(payload))["method"] == "tet4"

    def test_the_default_is_hex(self):
        from cadjoint.fem import SimMesh

        assert SimMesh(name="m", resolution=4).method == MeshMethod.HEX

    def test_an_unknown_method_names_the_members(self):
        from cadjoint.fem import SimMesh

        with pytest.raises(ValueError) as error:
            SimMesh(name="m", resolution=4, method="voxel")

        assert str(error.value) == "method must be one of ['hex', 'tet4', 'tet10'], got 'voxel'."


class TestSide:
    def test_the_enum_and_the_literal_select_the_same_nodes(self):
        from cadjoint.fem import Nodes

        literal = Nodes.side("+x")
        member = Nodes.side(Side.PLUS_X)

        assert literal == member
        assert literal.describe() == member.describe()
        assert type(literal.describe()["side"]) is str

    def test_an_unknown_side_names_the_members(self):
        from cadjoint.fem import Nodes

        with pytest.raises(ValueError) as error:
            Nodes.side("+w")

        assert str(error.value) == (
            "side must be one of ('+x', '-x', '+y', '-y', '+z', '-z'), got '+w'."
        )


class TestOptimization:
    def _optimization(self, study, **overrides):
        from cadjoint.optimize import Optimization

        settings = {"name": "o", "study": study, "metric": "mean"}
        settings.update(overrides)
        return Optimization(**settings)

    @pytest.fixture
    def study(self):
        from cadjoint.fem import Dirichlet, Nodes, ThermalStudy

        return ThermalStudy(
            name="s",
            resolution=4,
            conductivity=1.0,
            bcs=[Dirichlet(Nodes.side("-x"), 1.0)],
        )

    def test_published_tuples_are_the_enum_values(self):
        from cadjoint import optimize

        assert optimize.METHODS == values(OptimizerMethod) == ("adam", "sgd")
        assert optimize.METRICS == values(ObjectiveMetric) == ("mean", "max", "compliance")
        assert optimize.GRADIENT_PATHS == ("direct", "tesseract", "tesseract-dc")

    def test_the_enum_and_the_literal_describe_alike(self, study):
        literal = self._optimization(study, metric="max", method="sgd")
        member = self._optimization(study, metric=ObjectiveMetric.MAX, method=OptimizerMethod.SGD)

        assert literal.describe() == member.describe()
        assert type(literal.describe()["metric"]) is str
        assert type(literal.describe()["method"]) is str

    def test_aliases_resolve_to_the_canonical_member(self, study):
        assert self._optimization(study, gradient_path="plugins").gradient_path is (
            GradientPath.TESSERACT
        )
        assert self._optimization(study, gradient_path="plugins-dc").gradient_path is (
            GradientPath.TESSERACT_DC
        )
        assert self._optimization(study).gradient_path is GradientPath.DIRECT

    def test_unknown_values_name_the_members(self, study):
        with pytest.raises(ValueError, match=r"^method must be one of: adam, sgd\.$"):
            self._optimization(study, method="newton")
        with pytest.raises(
            ValueError, match=r"^metric must be one of: mean, max, compliance \(got 'rms'\)\.$"
        ):
            self._optimization(study, metric="rms")
        with pytest.raises(ValueError, match="gradient_path must be one of: direct, tesseract"):
            self._optimization(study, gradient_path="fastest")


class TestStudyPayloads:
    def test_boundary_conditions_report_their_enum_value(self):
        from cadjoint.fem import Dirichlet, Fixed, HeatFlux, Nodes, Traction

        payloads = [
            Dirichlet(Nodes.side("-x"), 1.0).describe(),
            HeatFlux(Nodes.side("+x"), 2.0).describe(),
            Fixed(Nodes.side("-y")).describe(),
            Traction(Nodes.side("+y"), (0.0, 0.0, 1.0)).describe(),
        ]

        assert [payload["type"] for payload in payloads] == list(values(BoundaryConditionType))
        assert all(type(payload["type"]) is str for payload in payloads)

    def test_study_kinds_are_the_enum_values(self):
        from cadjoint.fem import Dirichlet, ElasticStudy, Fixed, Nodes, ThermalStudy

        thermal = ThermalStudy(
            name="t", resolution=4, conductivity=1.0, bcs=[Dirichlet(Nodes.side("-x"), 1.0)]
        )
        elastic = ElasticStudy(
            name="e", resolution=4, youngs=1.0, poisson=0.3, bcs=[Fixed(Nodes.side("-x"))]
        )

        assert thermal.describe()["kind"] == StudyKind.THERMAL
        assert elastic.describe()["kind"] == StudyKind.ELASTIC
        assert type(thermal.describe()["kind"]) is str


class TestBackends:
    def test_the_registry_holds_the_built_in_names(self):
        from cadjoint.fem.backends import available_backends

        assert set(values(FemBackend)) <= set(available_backends())
        assert all(type(name) is str for name in available_backends())

    def test_the_registry_stays_open(self):
        """The enum names the built-ins; it does not close the registry."""
        from cadjoint.fem.backends import _REGISTRY, get_backend, register_backend

        sentinel = object()
        register_backend("third_party", lambda: sentinel)
        try:
            assert get_backend("third_party") is sentinel
        finally:
            _REGISTRY.pop("third_party")


class TestPlugins:
    def test_the_enum_and_the_literal_build_the_same_spec(self, tmp_path):
        from cadjoint.plugins.spec import PluginSpec

        api = tmp_path / "tesseract_api.py"
        api.write_text("", encoding="utf-8")
        literal = PluginSpec(name="p", kind="mesher", transport="local", api_path=api)
        member = PluginSpec(
            name="p", kind=PluginKind.MESHER, transport=PluginTransport.LOCAL, api_path=api
        )

        assert literal == member
        assert literal.to_mapping() == member.to_mapping()
        assert type(literal.to_mapping()["transport"]) is str
        assert literal.transport is PluginTransport.LOCAL

    def test_an_unknown_transport_names_the_members(self):
        from cadjoint.plugins.spec import PluginConfigError, PluginSpec

        with pytest.raises(PluginConfigError) as error:
            PluginSpec(name="p", kind="mesher", transport="carrier-pigeon")

        assert str(error.value) == (
            "plugin 'p': transport must be one of local, container, remote, python "
            "(got 'carrier-pigeon')."
        )

    def test_kinds_are_the_slots_the_pipeline_asks_for(self):
        """Derived from the registry, not restated: a slot the built-ins
        stop filling has to leave the enum with them.
        """
        from cadjoint import tier
        from cadjoint.plugins.registry import BUILTIN_DEFAULTS, BUILTIN_PACKAGES, KINDS

        assert KINDS == values(PluginKind)
        shipped = {kind for kind, _ in BUILTIN_PACKAGES.values()}
        # Every shipped package fills a slot the enum names, and the defaults
        # choose among the packages.  A retired slot (`qef`) or a new one
        # (`tet_mesher`) fails here the moment the two disagree.
        assert shipped <= set(KINDS)
        assert set(BUILTIN_DEFAULTS) <= set(KINDS)
        assert set(BUILTIN_DEFAULTS.values()) <= set(BUILTIN_PACKAGES)
        # The slots with no shipped package are exactly the private tier's:
        # they are discovered from an installed distribution, never bundled
        # (``research/two-tier.md`` §2.3).  A new public kind that forgets its
        # package lands here rather than silently resolving to nothing.
        assert set(KINDS) - shipped == set(tier.KINDS)
        assert set(BUILTIN_DEFAULTS).isdisjoint(tier.KINDS)

    def test_a_plugin_may_declare_a_kind_cadjoint_never_heard_of(self, tmp_path):
        """``PluginSpec.kind`` is open on purpose: it comes from config."""
        from cadjoint.plugins.spec import PluginSpec

        api = tmp_path / "tesseract_api.py"
        api.write_text("", encoding="utf-8")
        spec = PluginSpec(name="p", kind="acoustic_solver", api_path=api)

        assert spec.kind == "acoustic_solver"
        assert spec.to_mapping()["kind"] == "acoustic_solver"


class TestRequestSchemas:
    """Each model's JSON schema enumerates exactly the enum's members."""

    @pytest.mark.parametrize(
        ("model_name", "field", "option"),
        [
            ("add_study", "kind", StudyKind),
            ("add_study_bc", "bc_type", BoundaryConditionType),
            ("add_constraint", "kind", ConstraintKind),
            ("solve_sketch", "method", ConstraintSolveMethod),
            ("set_optimization_value", "argument", OptimizationArgument),
        ],
    )
    def test_schema_enumerates_the_members(self, model_name, field, option):
        from cadjoint.viewer.schema.requests import PATCH_REQUEST_MODELS

        schema = PATCH_REQUEST_MODELS[model_name].model_json_schema()
        reference = schema["properties"][field]
        name = (reference.get("$ref") or reference["allOf"][0]["$ref"]).rsplit("/", 1)[-1]

        assert name == option.__name__
        assert tuple(schema["$defs"][name]["enum"]) == values(option)

    def test_the_generated_typescript_is_a_union_of_the_same_literals(self):
        from cadjoint.viewer.schema.emit import TYPESCRIPT_PATH

        source = TYPESCRIPT_PATH.read_text(encoding="utf-8")
        for option in (
            StudyKind,
            BoundaryConditionType,
            ConstraintKind,
            ConstraintSolveMethod,
            OptimizationArgument,
        ):
            union = " | ".join(f'"{value}"' for value in values(option))
            assert f"export type {option.__name__} = {union};" in source


class TestValidatorMessages:
    """The rejection strings are derived, and read exactly as they always did."""

    def _rejected(self, operation, request):
        from cadjoint.viewer._patch_requests import PATCH_VALIDATORS

        error, _ = PATCH_VALIDATORS[operation](request)
        return error["error"]

    def test_study_kind(self):
        message = self._rejected("add_study", {"kind": "acoustic"})
        assert message == "Study `kind` must be `thermal` or `elastic`."

    def test_bc_type(self):
        message = self._rejected("add_study_bc", {"study": 0, "bc_type": "neumann"})
        assert message == "`bc_type` must be one of: dirichlet, heat_flux, fixed, traction."

    def test_mesh_method(self):
        message = self._rejected(
            "set_mesh_value", {"mesh": 0, "argument": "method", "value": "voxel"}
        )
        assert message == "Mesh `method` must be one of: hex, tet4, tet10."

    def test_solver_method(self):
        message = self._rejected("solve_sketch", {"line": 1, "method": "bfgs"})
        assert message == "Solver `method` must be `newton`, `adam`, or `sgd`."

    def test_constraint_kind(self):
        message = self._rejected("add_constraint", {"line": 1, "kind": "tangent", "indices": [0]})
        assert message == (
            "Constraint `kind` must be one of: coincident, distance, fixed, horizontal, "
            "parallel, perpendicular, vertical."
        )

    def test_the_solver_default_is_the_member_and_writes_the_literal(self):
        """The validator fills in the enum; the patch writes ``'newton'``."""
        from cadjoint.viewer._patch_requests import PATCH_VALIDATORS

        error, arguments = PATCH_VALIDATORS["solve_sketch"]({"line": 1})

        assert error is None
        assert arguments["method"] is ConstraintSolveMethod.NEWTON
        assert repr(arguments["method"]) == repr("newton")

    def test_the_patch_operations_reject_the_same_way(self):
        """The patch layer derives its own copies from the same enums."""
        from cadjoint.viewer.patch.errors import PatchError
        from cadjoint.viewer.patch.studies import add_study

        with pytest.raises(PatchError, match=r"^Study `kind` must be `thermal` or `elastic`\.$"):
            add_study("scene = None\n", "acoustic")
