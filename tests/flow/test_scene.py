"""``scenes/duct_sink.py``: the demonstration, run as a test.

A scene is the only place the pieces meet the way a user meets them --
parameters, an SDF, a study, a solve, a derivative -- so it is worth one
test that runs the real file rather than a reconstruction of it.  What is
asserted here is physics that would survive a rewrite of the scene: the air
takes heat away, the sink resists the flow, and energy is conserved.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "scenes" / "duct_sink.py"


def _run(script: str) -> subprocess.CompletedProcess:
    """Run ``script`` in a fresh interpreter rooted at the repository."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=900,
    )


@pytest.fixture(scope="module")
def scene():
    """Import ``scenes/duct_sink.py`` as a module."""
    spec = importlib.util.spec_from_file_location("duct_sink_scene", SCENE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        sys.modules.pop(spec.name, None)


@pytest.fixture(scope="module")
def solved(scene):
    """One conjugate solve of the scene, shared by every assertion."""
    from cadjoint import extract_parameters, functionalize

    free, fixed, _ = extract_parameters(scene.scene)
    return scene.cooling.solve(functionalize(scene.scene)(free, fixed))


class TestTheSceneCompiles:
    """A scene that cannot be opened in the viewer is a broken scene.

    These run in a **subprocess** on purpose.  What they check is a
    process-global -- ``jax_enable_x64`` -- and the flow suite's own conftest
    turns it on package-wide, so a check inside this process would pass on a
    scene that breaks every viewer that loads it.  The subprocess starts in
    the viewer's condition: single precision, nothing enabled.
    """

    def test_the_scene_compiles_to_wgsl_in_single_precision(self):
        """The regression this exists for.

        ``scenes/duct_sink.py`` used to call ``jax.config.update`` at module
        scope, which is a *process* setting: the viewer's compile worker
        imports the scene, every array afterwards became float64, and the
        WGSL emitter has nothing to emit for one -- WebGPU has no ``f64``.
        The solve scopes double precision to itself now
        (``cadjoint.flow.precision``), so importing the scene leaves the
        process exactly as it found it.
        """
        script = """
import jax
assert jax.config.jax_enable_x64 is False, "x64 was on before the scene loaded"
from cadjoint.backends.wgsl import compile_scene_to_wgsl
from cadjoint.viewer._worker_scene import _execute_scene
namespace = _execute_scene(open("scenes/duct_sink.py").read())
emitted = compile_scene_to_wgsl(namespace["scene"])
shader = emitted[0] if isinstance(emitted, tuple) else emitted
assert len(str(shader)) > 1000, "no shader emitted"
assert jax.config.jax_enable_x64 is False, "the scene left x64 on process-wide"
assert [s.name for s in namespace["__studies__"]] == ["duct-cooling"]
print("OK", len(str(shader)), namespace["__studies__"][0].name)
"""
        finished = _run(script)

        assert finished.returncode == 0, finished.stderr[-4000:]
        assert "OK" in finished.stdout
        assert "duct-cooling" in finished.stdout

    @pytest.mark.parametrize(
        "name", ["duct_sink.py", "starter.py", "bracket.py", "end_cap.py", "motor_shield.py"]
    )
    def test_every_scene_still_compiles(self, name):
        """Not only the new one: a precision leak from any scene would take
        the others down with it in the same worker process."""
        script = f"""
from cadjoint.backends.wgsl import compile_scene_to_wgsl
from cadjoint.viewer._worker_scene import _execute_scene
namespace = _execute_scene(open("scenes/{name}").read())
emitted = compile_scene_to_wgsl(namespace["scene"])
shader = emitted[0] if isinstance(emitted, tuple) else emitted
assert len(str(shader)) > 1000
print("OK")
"""
        finished = _run(script)

        assert finished.returncode == 0, finished.stderr[-4000:]
        assert "OK" in finished.stdout

    def test_the_viewer_compile_path_accepts_the_scene(self):
        """The whole worker path, not only the shader.

        Emitting WGSL is necessary and was not sufficient: the compile
        worker also serializes every declared study, and it assumes each
        boundary condition carries a node selection. A flow study's ``Inlet``
        does not, so the compile died with an ``AttributeError`` on a scene
        whose shader was fine. The worker asks the *condition* whether it
        serializes now -- the only question every condition can answer --
        and the payload model's ``nodes`` is optional.
        """
        script = """
from cadjoint.viewer._compile_worker import _compile_source
payload = _compile_source(open("scenes/duct_sink.py").read())
assert payload["ok"] is True, payload
assert len(payload["shader"]) > 1000, "no shader"
study, = payload["studies"]
assert study["kind"] == "flow" and study["name"] == "duct-cooling", study
assert study["resolution"] == [14, 26, 14] and study["mesh"] is None, study
types = [bc["type"] for bc in study["bcs"]]
assert types == ["inlet", "outlet", "walls", "heat_source"], types
assert all(bc["serializable"] for bc in study["bcs"]), study["bcs"]
assert study["bcs"][0]["velocity"] == [0.0, 0.02, 0.0], study["bcs"][0]
assert study["bcs"][3]["nodes"]["kind"] == "box", study["bcs"][3]
print("OK", len(payload["shader"]), types)
"""
        finished = _run(script)

        assert finished.returncode == 0, finished.stderr[-4000:]
        assert "OK" in finished.stdout

    def test_solving_leaves_the_precision_setting_alone(self):
        """The scope restores what it found, so a forward solve in a
        single-precision process does not strand it in float64."""
        script = """
import jax
import jax.numpy as jnp
from cadjoint.viewer._worker_scene import _execute_scene
namespace = _execute_scene(open("scenes/duct_sink.py").read())
study = namespace["__studies__"][0]
before = jax.config.jax_enable_x64
result = study.solve(chi=jnp.zeros(study.resolution))
assert jax.config.jax_enable_x64 is before, "solve() left x64 flipped"
assert jnp.zeros(3).dtype == jnp.float32, "the process is no longer float32"
print("OK", float(result.peak_temperature))
"""
        finished = _run(script)

        assert finished.returncode == 0, finished.stderr[-4000:]
        assert "OK" in finished.stdout


class TestTheDeclarationSerializes:
    """``_study_entries`` and the payload model, without running a solve.

    The compile test above proves the whole path works; these two say
    *which* part of it the flow study needed, so a change to either side
    fails here with a short message rather than in a 40-second subprocess.
    """

    def test_study_entries_serializes_a_flow_study(self):
        from cadjoint.viewer._worker_declarations import _study_entries
        from cadjoint.viewer._worker_scene import _execute_scene

        source = SCENE.read_text(encoding="utf-8")
        namespace = _execute_scene(source)

        entries = _study_entries(namespace["__studies__"], source)

        assert len(entries) == 1
        entry = entries[0]
        assert entry["kind"] == "flow"
        assert entry["mesh"] is None
        assert [bc["type"] for bc in entry["bcs"]] == [
            "inlet",
            "outlet",
            "walls",
            "heat_source",
        ]
        # The seam: a condition answers for itself, and three of these four
        # have no node selection to have been asked instead.
        assert all(bc["serializable"] for bc in entry["bcs"])
        assert [bool(bc.get("nodes")) for bc in entry["bcs"]] == [False, False, False, True]

    def test_the_payload_model_accepts_it(self):
        """The compile worker validates against this model before sending,
        so a shape it rejects is a scene the viewer cannot open."""
        from cadjoint.viewer._worker_declarations import _study_entries
        from cadjoint.viewer._worker_scene import _execute_scene
        from cadjoint.viewer.schema.payloads import StudyPayload

        source = SCENE.read_text(encoding="utf-8")
        namespace = _execute_scene(source)
        entry = _study_entries(namespace["__studies__"], source)[0]

        payload = StudyPayload.model_validate(entry)

        assert payload.kind == "flow"
        assert payload.bcs[0].nodes is None
        assert payload.bcs[0].velocity == [0.0, 0.02, 0.0]
        assert payload.bcs[3].nodes is not None


class TestTheSceneSolves:
    """The demonstration produces a coherent answer, and says so."""

    def test_the_solve_converges_and_conserves(self, solved):
        """Energy in equals energy out to round-off.

        On this scene that check has already earned itself twice: it is what
        caught the convective flux being the velocity instead of the mass
        flux, and it is what caught restarted GMRES stagnating at the
        scene's conductivity ratio and reporting a peak temperature forty
        times too low.
        """
        assert bool(jnp.all(jnp.isfinite(solved.temperature)))
        assert abs(float(solved.energy_imbalance)) < 1e-8

    def test_the_study_warns_about_nothing(self, solved):
        assert solved.warnings() == []

    def test_the_sink_is_hotter_than_the_air_leaving(self, solved):
        """Heat flows the right way: metal above the mixed air, air above
        the inlet.  Reversed, and the sign of the coupling would be wrong."""
        assert float(solved.peak_temperature) > float(solved.mean_temperature)
        assert float(solved.mean_temperature) > float(solved.bulk_outlet_temperature)
        assert float(solved.bulk_outlet_temperature) > 0.0

    def test_the_sink_costs_pressure(self, solved, scene):
        """A positive drop, and several times an empty duct's.

        The sign is worth pinning: the scene's first draft had the sink
        filling 87% of every cross-section, which drove the density down
        by 18% and inverted the drop. A blockage that reads as plausible in
        source can be nonsense on the lattice.
        """
        from cadjoint.flow import solve as flow_solve

        empty = flow_solve(
            jnp.zeros(scene.cooling.resolution),
            scene.cooling.flow_config(),
            inlet_velocity=jnp.asarray(scene.cooling.inlet.velocity),
        )

        assert float(solved.pressure_drop) > 0.0
        assert float(solved.pressure_drop) > 4.0 * float(empty.pressure_drop)

    def test_the_duct_is_open_where_it_should_be(self, solved):
        """Clear upstream and downstream, blocked only across the sink."""
        from cadjoint.flow import duct_walls

        wall = duct_walls(scene_shape := tuple(solved.chi.shape))
        open_cells = np.asarray(~wall, dtype=float)
        blockage = np.asarray(jnp.sum(solved.chi * (~wall), axis=(0, 2))) / open_cells.sum(
            axis=(0, 2)
        )

        assert scene_shape == (14, 26, 14)
        assert blockage[0] == pytest.approx(0.0, abs=1e-12)
        assert blockage[-1] == pytest.approx(0.0, abs=1e-12)
        assert 0.2 < blockage.max() < 0.6

    def test_the_regime_is_the_one_the_scene_claims(self, solved):
        """Laminar, forced, and resolved enough for the exponential scheme."""
        assert solved.reynolds == pytest.approx(25.0)
        assert solved.richardson < 0.1
        assert solved.peclet_cell < 2.0
