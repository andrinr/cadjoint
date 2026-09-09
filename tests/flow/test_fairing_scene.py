"""``scenes/duct_fairing.py``: immersed shape optimization, run as a test.

``tests/flow/test_scene.py`` covers the conjugate scene, which takes a
derivative and stops.  This file covers the one that *descends*: a body
immersed in a duct as a Brinkman solid fraction, a pressure drop off the
converged flow, and a declared ``Optimization`` walking the half-extents
downhill at constant volume.

Three things are worth a test here and they cost very different amounts.

**The declaration** is free -- importing the scene solves nothing -- and it
is where a rename or a stray free parameter shows up first.

**The precision contract** costs one flow solve and one adjoint, and is the
only new mechanism the scene needed: a flow objective differentiated from a
float32 process dies in the backward pass, because the solver's own x64
scope covers its forward call and has closed by the time :func:`jax.grad`
transposes it.  ``precision="double"`` on the optimization holds the flag
for the whole loop.  The test runs *both* spellings in one subprocess, so
what is pinned is not merely that the option works but that it is load
bearing -- if the failure it prevents ever stops happening, this says so
rather than quietly passing.

**The descent** costs minutes and is marked ``slow``.  It runs the scene's
own ``main``, which asserts its finite-difference check internally: the
relative error between the adjoint and a central difference has to fall by
about a hundred when the step size falls by ten, which is the second-order
convergence that makes the agreement mean something.  A wrong gradient that
happens to descend is exactly the failure this scene is exposed to, and the
loss curve does not reveal it.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "scenes" / "duct_fairing.py"


def _run(script: str, timeout: int = 900) -> subprocess.CompletedProcess:
    """Run ``script`` in a fresh interpreter rooted at the repository."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=timeout,
    )


@pytest.fixture(scope="module")
def scene():
    """Import ``scenes/duct_fairing.py`` as a module (no solve)."""
    spec = importlib.util.spec_from_file_location("duct_fairing_scene", SCENE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        sys.modules.pop(spec.name, None)


class TestTheDeclaration:
    """What the scene says it is, before anything is solved."""

    def test_it_declares_one_flow_study_and_one_optimization(self, scene):
        from cadjoint.flow import FLOW_STUDY_KIND, FlowStudy
        from cadjoint.optimize import Optimization

        assert isinstance(scene.drag, FlowStudy)
        assert scene.drag.describe()["kind"] == FLOW_STUDY_KIND
        assert isinstance(scene.streamline, Optimization)
        assert scene.streamline.study is None, "this is the objective form"

    def test_the_lattice_cells_are_cubes(self, scene):
        """The solve is in lattice units and the world size only decides
        where the SDF is sampled, so unequal spacings hand the solver a
        stretched duct.  ``FlowStudyResult.warnings`` reports it after a
        solve; this catches it at declaration, for free."""
        spacing = scene.drag.grid.spacing

        assert max(spacing) == pytest.approx(min(spacing), rel=1e-12)

    def test_only_the_shape_is_free(self, scene):
        """The body's position is pinned on purpose: a box free to move in a
        duct with a symmetric objective has a flat direction and a wall to
        drift into."""
        from cadjoint import extract_parameters

        free, _, _ = extract_parameters(scene.fairing)

        assert list(free) == ["fairing_size"]
        assert scene.streamline.describe()["parameters"] == ["fairing_size"]

    def test_the_descent_asks_for_double_precision(self, scene):
        from cadjoint.enums import Precision

        assert scene.streamline.precision is Precision.DOUBLE

    def test_the_objective_holds_the_volume(self, scene):
        """Without the second term the minimum is "delete the obstacle", so
        the weight being non-zero is the whole reason the answer is a shape
        rather than an absence."""
        assert scene.VOLUME_WEIGHT > 0.0
        assert scene.REFERENCE_VOLUME == pytest.approx(8 * 0.20 * 0.15 * 0.20)


class TestTheSceneCompiles:
    """A scene that cannot be opened in the viewer is a broken scene.

    In a **subprocess**, because what these check is the process-global
    ``jax_enable_x64`` and this package's conftest turns it on for the whole
    suite -- an in-process check would pass on a scene that breaks every
    viewer that loads it.
    """

    def test_the_scene_compiles_to_wgsl_in_single_precision(self):
        script = """
import jax
assert jax.config.jax_enable_x64 is False, "x64 was on before the scene loaded"
from cadjoint.backends.wgsl import compile_scene_to_wgsl
from cadjoint.viewer.worker.scene import _execute_scene
namespace = _execute_scene(open("scenes/duct_fairing.py").read())
emitted = compile_scene_to_wgsl(namespace["scene"])
shader = emitted[0] if isinstance(emitted, tuple) else emitted
assert len(str(shader)) > 1000, "no shader emitted"
assert jax.config.jax_enable_x64 is False, "the scene left x64 on process-wide"
assert [s.name for s in namespace["__studies__"]] == ["duct-drag"]
assert [o.name for o in namespace["__optimizations__"]] == ["streamline"]
print("OK", len(str(shader)))
"""
        finished = _run(script)

        assert finished.returncode == 0, finished.stderr[-4000:]
        assert "OK" in finished.stdout

    def test_the_viewer_compile_path_accepts_the_scene(self):
        """The whole worker path: shader, study payload and optimization
        payload, which is what the playground's panels read."""
        script = """
from cadjoint.viewer.worker.main import _compile_source
payload = _compile_source(open("scenes/duct_fairing.py").read())
assert payload["ok"] is True, payload
assert len(payload["shader"]) > 1000, "no shader"
study, = payload["studies"]
assert study["kind"] == "flow" and study["name"] == "duct-drag", study
assert study["resolution"] == [16, 32, 16], study
assert [bc["type"] for bc in study["bcs"]] == ["inlet", "outlet", "walls"], study
optimization, = payload["optimizations"]
assert optimization["name"] == "streamline", optimization
assert optimization["parameters"] == ["fairing_size"], optimization
assert optimization["study"] is None, optimization
print("OK", optimization["steps"], optimization["method"])
"""
        finished = _run(script)

        assert finished.returncode == 0, finished.stderr[-4000:]
        assert "OK" in finished.stdout


class TestThePrecisionContract:
    """The one mechanism the scene needed that did not exist before."""

    def test_a_flow_descent_runs_from_a_float32_process_only_in_double(self):
        """Both spellings, one subprocess, because the interesting assertion
        is the *difference* between them.

        ``precision="single"`` is what every existing optimization declares
        and what the viewer's optimize worker calls into with no ceremony at
        all; on a flow objective it raises in the backward pass.
        ``precision="double"`` is the same descent with the flag held.  If
        the single-precision path ever stops failing, the option has stopped
        being load bearing and this test says so.
        """
        script = """
import jax
assert jax.config.jax_enable_x64 is False
from cadjoint.viewer.worker.scene import _execute_scene
namespace = _execute_scene(open("scenes/duct_fairing.py").read())
streamline = namespace["__optimizations__"][0]

streamline.precision = "single"
try:
    streamline.run(steps=1)
except TypeError as exc:
    print("SINGLE-RAISED", type(exc).__name__)
else:
    raise AssertionError("the single-precision descent was expected to fail")

streamline.precision = "double"
run = streamline.run(steps=1)
assert jax.config.jax_enable_x64 is False, "the run left x64 on process-wide"
before = run.initial["fairing_size"]
after = run.parameters["fairing_size"]
assert before != after, (before, after)
print("DOUBLE-OK", before, after, run.history[0]["objective"])
"""
        finished = _run(script)

        assert finished.returncode == 0, finished.stderr[-4000:]
        assert "SINGLE-RAISED" in finished.stdout
        assert "DOUBLE-OK" in finished.stdout


@pytest.mark.slow
class TestTheDescent:
    """The demonstration itself, gradient check and all."""

    @pytest.fixture(scope="class")
    def output(self):
        finished = _run(
            'exec(open("scenes/duct_fairing.py").read(), {"__name__": "__main__"})',
            timeout=1800,
        )
        assert finished.returncode == 0, finished.stderr[-6000:]
        return finished.stdout

    def test_the_adjoint_agrees_with_a_central_difference(self, output):
        """``main`` asserts the second-order convergence internally, so
        reaching this line is most of the check; the parse is here so a
        failure names the axis rather than a return code."""
        rows = re.findall(r"^  ([xyz]): adjoint (\S+)\s+relative error", output, re.M)

        assert [axis for axis, _ in rows] == ["x", "y", "z"], output
        assert all(float(value) > 0.0 for _, value in rows), rows

    def test_growing_across_the_flow_costs_more_than_growing_along_it(self, output):
        """The trade the descent exploits, present in the gradient before a
        single step is taken."""
        values = [float(v) for _, v in re.findall(r"^  ([xyz]): adjoint (\S+)", output, re.M)]

        across, along, also_across = values
        assert across == pytest.approx(also_across, rel=1e-6), "the duct is x/z symmetric"
        assert across > 3.0 * along

    def test_the_objective_falls(self, output):
        start, end = re.search(r"^objective\s+(\S+) -> (\S+)", output, re.M).groups()

        assert float(end) < 0.75 * float(start), output

    def test_the_body_streamlines(self, output):
        """Longer along the flow, thinner across it: the aspect ratio rises
        and the frontal area falls."""
        aspect = re.search(r"^aspect \(y/x\)\s+(\S+) -> (\S+)", output, re.M).groups()
        frontal = re.search(r"^frontal area\s+(\S+) -> (\S+)", output, re.M).groups()

        assert float(aspect[1]) > float(aspect[0]) + 0.15, output
        assert float(frontal[1]) < float(frontal[0]), output

    def test_the_volume_is_held(self, output):
        ratio = float(re.search(r"^volume .*\((\S+)x\)", output, re.M).group(1))

        assert 0.95 < ratio < 1.05, output

    def test_the_gain_survives_the_constant_volume_counterfactual(self, output):
        """The volume drifts a little, so "the drag fell" is not yet "the
        shape improved".  ``main`` re-solves the *start* box scaled
        isotropically to the finished volume: same material, same
        proportions.  What separates that from the optimised box is shape
        alone, and it has to be most of the reduction."""
        shape_only = float(
            re.search(r"^attributable to shape alone\s+(\S+)%", output, re.M).group(1)
        )

        assert shape_only < -25.0, output
