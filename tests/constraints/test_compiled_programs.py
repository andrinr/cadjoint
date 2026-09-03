"""The constraint solve must stay one compiled program, whatever the step count.

The Newton projection used to run its corrections as a Python ``for`` loop over
eagerly dispatched JAX ops.  Eager dispatch compiles one XLA program *per
primitive operation*, so a solve of ``scenes/starter.py`` laid down 85 tiny
programs and 0.73 s of compilation — 66 % of the wall clock of executing that
scene at all — and every additional step added ~50 ms of Python dispatch on
top.  Rolled into a single ``lax.scan`` under one ``jax.jit`` it is one
program whose compilation does not move with ``steps``, which is the whole
reason asking for more steps is now affordable.

These tests pin that shape.  They are about *compilation*, not arithmetic:
:mod:`tests.constraints.test_redundant` pins what the step computes.
"""

from __future__ import annotations

import logging
import re

import jax.numpy as jnp
import pytest

from cadjoint.constraints import FixedConstraint, HorizontalConstraint, satisfy_constraints
from cadjoint.constraints.residual import (
    _collect_constraints,
    build_residual_fn,
    pack_param_dict,
)
from cadjoint.constraints.solve import _gradient_projection, _newton_projection
from cadjoint.construction import PolygonProfile
from cadjoint.enums import ConstraintSolveMethod
from cadjoint.geometry import Vector2

_COMPILING = re.compile(r"Compiling jit\(")


class _CompileCounter(logging.Handler):
    """Counts XLA compilations announced by ``jax.log_compiles``."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if _COMPILING.search(record.getMessage()):
            self.count += 1


def _count_compiles(work) -> int:
    import jax

    logger = logging.getLogger("jax")
    handler = _CompileCounter()
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        with jax.log_compiles():
            work()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)
    return handler.count


def _zigzag(points: int, tag: str):
    """A strip of points under horizontal relations, pinned at one end."""
    vertices = [
        Vector2(value=[0.5 * i, 0.2 * ((i % 2) - 0.5)], free=True, name=f"{tag}_p{i}")
        for i in range(points)
    ]
    FixedConstraint(vertices[0], [0.0, 0.0])
    for i in range(points - 1):
        HorizontalConstraint(vertices[i], vertices[i + 1])
    metadata = {v.name: v for v in vertices}
    flat_fn = build_residual_fn(_collect_constraints(metadata), metadata)
    return flat_fn, pack_param_dict({k: v.value for k, v in metadata.items()}, metadata)


def test_a_newton_solve_compiles_exactly_one_program():
    flat_fn, x0 = _zigzag(6, "one")
    compiles = _count_compiles(lambda: _newton_projection(flat_fn, x0, 2))
    assert compiles == 1


@pytest.mark.parametrize("steps", [1, 2, 32])
def test_the_program_count_does_not_grow_with_the_step_count(steps):
    """The trade the scanned loop buys: more steps, same compilation."""
    flat_fn, x0 = _zigzag(6, f"steps{steps}")
    assert _count_compiles(lambda: _newton_projection(flat_fn, x0, steps)) == 1


def test_the_residual_is_traced_a_constant_number_of_times():
    """A public proxy for the same thing, if the log format ever moves.

    A Python ``for`` loop calls the residual — and so re-traces it — once per
    step.  A ``lax.scan`` traces its body once, whatever ``length`` is.
    """
    traces: list[int] = []
    for steps in (2, 16):
        flat_fn, x0 = _zigzag(6, f"trace{steps}")
        calls = [0]

        def counted(x, inner=flat_fn, calls=calls):
            calls[0] += 1
            return inner(x)

        _newton_projection(counted, x0, steps)
        traces.append(calls[0])
    assert traces[0] == traces[1], f"residual re-traced per step: {traces}"


@pytest.mark.parametrize("steps", [4, 64])
def test_a_gradient_solve_also_compiles_one_program(steps):
    """The Optax paths are the ones users give large step counts."""
    flat_fn, x0 = _zigzag(6, f"adam{steps}")
    compiles = _count_compiles(
        lambda: _gradient_projection(flat_fn, x0, steps, ConstraintSolveMethod.ADAM)
    )
    assert compiles == 1


def test_a_whole_satisfy_call_is_dominated_by_the_one_solve_program():
    """End to end, packing and unpacking must not reintroduce a program cloud.

    The count is not pinned to an exact number — ``extract_parameters`` and
    ``apply_parameters`` dispatch eagerly and share their op-level programs
    with the rest of a scene — but it must stay small and, above all, flat in
    the step count.
    """
    profile = PolygonProfile([[0.0, 0.0], [2.0, 0.3], [0.0, 1.0]], name="end_to_end")
    FixedConstraint(profile.vertices[0], [0.0, 0.0])
    HorizontalConstraint(profile.vertices[0], profile.vertices[1])
    satisfy_constraints(profile, steps=2)  # warm the shared eager op programs
    few = _count_compiles(lambda: satisfy_constraints(profile, steps=2))
    many = _count_compiles(lambda: satisfy_constraints(profile, steps=64))
    assert few == many == 1, f"satisfy_constraints compiled {few} then {many} programs"


def test_more_steps_still_converge_further():
    """Guard the trade: cheap steps are only worth exposing if they do work."""
    profile = PolygonProfile([[0.0, 0.0], [2.0, 0.7], [0.0, 1.0]], name="converge")
    FixedConstraint(profile.vertices[0], [0.0, 0.0])
    HorizontalConstraint(profile.vertices[0], profile.vertices[1])
    metadata = {v.name: v for v in profile.vertices}
    flat_fn = build_residual_fn(_collect_constraints(metadata), metadata)
    x0 = pack_param_dict({k: v.value for k, v in metadata.items()}, metadata)
    _, losses = _newton_projection(flat_fn, x0, 4)
    assert len(losses) == 5
    assert losses[-1] <= losses[0]
    assert jnp.isfinite(jnp.asarray(losses)).all()
