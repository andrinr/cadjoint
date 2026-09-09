"""A sampler that maps a scene field must compile it, not dispatch it op by op.

``jax.vmap(field)(points)`` on a bare scene node runs the map *eagerly*: JAX
takes every primitive of the whole SDF tree through its own one-op program, so
what one sample costs follows how big the scene is rather than how many points
there are.  ``edge_hermite_data`` says so in its own comment and one
``jax.jit`` fixes it, but the same call was still written bare in a dozen
places — the lattice edge overlay above all, which reads every world-frame
leaf three times per request and ran a four-sweep Newton solve over all of
them a primitive at a time.

**What these tests assert, and why the number is one.**  Each sampler is given
a *fresh* field and must announce exactly one compilation.  A compiled map is
one program whatever the tree contains.  An eager map announces **zero** here,
which is the failure and not the ideal: its one-op programs are keyed on
primitive and shape, so a second field of the same shape is served from the
in-process cache and nothing is announced — while the request still pays a
Python dispatch per primitive per read, which is where the seconds went.
Count programs to see the shape of the computation; count seconds elsewhere.
The warm-up call before each measurement takes the shape-only programs (an
``iota``, a scatter — the same for any field) out of the count.

Same assertion, and the same reason for it, as
:mod:`tests.constraints.test_compiled_programs`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

_COMPILING = re.compile(r"Compiling jit\(")


class _CompileCounter(logging.Handler):
    """Counts the compilations announced by ``jax.log_compiles``."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if _COMPILING.search(record.getMessage()):
            self.count += 1


def _count_compiles(work: Callable[[], object]) -> int:
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


def _programs(run: Callable[[], object]) -> int:
    """What ``run`` compiles for a fresh field, past the shape-only programs."""
    run()
    return _count_compiles(run)


def _one_program(run: Callable[[], object], what: str) -> None:
    compiles = _programs(run)
    assert compiles == 1, (
        f"{what} compiled {compiles} programs for a fresh field; one means the map "
        "is compiled, zero means it is dispatched a primitive at a time"
    )


def _sphere(radius: float = 1.0, shift: float = 0.0) -> Callable[..., jax.Array]:
    """A fresh field object every call, so nothing is served from a trace cache."""

    def field(p):
        centre = jnp.asarray([shift, 0.0, 0.0])
        return jnp.sqrt(jnp.sum((p - centre) ** 2) + 1e-30) - radius

    return field


def _points(count: int, seed: int) -> np.ndarray:
    return np.asarray(np.random.default_rng(seed).normal(size=(count, 3)) * 0.4, dtype=np.float64)


def test_the_branch_classifier_compiles_one_program():
    """`active_branches` reads every operand of a hard CSG min/max."""
    from cadjoint.meshing.features import active_branches

    points = jnp.asarray(_points(64, 1), dtype=jnp.float32)

    def run() -> object:
        return active_branches([_sphere(), _sphere(0.6), _sphere(shift=0.4)], points)

    _one_program(run, "active_branches")


def test_the_seam_projection_compiles_one_program():
    """The lattice overlay's seam solve — the largest eager storm of the lot.

    Five reads of every used leaf's value and gradient, then four Newton
    sweeps of stacks, solves and einsums.  One program now, and one whatever
    the sweep count: the sweeps are a ``fori_loop``, so the leaves are lowered
    once rather than four times over.  Jitting *without* rolling measured
    worse than the eager path it replaced — ``research/performance.md`` §16.
    """
    from cadjoint.viewer._edge_overlay import _project_seam_groups

    rows = np.arange(24, dtype=np.int64)
    vertices = _points(24, 5)

    def run() -> object:
        leaves = [_sphere(), _sphere(0.9, shift=0.5)]
        return _project_seam_groups(leaves, [(rows, (0, 1))], vertices, 0.2)

    _one_program(run, "_project_seam_groups")


def test_the_seam_residual_compiles_one_program():
    """The acceptance test a projected seam group is judged by."""
    from cadjoint.viewer._edge_overlay import _seam_residual

    points = _points(24, 6)

    def run() -> object:
        return _seam_residual([_sphere(), _sphere(0.6), _sphere(shift=0.4)], points)

    _one_program(run, "_seam_residual")


def test_patch_signatures_compile_one_program():
    """Labelling points by owning leaf and patch reads every leaf and every patch."""
    from cadjoint.meshing.patch_fields import patch_signatures
    from cadjoint.sdf.primitives import Box, Sphere

    points = jnp.asarray(_points(32, 7), dtype=jnp.float32)

    def run() -> object:
        return patch_signatures(Sphere(radius=1.0) | Box(size=(0.8, 0.8, 0.8)), points)

    _one_program(run, "patch_signatures")


def test_the_material_field_is_sampled_by_one_program():
    """`sample_material_field` walks the whole material tree once per element."""
    from cadjoint.fem.properties import sample_material_field
    from cadjoint.sdf.primitives import Box, Sphere

    cells = np.arange(8 * 6, dtype=np.int64).reshape(6, 8)
    points = _points(int(cells.max()) + 1, 8)

    def run() -> object:
        return sample_material_field(Sphere(radius=1.0) | Box(size=(1.0, 1.0, 1.0)), points, cells)

    _one_program(run, "sample_material_field")
