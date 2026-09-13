"""The compiled projection must not grow with the number of incidence groups.

It once did.  The first compiled form of :func:`project_table` wrote a
Python loop over incidence groups inside the jitted function, which unrolls
at trace time: every group added a complete Newton solve — a linear solve,
an einsum and an ``eigvalsh`` over the whole field — to one program.  On a
real part with 147 groups that program climbed to 15.2 GB while tracing and
was killed, and every CI test job in the repository's history died there,
at the same position in the suite, with no output.

It shipped past verification because the cases measured had a handful of
groups and 11 groups respectively; neither exercised the dimension that
breaks.  This test exercises it: the same tiny scene, a rising group count,
and the program's equation count read directly.  Measured before the fix,
2 → 96 groups took the count from 1 501 to 29 161; after, 1 021 to 1 265.

Deliberately small — a two-surface scene and synthetic incidence — so it
runs in seconds and stays in the fast loop, because a guard that only runs
nightly would have caught this a day late.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from cadjoint.sdf.boolean import Union
from cadjoint.sdf.primitives import Box, Sphere
from cadjoint.zeroset import lower
from cadjoint.zeroset.evaluate import surfaces
from cadjoint.zeroset.project import _table_program


def _equations(jaxpr) -> int:
    """Every equation, including those inside nested jaxprs (jit, loops)."""
    n = 0
    for eqn in jaxpr.eqns:
        n += 1
        for sub in jax.core.jaxprs_in_params(eqn.params):
            n += _equations(sub)
    return n


def _program_size(model, n_groups: int, seed: int = 0) -> int:
    n_surf = len(list(surfaces(model)))
    groups = tuple(
        ((g % n_surf,), np.arange(g * 4, g * 4 + 4, dtype=np.int64)) for g in range(n_groups)
    )
    pts = jnp.asarray(np.random.default_rng(seed).uniform(-1.0, 1.0, size=(n_groups * 4 + 8, 3)))
    program = _table_program(model, groups, 8, None)
    return _equations(jax.make_jaxpr(program)(jnp.asarray(model.theta), pts).jaxpr)


def test_the_program_does_not_grow_with_the_group_count():
    model = lower(Union(Box(size=[0.5, 0.5, 0.5]), Sphere(radius=0.6), smoothness=0.0))
    few, many = _program_size(model, 8), _program_size(model, 96)
    # Twelve times more groups. The unrolled form grew 11.8x here; the
    # gathered form 1.1x. Anything past 2x means a loop is unrolling again.
    assert many <= 2 * few, (
        f"{many} equations at 96 groups against {few} at 8: the projection "
        "program is growing with the group count, which is how it reached 15 GB"
    )
