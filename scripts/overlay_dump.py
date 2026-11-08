"""Dump the viewer's lattice sharp layer for the clevis at a 1 mm cell (run under either checkout)."""
import math, sys, numpy as np
from cadjoint.meshing import GridSpec
from cadjoint.meshing.bounds import fit_bounds
from cadjoint.viewer._lattice_edges import _lattice_layers
from tests.meshing.fixtures import clevis_bracket
scene = clevis_bracket.scene
origin, extent = fit_bounds(scene)
grid = GridSpec.from_bounds(origin, extent, tuple(max(1, math.ceil(v - 1e-9)) for v in extent))
_, _, sharp = _lattice_layers(scene, grid)
np.save(sys.argv[1], sharp); print(len(sharp))
