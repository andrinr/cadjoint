# Benchmarks

`edge_detection_bench.py` exercises stage 1 of the differentiable meshing pipeline
(`cadjoint.meshing.edge_detection`) on three reference shapes — a unit sphere, a box with
half-extents (0.4, 0.5, 0.6), and a union of two spheres — across several grid resolutions.
For each case it reports crossing-edge counts, warm wall times for `sample_grid`,
`find_crossing_edges`, and `edge_hermite_data` (cold first-call compile time listed
separately), hermite-stage throughput in edges/second, root residuals, the sphere's
geometric error, and the autodiff dt/dr accuracy against the analytic value. When
scikit-image is installed, `skimage.measure.marching_cubes` on the same volumes is timed
for context. Run with:

    python benchmarks/edge_detection_bench.py --resolutions 16 32 64 [--repeats 3] [--json out.json]

`dual_contouring_bench.py` covers stage 3 (`cadjoint.meshing.dual_contouring`): extraction
wall time, watertightness, signed-volume error, 5th-percentile triangle minimum angle,
sharp-corner placement error against scikit-image marching cubes on the same volume, and
the wall time of one reverse-mode mesh-loss gradient. Run with:

    python benchmarks/dual_contouring_bench.py --resolutions 16 32 64 [--repeats 3] [--json out.json]

`suite_compile_probe.py` is a pytest plugin and its report: every XLA compile the
suite triggers, with its program, seconds, HLO size, module digest and the
repository frames that asked for it, one JSON line per compile per process. The
report separates what a cache-off run pays, the in-worker repeats a persistent
cache would serve from disk, and the globally unique programs that only changing
the code can make cheaper (`research/performance.md` §19). Run with:

    PYTHONPATH=.:benchmarks CADJOINT_NO_COMPILATION_CACHE=1 COMPILE_PROBE_DIR=/tmp/probe \
        pytest tests -m "not slow" -n 4 -p suite_compile_probe -p no:cacheprovider
    python benchmarks/suite_compile_probe.py /tmp/probe
