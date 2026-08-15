# Benchmarks

`edge_detection_bench.py` exercises stage 1 of the differentiable meshing pipeline
(`jaxcad.meshing.edge_detection`) on three reference shapes — a unit sphere, a box with
half-extents (0.4, 0.5, 0.6), and a union of two spheres — across several grid resolutions.
For each case it reports crossing-edge counts, warm wall times for `sample_grid`,
`find_crossing_edges`, and `edge_hermite_data` (cold first-call compile time listed
separately), hermite-stage throughput in edges/second, root residuals, the sphere's
geometric error, and the autodiff dt/dr accuracy against the analytic value. When
scikit-image is installed, `skimage.measure.marching_cubes` on the same volumes is timed
for context. Run with:

    python benchmarks/edge_detection_bench.py --resolutions 16 32 64 [--repeats 3] [--json out.json]
