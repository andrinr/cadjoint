# Viewport frame time (2026-10-18)

What made the playground viewport slow on the real parts, what fixed it, and
how to measure it again.

## Measuring

`frontend/tools/viewport-bench.mjs` drives a running playground through
Playwright's Metal-backed Chromium, loads each shipped scene, and reads the
timings the renderer publishes on `window`:

- `__cadjointBench(frames, interacting)`: preview frames while the camera
  turns, each waited on with `onSubmittedWorkDone`, so an upper bound on frame
  time that is the same bound before and after a change.
- `__cadjointBenchCompile()`: a cold compile (sources salted so no cache
  serves them) to the first finished preview frame and to the path tracer.
- a handle drag: the first free parameter overridden with its own value, so
  the renderer is in its live-drag state with the image unchanged.

```bash
python -m cadjoint.viewer.playground --port 6937
```

```bash
node frontend/tools/viewport-bench.mjs --port 6937 --out /tmp/bench --frames 60 --dpr 2
```

Other sessions share the GPU, and a single run of main has read anywhere from
52 to 75 ms on the same scene. The numbers below are medians of 60 frames,
then the median over three alternated runs at DPR 1 and the range over two at
DPR 2, main and this branch on two servers side by side.

## Results

Apple M-series, Metal through Dawn, Chromium. DPR 1 is a 660 x 778
framebuffer, DPR 2 is 1320 x 1556. Times in ms.

### DPR 1

| scene          | at rest      | camera gesture | handle drag  | cold compile to first frame |
| -------------- | ------------ | -------------- | ------------ | --------------------------- |
| starter        | 1.3 → 1.4    | 1.2 → 1.0      | 1.2 → 1.0    | 13 → 15                     |
| end_cap        | 10.0 → 3.2   | 9.6 → 3.2      | 9.0 → 3.2    | 40 → 21                     |
| cylinder_block | 55.6 → 6.2   | 54.5 → 6.0     | 55.2 → 7.1   | 172 → 38                    |
| motor_shield   | 53.8 → 17.6  | 55.3 → 8.5     | 54.8 → 14.3  | 187 → 69                    |

### DPR 2 (closer to the default Ultra tier on a Retina screen)

| scene          | at rest       | camera gesture | handle drag   | cold compile to first frame |
| -------------- | ------------- | -------------- | ------------- | --------------------------- |
| starter        | 3.4 → 3.0     | 3.4 → 3.1      | 3.4 → 3.0     | 15 → 20                     |
| end_cap        | 33-41 → 9-10  | 33-41 → 9-11   | 33-41 → 10-12 | 99-117 → 35-42              |
| cylinder_block | 163-174 → 20-22 | 161-189 → 10-14 | 168-188 → 13-14 | 503-593 → 67-76        |
| motor_shield   | 193 → 56-60   | 168-196 → 13-17 | 179-200 → 14-18 | 566-644 → 167-280      |

Starter is within noise: it was already cheap, and the gesture preview leaves
any scene under budget at full resolution.

The image at rest is unchanged. Against main, at most 10 of 513k pixels differ
at DPR 1, by at most 4/255 (main differs from itself by 11 pixels on
motor_shield), and at DPR 2 at most 51 of 2.05M, two of them by more than
8/255: single-pixel speckle where two silhouette edges cross.

## What it was

1. **The direct backend did no culling.** It has been the viewport's shader
   since it became the default, and unlike the traced form it evaluated every
   leaf of a part at every march step. Its own comment said so; the
   frontend's said culling was worth 2x. `compile_sdf_direct(culling=True)`
   now emits the same exact skip tests as `_culling.py` (unions, differences,
   pattern instances), with the boxes in the parameter buffer so an edit is
   still a buffer write.
2. **Uniform-buffer reads inside the skip branches.** Culling with the boxes
   in a `var<uniform>` block bought motor_shield nothing (52-62 ms against
   55 flat), and with every test failing it cost 108 ms. The same module with
   the block declared `var<storage, read>` drew in 18-33 ms. The flat module
   costs the same in either address space, so this only shows once there are
   branches. `PARAMETER_ADDRESS_SPACE` in `codegen.py` is now storage for
   both backends.
3. **Two buffer loads per polygon edge.** Profiles dominate the per-step cost
   (motor_shield reads 644 vertices per field evaluation). Carrying the
   previous vertex halved what was left: cylinder_block 11 → 6 ms, motor_shield
   34 → 18 ms.
4. **Full resolution during gestures.** A scene predicted over 12 ms is
   raymarched at a reduced scale while a gesture lasts and upscaled with its
   depth, under full-resolution overlays.

A handle drag keeps culling: each box records the parameter slots it was
computed from, and while a drag overrides one of them only those boxes are
opened (`OPEN_BOX_HALF`). Switching all culling off for the drag, the first
version, would have cost more than main (43-48 ms against 55, but 77-108 ms
before the storage change).

## Ruled out or left

- **Redundant redraws.** Rendering was already on demand (`invalidate` /
  `scheduleRender`); the FPS readout says "idle" at rest.
- **Pipeline rebuilds on parameter drags.** None, before or after
  (`e2e/shader.spec.ts`, `e2e/drag.spec.ts`).
- **Compile time.** Not the bottleneck in this harness (under 0.3 s to the
  first frame on every scene, after); the 12-25 s quoted in `setShaders` is
  not reproduced here.
- **Wide blends.** motor_shield's unions blend over k = 4 x smoothness up to
  0.48 m, so the exact tests rarely skip there. Anything faster would need a
  non-exact bound, which would move the surface.
- **Ray-box clipping of primary and shadow rays** against the scene's box
  would save the background's march steps; not done, since it needs a bounds
  function in both backends' viewer contract.
