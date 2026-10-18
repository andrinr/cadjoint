# Viewport frame time (2026-10-18)

Run a playground, then the benchmark against it (from `frontend/`):

```bash
python -m cadjoint.viewer.playground --port 6937
```

```bash
node tools/viewport-bench.mjs --port 6937 --out /tmp/bench --frames 60 --dpr 2
```

Other sessions share the GPU, so alternate runs of main and the branch and
take medians. Times in ms, median of 60 frames, main → branch, Apple M-series
(Metal, Chromium).

### DPR 1 (660 x 778)

| scene          | at rest     | camera gesture | handle drag | cold compile to first frame |
| -------------- | ----------- | -------------- | ----------- | --------------------------- |
| starter        | 1.3 → 1.4   | 1.2 → 1.0      | 1.2 → 1.0   | 13 → 15                     |
| end_cap        | 10.0 → 3.2  | 9.6 → 3.2      | 9.0 → 3.2   | 40 → 21                     |
| cylinder_block | 55.6 → 6.2  | 54.5 → 6.0     | 55.2 → 7.1  | 172 → 38                    |
| motor_shield   | 53.8 → 17.6 | 55.3 → 8.5     | 54.8 → 14.3 | 187 → 69                    |

### DPR 2 (1320 x 1556)

| scene          | at rest         | camera gesture  | handle drag     | cold compile to first frame |
| -------------- | --------------- | --------------- | --------------- | --------------------------- |
| starter        | 3.4 → 3.0       | 3.4 → 3.1       | 3.4 → 3.0       | 15 → 20                     |
| end_cap        | 33-41 → 9-10    | 33-41 → 9-11    | 33-41 → 10-12   | 99-117 → 35-42              |
| cylinder_block | 163-174 → 20-22 | 161-189 → 10-14 | 168-188 → 13-14 | 503-593 → 67-76             |
| motor_shield   | 193 → 56-60     | 168-196 → 13-17 | 179-200 → 14-18 | 566-644 → 167-280           |
