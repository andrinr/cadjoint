# ui-audit — a design linter for the playground UI

Eyeballing a dense tool UI misses the things that make it feel unfinished: a
label two pixels off its neighbour's baseline, a panel that quietly parks a
control below its own scroll line, a tenth font size that nobody meant to add.
`ui-audit` drives a real chromium through every mode, tab and popover of the
playground at several viewport sizes and measures those from computed style and
layout — so it reads the built CSS the way the browser resolves it, cascade and
custom properties included.

It reports; it never edits. The output is a JSON file for machines, a Markdown
summary for humans, and one screenshot per state.

## Running it

Start a playground and point the linter at it:

```bash
# terminal 1 — the server (stdlib only; serves cadjoint/viewer/static)
uv run python -m cadjoint.viewer.playground --port 8765

# terminal 2 — the audit
node frontend/tools/ui-audit/audit.mjs --port 8765 --out /tmp/ui-audit
```

`cadjoint/viewer/static` is the **committed build**, so it reflects
`frontend/src` only after `cd frontend && npm run build`. To audit the working
tree without rebuilding, run the Vite dev server against the same API and point
the linter there:

```bash
cd frontend && npm run dev                      # :5173, proxies the API
node frontend/tools/ui-audit/audit.mjs --port 5173 --out /tmp/ui-audit
```

Playwright comes from `frontend/node_modules`; no separate install is needed.

### Options

| flag | meaning |
| --- | --- |
| `--port N` | playground port (default 8765) |
| `--url URL` | full base URL, overrides `--port` |
| `--out DIR` | output directory (default `./ui-audit-out`) |
| `--states a,b` | subset of the state ids below |
| `--viewports WxH,…` | default `1440x900,1280x800` |
| `--settle MS` | wait after each state change (default 700) |
| `--solve` | solve a study and inspect a mesh first, so Results/Meshes are audited with real content (adds ~30s) |
| `--include-editor` | also run the layout checks inside the CodeMirror editor |
| `--headed` | show the browser |

A default run takes about 30 seconds; `--solve` about 60.

### States walked

`model`, `model-menu-file`, `model-tray`, `model-floating`,
`model-render-popover`, `sketch`, `sketch-solver`, `simulate-meshes`,
`simulate-studies`, `simulate-optimize`, `simulate-results` — each at every
viewport. The sketch states select a profile from the object tree first,
because the sketch panel only exists while one is selected. Since every panel
became a window, the object tree and the material browser are part of the
default desk rather than states of their own; `model-tray` parks both in the
tray and `model-floating` lifts one out of the grid, which are the two
arrangements `model` does not already show. A state that rearranged the dock
is undone through the Window menu's Reset layout. With `--solve` the simulate
states are visited first: leaving Simulate mode discards the solved result.

A state that cannot be reached is recorded under `skippedStates` rather than
failing the run.

### Output

```
<out>/ui-audit.json   full report: censuses, violations, per-instance detail
<out>/summary.md      the same thing readable, sorted by severity
<out>/shots/<state>@<w>x<h>.png
```

Violations are grouped into one entry per class, each with a total hit count, a
distinct-element count, the states and viewports it appears in, and up to three
example selectors (the JSON keeps up to 60 instances per class). Selectors are
short paths of at most three levels, cut at the first `data-testid`.

## What it checks

### 1. Type scale census

Every distinct `font-size` / `font-weight` / `line-height` / family in use on
an element that owns text, with a count and examples. If any CSS custom
property whose name mentions font/text/type/size/scale/leading/weight exists,
it is reported as `declaredTypeScale` for comparison; the stylesheet currently
declares none, so the census *is* the evidence from which a scale should be
chosen. Sort by count: the long tail of one- and two-use variants is the
cleanup list.

### 2. Overflow

- `overflow-x` / `overflow-y-clipped` — `scrollWidth > clientWidth + 1` (or the
  vertical equivalent) with no `auto`/`scroll` overflow and no
  `text-overflow: ellipsis`. Overflow caused only by an absolutely positioned
  descendant (a dropdown wider than its anchor) is excluded.
- `flex-child-missing-min-width` — a flex child of an overflowing row that
  still has `min-width: auto`, so it refuses to shrink. This is the usual
  cause of a row that blows out instead of truncating.
- `text-clipped` — text whose measured rects are cut by the first ancestor
  that clips that axis (`overflow: hidden`/`clip`, or the viewport via
  `body`). A scrollable ancestor ends the walk: the text is reachable.
- `scroll-content-hidden` — a legal scroll container that nonetheless parks a
  fifth or more of its content, or an entire interactive control, out of sight
  at rest. This is what users mean when they say a panel "overflows".
- `text-truncated-ellipsis` — informational: ellipsis truncation that is
  actually active, with the full string, so you can decide whether the box or
  the content is wrong.

### 3. Alignment

- `row-baselines-misaligned` — for each non-wrapping flex row, the first text
  line of each child, compared as `rect.bottom − 0.21 × font-size` (Chrome's
  text rects sit on the font box, so that approximates the baseline).
  Threshold 1px for a row whose children share a font size, 2px otherwise.
  Text inside an out-of-flow descendant is skipped: an open popover is not its
  anchor's first line, the same exclusion the overflow checks make.
  Rows that mix font sizes under `align-items: center` are reported at low
  severity: their baselines are *supposed* to differ, and the finding only
  matters if the row reads badly.
- `row-centres-misaligned` — vertical centres of the children of a flex row
  that is not stretching them, differing by more than 1px.
- `panel-left-edges-inconsistent` — sibling blocks stacked down a container
  (three or more) whose left edges do not agree within 1px, minus the ones
  that are visibly centred.
- `control-height-near-miss` — two controls of the same tag *and the same font
  size* whose heights differ by 0.5–3px. Same content, different padding: the
  near-miss that reads as sloppy where a 10px difference reads as deliberate.

### 4. Geometry regressions

- `offscreen-x` / `offscreen-y` — an interactive element past the viewport
  edge with no scrollable ancestor along that axis, i.e. unreachable.
- `interactive-overlap` — two interactive hit boxes intersecting by more than
  1px in both dimensions, neither containing the other. Pairs where either
  element sits inside an absolutely positioned box are marked `layered: true`
  and dropped to low severity: a popover covering the chrome behind it is
  intentional, two in-flow controls colliding is not.

### 5. Colour census and contrast

Every distinct computed text colour, background colour and border colour, with
counts and examples, so values outside the token set stand out. Contrast is
computed against the background actually painted behind the text: the ancestor
chain is composited, alpha included, stopping at the first opaque layer.
`contrast-text` flags below 4.5:1 (3:1 for large text — ≥24px, or ≥18.66px
bold); `contrast-control-border` flags a control border below 3:1 against its
surround.

## Known blind spots

- **Static states only.** Nothing is hovered, focused, dragged or opened
  mid-animation; `:hover`/`:focus-visible` styling and transition states are
  never measured. Dialogs behind a flow (Save As, the shortcut sheet, the BC
  builder, the trajectory player after an optimize run) are not visited.
- **Content is whatever the starter scene produces.** Long object names,
  many materials, a study with a dozen BCs, error notices — the layouts most
  likely to break are the ones this run never renders. `--solve` covers the
  solved Results and inspected Meshes tabs; the optimize trajectory player
  still is not covered.
- **Anything drawn on the canvas is invisible to it.** Overlay labels,
  gizmos, the field legend painted in WebGPU, and text over a canvas or a
  gradient are skipped for contrast (`painted: false`), because there is no
  DOM colour to composite against.
- **Visually hidden elements are skipped entirely** — a box clipped to
  nothing (`clip-path: inset(50%)`, `clip: rect(0 …)`, or a 1px box with
  `overflow: hidden`) is the sr-only pattern, and its text is *meant* to be
  visually unreachable. Measuring it as clipped or overflowing is a false
  positive; the dock library's `aria-live` announcer alone produced 42 of them.
- **CodeMirror's internals are exempt from the layout checks** (`--include-
  editor` re-enables them): its scroller, gutters and lines clip by design and
  otherwise drown everything else. Its colours and type *are* censused, since
  the syntax theme is defined in `EditorPane.tsx`.
- **The view cube (`.cube-stage`) is exempt** for the same reason: CSS 3D
  transforms make its bounding boxes meaningless in 2D.
- **The baseline model is an approximation** (a fixed 0.21 descent ratio), so
  rows mixing font families or very different sizes can produce soft findings.
  Trust the same-font-size rows first.
- **Contrast is WCAG, not APCA**, and it judges text against the composited
  background only — not against a busy image behind it or against a sibling
  that overlaps it.
- **No cross-run baseline.** The tool reports the current state; it does not
  diff against a stored snapshot, so it cannot by itself fail CI on a
  regression. Diff two `ui-audit.json` files if you want that.
- **Two viewports, one device scale.** Nothing below 1280px wide is checked,
  and no `deviceScaleFactor: 2` run, so hairline and sub-pixel rendering
  differences on a retina display are not represented.
