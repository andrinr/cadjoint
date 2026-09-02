# The cadjoint design language

*The settled specification. This is what the app is, not how it was found — the
exploration that produced it is in git history. Every value here is the value in
`frontend/src/tokens.ts`, which is the source of truth; `frontend/src/styles.css`
mirrors it and `frontend/test/tokens.test.ts` asserts the two cannot drift. If
this document and the code ever disagree, the code is right and this file is a
bug.*

---

## 1. The one sentence

> **Colour is a reading. Anything that is not a reading is drawn without hue.**

cadjoint paints measurements in viridis and magma. The chrome is the sheet those
measurements are printed on: paper, ink, and rules. Every hue on screen therefore
means a number, and the single accent is the one deliberate exception — and it is
never a mark, only a ground.

Everything below is a consequence of that.

---

## 2. What it is answering

`frontend/tools/ui-audit/audit.mjs` measured the playground before this work:

| measured | value |
| --- | --- |
| distinct colours in use | 85 |
| near-black surfaces | 13, adjacent steps **below the just-noticeable difference** |
| the brand accent | one hue at **12 different alphas** |
| distinct type variants | 49 |

None of those is a bug alone. Together they say there was no rule about when a
colour is allowed to appear, so the app answered "make this look distinct" by
inventing a value, eighty-five times. A design language is not a palette; it is
the set of sentences that make the eighty-sixth value obviously wrong.

Two facts about this product make its version of the problem specific:

- **The data already owns colour.** viridis for solved fields, magma for mesh
  quality, four BC hues, proposal cyan. A reader *measures* with those.
- **The users read numbers.** Tolerances, jacobians, aspect ratios, residuals,
  gradients. The typography is not a mood, it is an instrument face.

There is no free hue to give the chrome. Sampling both ramps at 400 stops and
bucketing every sample with chroma ≥ 0.04, **viridis ∪ magma claim 36 of 36 hue
buckets.** Relaxing to the read third of each ramp and excluding the BC hues and
proposal cyan leaves 24° of 360, in two slivers. The old chrome accents proved it
the hard way: every one sat on a ramp to within a third of a degree, and three of
six were inside the ≈10 ΔOKLab at which two colours read as the same colour —
the brand lime 9.8 from viridis(0.90), simulate's amber 4.7 from magma(0.87),
`danger` 1.8 from magma(0.73). A lime selection dot on a solved heat sink was,
measurably, a hot spot.

So the constraint is not "pick a safe hue". It is **spatial**, and it is lintable.

---

## 3. The ground is paper

One value, `#e6e6e9` (L 0.926, C 0.004), and **the chrome and the viewport share
it**. The seam between them measures `dL 0.0000 · contrast 1.00:1`. The viewport
is not a darker well or a lighter plate cut into the app: it is the same sheet,
and what marks its edge is a **rule**, not a step in luminance.

That is the structural idea in one line: **structure comes from rules, not
boxes.** One hairline weight, three contrasts, radius 0, no shadow.

### 3.1 Surfaces

Not a depth ladder — a paper ladder. The page is the ground and anything above it
is a lighter sheet laid on top. Two steps up, because a third would fall below a
JND.

| token | hex | L | job |
| --- | --- | --- | --- |
| `surface-viewport` | `#e6e6e9` | 0.926 | the viewport's ground |
| `surface-base`, `surface-bar` | `#e6e6e9` | 0.926 | the page, and the bars on it |
| `surface-bar-alt` | `#edecee` | 0.944 | sub-bars, and the hover state |
| `surface-float`, `surface-panel`, `surface-raised` | `#f8f7f8` | 0.977 | the dock sheet, panels, floating chrome |

**Hover is a step *down*.** `surface-raised-hover` is `#edecee`: on paper,
pressure darkens the sheet. A light UI that brightens on hover has nowhere to go
once the sheet is already white.

**No inset control.** A field drawn darker than the sheet it sits in reads as
disabled. Controls are separated by their edge, not by a hole.

### 3.2 Ink

Three levels, and nothing else. Text is never faded with `opacity`, because an
opacity fade has no assertable contrast ratio.

| token | hex | on paper |
| --- | --- | --- |
| `ink` | `#18161a` | 14.43:1 |
| `ink-2` | `#48464d` | 7.46:1 |
| `ink-3` | `#605e65` | 5.13:1 |
| `ink-on-accent` | `#0a0a0c` | — (drawn only on `accent`) |

All three clear AA on every surface, including the darkest. The worst measured
text pair anywhere in the shipped screens is 4.74:1 (`ok` at 12px on paper).

### 3.3 Rules

One weight — 1px, always — and three contrasts, because on a sheet the hierarchy
is carried by how dark a line is, not by how thick it is.

| token | hex | on paper | job |
| --- | --- | --- | --- |
| `rule` | `#9f9da5` | 2.15:1 | separation *within* a panel |
| `rule-strong` | `#747278` | 3.81:1 | between sections, control edges, and the viewport seam |
| `rule-heavy` | `#36343b` | 9.85:1 | the viewport frame and its corner marks |

`rule` is deliberately under 3:1 and is absent from `MEANINGFUL_NON_TEXT`: a
resting hairline inside a panel is structure, not state. `rule-strong` is the one
that has to be *found* — it is what says "the viewport starts here" when there is
no luminance step to say it — so it clears the non-text bar on both grounds.

A control's edge is something you must find, so it takes `rule-strong`; a
hairline that divides a panel is structure at 2.15:1. That distinction is what
took control borders from 70 to 0 without a fudge.

**Brutalist structure comes from coverage, not contrast.** The reference grid
measures 2.16:1; a heavy-border variant at 9.85:1 reads worse than the shipped
2.15:1. More rules, not darker ones.

### 3.4 The accent

**One hue, and it has exactly one job: a fill behind near-black type.**

`accent` `#f87318` measures **7.02:1 as a ground** under `ink-on-accent`
`#0a0a0c`, and **2.26:1 as ink** on paper. Those are not two options — one passes
and one fails. Anywhere a dark chrome would have drawn accent-coloured text or an
accent hairline, this design draws a filled block. `accent-press` `#c85d00` is
the pressed and hovered fill and the only tone allowed to draw an
accent-coloured *mark*: 3.36:1 on paper, which the accent itself cannot clear.

`tokens.test.ts` encodes the rule in both directions: ≥7:1 as a ground, <3:1 as
ink. The accent also appears as a low-alpha wash, tint, veil or edge on paper —
six strengths, each meaning something — but the solid block is the one that ever
carries type.

### 3.5 Status and taxonomy

Re-authored for paper: on a light ground a tone is **darkened**, not brightened,
to be read. Four tones, and the kind chips reuse them rather than inventing hues.

| token | hex | on paper |
| --- | --- | --- |
| `danger` | `#a8341c` | ≥4.5:1 |
| `danger-ink` | `#8a1f10` | ≥4.5:1 |
| `info` | `#0065b4` | 4.79:1 |
| `info-ink` | `#004a85` | ≥4.5:1 |
| `ok` | `#00734c` | 4.74:1 |

### 3.6 Inside the viewport rectangle

Everything the DOM draws inside the viewport — dimension labels, the hint bar,
the mode cue on the border — uses the same values chrome does, but owes a
separate rule: **inside the rectangle, nothing is coloured.** The field ramp is
the only hue there, which is what the measurements score as `FIELD WINS`, and an
achromatic annotation can never be mistaken for a value.

`viewport-ink` `#18161a` · `viewport-ink-2` `#605e65` · `viewport-mark`
`#48464d` · `viewport-mode` `#605e65`. The mode cue is one tone for all three
modes; the mode is named in words beside it.

The graticule is furniture, not data, so its three tones sit deliberately below
the 3:1 a meaningful mark owes and are held to a **band of 1.6–2.8:1** in
`test/graticule.test.ts`: above it the grid competes with the field, below it the
grid is invisible. `graticule-line` `#adadb3` · `graticule-axis` `#9c9ca2` ·
`graticule-frame` `#8f8f95`. The grid paints 1.4% of an empty viewport at 1.79:1.

### 3.7 Zoning: where a hue may be a fill

1. **Chrome hue lives outside the viewport rectangle.** Inside it, the only
   fills are the field ramp, the quality ramp, the four BC hues and proposal
   cyan.
2. **The panel is docked, not floating over the field.** This is the same
   decision as the zoning rule, not a separate one. A panel floating inside the
   rectangle must be achromatic to stay clean, which is the old problem
   restated; unzoned, it puts 26 709 px² of chrome hue inside the rectangle at
   95% of viridis' chroma, with the orange measuring 5.4 ΔOKLab from magma(0.73)
   — the same colour to the eye. Dock it outside and the conflict disappears:
   colour goes where the work is, and the viewport stays an instrument.
3. **Area governs how badly, adjacency governs whether.** A 970 px² accent probe
   still misreads when it sits beside a same-hue field reading. Area is
   necessary, not sufficient.
4. **A ramp may appear in a panel only as a legend**, and a legend is a bar
   ≤8px tall with both endpoints labelled. A ramp with no numbers beside it is
   decoration; delete it.
5. **Selection inside the viewport is achromatic and high-contrast**, never the
   accent. It must read over any field value.

---

## 4. Typography

**Mono is the house face.** Every label, tab, menu item, readout and unit in the
chrome is monospaced, because the panel is a view of a program and the furniture
should say so. Sans is kept for prose — hints, warnings, descriptions — which is
the only text here that is sentences.

**The one exception is the strings the program itself wrote.** A study called
`sink-conduction`, a mesh name, a tree label, a pane title: those are values from
`scene.py`, so they are quoted, not restyled. Chrome furniture is tracked
uppercase; a program's own string is neither.

### Tabular figures are non-negotiable

Numbers change while you look at them: a solve streams residuals, an optimize run
ticks an objective, a probe updates as the pointer moves. With proportional
figures `412.80 → 411.79` reflows its own row. Every numeric surface takes
`font-variant-numeric: tabular-nums` plus a monospaced family — belt and braces,
because some fallback stacks resolve to a proportional face.

- **Numbers are right-hung on a shared column**, not left-aligned after their
  label. Magnitude becomes edge alignment and the decimal point becomes a
  vertical rule you can read down.
- **The unit is `ink-3`, one size below the value, on the same baseline.** Never
  in the label — the unit belongs to the number, and when the number is empty the
  unit should be too.
- **The value is a size larger than its own label.** A 9px tracked label above a
  13px value reads as a readout; the same row at one size reads as a form.

### The scale

Six sizes. Below 9px is unreadable; above 15px belongs to the viewport, not the
chrome.

| token | px | job |
| --- | --- | --- |
| `text-3xs` | 9 | tracked small-caps labels |
| `text-2xs` | 10 | dense secondary, selector expressions |
| `text-xs` | 11 | control labels, tabs, hints |
| `text-sm` | 12 | body, the default |
| `text-md` | 13 | **numbers** — every value the user reads |
| `text-lg` | 15 | panel titles, and nothing else |

**Tracking is a function of size, not a house style.** At 9px the counters need
0.16em to stay open; at 15px the same value falls apart into letters. A single
`--tracking-caps` is the tell of a system that has not measured its own labels.

| `tracking-3xs` | `tracking-2xs` | `tracking-xs` | `tracking-sm` | `tracking-md` | `tracking-lg` |
| --- | --- | --- | --- | --- | --- |
| 0.16em | 0.13em | 0.1em | 0.08em | 0.06em | 0.04em |

Weights 400 / 500 / 600 / 700 — 700 is reserved for type on a filled accent
block, where it has a 7.02:1 ground under it. Leadings 1 (single-line controls),
1.25, 1.4 (dense stacks), 1.55 (prose).

---

## 5. Line, surface and shape

- **Hairlines over boxes.** The default way to separate two things is a 1px
  `rule` full-bleed to the container's padding edge. The default way to *group*
  them is a shared left edge and a shared baseline. A filled box is a last resort
  and needs a reason; "it is a card" is not one.
- **One radius, and it is zero.** Kept as a token so the decision has a name and
  one place to change. Everything here is a cell on a ruled sheet, and cells are
  square. Circles exist only where roundness carries "this is a point, not a
  region" — status dots, pipeline nodes.
- **No shadow.** `tokens.test.ts` asserts the stylesheet casts none. Elevation is
  a lighter sheet and an edge, not a cast.
- **One border weight**, 1px. A 2px border is not a stronger border, it is a
  different element: reserve 2px for the *active* indicator, where the doubling
  reads as state.
- **Sections are not numbered.** A first cut stamped `01`/`02` counter blocks on every panel and section head; the user judged them decoration without a purpose, and they are gone. A section is introduced by its kicker chip — the one word that names its kind (`LIBRARY`, `SERVER`, `FEM`) on the accent fill — and its title in tracked mono; order is carried by position, as everywhere else in this language.

Scales are deliberately short: six type sizes, six trackings, seven spacing
steps, one radius, five control heights, three durations. **If a value is not on
a scale it is a bug, not a nuance** — and `tokens.test.ts` fails on a raw hex
outside `:root`, a literal font-size, or a type size off the scale.

---

## 6. Three modes, one accent

Model → Sketch → Simulate is not three parallel worlds. It is **one pipeline**:
geometry, then the constraints that drive it, then the physics that scores it.
Gradients flow back along exactly that chain.

So `MODE_ACCENTS` binds all three modes to the same accent, and the mode is read
from:

- **the position of the filled cell** in the three-cell switcher, the way you
  read a segmented gauge;
- **the word** in the hint bar, which survives greyscale and colour-blindness;
- **the tool set** — rail, panel and shortcuts change wholesale, which is a much
  louder signal than a hue.

The one mode signal inside the viewport is the achromatic cue on the border. This
is strictly more information than three hues carried, at a cost of zero reserved
hues.

---

## 7. Density and rhythm

The target is 500+ elements per screen without the screen feeling loud.

- **Base grid 4px.** Spacing `2 · 4 · 6 · 8 · 12 · 16 · 24` — dense at the
  bottom, where panels live.
- **Control heights `22 · 26 · 30 · 34 · 36`.** Every interactive box picks one,
  which is what keeps labels, inputs and icons on a shared baseline inside a row.
- **Rows share one baseline.** Every panel row is `label | … | value` on a single
  baseline; a row that needs two lines hangs a 10px sub-line under the value,
  right-aligned, `ink-3`, ellipsised.
- **Value columns line up across panels.** Two panels side by side whose value
  columns do not align read as unrelated.
- **Density is bought with type, not with space.** When a panel is too tall, drop
  the label to 9px tracked and hang the value at 13px — do not shave 2px off the
  padding. Padding below 8px stops reading as padding and starts reading as a
  mistake.

---

## 8. Motion

Motion has exactly one job: **to show causality between an edit and its
consequence.** The product's claim is that a change to the source propagates
through geometry, meshing and physics. Movement that does not express that
propagation is noise on an instrument face.

| duration | meaning | example |
| --- | --- | --- |
| `dur-fast` 90ms | *acknowledgement* — the pointer's own feedback | hover tint, press |
| `dur-base` 160ms | *state* — something the user directly caused | tab switch, toggle, panel open |
| `dur-slow` 260ms | *propagation* — something changed **because** something else did | source rewritten → geometry recompiles |

One easing family: `ease` `cubic-bezier(0.2, 0, 0, 1)` for anything arriving,
`ease-inout` for anything that moves and settles in place.

**Causal pairs are sequenced, never simultaneous.** When a panel edit rewrites
the program, the changed source line highlights first, and only then does the
geometry update. Simultaneity teaches that the panel and the viewport are two
views; the stagger teaches that the panel *edits the code* and the code *makes
the geometry*. 60ms buys the product's core idea.

**What must never animate:**

- **Numbers.** A tweened number displays values that were never true. Values
  replace instantly; if a change needs marking, flash the row for 160ms.
- The camera under direct manipulation, ever.
- Anything during a solve except the progress indicator itself.
- Layout on data arrival. Reserve the space; do not grow into it.

`prefers-reduced-motion` collapses 90ms and 160ms to 0 and reduces the 260ms
propagation to a two-frame highlight — causality is information, so it survives
in a form that does not move.

### 8.1 Replay is not animation

> Playing a recorded trajectory back is not tweening: every frame shows a value
> the optimizer actually held at step *k*. The rule forbids inventing
> intermediates between two true values; it does not forbid showing true values
> in the order they occurred.
>
> **The test is mechanical: may this frame be exported as a row of the run log?**
> If yes it is replay. If no it is a tween, and it is forbidden.

### 8.2 The settle law

1. **Needles, bars and rules may settle. Digits may not.** A bar is not spelling
   anything; a digit mid-roll is a wrong number.
2. **Ballistics are VU:** 300 ms to 99% of the final value, overshoot 1.0–1.5%,
   τ ≈ 160 ms to 63%. For the residual head, a dragged tick, legend percentiles.
3. **Overshoot budget = half the displayed precision.** ANSI C39.1's definition
   of *dead-beat*, which makes overshoot computable instead of a taste call: a
   bar under a value shown to two decimals may overshoot 0.005 of its range.
4. **Peak marks are asymmetric (PPM):** attack ≈10 ms so one bad Newton
   iteration is never missed, decay ≈8.6 dB/s so the excursion stays readable.
5. The second-order settle is not a second easing family; it is a physical model
   confined to the marks named in (2) and (4).

---

## 9. Iconography

- **16px box, 1.25px stroke, no fills, miter joins, no rounded caps.** Drawn on
  the half-pixel so a vertical stroke lands on a device pixel; the stroke weight
  matches a hairline in feel, so an icon and a rule look like one pen.
- **Icons are `ink-2`, and colour is not available to them.** An icon that needs
  a colour to be understood needs a label instead.
- **Every rail icon has a text label at ≥1280px wide.** Icon-only is a space
  optimisation, not a design.
- **No icon-only destructive action, at any width.** Delete carries the word.
- Icons never appear inside the viewport rectangle. Gizmos are geometry, drawn by
  the renderer in the data vocabulary.

---

## 10. Character: where the personality lives

Character is found **in time and in precision, not in colour** — which is why no
rule above is bent for it. Every device here either ships or is queued behind a
named prerequisite.

### 10.1 Shipped

**The floor grid, the spacing readout and the title block.** The viewport rules
the **z = 0 plane** — the floor every scene stands on, since the library and every
scene are Z-up — with a minor line, a firmer major every fifth, and the two axes
a step above that: **1.36 / 1.58 / 1.69:1** on paper, fading outward from the
orbit target and again once a cell falls under a few pixels, so the far field
dissolves rather than aliasing. It is a per-fragment raycast in a fullscreen WGSL
pass at clip z = 1, depth-tested against the ray-miss depth: it sits behind the
solid, under every overlay, and costs one triangle. In Sketch mode, when the
active plane is not the floor, the whole grid steps back one level — the sketch's
own plane becomes the reference, but the floor still says which way up the world
is. There is no screen-space faceplate, no centre-axis ticks and no corner
brackets: a grid that does not live in the scene says nothing about where anything
is.

The readout states the grid spacing in real millimetres — the unit is not
invented; the STEP exporter is the one place the repo declares a length and it
stamps metres — always on a **1-2-5 rung**, and prefixed `>` whenever an on-screen
measurement is not to scale (perspective). Alt-wheel steps the rungs. The title
block sits bottom-right per ASME Y14.100: SCENE / STUDY / MESH / SOLVER / REV,
`—` where a field is empty. An empty or uncompiled viewport clears to paper and
rules the floor; it never renders black.

**The light viewport, and everything that assumed a black ground.** The
environment radiance is flat, because under the default orthographic camera every
primary ray shares one direction and a directional dome collapses to a single
colour that changed as you orbited. The FEM surface shades with a signed Lambert
term plus a facing-ratio contour, not `abs(dot(n, l))`, which folds the normal
sphere so every facet emits — correct when geometry is the only bright thing on
screen, wrong when the ground is brighter than the part. All ten overlay tones,
three gizmo axes, four BC hues and the constraint annotations are re-authored
dark-on-paper with measured contrast.

### 10.2 Queued, in order of delight per unit of cost

1. **The flag, and the hatch plate.** An aircraft OFF flag is spring-loaded *in*;
   only a good signal holds it out, so a dead instrument cannot hide its own
   flag. A stale or unavailable value is replaced **in its own slot** by an
   opaque 45° hatch at 2.5px stripes with a 1px `rule-strong` border, exactly the
   width the value column occupies. **Never grey the value; never remove the
   row.** An unmanufacturable interval on a parameter rule is hatched, not
   shortened — a barber pole says *you may not go here*, a missing region says
   nothing (AC 23.1311-1C §17.7(b)). When an auto-coupled input is pinned by
   hand, the result region carries an `UNCAL` annunciator and the viewport reads
   `MEAS UNCAL` until it is re-coupled. Cost: effectively none. Ship this even if
   nothing else does — it is a correctness feature wearing a character costume.
2. **First run and the empty states.** An empty scope is a ruled screen waiting
   for a trace, not a blank one: graticule drawn, readout present and prefixed
   `>`, title block present with `—` in every row it cannot fill, the legend a
   hatch with `—` endpoints. Empty copy has a form — name the missing
   precondition **in the language of the program** (anything from the source is
   mono), then exactly one clause of dry aside, then the action as a 9px tracked
   mono line. **The aside must be true and must teach something;** delete any
   sentence that is only a joke. The `cj` mark exercises its glyph set once per
   session (seven glyphs, 80 ms each, 560 ms total) — a nixie anti-poisoning
   routine, which is maintenance rather than ornament, and that is the only way
   ornament survives a system this strict. Never on tab or mode switches, and not
   at all under reduced motion.
3. **The trajectory tape.** The optimizer is the product's whole argument and it
   is currently a sparkline. Make a run a scrubbable tape: the geometry morphs,
   the objective curve draws itself, each parameter walks its own machinist's
   rule, and the source literals rewrite as it goes. 12 steps/s, total held to
   ≈3.4 s by dropping frames, never above 25 steps/s. The strip is **detented**
   — dragging snaps to the nearest recorded step, so it never lands between two
   real states. A parameter's track is a machinist's rule spanning its declared
   bounds: 20 minor divisions, every fifth major, no thumb, no fill, no track
   colour. Travel is shown by **ghosting, not blur** — the last five envelopes as
   thin achromatic outlines, drawn after the field, present only while moving.
   The objective is the one place editorial contrast is earned: a 30px mono
   number is licensed **only** on the objective of a live or replayed
   optimization. Prerequisite: the trajectory must be recorded and kept.
4. **The fill.** A 30-second solve is currently a spinner. Replace it with five
   annunciator cells that light in order — `COMPILE MESH ASSEMBLE SOLVE ADJOINT`,
   always allocated, never reflowing (the HP annunciator strip, whose whole
   virtue is that the layout does not move) — a mesh that builds bottom-up over
   the real duration of meshing, and a field that fills outward from its own
   boundary conditions, which is a *reading*: an iterative solve genuinely
   converges outward from its BCs. The residual is a nine-decade rule showing the
   solver's real iterates, replaced instantly. **No percentage of an unknowable
   whole.** RUN becomes STOP and drops the accent — a running job is not the
   identity. Prerequisite: stage events and a residual stream from the pipeline.
5. **The gradient gutter mark, and only the mark.** A 2px `info` rule in the
   editor gutter on lines the adjoint tape touches — no background tint, and only
   on lines carrying a *free* parameter, never the whole declaration. **At most
   fourteen marked lines:** ten marked lines is information, twenty is wallpaper.
   The animated adjoint wavefront was built and deferred; see §12.

### 10.3 The one break

> **One element per view may break the system, and it is always either the thing
> you act on or the thing that is moving right now.** Since there is no spare hue
> to spend, the break is spent in **luminance, length, plane or shape** — one of
> those four, once per view. The 30px objective is one; the breathing RUN key is
> one; the hatch plate is one. If a view has two breaks, one of them is wrong.

---

## 11. What this language refuses

1. **No hue in chrome that is not the accent.** If it is not a measurement and
   not the one accent block, it is achromatic.
2. **The accent is never ink.** 7.02:1 as a ground, 2.26:1 as ink: a filled block
   with near-black type, or nothing.
3. **No colour inside the viewport rectangle that is not data.** Selection and
   hover are achromatic; a chrome accent never enters the field.
4. **No opacity-faded text.** There are three ink levels; use one.
5. **No new surface.** Two steps above paper. If a proposed surface is under
   3 ΔOKLab from an existing one, it *is* that surface.
6. **No decorative gradient.** A gradient in cadjoint is a colormap, and a
   colormap without a labelled scale beside it is a lie.
7. **No text, and no colormap, on a backdrop that is a function of the data.** A
   surface carrying glyphs or a legend is opaque; its contrast and its surround
   must be constants the linter can assert. This is not a taste position about
   glass — see §12.6.
8. **No shadow, no glow, no neon.** Elevation is a lighter sheet and a 1px edge.
9. **No rounded corners.** One radius and it is zero; circles only where
   roundness means "a point, not a region".
10. **No animated numbers, no skeleton shimmer, no spinner over content that is
    already legible.**
11. **No indeterminate progress.** A run reports the stage it is in, in words,
    and a quantity that is genuinely known.
12. **No motion that changes a colour that is a reading.** Trails, ghosts and
    fronts are achromatic or geometry-clipped; a field pixel is either its true
    value or it is not painted.
13. **No second weight of hairline**, and no border used to say "this is
    interactive". Interactivity is said by height, baseline and hover.
14. **No icon without a word** for anything destructive, and no icon-only tool
    rail at desktop widths.
15. **No inset control.** A control is a plane, never a hole; a hole is what
    "disabled" looks like.
16. **No token whose name is a colour.** A repaint must be one file's worth of
    change.
17. **No value off the scales.** Six sizes, seven spacings, five control heights,
    one radius, three durations.

Two exclusions the user set, which are rules here and not preferences:

18. **No cropped display headline.** The reference's oversized headline running
    off the page edge is out. Identity rests on the `cj` block and the filled
    mode cell instead, and costs what it costs.
19. **No decorative accent squares.** Orange appears only where it marks
    something real — never pinned to grid nodes as ornament. The editor pane is
    framed by its rules alone.

---

## 12. Decisions, and the numbers behind them

The renders that produced these are in git history; the numbers are kept here so
the reasoning is recoverable without them.

### 12.1 Why paper, and what it costs

The field has a floor and a ceiling: **viridis runs L 0.290 (`#460155`) to L
0.914 (`#fae720`)**. A dark ground has to stay under the floor; a light ground
has to stay over the ceiling. Paper at **L 0.926 is brighter than 100% of
viridis** — but only by **0.012 L of headroom**.

Light is not a clean inversion of the dark constraint:

- **Text separates from the field 2.7× better on paper** — 0.085 L below the
  floor, against 0.031 L above the ceiling on dark.
- **Surface headroom collapses 8.7×.** On paper the chrome is separated from the
  data by **chroma alone** (paper is 19.2 ΔOKLab from its nearest viridis
  sample), not by lightness.

That trade is why the ink discipline in §3.2 and the zoning rule in §3.7 carry
more weight here than they would on a dark ground.

### 12.2 The seam is a rule, not a step

With chrome and viewport both at `#e6e6e9` the boundary measures `dL 0.0000 ·
contrast 1.00:1 · dE(OKLab) 0.00`, and the rule drawn on it sits at L 0.726 —
1.96:1 against both sides. A luminance step at the seam (viewport at `#f8f7f8`)
was built and measured at 1.17:1: it buys a boundary nobody needed and costs the
hot decile 1.37:1 instead of the shared-ground reading. The seam is drawn.

### 12.3 The plate as an object — measured, not chosen

With both chrome and viewport light, does the field stay the subject? In two
halves:

- **Solved: yes, and by more than on dark.** Geometry separates from paper at
  `dL 0.446` against `0.332` from a dark void, because paper is farther from
  viridis' body than black is.
- **Unsolved: no.** Grey geometry on paper measures `dL 0.099` with **51.9% of
  the part within a tenth of the ground**, and viewport chroma collapses to 0.01%
  of pixels while the orange chrome holds 2.74% — the accent becomes the most
  saturated thing on screen and the model the least. The shipped light viewport
  measures `dL 0.076` today; the old dark chrome was hiding it.
- **The hot end costs too.** With 0.012 L of headroom, 7.8% of a solved field
  sits within `dL 0.10` of its own ground and the hottest decile measures
  **1.20:1** — the yellow-green underside dissolves into the page.

The answer measured is **a dark plate set into the light page as an object, not a
well cut out of it**: it keeps the light system everywhere, restores the hot
decile to **5.54:1**, gives an unambiguous boundary (`dL 0.8146`, 16.42:1), lifts
the field's share of screen chroma to **11.35%**, and fixes the unsolved case for
free. `frontend/src/tokens.ts` still ships the shared-paper viewport, so this is
a recorded alternative, not a plan: the user chose the shared-paper viewport after seeing it live, and the plate stays on file as the measured answer should unsolved geometry on paper ever need the lightness step back.

### 12.4 Why the accent is orange, and only a fill

Six candidate accents were measured for type contrast on the block and distance
from both ramps. `#f87318` takes near-black type at 7.02:1 and sits 26.6 ΔOKLab
from viridis — the largest ramp distance of any candidate that also carries type.
Lime `#d9ff57` is 6.3 from viridis and 17.92:1 under black: legible, but it is a
*hot field sample*, which is the whole reason the old palette failed. Yellow is
4.4 from viridis, worse. Blue `#1f3fe0` and magenta `#e5007d` need white type,
which puts a second ink on the block.

About thirty sites were drawing the accent as ink. Converting every one to a
filled block with near-black type is the change that made the direction legible
rather than merely light.

### 12.5 Chrome temperature is a rounding error

At the chroma this language permits chrome (C ≤ 0.014), a **221° hue rotation**
moves every surface by less than or barely one JND — 0.73 to 2.25 ΔOKLab across
the whole ladder. Temperature only becomes visible at C ≥ 0.03, which is exactly
where a chrome hue starts competing with a reading. Anyone asking for warmer
greys is asking for a chroma this language has already declined.

### 12.6 Glass, with the receipt

Measured over a worst-case backdrop, a tuned translucent panel drops `ink-3` to
**3.42:1** and a naive one drops `ink-2` to 4.34:1 and `ink-3` to 2.68:1 — it
fails on the *third* ink level, which is exactly the level that carries labels
and units. The deeper objection is the spread, not the mean: background σ under a
column of glyphs goes from 0.016–0.020 to 0.048–0.077, so a translucent panel
does not have a contrast ratio, it has a *distribution that moves when the camera
orbits*.

The argument that settles it cannot be answered by turning a knob: **the panel
contains a colormap legend.** Measured ΔE76 of the panel ground as the field
behind it sweeps viridis(0) → viridis(1) is **36.4 at α 0.82, 18.9 at 0.91, 8.3
at 0.96, and 0.0 only at 1.00.** A viridis chip judged against an olive-shifted
ground is a correctness defect in an instrument. Below α ≈ 0.5, `ink-3` passes
through zero contrast and re-emerges as dark-on-light as the field heats — a
label that inverts polarity mid-solve.

On cost: **translucency itself is free; only the blur is not.** Isolated on an
M-series machine, α 0.85 without blur measures −0.02 ms/frame against opaque and
with `blur(20px)` measures **+0.49 ms** — ≈2.9% of a 60 fps budget. It scales
with the *canvas*, not the panel (+0.055 ms at 1.43 MPix, +0.627 at 12.86),
because a pixel-moving filter unions the panel's whole rect into frame damage
(`cc/trees/damage_tracker.cc`), and the viewport damages itself every frame of a
solve. Excalidraw removed `backdrop-filter` from panels over a live canvas for
this reason (#3505 / #3506); Linear declined Liquid Glass on legibility grounds;
Figma reverted UI3's floating panels because the canvas peeking around them was
distracting.

The need this was serving — *what is under the panel?* — is answered by a
**peek**: hold a key and the panel drops to 15% with its text hidden entirely,
for as long as it is held.

### 12.7 Rejected, with the measurement

- **Motion blur on morphing geometry.** Built and measured against the true
  frame: mean ΔE76 6.2, p95 11.0, max 21.9; **97% of on-part pixels above JND**,
  and it repaints **54% of the part's area outside the true silhouette**. The
  smear averages five viridis samples into one pixel, so the band is a
  temperature no node ever had. Rejected on correctness, not taste — ghosting
  gives the same sense of travel with zero colour error.
- **Odometer / split-flap digits.** Every frame of the roll shows a value that is
  partly one number and partly another, on a screen where people read tolerances.
  The ballistics belong on a needle or a bar, where nothing is being spelled.
- **The adjoint wavefront.** The mechanism is right and the ordering genuinely
  reads backward, but at 1× it is twenty-odd small flashes in sequence and lands
  closer to "busy" than to "wave". The permanent gutter mark carries most of the
  meaning for none of the motion. Revisit only once the marked-line set is proven
  to stay under fourteen.
- **Gradient magnitude as bars in the panel.** Five ramp-coloured bars above the
  field legend read as progress bars and compete with the legend. Draw the
  *direction* achromatically; let the magnitude be a signed number in mono.
- **0 / 10 / 90 / 100% rise-time markers** on the graticule. On a scope they are
  a normalisation ritual for a waveform; there is no analogue for a 3D field on a
  spatial axis, so they are four labelled ticks that mean nothing. Recorded here
  so nobody adds them back for the look of the thing.
- **One very large number per panel.** A 40px hero metric is a dashboard gesture.
  The one licensed exception is §10.2's optimizer objective.
- **Border-led hierarchy.** A resting border at 3:1 turns every panel into a
  drawn box. Borders move with elevation as an idea; here it is spent on the
  hairline ladder.

### 12.8 What was looked at

Tektronix 2213 (graticule divisions and subdivisions; the detent), 475A (minor
ticks on the centre axes only) and 2465 (`>` for an uncalibrated scale factor);
HP 5245L, 3478A and 8566B for annunciators and `MEAS UNCAL`; FAA-H-8083-15B and
AC 25-11B §6.2.1.7 for OFF flags in the location of the information they replace;
AC 23.1311-1C §17.7(b) for the 45° barber pole; the Garmin G1000 trend vector,
which is *absent* when the value is steady; ASME Y14.5-2018 §4.3.2 (decimal
places *are* the tolerance) and Y14.100-2017 (the title block); IEC 60268-17 /
ITU-R BS.2054-4 for VU ballistics and IEC 60268-10 for PPM asymmetry; ANSI
C39.1-1981 for dead-beat; nixie cathode-poisoning prevention routines; Braun's
ET 66, whose single yellow key licenses the one-break rule; Radix, Geist and
Linear for elevation-as-a-scale; Grether (1949) on the three-pointer altimeter,
the standing argument against any compact multi-ring dial.

Three claims that earlier drafts made and the sources do **not** support, dropped
here so they are not reintroduced: there is no published HP standard assigning
colours to functional groups; no primary HP source names Bauhaus, Braun or Rams
as an influence; and there is no "dot at 12" as a named Braun device — the
documented differentiator on Lubs' dials is tick *length*.

---

## 13. Making it stick

The language is only real if something fails on it.

- **`frontend/test/tokens.test.ts`** asserts that `styles.css` declares every
  token in `tokens.ts` with the same value; that no raw hex, literal font-size or
  off-scale type size escapes the token layer; that there is one radius and it is
  zero; that the stylesheet casts no shadow; that tracking is a function of size;
  that every text tone clears AA on every surface and every meaningful non-text
  tone clears 3:1; that the accent clears 7:1 as a ground and fails 3:1 as ink;
  that nothing drawn inside the viewport rectangle carries chroma; and that
  chrome and viewport sit on the same sheet.
- **`frontend/test/graticule.test.ts`** holds the floor-grid tones inside the
  1.6–2.8:1 furniture band — a band, not a floor.
- **`frontend/tools/ui-audit/audit.mjs`** drives a real Chromium through every
  mode, tab and popover at several widths and measures the result from computed
  style and layout. It reports; it never edits. The light chrome pass took it
  from 251 findings to 119 to **39**, with control borders 70 → 0 and 71
  `border-radius` declarations deleted.

Checks worth adding as the queued devices land:

- `field-floor` / `field-ceiling` — any chrome fill over 4 000 px² that crosses
  the ramp's bound on whichever side the ground is on.
- `ramp-adjacency` — any element painted with a ramp sample that is not inside a
  legend with two labelled endpoints.
- `numeric-face` — any element whose text matches `^[-+]?[0-9]` whose resolved
  family is not monospaced, or which lacks `tabular-nums`.
- `replayable` — any element whose text changes on more than two consecutive
  animation frames must be able to name the recorded step its value came from. A
  number that cannot is a tween.

---

## 14. Files

| what | where |
| --- | --- |
| the language | `research/design-language.md` (this file) |
| the tokens, and the truth | `frontend/src/tokens.ts` |
| the mirror | `frontend/src/styles.css` |
| the assertions | `frontend/test/tokens.test.ts`, `frontend/test/graticule.test.ts` |
| the linter | `frontend/tools/ui-audit/` |
| the chosen direction, as one panel | `research/design/brutalist/5-light-dock.png` |
| the same at 511 elements | `research/design/brutalist/8-light-dense.png` |
| its measurements | `research/design/brutalist/measurements.txt` |
| the system across the whole UI, with the plate as an object | `research/design/combined/6-dark-plate-object.png` |
| the all-light measurements | `research/design/combined/measurements.txt`, `measurements.json` |
| the light chrome as shipped, every mode and popover | `research/design/light-chrome/` |
| the light viewport, twelve states | `research/design/light-viewport/` |
| the graticule, readout and title block in the app | `research/design/graticule/` |
| the docking and window states | `research/design/windows/` |
| the banner and the live README capture | `research/design/banner/` |
| the refactor, before and after | `research/refactor/` |
