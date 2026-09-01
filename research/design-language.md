# A design language for cadjoint

*Proposal. Nothing here is implemented; `frontend/src` is owned by the layout/contrast
work landing alongside it. This document is the argument and the numbers, and
`research/design/` holds four rendered images to judge it by.*

---

## 1. The one sentence

> **Colour is a reading. Everything that is not a reading is drawn in the one
> colour the two ramps agree on — the darkness below their range.**

Call it **Zero**, after `t = 0`. cadjoint paints measurements in viridis and magma;
the chrome is what those ramps look like *before the data starts*. The interface is
literally the bottom of the colormap, and every hue on screen therefore means a
number.

That is the whole language. Everything below is a consequence of it.

---

## 2. What it is answering

`frontend/tools/ui-audit/audit.mjs` measured the current playground:

| measured | value |
| --- | --- |
| distinct colours in use | 85 |
| near-black surfaces | 13, of which adjacent steps are **below the just-noticeable difference** |
| the brand accent | one hue at **12 different alphas** |
| distinct type variants | 49 |

None of those numbers are bugs on their own. Together they say there is no rule
about when a colour is allowed to appear, so the app answers "make this look
distinct" by inventing a value, and it has invented eighty-five of them. A design
language is not a palette; it is the set of sentences that make the eighty-sixth
value obviously wrong.

Two things make cadjoint's version of that problem specific, and both push in the
same direction:

- **The data already owns colour.** viridis for solved fields, magma for mesh
  quality, four BC hues, proposal cyan. Those are not decoration; a reader
  *measures* with them.
- **The users read numbers.** Tolerances, jacobians, aspect ratios, residuals,
  gradients. The typography is not a mood, it is an instrument face.

---

## 3. Colour

### 3.1 The hypothesis, tested

> *Hypothesis under test: derive the chrome palette from desaturated, darkened
> extractions of viridis and magma, so chrome and data are provably one family and
> the ramps stay reserved for meaning.*

I built it and measured it. See `research/design/palette-from-colormaps.png` for the
same thing as a picture.

**Claim 1 — "the ramps stay reserved, so pick an accent hue outside them."
FALSE, and not marginally.**

Sampling both ramps at 400 stops and bucketing every sample with chroma ≥ 0.04:

```
hue buckets claimed by viridis ∪ magma:  36 of 36
hue buckets left free:                   none
```

Between them the two colormaps traverse the entire hue circle. There is no safe
accent hue. Relaxing the test to only the *read* third of each ramp (t ≥ 0.66 —
where the hot end and the good-quality end live, which is where a reader's eye
goes) frees 201°; then excluding ±25° around the four BC hues and the proposal
cyan leaves **24° of 360**, in two slivers at 184–185° and 331–354°. The most
isolated hue in the entire gamut is 331° — `#f895ed`, a magenta that is 26° from
the violet *fixed* BC. Technically legal, wrong for the product.

**Claim 2 — "the current accents are clear of the ramps." FALSE.** Every chrome
accent in `tokens.ts` today sits on a ramp, to within a third of a degree:

| token | hex | h | nearest ramp sample | Δh | ΔOKLab |
| --- | --- | --- | --- | --- | --- |
| `accent-model` | `#d9ff57` | 121° | viridis(0.90) `#bade29` | 0.0° | **9.8** |
| `accent-sketch` | `#7fd6f5` | 223° | viridis(0.39) `#29768e` | 0.1° | 30.4 |
| `accent-simulate` | `#ffb25c` | 67° | magma(0.87) `#fec285` | 0.2° | **4.7** |
| `danger` | `#ff8167` | 33° | magma(0.73) `#fa7a60` | 0.2° | **1.8** |
| `ok` | `#9fe7bd` | 158° | viridis(0.67) `#36b77a` | 0.1° | 18.2 |
| `info` | `#9adcf4` | 223° | viridis(0.39) `#29768e` | 0.3° | 33.0 |

Below ΔOKLab ≈ 10 two colours read as the same colour at chip size. Three of the six
are inside that threshold against the ramp sample they share a hue with. **The brand
lime is 9.8 from viridis(0.90), and 6.6 from viridis(1.00) — "hot" in a temperature
field.** A lime selection dot on a solved heat sink is, measurably, a hot spot.
Simulate's amber is **4.7** from magma(0.87), the colour of a well-shaped element,
and `danger` is **1.8** from magma(0.73) — the same colour, to the eye.

**Claim 3 — "desaturated ramp extractions give a chrome family whose provenance
is visible." FALSE at surface luminance.** Five surfaces built from five viridis
hues 216° apart, all at L 0.235 / C 0.012:

```
viridis(0.00) h 319° -> #211c22
viridis(0.25) h 267° -> #1b1e24
viridis(0.50) h 191° -> #18201f
viridis(0.75) h 145° -> #1b201b
viridis(1.00) h 103° -> #1f1e18
largest pairwise difference: 2.40 ΔOKLab   (JND at this size ≈ 2)
```

Hues nearly two thirds of the wheel apart land within one JND of each other. It is
not mud — mud would at least be visible. It is *nothing*. A palette whose organising
idea cannot be seen is not an organising idea.

**Claim 4 — "one hue, taken from where the ramps agree." TRUE, and worth
keeping.** viridis(0.00) sits at h 319°; magma(0.05) at h 280°. They are 39°
apart and both are violet-black: the two ramps agree about the colour of *nothing*.
Their midpoint, **h 299°**, is the chrome hue. Today's surfaces are tinted at
h 165° — viridis' **middle**, the hue of a lukewarm region, which is exactly the
wrong end of the ramp to borrow from.

### 3.2 Verdict

The hypothesis **half holds, and the half that fails is the important half.**

- Reject: deriving *a palette* (plural hues) from the ramps. Invisible, and it
  buys a provenance story nobody can perceive.
- Reject: the premise that reserving the ramps is achievable by choosing hues.
  There is no free hue to choose.
- Accept: deriving **one** chrome hue from the ramps' shared zero, and treating
  the whole neutral scale as that ramp continued below its data range.
- And then take the real consequence: **chrome carries no hue at all.** The tint
  at C ≤ 0.016 is in the hex values and below the threshold of hue perception, which
  is the point — it makes the greys belong without making them colours.

### 3.3 The palette

Ten values. Everything on hue 299°, chroma rising with lightness so the tint stays
constant in *appearance* rather than in numbers.

| role | hex | L | C | note |
| --- | --- | --- | --- | --- |
| `surface-void` | `#0a0a0c` | 0.145 | 0.006 | the viewport, and only the viewport |
| `surface-base` | `#121115` | 0.180 | 0.008 | app chrome, editor, bars |
| `surface-panel` | `#1b1a1f` | 0.220 | 0.010 | floating panels and docks |
| `surface-raised` | `#26242a` | 0.265 | 0.012 | cards, inputs, pressed states |
| `surface-hover` | `#312f37` | 0.310 | 0.014 | pointer feedback only |
| `line` | `#3a3741` | 0.345 | 0.016 | the structural hairline |
| `line-strong` | `#514e59` | 0.430 | 0.018 | control edges, hover |
| `ink-3` | `#93909a` | 0.660 | 0.014 | labels, units, secondary |
| `ink-2` | `#bbb9c1` | 0.790 | 0.012 | body |
| `ink` | `#edecf2` | 0.945 | 0.008 | values, active state |

Contrast, ink on surfaces (WCAG 2.1):

| | void | base | panel | raised | hover |
| --- | --- | --- | --- | --- | --- |
| `ink` | 16.83 | 16.00 | 14.74 | 13.04 | 11.23 |
| `ink-2` | 10.21 | 9.71 | 8.94 | 7.91 | 6.81 |
| `ink-3` | 6.34 | 6.03 | 5.55 | 4.91 | 4.23 |

All three ink levels clear AA (4.5:1) on every surface except `ink-3` on
`surface-hover` (4.23), which is why **`ink-3` is forbidden on `surface-hover`** —
a hovered row promotes its label to `ink-2`, which is what you want anyway.

Surface separability, the thing today's thirteen near-blacks fail:

```
void   -> base:    3.51 ΔOKLab
base   -> panel:   4.00
panel  -> raised:  4.50
raised -> hover:   4.50
total span, 5 steps: 16.52     (today: 8.70 across 8 steps, 5 of 7 below JND)
```

Five surfaces instead of thirteen, and every step visible. **If two surfaces are
not 3 ΔOKLab apart they are the same surface; delete one.**

Hairlines stay deliberately below 3:1 (`line` is 1.50:1 on panel). This is a
choice, not an oversight: pushing a resting border to 3:1 needs roughly `#5c6159`,
which turns every panel into a drawn box. A resting border is not what tells you a
control exists — its *label, its baseline and its right-hung value* do. State is
carried by luminance and by the focus ring, both of which clear 3:1 easily.

### 3.4 The rule that replaces "pick a safe hue"

Since no hue is free, the constraint has to be spatial. It is also lintable, which
is why it is the constraint I would actually ship:

1. **The viewport rectangle is the only place a hue may be a fill.** Field ramp,
   quality ramp, four BC hues, proposal cyan. Nothing else.
2. **Panel chrome is achromatic.** A ramp may appear inside a panel only as a
   legend, and a legend is a bar ≤ 8px tall with its endpoints labelled. A ramp
   with no numbers next to it is decoration; delete it.
3. **Chrome colour never exceeds a 2px stroke, a 10px chip, or text.** Confusion
   between chrome and data comes from area and adjacency, not from hue identity.
   A 2px amber rule beside a "heat flux" label cannot be mistaken for a magma
   region; a 200px amber panel header can.
4. **Selection inside the viewport is white-hot, never lime.** `#fcfbfe` with a
   1px `#0a0a0c` halo. White is above both ramps' entire range, so it is the one
   mark guaranteed to read over any field value. This is the single largest
   correction the language makes to today's app.
5. **The lime is identity, not state.** `#d9ff57` appears exactly twice per screen:
   the `cj` mark and the one committing action (**Run**). Both live in the top bar,
   600px from the nearest field pixel. It is never a selection, never a focus ring,
   never a mode, never inside the viewport rectangle.

Status tones move off the ramps' read band (23–159°) so a red error chip is not a
magma sample:

| role | hex | h | on panel |
| --- | --- | --- | --- |
| `danger` | `#f56b7c` | 15° | 5.98:1 |
| `info` | `#81cffc` | 235° | 10.08:1 |
| `ok` | `#78d1c0` | 180° | 9.62:1 |

Three, not six. `danger-ink`, `info-ink`, `accent-model-ink` are lightness variants
of tones that already pass AA and should not exist.

---

## 4. Typography

Two families with disjoint jobs, and the split is semantic rather than aesthetic:

- **Sans** (`-apple-system` / `Inter`) — prose, control labels, section names,
  anything the *interface* says.
- **Mono** (`ui-monospace` / `SF Mono` / `JetBrains Mono`) — **every number, every
  unit, and every identifier that exists in the source program.** If a string in the
  panel also appears in `scene.py` — `sink-conduction`, `fin_depth`, `tet10` — it is
  mono. That single rule makes the panel visibly a *view of the code*, which is what
  the product claims to be.

### Why tabular figures are non-negotiable

This is a UI where numbers change while you look at them: a solve streams residuals,
an optimize run ticks an objective forty times, a probe updates as the pointer
moves. With proportional figures a `1` is narrower than a `0`, so a value that goes
`412.80 → 411.79` **reflows its own row**, and a column of results cannot be scanned
as a column because the digits do not line up. Every numeric surface therefore takes
`font-variant-numeric: tabular-nums` plus a monospaced family — belt and braces,
because some fallback stacks resolve to a proportional face.

Two more consequences worth stating as rules:

- **Numbers are right-hung on a shared column**, not left-aligned after their label.
  Magnitude becomes visible as edge alignment, and the decimal point becomes a
  vertical rule you can read down.
- **The unit is `ink-3` at one size below the value, on the same baseline.** Never
  in the label ("conductivity (W/mK)"), because the unit belongs to the number, and
  when the number is empty the unit should be too.

### The scale

Five sizes, three weights, three leadings. Today's census found 49 variants; the
budget is 5 × 3 = 15 and in practice about eight combinations do all the work.

| token | px | job |
| --- | --- | --- |
| `text-3xs` | 9 | small-caps labels, tracked 0.15em, `ink-3` only |
| `text-2xs` | 10 | dense secondary, selector expressions |
| `text-xs` | 11 | control labels, tab labels, hints |
| `text-sm` | 12 | body, the default |
| `text-md` | 13 | **numbers** — every value the user reads |
| `text-lg` | 15 | panel titles, and nothing else |

Note what that does: **the number is a size larger than its own label.** In an
instrument, the reading is the biggest thing in the row. Today the label and the
value share a size, and the panel reads as a form; give the value 13 against a 9px
tracked label and the same panel reads as a readout.

Weights: 400 / 500 / 600. There is no 700 — at 11–13px on a dark surface, 700
blooms and reads as a colour change rather than a weight change. Leadings: 1.0
(single-line controls), 1.35 (dense stacks), 1.55 (prose). Small-caps labels are
`text-3xs` at 0.15em tracking, uppercase, mono; the tracking is what makes 9px
legible, and 9px is what keeps a label from competing with its value.

---

## 5. Line and surface

**Hairlines over boxes.** The default way to separate two things is a 1px `line`
rule, full-bleed to the container's padding edge. The default way to *group* things
is a shared left edge and a shared baseline. A filled box is a last resort and needs
a reason — "it floats over the viewport" is a reason, "it is a card" is not.

**Elevation is luminance, not shadow.** Four surfaces, each one step brighter than
its parent, and exactly **one** shadow in the entire app: the panel that floats over
the viewport, `0 20px 44px -20px rgba(0,0,0,0.92)`. Nothing else casts. A dark UI
that shadows everything ends up with objects that are lighter *and* darker than
their background, which is incoherent.

**Two radii, and they mean different things.** `0` for anything that is part of a
sheet (rows, rules, table cells, the legend bar) and `3px` for anything that is an
object you can act on (buttons, inputs, chips, the floating panel). A 12px radius is
a consumer gesture; nothing in a CAD tool is that soft. Circles exist only for
status dots and pipeline nodes, where roundness carries the "this is a point, not a
region" meaning.

**One border weight.** 1px, always. A 2px border is not a stronger border, it is a
different element — reserve 2px for the *active* indicator (the rule under the
selected tab, the left edge of the live block), where the doubling reads as state.

---

## 6. Three modes without three unrelated accents

The current answer is lime / cyan / amber — three hues that, as measured above, are
all ramp colours, and which say nothing about the modes' relationship. But Model →
Sketch → Simulate is not three parallel worlds. It is **one pipeline**: geometry,
then the constraints that drive it, then the physics that scores it. Gradients flow
back along exactly that chain. A sequence deserves a sequence, not three arbitrary
colours.

So: **modes are an ordinal, expressed by position; the accent hue count is one, and
it is zero-hue.**

- The mode switcher is one strip of three cells in source order. The active cell is
  `ink` on `surface-raised` with a 2px `ink` rule; the inactive cells are `ink-3` on
  nothing. You read *which* by where the bright cell sits, the way you read a
  segmented gauge.
- The viewport's 1px top edge carries the same three-cell tick, in `line-strong`
  with the current cell in `ink`. This is the only mode signal inside the viewport,
  it is achromatic, and it is 1px tall, so it cannot fight the field.
- The hint bar names the mode in words. Words survive colour-blindness, greyscale
  screenshots, and the fact that the app is looked at by people who did not choose
  the accents.
- Per-mode *tool sets* still change wholesale — the rail, the panel, the shortcuts.
  That is a much louder signal than a hue, and it is already true.

This is strictly more information than three hues carried (order is now visible, and
progress through the pipeline is expressible), at a cost of zero reserved hues.

---

## 7. Density and rhythm

The target is 500+ elements per screen without the screen feeling loud.

- **Base grid 4px.** Spacing steps: 2 · 4 · 8 · 12 · 16 · 24. Six, not seven —
  6px was doing the job of both 4 and 8 and is the single most common source of
  near-miss alignment in the audit.
- **Control heights: 22 · 26 · 30.** Three, not five. 34 and 36 differ from 30 by
  less than a text line and produce the `control-height-near-miss` findings.
- **Rows are 26px and share one baseline.** Every row in a panel is
  `label | … | value` on a single baseline; a row that needs two lines uses a 10px
  sub-line under the value, right-aligned, `ink-3`, ellipsised. That is how a node
  selector expression fits without breaking the rhythm.
- **Panel width 320px, label gutter 96px, value column right-hung to the padding
  edge.** Fixed, across every panel. Two panels side by side must have their value
  columns line up or the eye reads them as unrelated.
- **Density is bought with type, not with space.** When a panel is too tall, the
  fix is dropping the label to 9px tracked and hanging the value at 13px — not
  shaving 2px off the padding. Padding below 8px inside a control stops reading as
  padding and starts reading as a mistake.

---

## 8. Motion

Motion in cadjoint has exactly one job: **to show causality between an edit and its
consequence.** The product's entire claim is that a change to the source propagates
through geometry, meshing and physics. Movement that does not express that
propagation is noise on an instrument face.

Three durations, each with a meaning — not three speeds:

| duration | meaning | example |
| --- | --- | --- |
| 90ms | *acknowledgement* — the pointer's own feedback, no state changed yet | hover tint, press |
| 160ms | *state* — something the user directly caused | tab switch, toggle, panel open |
| 260ms | *propagation* — something changed **because** something else changed | source rewritten → geometry recompiles |

One easing family: `cubic-bezier(0.2, 0, 0, 1)` for anything arriving, its
symmetric sibling for anything that moves and settles.

The rule that makes propagation legible: **causal pairs are sequenced, never
simultaneous.** When a panel edit rewrites the program, the changed source line
highlights first, and only then does the geometry update. Simultaneity is cheaper to
implement and destroys the causal reading — the user learns that the panel and the
viewport are two views, instead of learning that the panel *edits the code* and the
code *makes the geometry*. 60ms of stagger buys the product's core idea.

**What must never animate:**

- **Numbers.** A tweened number displays values that were never true. In an app
  whose users read tolerances, that is a correctness bug, not a polish choice.
  Values replace instantly; if a change needs marking, flash the row's background
  for 160ms.
- The camera under direct manipulation, ever.
- Anything during a solve except the progress indicator itself. A busy UI during a
  30-second CalculiX run makes the run feel broken.
- Layout on data arrival. Reserve the space; do not grow into it.

`prefers-reduced-motion` collapses 90ms and 160ms to 0 and reduces the 260ms
propagation to a two-frame highlight — the causality is information, so it survives
in a form that does not move.

---

## 9. Iconography

- **16px box, 1.25px stroke, no fills, miter joins, no rounded caps.** Geometric,
  drawn on the half-pixel so a vertical stroke lands on a device pixel. The stroke
  weight matches `line-strong` in feel, so an icon and a hairline look like they came
  out of the same pen.
- **Icons are `ink-2`, and colour is not available to them.** An icon that needs a
  colour to be understood needs a label instead.
- **Every rail icon has a text label at ≥1280px wide.** Icon-only is a space
  optimisation, not a design; a CAD tool with 30 tools and no words is a memory test.
- **No icon-only destructive action, at any width.** Delete carries the word.
- Icons never appear inside the viewport rectangle. Gizmos are geometry; they are
  drawn by the renderer in the data vocabulary, not by the icon set.
- The `cj` mark is the only fill in the icon system, and the only lime in it.

---

## 10. What this language refuses

Stated as refusals because a language is only disciplined if it can say no:

1. **No hue in chrome.** No blue primary button, no green "success" panel, no
   coloured card headers. If it is not a measurement it is grey.
2. **No colour inside the viewport rectangle that is not data.** Selection is
   white-hot; hover is a 1px white edge; a chrome accent never enters the field.
3. **No opacity-faded text.** An `opacity: 0.6` label has no assertable contrast
   ratio. There are three ink levels; use one.
4. **No new near-black.** If a surface is under 3 ΔOKLab from an existing one it is
   that surface.
5. **No decorative gradient.** A gradient in cadjoint is a colormap, and a colormap
   without a labelled scale next to it is a lie.
6. **No glassmorphism, no blur-behind panels.** A translucent panel over a field
   means the panel's background is a function of the data behind it, which makes
   panel contrast unassertable and makes chrome look like it is measuring something.
   Panels are opaque, or 95% opaque over the viewport, and that is the ceiling.
7. **No glow, no neon, no coloured shadow.** One shadow, black, on one element.
8. **No rounded-everything.** Two radii, 0 and 3, each with a meaning.
9. **No animated numbers, no skeleton shimmer, no spinner over content that is
   already legible.**
10. **No second weight of hairline** and no border used to say "this is
    interactive". Interactivity is said by height, baseline and hover.
11. **No icon without a word** for anything destructive, and no icon-only tool rail
    at desktop widths.
12. **No token whose name is a colour.** `--accent-simulate`, not `--amber`. A
    repaint must be one file's worth of change.
13. **No value off the scales.** Five sizes, six spacings, three control heights,
    two radii, three durations, ten colours. A value not on a scale is a bug report,
    not a nuance.

---

## 11. Making it stick

The language is only real if the linter can fail on it. `ui-audit` already censuses
colours, type and geometry; four checks turn this document into a gate:

- `chrome-hue` — any computed colour with OKLCH chroma > 0.03 outside the viewport
  rectangle, that is not one of the three status tones or the two lime instances.
- `ramp-adjacency` — any element painted with a ramp sample that is not inside a
  legend with two labelled endpoints.
- `surface-collision` — any two backgrounds under 3 ΔOKLab in the same view.
- `numeric-face` — any element whose text matches `^[-+]?[0-9]` and whose resolved
  font family is not monospaced, or which lacks `tabular-nums`.

Plus the existing `test/tokens.test.ts` pattern: assert the ten hexes, the five
sizes and the contrast table, and assert that `styles.css` declares exactly those
values.

---

## 12. The three directions

All three share the palette, the type scale and the refusals above — that is the
point of a *language*. They differ in how structure is drawn, which is the decision
the language leaves open. Rendered at 1500×940 over a real solved field
(the geometry is the actual playground render from `research/refactor/before-4-studies.png`,
recoloured through viridis so the panels are judged against data, not against grey).

### A — **Instrument** · `research/design/direction-a-instrument.png`

**Thesis:** the panel is an instrument face — filled cards, hairline separations,
9px tracked labels above 13px monospaced values right-hung on a shared column, and
the only colour on screen is the field itself.

**Cost:** it is the most conservative of the three. It looks unmistakably
*correct* and only quietly *specific* — a reviewer could describe it as "a very
well-behaved dark tool UI", which is a real criticism of a brief that asked for
something unique. The identity load falls almost entirely on the mark, the
achromatic discipline and the numbers, and if the discipline slips even slightly it
degrades back into a generic dark app.

### B — **Plate** · `research/design/direction-b-plate.png`

**Thesis:** the panel is a drawing sheet. Nothing is filled; structure is a 96px
small-caps label gutter, full-bleed hairlines between every row, and a hard right
decimal column. It reads as a datasheet or a title block — the document a machinist
would recognise.

**Cost:** two real ones. **Affordance** — with no fills and no borders, an editable
value and a read-only stat look identical, so "what can I click" has to be carried
entirely by hover, which is invisible in a screenshot and unavailable on the first
look. And **flatness under density** — at 500 elements the uniform rule rhythm stops
segmenting and starts striping; the eye loses its place vertically because every row
weighs the same. It is the most beautiful of the three at a glance and the one I
trust least at 9pm on the third hour.

### C — **Adjoint** · `research/design/direction-c-adjoint.png`

**Thesis:** the differentiable identity is drawn rather than described. A spine runs
down the panel — the chain the gradient travels, mesh → study → objective → result —
with the live stage lit; every parameter a gradient flows through carries its
measured ∂J/∂p as a bar and a signed value; and the source lines the tape touches
are marked in the editor gutter with the same device, so the panel and the code say
the same thing.

**Cost:** it breaks its own rule under load. The sensitivity bars are ramp-coloured
because ∂J/∂p is genuinely a measurement — but five of them stacked above the field
legend means six coloured bars in one 320px column, and in the render they read as
progress bars, not as data. The code gutter marking lights 20 of 32 visible lines,
which is nearly everything, so the mark stops carrying information. And the top-bar
mode chain reads as a wizard: circles-connected-by-a-line implies a required order
that modes do not have. It also cannot be shipped honestly until sensitivities are
actually available per parameter; drawn from anything less it is a decoration
pretending to be an instrument.

---

## 13. Recommendation

**Ship A — Instrument — as the base, and fold in the two devices from B and C that
earned their place.**

Reasoning, having looked at all three rendered rather than described:

- A is the only one where **the field is unambiguously the subject.** In B the
  panel's rule rhythm is loud enough to pull the eye off the geometry; in C the
  coloured bars compete with the legend directly. A's panel recedes and the solved
  sink is what you look at first. That was the taste constraint that mattered most,
  and A is the only one that passes it.
- A survives density. B's uniform rules and C's per-row bars both add work per
  element; A adds none — a longer study list in A is just a longer study list.
- A's identity is not weak, it is *load-bearing elsewhere*: the achromatic rule is
  itself the distinctive thing, and it is only distinctive because everyone else's
  dark tool UI has a blue primary. The lime mark against ten greys and a viridis
  field is a more specific image than a third accent hue would be.
- C contains the single best idea in this whole exercise — **marking the source
  lines the gradient passes through, with the same device the panel uses** — but that
  idea belongs in the editor gutter, not replicated as five bars in the panel.

**What I would change about A after seeing it rendered:**

1. **Take B's right-hand decimal column into A's cards.** A currently right-aligns
   values within each field, so a two-column `conductivity | source` row has two
   independent columns and the values do not line up down the panel. B proved the
   single right column reads better; A should adopt it and drop to one field per row.
2. **Take B's 96px small-caps gutter for stat rows.** A's `nodes / elements /
   jacobian` triple is a three-column strip; as gutter-labelled rows it is scannable
   and it stops competing with the legend above it.
3. **Take C's gutter mark, and only that.** A 2px `info` rule in the editor gutter
   on lines the adjoint tape touches, with no background tint and — critically —
   only on lines carrying a *free* parameter, not on the whole declaration. Ten
   marked lines is information; twenty is wallpaper.
4. **The SOLVE button is wrong in all three.** It is a full-width `surface-raised`
   slab that reads as a section header, not an action. It should be a 30px control
   at the card's right edge with a 1px `line-strong` edge, so the card's own bottom
   rule stays the strongest horizontal in the block.
5. **Drop A's card fill by one step.** `surface-raised` at 0.265 against a
   `surface-panel` at 0.220 is correct numerically but the cards still read as
   drawers. 0.245 keeps them separable (2.5 ΔOKLab — at the edge, but the card also
   has a top hairline doing work) and lets the panel read as one sheet with divisions
   rather than as a container of boxes.
6. **The tab strip needs the badge rule fixed.** A dot on "Results" is the one
   coloured chrome element in the panel, at 4px; that is defensible under the ≤10px
   rule, but it should be `ink` rather than `info` — "there is something here" is
   not a status, and making it achromatic removes the last hue from the panel.

---

## 14. Files

| what | where |
| --- | --- |
| the language | `research/design-language.md` (this file) |
| palette test, as a picture | `research/design/palette-from-colormaps.png` |
| direction A | `research/design/direction-a-instrument.png` |
| direction B | `research/design/direction-b-plate.png` |
| direction C | `research/design/direction-c-adjoint.png` |
| reference (today) | `research/refactor/before-4-studies.png` |

The derivation scripts (`palette.mjs`, `accent.mjs`) and the standalone HTML
mockups live in the scratchpad workspace; every number quoted above is reproducible
from them.
