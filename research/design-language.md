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

## 14. Surfaces: light, depth and translucency

*Second pass. §3 fixed the hue and the refusals; it left the surfaces themselves
flat and very dark — five near-blacks between L 0.145 and 0.310, elevation by
luminance alone. This section asks whether that is right, and answers with nine
rendered variants of the same screen and the numbers behind each. Images in
`research/design/surfaces/`; every figure below is reproducible from
`variants.mjs` / `final.mjs` in the scratchpad workspace.*

### 14.1 The number the whole section turns on

**viridis' darkest sample is L 0.290** — `viridis(0.00) = #460155`, the coldest
pixel any solved field can contain. The ramp never goes below it:

```
viridis  L 0.290 (t=0.00, #460155)  ..  0.914 (t=1.00, #fae720)
magma    L 0.000 (t=0.00, #000000)  ..  0.970 (t=1.00, #fff8bf)
```

So the field has a **floor**, and chrome has a ceiling. A panel fill at L 0.335
is not merely "a bit light": it is measurably brighter than 11% of every
temperature field the app will ever draw, and the eye reads the brighter thing
as the subject. That single number decides how far up the ladder chrome may go,
and it is the reason variant 3 (GRAPHITE) fails and variant 9 does not.

magma reaches black, so it cannot set the ceiling; it is also the *quality* ramp,
which is read in short bursts. viridis is the ramp that owns the screen.

### 14.2 What the current stack actually measures

| ladder | span | steps | smallest step |
| --- | --- | --- | --- |
| cadjoint today (8 near-blacks) | **8.70** ΔOKLab | 7 | 0.47 |
| §3's proposal, ZERO (5 surfaces) | 16.52 | 4 | 3.51 |
| Linear's reported dark ladder | 7.82 | 4 | **0.44** |
| **this section's recommendation** | **24.41** | 6 | 2.72 |

Linear is worth putting in the table because it is the reference everyone
reaches for, and because its ladder is *also* compressed — its last two surfaces
(`#18191a` → `#191a1b`) are 0.44 ΔOKLab apart, a quarter of a JND. That is fine
for Linear: it is a text application on a black ground, its ceiling is the top of
the sRGB gamut, and it can carry hierarchy on hairlines because nothing on screen
competes with them. cadjoint has neither the same subject nor the same ceiling.
It has *more* headroom than Linear below its ceiling (0.110 → 0.290 is a usable
18 ΔOKLab of ladder) and a stronger reason to use it: the panel has to hold its
own against a colour-mapped field 40 px to its left.

### 14.3 The nine variants

All nine are the same DOM, the same layout and the same field image; only the
surface system changes. Renders at 1500×940, examined at 1× before judging.

| # | name | thesis | measured |
| --- | --- | --- | --- |
| 1 | **CONTROL · Today** | The shipped palette. 8 near-blacks on h 165° (viridis' *middle*), flat fills, elevation by shadow. | span 8.70 / 7 steps; 5 of 7 steps under JND. Cards and inputs are invisible as objects; the panel reads as one black hole with a border doing all the work. |
| 2 | **ZERO · direction A** | §3's answer: h 299°, five surfaces, flat fills, L 0.145–0.310. | span 16.52 / 4. Clearly better than today. But cards at Δ 4.5 from the panel still read as a tint, not a plate, and underlined values read as text rather than as controls. |
| 3 | **GRAPHITE · elevated, flat** | How light can chrome go? Gunmetal panels, L 0.125–0.45, flat fills, inputs as inset darker holes. | Panel L 0.335 → **outshines 11% of the viridis range**. Editor at L 0.265 turns 40% of the screen into a grey slab. And the inset inputs read as *disabled*: a value in a hole darker than its card looks switched off. |
| 4 | **PLATE · elevated + lit** | Graphite's ladder plus one global light direction: top-lit washes, 1px lit-top / dark-bottom edges, a gradient viewport well, inputs as lighter planes. | Same ladder, transformed. The lit inputs read as editable where Graphite's read as dead; the cards read as objects. Still too light: panel L 0.296–0.345 crosses the floor. |
| 5 | **WELL · ceilinged + lit** | PLATE's light model with the stack pulled back under the floor. L 0.115–0.325. | The first variant where nothing competes with the field and everything is still separable. Cards and controls poke 1–3 above the floor. |
| 6 | **WELL-WARM** | WELL on a warm neutral (h 78°) instead of the ramps' violet zero. | See §14.5 — the difference is **below JND at every surface**. |
| 7 | **GLASS** | The whole panel 68–70% opaque over `blur(28px) saturate(.45) brightness(.62)`. | ink-3 falls to **3.42:1**; backdrop σ under a column of glyphs **triples**. |
| 7b | **GLASS-NAIVE** | The same at 52–55% with `blur(16px)` and no brightness knockdown — the version copied off a marketing site. | ink-2 **4.34:1**, ink-3 **2.68:1**. Both fail AA. |
| 8 | **APERTURE** | Selective translucency: a 40%-opaque shell, every number-bearing block an opaque plate on top of it. | Contrast identical to fully opaque (ink-3 4.54 vs 4.58). Costs nothing — and buys almost nothing. |
| 9 | **PLATE-WELL** | **The recommendation.** WELL's architecture, re-spaced so no fill over 4 000 px² exceeds L 0.290; light model kept, noise dropped, translucency dropped. | span 24.41 / 6 steps, smallest 2.72; 0% of viridis outshone by any large fill. |

### 14.4 Light, not shadow

The finding that mattered most was not in the palette, it was in variants 3 vs 4:
**the same lightness ladder reads as dead or alive depending on whether there is
a light direction.** Graphite and Plate differ by nothing except a ±2.7 ΔOKLab
vertical wash and a pair of 1px edges, and Graphite's number fields read as
disabled while Plate's read as editable.

So the language gains one physical commitment: **there is a single light source,
it is above the screen, and every raised element is lit by it.**

- **Wash.** A panel or a card is a vertical gradient of **±1.4 ΔOKLab about its
  nominal** (panel `#25232a` → `#1e1d22`, amplitude 2.75; card `#2c2931` →
  `#26242b`, amplitude 2.21). Just over one JND — perceptible as a direction,
  never as a stripe.
- **Edges.** Anything raised carries `inset 0 1px 0 var(--edge-lit)` and
  `inset 0 -1px 0 var(--edge-dark)`. Those are 9.76 and 11.46 ΔOKLab from the
  surface they edge, i.e. **the 1px edge is a far stronger separator than the
  fill step it decorates** — which is the whole point: it buys separation without
  spending lightness budget against the field's floor.
- **Direction of elevation is inverted from today.** Controls are *lighter*
  planes, never darker holes. This is not taste: at panel L ≥ 0.25 a darker inset
  is the same signal the app uses for "disabled", and variant 3 shows it read that
  way.
- **Shadow stays at one.** §5's rule survives: exactly one cast shadow in the
  app, on the panel that floats over the viewport. The edge lights are not
  shadows; they are the same light model seen from the other side.
- **The viewport is a well.** `radial-gradient(118% 92% at 46% 33%, #161519,
  #040305)` — 5 ΔOKLab from centre to corner. It is below conscious notice and it
  does one job: it stops the viewport being a flat black rectangle and gives the
  geometry somewhere to sit.

This is what the industry converged on independently. Linear "trusts surface lift
and hairline borders to carry every bit of hierarchy", raising each level's white
overlay 0.02 → 0.04 → 0.05 rather than casting shadows; Radix's scale reserves
steps 3/4/5 for component background / hover / pressed, i.e. elevation *is* the
scale; Geist spends steps 100/200/300 on default / hover / active background and
400/500/600 on the matching borders. All three encode the same idea: **elevation
is a step on a scale, and a border is a second scale that moves with it.**

### 14.5 Temperature: not a decision

The hypothesis was that a warm neutral would read "made" where a cool one reads
"rendered". Measured, at the chroma the language actually permits chrome (C ≤ 0.014):

| surface | C | h 299° | h 78° | ΔOKLab |
| --- | --- | --- | --- | --- |
| void | 0.004 | `#050506` | `#060504` | 0.73 |
| base | 0.007 | `#18161a` | `#191714` | 1.37 |
| panel | 0.009 | `#232126` | `#24221d` | 1.85 |
| raised | 0.010 | `#2b2a2f` | `#2e2a25` | 1.85 |
| hover | 0.011 | `#353339` | `#37332e` | 1.95 |
| line | 0.013 | `#3e3c43` | `#413c36` | 2.25 |

A **221° hue rotation** — violet to warm, most of the way round the wheel — moves
every surface by less than or barely at one JND. Rendered side by side (variants 5
and 6) they are indistinguishable at 1×.

The temperature only becomes visible if chroma rises: Δ 3.76 at C 0.020, Δ 5.50 at
C 0.030, Δ 8.31 at C 0.045. But C > 0.03 is exactly what §11's `chrome-hue` lint
forbids, and for good reason — at that chroma the chrome has a hue, and every hue
on screen is supposed to be a number.

**Verdict: chrome temperature is not a design decision at this chroma; it is a
rounding error.** Keep h 299°, because it is derived from where the two ramps
agree and therefore costs nothing to justify. Anyone who wants the greys to feel
warmer is asking for C ≥ 0.03, and that is a different, larger argument that this
language has already lost on purpose. (Linear's own 2026 refresh moved "from a
cool, blue-ish hue toward a warmer gray" — which is a real change only because
their neutral carries more chroma than ours is allowed to.)

### 14.6 Banding: measured, and the fix is worse than the disease

Four gradients at the L ranges the system actually uses, rendered at 1500px wide
and measured across a scanline:

| gradient | distinct 8-bit values | median band width | **max single-step ΔOKLab** |
| --- | --- | --- | --- |
| viewport well 0.163 → 0.094 | 35 | 1px | 0.83 |
| panel wash 0.272 → 0.234 | 20 | 1px | 0.43 |
| card wash 0.345 → 0.296 | 25 | 1px | 0.39 |
| worst case 0.200 → 0.090 | 60 | 1px | 1.21 |
| *+ noise overlay (0.055 opacity)* | *132–250* | *1px* | ***1.08–2.14*** |

Median band width is **1px in every case**: the browser is already
error-diffusing the gradient, so there are no bands to see — the transitions are
dithered, not stepped. Every single step is under 1.25 ΔOKLab, well below the
JND of ≈ 2.

Adding a film-grain overlay raises the maximum step to 1.08–2.14 — **at or above
JND** — and multiplies the distinct values by 4–7×. At 1× the noise is visible as
mottling on the card wash, and the banding it was meant to hide is not.

**Verdict: no noise, no dither, no grain. The artifact does not exist at these
amplitudes, and the remedy is measurably louder than the disease.** This holds
because the washes are shallow by design (±1.4 ΔOKLab); a deeper gradient would
change the answer, which is another reason not to have one.

### 14.7 Translucency

The language currently refuses glassmorphism outright (§10.6). Having built and
measured it, the refusal is **right for the reason it gives, and for a second
reason it does not give — but it is stated too broadly.**

Background luminance under the glyph-bearing regions of the panel, over a
deliberately worst-case backdrop (the field repositioned so its bright viridis
region — mean L 0.331, p95 0.576, peak 0.748 — sits behind the panel):

| variant | bg L under glyphs | **σ** | ink | ink-2 | ink-3 |
| --- | --- | --- | --- | --- | --- |
| opaque (variant 5/9) | 0.238–0.285 | **0.020** | 12.24 | 7.41 | **4.58** ✓ |
| APERTURE, selective | 0.142–0.288 | **0.016** | 12.12 | 7.34 | **4.54** ✓ |
| GLASS, tuned | 0.239–0.363 | **0.048** | 9.12 | 5.52 | **3.42** ✗ |
| GLASS-NAIVE, blur only | 0.221–0.416 | **0.077** | 7.17 | 4.34 ✗ | **2.68** ✗ |

Three things fall out.

1. **Full translucency fails on the third ink level, not the first.** `ink` is
   never in danger — it clears AAA over anything. It is `ink-3`, the label and
   unit level, that collapses: 4.58 → 3.42 tuned, 2.68 naive. Since §7 puts every
   *label* at 9px `ink-3` and every *unit* at `ink-3` beside its value, glass
   breaks precisely the type that carries what a number means.
2. **The tuned version only survives because it is not really transparent.**
   `brightness(0.62)` is doing the work; it is a dimming filter with a blur
   attached. Take the knockdown away — variant 7b, which is what "add a
   backdrop-blur" means in practice — and ink-2 fails too.
3. **The real objection is σ, not the mean.** The spread of the background under
   a single column of glyphs goes from 0.016–0.020 (a designed, assertable,
   lintable constant) to 0.048–0.077. A translucent panel does not have a
   contrast ratio; it has a *distribution*, and the distribution moves when the
   camera orbits. That is what §10.6 means by "unassertable", and the numbers
   support it.

**A fourth thing, which is the one that should actually settle it, and which none
of the contrast maths above catches: the panel contains a colormap legend.**

`.sim-ramp` — the viridis bar with its two labelled endpoints — lives inside this
panel, and §3.4 permits it there precisely because "the chrome around it stays
neutral". Translucency breaks that invariant. The swatch pixels stay correct; the
*ground the eye judges them against* does not. Measured ΔE76 of the panel ground
as the field behind it sweeps viridis(0) → viridis(1):

| panel α | ground over `#440154` | ground over `#fde725` | ΔE76 |
| --- | --- | --- | --- |
| 0.82 | `#170e1b` | `#393712` | **36.4** |
| 0.91 | `#120f15` | `#232410` | **18.9** |
| 0.96 | `#101011` | `#17190f` | 8.3 |
| **1.00** | `#0e110f` | `#0e110f` | **0.0** |

Simultaneous contrast is not a subtlety here: a viridis chip read against an
olive-shifted ground is judged as a different colour than the same chip against a
neutral one. For a legend whose entire job is colour-matching against the field,
that is a **correctness defect in an instrument**, not a style preference. It is
also the only argument in this section that a translucency advocate cannot
answer by turning a knob — there is no alpha above 0 at which the ground is
constant.

There is a related failure below α ≈ 0.5: `ink-3` passes through zero contrast
(1.62:1) and re-emerges as dark-on-light as the field heats. A label that
*inverts polarity* mid-solve is worse than one that is merely dim.

**On the engineering cost, the earlier framing here was wrong and is corrected.**
A first measurement in this workspace — a 306×546 panel over a 2D canvas
repainted every frame — found no difference at all (opaque 4.10–4.25 ms/frame,
blurred 4.14–4.15; the two opaque runs differed by more than opaque differed from
blurred). That test was simply insensitive: it was CPU-bound on the canvas paint,
on a small surface. A properly isolated measurement (M-series, Chromium 151,
3024×1890 canvas) separates the two effects:

| condition | ms/frame | marginal |
| --- | --- | --- |
| no panel | 21.99 | — |
| opaque panel | 21.99 | −0.01 |
| α 0.85, **no blur** | 21.97 | **−0.02** |
| α 0.85, `blur(20px)` | 22.48 | **+0.49** |

**Translucency itself is free. Only the blur costs** — about 0.48 ms, ≈ 2.9% of a
60 fps budget. And the cost does not scale with the panel, it scales with the
*canvas*: holding the panel at 306×520 and varying only DPR gives +0.055 ms at
1.43 MPix, +0.479 at 5.72, +0.627 at 12.86. Sweeping the panel from 0.06 to 5.64
MPix leaves it flat at ~0.5 ms; sweeping the blur radius from 2 to 120px likewise
(Skia downsamples above sigma 4).

The mechanism is in Chromium's compositor, not in a blog post —
`cc/trees/damage_tracker.cc`:

```cpp
if (render_surface->BackdropFilters().HasFilterThatMovesPixels() &&
    intersects_damage_under) {
  damage_for_this_update_.Union(surface_rect_in_target_space, ...);
}
```

A blur is a pixel-moving filter, so **any damage beneath the panel unions the
panel's whole surface rect into the frame's damage**. The viewport damages itself
every frame for the whole accumulation window, so there is no cached-blur path
while a solve or a path-trace runs — exactly the window in which the GPU is most
contended. "It is only a 306px panel, it must be cheap" is the wrong model.

So the honest cost statement is: *translucency is free, blur is not, and the
reason blur is not free is a whole-canvas damage union that gets worse on the
larger displays this app is most likely to be used on.*

**Selective translucency passes the contrast test but fails a design one.**
APERTURE holds AA exactly (4.54 vs 4.58) because no glyph ever leaves an opaque
plate — a legitimate construction that costs nothing measurable. But what it buys
is a ~10px translucent frame, and Figma shipped precisely that idea in UI3:
floating panels pulled a few pixels off the edge so the canvas showed around
them. The reported complaint was that "designs seemed to peek out from behind
them in a way that was distracting", and Figma reverted to docked panels. A 10px
window onto the model is not enough to answer "what is under the panel" and is
enough to add motion at the edge of the thing you are reading.

**Two precedents worth naming, because both are closer to this product than any
marketing site.** Excalidraw — floating panels over a live canvas — removed
`backdrop-filter` and *kept* the translucency, for exactly the performance
reason above (issue #3505 / PR #3506). And Linear publicly declined Apple's
Liquid Glass APIs in October 2025, in terms that could have been written for
this section: *"refraction can make dense professional interfaces harder to
read. By relying on precise blurs, masking, and lighting, we maintained a sense
of depth without losing clarity."* The team building the closest analogue to
cadjoint's chrome looked at the most heavily marketed translucency system of the
decade and said no to it on legibility grounds.

**Verdict: the panel is opaque, and the refusal is restated so it says why.**
Replace §10.6:

> **6. No text, and no colormap, on a backdrop that is a function of the data.**
> A surface carrying glyphs or a legend is opaque — its contrast and its
> surround must be constants the linter can assert. This is not a taste
> position about glass: at any α < 1 the panel ground shifts by up to ΔE76 36
> as the field sweeps, which makes a viridis legend read differently depending
> on what is behind the panel, and below α ≈ 0.5 `ink-3` inverts polarity
> mid-solve.
>
> Translucency without blur is free and does not fail contrast on its own;
> `backdrop-filter` is not free, because a pixel-moving filter unions the
> panel's whole rect into frame damage every frame the viewport redraws. If a
> translucent panel is ever justified, it is translucent *without* a blur,
> α ≥ 0.96, it contains no ramp, and it honours `prefers-reduced-transparency`.
>
> The need this was trying to serve — *what is under the panel?* — is answered
> by a **peek**: hold a key and the panel drops to 15% with its text hidden
> entirely, for as long as it is held. That gives the whole panel's worth of
> geometry instead of a 10px sliver, it is unambiguous, and it never asks
> anyone to read a number off a moving ground.

### 14.8 The recommended stack

Nine tokens on the ladder, four wash values, two edge values, three inks. All on
h 299° with C = min(0.014, 0.045·L), so the tint stays constant in appearance.

| token | hex | L | C | Δ from previous | % of viridis outshone |
| --- | --- | --- | --- | --- | --- |
| `surface-void` | `#050406` | 0.110 | 0.007 | — | 0.0% |
| `surface-base` | `#131216` | 0.185 | 0.008 | 7.47 | 0.0% |
| `surface-bar` | `#19181d` | 0.212 | 0.010 | 2.72 | 0.0% |
| `surface-panel` | `#222026` | 0.248 | 0.012 | 3.59 | 0.0% |
| `surface-card` | `#29272e` | 0.278 | 0.013 | 2.95 | 0.0% |
| `surface-control` | `#343139` | 0.319 | 0.015 | 4.17 | 4.9% |
| `surface-hover` | `#3d3a42` | 0.355 | 0.014 | 3.51 | 10.1% |

Washes and edges:

| token | hex | L | job |
| --- | --- | --- | --- |
| `panel-lit` | `#25232a` | 0.261 | top of the panel wash |
| `panel-shade` | `#1e1d22` | 0.234 | bottom of the panel wash |
| `card-lit` | `#2c2931` | 0.287 | top of the card wash |
| `card-shade` | `#26242b` | 0.265 | bottom of the card wash |
| `edge-lit` | `#45424a` | 0.385 | the 1px lit top of anything raised |
| `edge-dark` | `#0b0b0e` | 0.151 | the 1px shaded bottom |
| `line` | `#3a3840` | 0.346 | structural hairline |
| `line-strong` | `#535159` | 0.440 | control edges, hover |
| `ink` | `#edecef` | 0.945 | values, active state |
| `ink-2` | `#bbbabe` | 0.791 | body, icons |
| `ink-3` | `#939197` | 0.660 | labels, units, secondary |

Span **24.41 ΔOKLab over six steps**, smallest step 2.72 — nearly three times
today's total range, with every step above JND. Panel wash amplitude 2.75, card
wash 2.21. `edge-lit` sits 9.76 from `card-lit` and `edge-dark` 11.46 from
`card-shade`.

Contrast, WCAG 2.1, ink on every resting surface (washes measured at both ends):

| | void | base | bar | panel↓ | panel↑ | card↓ | card↑ | control | hover |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ink` | 17.39 | 15.85 | 15.00 | 14.23 | 13.19 | 13.03 | 12.15 | 10.85 | 9.48 |
| `ink-2` | 10.60 | 9.66 | 9.14 | 8.68 | 8.04 | 7.94 | 7.41 | 6.62 | 5.78 |
| `ink-3` | 6.56 | 5.98 | 5.66 | 5.37 | 4.98 | 4.92 | 4.59 | **4.10** | **3.58** |

`ink` and `ink-2` clear AA everywhere. `ink-3` clears AA on every surface up to
and including `card-lit` (4.59) and **fails on `surface-control` and
`surface-hover`** — so, exactly as §3.3 already rules for hover: **`ink-3` is
forbidden on `surface-control` and `surface-hover`; a control's own label is
`ink-2`.** That is what you want anyway — the label inside a control is not
secondary, it is the control.

Distance to the ramps, so chrome is provably not a sample of anything:

| token | nearest viridis | nearest magma |
| --- | --- | --- |
| `surface-panel` | Δ 13.4 (t 0.00) | Δ 8.7 (t 0.10) |
| `surface-card` | Δ 12.6 (t 0.01) | Δ 10.1 (t 0.13) |
| `surface-control` | Δ 12.1 (t 0.08) | Δ 11.9 (t 0.16) |
| `line-strong` | Δ 8.6 (t 0.27) | Δ 15.9 (t 0.34) |

The tightest is `line-strong` at 8.6 from viridis(0.27) — a 1px hairline against
a ramp sample, which is inside §3.4's ≤ 2px stroke allowance.

### 14.9 Two new rules, and one amended

Added to §10's refusals:

14. **No chrome fill larger than 4 000 px² may exceed L 0.290** — viridis'
    darkest sample. Panels, cards, bars, the editor and the viewport are all
    subject to it. A 145×24 input is not, which is why `surface-control` is
    allowed at 0.319; a 306px-wide panel header is not exempt at any lightness.
15. **No inset control.** A control is a plane lit from above: `surface-control`
    fill, `edge-lit` on its top pixel, `edge-dark` on its bottom. A field drawn
    darker than the card it sits in reads as disabled, which is a different
    message.

Amended:

6. ~~No glassmorphism, no blur-behind panels.~~ → **No text, and no colormap, on
   a backdrop that is a function of the data** — as restated in full in §14.7.
   The panel is opaque; "what is under the panel" is a held peek to 15% with the
   text hidden, not a permanently translucent panel.
5. Unchanged in intent, sharpened in fact: **no decorative gradient** — but the
   ±1.4 ΔOKLab wash that expresses the light direction is not decoration and not
   a colormap; it is below the threshold at which a gradient reads as a gradient,
   and it is the mechanism that makes elevation legible without spending
   lightness against the field's floor.

And one lint check joins §11's four:

- `field-floor` — any element whose painted area exceeds 4 000 px² and whose
  background resolves to OKLab L > 0.290, outside the viewport rectangle.

### 14.10 Recommendation

**Ship variant 9, PLATE-WELL** (`research/design/surfaces/9-plate-well-recommended.png`).

- It is the only variant where the ladder is fully legible *and* nothing on it
  outshines the field. Variants 3 and 4 buy separability by crossing the floor;
  variants 1 and 2 stay under the floor by giving up separability. 9 gets both
  because the light model supplies separation that the fill values do not have
  to pay for.
- The light direction is the substantive change, not the lightness. It is what
  turns a number field from a hole into a plate, and it is one line of CSS per
  element rather than a new colour.
- The translucency work should ship as a *refusal with a receipt*: the numbers in
  §14.7 are the answer next time someone proposes glass, and the held peek is the
  feature that request was actually asking for.
- Noise and warmth were both hypotheses that measurement killed, which is the
  point of measuring. Neither costs anything to drop.

One thing this section deliberately does **not** settle. Every contrast figure
above is WCAG 2.1, which is the standard the existing `tokens.test.ts` asserts.
Under APCA — the perceptual model that actually predicts thin light-on-dark text
— `ink-3` at 9px is weak on the upper surfaces regardless of what the ladder
does, and no alpha or elevation choice fixes that. If that turns out to matter
more than surfaces do, the lever is the *type* (`ink-3` at 10px/500 rather than
9px/400), not the greys. Worth its own pass; out of scope for this one.

### 14.11 What was looked at

Beyond the nine renders: Radix Colors' 12-step scale (steps 1–2 app/subtle
background, 3/4/5 component background–hover–pressed, 6/7/8 borders, 11/12 text);
Vercel Geist's 10-step equivalent (100/200/300 background–hover–active,
400/500/600 the matching border triple); Linear's published UI redesign notes
(reduced dividers, a warmer and less saturated neutral, hierarchy carried by
surface lift rather than shadow) and its reported four-step ladder, measured
above; Linear's October 2025 statement declining Apple's Liquid Glass on
legibility grounds; Figma's UI3 floating-panel rollback and the forum record of
why; Excalidraw's removal of `backdrop-filter` from panels over a live canvas
(#3505 / #3506); Chromium's `cc/trees/damage_tracker.cc` for the mechanism; and
Blender theming guidance on keeping the viewport ground close enough to the
material that objects stay differentiable without producing retina burn.

**Rejected on purpose.** *One very large number per panel* — a 40px hero metric
is a dashboard gesture; at 500 elements it steals the eye from the field and
there is no single number in a study that deserves it. *Rounded, softened,
low-contrast separation* (Linear's direction) — right for a text app, wrong here,
because our separation problem is between chrome and a colour-mapped image, not
between rows of text. *Border-led hierarchy* (Geist's 400/500/600 triple) — a
resting border at 3:1 turns every panel into a drawn box, which §3.3 already
rejected; we take the *idea* that borders move with elevation and spend it on the
1px edge lights instead. *Glass, in all its 2026 forms* — for the reasons in
§14.7.


## 15. Files


| what | where |
| --- | --- |
| the language | `research/design-language.md` (this file) |
| palette test, as a picture | `research/design/palette-from-colormaps.png` |
| direction A | `research/design/direction-a-instrument.png` |
| direction B | `research/design/direction-b-plate.png` |
| direction C | `research/design/direction-c-adjoint.png` |
| reference (today) | `research/refactor/before-4-studies.png` |
| the surface study (§15) | `research/design/surfaces/` |
| control (today's palette) | `research/design/surfaces/1-control-today.png` |
| §3's proposal rendered | `research/design/surfaces/2-zero.png` |
| elevated, flat | `research/design/surfaces/3-graphite.png` |
| elevated, light-modelled | `research/design/surfaces/4-plate.png` |
| ceilinged, light-modelled | `research/design/surfaces/5-well.png` |
| warm neutral | `research/design/surfaces/6-well-warm.png` |
| translucent, worst-case backdrop | `research/design/surfaces/7-glass-worst-case.png` |
| translucent, no brightness knockdown | `research/design/surfaces/7b-glass-naive-worst-case.png` |
| selective translucency, worst case | `research/design/surfaces/8-aperture-worst-case.png` |
| **the recommendation** | `research/design/surfaces/9-plate-well-recommended.png` |
| all nine panels side by side | `research/design/surfaces/contact-sheet-panels.png` |
| the banding test | `research/design/surfaces/banding-test.png` |
| the character study (§16) | `research/design/playful/` |

The derivation scripts (`palette.mjs`, `accent.mjs`) and the standalone HTML
mockups live in the scratchpad workspace; every number quoted above is reproducible
from them.

---

## 16. Character: where the personality lives

*Third pass. §3 fixed the hue, §14 fixed the surfaces, and between them they
produced a language that is correct and severe. The criticism that prompted this
section — "it lacks personality" — is fair, and the reason is diagnosable: the
first two passes spent all of their attention on **static panels**, which is the
one part of this product that is like every other tool. Nothing had been asked of
the three things that are not: an optimizer that morphs a shape while its
objective falls, gradients that run backward through the program, and an
instrument lineage that is genuinely playful in real life.*

*So: no constraint from §3, §10 or §14 is overturned here. **Character is found in
time and in precision, not in colour.** Six ideas, rendered and animated in
`research/design/playful/`; two rejected with measurements.*

### 16.1 The one amendment, and the argument for it

§8 says **numbers must never animate**, because "a tweened number displays values
that were never true." That rule is right and it stays. But it does not apply to
what the tape does, and the distinction matters enough to write down:

> **Replay is not animation.** Playing a recorded trajectory back is not tweening
> — every frame shows a value the optimizer actually held at step *k*. The rule
> forbids inventing intermediates between two true values; it does not forbid
> showing a sequence of true values in the order they occurred. A tape may
> therefore run the objective, the parameters and the geometry through forty
> real states at twelve states per second, and it is showing *more* truth than a
> static end-state does, not less.
>
> The test is mechanical: **may this frame be exported as a row of the run log?**
> If yes it is replay. If no it is a tween, and it is forbidden.

Everything else in §8 survives intact: a *live* value still replaces instantly,
and §16.8's ballistics apply only to needles, bars and rules — never to digits.

### 16.2 THE TAPE — the trajectory, played

`research/design/playful/1-tape.png` · **`1-tape.gif`**

**Thesis:** the optimizer is the product's whole argument and it is currently a
sparkline. Make the run a scrubbable tape: the geometry morphs, the objective
curve draws itself, each parameter walks its own machinist's rule, and the source
literals rewrite as it goes.

- **Trigger.** A run completes, or the Optimize tab is opened on a run that has a
  tape. Autoplay once; thereafter scrub.
- **Rate.** 12 steps/s (83 ms per recorded step). Runs longer than ~60 steps hold
  the *total* at ≈3.4 s by dropping frames; never exceed 25 steps/s, past which
  the shape reads as flicker rather than as morph.
- **The strip.** One tick per recorded step across the card width: `#37343d`
  unplayed, `line-strong` played, `ink` at 3px and full height for the live step.
  Every 10th tick 72% height, the rest 44%. Dragging it **snaps to the nearest
  step** — the scrubber is detented, and it never lands between two real states.
- **The trace.** 292 × 66 inside the card. Graticule 10 × 4 divisions at
  `#37343d`; the two centre axes carry five subdivisions per division at
  `#4a474f` and nothing else does (this is the real Tektronix arrangement, §16.9).
  Trace 2.4px `ink`, drawn only as far as played. A writing head: 2px
  `line-strong` full-height cursor plus a 3.4px `ink` dot at the live sample.
- **The travel rules.** One per parameter, 292 × 14, spanning the parameter's
  declared bounds: 20 minor divisions with every fifth major (8px vs 4px), the
  interval swept so far as a 3px `ink-3` bar on the track, the start value as an
  11px `ink-3` tick, the live value as a 17px `ink` tick. This is a machinist's
  rule, not a slider — there is no thumb, no fill, and no track colour.
- **Ghosting, not blur.** The last five recorded envelopes at k−3 … k−15, top
  faces only, 1.6px `#edecef` at α 0.25 / 0.20 / 0.15 / 0.10 / 0.05, drawn
  **after** the field. White is above both ramps' entire range (§3.4-4), so a
  ghost can never be misread as a temperature. Ghosts exist only while the tape
  is moving or being scrubbed; at rest there are none. (Blur was built and
  measured; see §16.10.)
- **Type.** The objective is the one place editorial contrast is earned: 30px
  mono `ink` against a 9px tracked `ink-3` label. §14.11 rejected "one very large
  number per panel" as a dashboard gesture, and that rejection stands for
  *panels*; a 30px number is licensed **only** on the objective of a live or
  replayed optimization, because that number is the thing the run is about.
- **Reduced motion.** No autoplay. The tape renders at its final step, the whole
  trace drawn and the strip fully played. Scrubbing still works — it is
  user-driven, not motion.

**Cost.** One canvas redraw per step, which the viewport does anyway; the panel
work is three 292px canvases. Nothing scales with element count, so the 500-element
budget is untouched. The real cost is upstream: the trajectory has to be *recorded*
— parameters, objective and gradient per step — and kept.

**Verdict: ship. This is the single biggest opportunity and it survives being
watched.** The one honest caveat: with a fixed camera the ghosts are only legible
because this shape grows monotonically. On a trajectory that oscillates they will
overlap into hash, and the fallback is to ghost the *bounding envelope of the last
five steps* as one outline rather than five.

### 16.3 THE GRATICULE — the viewport becomes an instrument face

`research/design/playful/3-graticule.png` · **`3-detent.gif`**

**Thesis:** the viewport currently has no scale on it at all. Give it a real one,
in the Tektronix idiom, and the precision *is* the personality.

- **The graticule.** 10 horizontal × 8 vertical divisions, 1px at `#1e1d22`,
  drawn **behind** the geometry so the part occludes it and it can never compete
  with the field. The two centre axes — and only they — carry five subdivisions
  per division at `#332f38`, arms 5px (9px on every fifth). Four 26px corner
  brackets at `#413e47`, 2px: the frame stated four times rather than drawn as a
  box.
- **The gain readout.** Top-left, never moves:
  `H 10.0 mm/div   V 10.0 mm/div   VIEW ISO · +X+Y+Z`. 9px tracked `ink-3` key,
  12px mono `ink-2` value.
- **The detent.** Zoom snaps on a **1-2-5 ladder** (20 / 10 / 5 / 2 mm/div …) over
  160 ms on `cubic-bezier(0.2, 0, 0, 1)`, so one division is always an exact,
  stateable number of millimetres. The readout changes **at** the detent crossing,
  never between — it is never a lie mid-gesture. Free zoom stays available on a
  held modifier, and while off-detent **every affected readout is prefixed `>`**,
  which is exactly what a 2465 does with an uncalibrated scale factor.
- **The title block.** Bottom-right, ASME Y14.100's placement. Five rows,
  62px key / 136px value, 1px `line` rules: SCENE / STUDY / MESH / SOLVER / REV.
  9px tracked `ink-3` keys, 10px mono `ink-2` values, `—` where a field is empty.
  It lives in dead corner space and costs nothing at density.
- **The legend gains percentiles.** P10 and P90 ticks of the *actual* field
  distribution on the ramp bar, so the reader can see where the data lives inside
  the domain. This is the ramp doing more work, not decoration.
- **The probe.** White-hot 3.5px dot with a 1.5px `surface-void` halo, a 1px
  leader in the drafting idiom, and the value on an opaque `surface-void` plate.

**Cost.** One static canvas redrawn only on zoom. Zero per-element cost, zero
per-frame cost. The only risk is the detent: quantized zoom is unusual in CAD, so
ship free zoom as the default and the detent behind a modifier or a preference —
the readout is honest either way, which is the point of the `>`.

**Verdict: ship first.** It is the cheapest change that most alters the character
of the whole screen, and it fixes an actual gap (no spatial scale anywhere).

### 16.4 THE FILL — a run that says what it is doing

`research/design/playful/4-fill.png` · **`4-fill.gif`**

**Thesis:** a 30-second solve is currently a spinner. Replace it with five words
that light in order and a field that fills outward from its own boundary
conditions.

- **Annunciators.** Five equal cells across the card: `COMPILE  MESH  ASSEMBLE
  SOLVE  ADJOINT`. 9px mono at 0.13em. Unlit `#57555e` on a 2px `#2e2c34` top
  rule; live `ink` on a 2px `ink` rule; completed `ink-3` on `line-strong`. The
  legend words are always allocated and never reflow — this is the HP annunciator
  strip, and its whole virtue is that the layout does not move.
- **The mesh builds.** Bands below the front are drawn; bands above are 1.4px
  `surface-hover` wireframe. Bottom-up, over the real duration of the meshing
  stage.
- **The solve fills.** The same front, now separating solved field from unsolved
  wireframe, rising from the flux boundary. This is a *reading*: an iterative
  solve genuinely converges outward from its boundary conditions, so the wipe is
  not a loading bar wearing a costume.
- **The residual is a decade rule**, nine decades 1e−1 → 1e−10 with a major tick
  per decade, a 3px `ink-3` bar and a 14px `ink` head. The values are the solver's
  real iterates, replaced instantly. A percentage bar would be a fabrication;
  nobody knows what percent of a CG solve is done.
- **The RUN key.** No spinner anywhere. RUN becomes STOP, **drops the lime**
  (a running job is not the identity), and its background breathes between
  `surface-control` and `surface-hover` on a 1200 ms sine. That is the only moving
  thing on screen, which is exactly what §8 allows.
- **Reduced motion.** The annunciators still light in sequence — that is
  information, not motion. The front advances one stage-step at a time instead of
  continuously, and the key stops breathing.

**Cost.** The pipeline has to emit stage transitions and a residual stream. The
front is one extra comparison per band in the renderer.

**Verdict: ship.** It replaces the worst thirty seconds in the app, and the fill
is genuinely nice to watch without being a distraction, because it is monotone and
finite.

### 16.5 THE FLAG — staleness reported by presence

`research/design/playful/6-flag.png` · `6-flag-detail.png`

**Thesis:** an aircraft OFF flag is spring-loaded *in*; only a good signal holds
it out, so a dead instrument cannot hide its own flag. A greyed number can be
misread as a number. A covered one cannot.

- **The plate.** A stale or unavailable value is replaced, **in its own slot**, by
  an opaque 45° hatch: `#4a4750` over `#211f25` at 2.5px stripes, 1px
  `line-strong` border, exactly the width the value column occupies. Never grey
  the value. Never remove the row — a missing row says nothing at all.
- **UNCAL.** When any auto-coupled input is pinned by hand (mesh size, a bound, a
  time step), the *result region itself* carries an `UNCAL` annunciator — 9px
  tracked `ink` in a 1px `ink` box — and the viewport carries `MEAS UNCAL`
  top-right until it is re-coupled. The plot never silently lies.
- **Hatched bounds.** On a parameter rule, an unmanufacturable interval is drawn
  as 45° hatch clipped to the track, not as a shortened track. A barber pole says
  *you may not go here*; a missing region says nothing.
- **The trend stub.** From the live sample, a six-step projection of the current
  rate: 2px dashed `ink-3` at `[7, 6]`, with a 5px cross tick where it lands. It
  is **absent whenever the rate is below half the last displayed digit** — and
  that absence is the signal. A stub on screen means "still moving"; no stub means
  "converged", with no extra label.

**Cost.** Effectively none. Hatch is a `repeating-linear-gradient`; the trend stub
is two line segments.

**Verdict: ship, and ship the hatch plate even if nothing else in §16 lands.**
This is a correctness feature wearing a character costume, which is the best kind.

### 16.6 BACKWASH — the adjoint pass, drawn as order

`research/design/playful/2-backwash.png` · **`2-backwash.gif`**

**Thesis:** `cadjoint` = CAD + adjoint, and nothing in the UI expresses
reverse-mode. Direction C tried to draw the gradient as bars and failed for
competing with the field. The fix is to stop drawing the *quantity* and draw the
**direction**: one wavefront that runs backward along the chain, lighting each
station in reverse order, drawing no path at all.

- **Trigger.** A solve or optimizer step completes and an adjoint pass runs.
- **One wavefront position `u`**, 0 → 1.16 over **720 ms, linear**. A wavefront
  has a speed, not an easing.
- **Station brightness.** 0 before arrival; 1 at arrival; decaying linearly to the
  station's rest level over Δu = 0.16 (≈100 ms).
- **Stations, in adjoint order:** the objective value → the three ∂J/∂p rows →
  the boundary conditions in reverse listing order → the study scalars → the BC
  node sets in the viewport (white-hot 3.6px dots with a 1px `surface-void` halo)
  → the marked editor lines, **walked upward** → the three `Free(...)` literals.
- **The mark is one device everywhere:** 2px `ink` on the element's left edge
  (gutter-side in the editor) plus the row background at
  `rgba(237,236,239, 0.11 × brightness)`. §8 already licenses a 160 ms row flash
  to mark a change; this is that, sequenced.
- **Rest level 0 everywhere except free-parameter lines**, which keep a permanent
  0.34 gutter mark. That is §13's "take C's gutter mark, and only that", and it is
  the half of this idea that is durable.
- **Line budget: at most 14 marked lines**, and only lines carrying a free
  parameter or a differentiated call. §13's warning holds — ten marked lines is
  information, twenty is wallpaper.
- **The last beat is the product.** When the wave reaches the literals they
  rewrite, with a 160 ms `rgba(237,236,239,0.30)` row flash. The optimizer writing
  its own source is the thesis of the whole application; it should be the last
  thing the eye sees.
- **Reduced motion.** No wave. The end state in one frame: gutter marks on the
  free-parameter lines, gradients filled in, literals rewritten with a two-frame
  flash. Causality survives as an ordering in the DOM, not as a movement.

**Cost.** One CSS custom property per station per frame for 720 ms, once per
adjoint pass. Nothing scales with element count.

**Verdict: ship the gutter mark now, defer the wave.** Having watched the GIF
several times: **this is the idea that is more fun to describe than to watch.** As
a mechanism it is exactly right and the ordering genuinely reads backward, but at
1× it is twenty-odd small flashes in sequence, and it lands closer to "busy" than
to "wave". The permanent gutter mark on free-parameter lines carries most of the
meaning for none of the motion, and it is what I would build first. Revisit the
wave only once the mark has shipped and the marked-line set is proven to stay
under fourteen.

### 16.7 FIRST RUN — an empty app that is still an instrument

`research/design/playful/5-first-run.png` · `5-first-run-detail.png` · `5-mark.gif`

**Thesis:** an empty scope is not a blank screen; it is a ruled screen waiting for
a trace. Empty states are where an app shows it has a soul at zero cost to density,
and today they are plain sentences.

- **The empty viewport keeps its instrument.** Graticule drawn, gain readout
  present but prefixed `>` (there is nothing to calibrate against), title block
  present with `—` in every row it cannot fill. The legend bar becomes a 45° hatch
  at `#26242b`/`#1b1a1f` with `—` endpoints — the same "no data" texture as §16.5,
  so the vocabulary is one thing and not two.
- **Empty-state copy has a form.** Name the missing precondition **in the language
  of the program** (anything that appears in the source is `mono`), then exactly
  one clause of dry aside, then the action as a 9px tracked mono line. Three that
  survived reading them back:

  > **Studies · none** — A study needs a mesh and at least one `Dirichlet`.
  > *Without one the stiffness matrix is singular, and CalculiX will point that out
  > less politely than this panel does.*

  > **Optimize · none** — Three parameters are declared `Free` and nothing
  > differentiates them yet. *Name an objective and the tape starts recording.*

  > **No geometry** — `scene.py` declares three free parameters and no solid. *Add
  > a body, or press run and watch nothing happen very accurately.*

  The rule that keeps this from becoming cute: **the aside must be true and must
  teach something.** "Singular stiffness matrix" is a real failure a user will
  meet. Delete any sentence that is only a joke.
- **The mark.** On session wake, the `cj` mark exercises its own glyph set once —
  seven glyphs at 80 ms, 560 ms total — then settles. This is a nixie
  anti-poisoning routine, which is maintenance rather than ornament, and that is
  the only way ornament survives a system this strict. **Once per session**, never
  on tab or mode switches, and not at all under reduced motion. If it ever feels
  like a logo animation, it is too long; the whole budget is 560 ms.

**Cost.** None. This is HTML and one 560 ms text swap.

**Verdict: ship. Cheapest personality in the document.**

### 16.8 The settle law — where ballistics are allowed

One cross-cutting rule, because three of the six ideas want to move a mark toward
a value:

1. **Needles, bars and rules may settle. Digits may not.** A bar is not spelling
   anything, so an intermediate position is an approximation; a digit mid-roll is
   a wrong number.
2. **Ballistics are VU:** 300 ms to 99% of the final value, with an overshoot of
   1.0–1.5%; τ ≈ 160 ms to 63%. Use it for the residual head, the travel rule's
   live tick when a value is dragged, and the legend's percentile ticks.
3. **Overshoot budget = half the displayed precision.** This is ANSI C39.1's
   definition of *dead-beat* — critical damping is where overshoot "does not
   exceed an amount equal to one half the rated accuracy of the instrument". It
   makes overshoot **computable** instead of a taste call: a bar under a value
   shown to two decimals may overshoot by 0.005 of its range and no more.
4. **Peak marks are asymmetric (PPM):** attack ≈10 ms so a single bad Newton
   iteration is never missed, decay ≈8.6 dB/s so the excursion stays readable for
   two or three seconds. Applies to max stress, max residual, worst jacobian.
5. **One easing family still.** `cubic-bezier(0.2, 0, 0, 1)` for anything
   arriving; the second-order settle above is not a second easing family, it is a
   physical model, and it is confined to the four marks named in (2) and (4).

And one rule inherited from Braun, restated in this language's terms:

> **One element per view may break the system, and it is always either the thing
> you act on or the thing that is moving right now.** The ET 66 spends its single
> yellow key on `=`. Since we have no hue to spend, the break is spent in
> **luminance, length, plane or shape — one of those four, once per view.** In
> §16.2 it is the 30px objective; in §16.4 it is the breathing key; in §16.5 it is
> the hatch plate. If a view has two breaks, one of them is wrong.

### 16.9 What was looked at

Primary sources, since the point of this section is to borrow *mechanisms* rather
than vibes:

- **Tektronix 2213 operator's manual** — "internally marked on the faceplate…
  eight vertical and ten horizontal major divisions. Each major division is
  divided into five subdivisions"; the BEAM FIND momentary control; "to obtain a
  calibrated deflection factor, the VOLTS/DIV variable control must be in detent."
  `users.physics.unc.edu/~sean/Phys351/techresource/docs/2213%20User%20Manual.pdf`
- **Tektronix 475A data sheet** — "8 × 10 cm display. Horizontal and vertical
  centerlines further marked in 0.2 cm increments" — i.e. the minor ticks live
  **only** on the centre axes. `nscainc.com/wp-content/uploads/pdf/T_475A.pdf`
- **Tektronix 2465 manual** — the CRT readout prefixes an uncalibrated scale
  factor with `>`. `manualslib.com/manual/1400395/Tektronix-2465.html?page=33`
- **HP 5245L manual (1963)** — "annunciator" as a spec-table term: "total width
  of 8 digit display including illuminated units annunciator and auto-positioned
  decimal point indication". `kennethkuhn.com/hpmuseum/scans/hp5245l.pdf`
- **HP 3478A manual** — "The 12 character alphanumeric display includes 12
  dedicated annunciators"; "the right most digit on the display blinks (showing
  that the display is updated)."
  `manualslib.com/manual/1016815/Hp-3478a.html?page=21`
- **HP 8566B operating manual** — "If the amplitude or frequency becomes
  uncalibrated, 'MEAS UNCAL' appears in the right-hand side of the graticule";
  graticule/annotation/trace as independently blankable layers.
  `research.physics.illinois.edu/bezryadin/labprotocol/8566BOperating_ProgrammingManual.pdf`
- **Allen Inhelder, HP Measure, May 1965** — "Aesthetics are important to us, but
  everything we do must be tempered by the practical aspects of the problem."
  `hparchive.com/measure_magazine/HP-Measure-1965-05.pdf`
- **V&A, Braun ET 66** — the eye is "drawn to the yellow 'equals' button, the most
  frequently used function"; number keys polished, function keys matt, so the
  split survives in monochrome. `collections.vam.ac.uk/item/O1360553/`
- **Vitsœ, Rams' ten principles** — principle 8, "thorough down to the last
  detail… nothing must be arbitrary or left to chance", is what licenses spending
  the character budget on precision. `vitsoe.com/us/about/good-design`
- **FAA-H-8083-15B** — the OFF flag: "the device that indicates a usable or an
  unreliable signal may be an 'OFF' flag. It retracts from view when signal
  strength is sufficient." **AC 25-11B §6.2.1.7:** "Failure flags should be
  presented in the location of the information they reference or replace."
  `faa.gov/sites/faa.gov/files/regulations_policies/handbooks_manuals/aviation/FAA-H-8083-15B.pdf`
- **AC 23.1311-1C §17.7(b)** — "Incorporate a red arc or red barber pole extending
  from V_NE or V_MO upward to the end of the airspeed tape" — i.e. 45° hatch is
  the standard way to draw a forbidden region.
  `faa.gov/documentLibrary/media/Advisory_Circular/AC_23_1311-1C.pdf`
- **Garmin G1000 Pilot's Guide** — the trend vector: "the end of the trend vector
  displays approximately what airspeed will be reached in six seconds if the
  current rate of acceleration is maintained. **The trend vector is absent if the
  speed remains constant.**" `wayman.edu/files/G1000_CessnaNavIII_PilotsGuide.pdf`
- **Nixie cathode poisoning** — the prevention duty cycle: "for every 60 seconds
  when the tube is on, exercise every other digit for 0.2 s", implemented in
  practice as a "slot machine" routine.
  `docs.daliborfarny.com/nixie-tubes/1/en/topic/cathode-poisoning-prevention-routine`
- **ASME Y14.5-2018 §4.3.2** — "a dimension shall be expressed to the same number
  of decimal places as its tolerance." Decimal places *are* the tolerance.
  **ASME Y14.100-2017** for the lower-right title block.
- **IEC 60268-17 / ITU-R BS.2054-4** — VU ballistics: 99% of full-scale deflection
  in 300 ms, overshoot 1.0–1.5%, 63% at ≈160 ms. **IEC 60268-10** for PPM
  asymmetry (attack τ 1.7 ms, fallback 24 dB in 2.8 s ≈ 8.6 dB/s).
  `itu.int/dms_pub/itu-r/opb/rep/R-REP-BS.2054-4-2014-PDF-E.pdf`
- **ANSI C39.1-1981, via Simpson's glossary** — dead-beat: critical damping is
  where "overshoot is present but does not exceed an amount equal to one half the
  rated accuracy of the instrument."
  `simpsonelectric.com/technical-support/glossary-of-terms/`
- **Grether (1949) / NRL (1965) on the three-pointer altimeter** — over 7 seconds
  to read and >11% errors of 1 000 ft or more; misread roughly eight times more
  often than better designs. The standing argument against any compact
  multi-ring convergence dial.

Three things the earlier framing assumed that the sources **do not support**, and
which were dropped: there is no published HP standard assigning colours to
functional groups (the only consistent rule is blue = the shift layer, and
grouping is done by layout and named zones — which is a gift to an achromatic
system); no primary HP source names Bauhaus, Braun or Rams as an influence; and
there is no "dot at 12" as a named Braun device — the documented differentiator on
Lubs' dials is tick **length**, not a dot.

### 16.10 Rejected, with the measurement

`research/design/playful/7-rejected.png`

**A · Motion blur on the morphing geometry.** Built exactly as proposed: an
accumulation buffer of five filled steps at falling alpha under the live frame.
Measured against the true frame, pixel for pixel, over the part:

```
mean ΔE76  6.2      p95 11.0      max 21.9
97 % of on-part pixels are above JND
and it paints 54 % of the part's area again OUTSIDE the true silhouette
```

The smear averages five different viridis samples into one pixel, so the band is a
colour **no node ever had**. Colour is a reading; this one reads a temperature
that does not exist. **Rejected on correctness, not on taste** — and note that
ghosting (§16.2) delivers the same sense of travel with zero colour error, because
a 1px white outline is not a field value.

**B · Odometer / split-flap digits on a live value.** There is a real mechanism to
copy — a Geneva drive rolls only the column that is carrying and holds the higher
columns in dwell, which is why a real odometer shows one half-rolled digit and not
five. But every frame of the roll shows a value that is partly one number and
partly another, on a screen where people read tolerances. **§8's "numbers never
animate" is right and stays.** The ballistics belong on a needle or a bar, where
nothing is being spelled — which is what §16.8 does with them.

**C · The 0 / 10 / 90 / 100 % rise-time markers** on the left edge of the
graticule. Built, rendered, removed. On a scope they are a *normalisation ritual*
for a waveform — you position the trace so its zero touches 0% and its top touches
100%, then measure between 10% and 90%. There is no analogue for a 3D field on a
spatial axis, so on our graticule they are four labelled ticks that mean nothing.
Cited here only so nobody adds them back for the look of the thing. The idea does
have a home — as a "normalise to frame" gesture on the *objective trace*, which
would make two runs comparable by eye — and that is worth its own pass.

**D · Gradient magnitude as bars in the panel.** Already rejected in §12/C for
competing with the field legend; nothing found here rehabilitates it. §16.6 shows
what the idea should have been: draw the **direction**, in achromatic marks, and
let the magnitude be a signed number in mono.

### 16.11 Recommendation, ordered by delight per unit of cost

1. **THE GRATICULE + TITLE BLOCK** (§16.3). Static, zero per-frame cost, fixes a
   real gap — the viewport has no scale on it today — and it is the single change
   that most alters what the whole app feels like. Ship the graticule, the gain
   readout and the title block together; ship the detent behind a modifier.
2. **FIRST RUN, EMPTY STATES AND THE MARK** (§16.7). An afternoon. Personality is
   cheapest where there is no data to get in the way of it.
3. **THE FLAG** (§16.5). Nearly free, and the hatch plate is a correctness fix
   first and a character move second. Ship the plate even if nothing else lands.
4. **THE TAPE** (§16.2). The largest payoff in the document and the largest
   upstream cost — the trajectory must be recorded and kept. Worth it: this is the
   product's argument, animated.
5. **THE FILL** (§16.4). Needs stage events and a residual stream from the
   pipeline. Replaces the worst thirty seconds in the app.
6. **BACKWASH** (§16.6) — **the permanent gutter mark only.** Defer the wave until
   the mark has shipped; at 1× the wave reads busier than it reads backward, and
   the mark carries most of the meaning for none of the motion.

Two rules join §10's refusals:

16. **No motion that changes a colour that is a reading.** Trails, ghosts and
    fronts are achromatic or they are geometry-clipped; a field pixel is either
    its true value or it is not painted.
17. **No indeterminate progress.** A run reports the stage it is in, in words,
    and a quantity that is genuinely known (a residual, an element count, a step
    index). There is no spinner and no percentage of an unknowable whole.

And one joins §11's lint checks:

- `replayable` — any element whose text changes on more than two consecutive
  animation frames must be able to name the recorded step its value came from.
  A number that cannot is a tween.

### 16.12 Files

| what | where |
| --- | --- |
| the tape, still | `research/design/playful/1-tape.png` |
| **the tape, playing** | `research/design/playful/1-tape.gif` |
| the adjoint pass, still | `research/design/playful/2-backwash.png` |
| **the adjoint pass, running backward** | `research/design/playful/2-backwash.gif` |
| the graticule and title block | `research/design/playful/3-graticule.png` |
| **the zoom detent, 20 → 10 → 5 → 10 → 20 mm/div** | `research/design/playful/3-detent.gif` |
| a solve, mid-fill | `research/design/playful/4-fill.png` |
| **the whole run: compile → mesh → assemble → solve → adjoint** | `research/design/playful/4-fill.gif` |
| first run | `research/design/playful/5-first-run.png` |
| first run, viewport detail | `research/design/playful/5-first-run-detail.png` |
| **the mark's wake cycle** (shown twice; it runs once per session) | `research/design/playful/5-mark.gif` |
| flags, UNCAL and the trend stub | `research/design/playful/6-flag.png` |
| the same, panel detail | `research/design/playful/6-flag-detail.png` |
| **what was rejected, and the numbers** | `research/design/playful/7-rejected.png` |

Every screen above is the same 1500 × 940 layout as §12's three directions, on
§14.8's PLATE-WELL surfaces, rendered through Chromium at DPR 2. The mockups and
the measurement scripts live in the scratchpad workspace; the ΔE figures in
§16.10 come from `measure-blur.html` plus a Lab diff over the two renders.
