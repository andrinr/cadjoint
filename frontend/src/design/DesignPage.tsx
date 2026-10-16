/**
 * The design language, as a page of the playground.
 *
 * It states the axioms, derives the visual and functional rules from them,
 * and shows every rule as an example drawn with the tokens and primitives
 * the rule defines. The page obeys what it says: if it needs a value the
 * tokens do not have, the tokens are wrong.
 */

import { For, createSignal, onCleanup, onMount, type JSX } from "solid-js";
import {
  BoxIcon,
  CodeIcon,
  CursorIcon,
  CylinderIcon,
  DisplayIcon,
  DistanceIcon,
  ExtrudeIcon,
  HorizontalIcon,
  LoftIcon,
  ModelModeIcon,
  MoveIcon,
  PlayIcon,
  PolygonIcon,
  ResetIcon,
  RevolveIcon,
  RotateIcon,
  SimulateModeIcon,
  SketchModeIcon,
  SphereIcon,
  TrashIcon,
  VerticalIcon,
} from "../components/icons";
import { SHORTCUT_GROUPS } from "../shortcuts";
import { BC_TYPE_COLORS, PROPOSAL_COLOR, cssColor, fieldRampCss, qualityRampCss } from "../simColors";
import {
  CHROME,
  COLOR_ROLES,
  LAYOUT,
  TEXT_ROLES,
  type TextRole,
  DURATIONS,
  GRID_CONTROL_HEIGHTS,
  UNIT,
  type ChromeToken,
} from "../tokens";
import { sdfRampCss } from "../viewer/sdfRamp";
import {
  DragDemo,
  DurationDemo,
  EscapeDemo,
  GreyscaleDemo,
  InputDemo,
  JobDemo,
  LiteralDemo,
  ModeDemo,
  SelectionDemo,
  TweenDemo,
  UndoDemo,
} from "./demos";
import { Cross, Plate, Ratio, Rule, Section, Swatch, v } from "./kit";
import { DeskMap } from "./DeskMap";
import { FLYOUT_VARIANTS, FlyoutSpecimen } from "./FlyoutVariants";
import { Variations } from "./Variations";
import { AnchorsDrawing, DeskDrawing, LevelsDrawing, PanelAnatomy } from "./LayoutDrawings";
import { SelectSpecimen } from "./demos";
import { FeatureIndex, ProposedMenuBar } from "./FeatureIndex";
import { ViewportSpecimen } from "./ViewportSpecimen";

const SECTIONS = [
  { id: "system", num: "1", title: "System" },
  { id: "shape", num: "2", title: "Building blocks" },
  { id: "primitives", num: "3", title: "Primitives" },
  { id: "layout-primitives", num: "4", title: "Layout" },
  { id: "components", num: "5", title: "Components" },
  { id: "colour", num: "6", title: "Colour" },
  { id: "type", num: "7", title: "Type" },
  { id: "language", num: "8", title: "Language" },
  { id: "space", num: "9", title: "Space and size" },
  { id: "icons", num: "10", title: "Iconography" },
  { id: "motion", num: "11", title: "Motion" },
  { id: "viewport", num: "12", title: "Viewport" },
  { id: "layout", num: "13", title: "Where things are" },
  { id: "features", num: "14", title: "Every feature" },
  { id: "behaviour", num: "15", title: "Behaviour" },
  { id: "axioms", num: "16", title: "Principles" },
  { id: "changes", num: "17", title: "Record" },
];

/**
 * Values the playground ships today that the rules retire. They appear only
 * inside "Now" plates, as specimens of what is being replaced.
 */
const BEFORE = {
  edge: "#6a7f1a",
  handle: "#db3c1c",
  edgeSelected: "#915b16",
  plane: "#33608f",
  hoverTint: "#ff9e40",
  className: CHROME.danger,
  string: CHROME.ok,
  number: CHROME.warn,
  keyword: "#0065b4",
  cubeX: "#d21f2b",
  gizmoX: "#eb273f",
  cubeY: "#1f8a2e",
  gizmoY: "#1e8d39",
  cubeZ: "#1f5fd2",
  gizmoZ: "#3173f5",
} as const;

function Nav() {
  const [current, setCurrent] = createSignal("axioms");
  onMount(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((entry) => entry.isIntersecting);
        if (visible.length) setCurrent(visible[0].target.id);
      },
      { rootMargin: "-40% 0px -55% 0px" },
    );
    for (const section of SECTIONS) {
      const node = document.getElementById(section.id);
      if (node) observer.observe(node);
    }
    onCleanup(() => observer.disconnect());
  });
  return (
    <nav class="dl-nav" aria-label="Sections">
      <ol>
        <For each={SECTIONS}>
          {(section) => (
            <li>
              <a href={`#${section.id}`} class={current() === section.id ? "current" : ""}>
                <small>{section.num}</small>
                {section.title}
              </a>
            </li>
          )}
        </For>
      </ol>
    </nav>
  );
}

/** A tiny panel in the shipped language: kicker, title, rows. */
function MiniPanel(props: { kicker: string; title: string; children: JSX.Element; tabs?: string[] }) {
  return (
    <div class="dl-panel">
      <nav>
        <For each={props.tabs ?? [props.title]}>
          {(tab, index) => (
            <button type="button" class={index() === 0 ? "active" : ""}>
              {tab}
            </button>
          )}
        </For>
      </nav>
      <div>
        <div class="dl-panel-head">
          <span class="dl-kicker">{props.kicker}</span>
          <strong>{props.title}</strong>
        </div>
        {props.children}
      </div>
    </div>
  );
}

/** A12/K8: a tool rail with its middle group open, attached or not. */
function RailSpecimen(props: { attached?: boolean }) {
  return (
    <div class="dl-rail-demo">
      <span class="dl-rail-cell">
        <CursorIcon size={16} />
      </span>
      <span class={props.attached ? "dl-rail-group attached" : "dl-rail-group"}>
        <span class="dl-rail-cell open">
          <BoxIcon size={16} />
        </span>
        <span class={props.attached ? "dl-rail-flyout" : "dl-rail-flyout detached"}>
          <span class="dl-rail-cell">
            <BoxIcon size={16} />
          </span>
          <span class="dl-rail-cell">
            <SphereIcon size={16} />
          </span>
          <span class="dl-rail-cell">
            <CylinderIcon size={16} />
          </span>
        </span>
      </span>
      <span class="dl-rail-cell">
        <MoveIcon size={16} />
      </span>
    </div>
  );
}

/**
 * The system: every size derived from the unit and the line box. tokens.ts
 * carries the same values; this table is the page's reading of them.
 */
const SYSTEM = {
  unit: 4,
  line: 16,
  heights: [
    { name: "tag", px: 16, formula: "L", job: "kind chips, marks that are not pressed" },
    { name: "cell", px: 24, formula: "L + 2 × gap", job: "the one interactive height: buttons, fields, menu items, rows, chips" },
    { name: "bar", px: 32, formula: "L + 2 × inset", job: "strips: tab strips, the tool rail, the viewport controls" },
    { name: "toolbar", px: 40, formula: "bar + 2 × gap", job: "the app toolbar, which holds bars" },
  ],
  distances: [
    { name: "gap", px: 4, formula: "u", job: "between cells of one group; text to its cell's edge, vertically" },
    { name: "inset", px: 8, formula: "2u", job: "inside a sheet; a cell's text to its edge, horizontally" },
    { name: "section", px: 16, formula: "4u = L", job: "between sections; the viewport's furniture from its edges" },
  ],
  type: [
    { role: "caps", size: 11, weight: 600, job: "every label of the instrument: tabs, buttons, column heads, kickers" },
    { role: "body", size: 12, weight: 400, job: "names, values 2700.0, units, menu items, the editor" },
    { role: "title", size: 15, weight: 600, job: "a panel's title, on a 20px line box" },
    { role: "prose", size: 12, weight: 400, job: "sentences, in sans, on a 20px line box" },
  ],
} as const;

function System() {
  return (
    <Section
      id="system"
      num="1"
      title="System"
      intro="One unit and one line box, and every other size is derived from them. Nothing below is chosen per component: a height is a line box plus a distance on each side, a distance is a multiple of the unit, a text size is whatever sits on the line box. Adopted on 17 Oct 2026, with the choices from the Variations drawer: IBM Plex Mono, amber, warm ink, light lines."
    >
      <Rule
        id="Y1"
        why="One number to remember (4) and one to see (16). A layout built from them lines up without being told to, because there is nothing else it could be."
        single
        examples={
          <Plate label="the unit and the line box">
            <div class="dl-sys-row">
              <div class="dl-sys-unit" />
              <span class="t-body">u = 4px</span>
              <div class="dl-sys-line">
                <span class="t-body">a line of text, in its 16px box</span>
              </div>
              <span class="t-body">L = 4u = 16px</span>
            </div>
          </Plate>
        }
      >
        <b>u = 4px</b>, the unit. <b>L = 16px</b>, the line box: every single line of text sits in
        one, and an icon is one line box square.
      </Rule>
      <Rule
        id="Y2"
        why="Four heights, each a line box plus padding of one distance: so a cell, a bar and a toolbar hold the same text on the same baseline, and a bar holds a cell with a gap to spare."
        single
        examples={
          <Plate label="the heights, built from the line box" ground>
            <div class="dl-sys-heights">
              <For each={SYSTEM.heights}>
                {(h) => (
                  <div class="dl-sys-height">
                    <div class="dl-sys-box" style={{ height: `${h.px}px` }}>
                      <span class="dl-sys-text" />
                    </div>
                    <code>{h.px}</code>
                    <span class="t-body">{h.name}</span>
                    <span class="t-body" style={{ color: "var(--ink-3)" }}>{h.formula}</span>
                    <span class="t-prose">{h.job}</span>
                  </div>
                )}
              </For>
            </div>
          </Plate>
        }
      >
        <b>tag</b> 16 · <b>cell</b> 24 · <b>bar</b> 32 · <b>toolbar</b> 40. Anything clickable is a
        cell or larger (24 is the WCAG 2.2 minimum target).
      </Rule>
      <Rule
        id="Y3"
        why="Three distances that double: the gap between things in a group, the inset inside a sheet, the section between groups. A 2px nudge and a 12px stack were the old system's hesitations."
        single
        examples={
          <Plate label="the three distances">
            <div class="dl-ruler">
              <For each={SYSTEM.distances}>
                {(d) => (
                  <div>
                    <code>{d.name}</code>
                    <b class="t-body">{d.px}px</b>
                    <span class="dl-sys-dist">
                      <i style={{ width: `${d.px * 4}px` }} />
                      <span class="t-prose">
                        {d.formula} · {d.job}
                      </span>
                    </span>
                  </div>
                )}
              </For>
            </div>
          </Plate>
        }
      >
        <b>gap</b> 4 · <b>inset</b> 8 · <b>section</b> 16.
      </Rule>
      <Rule
        id="Y4"
        why="Three sizes, on the line box (11 and 12) or the one taller box (15 on 20). A value and a name are the same role: a row of label, value and unit is told apart by caps and ink, not by size."
        single
        examples={
          <Plate label="four roles, three sizes, on the line box">
            <div class="dl-sys-type">
              <For each={SYSTEM.type}>
                {(t) => (
                  <div>
                    <code>{t.role}</code>
                    <code>
                      {t.size} · {t.weight}
                    </code>
                    <span
                      class={`dl-sys-sample ${t.role}`}
                      style={{
                        "font-size": `${t.size}px`,
                        "font-weight": String(t.weight),
                      }}
                    >
                      {t.job}
                    </span>
                  </div>
                )}
              </For>
            </div>
          </Plate>
        }
      >
        <b>caps</b> 11 · <b>body</b> 12 · <b>title</b> 15 · <b>prose</b> 12 sans.
      </Rule>
      <Rule
        id="Y5"
        why="The whole system on one object: tab strip a bar, rows cells, the head a title line box, sections a section apart, an inset all round. Every edge falls on the 4px grid behind it."
        single
        examples={
          <Plate label="a panel, built from nothing but the module">
            <div class="dl-unit-grid" style={{ padding: "var(--section)" }}>
              <div class="dl-sys-panel">
                <div class="dl-sys-tabs">
                  <span class="active">studies</span>
                  <span>meshes</span>
                </div>
                <div class="dl-sys-body">
                  <div class="dl-sys-head">
                    <span class="dl-sys-kind">fem</span>
                    <span class="dl-sys-title">studies</span>
                  </div>
                  <div class="dl-sys-rows">
                    <div>
                      <span class="dl-sys-label">mesh</span>
                      <span class="dl-sys-val">sink-mesh</span>
                      <span />
                    </div>
                    <div>
                      <span class="dl-sys-label">conductivity</span>
                      <span class="dl-sys-val num">401.0</span>
                      <span class="dl-sys-unit-t">W/mK</span>
                    </div>
                    <div>
                      <span class="dl-sys-label">source</span>
                      <span class="dl-sys-val num">12.5</span>
                      <span class="dl-sys-unit-t">W</span>
                    </div>
                  </div>
                  <div class="dl-sys-actions">
                    <span class="dl-sys-cell">mesh</span>
                    <span class="dl-sys-cell primary">solve</span>
                  </div>
                </div>
              </div>
            </div>
          </Plate>
        }
      >
        A panel: a <b>bar</b> of tabs, an <b>inset</b>, a head on a title line box, a{" "}
        <b>section</b>, <b>cell</b> rows, a <b>section</b>, a row of cells a <b>gap</b> apart.
      </Rule>
    </Section>
  );
}

function Axioms() {
  return (
    <Section
      id="axioms"
      num="16"
      title="Principles"
      intro={
        <>
          Thirteen sentences: the reasons behind the system. Every rule on this page names the
          principle it comes from, and when two rules conflict, the one from the earlier principle
          wins. A1–A9 are about how things look, A2 and A10–A13 about how they behave.
        </>
      }
    >
      <Rule
        id="A1"
        why="So the chrome is quiet: paper, ink and rules. Saturated colour is spent on the geometry and on readings."
        examples={
          <>
            <Plate kind="do" label="the reading is the loudest thing" ground>
              <MiniPanel kicker="fem" title="Results">
                <div class="dl-legend">
                  <b>temperature</b>
                  <span>293.2 K</span>
                  <i style={{ background: fieldRampCss() }} />
                  <span>412.8 K</span>
                </div>
                <div class="dl-rows">
                  <div>
                    <span>peak</span>
                    <b>412.80</b>
                    <small>K</small>
                  </div>
                  <div>
                    <span>mean</span>
                    <b>338.07</b>
                    <small>K</small>
                  </div>
                </div>
              </MiniPanel>
            </Plate>
            <Plate kind="dont" label="chrome competes with the reading" ground>
              <div class="dl-panel">
                <nav>
                  <button type="button" class="active">Results</button>
                </nav>
                <div>
                  <div class="dl-panel-head">
                    <span class="dl-before-chip" style={{ background: v("accent"), color: v("ink-on-accent") }}>
                      fem
                    </span>
                    <strong>Results</strong>
                  </div>
                  <div class="dl-legend">
                    <b style={{ color: v("accent-press") }}>temperature</b>
                    <span>293.2 K</span>
                    <i style={{ background: fieldRampCss() }} />
                    <span>412.8 K</span>
                  </div>
                  <div class="dl-rows">
                    <div>
                      <span style={{ color: "#0065b4" }}>peak</span>
                      <b style={{ color: v("ok") }}>412.80</b>
                      <small>K</small>
                    </div>
                    <div>
                      <span style={{ color: "#0065b4" }}>mean</span>
                      <b style={{ color: v("warn") }}>338.07</b>
                      <small>K</small>
                    </div>
                  </div>
                </div>
              </div>
            </Plate>
          </>
        }
      >
        <b>The part and its numbers are the subject.</b> The interface is the sheet they are
        printed on. Nothing in the chrome is more salient than the geometry or a measured value.
      </Rule>

      <Rule
        id="A2"
        why="So every panel is an editor of literals, and the editor is never out of the loop. Type a value and press Enter."
        single
        examples={
          <Plate kind="try" label="a panel field writes its literal">
            <LiteralDemo />
          </Plate>
        }
      >
        <b>The program is the truth.</b> Every edit made anywhere is an edit to the source text,
        and the screen is a rendering of that text. The only state the program does not hold is
        the view: camera, layout, selection.
      </Rule>

      <Rule
        id="A3"
        why="So a reader can trust a hue: if it is coloured, it means something, and it means one thing. The table is in §6."
        examples={
          <>
            <Plate kind="do" label="hue only where a role is">
              <pre class="dl-code">
                <div>
                  <span>116</span>
                  <span>
                    base_l = <span class="kw">Vector2</span>(value=[-0.9, 0.0], <span class="free">free=True</span>,
                    name=<span class="str">"base_l"</span>)
                  </span>
                </div>
                <div>
                  <span>117</span>
                  <span class="com"># the deck stays level</span>
                </div>
                <div>
                  <span>118</span>
                  <span>
                    <span class="kw">from</span> cadjoint <span class="kw">import</span> <span class="err">Solidd</span>
                  </span>
                </div>
              </pre>
            </Plate>
            <Plate kind="dont" label="hue as decoration">
              <pre class="dl-code">
                <div>
                  <span>116</span>
                  <span>
                    base_l = <span style={{ color: BEFORE.className }}>Vector2</span>(value=[
                    <span style={{ color: BEFORE.number }}>-0.9</span>, <span style={{ color: BEFORE.number }}>0.0</span>], free=
                    <span style={{ color: BEFORE.number }}>True</span>, name=<span style={{ color: BEFORE.string }}>"base_l"</span>)
                  </span>
                </div>
                <div>
                  <span>117</span>
                  <span class="com"># the deck stays level</span>
                </div>
                <div>
                  <span>118</span>
                  <span>
                    <span style={{ color: BEFORE.keyword }}>from</span> cadjoint{" "}
                    <span style={{ color: BEFORE.keyword }}>import</span> <span style={{ color: BEFORE.className }}>Solidd</span>
                  </span>
                </div>
              </pre>
            </Plate>
          </>
        }
      >
        <b>Hue is meaning.</b> Every hue on screen belongs to exactly one role in a closed table.
        Anything without a role is drawn without hue: ink on paper.
      </Rule>
      <p class="dl-note">
        In the second plate a class name is in the error red and a typo is not marked at all; in
        the first, the only red on screen is the typo.
      </p>

      <Rule
        id="A4"
        why="So colour-blind readers and greyscale screenshots lose nothing. Toggle the plate to greyscale: every state still reads."
        single
        examples={
          <Plate kind="try" label="statuses in greyscale">
            <GreyscaleDemo>
              <div class="dl-stack">
                <div class="dl-row-of">
                  <span class="dl-status">
                    <i />
                    compiled · rev 14
                  </span>
                  <span class="dl-status stale">
                    <i />
                    out of date
                  </span>
                  <span class="dl-badge free">17 free</span>
                  <span class="dl-chip violated">
                    distance · P1–P2 = 1.8 ≠ 1.74
                    <button type="button">
                      <Cross />
                    </button>
                  </span>
                </div>
                <div class="dl-notice error">
                  <b>Error</b>
                  <p>
                    Line 118: <code>Solidd</code> is not defined. Did you mean <code>Solid</code>?
                  </p>
                  <span />
                </div>
              </div>
            </GreyscaleDemo>
          </Plate>
        }
      >
        <b>Never by hue alone.</b> Every role also has a carrier that survives greyscale: a word, a
        shape, a position or a weight.
      </Rule>

      <Rule
        id="A5"
        why="So legibility is asserted, not judged. Every ratio on this page is computed from the shipped tokens."
        single
        examples={
          <Plate label="the bars, measured on the two sheets">
            <table class="dl-table">
              <thead>
                <tr>
                  <th>pair</th>
                  <th>owes</th>
                  <th class="num">on ground</th>
                  <th class="num">on sheet</th>
                </tr>
              </thead>
              <tbody>
                <For
                  each={[
                    { fg: "ink-3", need: 4.5 as number | [number, number], what: "text" },
                    { fg: "rule-strong", need: 3, what: "a mark that means something" },
                    { fg: "rule", need: [1.4, 2.8] as [number, number], what: "furniture: visible, not competing" },
                  ]}
                >
                  {(row) => (
                    <tr>
                      <td>
                        <i class="dl-dot" style={{ background: v(row.fg) }} />
                        {row.fg}
                      </td>
                      <td class="prose">{row.what}</td>
                      <td class="num">
                        <Ratio fg={row.fg as ChromeToken} bg="surface-base" need={row.need} />
                      </td>
                      <td class="num">
                        <Ratio fg={row.fg as ChromeToken} bg="surface-float" need={row.need} />
                      </td>
                    </tr>
                  )}
                </For>
              </tbody>
            </table>
          </Plate>
        }
      >
        <b>Contrast is a number.</b> Text clears 4.5:1 on its ground. A mark that carries meaning
        clears 3:1. Furniture sits in a band below that, 1.4–2.8:1.
      </Rule>

      <Rule
        id="A6"
        why={`So everything lines up without being told to. The grid behind this plate is the unit, ${UNIT}px.`}
        single
        examples={
          <Plate label="a row on the unit grid">
            <div class="dl-unit-grid" style={{ padding: "var(--section)" }}>
              <div class="dl-row-of">
                <button type="button" class="dl-btn">Button · 28</button>
                <input class="dl-input" style={{ width: "calc(var(--unit) * 20)" }} value="0.75" />
                <span class="dl-tag">tag · 20</span>
                <button type="button" class="dl-btn h-sm">Small · 24</button>
              </div>
            </div>
          </Plate>
        }
      >
        <b>One unit.</b> Every length is a multiple of the unit, 4px. Half a unit is allowed inside
        a control. The one shorter length is the hairline, 1px.
      </Rule>

      <Rule
        id="A7"
        why="So a dense screen stays calm: a line costs less attention than a box, and paper has no depth to fake."
        examples={
          <>
            <Plate kind="do" label="rules separate, edges group">
              <div class="dl-rows">
                <div>
                  <span>mesh</span>
                  <b>sink-mesh</b>
                  <small />
                </div>
                <div>
                  <span>cells</span>
                  <b>48 211</b>
                  <small />
                </div>
                <div>
                  <span>min jacobian</span>
                  <b>0.31</b>
                  <small />
                </div>
              </div>
            </Plate>
            <Plate kind="dont" label="boxes, shadows, radii">
              <div class="dl-stack">
                <For each={[["mesh", "sink-mesh"], ["cells", "48 211"], ["min jacobian", "0.31"]]}>
                  {([label, value]) => (
                    <div
                      style={{
                        display: "flex",
                        "justify-content": "space-between",
                        padding: "var(--inset) var(--inset)",
                        "border-radius": "8px",
                        background: v("surface-float"),
                        "box-shadow": "0 2px 6px rgba(0,0,0,.18)",
                      }}
                    >
                      <span style={{ color: v("ink-3") }}>{label}</span>
                      <b>{value}</b>
                    </div>
                  )}
                </For>
              </div>
            </Plate>
          </>
        }
      >
        <b>Structure is drawn, not stacked.</b> Things are separated by rules and grouped by shared
        edges. No shadow, no radius, no translucency behind text.
      </Rule>

      <Rule
        id="A8"
        why="So a glance tells you who is speaking: mono is the instrument, sans is a person."
        single
        examples={
          <Plate label="two voices in one row">
            <div class="dl-field">
              <label>conductivity</label>
              <input class="dl-input" value="401" />
              <small>W/mK</small>
            </div>
            <p class="dl-note">Copper at room temperature. The solver reads it per cell from the material field.</p>
          </Plate>
        }
      >
        <b>Two voices.</b> What the program says (names, values, units, the instrument's labels) is
        mono, and a string from the program is set exactly as written. What a person says
        (sentences) is sans. Capitals are for labels of a word or two, never for a list or a
        sentence.
      </Rule>

      <Rule id="A9" why="So movement carries information. The demos are in §11.">
        <b>Motion shows causality.</b> Something moves only to show that one thing happened because
        of another. Anything else is still.
      </Rule>

      <Rule
        id="A10"
        why="So you never wonder whether something is happening. Press Run study, and Stop while it runs."
        single
        examples={
          <Plate kind="try" label="a job that answers">
            <JobDemo />
          </Plate>
        }
      >
        <b>Every action answers.</b> It is acknowledged within 100 ms, says what it is doing while it
        runs, and ends in a visible state: done, failed with the reason, or cancelled.
      </Rule>

      <Rule
        id="A11"
        why="So nothing is a trap. Focus the plate and press Esc repeatedly."
        single
        examples={
          <Plate kind="try" label="the escape ladder">
            <EscapeDemo />
          </Plate>
        }
      >
        <b>Every step can be taken back.</b> An edit is undoable, a gesture abandonable, a mode
        escapable, one step at a time.
      </Rule>

      <Rule
        id="A12"
        why="So the hand learns the screen. A control that moves, grows into a different size when it opens, or slides under another one has to be found again every time."
        examples={
          <>
            <Plate kind="do" label="it opens at its parent's row, at its own size, pointing back" ground>
              <FlyoutSpecimen variant="pointer" />
            </Plate>
            <Plate kind="dont" label="a second card, somewhere near" ground>
              <RailSpecimen />
            </Plate>
          </>
        }
      >
        <b>Things stay where they are.</b> An element keeps its place and its size when it opens,
        changes state, or the mode changes, and it opens from where it is. Space for what will
        arrive is reserved, and nothing covers what it does not own.
      </Rule>

      <Rule
        id="A13"
        why="So what is on screen is worth reading. A fact shown twice is one fact and one distraction, and the two copies can disagree."
        examples={
          <>
            <Plate kind="do" label="the mode is said by the switcher alone">
              <div class="dl-stack">
                <div class="dl-seg">
                  <button type="button">Model</button>
                  <button type="button" class="active">Sketch</button>
                  <button type="button">Simulate</button>
                </div>
                <span class="dl-hint">Distance: choose a second point in the same sketch</span>
              </div>
            </Plate>
            <Plate kind="dont" label="said again at the foot of the viewport">
              <div class="dl-stack">
                <div class="dl-seg">
                  <button type="button">Model</button>
                  <button type="button" class="active">Sketch</button>
                  <button type="button">Simulate</button>
                </div>
                <span class="dl-row-of">
                  <span class="dl-tag">sketch</span>
                  <span class="dl-hint">Drag handles or the gizmo · Drag to orbit · Right-drag to pan</span>
                </span>
              </div>
            </Plate>
          </>
        }
      >
        <b>Say it once.</b> Each fact is stated in one place, the place that owns it. A command may
        have several ways in (menu, rail, key) but it has one name; a state has one display; a
        resting instruction is not shown at all.
      </Rule>
    </Section>
  );
}

function Colour() {
  const bcEntries = Object.entries(BC_TYPE_COLORS) as [string, readonly [number, number, number]][];
  return (
    <Section
      id="colour"
      num="6"
      title="Colour"
      intro="Paper, three inks, three rules, and a closed table of roles. Ratios are measured live against the ground (#e6e6e9) and the sheet (#f8f7f8)."
    >
      <h3>The ground (A1, A7)</h3>
      <Rule id="C1" why="The seam between chrome and viewport is a rule, not a step in brightness: one sheet, ruled.">
        Chrome and viewport share one ground. Above it there are two sheets, and no third: a surface
        within 3 ΔOKLab of an existing one is that surface. Hover is a step <i>down</i>, because on
        paper pressure darkens.
      </Rule>
      <div class="dl-swatches">
        <Swatch token="surface-base" job="the ground: page and viewport" ink="ink" />
        <Swatch token="surface-bar-alt" job="sub-bars, and hover" ink="ink" />
        <Swatch token="surface-float" job="the sheet: panels, popovers, dialogs" ink="ink" />
      </div>

      <h3>Ink and rules (A5)</h3>
      <Rule id="C2" why="An opacity fade has no contrast ratio that can be asserted, so text is never faded.">
        Three inks and no fourth. Three rules, one weight.
      </Rule>
      <div class="dl-swatches">
        <Swatch token="ink" job="text, selection" need={4.5} />
        <Swatch token="ink-2" job="secondary text, icons, hover" need={4.5} />
        <Swatch token="ink-3" job="labels, units, construction" need={4.5} />
        <Swatch token="rule" job="hairlines inside a panel" need={[1.4, 2.8]} />
        <Swatch token="rule-strong" job="control edges, the seam" need={3} />
        <Swatch token="rule-heavy" job="viewport frame, active line" need={3} />
      </div>

      <h3>Roles (A3, A4)</h3>
      <Rule id="C3" why="A new hue needs a new row here first. The data roles live inside the viewport and in legends only.">
        The table is closed. Each role names what it means, what else carries it, and where it may
        appear.
      </Rule>
      <table class="dl-table">
        <thead>
          <tr>
            <th>role</th>
            <th>token</th>
            <th>means</th>
            <th>carrier</th>
            <th>zone</th>
            <th class="num">ground</th>
          </tr>
        </thead>
        <tbody>
          <For each={COLOR_ROLES}>
            {(role) => (
              <tr>
                <td>
                  <i class="dl-dot" style={{ background: v(role.token!) }} />
                  {role.role}
                </td>
                <td>
                  {role.token} <small style={{ color: v("ink-3") }}>{CHROME[role.token!]}</small>
                </td>
                <td class="prose">{role.means}</td>
                <td class="prose">{role.carrier}</td>
                <td>{role.zones.join(", ")}</td>
                <td class="num">
                  <Ratio fg={role.token!} bg="surface-base" need={role.token === "accent" ? undefined : 3} />
                  {role.token === "accent" ? <small style={{ color: v("ink-3") }}> fill only</small> : null}
                </td>
              </tr>
            )}
          </For>
          <tr>
            <td>
              <i class="dl-dot" style={{ background: fieldRampCss() }} />
              field
            </td>
            <td>viridis</td>
            <td class="prose">a solved nodal field</td>
            <td class="prose">a legend with labelled ends</td>
            <td>viewport, legend</td>
            <td />
          </tr>
          <tr>
            <td>
              <i class="dl-dot" style={{ background: qualityRampCss() }} />
              quality
            </td>
            <td>magma</td>
            <td class="prose">mesh quality</td>
            <td class="prose">a legend with labelled ends</td>
            <td>viewport, legend</td>
            <td />
          </tr>
          <tr>
            <td>
              <i class="dl-dot" style={{ background: sdfRampCss() }} />
              distance
            </td>
            <td>sdf ramp</td>
            <td class="prose">signed distance: inside violet, outside ochre</td>
            <td class="prose">the zero isoline</td>
            <td>viewport, legend</td>
            <td />
          </tr>
          <tr>
            <td>
              <i class="dl-dot" style={{ background: `linear-gradient(90deg, ${bcEntries.map(([, c]) => cssColor(c)).join(", ")})` }} />
              boundary
            </td>
            <td>bc-*</td>
            <td class="prose">the kind of each boundary condition</td>
            <td class="prose">a swatch beside its name</td>
            <td>viewport, BC list</td>
            <td />
          </tr>
          <tr>
            <td>
              <i class="dl-dot" style={{ background: cssColor(PROPOSAL_COLOR) }} />
              proposal
            </td>
            <td>proposal</td>
            <td class="prose">the live selection preview while building a BC</td>
            <td class="prose">exists only while building</td>
            <td>viewport</td>
            <td />
          </tr>
        </tbody>
      </table>
      <div class="dl-examples">
        <Plate label="the data roles, as legends">
          <div class="dl-legend">
            <b>field</b>
            <span>293.2 K</span>
            <i style={{ background: fieldRampCss() }} />
            <span>412.8 K</span>
          </div>
          <div class="dl-legend">
            <b>quality</b>
            <span>0.00</span>
            <i style={{ background: qualityRampCss() }} />
            <span>1.00</span>
          </div>
          <div class="dl-legend">
            <b>distance</b>
            <span>−0.25 m</span>
            <i style={{ background: sdfRampCss() }} />
            <span>+0.25 m</span>
          </div>
          <div style={{ "padding-top": "var(--inset)" }}>
            <For each={bcEntries}>
              {([name, color]) => (
                <span class="dl-bc">
                  <i style={{ background: cssColor(color) }} />
                  {name}
                </span>
              )}
            </For>
          </div>
        </Plate>
      </div>

      <h3>No hue for these</h3>
      <Rule
        id="C4"
        why="Selection has to read over any field value, and only the heaviest ink does. A constraint holds geometry still, and stillness has no hue."
      >
        Selection is <code>ink</code>. Hover is <code>ink-2</code>. Construction geometry is{" "}
        <code>ink-3</code>, and <code>rule</code> when locked. Constraints are <code>ink</code>, and a
        violated constraint is <code>danger</code>. A label of <i>kind</i> (kickers, kind chips) is
        inverse ink.
      </Rule>

      <h3>The accent (A3, A5)</h3>
      <Rule
        id="C5"
        why="As a ground under near-black type it measures well over AA; as text or a hairline on paper it fails even the 3:1 mark bar. One composition passes, so it is the only one."
        examples={
          <>
            <Plate kind="do" label="a filled block">
              <div class="dl-row-of">
                <button type="button" class="dl-btn primary">Run</button>
                <div class="dl-seg">
                  <button type="button">Model</button>
                  <button type="button" class="active">Sketch</button>
                  <button type="button">Simulate</button>
                </div>
              </div>
              <p class="dl-note">
                ink-on-accent on accent: <Ratio fg="ink-on-accent" bg="accent" need={4.5} />
              </p>
            </Plate>
            <Plate kind="dont" label="accent as ink or outline">
              <div class="dl-row-of">
                <span style={{ color: v("accent"), "font-weight": "var(--weight-semibold)" }}>SELECTION</span>
                <span style={{ padding: "var(--inset)", border: `var(--hairline) solid ${v("accent")}` }}>aluminium</span>
              </div>
              <p class="dl-note">
                accent on ground: <Ratio fg="accent" bg="surface-base" need={3} />
              </p>
            </Plate>
          </>
        }
      >
        The accent means <i>current</i>: the active mode, the selected row, the primary action, the
        source span being edited. It is a fill with <code>ink-on-accent</code> type, never ink.{" "}
        The block does not change under the pointer: a state that is on is one amber, never a
        darker press shade. <code>accent-press</code> survives only as the one accent-coloured mark
        on paper, where the fill itself fails 3:1.
      </Rule>
      <Rule
        id="C6"
        why="If kickers, chips and outlines also take the accent, it stops meaning anything."
        examples={
          <>
            <Plate kind="do" label="one current thing per region">
              <MiniPanel kicker="selection" title="Properties">
                <div class="dl-rows">
                  <div>
                    <span>sketch</span>
                    <b>fin comb</b>
                    <small />
                  </div>
                </div>
              </MiniPanel>
            </Plate>
            <Plate kind="dont" label="accent as a label of kind">
              <div class="dl-panel">
                <nav>
                  <button type="button" class="active">Properties</button>
                </nav>
                <div>
                  <div class="dl-panel-head">
                    <span class="dl-before-chip" style={{ background: v("accent"), color: v("ink-on-accent") }}>
                      selection
                    </span>
                    <strong>Properties</strong>
                  </div>
                  <div class="dl-rows">
                    <div>
                      <span>sketch</span>
                      <b>fin comb</b>
                      <small />
                    </div>
                  </div>
                </div>
              </div>
            </Plate>
          </>
        }
      >
        A screen holds few accent blocks: the mode, the focused panel's primary action, the
        selection, the armed tool.
      </Rule>

      <h3>Zones</h3>
      <Rule
        id="C7"
        why="A chrome hue inside the viewport could be read as a value. A ramp without numbers is decoration. A recoloured field pixel is a wrong reading."
        examples={
          <>
            <Plate kind="do" label="hover over a field is an outline" ground>
              <svg viewBox="0 0 200 80" style={{ width: "100%" }}>
                <defs>
                  <linearGradient id="dl-zone-do" x1="0" x2="1">
                    <stop offset="0" stop-color={cssColor([0.267, 0.005, 0.329])} />
                    <stop offset="0.5" stop-color={cssColor([0.128, 0.567, 0.551])} />
                    <stop offset="1" stop-color={cssColor([0.993, 0.906, 0.144])} />
                  </linearGradient>
                </defs>
                <rect x="10" y="10" width="180" height="60" fill="url(#dl-zone-do)" />
                <rect x="70" y="10" width="60" height="60" fill="none" stroke={v("ink")} stroke-width="2" />
              </svg>
            </Plate>
            <Plate kind="dont" label="hover tints the reading" ground>
              <svg viewBox="0 0 200 80" style={{ width: "100%" }}>
                <defs>
                  <linearGradient id="dl-zone-dont" x1="0" x2="1">
                    <stop offset="0" stop-color={cssColor([0.267, 0.005, 0.329])} />
                    <stop offset="0.5" stop-color={cssColor([0.128, 0.567, 0.551])} />
                    <stop offset="1" stop-color={cssColor([0.993, 0.906, 0.144])} />
                  </linearGradient>
                </defs>
                <rect x="10" y="10" width="180" height="60" fill="url(#dl-zone-dont)" />
                <rect x="70" y="10" width="60" height="60" fill={BEFORE.hoverTint} fill-opacity="0.45" />
              </svg>
            </Plate>
          </>
        }
      >
        Chrome hue stays outside the viewport. Inside it, only data and direction are coloured. A
        ramp in a panel is a legend at most 8px tall with both ends labelled. Hover and selection
        over a field are outlines. No text sits on a data-dependent backdrop.
      </Rule>
      <h3>The editor</h3>
      <Rule
        id="C9"
        why="Syntax highlighting is not a role, so it has no hue (A3). The program's values are the subject (A1), so they carry the weight, and the names around them recede."
        examples={
          <>
            <Plate kind="do" label="values carry the line; one hue for free">
              <pre class="dl-code">
                <div>
                  <span>79</span>
                  <span>
                    <span class="kwarg">density</span>
                    <span class="punct">=</span>
                    <span class="num">8940.0</span>
                    <span class="punct">,</span>
                  </span>
                </div>
                <div>
                  <span>116</span>
                  <span>
                    base_l <span class="punct">=</span> Vector2<span class="punct">(</span>
                    <span class="kwarg">value</span>
                    <span class="punct">=[</span>
                    <span class="num">-0.9</span>
                    <span class="punct">,</span> <span class="num">0.0</span>
                    <span class="punct">],</span> <span class="free">free=True</span>
                    <span class="punct">)</span>
                  </span>
                </div>
                <div>
                  <span>117</span>
                  <span class="com"># the deck stays level</span>
                </div>
              </pre>
            </Plate>
            <Plate kind="dont" label="one ink for everything">
              <pre class="dl-code">
                <div>
                  <span>79</span>
                  <span>density=8940.0,</span>
                </div>
                <div>
                  <span>116</span>
                  <span>base_l = Vector2(value=[-0.9, 0.0], free=True)</span>
                </div>
                <div>
                  <span>117</span>
                  <span># the deck stays level</span>
                </div>
              </pre>
            </Plate>
          </>
        }
      >
        Numbers and booleans are semibold <code>ink</code>; declared names <code>ink</code>;
        strings <code>ink-2</code>; keyword-argument names, punctuation and comments{" "}
        <code>ink-3</code> (comments italic). The editor's hues are its roles:{" "}
        <code>free=True</code> in <code>param</code>, diagnostics in <code>danger</code> and{" "}
        <code>warn</code> with a gutter shape, and the accent veil behind the span a viewport edit
        is rewriting.
      </Rule>

      <Rule id="C8" why="ok and axis-y are both green, traction and axis-x both red: they never label the same thing in the same place.">
        Two roles may share a hue family only across zones or modes.
      </Rule>
    </Section>
  );
}

function Type() {
  const jobs: Record<TextRole, string> = {
    caps: "labels, kickers, tabs, buttons, menu titles",
    body: "names, values 2700.0, units, list rows, the editor",
    title: "a panel's title, and nothing else",
    prose: "Sentences: hints, help, empty states, dialog text.",
  };
  const roles = Object.entries(TEXT_ROLES) as [TextRole, (typeof TEXT_ROLES)[TextRole]][];
  return (
    <Section
      id="type"
      num="7"
      title="Type"
      intro="Four text roles, and no other way to set text. A role is a whole style — size, weight, leading, tracking, case and figures — taken together; hierarchy inside a component comes from choosing a role and an ink level, never from adjusting one property."
    >
      <Rule
        id="T0"
        why="Each role looks different from every other at a glance; two that did not were merged (label and control into caps; value and code into body, since a monospace face already sets digits on one width). A component that needs a fifth is a component that has not decided what its text is. test/tokens.test.ts fails on any font, size, weight, tracking or case set outside a role."
        single
        examples={
          <Plate label="the four roles, as the app sets them">
            <div class="dl-type">
              <For each={roles}>
                {([role, spec]) => (
                  <div>
                    <code>{role}</code>
                    <code>
                      {spec.size} · {spec.weight}
                    </code>
                    <span class={`t-${role}`}>{jobs[role]}</span>
                  </div>
                )}
              </For>
            </div>
          </Plate>
        }
      >
        <b>caps</b> 11px tracked capitals · <b>body</b> 12px as written (names, values, the editor) · <b>title</b> 15px tracked capitals · <b>prose</b> 12px sans. Three sizes, two weights (400, 600).
      </Rule>
      <Rule
        id="T1"
        why="Between 9 and 13px a 1px step is about 10% of the size, just above the visible threshold; the roles skip 10px because a step that small between two roles reads as an accident. Tracking opens as the size falls, and only capitals are tracked."
      >
        The raw scale (six sizes, four weights, six trackings in tokens.ts) is what the roles are
        cut from. Components never use it directly.
      </Rule>
      <Rule
        id="T2"
        why="A program's string is a value, so it is quoted, not restyled. Uppercasing it would change what it says."
        examples={
          <>
            <Plate kind="do" label="furniture tracked, names verbatim">
              <div class="dl-row-of">
                <span class="dl-tag">thermal</span>
                <span class="t-body">sink-conduction</span>
              </div>
            </Plate>
            <Plate kind="dont" label="the program's name restyled">
              <div class="dl-row-of">
                <span class="dl-tag">thermal</span>
                <span style={{ "font-weight": "var(--weight-semibold)", "text-transform": "uppercase", "letter-spacing": "var(--tracking-xs)" }}>
                  sink-conduction
                </span>
              </div>
            </Plate>
          </>
        }
      >
        Uppercase is for furniture only. Strings from the program (<code>fin comb</code>,{" "}
        <code>sink-conduction</code>) are set exactly as written.
      </Rule>
      <Rule
        id="T3"
        why="Numbers change while you watch them. Tabular figures keep the row still; a shared right edge makes magnitude visible; the decimal point becomes a column."
        examples={
          <>
            <Plate kind="do" label="tabular, right-hung, unit after">
              <div class="dl-rows">
                <For each={[["peak", "412.80"], ["mean", "338.07"], ["min", "293.21"], ["Δ", "119.59"]]}>
                  {([label, value]) => (
                    <div>
                      <span>{label}</span>
                      <b>{value}</b>
                      <small>K</small>
                    </div>
                  )}
                </For>
              </div>
            </Plate>
            <Plate kind="dont" label="proportional, left-aligned, unit in label">
              <div class="dl-stack" style={{ gap: "var(--gap)", "font-family": "var(--font-sans)" }}>
                <For each={[["peak (K)", "412.80"], ["mean (K)", "338.07"], ["min (K)", "293.21"], ["Δ (K)", "119.59"]]}>
                  {([label, value]) => (
                    <div>
                      <span style={{ color: v("ink-3") }}>{label}: </span>
                      <b class="t-prose">{value}</b>
                    </div>
                  )}
                </For>
              </div>
            </Plate>
          </>
        }
      >
        Values are tabular, right-aligned on a shared column, one size above their label. The unit
        follows the value in <code>ink-3</code>, one size smaller, on the same baseline.
      </Rule>
    </Section>
  );
}

function Space() {
  const heights = Object.entries(GRID_CONTROL_HEIGHTS) as [string, number][];
  const jobs: Record<string, string> = {
    "control-tag": "non-interactive tags",
    "control-sm": "the smallest clickable: dense rows",
    "control-md": "inputs, buttons in panels",
    "control-lg": "toolbar, tool rail",
    "control-xl": "the top bar",
  };
  return (
    <Section id="space" num="9" title="Space and size" intro={`Every length is a multiple of ${UNIT}px (A6).`}>
      <Rule
        id="S0"
        why="A distance named by its job is the same everywhere that job recurs, so two panels built months apart still line up. Turn on View › Show › Layout grid (⇧G) in the playground: it draws the unit over the whole UI and outlines whatever is off it."
        single
        examples={
          <Plate label="the three layout distances, bars drawn at 4×">
            <div class="dl-ruler">
              <For each={Object.entries(LAYOUT) as [string, number][]}>
                {([name, px]) => (
                  <div>
                    <code>{name}</code>
                    <b>{px}px</b>
                    <i style={{ width: `${px * 4}px` }} />
                  </div>
                )}
              </For>
            </div>
            <p class="dl-note">
              gap — between cells of one group · inset — inside a sheet · section — between
              blocks, and the viewport's furniture from its edges.
            </p>
          </Plate>
        }
      >
        Layout uses three distances and no others: <b>gap</b> 4, <b>inset</b> 8, <b>section</b> 16. The spacing scale they are cut from is not used by
        components; the test fails on it.
      </Rule>
      <Rule
        id="S2"
        why="24px is the WCAG 2.2 minimum target (2.5.8), so nothing clickable is smaller. Every control in a row takes one height, so labels, inputs and icons share a baseline."
        single
        examples={
          <Plate label="control heights, five to nine units">
            <div class="dl-ruler">
              <For each={heights}>
                {([token, px]) => (
                  <div>
                    <code>{token}</code>
                    <b>{px}px</b>
                    <div class="dl-row-of">
                      {token === "control-tag" ? (
                        <span class="dl-tag">tag</span>
                      ) : (
                        <button
                          type="button"
                          class={`dl-btn ${{ "control-sm": "h-sm", "control-md": "", "control-lg": "h-lg", "control-xl": "h-xl" }[token]}`}
                        >
                          {jobs[token]}
                        </button>
                      )}
                      {token === "control-tag" ? <span style={{ color: v("ink-3") }}>{jobs[token]}</span> : null}
                    </div>
                  </div>
                )}
              </For>
            </div>
          </Plate>
        }
      >
        Control heights are 20 · 24 · 28 · 32 · 36.
      </Rule>
      <Rule
        id="S3"
        why="A 2px line used for emphasis would be read as state."
        examples={
          <>
            <Plate kind="do" label="2px means active">
              <div class="dl-panel">
                <nav>
                  <button type="button" class="active">Studies</button>
                  <button type="button">Meshes</button>
                </nav>
                <div>
                  <input class="dl-input" value="0.75" style={{ outline: "var(--stroke-active) solid var(--ink)", "outline-offset": "calc(var(--hairline) * -1)" }} />
                </div>
              </div>
            </Plate>
            <Plate kind="dont" label="2px for emphasis">
              <div style={{ padding: "var(--inset)", border: "var(--stroke-active) solid var(--rule-heavy)" }}>
                Important settings
              </div>
            </Plate>
          </>
        }
      >
        The hairline (1px) is the only border weight. The 2px line means one thing: <i>this is the
        active one</i>, as on the active tab or the focused input.
      </Rule>
    </Section>
  );
}

function Shape() {
  return (
    <Section
      id="shape"
      num="2"
      title="Building blocks"
      intro="The atoms everything is drawn with: three lines, three planes, two blocks, a handful of marks, four text roles and three distances; no two of them look alike. Each is a token, whole, generated from tokens.ts; the primitives in the next section are formulas of these and nothing else. Derived from A7: things are drawn, not stacked."
    >
      <Rule
        id="B0"
        why="Everything on screen is drawn with these and nothing else. Learn this plate and every panel, menu and overlay in the app is one of its combinations; a thing that cannot be described with it is a bug."
        single
        examples={
          <Plate label="the atlas: every atom, side by side">
            <div class="dl-atlas">
              <section>
                <span class="t-caps">lines</span>
                <For each={["line", "active", "pending"]}>
                  {(name) => (
                    <div class="dl-atlas-line">
                      <i style={{ "border-top": name === "line" ? "var(--line)" : `var(--line-${name})` }} />
                      <span class="t-body">{name}</span>
                    </div>
                  )}
                </For>
              </section>
              <section>
                <span class="t-caps">planes</span>
                <For each={["ground", "sheet", "step"]}>
                  {(name) => (
                    <div class="dl-atlas-plane" style={{ background: `var(--plane-${name})` }}>
                      <span class="t-body">{name}</span>
                    </div>
                  )}
                </For>
              </section>
              <section>
                <span class="t-caps">blocks</span>
                <span class="dl-blk current t-caps">current</span>
                <span class="dl-blk kind t-caps">kind</span>
              </section>
              <section>
                <span class="t-caps">marks</span>
                <div class="dl-atlas-marks">
                  <i class="dl-mark dot" />
                  <i class="dl-mark ring" />
                  <i class="dl-mark square" />
                  <i class="dl-mark triangle" />
                  <b class="t-body">✓</b>
                  <b class="t-body">›</b>
                </div>
              </section>
              <section>
                <span class="t-caps">text</span>
                <span class="t-caps">caps</span>
                <span class="t-body">body 2700.0</span>
                <span class="t-title">title</span>
                <span class="t-prose">Prose sentence.</span>
              </section>
              <section>
                <span class="t-caps">distances</span>
                <For each={Object.entries(LAYOUT) as [string, number][]}>
                  {([name, px]) => (
                    <div class="dl-atlas-line">
                      <i class="dl-atlas-dist" style={{ width: `${px * 2}px` }} />
                      <span class="t-body">
                        {name} {px}
                      </span>
                    </div>
                  )}
                </For>
              </section>
            </div>
          </Plate>
        }
      >
        Six kinds of atom: three <b>lines</b>, three <b>planes</b>, two <b>blocks</b>, the{" "}
        <b>marks</b>, four <b>text</b> roles and three <b>distances</b>. The rules below define each kind; the
        sections after build from them.
      </Rule>
      <Rule
        id="B1"
        why="Three lines because there are three things a line can say: here is an edge, this is the active one, this is not there yet. Separators, edges and the viewport frame were once three greys nobody could tell apart; they are one line now. A border written any other way fails test/tokens.test.ts."
        single
        examples={
          <Plate label="the three lines">
            <table class="dl-table">
              <tbody>
                <For
                  each={[
                    ["line", "every structural line: separators, the edge of a sheet, a field, a cell, the viewport's frame"],
                    ["line-active", "this is the active one: a tab, focus, the selected edge"],
                    ["line-pending", "not yet: an add button, a drop target, an expression"],
                  ]}
                >
                  {([token, job]) => (
                    <tr>
                      <td>{token}</td>
                      <td style={{ width: "calc(var(--unit) * 40)" }}>
                        <div style={{ "border-top": `var(--${token})` }} />
                      </td>
                      <td class="prose">{job}</td>
                    </tr>
                  )}
                </For>
              </tbody>
            </table>
          </Plate>
        }
      >
        <b>line</b>, <b>active</b>, <b>pending</b>. One hairline for all structure; the 2px line means
        only <i>active</i>; dashed means only <i>not yet</i>.
      </Rule>
      <Rule
        id="B2"
        why="Elevation is a lighter sheet, never a shadow; hover is a step down, because pressure darkens paper."
        single
        examples={
          <Plate label="the three planes" ground>
            <div class="dl-cluster-demo">
              <For each={[["plane-ground", "ground: page, viewport"], ["plane-sheet", "sheet: panels, popovers, fields"], ["plane-step", "step: hover, sub-bars"]]}>
                {([token, job]) => (
                  <div class="dl-plane" style={{ background: `var(--${token})` }}>
                    <span class="t-caps">{token}</span>
                    <span class="t-prose">{job}</span>
                  </div>
                )}
              </For>
            </div>
          </Plate>
        }
      >
        <b>ground</b>, <b>sheet</b>, <b>step</b>. Nothing is drawn on a fourth.
      </Rule>
      <Rule
        id="B3"
        why="A block says something by its fill, so there are exactly as many as there are things to say: current, and kind."
        single
        examples={
          <Plate label="the two blocks">
            <div class="l-cluster">
              <span class="dl-blk current t-caps">Sketch</span>
              <span class="dl-blk current t-caps">Run</span>
              <span class="dl-blk kind t-caps">thermal</span>
              <span class="dl-blk kind t-caps">library</span>
            </div>
          </Plate>
        }
      >
        <b>current</b>: the accent with ink-on-accent type, the thing you act in or on. <b>kind</b>:
        ink with sheet type, a label of what something is.
      </Rule>
      <Rule
        id="B4"
        why="A mark is a shape that says a state without a word or a hue (A4). Circles mean a point; squares mean held; a triangle warns."
        single
        examples={
          <Plate label="the marks">
            <div class="dl-marks">
              <span>
                <i class="dl-mark dot" />
                <small class="t-caps">dot</small>
                <small class="t-prose">done, a status</small>
              </span>
              <span>
                <i class="dl-mark ring" />
                <small class="t-caps">ring</small>
                <small class="t-prose">a free point</small>
              </span>
              <span>
                <i class="dl-mark square" />
                <small class="t-caps">square</small>
                <small class="t-prose">held, an error</small>
              </span>
              <span>
                <i class="dl-mark triangle" />
                <small class="t-caps">triangle</small>
                <small class="t-prose">a warning</small>
              </span>
              <span>
                <b class="t-body">✓</b>
                <small class="t-caps">check</small>
                <small class="t-prose">on, current value</small>
              </span>
              <span>
                <b class="t-body">›</b>
                <small class="t-caps">chevron</small>
                <small class="t-prose">opens to the side</small>
              </span>
            </div>
          </Plate>
        }
      >
        <b>dot</b>, <b>ring</b>, <b>square</b>, <b>triangle</b>, <b>check</b>, <b>chevron</b>, and the
        corner caret that says a tool has more behind it.
      </Rule>
      <Rule
        id="E1"
        why="Elevation is a lighter sheet and an edge. A shadow on paper is a picture of a card on a desk."
        examples={
          <>
            <Plate kind="do" label="a popover: sheet and edge" ground>
              <div class="dl-menu">
                <button type="button">
                  <span>✓</span>
                  <span>Editor</span>
                  <kbd />
                </button>
                <button type="button">
                  <span>✓</span>
                  <span>Object tree</span>
                  <kbd />
                </button>
                <button type="button">
                  <span />
                  <span>Meshes</span>
                  <kbd />
                </button>
              </div>
            </Plate>
            <Plate kind="dont" label="a floating card" ground>
              <div style={{ width: "calc(var(--unit) * 64)", padding: "var(--inset)", background: v("surface-float"), "border-radius": "10px", "box-shadow": "0 8px 24px rgba(0,0,0,.2)" }}>
                <div style={{ padding: "var(--gap)" }}>✓ Editor</div>
                <div style={{ padding: "var(--gap)" }}>✓ Object tree</div>
                <div style={{ padding: "var(--gap)" }}>Meshes</div>
              </div>
            </Plate>
          </>
        }
      >
        Radius is zero; circles appear only where roundness means <i>a point, not a region</i>{" "}
        (status dots, vertex handles). There are no shadows.
      </Rule>
      <Rule
        id="E2"
        why="A modal blocks everything else, so it is the only thing that dims what is behind it."
        single
        examples={
          <Plate label="a dialog: scrim, sheet, prose, buttons bottom-right">
            <div class="dl-scrim">
              <div class="dl-dialog" role="dialog" aria-label="Overwrite">
                <h4>Overwrite heat_sink.py?</h4>
                <p>The file on disk changed since you opened it. Saving replaces it, and undo cannot bring the disk version back.</p>
                <footer>
                  <button type="button" class="dl-btn">Cancel</button>
                  <button type="button" class="dl-btn primary">Overwrite</button>
                </footer>
              </div>
            </div>
          </Plate>
        }
      >
        A modal dialog is a sheet with an edge over an ink scrim at 42%.
      </Rule>
      <Rule
        id="E3"
        why="A field darker than its sheet reads as a hole, and a hole is what disabled looks like."
        examples={
          <>
            <Plate kind="do" label="a control is a plane">
              <input class="dl-input" value="0.75" />
            </Plate>
            <Plate kind="dont" label="an inset control">
              <input class="dl-input" value="0.75" style={{ background: v("surface-bar-alt"), "border-color": "transparent" }} />
            </Plate>
          </>
        }
      >
        No inset controls. Inputs are the sheet with a <code>rule-strong</code> edge.
      </Rule>
      <Rule
        id="E4"
        why="One edge style for one idea: something you could make, a place you could drop."
        single
        examples={
          <Plate label="dashed: an add button, a drop target, a read-only expression">
            <div class="dl-row-of">
              <button type="button" class="dl-btn add">+ Boundary condition</button>
              <div style={{ padding: "var(--inset) var(--section)", border: "var(--hairline) dashed var(--rule-strong)", color: v("ink-3") }}>
                Drop a material here
              </div>
              <input class="dl-input" readonly value="0.5 * base_w" style={{ width: "calc(var(--unit) * 32)" }} />
            </div>
          </Plate>
        }
      >
        A dashed edge means <i>not yet</i>, or <i>not here</i>: an add button, a drop target, a value
        written as an expression that the panel will not overwrite.
      </Rule>
    </Section>
  );
}

function Icons() {
  const icons: [string, (props: { size?: number }) => JSX.Element][] = [
    ["select", CursorIcon],
    ["polygon", PolygonIcon],
    ["box", BoxIcon],
    ["sphere", SphereIcon],
    ["cylinder", CylinderIcon],
    ["move", MoveIcon],
    ["rotate", RotateIcon],
    ["extrude", ExtrudeIcon],
    ["revolve", RevolveIcon],
    ["loft", LoftIcon],
    ["distance", DistanceIcon],
    ["horizontal", HorizontalIcon],
    ["vertical", VerticalIcon],
    ["model", ModelModeIcon],
    ["sketch", SketchModeIcon],
    ["simulate", SimulateModeIcon],
    ["display", DisplayIcon],
    ["source", CodeIcon],
    ["run", PlayIcon],
    ["reset", ResetIcon],
    ["delete", TrashIcon],
  ];
  return (
    <Section id="icons" num="10" title="Iconography" intro="The playground's icon set, as shipped, at 16px.">
      <Rule
        id="I1"
        why="The stroke matches the hairline in feel, so an icon and a rule look drawn by one pen. Colour is not available to icons (A3): an icon that needs a colour needs a word."
        single
        examples={
          <Plate label="the set as shipped, ink-2">
            <div class="dl-icons">
              <For each={icons}>
                {([name, Icon]) => (
                  <div>
                    <Icon size={16} />
                    {name}
                  </div>
                )}
              </For>
            </div>
          </Plate>
        }
      >
        16px box, 1.25px stroke, no fills, miter joins and square caps, drawn on the half pixel.
        Icons are <code>ink-2</code>, and <code>ink</code> when active.
      </Rule>
      <Rule
        id="I2"
        why="An icon-only delete is a guess with a cost."
        examples={
          <>
            <Plate kind="do" label="the word carries the action">
              <div class="dl-row-of">
                <button type="button" class="dl-btn destructive">
                  <TrashIcon size={16} />
                  Delete study
                </button>
                <button type="button" class="dl-btn quiet" title="Display options">
                  <DisplayIcon size={16} />
                </button>
              </div>
            </Plate>
            <Plate kind="dont" label="a coloured icon, alone">
              <div class="dl-row-of">
                <button type="button" class="dl-btn" style={{ color: v("danger") }}>
                  <TrashIcon size={16} />
                </button>
              </div>
            </Plate>
          </>
        }
      >
        The tool rail is icons only, so it stays a narrow card beside the model: each icon's word
        is in its tooltip, and the hint bar names the armed tool. A destructive action carries the
        word unless undo reaches it (A11); the rail's Delete is undoable, so it is an icon there and
        a word in the Edit menu. Gizmos and handles are geometry, not icons.
      </Rule>
    </Section>
  );
}

function Motion() {
  return (
    <Section id="motion" num="11" title="Motion" intro="Derived from A9. Three durations, named by what they mean, and one easing family.">
      <Rule
        id="M1"
        why={`${DURATIONS["dur-fast"]} ms acknowledges the pointer, ${DURATIONS["dur-base"]} ms shows a state you caused, ${DURATIONS["dur-slow"]} ms shows something that changed because something else did.`}
        single
        examples={
          <Plate kind="try" label="the three durations">
            <DurationDemo />
          </Plate>
        }
      >
        Acknowledgement, state, propagation.
      </Rule>
      <Rule
        id="M2"
        why="Simultaneity teaches that the panel and the viewport are two views. The stagger teaches that the panel edits the code, and the code makes the geometry. Try it in A2."
      >
        Causal pairs are sequenced, never simultaneous: the changed source span lights first, the
        geometry follows.
      </Rule>
      <Rule
        id="M3"
        why="A tweened number shows values that were never true. Press New result in both plates."
        examples={
          <>
            <Plate kind="do" label="digits replace">
              <TweenDemo tween={false} />
            </Plate>
            <Plate kind="dont" label="digits roll">
              <TweenDemo tween />
            </Plate>
          </>
        }
      >
        Never animated: digits, the camera under direct manipulation, layout when data arrives.
        Replaying a recorded optimisation is not animation: every frame is a state the optimiser
        held.
      </Rule>
      <Rule id="M4" why="Causality is information, so it survives in a form that does not move.">
        Under <code>prefers-reduced-motion</code>, acknowledgement and state become instant and
        propagation becomes a two-frame highlight.
      </Rule>
    </Section>
  );
}

function Primitives() {
  const formulas: [string, string, () => JSX.Element][] = [
    ["sheet", "plane-sheet + line + inset", () => <div class="dl-sheet-demo" />],
    ["rule", "line, full width inside the sheet's inset", () => <div style={{ width: "calc(var(--unit) * 40)", "border-top": "var(--line)" }} />],
    ["cell", "a cell height + caps text; a line when it can be pressed, a block when current", () => (
      <span class="l-cluster">
        <button type="button" class="dl-btn h-sm">Mesh</button>
        <button type="button" class="dl-btn h-sm primary">Run</button>
      </span>
    )],
    ["field", "a sheet at cell height + body text, right-hung", () => <input class="dl-input" value="0.75" style={{ width: "calc(var(--unit) * 24)" }} />],
    ["chip", "tag height + body text on a sheet with a line; or a kind block with caps text", () => (
      <span class="l-cluster">
        <span class="dl-chip">fix · P1</span>
        <span class="dl-tag">thermal</span>
      </span>
    )],
    ["row", "l-row: caps label | body value | body unit, on the panel tracks", () => (
      <div class="dl-rows" style={{ width: "calc(var(--unit) * 64)" }}>
        <div>
          <span>density</span>
          <b>2700</b>
          <small>kg/m³</small>
        </div>
      </div>
    )],
    ["menu", "sheet + 24px body rows + a check column + a key column", () => (
      <div class="dl-menu" style={{ width: "calc(var(--unit) * 40)" }}>
        <button type="button">
          <span>✓</span>
          <span>Editor</span>
          <kbd />
        </button>
      </div>
    )],
  ];
  return (
    <Section
      id="primitives"
      num="3"
      title="Primitives"
      intro="Seven primitives, each a formula of the building blocks (§2). Every component is a composition of these; a component that seems to need an eighth is the place to stop and ask whether it is really new."
    >
      <Rule
        id="P1"
        why="Composition instead of invention: a panel is a sheet of ruled rows; a toolbar is cells; a chip list is chips in a cluster. Two things that look alike are the same primitive, so they behave alike."
        single
        examples={
          <Plate label="each primitive, and what it is made of" ground>
            <table class="dl-table">
              <thead>
                <tr>
                  <th>primitive</th>
                  <th>made of</th>
                  <th>specimen</th>
                </tr>
              </thead>
              <tbody>
                <For each={formulas}>
                  {([name, recipe, specimen]) => (
                    <tr>
                      <td>{name}</td>
                      <td class="prose">{recipe}</td>
                      <td>{specimen()}</td>
                    </tr>
                  )}
                </For>
              </tbody>
            </table>
          </Plate>
        }
      >
        <b>sheet</b>, <b>rule</b>, <b>cell</b>, <b>field</b>, <b>chip</b>, <b>row</b>, <b>menu</b>.
        None has a variant that carries no meaning.
      </Rule>
    </Section>
  );
}

function Layout0() {
  return (
    <Section
      id="layout-primitives"
      num="4"
      title="Layout"
      intro="Where things go, at every scale, and why. Each rule is a drawing, dimensioned in the system's numbers (§1) and generated from them, so a change to the module redraws the page."
    >
      <Rule
        id="L1"
        why="One way to separate siblings per level means you can tell the level of anything by how it is separated: a line between rows, space between blocks, an edge between windows. Two ways at one level (a rule and a gap) is the commonest way a panel starts to look busy."
        single
        examples={
          <Plate label="the six levels, each separated in one way" ground>
            <LevelsDrawing />
          </Plate>
        }
      >
        Six levels — <b>desk</b>, <b>window</b>, <b>sheet</b>, <b>block</b>, <b>list</b>, <b>row</b> — and
        each separates its children in exactly one way: windows share an edge, a sheet has an inset,
        blocks are a section of space apart, list rows have a line between them, a row is a cell
        high. Space never separates rows; a line never separates blocks.
      </Rule>
      <Rule
        id="L2"
        why="Shared tracks make every value in the app sit on one column, so two panels side by side, or one above the other, read as one table. A component chooses which tracks it spans; it does not choose widths."
        single
        examples={
          <Plate label="a panel on the panel grid, with its rhythm" ground>
            <PanelAnatomy />
          </Plate>
        }
      >
        Every sheet has the same four tracks inside its inset: <b>label</b> 96 · <b>value</b> the rest ·{" "}
        <b>unit</b> 40 · <b>action</b> 24, an inset apart. Rows take label | value | unit (| action);
        heads, prose and chip clusters span all four; a sheet's own actions end-align on the action
        track.
      </Rule>
      <Rule
        id="L3"
        why="If every block is a sum of line boxes and distances, every panel is a whole number of units tall, and the layout grid (⇧G) can show at once anything that is not."
        single
      >
        <b>Vertical rhythm.</b> Every height is a sum of line boxes and distances: a head is 16 + 4 +
        20 = 40, a row 24, a block apart 16, a bar 32. Text sits on its line box; a line box sits a gap
        inside its cell.
      </Rule>
      <Rule
        id="L4"
        why="Edges are what the eye uses to group; one left and one right edge per sheet means everything inside belongs to the sheet. Centring breaks edges, so it is kept for a label inside its own cell."
        single
      >
        <b>Alignment.</b> One left edge and one right edge per sheet, an inset in. Names and labels
        hang left; values and actions hang right; numbers align on their last digit. Nothing is
        centred except a label in its own cell.
      </Rule>
      <Rule
        id="L5"
        why="Reading order is causal order (A2), and the same columns in every mode keep the model under the pointer when the mode changes (N2)."
        single
        examples={
          <Plate label="the desk at its default proportions" ground>
            <DeskDrawing />
          </Plate>
        }
      >
        <b>The desk.</b> Three columns — the program 460, the viewport the rest, the column 320 — and
        four rows: the menu bar (a bar, 32), the toolbar (40), the desk, the tray (a cell, 24).
      </Rule>
      <Rule
        id="L6"
        why="The centre third is the part's; furniture that drifts toward it competes with the subject (A1). Anchoring each piece to a corner and stacking it inward means a new readout has a place before it is drawn."
        single
        examples={
          <Plate label="the viewport's anchors" ground>
            <AnchorsDrawing />
          </Plate>
        }
      >
        <b>The viewport.</b> Six anchors a section in from the edges — top-left, top-right,
        bottom-left, bottom-centre, bottom-right — and each piece of furniture stacks inward from its
        anchor. The centre third is kept for the part.
      </Rule>
      <Rule
        id="L7"
        why="The rules above, as the classes that implement them: choosing an arrangement chooses its distance."
        single
      >
        <b>The arrangements</b>, generated as classes from tokens.ts: <b>l-inset</b> (a sheet's inset),{" "}
        <b>l-stack</b> (blocks a section apart), <b>l-ruled</b> (rows with a line between),{" "}
        <b>l-cluster</b> (cells a gap apart, wrapping), <b>l-row</b> (the panel tracks), <b>l-split</b>{" "}
        (a head: start and end).
      </Rule>
    </Section>
  );
}

function Components() {
  return (
    <Section id="components" num="5" title="Components" intro="Each component is a composition of the primitives (§3), arranged by the layout primitives (§4), set in the text roles (T0).">
      <h3>Panel</h3>
      <Rule
        id="K1"
        single
        why="Docked, not floated over the viewport, unless the user floats it."
        examples={
          <Plate label="tab strip, header, rows" ground>
            <MiniPanel kicker="library" title="Materials" tabs={["Materials", "Optimize"]}>
              <div class="dl-rows">
                <div>
                  <span>aluminium</span>
                  <b>1 applied</b>
                  <small />
                </div>
                <div>
                  <span>copper</span>
                  <b>1 applied</b>
                  <small />
                </div>
                <div class="free">
                  <span>free parameters</span>
                  <b>17</b>
                  <small />
                </div>
                <div>
                  <span>density</span>
                  <b>2700</b>
                  <small>kg/m³</small>
                  <sub>aluminium_6061 · 293 K</sub>
                </div>
              </div>
            </MiniPanel>
          </Plate>
        }
      >
        A panel is a tab strip, then a sheet with 8px padding. The body opens with an inverse-ink
        kicker naming its kind and the title in 15px tracked mono. Rows are label, value, unit on one
        baseline; a second line hangs under the value.
      </Rule>

      <h3>Buttons</h3>
      <Rule
        id="K2"
        why="One primary per panel, so the accent stays the current action."
        single
        examples={
          <Plate label="variants and states">
            <table class="dl-table">
              <thead>
                <tr>
                  <th>variant</th>
                  <th>rest</th>
                  <th>pressed</th>
                  <th>disabled</th>
                </tr>
              </thead>
              <tbody>
                <For
                  each={[
                    ["primary", "primary", "Solve"],
                    ["button", "", "Mesh"],
                    ["quiet", "quiet", "Reset"],
                    ["destructive", "destructive", "Delete study"],
                    ["add", "add", "+ Study"],
                  ]}
                >
                  {([name, cls, label]) => (
                    <tr>
                      <td>{name}</td>
                      <td>
                        <button type="button" class={`dl-btn ${cls}`}>{label}</button>
                      </td>
                      <td>
                        <button type="button" class={`dl-btn ${cls} pressed`}>
                          {label}
                        </button>
                      </td>
                      <td>
                        <button type="button" class={`dl-btn ${cls}`} disabled title="Nothing to solve: add a study first">
                          {label}
                        </button>
                      </td>
                    </tr>
                  )}
                </For>
              </tbody>
            </table>
          </Plate>
        }
      >
        Primary is the accent block. A button is the sheet with a <code>rule-strong</code> edge;
        hover steps down to <code>surface-bar-alt</code>; pressed takes an <code>ink</code> edge. A
        disabled button keeps its shape, loses its edge contrast, and says why in its tooltip.
      </Rule>

      <h3>Inputs</h3>
      <Rule
        id="K3"
        why="Type a non-number into base width, then press Enter: the text stays, nothing is written."
        single
        examples={
          <Plate kind="try" label="rest, focus, invalid, read-only expression, disabled">
            <InputDemo />
          </Plate>
        }
      >
        An input is the sheet with a <code>rule-strong</code> edge and right-aligned tabular mono;
        a select is the same field with a drawn ink chevron and its value as the program writes it.
        Focus is a 2px <code>ink</code> edge; invalid is a <code>danger</code> edge with the text
        kept; an expression is read-only and dashed.
      </Rule>

      <h3>Chips and badges</h3>
      <Rule
        id="K4"
        single
        examples={
          <Plate label="kind, value, violated, count, free count">
            <div class="dl-row-of">
              <span class="dl-tag">thermal</span>
              <span class="dl-tag">elastic</span>
              <span class="dl-chip">
                distance · P1–P2 = 1.8
                <button type="button">
                  <Cross />
                </button>
              </span>
              <span class="dl-chip violated">
                equal · P1P15–P2P4 ≠ 0.02
                <button type="button">
                  <Cross />
                </button>
              </span>
              <span class="dl-badge" title="3 constraints">3</span>
              <span class="dl-badge free" title="17 free parameters">17</span>
            </div>
          </Plate>
        }
      >
        A kind chip is inverse ink. A value chip is the sheet with a <code>rule</code> edge, the value
        as written, and a × to remove it. A count badge is <code>ink</code> on{" "}
        <code>surface-bar-alt</code>; the free-parameter count is the one badge with a hue.
      </Rule>

      <h3>Menus</h3>
      <Rule
        id="K5"
        why="A fixed leading column keeps every label aligned whether or not it carries a check or a dot."
        single
        examples={
          <Plate label="a menu, opened from the bar" ground menu>
            <div class="dl-menubar">
              <button type="button">File</button>
              <button type="button">Edit</button>
              <div class="dl-menubar-item">
                <button type="button" class="open" aria-expanded="true">
                  Window
                </button>
                <div class="dl-menu dl-dropdown" role="menu">
                  <button type="button">
                    <span>●</span>
                    <span>Model mode</span>
                    <kbd>Esc</kbd>
                  </button>
                  <button type="button">
                    <span />
                    <span>Sketch mode</span>
                    <kbd>M</kbd>
                  </button>
                  <hr />
                  <button type="button">
                    <span>✓</span>
                    <span>Editor</span>
                    <kbd />
                  </button>
                  <button type="button">
                    <span />
                    <span>Processes</span>
                    <kbd />
                  </button>
                  <button type="button" disabled title="Results has nothing to show until a study is solved">
                    <span />
                    <span>Float results</span>
                    <kbd />
                  </button>
                  <hr />
                  <button type="button">
                    <span />
                    <span>Reset layout</span>
                    <kbd />
                  </button>
                </div>
              </div>
              <button type="button">Help</button>
            </div>
          </Plate>
        }
      >
        A menu is the sheet with a <code>rule-strong</code> edge; items are 24px, in one column,
        shortcuts right-aligned in <code>ink-3</code>. The bar's titles are furniture, tracked
        uppercase; the items are commands read down a list, so they are mono in sentence case, on
        one line, with section labels as kickers on the items' left edge. The open menu's title
        is the accent block.
      </Rule>

      <h3>Notices</h3>
      <Rule
        id="K6"
        why="A timer decides when you have read something. Feedback goes where its cause is, and stays until the cause is gone."
        examples={
          <>
            <Plate kind="do" label="in place, in the program's words">
              <div class="dl-notice error">
                <b>Error</b>
                <p>
                  Line 118: <code>Solidd</code> is not defined. Did you mean <code>Solid</code>?
                </p>
                <button type="button" class="dl-btn quiet h-sm">Go to line</button>
                <details>
                  <summary>Traceback</summary>
                  <pre>NameError: name 'Solidd' is not defined</pre>
                </details>
              </div>
            </Plate>
            <Plate kind="dont" label="a toast with a traceback">
              <div style={{ padding: "var(--inset)", background: v("ink"), color: v("surface-float"), "border-radius": "6px", "font-size": "var(--text-2xs)" }}>
                Traceback (most recent call last): File "worker.py", line 88, in compile … NameError: name 'Solidd' is not defined
                <div style={{ "margin-top": "var(--gap)", color: v("ink-3") }}>disappears in 4 s</div>
              </div>
            </Plate>
          </>
        }
      >
        No toasts on a timer. Feedback goes in place (the default), in the status line (the state of
        the whole program), or in the viewport's notice bar (a failure of the whole picture). It
        names the line, says what to do, and keeps the traceback behind a disclosure.
      </Rule>

      <h3>Status and empty states</h3>
      <Rule
        id="K7"
        examples={
          <>
            <Plate label="the status line">
              <div class="dl-stack">
                <span class="dl-status">
                  <i />
                  compiled · rev 14 · 1.2 s
                </span>
                <span class="dl-status stale">
                  <i />
                  out of date · compiling rev 15
                </span>
              </div>
            </Plate>
            <Plate label="an empty state">
              <div class="dl-empty">
                <p>
                  No results yet. <code>sink-conduction</code> has not been solved.
                </p>
                <div>
                  <button type="button" class="dl-btn primary h-sm">Solve</button>
                </div>
              </div>
            </Plate>
          </>
        }
      >
        The status line says whether the screen matches the code. An empty state names the missing
        precondition in the program's terms and offers the action.
      </Rule>

      <h3>Flyouts and submenus</h3>
      <Rule
        id="K8a"
        why="Chosen 17 Oct 2026: D. The rail keeps its edge whole, the flyout's cells sit on the parent's row at the parent's size, and the notch says where it came from. The other three are kept as the record of the choice."
        single
        examples={
          <div class="dl-fly-choices">
            <For each={FLYOUT_VARIANTS}>
              {(variant) => (
                <Plate label={variant.name} ground caption={variant.idea}>
                  <FlyoutSpecimen variant={variant.id} />
                </Plate>
              )}
            </For>
          </div>
        }
      >
        <b>A tool group opens as D:</b> a separate sheet a section out from the rail, its cells on the
        parent's row, a notch pointing at the parent. A, B and C were the alternatives.
      </Rule>
      <Rule
        id="K8"
        why="A12: an expansion keeps its parent's size and its parent's row, and says where it came from; a second card floating nearby at another size makes you find its origin."
        single
        examples={
          <Plate label="a menu with a submenu open beside its parent" ground>
            <div class="dl-row-of" style={{ "align-items": "flex-start", gap: "0" }}>
              <div class="dl-menu" style={{ width: "calc(var(--unit) * 52)" }}>
                <button type="button">
                  <span />
                  <span>Create</span>
                  <kbd>›</kbd>
                </button>
                <button type="button" style={{ background: "var(--surface-bar-alt)" }}>
                  <span />
                  <span>Shape</span>
                  <kbd>›</kbd>
                </button>
                <button type="button">
                  <span />
                  <span>Constrain</span>
                  <kbd>›</kbd>
                </button>
              </div>
              <div class="dl-menu" style={{ width: "calc(var(--unit) * 44)", "margin-top": "calc(var(--unit) * 6)" }}>
                <button type="button">
                  <span />
                  <span>Extrude</span>
                  <kbd>E</kbd>
                </button>
                <button type="button">
                  <span />
                  <span>Revolve</span>
                  <kbd />
                </button>
                <button type="button">
                  <span />
                  <span>Loft</span>
                  <kbd />
                </button>
              </div>
            </div>
          </Plate>
        }
      >
        A tool flyout is a sheet a section out from the rail, the same 32px cells as its parent on the
        parent's row, with a notch pointing back; the parent is marked by its fill. A submenu opens to
        the side of its parent item, top edge on the parent's row, with a › where the parent's
        shortcut would be.
      </Rule>

      <h3>Platform controls</h3>
      <Rule
        id="K9"
        why="A native select's open list is the operating system's menu: system font, system highlight, no token reaches it. An unmeasurable surface is outside A3 and A5, so the control is rebuilt from the primitives."
        single
        examples={
          <Plate kind="try" label="the playground's Select">
            <SelectSpecimen />
          </Plate>
        }
      >
        Every control is built from the primitives. A platform control whose look the tokens cannot
        reach (a select's list, a colour picker, a date field) is replaced, not themed.
      </Rule>
    </Section>
  );
}

function Layout() {
  return (
    <Section
      id="layout"
      num="13"
      title="Where things are"
      intro="Navigation is a map you learn once. Every region answers one question, every window has one name, and every action has one home."
    >
      <Rule
        id="N1"
        why="Reading order is causal order (A2): the program on the left makes the picture in the middle, and the right column shows what follows from what you selected in it."
        single
        examples={
          <Plate kind="try" label="the default desk; pick a mode in it, point at a region">
            <DeskMap />
          </Plate>
        }
      >
        The desk has fixed regions. From top to bottom: the menu bar, the toolbar (mode on the left,
        program status and Run on the right), the desk, and the tray. Left to right across the desk:
        the program, the picture, the consequences.
      </Rule>
      <Rule
        id="N2"
        why="Selecting a sketch enters Sketch mode under the pointer; if the viewport moved when the mode changed, the model would jump away from the click that caused it."
      >
        The columns are the same in every mode. A mode changes what is in the right column, not
        where the viewport is.
      </Rule>
      <Rule id="N3" why="Top to bottom is cause to effect, the same order as left to right.">
        The right column runs from what to what follows: the selection (Objects), then its arguments
        (Properties), then what depends on it. In Simulate: setup (Studies, Meshes) above outcomes
        (Results, Optimize).
      </Rule>
      <Rule
        id="N4"
        why="A thing with two names is two things to a newcomer. The Window menu is the index of the desk, so it uses the tab's own words."
        examples={
          <>
            <Plate kind="do" label="one name per place">
              <div class="dl-menu flush">
                <For each={["scene.py", "Objects", "Properties", "Sketch"]}>
                  {(name) => (
                    <button type="button">
                      <span>✓</span>
                      <span style={{ "text-transform": "none" }}>{name}</span>
                      <kbd>tab: {name}</kbd>
                    </button>
                  )}
                </For>
              </div>
            </Plate>
            <Plate kind="dont" label="the menu and the tab disagree">
              <div class="dl-menu flush">
                <For each={[["Editor", "scene.py"], ["Object tree", "Objects"], ["Properties", "Properties"], ["Sketch panel", "Sketch"]]}>
                  {([menu, tab]) => (
                    <button type="button">
                      <span>✓</span>
                      <span style={{ "text-transform": "none" }}>{menu}</span>
                      <kbd>tab: {tab}</kbd>
                    </button>
                  )}
                </For>
              </div>
            </Plate>
          </>
        }
      >
        Every window is reachable from the Window menu by the name on its tab, and nothing is lost
        by closing it.
      </Rule>
      <Rule
        id="N5"
        why="You find an action by asking what it acts on. Two homes for one action, or one action under two names, makes both harder to find."
        single
        examples={
          <Plate label="where an action lives">
            <table class="dl-table">
              <thead>
                <tr>
                  <th>it acts on</th>
                  <th>it lives</th>
                  <th>for example</th>
                </tr>
              </thead>
              <tbody>
                <For
                  each={[
                    ["one declaration", "on its card or row", "× on a constraint chip, Delete study, Solve on a study card"],
                    ["the whole program", "the toolbar, right", "Run, Reset, render settings"],
                    ["the application", "the menu bar", "Save as, Reset layout, Keyboard shortcuts"],
                    ["the picture", "the tool rail, the keyboard", "Polygon (P), Move (G), views (1 3 7)"],
                    ["the pipeline", "the toolbar, left", "Model · Sketch · Simulate (M)"],
                  ]}
                >
                  {([what, where, example]) => (
                    <tr>
                      <td>{what}</td>
                      <td>{where}</td>
                      <td class="prose">{example}</td>
                    </tr>
                  )}
                </For>
              </tbody>
            </table>
          </Plate>
        }
      >
        An action lives on what it acts on. A menu item shows its shortcut, so the menu teaches the
        keyboard.
      </Rule>
      <Rule
        id="N6"
        why="Corners are learned once. A readout that moves is a readout you have to look for."
      >
        The viewport's furniture has fixed places: the tool rail, a narrow floating card,
        top-left with the readout beside it; projection and render settings over the view cube,
        top-right; the legend bottom-left, the performance readout bottom-right, the hint
        bottom-centre (V3). Nothing else is pinned there, and nothing covers anything else.
      </Rule>
      <Rule
        id="N7"
        why="You should always be able to answer where am I, and how do I get back, without trying something."
      >
        You always know where you are and how to get back: the mode is named in the switcher and in
        the hint bar, the selection is named in the toolbar, Escape walks back one step (F6), and
        Reset layout restores the desk (F11).
      </Rule>
      <Rule id="N8" why="A monitor that takes a column by default is a monitor in the way, but it must be one click away, because a solve keeps running while you model.">
        Monitors and browsers (Processes, Scenes) are parked in the tray, named, one click away.
      </Rule>
    </Section>
  );
}

function Features() {
  return (
    <Section
      id="features"
      num="14"
      title="Every feature"
      intro="Every feature of the playground, everywhere it can be reached today, and where these rules put it. The table and the menu bar below are rendered from one list, design/features.ts."
    >
      <Rule
        id="N9"
        why="The menu bar is where you look when you do not know where to look. If a command is only on a rail, a gesture or a key, the only way to find it is to already know it."
      >
        The menu bar is the complete index. Every command has a menu home; the toolbar, the rail,
        the panels and the keys are shortcuts to it, and the menu shows the key. A command that does
        not apply in the current mode is shown disabled, with the reason; the rail and the panels
        show only what applies.
      </Rule>
      <Rule
        id="N10"
        why="Grouping follows N5: by what the command acts on. File for the document and the run, Edit for the selection, View for the picture, Model for geometry, Simulate for studies and their outcomes, Window for the desk, Help for the rest. Breadth is spent sparingly (A13): the rail's groups nest as Model's submenus rather than taking three titles of their own."
        single
        examples={
          <Plate kind="try" label="the menu bar as shipped; switch the mode and open the menus">
            <ProposedMenuBar />
          </Plate>
        }
      >
        Seven menus, grouped by what their commands act on, in the order you work: file, selection,
        picture, geometry, physics, desk, help. A section that is a set of one kind (Camera, Create,
        Constrain) is a submenu that opens to the side; a short section is a labelled block.
      </Rule>
      <Rule
        id="N11"
        why="Read down a column to see a kind of home; read along a row to see every way to one feature."
        single
        examples={
          <Plate label="the feature index">
            <FeatureIndex />
          </Plate>
        }
      >
        One name per command in every home, at least one keyboard path for every frequent command,
        and no command reachable only by a gesture.
      </Rule>
    </Section>
  );
}

function Viewport() {
  return (
    <Section id="viewport" num="12" title="Viewport" intro="The viewport is an instrument face (A1). Its overlays are achromatic unless they are data or direction (C7).">
      <Rule
        id="V1"
        why="Free versus held is told by shape and by the one hue that means free everywhere: the blue free=True takes in the editor is the blue of a vertex a drag or the optimiser may move."
        single
        examples={
          <Plate label="sketch mode" ground>
            <ViewportSpecimen />
          </Plate>
        }
      >
        A free vertex is a hollow <code>param</code> ring; a held vertex is a filled square; the selected vertex is a
        filled <code>ink</code> circle. Edges go <code>ink-3</code> → <code>ink-2</code> (hover) →{" "}
        <code>ink</code> (selected, 2px); a locked edge is <code>rule</code> (solid: dashed means <i>not yet</i>, E4). Dimension
        labels are <code>ink</code> on an opaque tab, and a violated one is <code>danger</code> and
        says by how much. The gizmo and the triad are the only other hues.
      </Rule>
      <Rule
        id="V2"
        why="The field is the only colour inside the rectangle: hover is an outline, the legend is opaque and labelled at both ends."
        single
        examples={
          <Plate label="simulate mode" ground>
            <ViewportSpecimen field />
          </Plate>
        }
      >
        Over a solved field, hover and selection are drawn, not tinted.
      </Rule>
      <Rule id="V3" why="A spacing on an unnamed plane is not a measurement; a value on a perspective view is not to scale.">
        The graticule is furniture, held to 1.4–2.8:1. It rules the world plane the camera most
        nearly faces and states its spacing on a 1-2-5 ladder (<code>GRID 100 mm · XZ</code>), prefixed{" "}
        <code>&gt;</code> when the view is not to scale; because the spacing follows the zoom, the
        grid steps are a detented zoom (View › Zoom, <code>[</code> <code>]</code>). The performance
        readout states what only the viewport can: FPS (<code>idle</code> when nothing is drawing),
        CPU time per frame, the render resolution, node count, and mesh elements and triangles once
        a mesh is shown; sampled twice a second, never tweened. The hint line speaks only for a step in
        progress (the second pick of a constraint, the face a sketch will land on, why that face
        cannot be used) or to name the hovered handle or plane; at rest it is empty, because the
        mode switcher already names the mode.
      </Rule>
    </Section>
  );
}

function Behaviour() {
  return (
    <Section id="behaviour" num="15" title="Behaviour" intro="The functional rules. They come from A2 (the program is the truth), A10 (every action answers) and A11 (every step can be taken back).">
      <h3>The program is the truth</h3>
      <Rule id="F1" why="An expression is not a number: overwriting it with one would silently break the program.">
        Every panel is an editor of literals. An argument written as an expression is shown as
        written and read-only (see K3).
      </Rule>
      <Rule
        id="F2"
        why="Click a row, a shape or a line: all three follow. The editor scrolls but does not take focus."
        single
        examples={
          <Plate kind="try" label="one selection, three views">
            <SelectionDemo />
          </Plate>
        }
      >
        One selection: viewport, object tree, properties and editor show the same thing. A precise
        selection reveals its literal; a whole object reveals its declaration.
      </Rule>
      <Rule id="F3" why="A stale picture shown as current is a wrong answer.">
        The screen says whether it matches the code (K7). A stale value is marked in place, never
        silently kept. The newest edit wins: an older answer arriving late is discarded.
      </Rule>

      <h3>Direct manipulation</h3>
      <Rule
        id="F4"
        why="Drag the handle along the rule: it snaps to detents and previews; release writes the source once; Esc during the drag puts it back."
        single
        examples={
          <Plate kind="try" label="preview, commit on release, Esc abandons">
            <DragDemo />
          </Plate>
        }
      >
        A drag shows its effect live and writes the source once, on release: one gesture, one undo
        step. Escape during a gesture returns the value to where it was. A snap draws what it snapped
        to, for as long as it holds.
      </Rule>

      <h3>Modes and tools</h3>
      <Rule
        id="F5"
        why="Switch mode, arm a tool: the rail, the hint and the tag change together."
        single
        examples={
          <Plate kind="try" label="mode, armed tool, hint">
            <ModeDemo />
          </Plate>
        }
      >
        Modes are a pipeline, Model → Sketch → Simulate; a mode changes the tool set, panels and
        shortcuts wholesale. One tool is armed at a time and is the accent cell; finishing or
        cancelling it returns to Select. A grouped tool remembers its last choice.
      </Rule>

      <h3>Escape and undo</h3>
      <Rule id="F6" why="The ladder in A11, in order: gesture, command, tool, selection, mode. When nothing applies, Escape does nothing.">
        Escape cancels one thing per press, most specific first.
      </Rule>
      <Rule
        id="F7"
        why="Remove a constraint: no dialog, and the undo is right there."
        single
        examples={
          <Plate kind="try" label="undo replaces confirmation">
            <UndoDemo />
          </Plate>
        }
      >
        Undo is over the source text: each run, patch and gesture is one step; in the editor it is
        the editor's own history. A deletion that can be undone is not confirmed. Confirmation (E2)
        is only for what undo cannot reach: overwriting a file, discarding unsaved work.
      </Rule>

      <h3>Feedback and long work</h3>
      <Rule
        id="F8"
        why="This one fails at SOLVE, and says why in the program's language."
        single
        examples={
          <Plate kind="try" label="a job that fails well">
            <JobDemo fail />
          </Plate>
        }
      >
        Within 100 ms the pressed control changes state. After about a second the work is a job,
        with its stage in words and a stop control (A10). A result outlives its panel: closing it or
        switching mode loses nothing. No indeterminate spinner over legible content, no percentage
        of an unknown total.
      </Rule>

      <h3>Inputs</h3>
      <Rule
        id="F9"
        examples={
          <>
            <Plate kind="do" label="precision as written">
              <div class="dl-field">
                <label>tip width</label>
                <input class="dl-input" value="0.5" />
                <small>m</small>
              </div>
            </Plate>
            <Plate kind="dont" label="reformatted, unit in the field">
              <div class="dl-field">
                <label>tip width</label>
                <input class="dl-input" value="0.5000 m" />
                <small />
              </div>
            </Plate>
          </>
        }
      >
        A value commits on Enter or blur, and only if valid. Units are shown beside the field, never
        typed. Numbers keep their written precision. A disabled control says why; a control that
        cannot apply in this mode is not shown.
      </Rule>

      <h3>Keyboard</h3>
      <Rule
        id="F10"
        why="The table below is rendered from shortcuts.ts, the same record the key handlers read, so the sheet cannot drift from the keys."
        single
        examples={
          <Plate label="the shortcut sheet, live">
            <table class="dl-table">
              <tbody>
                <For each={SHORTCUT_GROUPS.slice(0, 3)}>
                  {(group) => (
                    <For each={group.items}>
                      {(item, index) => (
                        <tr>
                          <td style={{ color: v("ink-3") }}>{index() === 0 ? group.title : ""}</td>
                          <td>
                            <kbd>{item.keys}</kbd>
                          </td>
                          <td class="prose">{item.action}</td>
                        </tr>
                      )}
                    </For>
                  )}
                </For>
              </tbody>
            </table>
          </Plate>
        }
      >
        Every action is reachable from the keyboard and listed in the Help sheet. Single-key
        shortcuts act only when the viewport has focus. Focus order follows reading order, and focus
        is always visible: a 2px <code>ink</code> ring.
      </Rule>

      <h3>Layout</h3>
      <Rule
        id="F11"
        examples={
          <>
            <Plate kind="do" label="the slot exists before the data">
              <div class="dl-rows">
                <div>
                  <span>peak</span>
                  <b>—</b>
                  <small />
                </div>
                <div>
                  <span>mean</span>
                  <b>—</b>
                  <small />
                </div>
              </div>
            </Plate>
            <Plate kind="dont" label="rows appear, the panel jumps">
              <p class="dl-note" style={{ margin: "0" }}>
                (empty until the solve returns, then three rows push everything below them down)
              </p>
            </Plate>
          </>
        }
      >
        Windows dock, tab, float and minimise; the layout is remembered per mode, and Reset layout
        always restores a working default. Closing a window loses nothing. Space for arriving data is
        reserved.
      </Rule>

      <h3>Defaults</h3>
      <Rule
        id="F12"
        why="A1: the first picture someone sees is the part. A default they have to raise to see it properly is a default that hid it; a slower machine can step down, and the performance readout says when to."
      >
        Defaults are the best the machine can sustain: the Ultra tier, rendered at device resolution.
        A user lowers a setting; they never have to discover that they could raise it.
      </Rule>
    </Section>
  );
}

function Language() {
  const rows = [
    ["furniture", "MESH · SOLVE · BOUNDARY CONDITION", "uppercase tracked mono"],
    ["commands in a list", "Save as… · Show generated WGSL", "mono, sentence case, one line"],
    ["program strings", "fin comb · sink-conduction · base_l", "verbatim mono"],
    ["sentences", "Drag a swatch onto an object", "sans, sentence case, no period on one clause"],
    ["quantities", "1.00 m · 412.8 K · 2700 kg/m³", "SI, unit as a symbol after the value"],
    ["empty", "—", "an em dash; never 0, N/A or blank"],
  ];
  return (
    <Section id="language" num="8" title="Language" intro="Words are part of the interface, and they follow A8.">
      <Rule
        id="L1"
        single
        examples={
          <Plate label="five kinds of string">
            <table class="dl-table">
              <thead>
                <tr>
                  <th>kind</th>
                  <th>example</th>
                  <th>set as</th>
                </tr>
              </thead>
              <tbody>
                <For each={rows}>
                  {([kind, example, how], index) => (
                    <tr>
                      <td>{kind}</td>
                      <td
                        style={
                          index() === 0
                            ? { "text-transform": "uppercase", "letter-spacing": "var(--tracking-xs)", "font-size": "var(--text-xs)" }
                            : index() === 2
                              ? { "font-family": "var(--font-sans)" }
                              : { "font-variant-numeric": "tabular-nums" }
                        }
                      >
                        {example}
                      </td>
                      <td class="prose">{how}</td>
                    </tr>
                  )}
                </For>
              </tbody>
            </table>
          </Plate>
        }
      >
        Furniture is uppercase tracked mono. Program strings are verbatim. Sentences are sans.
        Quantities are SI. An empty field is an em dash.
      </Rule>
    </Section>
  );
}

function Change(props: { n: number; title: string; before: JSX.Element; after: JSX.Element; why: JSX.Element }) {
  return (
    <Rule
      id={`Δ${props.n}`}
      why={props.why}
      examples={
        <>
          <Plate kind="before">{props.before}</Plate>
          <Plate kind="after">{props.after}</Plate>
        </>
      }
    >
      <b>{props.title}</b>
    </Rule>
  );
}

function Changes() {
  const handle = (fill: string, stroke?: string, square?: boolean) => () =>
    square ? (
      <rect x="-4" y="-4" width="8" height="8" fill={fill} />
    ) : (
      <circle r="4.5" fill={fill} stroke={stroke} stroke-width="1.5" />
    );
  const sketch = (edge: string, handles: (() => JSX.Element)[], selected: string, plane: string) => (
    <svg viewBox="0 0 220 100" style={{ width: "100%" }}>
      <rect x="10" y="10" width="200" height="80" fill={plane} fill-opacity="0.09" stroke={plane} stroke-width="1" />
      <path d="M40 75 L180 75 L180 30 L140 30 L140 55 L80 55 L80 30 L40 30 Z" fill="none" stroke={edge} stroke-width="1.5" />
      <line x1="140" y1="30" x2="180" y2="30" stroke={selected} stroke-width="2" />
      <For each={[[40, 75], [180, 75], [180, 30], [140, 30], [140, 55], [80, 55], [80, 30], [40, 30]]}>
        {([x, y], index) => <g transform={`translate(${x} ${y})`}>{handles[index() % handles.length]()}</g>}
      </For>
    </svg>
  );
  return (
    <Section
      id="changes"
      num="17"
      title="Record"
      intro="What the playground did before this language, and what it does now. Each pair is the old behaviour, then the rule as shipped; together they are the reasons the rules above read as they do."
    >
      <Change
        n={1}
        title="Viewport overlays become achromatic"
        why="Olive edges, red handles and blue planes were hues with no role (C7, V1)."
        before={sketch(BEFORE.edge, [handle(BEFORE.handle)], BEFORE.edgeSelected, BEFORE.plane)}
        after={sketch(
          CHROME["ink-3"],
          [handle(CHROME["ink-3"], undefined, true), handle(CHROME["ink-3"], undefined, true), handle(CHROME["surface-float"], CHROME["ink-2"])],
          CHROME.ink,
          CHROME["ink-3"],
        )}
      />
      <Change
        n={2}
        title="Kickers and kind chips become inverse ink"
        why="Six accent blocks on one screen, and the accent stopped meaning current (C6)."
        before={
          <div class="dl-row-of">
            <For each={["read-only", "selection", "library", "fem", "thermal"]}>
              {(word) => (
                <span class="dl-before-chip" style={{ background: v("accent"), color: v("ink-on-accent") }}>
                  {word}
                </span>
              )}
            </For>
            <span class="dl-before-chip" style={{ background: "#0065b4", color: v("surface-float") }}>
              elastic
            </span>
          </div>
        }
        after={
          <div class="dl-row-of">
            <For each={["read-only", "selection", "library", "fem", "thermal", "elastic"]}>{(word) => <span class="dl-tag">{word}</span>}</For>
          </div>
        }
      />
      <Change
        n={3}
        title="Syntax highlighting goes achromatic"
        why="Class names were drawn in error red, strings in success green, numbers in warning amber (A3)."
        before={
          <pre class="dl-code">
            <div>
              <span>32</span>
              <span>
                <span style={{ color: BEFORE.keyword }}>from</span> cadjoint.constraints <span style={{ color: BEFORE.keyword }}>import</span> (
                <span style={{ color: BEFORE.className }}>DistanceConstraint</span>)
              </span>
            </div>
          </pre>
        }
        after={
          <pre class="dl-code">
            <div>
              <span>32</span>
              <span>
                <span class="kw">from</span> cadjoint.constraints <span class="kw">import</span> (DistanceConstraint)
              </span>
            </div>
          </pre>
        }
      />
      <Change
        n={4}
        title="info becomes param, and means only free parameter"
        why="The blue was a constraint count, a residual line, a plane label and the elastic chip: four meanings (A3)."
        before={
          <div class="dl-row-of">
            <span class="dl-badge" style={{ background: "#0065b4", color: v("surface-float") }} title="constraints">
              17
            </span>
            <span style={{ color: "#004a85" }}>FIN COMB</span>
          </div>
        }
        after={
          <div class="dl-row-of">
            <span class="dl-badge" title="17 constraints">17</span>
            <span class="dl-badge free" title="12 free parameters">12 free</span>
            <span style={{ color: v("ink-2") }}>FIN COMB</span>
          </div>
        }
      />
      <Change
        n={5}
        title="One set of axis colours"
        why="The gizmo and the view-cube triad drew different reds, greens and blues. Both use axis-x/y/z, dark enough that the letters clear 4.5:1."
        before={
          <table class="dl-table">
            <tbody>
              <For each={[["X", BEFORE.gizmoX, BEFORE.cubeX], ["Y", BEFORE.gizmoY, BEFORE.cubeY], ["Z", BEFORE.gizmoZ, BEFORE.cubeZ]]}>
                {([axis, gizmo, cube]) => (
                  <tr>
                    <td>{axis}</td>
                    <td>
                      <i class="dl-dot" style={{ background: gizmo }} />
                      gizmo <Ratio fg={gizmo} bg="surface-base" />
                    </td>
                    <td>
                      <i class="dl-dot" style={{ background: cube }} />
                      cube <Ratio fg={cube} bg="surface-base" />
                    </td>
                  </tr>
                )}
              </For>
            </tbody>
          </table>
        }
        after={
          <table class="dl-table">
            <tbody>
              <For each={["axis-x", "axis-y", "axis-z"] as ChromeToken[]}>
                {(token) => (
                  <tr>
                    <td>{token.slice(-1).toUpperCase()}</td>
                    <td>
                      <i class="dl-dot" style={{ background: v(token) }} />
                      {token} <Ratio fg={token} bg="surface-base" need={4.5} />
                    </td>
                  </tr>
                )}
              </For>
            </tbody>
          </table>
        }
      />
      <Change
        n={6}
        title="Control heights move onto the grid"
        why="22 · 26 · 30 · 34 are not multiples of the unit, and 22 is under the 24px target (A6, S2)."
        before={
          <div class="dl-row-of">
            <For each={[22, 26, 30, 34]}>
              {(h) => (
                <button type="button" class="dl-btn" style={{ height: `${h}px` }}>
                  {h}
                </button>
              )}
            </For>
          </div>
        }
        after={
          <div class="dl-row-of">
            <span class="dl-tag">20</span>
            <button type="button" class="dl-btn h-sm">24</button>
            <button type="button" class="dl-btn">28</button>
            <button type="button" class="dl-btn h-lg">32</button>
          </div>
        }
      />
      <Change
        n={7}
        title="The accent never draws an outline"
        why={
          <>
            The selected material card's orange border measured <Ratio fg="accent" bg="surface-base" need={3} /> (C5).
          </>
        }
        before={
          <div style={{ display: "inline-block", padding: "var(--inset)", border: `var(--hairline) solid ${v("accent")}` }}>
            aluminium
            <br />
            <small style={{ color: v("ink-3") }}>1 applied</small>
          </div>
        }
        after={
          <div style={{ display: "inline-grid", border: `var(--hairline) solid ${v("rule-strong")}` }}>
            <span style={{ padding: "var(--inset)" }}>
              <small style={{ color: v("ink-3") }}>1 applied</small>
            </span>
            <span style={{ padding: "var(--gap) var(--inset)", background: v("accent"), color: v("ink-on-accent"), "font-weight": "var(--weight-bold)" }}>
              aluminium
            </span>
          </div>
        }
      />
      <Change
        n={8}
        title="Menus align in one column"
        why="The mode items were centred while the rest were left-aligned (K5)."
        before={
          <div class="dl-menu" style={{ width: "calc(var(--unit) * 52)" }}>
            <button type="button" style={{ "grid-template-columns": "1fr", "text-align": "center" }}>
              <span>Model mode</span>
            </button>
            <button type="button" style={{ "grid-template-columns": "1fr", "text-align": "center" }}>
              <span>Sketch mode</span>
            </button>
            <button type="button">
              <span>✓</span>
              <span>Editor</span>
              <kbd />
            </button>
          </div>
        }
        after={
          <div class="dl-menu" style={{ width: "calc(var(--unit) * 52)" }}>
            <button type="button">
              <span>●</span>
              <span>Model mode</span>
              <kbd>Esc</kbd>
            </button>
            <button type="button">
              <span />
              <span>Sketch mode</span>
              <kbd>M</kbd>
            </button>
            <button type="button">
              <span>✓</span>
              <span>Editor</span>
              <kbd />
            </button>
          </div>
        }
      />
      <Change
        n={9}
        title="Icons: square caps, miter joins, no fills"
        why="The shipped set uses round caps and joins and filled dots, against I1."
        before={
          <div class="dl-row-of">
            <span style={{ color: v("ink-3") }}>round caps and joins, filled handles, a 24-unit box</span>
          </div>
        }
        after={
          <div class="dl-row-of">
            <PolygonIcon size={16} />
            <MoveIcon size={16} />
            <DistanceIcon size={16} />
            <span style={{ color: v("ink-3") }}>redrawn: 16px, 1.25 stroke, square caps, hollow points</span>
          </div>
        }
      />
      <Change
        n={10}
        title="Dark-theme leftovers and undefined tokens go"
        why="White-alpha fills, a lime and a cyan rgba, a bluish element-edge tone and an undefined --type-body-small survive from the dark theme."
        before={
          <div class="dl-row-of">
            <i class="dl-dot" style={{ background: "rgb(217, 255, 87)" }} />
            <i class="dl-dot" style={{ background: "rgb(120, 198, 231)" }} />
            <i class="dl-dot" style={{ background: "rgb(219, 230, 240)" }} />
          </div>
        }
        after={<span style={{ color: v("ink-3") }}>only tokens</span>}
      />
      <Change
        n={11}
        title="The Window menu uses the tabs' names"
        why="The menu toggles Editor, Object tree and Sketch panel; the tabs say scene.py, Objects and Sketch; the Float items in the same menu already use the tab names (N4)."
        before={
          <div class="dl-menu flush">
            <For each={["Editor", "Object tree", "Sketch panel", "Float scene.py", "Float objects"]}>
              {(name) => (
                <button type="button">
                  <span />
                  <span>{name}</span>
                  <kbd />
                </button>
              )}
            </For>
          </div>
        }
        after={
          <div class="dl-menu flush">
            <For each={["scene.py", "Objects", "Sketch"]}>
              {(name) => (
                <button type="button">
                  <span>✓</span>
                  <span style={{ "text-transform": "none" }}>{name}</span>
                  <kbd>float</kbd>
                </button>
              )}
            </For>
          </div>
        }
      />
      <Change
        n={12}
        title="The tool rail is a narrow card"
        why="Words beside the icons made the card three times as wide and pushed it over the readout (A12)."
        before={<p class="dl-note" style={{ margin: "0" }}>A 170px column of icon-over-word cards, covering the GRID readout</p>}
        after={<p class="dl-note" style={{ margin: "0" }}>A 42px card of centred 32px icons, the words in tooltips, the readout beside it</p>}
      />
      <Change
        n={13}
        title="Tool flyouts open from their parent"
        why="The flyout was a second card at a different size, a gap away (A12, K8)."
        before={<p class="dl-note" style={{ margin: "0" }}>A padded card 16px to the side, larger cells</p>}
        after={<p class="dl-note" style={{ margin: "0" }}>The parent's row continued: same cells, flush, sharing the open parent's edge</p>}
      />
      <Change
        n={14}
        title="The hint line says only what is not said elsewhere"
        why="The mode was named twice and the orbit instructions shown forever (A13)."
        before={<p class="dl-note" style={{ margin: "0" }}>SKETCH · Drag handles or the gizmo · Drag to orbit · Right-drag to pan</p>}
        after={<p class="dl-note" style={{ margin: "0" }}>Silent at rest; the next step of a command, or the name of what is hovered</p>}
      />
      <Change
        n={15}
        title="The title block becomes a performance readout"
        why="SCENE, STUDY, MESH, SOLVER and REV were all stated elsewhere (A13); what only the viewport knows is what drawing costs."
        before={<p class="dl-note" style={{ margin: "0" }}>SCENE untitled · STUDY — · MESH — · SOLVER — · REV unsaved</p>}
        after={<p class="dl-note" style={{ margin: "0" }}>FPS idle · CPU — · RES 1696 × 1104 px · NODES 12</p>}
      />
      <Change
        n={16}
        title="Selects are built, not themed"
        why="A native select's open list is the system menu, out of reach of every token (K9)."
        before={<p class="dl-note" style={{ margin: "0" }}>The OS dropdown: system font, blue highlight, uppercase value</p>}
        after={<p class="dl-note" style={{ margin: "0" }}>The menu primitive: sheet, 24px options, a check on the current value</p>}
      />
      <Change
        n={17}
        title="Seven menus, with submenus"
        why="Ten titles spent a row on commands the rail and keys already reach (A13, N10)."
        before={<p class="dl-note" style={{ margin: "0" }}>File Edit View Create Shape Constrain Simulate Program Window Help</p>}
        after={<p class="dl-note" style={{ margin: "0" }}>File Edit View Model Simulate Window Help — Model › Create, Shape, Constrain</p>}
      />
      <Change
        n={18}
        title="One selection in the tree"
        why="An inspected operator and a selected object could both be lit (F2, C6)."
        before={<p class="dl-note" style={{ margin: "0" }}>revolve line 203 and bush_a both accent blocks</p>}
        after={<p class="dl-note" style={{ margin: "0" }}>Picking either clears the other</p>}
      />
      <Change
        n={19}
        title="The tree's root is the file"
        why="Two rows called scene meant two different things (A13)."
        before={<p class="dl-note" style={{ margin: "0" }}>scene · 12 objects … scene · Union · 7 operands</p>}
        after={<p class="dl-note" style={{ margin: "0" }}>scene.py · 12 objects … scene · Union · 7 operands</p>}
      />
      <Change
        n={20}
        title="42 ways to set text become 7"
        why="The audit counted 42 distinct size/weight/leading combinations (T0)."
        before={<p class="dl-note" style={{ margin: "0" }}>9, 10, 11, 12, 13 and 15px at 400, 500, 600 and 700, freely combined</p>}
        after={<p class="dl-note" style={{ margin: "0" }}>label · control · body · value · title · prose · code</p>}
      />
      <Change
        n={21}
        title="Spacing becomes five named distances"
        why="Seven raw steps were picked per rule, so the same job got different gaps (S0)."
        before={<p class="dl-note" style={{ margin: "0" }}>2 · 4 · 6 · 8 · 12 · 16 · 24, chosen case by case</p>}
        after={<p class="dl-note" style={{ margin: "0" }}>nudge 2 · gap 4 · inset 8 · stack 12 · section 16, by job</p>}
      />
      <Change
        n={22}
        title="Ultra renders at device resolution"
        why="The default tier's budget halved the pixels of a Retina viewport (F12)."
        before={<p class="dl-note" style={{ margin: "0" }}>1.6 M pixels: soft on a 2× screen</p>}
        after={<p class="dl-note" style={{ margin: "0" }}>4.2 M pixels: device resolution, about 10 ms a frame</p>}
      />
      <p class="dl-note">
        {`Also: the docs site moved off its stock theme onto these tokens.`}
      </p>
    </Section>
  );
}

export function DesignPage() {
  return (
    <>
      <header class="dl-top">
        <b>CJ</b>
        <strong>cadjoint</strong>
        <span>Design language · draft</span>
        <span class="dl-top-end">
          <Variations />
          <a class="dl-btn quiet h-sm" href="/">
            Playground
          </a>
        </span>
      </header>
      <div class="dl">
        <Nav />
        <main class="dl-main">
          <header>
            <h1>Design language</h1>
            <p>
              The system the playground is built from: one unit and one line box from which every
              size is derived, the atoms drawn with them, the primitives and arrangements made of
              the atoms, and the components made of those. The principles behind it come last. A rule that cannot be traced to an axiom is a bug in this page; a screen that
              contradicts a rule is a bug in the screen. This page is drawn with the same tokens and
              obeys the same rules. The measurements behind the choices are in{" "}
              <code>research/design-language.md</code>.
            </p>
          </header>
          <System />
          <Shape />
          <Primitives />
          <Layout0 />
          <Components />
          <Colour />
          <Type />
          <Language />
          <Space />
          <Icons />
          <Motion />
          <Viewport />
          <Layout />
          <Features />
          <Behaviour />
          <Axioms />
          <Changes />
        </main>
      </div>
    </>
  );
}

