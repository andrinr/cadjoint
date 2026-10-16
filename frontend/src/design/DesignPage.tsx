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
  DURATIONS,
  GRID_CONTROL_HEIGHTS,
  SPACE,
  TRACKING,
  TYPE_SCALE,
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
import { ViewportSpecimen } from "./ViewportSpecimen";

const SECTIONS = [
  { id: "axioms", num: "1", title: "Axioms" },
  { id: "colour", num: "2", title: "Colour" },
  { id: "type", num: "3", title: "Type" },
  { id: "space", num: "4", title: "Space and size" },
  { id: "shape", num: "5", title: "Shape and elevation" },
  { id: "icons", num: "6", title: "Iconography" },
  { id: "motion", num: "7", title: "Motion" },
  { id: "components", num: "8", title: "Components" },
  { id: "viewport", num: "9", title: "Viewport" },
  { id: "behaviour", num: "10", title: "Behaviour" },
  { id: "language", num: "11", title: "Language" },
  { id: "changes", num: "12", title: "What changes" },
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
  keyword: CHROME.info,
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

function Axioms() {
  return (
    <Section
      id="axioms"
      num="1"
      title="Axioms"
      intro={
        <>
          Eleven sentences. Everything after this section is derived from them, and each
          derived rule names the axiom it comes from. When two rules conflict, the one from the
          earlier axiom wins.
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
                      <span style={{ color: v("info") }}>peak</span>
                      <b style={{ color: v("ok") }}>412.80</b>
                      <small>K</small>
                    </div>
                    <div>
                      <span style={{ color: v("info") }}>mean</span>
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
        why="So a reader can trust a hue: if it is coloured, it means something, and it means one thing. The table is in §2."
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
            <div class="dl-unit-grid" style={{ padding: "var(--space-6)" }}>
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
                        padding: "var(--space-3) var(--space-4)",
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
        mono. What a person says (sentences) is sans.
      </Rule>

      <Rule id="A9" why="So movement carries information. The demos are in §7.">
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
    </Section>
  );
}

function Colour() {
  const bcEntries = Object.entries(BC_TYPE_COLORS) as [string, readonly [number, number, number]][];
  return (
    <Section
      id="colour"
      num="2"
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
          <div style={{ "padding-top": "var(--space-4)" }}>
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
                <span style={{ padding: "var(--space-3)", border: `var(--hairline) solid ${v("accent")}` }}>aluminium</span>
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
        <code>accent-press</code> is its pressed state and the only accent-coloured mark.
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
      <Rule id="C8" why="ok and axis-y are both green, traction and axis-x both red: they never label the same thing in the same place.">
        Two roles may share a hue family only across zones or modes.
      </Rule>
    </Section>
  );
}

function Type() {
  const sizes = Object.entries(TYPE_SCALE) as [string, number][];
  const jobs: Record<string, string> = {
    "text-3xs": "tracked uppercase labels and kickers",
    "text-2xs": "dense secondary text, selector expressions",
    "text-xs": "control labels, tabs, menu items",
    "text-sm": "body; the default",
    "text-md": "values the user reads",
    "text-lg": "panel titles, and nothing else",
  };
  const tracking = Object.values(TRACKING);
  return (
    <Section id="type" num="3" title="Type" intro="System mono and system sans, six sizes, three weights.">
      <Rule
        id="T1"
        why="Between 9 and 13px a 1px step is about 10% of the size, just above the visible threshold. Then one jump, to the single title size."
        single
        examples={
          <Plate label="the scale, with its tracking">
            <div class="dl-type">
              <For each={sizes}>
                {([token, px], index) => (
                  <div>
                    <code>
                      {token} · {tracking[index()]}
                    </code>
                    <code>{px}px</code>
                    <span
                      style={{
                        "font-size": `var(--${token})`,
                        "letter-spacing": `var(--tracking-${token.replace("text-", "")})`,
                        "text-transform": index() === 0 ? "uppercase" : "none",
                        "font-weight": index() === 5 ? "var(--weight-semibold)" : "var(--weight-medium)",
                      }}
                    >
                      {jobs[token]}
                    </span>
                  </div>
                )}
              </For>
            </div>
          </Plate>
        }
      >
        Six sizes: 9, 10, 11, 12, 13 and 15px. Tracking is a function of size, from 0.16em at 9px
        down to 0.04em at 15px. Weights 400, 500, 600; 700 only on an accent block.
      </Rule>
      <Rule
        id="T2"
        why="A program's string is a value, so it is quoted, not restyled. Uppercasing it would change what it says."
        examples={
          <>
            <Plate kind="do" label="furniture tracked, names verbatim">
              <div class="dl-row-of">
                <span class="dl-tag">thermal</span>
                <span style={{ "font-weight": "var(--weight-semibold)" }}>sink-conduction</span>
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
              <div class="dl-stack" style={{ gap: "var(--space-2)", "font-family": "var(--font-sans)" }}>
                <For each={[["peak (K)", "412.80"], ["mean (K)", "338.07"], ["min (K)", "293.21"], ["Δ (K)", "119.59"]]}>
                  {([label, value]) => (
                    <div>
                      <span style={{ color: v("ink-3") }}>{label}: </span>
                      <b style={{ "font-weight": "var(--weight-medium)" }}>{value}</b>
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
  const spaces = Object.entries(SPACE) as [string, number][];
  const heights = Object.entries(GRID_CONTROL_HEIGHTS) as [string, number][];
  const jobs: Record<string, string> = {
    "control-tag": "non-interactive tags",
    "control-sm": "the smallest clickable: dense rows",
    "control-md": "inputs, buttons in panels",
    "control-lg": "toolbar, tool rail",
    "control-xl": "the top bar",
  };
  return (
    <Section id="space" num="4" title="Space and size" intro={`Every length is a multiple of ${UNIT}px (A6).`}>
      <Rule
        id="S1"
        why="When a panel runs out of room, density comes from type (a 9px label over a 13px value), not from tighter gaps. Padding under 8px reads as a mistake."
        single
        examples={
          <Plate label="the spacing scale, bars drawn at 4×">
            <div class="dl-ruler">
              <For each={spaces}>
                {([token, px]) => (
                  <div>
                    <code>{token}</code>
                    <b>{px}px</b>
                    <i style={{ width: `${px * 4}px` }} />
                  </div>
                )}
              </For>
            </div>
          </Plate>
        }
      >
        Spacing is 2 · 4 · 6 · 8 · 12 · 16 · 24, half a unit to six units. The half-unit steps live
        inside controls.
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
              <div style={{ padding: "var(--space-4)", border: "var(--stroke-active) solid var(--rule-heavy)" }}>
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
    <Section id="shape" num="5" title="Shape and elevation" intro="Derived from A7: things are drawn, not stacked.">
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
              <div style={{ width: "calc(var(--unit) * 64)", padding: "var(--space-3)", background: v("surface-float"), "border-radius": "10px", "box-shadow": "0 8px 24px rgba(0,0,0,.2)" }}>
                <div style={{ padding: "var(--space-2)" }}>✓ Editor</div>
                <div style={{ padding: "var(--space-2)" }}>✓ Object tree</div>
                <div style={{ padding: "var(--space-2)" }}>Meshes</div>
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
              <div style={{ padding: "var(--space-4) var(--space-6)", border: "var(--hairline) dashed var(--rule-strong)", color: v("ink-3") }}>
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
    <Section id="icons" num="6" title="Iconography" intro="The playground's icon set, as shipped, at 16px.">
      <Rule
        id="I1"
        why="The stroke matches the hairline in feel, so an icon and a rule look drawn by one pen. Colour is not available to icons (A3): an icon that needs a colour needs a word."
        single
        examples={
          <Plate label="the set as shipped; its round caps are Δ9">
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
        Every tool-rail icon has a text label at 1280px and wider. A destructive action always
        carries the word. Icons never appear inside the viewport: gizmos and handles are geometry.
      </Rule>
    </Section>
  );
}

function Motion() {
  return (
    <Section id="motion" num="7" title="Motion" intro="Derived from A9. Three durations, named by what they mean, and one easing family.">
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

function Components() {
  return (
    <Section id="components" num="8" title="Components" intro="Each component is made only of the tokens above. These are the guide's primitives; the playground adopts them.">
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
                        <button type="button" class={`dl-btn ${cls} pressed`} style={cls === "primary" ? { background: v("accent-press"), "border-color": v("accent-press") } : undefined}>
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
        An input is the sheet with a <code>rule-strong</code> edge and right-aligned tabular mono.
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
        shortcuts right-aligned in <code>ink-3</code>. The open menu's title is the accent block.
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
              <div style={{ padding: "var(--space-4)", background: v("ink"), color: v("surface-float"), "border-radius": "6px", "font-size": "var(--text-2xs)" }}>
                Traceback (most recent call last): File "worker.py", line 88, in compile … NameError: name 'Solidd' is not defined
                <div style={{ "margin-top": "var(--space-2)", color: v("ink-3") }}>disappears in 4 s</div>
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
    </Section>
  );
}

function Viewport() {
  return (
    <Section id="viewport" num="9" title="Viewport" intro="The viewport is an instrument face (A1). Its overlays are achromatic unless they are data or direction (C7).">
      <Rule
        id="V1"
        why="Free versus held is shape, so the free-parameter hue stays in the chrome and the viewport keeps its colour for readings."
        single
        examples={
          <Plate label="sketch mode" ground>
            <ViewportSpecimen />
          </Plate>
        }
      >
        A free vertex is a hollow circle; a held vertex is a filled square; the selected vertex is a
        filled <code>ink</code> circle. Edges go <code>ink-3</code> → <code>ink-2</code> (hover) →{" "}
        <code>ink</code> (selected, 2px); a locked edge is a dashed <code>rule</code>. Dimension
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
        <code>&gt;</code> when the view is not to scale. The title block lists SCENE, STUDY, MESH,
        SOLVER, REV, with <code>—</code> where a field is empty. The hint bar names the mode as an
        inverse tag, then one sentence about what the pointer does now.
      </Rule>
    </Section>
  );
}

function Behaviour() {
  return (
    <Section id="behaviour" num="10" title="Behaviour" intro="The functional rules. They come from A2 (the program is the truth), A10 (every action answers) and A11 (every step can be taken back).">
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
    </Section>
  );
}

function Language() {
  const rows = [
    ["furniture", "MESH · SOLVE · BOUNDARY CONDITION", "uppercase tracked mono"],
    ["program strings", "fin comb · sink-conduction · base_l", "verbatim mono"],
    ["sentences", "Drag a swatch onto an object", "sans, sentence case, no period on one clause"],
    ["quantities", "1.00 m · 412.8 K · 2700 kg/m³", "SI, unit as a symbol after the value"],
    ["empty", "—", "an em dash; never 0, N/A or blank"],
  ];
  return (
    <Section id="language" num="11" title="Language" intro="Words are part of the interface, and they follow A8.">
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
      num="12"
      title="What changes"
      intro="What the playground does today that these rules retire. Each pair is now, then the rule."
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
            <span class="dl-before-chip" style={{ background: v("info"), color: v("surface-float") }}>
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
            <span class="dl-badge" style={{ background: v("info"), color: v("surface-float") }} title="constraints">
              17
            </span>
            <span style={{ color: v("info-ink") }}>FIN COMB</span>
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
          <div style={{ display: "inline-block", padding: "var(--space-4)", border: `var(--hairline) solid ${v("accent")}` }}>
            aluminium
            <br />
            <small style={{ color: v("ink-3") }}>1 applied</small>
          </div>
        }
        after={
          <div style={{ display: "inline-grid", border: `var(--hairline) solid ${v("rule-strong")}` }}>
            <span style={{ padding: "var(--space-4)" }}>
              <small style={{ color: v("ink-3") }}>1 applied</small>
            </span>
            <span style={{ padding: "var(--space-2) var(--space-4)", background: v("accent"), color: v("ink-on-accent"), "font-weight": "var(--weight-bold)" }}>
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
            <PolygonIcon size={16} />
            <MoveIcon size={16} />
            <DistanceIcon size={16} />
            <span style={{ color: v("ink-3") }}>round caps, filled handles</span>
          </div>
        }
        after={
          <div class="dl-row-of">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.25" stroke-linecap="square" stroke-linejoin="miter">
              <path d="M3.5 12.5 L2.5 6.5 L8 2.5 L13.5 6.5 L12.5 12.5 Z" />
            </svg>
            <span style={{ color: v("ink-3") }}>to be redrawn</span>
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
      <p class="dl-note">
        {`Also: the docs site moves off its stock theme onto these tokens, and the playground adopts the dl- primitives on this page as its components.`}
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
        <a class="dl-btn quiet h-sm" href="/">
          Playground
        </a>
      </header>
      <div class="dl">
        <Nav />
        <main class="dl-main">
          <header>
            <h1>Design language</h1>
            <p>
              The rules the playground is built from: eleven axioms, and everything derived from
              them. A rule that cannot be traced to an axiom is a bug in this page; a screen that
              contradicts a rule is a bug in the screen. This page is drawn with the same tokens and
              obeys the same rules. The measurements behind the choices are in{" "}
              <code>research/design-language.md</code>.
            </p>
          </header>
          <Axioms />
          <Colour />
          <Type />
          <Space />
          <Shape />
          <Icons />
          <Motion />
          <Components />
          <Viewport />
          <Behaviour />
          <Language />
          <Changes />
        </main>
      </div>
    </>
  );
}

