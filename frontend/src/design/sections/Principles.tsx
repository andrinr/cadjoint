import { For } from "solid-js";
import { fieldRampCss } from "../../simColors";
import { UNIT, type ChromeToken } from "../../tokens";
import { EscapeDemo, GreyscaleDemo, JobDemo, LiteralDemo } from "../demos";
import { FlyoutSpecimen } from "../FlyoutSpecimen";
import { BoxIcon, CursorIcon, CylinderIcon, MoveIcon, SphereIcon } from "../../components/icons";
import { Chip, DONT, Legend, Line, MiniPanel, Plate, Ratio, Rows, Rule, Seg, Status, v, type SectionDef } from "../kit";

const READINGS = [
  ["peak", "412.80", "K"],
  ["mean", "338.07", "K"],
] as const;

const MESH = [
  ["mesh", "sink-mesh"],
  ["cells", "48 211"],
  ["min jacobian", "0.31"],
] as const;

const BARS: { fg: ChromeToken; need: number | [number, number]; what: string }[] = [
  { fg: "ink-3", need: 4.5, what: "text" },
  { fg: "rule-strong", need: 3, what: "a mark that means something" },
  { fg: "rule", need: [1.4, 2.8], what: "furniture: visible, not competing" },
];

/** A12 Don't: the open group as a second card, larger, off its parent's row. */
const DetachedFlyout = () => (
  <div class="dl-fly-rail">
    <span class="dl-fly-cell">
      <CursorIcon size={16} />
    </span>
    <span class="dl-fly-group">
      <span class="dl-fly-cell open">
        <BoxIcon size={16} />
      </span>
      <span class="dl-fly-detached">
        <For each={[BoxIcon, SphereIcon, CylinderIcon]}>
          {(Icon) => (
            <span class="dl-fly-cell">
              <Icon size={16} />
            </span>
          )}
        </For>
      </span>
    </span>
    <span class="dl-fly-cell">
      <MoveIcon size={16} />
    </span>
  </div>
);

export const principles: SectionDef = {
  id: "axioms",
  title: "Principles",
  intro:
    "The reasons behind every rule above. A1–A9 are about how things look, A2 and A10–A13 how they behave; when two rules conflict, the one from the earlier principle wins.",
  body: () => (
    <>
      <Rule
        id="A1"
        why="So the chrome is quiet: paper, ink and rules. Saturated colour is spent on the geometry and on readings."
        examples={
          <>
            <Plate kind="do" label="the reading is the loudest thing" ground>
              <MiniPanel kicker="fem" title="Results">
                <Legend name="temperature" from="293.2 K" to="412.8 K" ramp={fieldRampCss()} />
                <Rows rows={READINGS} />
              </MiniPanel>
            </Plate>
            <Plate kind="dont" label="chrome competes with the reading" ground>
              <MiniPanel kicker="fem" title="Results" kickerFill="accent">
                <div class="dl-legend">
                  <b style={{ color: v("accent-press") }}>temperature</b>
                  <span>293.2 K</span>
                  <i style={{ background: fieldRampCss() }} />
                  <span>412.8 K</span>
                </div>
                <div class="dl-rows">
                  <For each={READINGS}>
                    {([label, value, unit], index) => (
                      <div>
                        <span style={{ color: DONT.keyword }}>{label}</span>
                        <b style={{ color: v(index() === 0 ? "ok" : "warn") }}>{value}</b>
                        <small>{unit}</small>
                      </div>
                    )}
                  </For>
                </div>
              </MiniPanel>
            </Plate>
          </>
        }
      >
        <b>The part and its numbers are the subject.</b> The interface is the sheet they are printed on; nothing in the
        chrome is more salient than the geometry or a measured value.
      </Rule>

      <Rule
        id="A2"
        why="So every panel is an editor of literals, and the editor is never out of the loop."
        examples={
          <Plate kind="try" label="type a value and press Enter: the panel writes its literal">
            <LiteralDemo />
          </Plate>
        }
      >
        <b>The program is the truth.</b> Every edit made anywhere is an edit to the source text, and the screen is a
        rendering of that text. The only state the program does not hold is the view: camera, layout, selection.
      </Rule>

      <Rule
        id="A3"
        why="So a reader can trust a hue: if it is coloured it means something, and one thing (C3)."
        examples={
          <>
            <Plate kind="do" label="hue only where a role is: the one red is the typo">
              <pre class="dl-code">
                <Line n={116}>
                  base_l = <span class="kw">Vector2</span>(value=[-0.9, 0.0], <span class="free">free=True</span>, name=
                  <span class="str">"base_l"</span>)
                </Line>
                <Line n={117}>
                  <span class="com"># the deck stays level</span>
                </Line>
                <Line n={118}>
                  <span class="kw">from</span> cadjoint <span class="kw">import</span> <span class="err">Solidd</span>
                </Line>
              </pre>
            </Plate>
            <Plate kind="dont" label="hue as decoration: the class is red, the typo unmarked">
              <pre class="dl-code">
                <Line n={116}>
                  base_l = <span style={{ color: DONT.className }}>Vector2</span>(value=[
                  <span style={{ color: DONT.number }}>-0.9</span>, <span style={{ color: DONT.number }}>0.0</span>], free=
                  <span style={{ color: DONT.number }}>True</span>, name=<span style={{ color: DONT.string }}>"base_l"</span>)
                </Line>
                <Line n={117}>
                  <span class="com"># the deck stays level</span>
                </Line>
                <Line n={118}>
                  <span style={{ color: DONT.keyword }}>from</span> cadjoint <span style={{ color: DONT.keyword }}>import</span>{" "}
                  <span style={{ color: DONT.className }}>Solidd</span>
                </Line>
              </pre>
            </Plate>
          </>
        }
      >
        <b>Hue is meaning.</b> Every hue on screen belongs to exactly one role in a closed table. Anything without a role
        is ink on paper.
      </Rule>

      <Rule
        id="A4"
        why="So colour-blind readers and greyscale screenshots lose nothing."
        examples={
          <Plate kind="try" label="toggle the plate to greyscale: every state still reads">
            <GreyscaleDemo>
              <div class="dl-stack">
                <div class="dl-row-of">
                  <Status>compiled · rev 14</Status>
                  <Status stale>out of date</Status>
                  <span class="dl-badge free">17 free</span>
                  <Chip violated>distance · P1–P2 = 1.8 ≠ 1.74</Chip>
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
        <b>Never by hue alone.</b> Every role also has a carrier that survives greyscale: a word, a shape, a position or a
        weight.
      </Rule>

      <Rule
        id="A5"
        why="So legibility is asserted, not judged: every ratio on this page is computed from the shipped tokens."
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
                <For each={BARS}>
                  {(row) => (
                    <tr>
                      <td>
                        <i class="dl-dot" style={{ background: v(row.fg) }} />
                        {row.fg}
                      </td>
                      <td class="prose">{row.what}</td>
                      <td class="num">
                        <Ratio fg={row.fg} bg="surface-base" need={row.need} />
                      </td>
                      <td class="num">
                        <Ratio fg={row.fg} bg="surface-float" need={row.need} />
                      </td>
                    </tr>
                  )}
                </For>
              </tbody>
            </table>
          </Plate>
        }
      >
        <b>Contrast is a number.</b> Text clears 4.5:1 on its ground, a mark that carries meaning 3:1, and furniture sits
        in a band below that, 1.4–2.8:1.
      </Rule>

      <Rule
        id="A6"
        why={`So everything lines up without being told to. The grid behind the plate is the unit, ${UNIT}px.`}
        examples={
          <Plate label="a row on the unit grid">
            <div class="dl-unit-grid" style={{ padding: "var(--section)" }}>
              <div class="dl-row-of">
                <button type="button" class="dl-btn">
                  Button · 32
                </button>
                <input class="dl-input" style={{ width: "calc(var(--unit) * 20)" }} value="0.75" />
                <span class="dl-tag">tag · 16</span>
                <button type="button" class="dl-btn h-sm">
                  Small · 24
                </button>
              </div>
            </div>
          </Plate>
        }
      >
        <b>One unit.</b> Every length is a multiple of the unit (Y1). Half a unit is allowed inside a control; the one
        shorter length is the hairline, 1px.
      </Rule>

      <Rule
        id="A7"
        why="So a dense screen stays calm: a line costs less attention than a box, and paper has no depth to fake."
        examples={
          <>
            <Plate kind="do" label="rules separate, edges group">
              <Rows rows={MESH} />
            </Plate>
            <Plate kind="dont" label="boxes, shadows, radii">
              <div class="dl-stack">
                <For each={MESH}>
                  {([label, value]) => (
                    <div
                      style={{
                        display: "flex",
                        "justify-content": "space-between",
                        padding: "var(--inset)",
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
        <b>Structure is drawn, not stacked.</b> Things are separated by rules and grouped by shared edges. No shadow, no
        radius, no translucency behind text.
      </Rule>

      <Rule
        id="A8"
        why="So a glance tells you who is speaking: mono is the instrument, sans is a person."
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
        <b>Two voices.</b> What the program says (names, values, units, the instrument's labels) is mono, and its strings
        are set as written. What a person says is sans. Capitals are for labels of a word or two.
      </Rule>

      <Rule id="A9" why="So movement carries information (M1–M4).">
        <b>Motion shows causality.</b> Something moves only to show that one thing happened because of another. Anything
        else is still.
      </Rule>

      <Rule
        id="A10"
        why="So you never wonder whether something is happening."
        examples={
          <Plate kind="try" label="run the study; stop it while it runs">
            <JobDemo />
          </Plate>
        }
      >
        <b>Every action answers.</b> It is acknowledged within 100 ms, says what it is doing while it runs, and ends in a
        visible state: done, failed with the reason, or cancelled.
      </Rule>

      <Rule
        id="A11"
        why="So nothing is a trap."
        examples={
          <Plate kind="try" label="focus the plate and press Esc repeatedly">
            <EscapeDemo />
          </Plate>
        }
      >
        <b>Every step can be taken back.</b> An edit is undoable, a gesture abandonable, a mode escapable, one step at a
        time.
      </Rule>

      <Rule
        id="A12"
        why="So the hand learns the screen: a control that moves has to be found again every time."
        examples={
          <>
            <Plate kind="do" label="it opens at its parent's row, at its own size, pointing back" ground>
              <FlyoutSpecimen />
            </Plate>
            <Plate kind="dont" label="a second card, somewhere near" ground>
              <DetachedFlyout />
            </Plate>
          </>
        }
      >
        <b>Things stay where they are.</b> An element keeps its place and size when it opens, changes state, or the mode
        changes, and it opens from where it is. Space for what will arrive is reserved, and nothing covers what it does
        not own.
      </Rule>

      <Rule
        id="A13"
        why="So what is on screen is worth reading: a fact shown twice is one fact and one distraction, and the copies can disagree."
        examples={
          <>
            <Plate kind="do" label="the mode is said by the switcher alone">
              <div class="dl-stack">
                <Seg active="Sketch" />
                <span class="dl-hint">Distance: choose a second point in the same sketch</span>
              </div>
            </Plate>
            <Plate kind="dont" label="said again at the foot of the viewport">
              <div class="dl-stack">
                <Seg active="Sketch" />
                <span class="dl-row-of">
                  <span class="dl-tag">sketch</span>
                  <span class="dl-hint">Drag handles or the gizmo · Drag to orbit · Right-drag to pan</span>
                </span>
              </div>
            </Plate>
          </>
        }
      >
        <b>Say it once.</b> Each fact is stated in one place, the place that owns it. A command may have several ways in
        but one name; a state has one display; a resting instruction is not shown at all.
      </Rule>
    </>
  ),
};
