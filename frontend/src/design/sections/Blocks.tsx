import { For } from "solid-js";
import { LAYOUT } from "../../tokens";
import { MenuItem, Plate, Rule, v, type SectionDef } from "../kit";

const LINES = [
  ["line", "every structural line: separators, the edge of a sheet, a field, a cell, the viewport's frame"],
  ["line-active", "this is the active one: a tab, focus, the selected edge"],
  ["line-pending", "not yet: an add button, a drop target, an expression"],
] as const;

const PLANES = [
  ["plane-ground", "ground: page, viewport"],
  ["plane-sheet", "sheet: panels, popovers, fields"],
  ["plane-step", "step: hover, sub-bars"],
] as const;

const MARKS = [
  ["dot", "done, a status"],
  ["ring", "a free point"],
  ["square", "held, an error"],
  ["triangle", "a warning"],
  ["✓", "on, current value"],
  ["›", "opens to the side"],
] as const;

const MARK_NAMES: Record<string, string> = { "✓": "check", "›": "chevron" };

const Mark = (props: { mark: string }) =>
  props.mark in MARK_NAMES ? <b class="t-body">{props.mark}</b> : <i class={`dl-mark ${props.mark}`} />;

const DISTANCES = Object.entries(LAYOUT) as [string, number][];

export const blocks: SectionDef = {
  id: "shape",
  title: "Building blocks",
  intro: "The atoms everything is drawn with, each a token generated from tokens.ts. Derived from A7.",
  body: () => (
    <>
      <Rule
        id="B0"
        why="Every panel, menu and overlay in the app is a combination of these; a thing that cannot be described with them is a bug."
        examples={
          <Plate label="the atlas: every atom, side by side">
            <div class="dl-atlas">
              <section>
                <span class="t-caps">lines</span>
                <For each={LINES}>
                  {([token]) => (
                    <div class="dl-atlas-line">
                      <i style={{ "border-top": `var(--${token})` }} />
                      <span class="t-body">{token === "line" ? "line" : token.slice(5)}</span>
                    </div>
                  )}
                </For>
              </section>
              <section>
                <span class="t-caps">planes</span>
                <For each={PLANES}>
                  {([token]) => (
                    <div class="dl-atlas-plane" style={{ background: v(token) }}>
                      <span class="t-body">{token.slice(6)}</span>
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
                  <For each={MARKS}>{([mark]) => <Mark mark={mark} />}</For>
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
                <For each={DISTANCES}>
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
        Six kinds of atom: three <b>lines</b>, three <b>planes</b>, two <b>blocks</b>, the <b>marks</b>, four{" "}
        <b>text</b> roles and three <b>distances</b>, no two alike.
      </Rule>
      <Rule
        id="B1"
        why="There are three things a line can say: here is an edge, this is the active one, this is not there yet. A border written any other way fails test/tokens.test.ts."
        examples={
          <Plate label="the three lines">
            <table class="dl-table">
              <tbody>
                <For each={LINES}>
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
        <b>line</b>, <b>active</b>, <b>pending</b>. One hairline for all structure; the 2px line means only{" "}
        <i>active</i>; dashed means only <i>not yet</i>.
      </Rule>
      <Rule
        id="B2"
        why="Elevation is a lighter sheet, not a shadow (E1); hover is the step (C1)."
        examples={
          <Plate label="the three planes" ground>
            <div class="dl-cluster-demo">
              <For each={PLANES}>
                {([token, job]) => (
                  <div class="dl-plane" style={{ background: v(token) }}>
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
        why="A block speaks by its fill, so there are as many as there are things to say."
        examples={
          <Plate label="the two blocks">
            <div class="l-cluster">
              <For each={["Sketch", "Run"]}>{(word) => <span class="dl-blk current t-caps">{word}</span>}</For>
              <For each={["thermal", "library"]}>{(word) => <span class="dl-blk kind t-caps">{word}</span>}</For>
            </div>
          </Plate>
        }
      >
        <b>current</b>: the accent under ink-on-accent type, the thing you act in or on. <b>kind</b>: ink under sheet
        type, a label of what something is.
      </Rule>
      <Rule
        id="B4"
        why="A mark says a state without a word or a hue (A4): circles mean a point, squares held, a triangle warns."
        examples={
          <Plate label="the marks">
            <div class="dl-marks">
              <For each={MARKS}>
                {([mark, job]) => (
                  <span>
                    <Mark mark={mark} />
                    <small class="t-caps">{MARK_NAMES[mark] ?? mark}</small>
                    <small class="t-prose">{job}</small>
                  </span>
                )}
              </For>
            </div>
          </Plate>
        }
      >
        <b>dot</b>, <b>ring</b>, <b>square</b>, <b>triangle</b>, <b>check</b>, <b>chevron</b>, and the corner caret
        that says a tool has more behind it.
      </Rule>
      <Rule
        id="E1"
        why="A shadow on paper is a picture of a card on a desk."
        examples={
          <>
            <Plate kind="do" label="a popover: sheet and edge" ground>
              <div class="dl-menu">
                <MenuItem mark="✓" label="Editor" />
                <MenuItem mark="✓" label="Object tree" />
                <MenuItem label="Meshes" />
              </div>
            </Plate>
            <Plate kind="dont" label="a floating card" ground>
              <div
                style={{
                  width: "calc(var(--unit) * 64)",
                  padding: "var(--inset)",
                  background: v("surface-float"),
                  "border-radius": "10px",
                  "box-shadow": "0 8px 24px rgba(0,0,0,.2)",
                }}
              >
                <For each={["✓ Editor", "✓ Object tree", "Meshes"]}>
                  {(item) => <div style={{ padding: "var(--gap)" }}>{item}</div>}
                </For>
              </div>
            </Plate>
          </>
        }
      >
        Radius is zero and there are no shadows (A7). Circles appear only where roundness means <i>a point, not a
        region</i>: status dots, vertex handles.
      </Rule>
      <Rule
        id="E2"
        why="A modal blocks everything else, so it alone dims what is behind it."
        examples={
          <Plate label="a dialog: scrim, sheet, prose, buttons bottom-right">
            <div class="dl-scrim">
              <div class="dl-dialog" role="dialog" aria-label="Overwrite">
                <h4>Overwrite heat_sink.py?</h4>
                <p>
                  The file on disk changed since you opened it. Saving replaces it, and undo cannot bring the disk version
                  back.
                </p>
                <footer>
                  <button type="button" class="dl-btn">
                    Cancel
                  </button>
                  <button type="button" class="dl-btn primary">
                    Overwrite
                  </button>
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
              <input
                class="dl-input"
                value="0.75"
                style={{ background: v("surface-bar-alt"), "border-color": "transparent" }}
              />
            </Plate>
          </>
        }
      >
        No inset controls. Inputs are the sheet with a <code>rule-strong</code> edge.
      </Rule>
      <Rule
        id="E4"
        why="One edge style for one idea: something you could make, a place you could drop."
        examples={
          <Plate label="dashed: an add button, a drop target, a read-only expression">
            <div class="dl-row-of">
              <button type="button" class="dl-btn add">
                + Boundary condition
              </button>
              <div
                style={{
                  padding: "var(--inset) var(--section)",
                  border: "var(--hairline) dashed var(--rule-strong)",
                  color: v("ink-3"),
                }}
              >
                Drop a material here
              </div>
              <input class="dl-input" readonly value="0.5 * base_w" style={{ width: "calc(var(--unit) * 32)" }} />
            </div>
          </Plate>
        }
      >
        The pending line (B1) marks an add button, a drop target, and a value written as an expression that the panel
        will not overwrite.
      </Rule>
    </>
  ),
};
