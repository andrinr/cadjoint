import { For } from "solid-js";
import { SHORTCUT_GROUPS } from "../../shortcuts";
import { DragDemo, JobDemo, ModeDemo, SelectionDemo, UndoDemo } from "../demos";
import { Plate, Rows, Rule, v, type SectionDef } from "../kit";

export const behaviour: SectionDef = {
  id: "behaviour",
  title: "Behaviour",
  intro: "The functional rules, from A2, A10 and A11.",
  body: () => (
    <>
      <h3>The program is the truth</h3>
      <Rule id="F1" why="An expression is not a number: overwriting it with one would silently break the program.">
        A panel edits literals only (A2). An argument written as an expression is shown as written and read-only (K3).
      </Rule>
      <Rule
        id="F2"
        why="One thing selected is one thing, wherever you look at it."
        examples={
          <Plate kind="try" label="click a row, a shape or a line: all three follow">
            <SelectionDemo />
          </Plate>
        }
      >
        One selection: viewport, object tree, properties and editor show the same thing, and the editor scrolls to it
        without taking focus. A precise selection reveals its literal; a whole object its declaration.
      </Rule>
      <Rule id="F3" why="A stale picture shown as current is a wrong answer.">
        The screen says whether it matches the code (K7). A stale value is marked in place, never silently kept. The newest
        edit wins: an older answer arriving late is discarded.
      </Rule>

      <h3>Direct manipulation</h3>
      <Rule
        id="F4"
        why="One gesture is one change to the program, and abandoning it leaves the program as it was."
        examples={
          <Plate kind="try" label="drag the handle; press Esc during the drag">
            <DragDemo />
          </Plate>
        }
      >
        A drag shows its effect live and writes the source once, on release: one gesture, one undo step. Escape during
        a gesture puts the value back. A snap draws what it snapped to, for as long as it holds.
      </Rule>

      <h3>Modes and tools</h3>
      <Rule
        id="F5"
        why="One armed tool, so a click in the viewport means one thing."
        examples={
          <Plate kind="try" label="switch mode, arm a tool">
            <ModeDemo />
          </Plate>
        }
      >
        Modes are a pipeline, Model → Sketch → Simulate; a mode changes the tool set, panels and shortcuts wholesale. One
        tool is armed at a time, as the current block; finishing or cancelling it returns to Select. A grouped tool
        remembers its last choice.
      </Rule>

      <h3>Escape and undo</h3>
      <Rule id="F6" why="A11's ladder, in order: gesture, command, tool, selection, mode. When nothing applies, Escape does nothing.">
        Escape cancels one thing per press, most specific first.
      </Rule>
      <Rule
        id="F7"
        why="A dialog interrupts every deletion to guard against a few; undo guards all of them without asking."
        examples={
          <Plate kind="try" label="remove a constraint, then undo">
            <UndoDemo />
          </Plate>
        }
      >
        Undo is over the source text: each run, patch and gesture is one step; in the editor it is the editor's own history.
        A deletion that can be undone is not confirmed; confirmation (E2) is for what undo cannot reach.
      </Rule>

      <h3>Feedback and long work</h3>
      <Rule
        id="F8"
        why="A failure is a result, said in the program's language (K6)."
        examples={
          <Plate kind="try" label="a job that fails at solve">
            <JobDemo fail />
          </Plate>
        }
      >
        After about a second, work becomes a job: its stage in words and a stop control (A10). A result outlives its
        panel. No indeterminate spinner over legible content, no percentage of an unknown total.
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
        A value commits on Enter or blur, and only if valid. Units sit beside the field, never typed. Numbers keep their
        written precision.
      </Rule>

      <h3>Keyboard</h3>
      <Rule
        id="F10"
        why="The sheet is rendered from shortcuts.ts, the record the key handlers read, so it cannot drift from the keys."
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
        Every action is reachable from the keyboard and listed in the Help sheet. Single-key shortcuts act only when the
        viewport has focus. Focus order follows reading order, and focus is always visible as the active line.
      </Rule>

      <h3>Layout</h3>
      <Rule
        id="F11"
        examples={
          <>
            <Plate kind="do" label="the slot exists before the data">
              <Rows
                rows={[
                  ["peak", "—"],
                  ["mean", "—"],
                ]}
              />
            </Plate>
            <Plate kind="dont" label="rows appear, the panel jumps">
              <p class="dl-note" style={{ margin: "0" }}>
                (empty until the solve returns, then three rows push everything below them down)
              </p>
            </Plate>
          </>
        }
      >
        Windows dock, tab, float and minimise; the layout is remembered per mode, and Reset layout restores the default.
        Space for arriving data is reserved (A12).
      </Rule>

      <h3>Defaults</h3>
      <Rule
        id="F12"
        why="The first picture someone sees is the part (A1); a slower machine steps down, and the performance readout says when."
      >
        Defaults are the best the machine can sustain: the Ultra tier, at device resolution. A user lowers a setting; they
        never have to discover that they could raise it.
      </Rule>
    </>
  ),
};
