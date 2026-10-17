import { For } from "solid-js";
import { DeskMap } from "../DeskMap";
import { MenuItem, Plate, Rule, type SectionDef } from "../kit";

/** Window menu entry → the name on its tab. */
const ONE_NAME = [
  ["scene.py", "scene.py"],
  ["Objects", "Objects"],
  ["Properties", "Properties"],
  ["Sketch", "Sketch"],
] as const;
const TWO_NAMES = [
  ["Editor", "scene.py"],
  ["Object tree", "Objects"],
  ["Properties", "Properties"],
  ["Sketch panel", "Sketch"],
] as const;

const HOMES = [
  ["one declaration", "on its card or row", "× on a constraint chip, Delete study, Solve on a study card"],
  ["the whole program", "the toolbar, right", "Run, the program status, Stop on the job chip"],
  ["the application", "the menu bar", "Save as, Reset layout, Keyboard shortcuts"],
  ["the picture", "the tool rail, the keyboard", "Polygon (P), Move (G), views (1 3 7)"],
  ["the pipeline", "the toolbar, left", "Model · Sketch · Simulate (M)"],
] as const;

const WindowMenu = (props: { names: readonly (readonly [string, string])[] }) => (
  <div class="dl-menu flush">
    <For each={props.names}>{([menu, tab]) => <MenuItem mark="✓" label={menu} keys={`tab: ${tab}`} />}</For>
  </div>
);

export const navigation: SectionDef = {
  id: "layout",
  title: "Where things are",
  intro: "A map learned once: every region answers one question, every window has one name, every action one home.",
  body: () => (
    <>
      <Rule
        id="N1"
        why="Reading order is causal order (A2): the program makes the picture, and the right column shows what follows from the selection."
        examples={
          <Plate kind="try" label="the default desk; pick a mode in it, point at a region">
            <DeskMap />
          </Plate>
        }
      >
        The desk has fixed regions (L5): left to right, the program, the picture, the consequences. The toolbar holds the
        mode on the left, the program status and Run on the right.
      </Rule>
      <Rule
        id="N2"
        why="Selecting a sketch enters Sketch mode under the pointer; if the viewport moved with the mode, the model would jump away from the click."
      >
        The columns are the same in every mode. A mode changes what is in the right column, not where the viewport is.
      </Rule>
      <Rule id="N3" why="Top to bottom is cause to effect, as left to right is.">
        The right column runs from the selection (Objects) to its arguments (Properties) to what depends on it. In
        Simulate: setup (Studies, Meshes) above outcomes (Results, Optimize).
      </Rule>
      <Rule
        id="N4"
        why="A thing with two names is two things to a newcomer; the Window menu is the index of the desk, so it uses the tabs' words."
        examples={
          <>
            <Plate kind="do" label="one name per place">
              <WindowMenu names={ONE_NAME} />
            </Plate>
            <Plate kind="dont" label="the menu and the tab disagree">
              <WindowMenu names={TWO_NAMES} />
            </Plate>
          </>
        }
      >
        Every window is reachable from the Window menu by the name on its tab, and nothing is lost by closing it.
      </Rule>
      <Rule
        id="N5"
        why="You find an action by asking what it acts on; two homes or two names make both harder to find."
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
                <For each={HOMES}>
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
        An action lives on what it acts on. A menu item shows its shortcut, so the menu teaches the keyboard.
      </Rule>
      <Rule id="N6" why="Corners are learned once; a readout that moves has to be looked for.">
        The viewport's furniture sits on L6's anchors: the tool rail top-left with the readout beside it; projection and
        render settings over the view cube, top-right; the legend bottom-left, the hint bottom-centre, the performance
        readout bottom-right. Nothing else is pinned there, and nothing covers anything else.
      </Rule>
      <Rule id="N7" why="Where am I, and how do I get back, should be answerable without trying something.">
        The switcher names the mode, the toolbar names the selection, Escape walks back one step (F6), and Reset layout
        restores the desk (F11).
      </Rule>
      <Rule
        id="N8"
        why="A monitor that takes a column by default is in the way, but a solve keeps running while you model, so it stays one click away."
      >
        Monitors and browsers (Processes, Scenes) are parked in the tray, named, one click away.
      </Rule>
    </>
  ),
};
