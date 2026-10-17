import { AnchorsDrawing, DeskDrawing, LevelsDrawing, PanelAnatomy } from "../LayoutDrawings";
import { Plate, Rule, type SectionDef } from "../kit";

export const layout: SectionDef = {
  id: "layout-primitives",
  title: "Layout",
  intro: "Where things go, at every scale. The drawings are generated from the module (Y1–Y3), so a change to it redraws them.",
  body: () => (
    <>
      <Rule
        id="L1"
        why="The level of anything can then be read from how it is separated."
        examples={
          <Plate label="the six levels, each separated in one way" ground>
            <LevelsDrawing />
          </Plate>
        }
      >
        Six levels, each separating its children in one way: <b>desk</b> windows share an edge, a <b>window</b>'s
        sheet has an inset, <b>blocks</b> are a section apart, <b>list</b> rows have a line between them, a{" "}
        <b>row</b> is a cell high. Space never separates rows; a line never separates blocks.
      </Rule>
      <Rule
        id="L2"
        why="Shared tracks put every value in the app on one column, so two panels read as one table."
        examples={
          <Plate label="a panel on the panel grid, with its rhythm" ground>
            <PanelAnatomy />
          </Plate>
        }
      >
        Every sheet has four tracks inside its inset: <b>label</b> 96 · <b>value</b> the rest · <b>unit</b> 40 ·{" "}
        <b>action</b> 24, an inset apart. Rows take label | value | unit (| action); heads, prose and chip clusters span
        all four; a sheet's own actions end-align on the action track.
      </Rule>
      <Rule
        id="L3"
        why="Every panel is then a whole number of units tall, and the layout grid (⇧G) shows at once anything that is not."
      >
        <b>Vertical rhythm.</b> Every height is a sum of line boxes and distances: a head is 16 + 4 + 20 = 40, a row 24, a
        block apart 16, a bar 32.
      </Rule>
      <Rule id="L4" why="Edges are what the eye groups by; centring breaks them.">
        <b>Alignment.</b> One left and one right edge per sheet, an inset in. Names and labels hang left; values and
        actions hang right; numbers align on their last digit. Nothing is centred except a label in its own cell.
      </Rule>
      <Rule
        id="L5"
        why="Fixed proportions keep the viewport where it is when the mode changes (N2)."
        examples={
          <Plate label="the desk at its default proportions" ground>
            <DeskDrawing />
          </Plate>
        }
      >
        <b>The desk.</b> Three columns: the program 460, the viewport the rest, the column 320. Four rows: the menu bar
        32, the toolbar 40, the desk, the tray 24.
      </Rule>
      <Rule
        id="L6"
        why="The centre third is the part's (A1); a piece anchored to a corner has a place before it is drawn."
        examples={
          <Plate label="the viewport's anchors" ground>
            <AnchorsDrawing />
          </Plate>
        }
      >
        <b>The viewport.</b> Furniture hangs from anchors a section in from the edges and stacks inward from them; what
        goes where is N6.
      </Rule>
      <Rule id="L7" why="Choosing an arrangement chooses its distance.">
        <b>The arrangements</b>, generated as classes from tokens.ts: <b>l-inset</b> (a sheet's inset), <b>l-stack</b>{" "}
        (blocks a section apart), <b>l-ruled</b> (rows with a line between), <b>l-cluster</b> (cells a gap apart,
        wrapping), <b>l-row</b> (the panel tracks), <b>l-split</b> (a head: start and end).
      </Rule>
    </>
  ),
};
