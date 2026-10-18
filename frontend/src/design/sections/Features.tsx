import { FeatureIndex, MenuBarSpecimen } from "../FeatureIndex";
import { Plate, Rule, type SectionDef } from "../kit";

export const features: SectionDef = {
  id: "features",
  title: "Every feature",
  intro: "Every feature and every place it can be reached; the table and the menu bar are rendered from design/features.ts.",
  body: () => (
    <>
      <Rule
        id="N9"
        why="The menu bar is where you look when you do not know where to look; a command only on a rail or a key is found only by those who know it."
      >
        The menu bar is the complete index: every command has a menu home, and the toolbar, rail, panels and keys are
        shortcuts to it. A command that does not apply in the mode is shown disabled with the reason; the rail and panels
        show only what applies.
      </Rule>
      <Rule
        id="N10"
        why="Grouping follows N5; breadth is spent sparingly (A13), so the rail's groups nest as Model's submenus rather than taking titles of their own."
        examples={
          <Plate kind="try" label="the menu bar as shipped; switch the mode and open the menus">
            <MenuBarSpecimen />
          </Plate>
        }
      >
        Seven menus in the order you work: file, selection, picture, geometry, physics, desk, help. A section that is a set
        of one kind (Camera, Create, Constrain) is a submenu; a short section is a labelled block.
      </Rule>
      <Rule
        id="N11"
        why="Read down a column to see a kind of home; read along a row to see every way to one feature."
        examples={
          <Plate label="the feature index">
            <FeatureIndex />
          </Plate>
        }
      >
        One name per command in every home, a keyboard path for every frequent command, and no command reachable only by a
        gesture.
      </Rule>
    </>
  ),
};
