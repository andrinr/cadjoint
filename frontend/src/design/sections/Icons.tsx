import { For, type JSX } from "solid-js";
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
} from "../../components/icons";
import { Plate, Rule, v, type SectionDef } from "../kit";

const ICONS: [string, (props: { size?: number }) => JSX.Element][] = [
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

export const icons: SectionDef = {
  id: "icons",
  title: "Iconography",
  body: () => (
    <>
      <Rule
        id="I1"
        why="The stroke matches the hairline, so an icon and a rule look drawn by one pen. Icons get no colour (A3): an icon that needs one needs a word."
        examples={
          <Plate label="the set as shipped, ink-2">
            <div class="dl-icons">
              <For each={ICONS}>
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
        16px box, 1.25px stroke, no fills, miter joins and square caps, drawn on the half pixel. Icons are{" "}
        <code>ink-2</code>, and <code>ink</code> when active.
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
        An icon alone carries its word in its tooltip. A destructive action carries the word unless undo reaches it
        (A11): the rail's Delete is an icon, the Edit menu's a word. Gizmos and handles are geometry, not icons.
      </Rule>
    </>
  ),
};
