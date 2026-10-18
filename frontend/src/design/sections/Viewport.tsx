import { ViewportSpecimen } from "../ViewportSpecimen";
import { Plate, Rule, type SectionDef } from "../kit";

export const viewport: SectionDef = {
  id: "viewport",
  title: "Viewport",
  intro: "An instrument face (A1): overlays are achromatic unless they are data, direction or the skeleton (C7, C10).",
  body: () => (
    <>
      <Rule
        id="V1"
        why="Free versus held is told by shape, and by param, the one hue that means free everywhere: in the editor and on a vertex the optimiser may move."
        examples={
          <Plate label="sketch mode" ground>
            <ViewportSpecimen />
          </Plate>
        }
      >
        Edges take C10's construction ladder, the selected one <code>ink</code> at 2px; behind an x-rayed part they are
        dashed and faint. A free vertex is a hollow <code>param</code> ring, a held one a filled{" "}
        <code>construction</code> square, the selected one a filled <code>ink</code> circle, each in a sheet casing. A
        dimension label is <code>ink</code> on an opaque tab; a violated one is <code>danger</code> and says by how much.
        The gizmo and the triad are the only other hues. The skeleton has its own switch in the viewport bar, beside
        projection (<code>⇧C</code>).
      </Rule>
      <Rule
        id="V2"
        why="The picture is a measurement: nothing drawn over it may change a value."
        examples={
          <Plate label="simulate mode" ground>
            <ViewportSpecimen field />
          </Plate>
        }
      >
        Over a solved field the field is the only colour: hover and selection are outlines, the legend opaque and
        labelled at both ends (C7), and the skeleton steps back to the ink ladder (<code>ink-3</code> at rest).
      </Rule>
      <Rule id="V3" why="A spacing on an unnamed plane is not a measurement; a value on a perspective view is not to scale.">
        The graticule is furniture (1.4–2.8:1) on the world plane the camera most nearly faces, its spacing stated on a
        1-2-5 ladder (<code>GRID 100 mm · XZ</code>), prefixed <code>&gt;</code> when the view is not to scale; its steps
        are a detented zoom (<code>[</code> <code>]</code>). The performance readout states what only the viewport can (FPS
        or <code>idle</code>, frame time, resolution, nodes, mesh size), sampled twice a second, never tweened. The hint
        speaks only for a step in progress or a hovered handle; at rest it is empty (A13).
      </Rule>
    </>
  ),
};
