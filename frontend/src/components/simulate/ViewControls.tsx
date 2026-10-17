/**
 * What the viewport shows once a mesh or a result is loaded.
 *
 * These controls belong to neither tab: they apply to whatever is currently
 * displayed, so they sit under the tab strip and stay put while the user
 * moves between Meshes, Studies and Results.
 *
 * The viewport switch is the important one. A loaded mesh never owns the
 * viewport for good — "Scene" returns to the raymarched SDF and keeps the
 * loaded state one click away. ("Both" is deliberately absent: the mesh sits
 * inside the opaque raymarched solid, so a composite would only hide it.)
 */

import { Segmented, ToggleSwitch, axisOptions } from "../ui";

const SLICE_AXES = axisOptions("simulate-slice", (axis) => `Slice along ${axis}`);
import type { SimulateController, ViewportMode } from "./controller";

const VIEWPORTS = [
  {
    value: "scene" as ViewportMode,
    label: "Scene",
    title: "Show the raymarched scene; the loaded mesh stays ready",
    testId: "simulate-viewport-scene",
  },
  {
    value: "mesh" as ViewportMode,
    label: "Mesh",
    title: "Show the loaded mesh or solved field",
    testId: "simulate-viewport-mesh",
  },
];

export interface ViewControlsProps {
  sim: SimulateController;
}

export function ViewControls(props: ViewControlsProps) {
  const sim = () => props.sim;

  return (
    <>
      <div class="sim-row sim-view-control">
        <Segmented
          class="sim-viewport"
          testId="simulate-viewport"
          options={VIEWPORTS}
          value={sim().viewportMode()}
          onSelect={(mode) => sim().setViewport(mode)}
        />
        <ToggleSwitch
          compact
          title={
            sim().hasEdges()
              ? "Hairline element boundary edges over the surface"
              : "This payload carries no element edges"
          }
          checked={sim().showEdges() && sim().hasEdges()}
          disabled={!sim().hasEdges()}
          onChange={(checked) => sim().setShowEdges(checked)}
          testId="simulate-edges"
        >
          Element edges
        </ToggleSwitch>
      </div>
      <div class="sim-row sim-slice">
        <ToggleSwitch
          compact
          checked={sim().slice().enabled}
          onChange={(enabled) => sim().applySlice({ enabled })}
          testId="simulate-slice-enabled"
        >
          Slice
        </ToggleSwitch>
        <Segmented
          options={SLICE_AXES}
          value={sim().slice().axis}
          onSelect={(axis) => sim().applySlice({ axis })}
        />
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={sim().slice().fraction}
          disabled={!sim().slice().enabled}
          onInput={(event) =>
            sim().applySlice({ fraction: Number(event.currentTarget.value) })
          }
          data-testid="simulate-slice-fraction"
        />
      </div>
    </>
  );
}
