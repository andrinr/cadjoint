/**
 * Four ways a tool rail's group can open, drawn at the rail's real
 * geometry (a sheet with a gap of padding, 32px cells a gap apart), so the
 * choice is made on the actual shape rather than on a description.
 */

import { BoxIcon, CursorIcon, CylinderIcon, MoveIcon, PolygonIcon, SphereIcon } from "../components/icons";

export type FlyoutVariant = "flush" | "opened" | "column" | "pointer";

export const FLYOUT_VARIANTS: { id: FlyoutVariant; name: string; idea: string }[] = [
  { id: "flush", name: "A · flush at the edge", idea: "The rail's edge stays whole; the flyout butts against it at the parent's row. The parent is marked by its fill." },
  { id: "opened", name: "B · the edge opens", idea: "The rail's edge is cut exactly at the parent's row and the flyout continues the rail outward, like a drawer." },
  { id: "column", name: "C · a second column", idea: "A second rail beside the first, the same cells, starting at the parent's row and running down." },
  { id: "pointer", name: "D · separate, pointing", idea: "A sheet a gap away with a notch at the parent. The conventional answer, for comparison." },
];

export function FlyoutSpecimen(props: { variant: FlyoutVariant }) {
  return (
    <div class={`dl-fly dl-fly-${props.variant}`}>
      <div class="dl-fly-rail">
        <span class="dl-fly-cell">
          <CursorIcon size={16} />
        </span>
        <span class="dl-fly-cell parent">
          <BoxIcon size={16} />
        </span>
        <span class="dl-fly-cell">
          <PolygonIcon size={16} />
        </span>
        <span class="dl-fly-cell">
          <MoveIcon size={16} />
        </span>
        <span class="dl-fly-out">
          <span class="dl-fly-cell">
            <BoxIcon size={16} />
          </span>
          <span class="dl-fly-cell">
            <SphereIcon size={16} />
          </span>
          <span class="dl-fly-cell">
            <CylinderIcon size={16} />
          </span>
        </span>
        <span class="dl-fly-cut" />
      </div>
    </div>
  );
}
