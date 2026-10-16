/**
 * A tool rail with one group open, drawn at the rail's real geometry (a
 * sheet with a gap of padding, 32px cells a gap apart): the flyout is a
 * separate sheet a section out, its cells on the parent's row, a notch
 * pointing at the parent, the parent marked by its fill (K8a).
 */

import { BoxIcon, CursorIcon, CylinderIcon, MoveIcon, PolygonIcon, SphereIcon } from "../components/icons";

export function FlyoutSpecimen() {
  return (
    <div class="dl-fly">
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
      </div>
    </div>
  );
}
