/**
 * Where each sketch plane's label sits on screen.
 *
 * The plane's frame is drawn by the GPU; its name is text, so it is drawn by
 * the browser, projected here through the same view descriptor the renderer
 * used. One mark per plane that has a sketch, plus one for the placement
 * preview while the sketch tool is armed. Pure, like `constraintMarks.ts`.
 */

import type { PlaneFrame } from "../../planes";
import { labelAnchor } from "../../planes";
import { projectPoint } from "../../viewer/math";
import type { PickView } from "../../viewer/hittest";

export interface PlaneMark {
  /** The sketch the plane belongs to; null for the placement preview. */
  nodeId: string | null;
  label: string;
  /** CSS pixels inside the canvas. */
  x: number;
  y: number;
  /** A plane the source derived from other geometry: shown, not draggable. */
  derived: boolean;
  preview: boolean;
}

export interface PlaneMarkGeometry {
  width: number;
  height: number;
  marks: PlaneMark[];
}

export function buildPlaneMarks(
  view: PickView,
  width: number,
  height: number,
  frames: readonly PlaneFrame[],
  preview: PlaneFrame | null,
): PlaneMarkGeometry {
  if (width <= 0 || height <= 0) return { width, height, marks: [] };
  const scaleX = width / Math.max(view.width, 1);
  const scaleY = height / Math.max(view.height, 1);
  const marks: PlaneMark[] = [];
  const place = (frame: PlaneFrame, isPreview: boolean) => {
    const point = projectPoint(labelAnchor(frame), view);
    if (!point.visible) return;
    marks.push({
      nodeId: frame.nodeId,
      label: frame.label,
      x: point.x * scaleX,
      y: point.y * scaleY,
      derived: frame.derived,
      preview: isPreview,
    });
  };
  for (const frame of frames) place(frame, false);
  if (preview) place(preview, true);
  return { width, height, marks };
}
