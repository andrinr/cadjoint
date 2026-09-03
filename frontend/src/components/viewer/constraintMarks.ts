/**
 * Where constraint annotations land on screen.
 *
 * Sketch constraints and object relations are drawn as DOM/SVG marks over the
 * canvas rather than by the renderer, so something has to project them into
 * CSS pixels every time the camera or the viewport moves. That projection is
 * pure — construction tree and view in, screen-space marks out — so it lives
 * here, away from the pane's pointer plumbing, and can be reasoned about (and
 * tested) without a GPU or a browser.
 *
 * Only the two kinds that carry geometry are placed: `fixed` pins and
 * `distance` dimensions. The relational kinds (horizontal, parallel, …) are
 * shown as panel chips instead and never reach this module.
 */

import type { ConstructionNode, ConstructionRelation } from "../../types";
import { projectPoint, type Vec3 } from "../../viewer/math";
import type { PickView } from "../../viewer/hittest";

/** A pin glyph anchored beside a fixed vertex or object. */
export interface FixedMark {
  x: number;
  y: number;
  key: string;
  scope: "sketch" | "object";
}

/** A dimension line with its witness lines and label, all in CSS pixels. */
export interface DistanceMark {
  ax: number;
  ay: number;
  bx: number;
  by: number;
  dax: number;
  day: number;
  dbx: number;
  dby: number;
  labelX: number;
  labelY: number;
  label: string;
  key: string;
  scope: "sketch" | "object";
}

export interface ConstraintOverlayGeometry {
  width: number;
  height: number;
  fixed: FixedMark[];
  distance: DistanceMark[];
}

/**
 * Project every drawable constraint into the overlay's viewBox.
 *
 * `width`/`height` are the canvas's CSS size; the view descriptor is the
 * renderer's own, so the marks line up with what was drawn. A zero-sized
 * canvas yields an empty overlay rather than a division by zero.
 */
export function buildConstraintOverlay(
  view: PickView,
  width: number,
  height: number,
  displayedNodes: readonly ConstructionNode[],
  relations: readonly ConstructionRelation[],
): ConstraintOverlayGeometry {
  if (width <= 0 || height <= 0) return { width, height, fixed: [], distance: [] };
  const scaleX = width / Math.max(view.width, 1);
  const scaleY = height / Math.max(view.height, 1);
  const screen = (world: Vec3) => {
    const point = projectPoint(world, view);
    return {
      x: point.x * scaleX,
      y: point.y * scaleY,
      visible: point.visible,
    };
  };
  const fixed: FixedMark[] = [];
  const distance: DistanceMark[] = [];

  const addDistance = (
    first: Vec3,
    second: Vec3,
    value: number,
    key: string,
    scope: "sketch" | "object",
  ) => {
    const a = screen(first);
    const b = screen(second);
    if (!a.visible || !b.visible) return;
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const length = Math.hypot(dx, dy);
    if (length < 2) return;
    const offsetX = (-dy / length) * 17;
    const offsetY = (dx / length) * 17;
    const dax = a.x + offsetX;
    const day = a.y + offsetY;
    const dbx = b.x + offsetX;
    const dby = b.y + offsetY;
    distance.push({
      ax: a.x,
      ay: a.y,
      bx: b.x,
      by: b.y,
      dax,
      day,
      dbx,
      dby,
      labelX: (dax + dbx) * 0.5,
      labelY: (day + dby) * 0.5 - 5,
      label: Number(value.toPrecision(4)).toString(),
      key,
      scope,
    });
  };

  for (const profile of displayedNodes) {
    if (profile.kind !== "profile") continue;
    for (let index = 0; index < profile.constraints.length; index++) {
      const constraint = profile.constraints[index];
      if (constraint.kind === "fixed") {
        const vertex = profile.vertices[constraint.vertices[0]];
        if (!vertex) continue;
        const point = screen(vertex.world);
        if (point.visible) {
          fixed.push({
            x: point.x + 9,
            y: point.y - 9,
            key: `${profile.id}-fixed-${index}`,
            scope: "sketch",
          });
        }
        continue;
      }

      // Only distance constraints draw a dimension; relational kinds
      // (horizontal, parallel, …) are shown as panel chips instead.
      if (constraint.kind !== "distance") continue;
      const first = profile.vertices[constraint.vertices[0]];
      const second = profile.vertices[constraint.vertices[1]];
      if (!first || !second || typeof constraint.value !== "number") continue;
      addDistance(
        first.world,
        second.world,
        constraint.value,
        `${profile.id}-distance-${index}`,
        "sketch",
      );
    }
  }

  const displayedById = new Map(displayedNodes.map((node) => [node.id, node]));
  for (let index = 0; index < relations.length; index++) {
    const relation = relations[index];
    if (relation.kind === "fixed") {
      const node = displayedById.get(relation.nodes[0]);
      if (!node?.transform) continue;
      const point = screen(node.transform.position);
      if (point.visible) {
        fixed.push({
          x: point.x + 9,
          y: point.y - 9,
          key: `object-fixed-${index}`,
          scope: "object",
        });
      }
      continue;
    }
    const first = displayedById.get(relation.nodes[0]);
    const second = displayedById.get(relation.nodes[1]);
    if (!first?.transform || !second?.transform || typeof relation.value !== "number") {
      continue;
    }
    addDistance(
      first.transform.position,
      second.transform.position,
      relation.value,
      `object-distance-${index}`,
      "object",
    );
  }
  return { width, height, fixed, distance };
}
