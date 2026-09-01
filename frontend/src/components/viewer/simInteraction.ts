/**
 * Pointer interaction with the displayed FEM surface.
 *
 * In simulate mode the viewport stops being a modelling surface and becomes
 * an inspection one: a click either probes the field at the nearest vertex or,
 * while the study builder has picking armed, proposes a boundary-condition
 * region. Both work on projected vertices, so occlusion is approximate — good
 * enough for a probe and for a proposal the user still confirms in the panel.
 *
 * Nothing here commits anything. Probes land in a shared signal the panel
 * mirrors, and BC proposals only pre-fill the builder.
 */

import { nearestVertex, sphereProposal } from "../../bcPick";
import { bcPickArmed, editingMode, setBcProposal, setSimProbe, simView } from "../../state";
import type { PickView } from "../../viewer/hittest";
import type { Renderer } from "../../viewer/renderer";

export interface SimInteractionContext {
  /** The canvas, read lazily: the ref is only bound after the first render. */
  canvas: () => HTMLCanvasElement;
  renderer: Renderer;
  pickView: () => PickView;
}

export function createSimInteraction(context: SimInteractionContext) {
  /** Whether pointer input should target the displayed FEM surface. */
  const simInteractive = () =>
    editingMode() === "simulate" &&
    simView() !== null &&
    context.renderer.simulationActive;

  /** Framebuffer px → CSS px, for DOM chips anchored to projected points. */
  const toCss = (x: number, y: number) => {
    const canvas = context.canvas();
    return {
      x: (x * canvas.clientWidth) / Math.max(canvas.width, 1),
      y: (y * canvas.clientHeight) / Math.max(canvas.height, 1),
    };
  };

  /**
   * A completed click on the FEM surface.
   *
   * Armed BC picking proposes a Nodes.sphere around the picked point (radius
   * from the mesh cell spacing); otherwise the click probes: a chip shows the
   * nearest vertex's position and the active field value.
   */
  const handleSimTap = (x: number, y: number) => {
    const view = simView();
    if (!view) return;
    const hit = nearestVertex(view.payload.positions, x, y, context.pickView());
    if (!hit) {
      setSimProbe(null);
      return;
    }
    if (bcPickArmed()) {
      setBcProposal(sphereProposal(hit.world, view.info?.grid ?? null));
      return;
    }
    const anchor = toCss(hit.x, hit.y);
    setSimProbe({
      x: anchor.x,
      y: anchor.y,
      world: hit.world,
      value: view.scalars[hit.index] ?? 0,
      label: view.fieldLabel,
    });
  };

  /** Rubber-band rectangle for the current bcrect gesture, in CSS px. */
  const rectFromGesture = (rect: { x0: number; y0: number; x1: number; y1: number }) => {
    const a = toCss(rect.x0, rect.y0);
    const b = toCss(rect.x1, rect.y1);
    return {
      left: Math.min(a.x, b.x),
      top: Math.min(a.y, b.y),
      width: Math.abs(b.x - a.x),
      height: Math.abs(b.y - a.y),
    };
  };

  return { simInteractive, toCss, handleSimTap, rectFromGesture };
}
