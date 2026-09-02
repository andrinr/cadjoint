/**
 * The one thing the dock announces to the panels inside it.
 *
 * Laying out a new desk tears the grid down and builds it again, which moves
 * every panel's element between the library's own wrappers. Almost nothing
 * cares — Solid keeps its subscriptions and the DOM comes back intact — but
 * the viewport holds a WebGPU swap chain, and a re-attached canvas is worth
 * reconfiguring. A window event rather than a prop, because the dock does not
 * know what is inside its panels and should not have to.
 */
export const DOCK_REBUILT_EVENT = "cadjoint:dock-rebuilt";
