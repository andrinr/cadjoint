/**
 * The private tier, as this server has it: read once, shown everywhere.
 *
 * cadjoint's exact paths — the derived B-rep, its edge curves, analytic STEP,
 * the node map that lets a Gmsh mesh follow the design — are the private
 * tier's (diff-brep). Without it each of those features still works, on the
 * dual-contouring path, and the viewer says so at the feature rather than
 * letting it degrade silently: a value chip with what the feature is in this
 * tier ("Faceted · public tier") and, where the exact path is missing, the
 * server's one-line reason. Both strings come from `cadjoint.tier`, so the
 * library's refusals and the viewer's tags are one wording.
 *
 * `/api/capabilities` is asked once, after the session starts. Until it
 * answers, `tierKind` returns null and nothing is tagged.
 */

import { createSignal } from "solid-js";
import * as api from "./api";
import type { CapabilitiesResponse, KindStatus } from "./types";

/** The private kinds the viewer tags (`cadjoint.tier.KINDS`). */
export type TierKind = "brep" | "feature_edges" | "step_export" | "node_map" | "drag";

const [capabilities, setCapabilities] = createSignal<CapabilitiesResponse | null>(null);

export { capabilities };

/** Ask the server once; a failure leaves everything untagged rather than wrong. */
export async function loadCapabilities(): Promise<void> {
  try {
    setCapabilities(await api.capabilities());
  } catch {
    setCapabilities(null);
  }
}

/** State the tier outright: for the static demo, which has no server to ask, and for tests. */
export function stateTier(value: CapabilitiesResponse | null): void {
  setCapabilities(value);
}

/** One kind's standing, or null before the server has said. */
export function tierKind(kind: TierKind): KindStatus | null {
  return capabilities()?.kinds[kind] ?? null;
}

/** Whether the exact path for `kind` is known to be missing. */
export function tierMissing(kind: TierKind): boolean {
  const status = tierKind(kind);
  return status !== null && !status.available;
}
