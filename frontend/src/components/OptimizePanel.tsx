/**
 * Optimize panel — the Model-mode dock section hosting the optimization
 * cards. All the behavior lives in OptimizeCards, which the Simulate
 * panel's Optimize tab shares, so runs behave identically from both homes.
 */

import { OptimizeCards, type OptimizeCardsProps } from "./OptimizeCards";

export type OptimizePanelProps = OptimizeCardsProps;

export function OptimizePanel(props: OptimizePanelProps) {
  return (
    <aside class="sim-panel optimize-panel" data-testid="optimize-panel">
      <header>
        <span>
          <small>Adjoint</small>
          Optimize
        </span>
      </header>
      <OptimizeCards
        onPatch={props.onPatch}
        onAdoptSource={props.onAdoptSource}
        onGhostCompile={props.onGhostCompile}
      />
    </aside>
  );
}
