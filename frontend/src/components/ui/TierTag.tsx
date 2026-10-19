import { Show } from "solid-js";
import { tierKind, type TierKind } from "../../tier";

/**
 * What a feature is in this install, at the point of use (K4, tier chip).
 *
 * A value chip — the sheet with a hairline edge — because it states a fact
 * about the source, not a kind and not a status: "Exact · private tier" or
 * "Faceted · public tier". The public form's edge is dashed — still the one
 * 1px weight — so the two read apart without a hue, and its tooltip is the
 * server's sentence. Renders nothing until the server has answered.
 */
export function TierTag(props: { kind: TierKind; testId?: string }) {
  const status = () => tierKind(props.kind);
  return (
    <Show when={status()}>
      {(entry) => (
        <span
          class="tier-tag"
          classList={{ "tier-public": !entry().available }}
          title={entry().message ?? undefined}
          data-testid={props.testId ?? `tier-${props.kind}`}
          data-available={entry().available ? "true" : "false"}
        >
          {entry().label}
        </span>
      )}
    </Show>
  );
}

/**
 * The one-line reason under a feature whose exact path is missing; nothing
 * when the private tier fills it.
 */
export function TierReason(props: { kind: TierKind; testId?: string }) {
  const message = () => {
    const entry = tierKind(props.kind);
    return entry && !entry.available ? entry.message : null;
  };
  return (
    <Show when={message()}>
      <small class="tier-reason" data-testid={props.testId ?? `tier-reason-${props.kind}`}>
        {message()}
      </small>
    </Show>
  );
}
