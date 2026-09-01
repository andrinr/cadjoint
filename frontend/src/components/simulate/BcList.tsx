/**
 * One study's boundary conditions, as rows you can edit or remove.
 *
 * Every row is a `Nodes.*` selection plus a value in the source, so the row
 * shows the selection as the code spells it and edits the value in place.
 * Selections written as a Python predicate cannot be evaluated client-side;
 * those rows say so and drop their preview instead of guessing. Hovering a
 * row paints its nodes on the displayed mesh, which is why the list owns the
 * hover signal rather than the study card.
 */

import { For, Index, Show } from "solid-js";
import { BC_LABELS, bcValue, deleteBcRequest, describeSelection, setBcValueRequest } from "../../studies";
import { selectionEvaluable } from "../../selectionEval";
import { AXIS_LABELS, NumberInput } from "../ui";
import { bcSwatch } from "./colors";
import type { SimulateController } from "./controller";
import type { StudyPayload } from "../../types";

export interface BcListProps {
  sim: SimulateController;
  study: StudyPayload;
}

export function BcList(props: BcListProps) {
  const sim = () => props.sim;

  return (
    <ul class="sim-bcs" data-testid={`simulate-bcs-${props.study.name}`}>
      <For each={props.study.bcs}>
        {(bc, bcIndex) => (
          <li
            classList={{ "sim-bc-readonly": !bc.serializable }}
            onMouseEnter={() =>
              sim().setHoveredBc({ study: props.study.index, bc: bcIndex() })
            }
            onMouseLeave={() => sim().setHoveredBc(null)}
          >
            <div class="sim-bc-main">
              <span class="sim-bc-type">
                <i class="sim-bc-swatch" style={{ background: bcSwatch(bc.type) }} />
                {BC_LABELS[bc.type]}
              </span>
              <code title={describeSelection(bc.nodes)}>{describeSelection(bc.nodes)}</code>
              <Show when={!selectionEvaluable(bc.nodes)}>
                <small class="sim-note">no preview (predicate)</small>
              </Show>
            </div>
            <Show
              when={bc.serializable}
              fallback={<small class="sim-note">edit in code</small>}
            >
              <Show when={bcValue(bc) !== null}>
                <Show
                  when={bc.type === "traction"}
                  fallback={
                    <NumberInput
                      value={bcValue(bc) as number}
                      disabled={sim().solving() !== null}
                      title={BC_LABELS[bc.type]}
                      testId={`simulate-bc-value-${props.study.name}-${bcIndex()}`}
                      onCommit={(value) =>
                        void sim().patch(
                          setBcValueRequest(props.study, bcIndex(), value),
                        )
                      }
                    />
                  }
                >
                  <span class="sim-vector">
                    <Index each={bcValue(bc) as number[]}>
                      {(component, index) => (
                        <NumberInput
                          value={component()}
                          disabled={sim().solving() !== null}
                          title={`Traction ${AXIS_LABELS[index]}`}
                          onCommit={(value) => {
                            const vector = [...(bcValue(bc) as number[])];
                            vector[index] = value;
                            void sim().patch(
                              setBcValueRequest(props.study, bcIndex(), vector),
                            );
                          }}
                        />
                      )}
                    </Index>
                  </span>
                </Show>
              </Show>
              <button
                type="button"
                class="sim-delete"
                onClick={() =>
                  void sim().patch(deleteBcRequest(props.study, bcIndex()))
                }
                title="Remove this boundary condition"
                aria-label="Remove boundary condition"
                data-testid={`simulate-bc-delete-${props.study.name}-${bcIndex()}`}
              >
                ×
              </button>
            </Show>
          </li>
        )}
      </For>
    </ul>
  );
}
