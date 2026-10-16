/**
 * Studies tab — the declarations, their boundary conditions, and Solve.
 *
 * Each card is one `ThermalStudy(...)` or `ElasticStudy(...)` in the program.
 * Its numeric arguments, the mesh it discretizes and the object it is
 * restricted to are all plain source rewrites through /patch, so the card and
 * the code never disagree. Solving posts only the study's name: the server
 * re-derives the mesh and the BCs from the declaration.
 *
 * A study is solvable once it has at least one BC, which is why the BC list
 * and its builder live inside the card rather than in a tab of their own.
 */

import { For, Show, createSignal, onCleanup } from "solid-js";
import { setStudyDomainRequest, setStudyMeshRequest } from "../../meshes";
import { simMeshes, simView, studies } from "../../state";
import {
  BC_LABELS,
  addStudyRequest,
  bcTypesFor,
  deleteStudyRequest,
  setArgumentRequest,
  studyArguments,
} from "../../studies";
import { Card, CardHeader, CardList, NumberField, ToggleSwitch } from "../ui";
import { BcBuilder } from "./BcBuilder";
import { BcList } from "./BcList";
import { bcSwatch } from "./colors";
import type { SimulateController } from "./controller";

export interface StudiesTabProps {
  sim: SimulateController;
}

/** The kinds "+ Study" offers, named as Simulate › New study names them. */
const STUDY_KINDS: { kind: "thermal" | "elastic" | "flow"; label: string; hint: string }[] = [
  { kind: "thermal", label: "Thermal", hint: "A ThermalStudy: steady heat conduction" },
  { kind: "elastic", label: "Elastic", hint: "An ElasticStudy: linear elasticity" },
  { kind: "flow", label: "Flow", hint: "A flow study on its own lattice" },
];

export function StudiesTab(props: StudiesTabProps) {
  const sim = () => props.sim;
  const [kindsOpen, setKindsOpen] = createSignal(false);
  let anchor: HTMLDivElement | undefined;

  // A small menu closes like any other: a click elsewhere, or Escape —
  // which it claims in the capture phase so the viewport's Escape ladder
  // does not also fire underneath it.
  const onPointerDown = (event: PointerEvent) => {
    if (kindsOpen() && anchor && !anchor.contains(event.target as Node)) setKindsOpen(false);
  };
  const onKeyDown = (event: KeyboardEvent) => {
    if (event.key !== "Escape" || !kindsOpen()) return;
    event.stopPropagation();
    setKindsOpen(false);
  };
  window.addEventListener("pointerdown", onPointerDown, true);
  document.addEventListener("keydown", onKeyDown, true);
  onCleanup(() => {
    window.removeEventListener("pointerdown", onPointerDown, true);
    document.removeEventListener("keydown", onKeyDown, true);
  });

  return (
    <>
      <Show
        when={studies().length > 0}
        fallback={
          <p class="sim-help" data-testid="simulate-empty">
            No studies declared. Add one — it becomes a ThermalStudy or
            ElasticStudy call in the code, and stays editable from either side.
          </p>
        }
      >
        <CardList testId="simulate-studies">
          <For each={studies()}>
            {(study) => (
              <Card testId={`simulate-study-${study.name}`}>
                <CardHeader
                  kind={study.kind}
                  kindClass={`sim-kind-${study.kind}`}
                  name={study.name}
                  onDelete={() => void sim().patch(deleteStudyRequest(study))}
                  deleteTitle="Delete this study from the code"
                  deleteAriaLabel={`Delete study ${study.name}`}
                  deleteTestId={`simulate-delete-${study.name}`}
                />

                <Show
                  when={study.editable}
                  fallback={
                    <p class="sim-note">Defined dynamically in code — edit it there.</p>
                  }
                >
                  <div class="sim-args">
                    <For each={studyArguments(study)}>
                      {(argument) => (
                        <NumberField
                          label={argument.key}
                          value={argument.value}
                          disabled={sim().solving() !== null}
                          testId={`simulate-arg-${study.name}-${argument.key}`}
                          onCommit={(value) =>
                            void sim().patch(
                              setArgumentRequest(study, argument.key, value),
                            )
                          }
                        />
                      )}
                    </For>
                    {/* The mesh/domain a study discretizes: a declared
                        SimMesh by name, or a named object restricting the
                        implicit mesh. Both are plain source rewrites.
                        A flow study discretizes neither — it fills a lattice
                        of its own — so it is offered no mesh to pick. */}
                    <Show when={simMeshes().length > 0 && study.kind !== "flow"}>
                      <label>
                        <span>mesh</span>
                        <select
                          value={study.mesh ?? ""}
                          disabled={sim().solving() !== null}
                          onChange={(event) => {
                            const value = event.currentTarget.value;
                            if (value) void sim().patch(setStudyMeshRequest(study, value));
                          }}
                          data-testid={`simulate-mesh-${study.name}`}
                        >
                          <Show when={study.mesh === null}>
                            <option value="">(implicit)</option>
                          </Show>
                          <For each={simMeshes()}>
                            {(mesh) => <option value={mesh.name}>{mesh.name}</option>}
                          </For>
                        </select>
                      </label>
                    </Show>
                    <Show when={study.mesh === null && sim().namedObjects().length > 0}>
                      <label>
                        <span>domain</span>
                        <select
                          value={study.domain?.name ?? ""}
                          disabled={sim().solving() !== null}
                          onChange={(event) => {
                            const value = event.currentTarget.value;
                            if (value) {
                              void sim().patch(setStudyDomainRequest(study, value));
                            }
                          }}
                          data-testid={`simulate-domain-${study.name}`}
                        >
                          <Show when={study.domain === null}>
                            <option value="">(whole scene)</option>
                          </Show>
                          <Show when={study.domain && study.domain.name === null}>
                            <option value="">{`(${study.domain!.type})`}</option>
                          </Show>
                          <For each={sim().domainOptions(study.domain?.name)}>
                            {(name) => <option value={name}>{name}</option>}
                          </For>
                        </select>
                      </label>
                    </Show>
                  </div>

                  <Show when={simView()}>
                    <ToggleSwitch
                      compact
                      class="sim-show-bcs"
                      checked={sim().showBcs() === study.index}
                      onChange={(checked) =>
                        sim().setShowBcs(checked ? study.index : null)
                      }
                      testId={`simulate-show-bcs-${study.name}`}
                    >
                      Show BCs on mesh
                    </ToggleSwitch>
                    <Show when={sim().showBcs() === study.index}>
                      <div class="sim-bc-legend" data-testid="simulate-bc-legend">
                        <For each={bcTypesFor(study.kind)}>
                          {(type) => (
                            <span>
                              <i style={{ background: bcSwatch(type) }} />
                              {BC_LABELS[type]}
                            </span>
                          )}
                        </For>
                      </div>
                    </Show>
                  </Show>

                  <BcList sim={sim()} study={study} />

                  <Show
                    when={sim().building() === study.index}
                    fallback={
                      <button
                        type="button"
                        class="sim-add-bc"
                        onClick={() => sim().openBuilder(study)}
                        data-testid={`simulate-add-bc-${study.name}`}
                      >
                        + Boundary condition
                      </button>
                    }
                  >
                    <BcBuilder sim={sim()} study={study} />
                  </Show>
                </Show>

                {/* A solve is minutes of somebody's machine, so the button
                    that started it is the button that stops it: once the
                    registry has told us which job is running, Solve becomes
                    Cancel in place. Before that it reads as working, because
                    there is nothing to cancel yet. */}
                <Show
                  when={sim().solving() === study.name}
                  fallback={
                    <button
                      type="button"
                      class="sim-run"
                      disabled={
                        sim().solving() !== null ||
                        sim().unavailable() ||
                        study.bcs.length === 0
                      }
                      onClick={() => void sim().solve(study)}
                      title={
                        study.bcs.length > 0
                          ? "Mesh the scene and run this study"
                          : "Add at least one boundary condition first"
                      }
                      data-testid={`simulate-run-${study.name}`}
                    >
                      Solve
                    </button>
                  }
                >
                  <button
                    type="button"
                    class="sim-run"
                    disabled={sim().solveJob() === null}
                    onClick={() => void sim().cancelActive()}
                    title="Stop this solve and free the worker"
                    data-testid={`simulate-cancel-${study.name}`}
                  >
                    {sim().solveJob() ? "Cancel" : "Meshing + solving…"}
                  </button>
                </Show>
              </Card>
            )}
          </For>
        </CardList>
      </Show>

      {/* One add button, the kinds behind it — the same three as
          Simulate › New study. Three buttons side by side truncated in a
          320px column. */}
      <div class="sim-add-study-anchor" ref={anchor}>
        <button
          type="button"
          class="sim-add-bc sim-add-study-toggle"
          aria-haspopup="menu"
          aria-expanded={kindsOpen()}
          onClick={() => setKindsOpen(!kindsOpen())}
          title="Declare a new study in the code"
          data-testid="simulate-add-study"
        >
          + Study
        </button>
        <Show when={kindsOpen()}>
          <div class="menu-dropdown study-kinds" role="menu" aria-label="Study kind">
            <For each={STUDY_KINDS}>
              {(entry) => (
                <button
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    setKindsOpen(false);
                    void sim().patch(addStudyRequest(entry.kind));
                  }}
                  title={entry.hint}
                  data-testid={`simulate-add-${entry.kind}`}
                >
                  <i class="menu-check" />
                  <span>{entry.label}</span>
                </button>
              )}
            </For>
          </div>
        </Show>
      </div>
    </>
  );
}
