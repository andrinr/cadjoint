/**
 * Meshes tab — declare, generate, and judge discretizations.
 *
 * Each card is one `SimMesh(...)` in the program: its resolution, bounds,
 * padding and element family are edited straight into the source through
 * /patch, and "Generate mesh" builds it server-side and shows it with a
 * quality heatmap. The empty state matters as much as the list: a scene with
 * no declared mesh still solves on each study's implicit mesh, so the
 * fallback offers to generate and inspect that one.
 *
 * State and requests live in the panel controller; this file is the view.
 */

import { For, Show } from "solid-js";
import {
  addMeshRequest,
  deleteMeshRequest,
  meshArguments,
  setMeshValueRequest,
} from "../../meshes";
import { simMeshes, studies } from "../../state";
import {
  Card,
  CardHeader,
  CardList,
  NumberField,
  SectionHead,
  Segmented,
  VectorField,
} from "../ui";
import { MeshStats } from "./MeshStats";
import type { SimulateController } from "./controller";
import type { SimMeshPayload } from "../../types";

/** Element type: hex is the lattice-native default; tet10 resolves curved
    boundaries better at a solve cost; tet4 exists but measures stiff
    (locking) — prefer tet10. */
const METHODS = [
  { value: "hex", label: "Hex", title: "Hexahedra: fast, lattice-aligned" },
  { value: "tet4", label: "Tet4", title: "Linear tets: stiff — prefers Tet10" },
  { value: "tet10", label: "Tet10", title: "Quadratic tets: accurate boundary, slower" },
] as const;

/** Triplet arguments are edited component-wise; the rest are single values. */
type TripletKey = "resolution" | "bounds" | "size";

export interface MeshesTabProps {
  sim: SimulateController;
}

export function MeshesTab(props: MeshesTabProps) {
  const sim = () => props.sim;

  const methodOptions = (mesh: SimMeshPayload) =>
    METHODS.map((method) => ({
      value: method.value as string,
      label: method.label,
      title: method.title,
      testId: `mesh-method-${mesh.name}-${method.value}`,
    }));

  return (
    <>
      <SectionHead title="Meshes" testId="mesh-panel">
        <button
          type="button"
          class="sim-add-inline"
          onClick={() => void sim().patch(addMeshRequest())}
          title="Declare a SimMesh in the code"
          data-testid="mesh-add"
        >
          + Mesh
        </button>
      </SectionHead>
      <Show
        when={simMeshes().length > 0}
        fallback={
          <div class="sim-help" data-testid="mesh-empty">
            <p>
              No meshes declared — each study builds its own default mesh from
              its resolution and bounds when you solve. Declare a SimMesh(...)
              in the program to name it, share it between studies, choose
              hex/tet10, and inspect its quality here.
            </p>
            <For each={studies()}>
              {(study) => (
                <button
                  type="button"
                  class="sim-add-inline"
                  disabled={sim().inspecting() !== null || sim().solving() !== null}
                  onClick={() => void sim().inspect({ name: study.name })}
                  title={`Build and display the mesh ${study.name} would solve on`}
                  data-testid={`mesh-generate-study-${study.name}`}
                >
                  {sim().inspecting() === study.name
                    ? "Generating…"
                    : `Generate ${study.name}'s mesh`}
                </button>
              )}
            </For>
            <Show when={sim().inspected()}>
              {(current) => <MeshStats info={current().info} />}
            </Show>
          </div>
        }
      >
        <CardList testId="mesh-list">
          <For each={simMeshes()}>
            {(mesh) => (
              <Card testId={`mesh-${mesh.name}`}>
                <CardHeader
                  kind="mesh"
                  kindClass="sim-kind-mesh"
                  name={mesh.name}
                  onDelete={() => void sim().patch(deleteMeshRequest(mesh))}
                  deleteTitle="Delete this mesh from the code"
                  deleteAriaLabel={`Delete mesh ${mesh.name}`}
                  deleteTestId={`mesh-delete-${mesh.name}`}
                />

                <Show
                  when={mesh.editable}
                  fallback={
                    <p class="sim-note">Defined dynamically in code — edit it there.</p>
                  }
                >
                  <For each={meshArguments(mesh)}>
                    {(argument) =>
                      Array.isArray(argument.value) ? (
                        <VectorField
                          label={argument.key}
                          value={argument.value}
                          step={argument.key === "resolution" ? "1" : "0.1"}
                          disabled={sim().solving() !== null}
                          testId={(component) =>
                            `mesh-arg-${mesh.name}-${argument.key}-${component}`
                          }
                          onCommit={(component, next) => {
                            const triplet = [...(argument.value as number[])];
                            triplet[component] =
                              argument.key === "resolution" ? Math.round(next) : next;
                            void sim().patch(
                              setMeshValueRequest(
                                mesh,
                                argument.key as TripletKey,
                                triplet,
                              ),
                            );
                          }}
                        />
                      ) : (
                        <NumberField
                          class="sim-builder-vector"
                          label={argument.key}
                          step={argument.key === "resolution" ? "1" : "0.05"}
                          value={argument.value as number}
                          disabled={sim().solving() !== null}
                          testId={`mesh-arg-${mesh.name}-${argument.key}`}
                          onCommit={(value) =>
                            void sim().patch(
                              setMeshValueRequest(
                                mesh,
                                argument.key,
                                argument.key === "resolution" ? Math.round(value) : value,
                              ),
                            )
                          }
                        />
                      )
                    }
                  </For>
                  <Show when={mesh.domain}>
                    {(domain) => (
                      <label class="sim-builder-vector">
                        <span>domain</span>
                        <select
                          value={domain().name ?? ""}
                          disabled={sim().solving() !== null}
                          onChange={(event) => {
                            const value = event.currentTarget.value;
                            if (value) {
                              void sim().patch(setMeshValueRequest(mesh, "domain", value));
                            }
                          }}
                          data-testid={`mesh-domain-${mesh.name}`}
                        >
                          <Show when={domain().name === null}>
                            <option value="">{`(${domain().type})`}</option>
                          </Show>
                          <For each={sim().namedObjects()}>
                            {(name) => <option value={name}>{name}</option>}
                          </For>
                        </select>
                      </label>
                    )}
                  </Show>

                  <Segmented
                    class="sim-method"
                    testId={`mesh-method-${mesh.name}`}
                    options={methodOptions(mesh)}
                    value={mesh.method ?? "hex"}
                    disabled={sim().solving() !== null}
                    onSelect={(method) =>
                      void sim().patch(setMeshValueRequest(mesh, "method", method))
                    }
                  />
                  <p class="sim-note sim-method-hint">
                    Hex: fast, lattice-aligned · Tet10: accurate boundary, slower
                  </p>
                </Show>

                <button
                  type="button"
                  class="sim-run"
                  disabled={sim().inspecting() !== null || sim().solving() !== null}
                  onClick={() => void sim().inspect(mesh)}
                  title="Build this mesh and show it with its quality heatmap"
                  data-testid={`mesh-inspect-${mesh.name}`}
                >
                  {sim().inspecting() === mesh.name
                    ? "Generating…"
                    : sim().inspected()?.name === mesh.name
                      ? "Regenerate"
                      : "Generate mesh"}
                </button>

                <Show when={sim().inspected()?.name === mesh.name}>
                  <MeshStats
                    info={sim().inspected()!.info}
                    histogram={sim().histogram()}
                  />
                </Show>
              </Card>
            )}
          </For>
        </CardList>
      </Show>
    </>
  );
}
