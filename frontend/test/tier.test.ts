import { afterEach, describe, expect, it } from "vitest";
import { stateTier, tierKind, tierMissing } from "../src/tier";
import type { CapabilitiesResponse, KindStatus } from "../src/types";

const kind = (name: string, available: boolean): KindStatus => ({
  kind: name,
  available,
  plugin: available ? name : null,
  version: available ? "0.2.0" : null,
  contract_version: available ? 1 : null,
  compatible: available,
  reason: available ? null : "not registered",
  label: available ? "Exact · private tier" : "Dual contouring · public tier",
  message: available ? null : "Exact B-rep extraction is part of the private tier (diff-brep).",
});

const status = (available: boolean): CapabilitiesResponse => ({
  ok: true,
  installed: available,
  provider: "diff-brep",
  contract_version: 1,
  kinds: Object.fromEntries(
    ["brep", "feature_edges", "step_export", "node_map", "drag"].map((name) => [name, kind(name, available)]),
  ),
});

afterEach(() => stateTier(null));

describe("the tier store", () => {
  it("tags nothing and refuses nothing before the server has answered", () => {
    expect(tierKind("brep")).toBeNull();
    expect(tierMissing("brep")).toBe(false);
  });

  it("reports a missing kind with the server's label and sentence", () => {
    stateTier(status(false));
    expect(tierMissing("step_export")).toBe(true);
    expect(tierKind("brep")?.label).toBe("Dual contouring · public tier");
    expect(tierKind("brep")?.message).toMatch(/private tier/);
  });

  it("reports a filled kind as exact, with no sentence", () => {
    stateTier(status(true));
    expect(tierMissing("brep")).toBe(false);
    expect(tierKind("brep")?.message).toBeNull();
  });
});
