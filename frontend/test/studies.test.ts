import { describe, expect, it } from "vitest";
import {
  addBcRequest,
  addStudyRequest,
  bcTypesFor,
  bcValue,
  defaultDraft,
  deleteBcRequest,
  deleteStudyRequest,
  describeSelection,
  draftSelection,
  isEditableStudyKind,
  setArgumentRequest,
  setBcValueRequest,
  studyArguments,
} from "../src/studies";
import type { StudyPayload } from "../src/types";

const study: StudyPayload = {
  index: 1,
  stableId: null,
  name: "heat",
  kind: "thermal",
  resolution: 20,
  bounds: [0, 0, 0],
  size: [2, 2, 2],
  mesh: null,
  domain: null,
  material: { conductivity: 1.5 },
  source: 0.25,
  line: 10,
  span: [100, 200],
  editable: true,
  bcs: [],
};

describe("selection summaries", () => {
  it("formats every primitive selection kind", () => {
    expect(
      describeSelection({ kind: "box", min_corner: [0, 0, 0], max_corner: [1, 2.5, 1] }),
    ).toBe("box [0, 0, 0] → [1, 2.5, 1]");
    expect(describeSelection({ kind: "sphere", center: [1, 0, 0], radius: 0.5 })).toBe(
      "sphere [1, 0, 0] r 0.5",
    );
    expect(
      describeSelection({ kind: "halfspace", point: [0, 0, 1], normal: [0, 0, 1] }),
    ).toBe("halfspace at [0, 0, 1] · n [0, 0, 1]");
    expect(describeSelection({ kind: "side", side: "+x", tol: null })).toBe("side +x");
    expect(describeSelection({ kind: "side", side: "-z", tol: 0.05 })).toBe(
      "side -z ± 0.05",
    );
    expect(describeSelection({ kind: "predicate", name: "bolt_region" })).toBe(
      "predicate bolt_region()",
    );
  });

  it("renders composites with logic glyphs, not with nesting noise", () => {
    const composite = describeSelection({
      kind: "and",
      operands: [
        { kind: "side", side: "+x", tol: null },
        {
          kind: "not",
          operand: { kind: "sphere", center: [0, 0, 0], radius: 1 },
        },
      ],
    });
    expect(composite).toBe("side +x ∧ ¬(sphere [0, 0, 0] r 1)");
    expect(
      describeSelection({
        kind: "or",
        operands: [
          { kind: "side", side: "+y", tol: null },
          { kind: "side", side: "-y", tol: null },
        ],
      }),
    ).toBe("side +y ∨ side -y");
  });
});

describe("bc metadata", () => {
  it("offers the BC types matching the study kind", () => {
    expect(bcTypesFor("thermal")).toEqual(["dirichlet", "heat_flux"]);
    expect(bcTypesFor("elastic")).toEqual(["fixed", "traction"]);
  });

  it("extracts the editable value per BC type", () => {
    const nodes = { kind: "side", side: "+x", tol: null } as const;
    const row = { nodes, serializable: true, span: null, stableId: null } as const;
    expect(bcValue({ ...row, type: "dirichlet", value: 300 })).toBe(300);
    expect(bcValue({ ...row, type: "heat_flux", flux: 5 })).toBe(5);
    expect(bcValue({ ...row, type: "traction", vector: [0, 0, -1] })).toEqual([0, 0, -1]);
    expect(bcValue({ ...row, type: "fixed" })).toBeNull();
  });

  it("lists a study's editable numeric arguments", () => {
    expect(studyArguments(study)).toEqual([
      { key: "resolution", value: 20 },
      { key: "conductivity", value: 1.5 },
      { key: "source", value: 0.25 },
    ]);
    const elastic: StudyPayload = {
      ...study,
      kind: "elastic",
      material: { youngs: 200, poisson: 0.3 },
      source: undefined,
      resolution: [8, 8, 16],
    };
    // A per-axis resolution is not a single scalar field; only material shows.
    expect(studyArguments(elastic)).toEqual([
      { key: "youngs", value: 200 },
      { key: "poisson", value: 0.3 },
    ]);
  });
});

describe("patch request builders", () => {
  it("builds add/delete study requests addressed by index", () => {
    expect(addStudyRequest("elastic")).toEqual({ op: "add_study", kind: "elastic" });
    expect(deleteStudyRequest(study)).toEqual({ op: "delete_study", study: 1 });
    expect(deleteBcRequest(study, 2)).toEqual({ op: "delete_study_bc", study: 1, bc: 2 });
  });

  it("builds value edits for BCs and constructor arguments", () => {
    expect(setBcValueRequest(study, 0, 450)).toEqual({
      op: "set_study_value",
      study: 1,
      bc: 0,
      value: 450,
    });
    expect(setArgumentRequest(study, "resolution", 32)).toEqual({
      op: "set_study_value",
      study: 1,
      argument: "resolution",
      value: 32,
    });
  });

  it("converts the builder draft into an add_study_bc body", () => {
    const draft = defaultDraft("thermal");
    expect(draftSelection(draft)).toEqual({ kind: "side", side: "+x", tol: null });
    expect(addBcRequest(study, draft)).toEqual({
      op: "add_study_bc",
      study: 1,
      bc_type: "dirichlet",
      selection: { kind: "side", side: "+x", tol: null },
      value: 100,
    });

    const box = {
      ...defaultDraft("thermal"),
      selectionKind: "box" as const,
      minCorner: [0, 0, 0] as [number, number, number],
      maxCorner: [1, 1, 1] as [number, number, number],
      bcType: "heat_flux" as const,
      value: 7,
    };
    expect(addBcRequest(study, box)).toEqual({
      op: "add_study_bc",
      study: 1,
      bc_type: "heat_flux",
      selection: { kind: "box", min_corner: [0, 0, 0], max_corner: [1, 1, 1] },
      value: 7,
    });
  });

  it("sends a vector for traction and omits the value for fixed supports", () => {
    const elastic = { ...study, kind: "elastic" as const };
    const traction = {
      ...defaultDraft("elastic"),
      bcType: "traction" as const,
      vector: [0, 0, -9.81] as [number, number, number],
    };
    expect(addBcRequest(elastic, traction).value).toEqual([0, 0, -9.81]);

    const fixed = { ...defaultDraft("elastic"), bcType: "fixed" as const };
    expect("value" in addBcRequest(elastic, fixed)).toBe(false);
  });
});

describe("a flow study is reported but not authored", () => {
  // `StudyPayload.kind` is wider than the `StudyKind` enum on purpose: the
  // payload reports what a program contains, the enum names what the GUI can
  // build. A flow study is declared in the scene and has no patch operations
  // behind it, so the panel must read one without offering to edit it — and
  // must not throw when it meets one, which is what happened before this.
  it("offers no boundary-condition types to add", () => {
    expect(bcTypesFor("flow")).toEqual([]);
    expect(bcTypesFor("thermal")).toEqual(["dirichlet", "heat_flux"]);
    expect(bcTypesFor("elastic")).toEqual(["fixed", "traction"]);
  });

  it("is not an editable kind, and the two that are still are", () => {
    expect(isEditableStudyKind("flow")).toBe(false);
    expect(isEditableStudyKind("thermal")).toBe(true);
    expect(isEditableStudyKind("elastic")).toBe(true);
  });
});
