/**
 * Pure helpers for the study panel: readable selection summaries and the
 * request bodies for the study patch operations.
 *
 * Studies are declared in the scene program; the panel only ever edits them
 * through /patch source edits, so everything here is a straight mapping from
 * payload shapes to patch-request shapes with no state of its own.
 */

import type {
  StudyBc,
  StudyBcType,
  StudyPayload,
  StudyPayloadKind,
  StudySelection,
} from "./types";
import { byId } from "./identity";

/** Compact numeric formatting for selection summaries: drop trailing zeros. */
const num = (value: number): string => {
  const fixed = Math.abs(value) >= 1000 ? value.toPrecision(4) : value.toFixed(2);
  return String(Number(fixed));
};

const vec = (values: number[]): string => `[${values.map(num).join(", ")}]`;

/** One-line human summary of a selection description, composites included. */
export function describeSelection(selection: StudySelection): string {
  switch (selection.kind) {
    case "box":
      return `box ${vec(selection.min_corner)} → ${vec(selection.max_corner)}`;
    case "sphere":
      return `sphere ${vec(selection.center)} r ${num(selection.radius)}`;
    case "halfspace":
      return `halfspace at ${vec(selection.point)} · n ${vec(selection.normal)}`;
    case "side":
      return selection.tol === null
        ? `side ${selection.side}`
        : `side ${selection.side} ± ${num(selection.tol)}`;
    case "predicate":
      return `predicate ${selection.name}()`;
    case "and":
      return selection.operands.map(describeSelection).join(" ∧ ");
    case "or":
      return selection.operands.map(describeSelection).join(" ∨ ");
    case "not":
      return `¬(${describeSelection(selection.operand)})`;
  }
}

export const BC_LABELS: Record<StudyBcType, string> = {
  dirichlet: "Fixed value",
  heat_flux: "Heat flux",
  fixed: "Fixed support",
  traction: "Traction",
};

/**
 * BC types that make sense for a study kind.
 *
 * A flow study's conditions — inlet, outlet, walls, a heat source — are
 * declared in the scene and have no patch operations behind them, so the
 * panel has nothing to offer for one and says so by offering nothing. That
 * is why the kind is wider here than `StudyKind`: the payload reports what a
 * program *contains*, the enum names what the GUI can *author*, and flow is
 * currently the first that is one without the other.
 */
export function bcTypesFor(kind: StudyPayloadKind): StudyBcType[] {
  if (kind === "thermal") return ["dirichlet", "heat_flux"];
  if (kind === "elastic") return ["fixed", "traction"];
  return [];
}

/** Whether the GUI can add and edit this kind of study's conditions. */
export function isEditableStudyKind(kind: StudyPayloadKind): kind is "thermal" | "elastic" {
  return kind === "thermal" || kind === "elastic";
}

/** The scalar/vector a BC row edits, or null for `fixed` (no value). */
export function bcValue(bc: StudyBc): number | [number, number, number] | null {
  if (bc.type === "dirichlet") return bc.value ?? 0;
  if (bc.type === "heat_flux") return bc.flux ?? 0;
  if (bc.type === "traction") return bc.vector ?? [0, 0, 0];
  return null;
}

/** The numeric constructor arguments a study kind exposes for editing. */
export function studyArguments(study: StudyPayload): { key: string; value: number }[] {
  const rows: { key: string; value: number }[] = [];
  if (typeof study.resolution === "number") {
    rows.push({ key: "resolution", value: study.resolution });
  }
  // A study's material map carries numbers for the properties stated in the
  // declaration and the sentinel string "material" for the ones it defers to
  // the assigned Material. Only the numbers are literals in the source, so
  // only the numbers get an editable row; a deferred property is shown by the
  // material chip on the card instead of by a field that would rewrite it.
  for (const [key, value] of Object.entries(study.material ?? {})) {
    if (typeof value === "number") rows.push({ key, value });
  }
  if (study.kind === "thermal") {
    rows.push({ key: "source", value: typeof study.source === "number" ? study.source : 0 });
  }
  return rows;
}

/** Selection kinds the builder form offers (predicates are code-only). */
export type BuilderSelectionKind = "side" | "box" | "sphere" | "halfspace";

/** Editable state of the add-BC builder form, converted on submit. */
export interface BcDraft {
  bcType: StudyBcType;
  selectionKind: BuilderSelectionKind;
  side: string;
  minCorner: [number, number, number];
  maxCorner: [number, number, number];
  center: [number, number, number];
  radius: number;
  point: [number, number, number];
  normal: [number, number, number];
  value: number;
  vector: [number, number, number];
}

export function defaultDraft(kind: StudyPayloadKind): BcDraft {
  return {
    bcType: kind === "thermal" ? "dirichlet" : "fixed",
    selectionKind: "side",
    side: "+x",
    minCorner: [0, 0, 0],
    maxCorner: [1, 1, 1],
    center: [0, 0, 0],
    radius: 0.5,
    point: [0, 0, 0],
    normal: [0, 0, 1],
    value: kind === "thermal" ? 100 : 0,
    vector: [0, 0, -1],
  };
}

export function draftSelection(draft: BcDraft): StudySelection {
  switch (draft.selectionKind) {
    case "side":
      return { kind: "side", side: draft.side, tol: null };
    case "box":
      return { kind: "box", min_corner: [...draft.minCorner], max_corner: [...draft.maxCorner] };
    case "sphere":
      return { kind: "sphere", center: [...draft.center], radius: draft.radius };
    case "halfspace":
      return { kind: "halfspace", point: [...draft.point], normal: [...draft.normal] };
  }
}

/** Body for POST /patch adding the drafted BC (App prepends `source`). */
export function addBcRequest(study: StudyPayload, draft: BcDraft): Record<string, unknown> {
  const body: Record<string, unknown> = {
    op: "add_study_bc",
    ...byId(study),
    study: study.index,
    bc_type: draft.bcType,
    selection: draftSelection(draft),
  };
  // `value` is required for valued BCs and forbidden for fixed supports.
  if (draft.bcType === "traction") body.value = [...draft.vector];
  else if (draft.bcType !== "fixed") body.value = draft.value;
  return body;
}

export function addStudyRequest(kind: "thermal" | "elastic"): Record<string, unknown> {
  return { op: "add_study", kind };
}

export function deleteStudyRequest(study: StudyPayload): Record<string, unknown> {
  return { op: "delete_study", ...byId(study), study: study.index };
}

export function deleteBcRequest(study: StudyPayload, bc: number): Record<string, unknown> {
  return { op: "delete_study_bc", ...byId(study), study: study.index, bc };
}

export function setBcValueRequest(
  study: StudyPayload,
  bc: number,
  value: number | number[],
): Record<string, unknown> {
  return { op: "set_study_value", ...byId(study), study: study.index, bc, value };
}

export function setArgumentRequest(
  study: StudyPayload,
  argument: string,
  value: number,
): Record<string, unknown> {
  return { op: "set_study_value", ...byId(study), study: study.index, argument, value };
}
