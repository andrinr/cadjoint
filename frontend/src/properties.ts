/**
 * Which construction elements the properties window shows, and how.
 *
 * The window follows two signals: the node selection the viewport and the
 * object tree share, and the element the tree's operator rows point at
 * directly (an extrusion has no outline to click). The rules that turn those
 * into a list of sections — a sketch brings its plane along, a plane
 * selection shows the plane first, an inspected element wins over the
 * selection — are pure, so they are tested without a DOM.
 */

import type {
  ConstructionArgument,
  ConstructionElement,
  ConstructionNode,
  Selection,
} from "./types";

/** Human label for an element's kind, by the call that built it. */
export function kindLabel(element: ConstructionElement): string {
  switch (element.kind) {
    case "sketch":
      return "Sketch";
    case "plane":
      return element.call === "SketchPlane" ? "Sketch plane" : "Derived sketch plane";
    case "primitive":
      return element.call.charAt(0).toUpperCase() + element.call.slice(1);
    case "feature":
      return element.call.charAt(0).toUpperCase() + element.call.slice(1);
    case "boolean":
      return element.call;
  }
}

/** What the section is titled: the literal name, the variable, or the call. */
export function elementTitle(element: ConstructionElement): string {
  return element.name ?? element.variable ?? element.call;
}

/** The element a node was built by, matched by stable id first, then by line. */
export function elementForNode(
  elements: readonly ConstructionElement[],
  node: ConstructionNode | undefined,
): ConstructionElement | null {
  if (!node) return null;
  if (node.stableId) {
    const byId = elements.find((element) => element.stableId === node.stableId);
    if (byId) return byId;
  }
  if (node.line === null) return null;
  const kind = node.kind === "profile" ? "sketch" : "primitive";
  return elements.find((element) => element.kind === kind && element.line === node.line) ?? null;
}

/** The plane element that belongs to a sketch element. */
export function planeOf(
  elements: readonly ConstructionElement[],
  sketch: ConstructionElement,
): ConstructionElement | null {
  return elements.find((element) => element.kind === "plane" && element.owner === sketch.id) ?? null;
}

/** The feature element an operator chip (kind + line) names. */
export function elementForOperator(
  elements: readonly ConstructionElement[],
  kind: string,
  line: number,
): ConstructionElement | null {
  return (
    elements.find(
      (element) => element.kind === "feature" && element.call === kind && element.line === line,
    ) ?? null
  );
}

/**
 * The sections the window shows, in order.
 *
 * An inspected element takes precedence; it is what the user just pointed
 * at. Otherwise the selection decides: a sketch is shown with its plane
 * under it, a plane selection puts the plane first, a primitive stands
 * alone. Nothing selected is an empty list, and the window explains itself.
 */
export function resolveSections(
  elements: readonly ConstructionElement[],
  nodes: readonly ConstructionNode[],
  selection: Selection | null,
  inspected: string | null,
): ConstructionElement[] {
  if (inspected) {
    const element = elements.find((item) => item.id === inspected);
    if (element) {
      if (element.kind === "sketch") {
        const plane = planeOf(elements, element);
        return plane ? [element, plane] : [element];
      }
      return [element];
    }
  }
  if (!selection) return [];
  const node = nodes.find((item) => item.id === selection.nodeId);
  const element = elementForNode(elements, node);
  if (!element) return [];
  if (element.kind !== "sketch") return [element];
  const plane = planeOf(elements, element);
  if (!plane) return [element];
  return selection.part === "plane" ? [plane, element] : [element, plane];
}

/** Short, source-like rendering of a value for a read-only row. */
export function formatArgument(argument: ConstructionArgument): string {
  if (argument.kind === "string") return String(argument.value ?? "");
  if (argument.kind === "reference") return String(argument.value ?? argument.text);
  if (argument.kind === "expression") {
    return typeof argument.value === "string" ? argument.value : argument.text;
  }
  const value = argument.value;
  if (Array.isArray(value)) return `[${value.map(formatNumber).join(", ")}]`;
  return typeof value === "number" ? formatNumber(value) : argument.text;
}

function formatNumber(value: number): string {
  const rounded = Number(value.toPrecision(6));
  return Object.is(rounded, -0) ? "0" : String(rounded);
}

/** A plane argument the payload could not state, filled from the runtime frame. */
export function runtimePlaneRows(
  element: ConstructionElement,
  nodes: readonly ConstructionNode[],
): { name: string; text: string }[] {
  if (element.kind !== "plane") return [];
  const stated = new Set(element.arguments.map((argument) => argument.name));
  if (stated.has("origin") && stated.has("normal")) return [];
  const owner = nodes.find((node) => node.stableId === element.owner);
  const plane = owner?.plane;
  if (!plane) return [];
  const rows: { name: string; text: string }[] = [];
  if (!stated.has("origin")) {
    rows.push({ name: "origin", text: `[${plane.origin.map(formatNumber).join(", ")}]` });
  }
  if (!stated.has("normal")) {
    rows.push({ name: "normal", text: `[${plane.normal.map(formatNumber).join(", ")}]` });
  }
  return rows;
}
