/**
 * Pure derivation of the visual scene-composition tree.
 *
 * Built from what the compiler reports: the construction nodes, the elements,
 * and the evaluated scene tree — which boolean consumes which solid, as the
 * program's `scene` actually came out. The Object tree panel renders this
 * read-only; editing always happens through the source or the viewport tools.
 *
 * The scene tree is a DAG (`Union(bolt, Mirror(bolt))` reads `bolt` twice),
 * so an item reached a second time is drawn as a reference to the first
 * rather than as a second object. A worker too old to report the tree gets
 * the flat list it always had.
 */

import type { SceneTree, SceneTreeItem } from "../../cadjoint/viewer/schema/payloads";
import type { ConstructionElement, ConstructionNode } from "./types";

export type { SceneTree, SceneTreeItem };

export interface SceneTreeRow {
  /** Stable key for rendering and collapse state: the row's path from the root. */
  key: string;
  /**
   * The row's name for tests and the DOM: the node, element or item id it
   * draws on its first occurrence (unique), its path on a reference.
   */
  testKey: string;
  /** Construction node this row selects, or null for scene/operator rows. */
  nodeId: string | null;
  /**
   * Construction element this row points the properties window at, when the
   * row is not a node: an operator, or a boolean. Null for the scene row and
   * for rows whose node carries the element itself.
   */
  elementId: string | null;
  /** Scene-tree item this row draws, when it came from the evaluated tree. */
  itemId: string | null;
  depth: number;
  /**
   * Icon/label vocabulary: the file, the solids, a sketch's operator, a
   * boolean, a pattern-like transform, a raw SDF leaf, or a grouping row.
   */
  kind:
    | "scene"
    | "profile"
    | "box"
    | "sphere"
    | "cylinder"
    | "operator"
    | "boolean"
    | "transform"
    | "sdf"
    | "group";
  label: string;
  /** Compact dimensions or point-count summary, when derivable. */
  detail: string | null;
  /** Named material assigned to the node, when any. */
  material: string | null;
  /** Sketch-level constraint count (0 hides the badge). */
  constraintCount: number;
  /** True when the row has children and can collapse. */
  group: boolean;
  /** How many operands in the scene tree read this item (0/1: not shared). */
  uses: number;
  /** True on every occurrence after the first: a reference, drawn without children. */
  reference: boolean;
}

/** Format one number the way the source panes do: short, no trailing zeros. */
function formatNumber(value: number): string {
  const rounded = Number(value.toFixed(2));
  return Object.is(rounded, -0) ? "0" : String(rounded);
}

/** Compact dimension summary for a solid, from its transform payload. */
export function dimensionSummary(node: ConstructionNode): string | null {
  if (node.kind === "profile") {
    const count = node.vertices.length;
    return count > 0 ? `${count} point${count === 1 ? "" : "s"}` : null;
  }
  const dimensions = node.transform?.dimensions;
  if (!dimensions) return null;
  const parts: string[] = [];
  const size = dimensions.size;
  if (Array.isArray(size)) parts.push(size.map(formatNumber).join(" × "));
  if (typeof dimensions.radius === "number") {
    parts.push(`r ${formatNumber(dimensions.radius)}`);
  }
  if (typeof dimensions.height === "number") {
    parts.push(`h ${formatNumber(dimensions.height)}`);
  }
  return parts.length > 0 ? parts.join(" · ") : null;
}

function plural(count: number, noun: string): string {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

/** A solid's row, with its operators nested beneath it in source order. */
function pushNode(
  rows: SceneTreeRow[],
  node: ConstructionNode,
  elements: readonly ConstructionElement[],
  key: string,
  depth: number,
  extra: { itemId?: string; uses?: number; reference?: boolean; group?: boolean } = {},
): void {
  const reference = extra.reference ?? false;
  const testKey = reference ? key : node.id;
  rows.push({
    key,
    testKey,
    nodeId: node.id,
    elementId: null,
    itemId: extra.itemId ?? null,
    depth,
    kind: node.kind,
    label: node.name ?? node.kind,
    detail: reference ? "shared" : dimensionSummary(node),
    material: node.material,
    constraintCount: reference ? 0 : node.constraints.length,
    group: !reference && (node.operators.length > 0 || (extra.group ?? false)),
    uses: extra.uses ?? 0,
    reference,
  });
  if (reference) return;
  for (const operator of node.operators) {
    const element = elements.find(
      (item) =>
        item.kind === "feature" && item.call === operator.kind && item.line === operator.line,
    );
    rows.push({
      key: `${key}-op-${operator.kind}-${operator.line}`,
      testKey: `${testKey}-op-${operator.kind}-${operator.line}`,
      nodeId: null,
      elementId: element?.id ?? null,
      itemId: null,
      depth: depth + 1,
      kind: "operator",
      label: operator.kind,
      detail: `line ${operator.line}`,
      material: null,
      constraintCount: 0,
      group: false,
      uses: 0,
      reference: false,
    });
  }
}

/** Icons the leaf classes of the SDF library share with the construction kinds. */
const LEAF_KINDS: Record<string, SceneTreeRow["kind"]> = {
  Box: "box",
  Sphere: "sphere",
  Cylinder: "cylinder",
};

/** How many operand lists read each item: 2+ means a shared reference. */
function useCounts(tree: SceneTree): Map<string, number> {
  const uses = new Map<string, number>();
  for (const item of tree.items) {
    for (const operand of item.operands) uses.set(operand, (uses.get(operand) ?? 0) + 1);
  }
  for (const root of tree.roots) uses.set(root, (uses.get(root) ?? 0) + 1);
  return uses;
}

/**
 * The short name of an item, for a parent's label hint: its own name, its
 * construction node's, an unnamed pattern's or mirror's operand (a copy of
 * `bolt` reads as `bolt`), or — for an unnamed boolean or leaf — its call.
 */
function shortName(
  item: SceneTreeItem,
  items: ReadonlyMap<string, SceneTreeItem>,
  nodesById: ReadonlyMap<string, ConstructionNode>,
): string {
  if (item.node !== null) {
    const node = nodesById.get(item.node);
    if (node) return node.name ?? node.kind;
  }
  if (item.name !== null) return item.name;
  const only = item.kind === "transform" ? items.get(item.operands[0]) : undefined;
  return only ? shortName(only, items, nodesById) : item.call;
}

/**
 * The label of a boolean or transform: its name, else the operation and a
 * hint of what it combines — "Union of rail_left, rail_right…".
 */
export function itemLabel(
  item: SceneTreeItem,
  items: ReadonlyMap<string, SceneTreeItem>,
  nodesById: ReadonlyMap<string, ConstructionNode>,
  element?: ConstructionElement,
): string {
  const named = item.name ?? element?.variable ?? element?.name ?? null;
  if (named !== null) return named;
  const operands = item.operands
    .map((id) => items.get(id))
    .filter((operand): operand is SceneTreeItem => operand !== undefined);
  if (operands.length === 0) return item.call;
  // `Union(bolt, Mirror(bolt))` names one thing, not two.
  const names = [...new Set(operands.map((operand) => shortName(operand, items, nodesById)))];
  const shown = names.slice(0, 2);
  return `${item.call} of ${shown.join(", ")}${names.length > 2 ? "…" : ""}`;
}

/**
 * Every construction node at or under *itemId*: what selecting a boolean
 * lights up in the viewport.
 */
export function descendantNodes(tree: SceneTree | null | undefined, itemId: string): Set<string> {
  const out = new Set<string>();
  if (!tree) return out;
  const items = new Map(tree.items.map((item) => [item.id, item]));
  const seen = new Set<string>();
  const stack = [itemId];
  while (stack.length > 0) {
    const id = stack.pop()!;
    if (seen.has(id)) continue;
    seen.add(id);
    const item = items.get(id);
    if (!item) continue;
    if (item.node !== null) out.add(item.node);
    stack.push(...item.operands);
  }
  return out;
}

/** The tree rows from the evaluated scene: roots first, then what no root reaches. */
function buildFromSceneTree(
  nodes: ConstructionNode[],
  elements: readonly ConstructionElement[],
  fileName: string,
  tree: SceneTree,
): SceneTreeRow[] {
  const items = new Map(tree.items.map((item) => [item.id, item]));
  const nodesById = new Map(nodes.map((node) => [node.id, node]));
  const elementsById = new Map(elements.map((element) => [element.id, element]));
  const uses = useCounts(tree);
  const reached = new Set(tree.items.flatMap((item) => (item.node !== null ? [item.node] : [])));
  const unused = nodes.filter((node) => !reached.has(node.id));
  const objectCount =
    reached.size + tree.items.filter((item) => item.kind === "boolean").length;

  const rows: SceneTreeRow[] = [
    {
      key: "scene",
      testKey: "scene",
      nodeId: null,
      elementId: null,
      itemId: null,
      depth: 0,
      kind: "scene",
      label: fileName,
      detail: objectCount > 0 ? plural(objectCount, "object") : "empty",
      material: null,
      constraintCount: 0,
      group: tree.roots.length > 0,
      uses: 0,
      reference: false,
    },
  ];
  const drawn = new Set<string>();

  const visit = (id: string, parentKey: string, depth: number) => {
    const item = items.get(id);
    if (!item) return;
    const key = `${parentKey}/${id}`;
    const reference = drawn.has(id);
    drawn.add(id);
    const count = uses.get(id) ?? 0;
    const node = item.node !== null ? nodesById.get(item.node) : undefined;
    if (node) {
      pushNode(rows, node, elements, key, depth, {
        itemId: id,
        uses: count,
        reference,
        group: item.operands.length > 0,
      });
    } else {
      const element = item.element !== null ? elementsById.get(item.element) : undefined;
      const label = itemLabel(item, items, nodesById, element);
      const operands = item.operands.length;
      const kind: SceneTreeRow["kind"] =
        item.kind === "boolean"
          ? "boolean"
          : item.kind === "transform"
            ? "transform"
            : (LEAF_KINDS[item.call] ?? "sdf");
      // A label already spelling the call ("Union of …") does not repeat it.
      const call = label.startsWith(item.call) ? null : item.call;
      const detail = reference
        ? "shared"
        : item.kind === "boolean"
          ? [call, plural(operands, "operand")].filter(Boolean).join(" · ")
          : call;
      rows.push({
        key,
        testKey: reference ? key : id,
        nodeId: null,
        elementId: element?.id ?? null,
        itemId: id,
        depth,
        kind,
        label,
        detail,
        material: null,
        constraintCount: 0,
        group: !reference && operands > 0,
        uses: count,
        reference,
      });
    }
    if (reference) return;
    for (const operand of item.operands) visit(operand, key, depth + 1);
  };
  for (const root of tree.roots) visit(root, "scene", 1);

  // Sketches and solids the program built but `scene` never reads: kept
  // reachable, under their own heading rather than mixed into the tree.
  if (unused.length > 0) {
    rows.push({
      key: "unused",
      testKey: "unused",
      nodeId: null,
      elementId: null,
      itemId: null,
      depth: 1,
      kind: "group",
      label: "Not in scene",
      detail: plural(unused.length, "object"),
      material: null,
      constraintCount: 0,
      group: true,
      uses: 0,
      reference: false,
    });
    for (const node of unused) pushNode(rows, node, elements, `unused/${node.id}`, 2);
  }
  return rows;
}

/**
 * The scene → boolean → solid → operator rows the Object tree draws.
 *
 * With the evaluated scene tree, rows follow the composition: booleans nest
 * their operands in argument order, each solid appears where it is used, and
 * a sketch's operators nest beneath it. Without one, or with one that has no
 * roots (before the first compile, or a `scene` that evaluates to nothing),
 * nodes appear as the compiler reported them and the booleans follow, flat.
 */
export function buildSceneTree(
  nodes: ConstructionNode[],
  elements: readonly ConstructionElement[] = [],
  /** The root row's label: the program's file name (not "scene", which the starter also binds). */
  fileName = "scene.py",
  tree: SceneTree | null = null,
): SceneTreeRow[] {
  if (tree && tree.roots.length > 0) return buildFromSceneTree(nodes, elements, fileName, tree);
  const booleans = elements.filter((element) => element.kind === "boolean");
  const count = nodes.length + booleans.length;
  const rows: SceneTreeRow[] = [
    {
      key: "scene",
      testKey: "scene",
      nodeId: null,
      elementId: null,
      itemId: null,
      depth: 0,
      kind: "scene",
      label: fileName,
      detail: count > 0 ? `${count} object${count === 1 ? "" : "s"}` : "empty",
      material: null,
      constraintCount: 0,
      group: count > 0,
      uses: 0,
      reference: false,
    },
  ];
  for (const node of nodes) pushNode(rows, node, elements, node.id, 1);
  // The booleans that combine the objects above, in source order: they have
  // no outline in the viewport, so the tree is where they can be reached.
  for (const boolean of booleans) {
    const operands = boolean.arguments.find((argument) => argument.name === "operands");
    const names = operands?.text ? operands.text.split(",").length : 0;
    rows.push({
      key: boolean.id,
      testKey: boolean.id,
      nodeId: null,
      elementId: boolean.id,
      itemId: null,
      depth: 1,
      kind: "boolean",
      label: boolean.variable ?? boolean.name ?? boolean.call,
      detail: `${boolean.call} · ${plural(names, "operand")}`,
      material: null,
      constraintCount: 0,
      group: false,
      uses: 0,
      reference: false,
    });
  }
  return rows;
}

/** Rows visible given a set of collapsed group keys. */
export function visibleRows(
  rows: SceneTreeRow[],
  collapsed: ReadonlySet<string>,
): SceneTreeRow[] {
  const out: SceneTreeRow[] = [];
  let hideDeeperThan: number | null = null;
  for (const row of rows) {
    if (hideDeeperThan !== null && row.depth > hideDeeperThan) continue;
    hideDeeperThan = null;
    out.push(row);
    if (row.group && collapsed.has(row.key)) hideDeeperThan = row.depth;
  }
  return out;
}
