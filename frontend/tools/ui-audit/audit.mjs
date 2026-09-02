#!/usr/bin/env node
/**
 * ui-audit — a design linter for the cadjoint playground.
 *
 * Drives a real chromium through every editing mode and Simulate tab at
 * several viewport sizes and measures what eyeballing misses: the type scale
 * actually in use, elements that overflow or clip their text, rows whose
 * children do not share a baseline, panels whose blocks do not share a left
 * edge, controls whose heights differ by a hair, interactive elements that
 * overlap or fall off screen, and every colour in use with its WCAG contrast
 * against the background it is actually painted on.
 *
 * Everything is measured from computed style and layout in the live page, so
 * it sees the built CSS as the browser resolves it — cascade, custom
 * properties and all — rather than what the stylesheet source appears to say.
 *
 * Usage:
 *   node frontend/tools/ui-audit/audit.mjs [--port N] [--out DIR] [options]
 *
 * See README.md next to this file for the full contract.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const { chromium } = require(path.join(HERE, "../../node_modules/@playwright/test"));

// Chromium needs these to get a real GPU adapter on macOS; the playground
// paints its viewport with WebGPU and lays out around the canvas.
const GPU_ARGS = [
  "--enable-unsafe-webgpu",
  "--enable-gpu",
  "--use-angle=metal",
  "--ignore-gpu-blocklist",
];

const VIEWPORTS = [
  { width: 1440, height: 900 },
  { width: 1280, height: 800 },
];

/**
 * Every UI state the linter walks, in visit order.
 *
 * `open` clicks a control before measuring (menus, popovers, side panels) and
 * `expect` is the testid that must then be visible for the state to count;
 * `close` (default: click `open` again, then Escape) puts it back.
 */
const STATES = [
  { id: "model", mode: "model" },
  { id: "model-menu-file", mode: "model", open: "menu-file", expect: "menu-file-save-as" },
  // Every panel is a window now, and the two states that used to toggle the
  // object tree and the material browser open are simply the default desk:
  // both are docked from the start. What `model` does not already show is a
  // window that has left the grid, so those two ids are spent on the two
  // arrangements the window system added — parked in the tray, and floating.
  { id: "model-tray", mode: "model", click: ["object-tree-close", "material-close"], expect: "window-tray", reset: true },
  { id: "model-floating", mode: "model", click: ["menu-window", "menu-window-float-objects"], expect: "object-tree-panel", reset: true },
  { id: "model-render-popover", mode: "model", open: "display-options", expect: "render-popover" },
  // The sketch panel only exists while a profile is selected, so these two
  // states pick one out of the object tree first.
  { id: "sketch", mode: "sketch", selectProfile: true, expect: "sketch-panel" },
  { id: "sketch-solver", mode: "sketch", selectProfile: true, open: "solver-toggle", expect: "solver-panel" },
  { id: "simulate-meshes", mode: "simulate", tab: "meshes" },
  { id: "simulate-studies", mode: "simulate", tab: "studies" },
  { id: "simulate-optimize", mode: "simulate", tab: "optimize" },
  { id: "simulate-results", mode: "simulate", tab: "results" },
];

function parseArgs(argv) {
  const args = {
    port: 8765,
    out: path.join(process.cwd(), "ui-audit-out"),
    settle: 700,
    timeout: 90_000,
    headed: false,
    states: null,
    viewports: null,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    const next = () => argv[(i += 1)];
    if (flag === "--port") args.port = Number(next());
    else if (flag === "--url") args.url = next();
    else if (flag === "--out") args.out = path.resolve(next());
    else if (flag === "--settle") args.settle = Number(next());
    else if (flag === "--timeout") args.timeout = Number(next());
    else if (flag === "--headed") args.headed = true;
    else if (flag === "--include-editor") args.includeEditor = true;
    else if (flag === "--solve") args.solve = true;
    else if (flag === "--states") args.states = next().split(",").map((s) => s.trim());
    else if (flag === "--viewports")
      args.viewports = next()
        .split(",")
        .map((spec) => {
          const [width, height] = spec.trim().split("x").map(Number);
          return { width, height };
        });
    else if (flag === "--help" || flag === "-h") {
      process.stdout.write(
        [
          "node frontend/tools/ui-audit/audit.mjs [options]",
          "  --port N          playground port (default 8765)",
          "  --url URL         full base URL, overrides --port",
          "  --out DIR         output directory (default ./ui-audit-out)",
          "  --states a,b      subset of: " + STATES.map((s) => s.id).join(","),
          "  --viewports WxH,… default 1440x900,1280x800",
          "  --settle MS       wait after each state change (default 700)",
          "  --include-editor  also run layout checks inside the CodeMirror editor",
          "  --solve           solve a study and inspect a mesh first, so the",
          "                    Results/Meshes tabs are audited with real content",
          "  --headed          show the browser",
          "",
        ].join("\n"),
      );
      process.exit(0);
    }
  }
  return args;
}

/* ------------------------------------------------------------------ */
/* The in-page collector. Runs inside the browser; must be self-contained. */
/* ------------------------------------------------------------------ */

function collectInPage(options) {
  const SKIP_TAGS = new Set(["script", "style", "meta", "link", "title", "head", "br"]);
  const INTERACTIVE = "a[href],button,input,select,textarea,summary,[role=button],[role=tab],[tabindex]:not([tabindex='-1'])";
  // Subtrees whose layout is owned by a third party (CodeMirror) or by a CSS
  // 3D transform (the view cube): their boxes are meaningless to a 2D linter.
  // Colour and type are still ours, so only the layout checks are exempted.
  const LAYOUT_EXEMPT = options && options.includeEditor
    ? ".cube-stage"
    : ".cm-scroller, .cm-content, .cm-gutters, .cm-tooltip, .cm-panels, .cube-stage";

  const out = { type: [], color: [], violations: [], stats: {} };
  const add = (cls, severity, el, detail) =>
    out.violations.push({ class: cls, severity, selector: sel(el), text: snippet(el), detail });

  function snippet(el) {
    const t = (el.textContent || "").replace(/\s+/g, " ").trim();
    return t.length > 48 ? t.slice(0, 48) + "…" : t;
  }

  function sel(el) {
    const parts = [];
    let node = el;
    for (let depth = 0; node && node.nodeType === 1 && depth < 3; depth += 1) {
      let piece = node.tagName.toLowerCase();
      const tid = node.getAttribute && node.getAttribute("data-testid");
      if (tid) {
        parts.unshift(`${piece}[data-testid=${tid}]`);
        return parts.join(" > ");
      }
      const cls = (node.getAttribute("class") || "")
        .trim()
        .split(/\s+/)
        .filter(Boolean)
        .slice(0, 2);
      if (cls.length) piece += "." + cls.join(".");
      else if (node.parentElement) {
        const sameTag = Array.from(node.parentElement.children).filter(
          (c) => c.tagName === node.tagName,
        );
        if (sameTag.length > 1) piece += `:nth-of-type(${sameTag.indexOf(node) + 1})`;
      }
      parts.unshift(piece);
      node = node.parentElement;
    }
    return parts.join(" > ");
  }

  /* ---------- colour helpers ---------- */
  function parseColor(value) {
    if (!value) return null;
    const m = /^rgba?\(([^)]+)\)$/.exec(value.trim());
    if (!m) return null;
    const parts = m[1].split(/[,\s/]+/).filter(Boolean).map(Number);
    if (parts.length < 3 || parts.some((n) => Number.isNaN(n))) return null;
    return { r: parts[0], g: parts[1], b: parts[2], a: parts.length > 3 ? parts[3] : 1 };
  }
  const over = (fg, bg) => ({
    r: fg.r * fg.a + bg.r * (1 - fg.a),
    g: fg.g * fg.a + bg.g * (1 - fg.a),
    b: fg.b * fg.a + bg.b * (1 - fg.a),
    a: 1,
  });
  function luminance(c) {
    const ch = [c.r, c.g, c.b].map((v) => {
      const s = v / 255;
      return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
    });
    return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2];
  }
  const contrast = (a, b) => {
    const la = luminance(a);
    const lb = luminance(b);
    return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
  };
  const hex = (c) =>
    "#" +
    [c.r, c.g, c.b]
      .map((v) => Math.round(v).toString(16).padStart(2, "0"))
      .join("") +
    (c.a < 1 ? ` @${c.a.toFixed(2)}` : "");

  /** Composite the painted background behind `el` (self included). */
  function paintedBackground(el) {
    const layers = [];
    let node = el;
    let painted = true; // false when something un-modelable (image/gradient/canvas) intervenes
    while (node && node.nodeType === 1) {
      const cs = getComputedStyle(node);
      if (cs.backgroundImage && cs.backgroundImage !== "none") painted = false;
      if (node.tagName === "CANVAS") painted = false;
      const bg = parseColor(cs.backgroundColor);
      if (bg && bg.a > 0) {
        layers.push(bg);
        if (bg.a >= 0.999) break;
      }
      node = node.parentElement;
    }
    let base = { r: 255, g: 255, b: 255, a: 1 };
    for (let i = layers.length - 1; i >= 0; i -= 1) base = over(layers[i], base);
    return { color: base, painted };
  }

  /* ---------- element sweep ---------- */
  const all = Array.from(document.querySelectorAll("*")).filter((el) => {
    if (SKIP_TAGS.has(el.tagName.toLowerCase())) return false;
    if (el.namespaceURI === "http://www.w3.org/2000/svg" && el.tagName.toLowerCase() !== "svg")
      return false;
    return true;
  });

  const info = new Map();
  for (const el of all) {
    const cs = getComputedStyle(el);
    if (cs.display === "none" || cs.visibility === "hidden" || cs.opacity === "0") continue;
    const rect = el.getBoundingClientRect();
    if (rect.width < 1 || rect.height < 1) continue;
    // Visually hidden, the sr-only way: a one-pixel box clipped to nothing so
    // that only assistive technology reads it. Its text is *meant* to be
    // visually unreachable, so measuring it as clipped, overflowing or
    // off-scale is a false positive — the dock library's aria-live announcer
    // alone produced 42 of them across a run.
    const clippedAway =
      cs.clipPath === "inset(50%)" ||
      /^rect\(0px[,)\s]/.test(cs.clip || "") ||
      (rect.width <= 1 && rect.height <= 1 && (cs.overflow === "hidden" || cs.overflow === "clip"));
    if (clippedAway) continue;
    const ownText = Array.from(el.childNodes).some(
      (n) => n.nodeType === 3 && n.textContent.trim().length > 0,
    );
    const exempt = el.closest(LAYOUT_EXEMPT) !== null;
    info.set(el, { el, cs, rect, ownText, exempt });
  }
  out.stats.elements = info.size;
  out.stats.layoutExempt = [...info.values()].filter((r) => r.exempt).length;
  /** Layout checks run on our own chrome only. */
  const layout = () => [...info.values()].filter((r) => !r.exempt);

  /* ---------- 1. type-scale census ---------- */
  for (const rec of info.values()) {
    if (!rec.ownText) continue;
    const { cs } = rec;
    out.type.push({
      key: [cs.fontSize, cs.fontWeight, cs.lineHeight, cs.fontFamily.split(",")[0].replace(/["']/g, "")].join("|"),
      fontSize: cs.fontSize,
      fontWeight: cs.fontWeight,
      lineHeight: cs.lineHeight,
      fontFamily: cs.fontFamily.split(",")[0].replace(/["']/g, ""),
      letterSpacing: cs.letterSpacing,
      textTransform: cs.textTransform,
      selector: sel(rec.el),
      text: snippet(rec.el),
    });
  }

  /* ---------- 5. colour census + contrast ---------- */
  for (const rec of info.values()) {
    const { cs, el, ownText } = rec;
    const push = (role, raw) => {
      const c = parseColor(raw);
      out.color.push({
        role,
        value: c ? hex(c) : raw,
        raw,
        selector: sel(el),
      });
    };
    if (ownText) push("text", cs.color);
    const bg = parseColor(cs.backgroundColor);
    if (bg && bg.a > 0) push("background", cs.backgroundColor);
    for (const side of ["Top", "Right", "Bottom", "Left"]) {
      if (parseFloat(cs[`border${side}Width`]) > 0 && cs[`border${side}Style`] !== "none")
        push("border", cs[`border${side}Color`]);
    }

    if (!ownText) continue;
    const fg = parseColor(cs.color);
    if (!fg || fg.a === 0) continue;
    const behind = paintedBackground(el.parentElement ?? el);
    const selfBg = parseColor(cs.backgroundColor);
    let base = behind.color;
    if (selfBg && selfBg.a > 0) base = over(selfBg, base);
    if (!behind.painted) continue; // text over canvas/gradient — cannot measure
    const ratio = contrast(over(fg, base), base);
    const size = parseFloat(cs.fontSize);
    const bold = Number(cs.fontWeight) >= 700;
    const large = size >= 24 || (size >= 18.66 && bold);
    const need = large ? 3 : 4.5;
    if (ratio < need) {
      add("contrast-text", ratio < need - 1.5 ? "high" : "medium", el, {
        ratio: Number(ratio.toFixed(2)),
        required: need,
        color: hex(fg),
        background: hex(base),
        fontSize: cs.fontSize,
        fontWeight: cs.fontWeight,
      });
    }
  }

  // Non-text contrast: interactive borders / control fills against their surround.
  for (const el of document.querySelectorAll("button,input,select,textarea,[role=tab]")) {
    const rec = info.get(el);
    if (!rec) continue;
    const cs = rec.cs;
    const bw = parseFloat(cs.borderTopWidth);
    if (!(bw > 0) || cs.borderTopStyle === "none") continue;
    const bc = parseColor(cs.borderTopColor);
    if (!bc || bc.a === 0) continue;
    const behind = paintedBackground(el.parentElement ?? el);
    if (!behind.painted) continue;
    const ratio = contrast(over(bc, behind.color), behind.color);
    if (ratio < 3)
      add("contrast-control-border", "low", el, {
        ratio: Number(ratio.toFixed(2)),
        required: 3,
        border: hex(bc),
        background: hex(behind.color),
      });
  }

  /* ---------- 2. overflow ---------- */
  const scrollableX = (cs) => cs.overflowX === "auto" || cs.overflowX === "scroll";
  const scrollableY = (cs) => cs.overflowY === "auto" || cs.overflowY === "scroll";

  /**
   * scrollWidth counts absolutely positioned descendants, so a dropdown wider
   * than the button it hangs off reads as "overflow". That is by design.
   */
  function overflowIsPositioned(el) {
    const box = el.getBoundingClientRect();
    for (const kid of el.querySelectorAll("*")) {
      const kcs = getComputedStyle(kid);
      if (kcs.position !== "absolute" && kcs.position !== "fixed") continue;
      const r = kid.getBoundingClientRect();
      if (r.width > 0 && (r.right > box.right + 1 || r.left < box.left - 1)) return true;
    }
    return false;
  }

  for (const rec of layout()) {
    const { el, cs } = rec;
    if (el === document.documentElement || el.tagName === "BODY") continue;
    const overX = el.scrollWidth - el.clientWidth;
    const overY = el.scrollHeight - el.clientHeight;
    const ellipsis = cs.textOverflow === "ellipsis";
    if (overX > 1 && ellipsis) {
      // Not a bug by itself, but the fixing agent needs the list: this is
      // exactly what "text is cut off" looks like to a user.
      add("text-truncated-ellipsis", "low", el, {
        by: overX,
        clientWidth: el.clientWidth,
        scrollWidth: el.scrollWidth,
        full: (el.textContent || "").replace(/\s+/g, " ").trim().slice(0, 120),
      });
    }
    if (overX > 1 && !scrollableX(cs) && !ellipsis && !overflowIsPositioned(el)) {
      add("overflow-x", cs.overflowX === "visible" ? "high" : "medium", el, {
        by: overX,
        clientWidth: el.clientWidth,
        scrollWidth: el.scrollWidth,
        overflowX: cs.overflowX,
        whiteSpace: cs.whiteSpace,
      });
    }
    // A scroll container is legal, but one that parks most of its content —
    // or a whole control — out of sight at rest reads as "the panel is cut
    // off". That is what a user calls an overflow.
    if (scrollableY(cs) && overY > 16) {
      const box = el.getBoundingClientRect();
      const buried = Array.from(el.querySelectorAll(INTERACTIVE)).filter((kid) => {
        const r = kid.getBoundingClientRect();
        return r.height > 0 && (r.top >= box.bottom - 1 || r.bottom <= box.top + 1);
      });
      const fraction = overY / el.scrollHeight;
      if (buried.length || fraction > 0.2)
        add("scroll-content-hidden", buried.length ? "high" : "medium", el, {
          hiddenPx: overY,
          hiddenFraction: Number(fraction.toFixed(2)),
          clientHeight: el.clientHeight,
          scrollHeight: el.scrollHeight,
          buriedControls: buried.length,
          buriedExamples: buried.slice(0, 3).map(sel),
        });
    }
    if (overY > 1 && !scrollableY(cs) && cs.overflowY !== "visible") {
      add("overflow-y-clipped", "high", el, {
        by: overY,
        clientHeight: el.clientHeight,
        scrollHeight: el.scrollHeight,
        overflowY: cs.overflowY,
      });
    }

    // Flex child of an overflowing row without min-width:0 — the classic cause.
    const parent = el.parentElement;
    if (!parent) continue;
    const pcs = getComputedStyle(parent);
    const isRow =
      (pcs.display === "flex" || pcs.display === "inline-flex") &&
      pcs.flexDirection.startsWith("row");
    if (!isRow) continue;
    const parentOverflows = parent.scrollWidth - parent.clientWidth > 1;
    if (parentOverflows && cs.minWidth === "auto" && overX > 1) {
      add("flex-child-missing-min-width", "high", el, {
        parent: sel(parent),
        childOverBy: overX,
        parentOverBy: parent.scrollWidth - parent.clientWidth,
      });
    }
  }

  // Text clipped by a clipping ancestor (fixed height / overflow hidden).
  for (const rec of layout()) {
    if (!rec.ownText) continue;
    const { el, cs } = rec;
    if (cs.textOverflow === "ellipsis") continue;
    const range = document.createRange();
    range.selectNodeContents(el);
    const rects = Array.from(range.getClientRects()).filter((r) => r.width > 0 && r.height > 0);
    if (!rects.length) continue;
    const text = rects.reduce(
      (acc, r) => ({
        top: Math.min(acc.top, r.top),
        bottom: Math.max(acc.bottom, r.bottom),
        left: Math.min(acc.left, r.left),
        right: Math.max(acc.right, r.right),
      }),
      { top: Infinity, bottom: -Infinity, left: Infinity, right: -Infinity },
    );
    // Walk out to the first box that clips each axis. A scrollable ancestor
    // ends the walk for that axis: the text is reachable by scrolling.
    let node = el;
    let checkX = true;
    let checkY = true;
    const cut = { right: 0, left: 0, bottom: 0, top: 0 };
    let culprit = null;
    while (node && node.nodeType === 1 && (checkX || checkY)) {
      const ncs = node === el ? cs : getComputedStyle(node);
      if (checkX && scrollableX(ncs)) checkX = false;
      if (checkY && scrollableY(ncs)) checkY = false;
      const clipsX = checkX && (ncs.overflowX === "hidden" || ncs.overflowX === "clip");
      const clipsY = checkY && (ncs.overflowY === "hidden" || ncs.overflowY === "clip");
      if (clipsX || clipsY) {
        const box = node.getBoundingClientRect();
        if (clipsX) {
          cut.right = Math.max(cut.right, text.right - (box.right - parseFloat(ncs.borderRightWidth)));
          cut.left = Math.max(cut.left, box.left + parseFloat(ncs.borderLeftWidth) - text.left);
          checkX = false;
        }
        if (clipsY) {
          cut.bottom = Math.max(cut.bottom, text.bottom - (box.bottom - parseFloat(ncs.borderBottomWidth)));
          cut.top = Math.max(cut.top, box.top + parseFloat(ncs.borderTopWidth) - text.top);
          checkY = false;
        }
        culprit = node;
      }
      node = node.parentElement;
    }
    const worst = Math.max(cut.right, cut.left, cut.bottom, cut.top);
    if (culprit && worst > 1.5) {
      add("text-clipped", "high", el, {
        clippedBy: sel(culprit),
        cutPx: Number(worst.toFixed(1)),
        side: Object.entries(cut).sort((a, b) => b[1] - a[1])[0][0],
      });
    }
  }

  /* ---------- 3. alignment ---------- */
  // 3a. rows: children of a flex row should share a baseline / centre.
  const DESCENT = 0.21; // fraction of font-size below the baseline, Chrome text rects
  for (const rec of layout()) {
    const { el, cs } = rec;
    if (!(cs.display === "flex" || cs.display === "inline-flex")) continue;
    if (!cs.flexDirection.startsWith("row")) continue;
    if (cs.flexWrap === "wrap") continue;
    const kids = Array.from(el.children)
      .map((c) => info.get(c))
      .filter(Boolean)
      .filter((k) => k.cs.position !== "absolute" && k.cs.position !== "fixed");
    if (kids.length < 2) continue;

    // Centres — only meaningful when the row is not stretching its children.
    if (cs.alignItems !== "stretch" && cs.alignItems !== "normal") {
      const centres = kids.map((k) => k.rect.top + k.rect.height / 2);
      const spread = Math.max(...centres) - Math.min(...centres);
      if (spread > 1) {
        add("row-centres-misaligned", spread > 4 ? "medium" : "low", el, {
          spreadPx: Number(spread.toFixed(1)),
          alignItems: cs.alignItems,
          children: kids.map((k) => ({
            selector: sel(k.el),
            centre: Number((k.rect.top + k.rect.height / 2).toFixed(1)),
            height: Number(k.rect.height.toFixed(1)),
          })),
        });
      }
    }

    // Baselines of the first text line in each child that carries text.
    const baselines = [];
    /** Text inside an out-of-flow descendant is not the child's own line. */
    const inFlowWithin = (node, root) => {
      for (let n = node.parentElement; n && n !== root; n = n.parentElement) {
        const p = getComputedStyle(n).position;
        if (p === "absolute" || p === "fixed") return false;
      }
      return true;
    };
    for (const kid of kids) {
      const walker = document.createTreeWalker(kid.el, NodeFilter.SHOW_TEXT);
      let textNode = walker.nextNode();
      while (
        textNode &&
        (!textNode.textContent.trim() || !inFlowWithin(textNode, kid.el))
      )
        textNode = walker.nextNode();
      if (!textNode) continue;
      const range = document.createRange();
      range.selectNodeContents(textNode);
      const r = range.getClientRects()[0];
      if (!r) continue;
      const size = parseFloat(getComputedStyle(textNode.parentElement).fontSize);
      baselines.push({ selector: sel(kid.el), baseline: r.bottom - DESCENT * size, size });
    }
    if (baselines.length >= 2) {
      const values = baselines.map((b) => b.baseline);
      const spread = Math.max(...values) - Math.min(...values);
      const mixedSize = new Set(baselines.map((b) => b.size)).size > 1;
      const limit = mixedSize ? 2 : 1;
      // Children of different font sizes in a centre-aligned row are supposed
      // to differ; only same-size rows and baseline-aligned rows are certain.
      const certain = !mixedSize || cs.alignItems === "baseline";
      if (spread > limit) {
        add("row-baselines-misaligned", certain && spread > 2 ? "medium" : "low", el, {
          spreadPx: Number(spread.toFixed(1)),
          mixedFontSizes: mixedSize,
          alignItems: cs.alignItems,
          children: baselines.map((b) => ({
            selector: b.selector,
            baseline: Number(b.baseline.toFixed(1)),
            fontSize: b.size,
          })),
        });
      }
    }
  }

  // 3b. down a panel: sibling blocks should share a left edge.
  for (const rec of layout()) {
    const { el, cs } = rec;
    const isColumn =
      cs.display === "block" ||
      cs.display === "flow-root" ||
      ((cs.display === "flex" || cs.display === "inline-flex") &&
        cs.flexDirection.startsWith("column"));
    if (!isColumn) continue;
    const kids = Array.from(el.children)
      .map((c) => info.get(c))
      .filter(Boolean)
      .filter(
        (k) =>
          k.cs.position !== "absolute" &&
          k.cs.position !== "fixed" &&
          k.cs.float === "none" &&
          k.cs.display !== "inline" &&
          k.rect.width > 8,
      );
    if (kids.length < 3) continue;
    const lefts = kids.map((k) => Number(k.rect.left.toFixed(1)));
    const counts = new Map();
    for (const l of lefts) counts.set(l, (counts.get(l) ?? 0) + 1);
    const mode = [...counts.entries()].sort((a, b) => b[1] - a[1])[0][0];
    const strays = kids.filter((k) => Math.abs(k.rect.left - mode) > 1);
    // Centred children are a deliberate exception.
    const meaningful = strays.filter((k) => {
      const rightGap = el.getBoundingClientRect().right - k.rect.right;
      const leftGap = k.rect.left - el.getBoundingClientRect().left;
      return Math.abs(rightGap - leftGap) > 2;
    });
    if (meaningful.length && meaningful.length < kids.length) {
      add("panel-left-edges-inconsistent", "medium", el, {
        commonLeft: mode,
        strays: meaningful.slice(0, 6).map((k) => ({
          selector: sel(k.el),
          left: Number(k.rect.left.toFixed(1)),
          deltaPx: Number((k.rect.left - mode).toFixed(1)),
        })),
        siblings: kids.length,
      });
    }
  }

  // 3c. near-miss control heights across the whole state.
  const controls = [];
  for (const el of document.querySelectorAll("button,input,select,textarea")) {
    const rec = info.get(el);
    if (!rec || rec.exempt) continue;
    if (["checkbox", "radio", "range", "color"].includes(el.type)) continue;
    const section = (() => {
      let n = el.parentElement;
      while (n && n !== document.body) {
        if (n.getAttribute("data-testid") || (n.className || "").trim()) return sel(n);
        n = n.parentElement;
      }
      return "body";
    })();
    controls.push({
      kind: el.tagName.toLowerCase(),
      height: Number(rec.rect.height.toFixed(1)),
      fontSize: rec.cs.fontSize,
      selector: sel(el),
      section,
    });
  }
  out.stats.controls = controls.length;
  // Compare like with like: same element kind AND same font size, so a
  // difference in height is a difference in padding/border, not in content.
  const byKind = new Map();
  for (const c of controls) {
    const key = `${c.kind} @${c.fontSize}`;
    if (!byKind.has(key)) byKind.set(key, []);
    byKind.get(key).push(c);
  }
  for (const [kind, list] of byKind) {
    const heights = [...new Set(list.map((c) => c.height))].sort((a, b) => a - b);
    for (let i = 0; i < heights.length; i += 1) {
      for (let j = i + 1; j < heights.length; j += 1) {
        const delta = Number((heights[j] - heights[i]).toFixed(1));
        if (delta >= 0.5 && delta <= 3) {
          const a = list.find((c) => c.height === heights[i]);
          const b = list.find((c) => c.height === heights[j]);
          out.violations.push({
            class: "control-height-near-miss",
            severity: "low",
            selector: `${kind} ${heights[i]}px vs ${heights[j]}px`,
            text: "",
            detail: {
              kind,
              deltaPx: delta,
              a: { selector: a.selector, section: a.section, height: a.height, fontSize: a.fontSize },
              b: { selector: b.selector, section: b.section, height: b.height, fontSize: b.fontSize },
              countA: list.filter((c) => c.height === heights[i]).length,
              countB: list.filter((c) => c.height === heights[j]).length,
            },
          });
        }
      }
    }
  }

  /* ---------- 4. geometry regressions ---------- */
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const scrollsAlong = (el, axis) => {
    let n = el.parentElement;
    while (n && n.nodeType === 1) {
      const c = getComputedStyle(n);
      const ov = axis === "x" ? c.overflowX : c.overflowY;
      if (ov === "auto" || ov === "scroll") return true;
      n = n.parentElement;
    }
    return false;
  };

  const isLayered = (el) => {
    let n = el;
    while (n && n.nodeType === 1 && n !== document.body) {
      const p = getComputedStyle(n).position;
      if (p === "absolute" || p === "fixed") return true;
      n = n.parentElement;
    }
    return false;
  };

  const interactive = Array.from(document.querySelectorAll(INTERACTIVE))
    .map((el) => info.get(el))
    .filter(Boolean)
    .filter((rec) => !rec.exempt);
  out.stats.interactive = interactive.length;

  for (const rec of interactive) {
    const { el, rect } = rec;
    const outLeft = -rect.left;
    const outRight = rect.right - vw;
    const outTop = -rect.top;
    const outBottom = rect.bottom - vh;
    const horizontal = Math.max(outLeft, outRight);
    const vertical = Math.max(outTop, outBottom);
    if (horizontal > 1 && !scrollsAlong(el, "x"))
      add("offscreen-x", "high", el, {
        byPx: Number(horizontal.toFixed(1)),
        rect: { left: Math.round(rect.left), right: Math.round(rect.right) },
        viewportWidth: vw,
      });
    if (vertical > 1 && !scrollsAlong(el, "y"))
      add("offscreen-y", "high", el, {
        byPx: Number(vertical.toFixed(1)),
        rect: { top: Math.round(rect.top), bottom: Math.round(rect.bottom) },
        viewportHeight: vh,
      });
  }

  for (let i = 0; i < interactive.length; i += 1) {
    for (let j = i + 1; j < interactive.length; j += 1) {
      const a = interactive[i];
      const b = interactive[j];
      if (a.el.contains(b.el) || b.el.contains(a.el)) continue;
      const w = Math.min(a.rect.right, b.rect.right) - Math.max(a.rect.left, b.rect.left);
      const h = Math.min(a.rect.bottom, b.rect.bottom) - Math.max(a.rect.top, b.rect.top);
      if (w > 1 && h > 1) {
        // A popup layered on purpose (anything inside an absolutely
        // positioned box) is a lesser finding than two in-flow controls that
        // collide.
        const layered = isLayered(a.el) || isLayered(b.el);
        add("interactive-overlap", layered ? "low" : "high", a.el, {
          other: sel(b.el),
          overlapPx: { w: Number(w.toFixed(1)), h: Number(h.toFixed(1)) },
          layered,
        });
      }
    }
  }

  return out;
}

/* ------------------------------------------------------------------ */
/* Driver                                                              */
/* ------------------------------------------------------------------ */

async function waitForReady(page, timeout) {
  await page.waitForSelector("[data-testid=status]", { timeout });
  await page
    .waitForFunction(
      () => {
        const status = document.querySelector("[data-testid=status]");
        return status && !/compil/i.test(status.textContent ?? "");
      },
      { timeout },
    )
    .catch(() => {});
}

async function enterState(page, state, settle) {
  if (state.selectProfile) {
    // The object tree is docked in both the Model and Sketch desks, so the
    // row is already on screen; nothing has to be opened to reach it.
    await page.locator("[data-testid^=tree-row-profile]").first().click({ timeout: 8_000 });
  }
  await page.getByTestId(`editmode-${state.mode}`).click();
  await page.waitForTimeout(150);
  if (state.tab) {
    const tab = page.getByTestId(`sim-tab-${state.tab}`);
    await tab.waitFor({ state: "visible", timeout: 15_000 });
    await tab.click();
  }
  for (const testid of state.click ?? []) {
    await page.getByTestId(testid).click({ timeout: 8_000 });
    await page.waitForTimeout(120);
  }
  if (state.open) await page.getByTestId(state.open).click({ timeout: 8_000 });
  if (state.expect)
    await page.getByTestId(state.expect).waitFor({ state: "visible", timeout: 8_000 });
  await page.waitForTimeout(settle);
}

/** Put an overlay state back so the next state starts from a clean chrome. */
async function leaveState(page, state) {
  // A state that rearranged the dock cannot be undone by clicking its opener
  // again: the Window menu's own Reset layout is the way back to the desk.
  if (state.reset) {
    try {
      await page.keyboard.press("Escape");
      await page.getByTestId("menu-window").click({ timeout: 5_000 });
      await page.getByTestId("menu-window-reset").click({ timeout: 5_000 });
      await page.waitForTimeout(250);
    } catch {
      /* nothing to restore; the next state re-enters its mode anyway */
    }
    return;
  }
  if (!state.open) return;
  try {
    await page.getByTestId(state.close ?? state.open).click({ timeout: 5_000 });
  } catch {
    /* toggling failed; Escape below is the fallback */
  }
  await page.keyboard.press("Escape");
  await page.waitForTimeout(120);
}

/**
 * `--solve` prelude: populate the states that are empty on a fresh session —
 * solve the first declared study (fills Results and the view controls) and
 * inspect the first declared mesh (fills the mesh report and histogram).
 * Best effort: a server without the FEM extra just leaves those tabs empty.
 */
async function populate(page, warn) {
  await page.getByTestId("editmode-simulate").click();
  // Mesh first, then the solve: inspecting a mesh replaces the panel's
  // current result, so solving last leaves both tabs populated.
  try {
    await page.getByTestId("sim-tab-meshes").click();
    await page.locator("[data-testid^=mesh-inspect-]").first().click({ timeout: 10_000 });
    await page.waitForSelector("[data-testid=mesh-stats]", { timeout: 240_000 });
  } catch (error) {
    warn(`--solve: mesh did not inspect (${String(error).split("\n")[0]})`);
  }
  try {
    await page.getByTestId("sim-tab-studies").click();
    await page.locator("[data-testid^=simulate-run-]").first().click({ timeout: 10_000 });
    await page.waitForSelector("[data-testid=simulate-result-summary]", { timeout: 240_000 });
  } catch (error) {
    warn(`--solve: study did not solve (${String(error).split("\n")[0]})`);
  }
  // Stay in Simulate mode: leaving it clears the published result, so the
  // caller must visit the simulate states before anything else.
  await page.waitForTimeout(300);
}

/** Read the declared type scale from CSS custom properties, if any exist. */
async function readDeclaredScale(page) {
  return page.evaluate(() => {
    const found = {};
    for (const sheet of Array.from(document.styleSheets)) {
      let rules;
      try {
        rules = sheet.cssRules;
      } catch {
        continue;
      }
      for (const rule of Array.from(rules ?? [])) {
        if (!rule.style) continue;
        for (const prop of Array.from(rule.style)) {
          if (!prop.startsWith("--")) continue;
          if (!/(font|text|type|size|scale|leading|weight)/i.test(prop)) continue;
          found[prop] = rule.style.getPropertyValue(prop).trim();
        }
      }
    }
    return found;
  });
}

function tally(samples, keyOf) {
  const map = new Map();
  for (const s of samples) {
    const key = keyOf(s);
    if (!map.has(key)) map.set(key, { key, count: 0, examples: [], states: new Set() });
    const entry = map.get(key);
    entry.count += 1;
    entry.states.add(s.state);
    if (entry.examples.length < 3 && !entry.examples.some((e) => e.selector === s.selector))
      entry.examples.push({ selector: s.selector, text: s.text, state: s.state });
    entry.sample = s;
  }
  return [...map.values()]
    .map((e) => ({ ...e, states: [...e.states].sort() }))
    .sort((a, b) => b.count - a.count);
}

const SEVERITY_RANK = { high: 0, medium: 1, low: 2 };

const CLASS_DOC = {
  "offscreen-x": "Interactive element extends past the viewport horizontally with no scrollable ancestor — unreachable.",
  "offscreen-y": "Interactive element extends past the viewport vertically with no scrollable ancestor — unreachable.",
  "interactive-overlap": "Two interactive elements' hit boxes intersect; the one on top steals the clicks.",
  "text-clipped": "Text is cut off by a clipping ancestor (overflow hidden / fixed height) with no ellipsis.",
  "overflow-x": "scrollWidth exceeds clientWidth with no auto/scroll overflow and no text-overflow: ellipsis.",
  "text-truncated-ellipsis": "Informational: text is actively truncated to an ellipsis, so the user cannot read it in full.",
  "overflow-y-clipped": "scrollHeight exceeds clientHeight inside a non-scrollable clipping box — content is silently lost.",
  "scroll-content-hidden": "A scroll container parks a fifth or more of its content — or a whole control — out of sight at rest.",
  "flex-child-missing-min-width": "A flex child of an overflowing row keeps min-width: auto, so it refuses to shrink.",
  "contrast-text": "Text contrast against its painted background is below the WCAG AA threshold.",
  "contrast-control-border": "A control's border contrast against its surround is below 3:1.",
  "row-centres-misaligned": "Children of a flex row do not share a vertical centre.",
  "row-baselines-misaligned": "First text lines in a flex row's children do not sit on one baseline.",
  "panel-left-edges-inconsistent": "Sibling blocks stacked down a container do not share a left edge.",
  "control-height-near-miss": "Two controls of the same kind differ in height by 0.5–3px — reads as sloppy rather than deliberate.",
};

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const base = args.url ?? `http://127.0.0.1:${args.port}`;
  const states = args.states
    ? STATES.filter((s) => args.states.includes(s.id))
    : STATES;
  const viewports = args.viewports ?? VIEWPORTS;
  const started = Date.now();

  fs.mkdirSync(path.join(args.out, "shots"), { recursive: true });

  const browser = await chromium.launch({ args: GPU_ARGS, headless: !args.headed });
  const pageErrors = [];
  const skipped = [];
  const typeSamples = [];
  const colorSamples = [];
  const rawViolations = [];
  const shots = [];
  let declaredScale = null;

  for (const viewport of viewports) {
    const vpName = `${viewport.width}x${viewport.height}`;
    const page = await browser.newPage({ viewport, deviceScaleFactor: 1 });
    page.on("pageerror", (e) => pageErrors.push(`${vpName}: ${e.message}`));
    await page.goto(`${base}/`, { waitUntil: "load", timeout: args.timeout });
    await waitForReady(page, args.timeout);
    if (!declaredScale) declaredScale = await readDeclaredScale(page);
    // With --solve the simulate states go first: they are the only ones that
    // can see the solved result, which leaving Simulate mode discards.
    const order = args.solve
      ? [...states.filter((s) => s.mode === "simulate"), ...states.filter((s) => s.mode !== "simulate")]
      : states;
    if (args.solve)
      await populate(page, (message) => skipped.push(`${vpName}: ${message}`));

    for (const state of order) {
      try {
        await enterState(page, state, args.settle);
      } catch (error) {
        skipped.push(`${state.id} @ ${vpName}: ${String(error).split("\n")[0]}`);
        process.stderr.write(`  SKIPPED ${state.id} @ ${vpName}\n`);
        await leaveState(page, state).catch(() => {});
        continue;
      }
      const shot = path.join(args.out, "shots", `${state.id}@${vpName}.png`);
      await page.screenshot({ path: shot });
      shots.push(shot);
      const result = await page.evaluate(collectInPage, {
        includeEditor: Boolean(args.includeEditor),
      });
      const stamp = (s) => ({ ...s, state: state.id, viewport: vpName });
      typeSamples.push(...result.type.map(stamp));
      colorSamples.push(...result.color.map(stamp));
      rawViolations.push(...result.violations.map(stamp));
      process.stderr.write(
        `  ${state.id} @ ${vpName}: ${result.stats.elements} elements, ` +
          `${result.violations.length} raw findings\n`,
      );
      await leaveState(page, state);
    }
    await page.close();
  }
  await browser.close();

  /* ---------- aggregate ---------- */
  const typeCensus = tally(typeSamples, (s) => s.key).map((e) => ({
    fontSize: e.sample.fontSize,
    fontWeight: e.sample.fontWeight,
    lineHeight: e.sample.lineHeight,
    fontFamily: e.sample.fontFamily,
    letterSpacing: e.sample.letterSpacing,
    textTransform: e.sample.textTransform,
    count: e.count,
    states: e.states,
    examples: e.examples,
  }));

  const colorCensus = tally(colorSamples, (s) => `${s.role}|${s.value}`).map((e) => ({
    role: e.sample.role,
    value: e.sample.value,
    count: e.count,
    states: e.states,
    examples: e.examples.map((x) => x.selector),
  }));

  const classes = new Map();
  for (const v of rawViolations) {
    if (!classes.has(v.class))
      classes.set(v.class, {
        class: v.class,
        severity: v.severity,
        description: CLASS_DOC[v.class] ?? "",
        count: 0,
        distinctSelectors: new Set(),
        states: new Set(),
        viewports: new Set(),
        instances: [],
      });
    const entry = classes.get(v.class);
    entry.count += 1;
    if (SEVERITY_RANK[v.severity] < SEVERITY_RANK[entry.severity]) entry.severity = v.severity;
    entry.distinctSelectors.add(v.selector);
    entry.states.add(v.state);
    entry.viewports.add(v.viewport);
    entry.instances.push(v);
  }

  const violations = [...classes.values()]
    .map((entry) => {
      // One example per distinct selector, worst first, capped at 3.
      const bySelector = new Map();
      for (const inst of entry.instances)
        if (!bySelector.has(inst.selector)) bySelector.set(inst.selector, inst);
      const examples = [...bySelector.values()]
        .sort((a, b) => SEVERITY_RANK[a.severity] - SEVERITY_RANK[b.severity])
        .slice(0, 3);
      return {
        class: entry.class,
        severity: entry.severity,
        description: entry.description,
        count: entry.count,
        distinctElements: entry.distinctSelectors.size,
        states: [...entry.states].sort(),
        viewports: [...entry.viewports].sort(),
        examples,
        instances: [...bySelector.values()].slice(0, 60),
      };
    })
    .sort(
      (a, b) =>
        SEVERITY_RANK[a.severity] - SEVERITY_RANK[b.severity] ||
        b.distinctElements - a.distinctElements,
    );

  const report = {
    generatedAt: new Date().toISOString(),
    base,
    runtimeSeconds: Number(((Date.now() - started) / 1000).toFixed(1)),
    states: states.map((s) => s.id),
    viewports: viewports.map((v) => `${v.width}x${v.height}`),
    declaredTypeScale: declaredScale,
    pageErrors,
    skippedStates: skipped,
    summary: {
      typeVariants: typeCensus.length,
      colorVariants: colorCensus.length,
      violationClasses: violations.length,
      totalFindings: rawViolations.length,
      bySeverity: {
        high: violations.filter((v) => v.severity === "high").length,
        medium: violations.filter((v) => v.severity === "medium").length,
        low: violations.filter((v) => v.severity === "low").length,
      },
    },
    typeCensus,
    colorCensus,
    violations,
    shots,
  };

  const jsonPath = path.join(args.out, "ui-audit.json");
  fs.writeFileSync(jsonPath, JSON.stringify(report, null, 2));
  fs.writeFileSync(path.join(args.out, "summary.md"), renderSummary(report));
  process.stdout.write(renderSummary(report));
  process.stdout.write(`\nJSON: ${jsonPath}\nShots: ${path.join(args.out, "shots")}\n`);
}

function renderSummary(report) {
  const lines = [];
  const L = (s = "") => lines.push(s);
  L(`# UI audit — ${report.base}`);
  L();
  L(
    `${report.summary.totalFindings} findings in ${report.summary.violationClasses} classes ` +
      `(${report.summary.bySeverity.high} high / ${report.summary.bySeverity.medium} medium / ` +
      `${report.summary.bySeverity.low} low) · ${report.summary.typeVariants} type variants · ` +
      `${report.summary.colorVariants} colour variants · ${report.runtimeSeconds}s`,
  );
  L();
  L(`States: ${report.states.join(", ")}`);
  L(`Viewports: ${report.viewports.join(", ")}`);
  if (report.pageErrors.length) {
    L();
    L(`## Page errors`);
    for (const e of [...new Set(report.pageErrors)]) L(`- ${e}`);
  }
  if (report.skippedStates.length) {
    L();
    L(`## Skipped states (could not be reached)`);
    for (const s of report.skippedStates) L(`- ${s}`);
  }
  L();
  L(`## Type scale census`);
  L();
  L(`| size | weight | line-height | family | uses | example |`);
  L(`| --- | --- | --- | --- | --- | --- |`);
  for (const t of report.typeCensus)
    L(
      `| ${t.fontSize} | ${t.fontWeight} | ${t.lineHeight} | ${t.fontFamily} | ${t.count} | ` +
        `\`${t.examples[0]?.selector ?? ""}\` |`,
    );
  L();
  L(`## Colour census`);
  L();
  L(`| role | value | uses | example |`);
  L(`| --- | --- | --- | --- |`);
  for (const c of report.colorCensus)
    L(`| ${c.role} | ${c.value} | ${c.count} | \`${c.examples[0] ?? ""}\` |`);
  L();
  L(`## Violations`);
  for (const v of report.violations) {
    L();
    L(`### [${v.severity}] ${v.class} — ${v.distinctElements} elements (${v.count} hits)`);
    L(v.description);
    L(`States: ${v.states.join(", ")} · viewports: ${v.viewports.join(", ")}`);
    for (const ex of v.examples) {
      L(`- \`${ex.selector}\`${ex.text ? ` — "${ex.text}"` : ""}`);
      L(`  ${JSON.stringify(ex.detail)}`);
    }
  }
  L();
  return lines.join("\n");
}

main().catch((error) => {
  process.stderr.write(`ui-audit failed: ${error.stack ?? error}\n`);
  process.exit(1);
});
