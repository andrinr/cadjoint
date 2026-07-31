/**
 * CodeMirror editor pane.
 *
 * Besides editing, this is the other half of viewer/code parity: selecting a
 * sketch vertex in the 3D view highlights and scrolls to the exact characters
 * that define it.
 */

import { defaultKeymap, history, historyKeymap, indentWithTab } from "@codemirror/commands";
import { python } from "@codemirror/lang-python";
import { HighlightStyle, syntaxHighlighting } from "@codemirror/language";
import { tags } from "@lezer/highlight";
import { EditorState, StateEffect, StateField, type Extension } from "@codemirror/state";
import {
  Decoration,
  EditorView,
  keymap,
  lineNumbers,
  type DecorationSet,
} from "@codemirror/view";
import { createEffect, onCleanup, onMount } from "solid-js";
import { consoleText, setDirty, setSource, source } from "../state";

/** Highlight the character span of the selected sketch vertex. */
const setHighlight = StateEffect.define<{ from: number; to: number } | null>();

const highlightMark = Decoration.mark({ class: "cm-vertex-highlight" });

const highlightField = StateField.define<DecorationSet>({
  create: () => Decoration.none,
  update(value, transaction) {
    let next = value.map(transaction.changes);
    for (const effect of transaction.effects) {
      if (effect.is(setHighlight)) {
        next = effect.value
          ? Decoration.set([highlightMark.range(effect.value.from, effect.value.to)])
          : Decoration.none;
      }
    }
    return next;
  },
  provide: (field) => EditorView.decorations.from(field),
});

/**
 * Syntax colours for the editor.
 *
 * CodeMirror ships no highlighting unless a style is installed — the language
 * package only supplies the parser — so this defines one on the app's palette
 * rather than pulling in the default light-leaning theme.
 */
const highlightStyle = HighlightStyle.define([
  { tag: tags.comment, color: "#6d746a", fontStyle: "italic" },
  { tag: tags.keyword, color: "#d9ff57" },
  { tag: [tags.controlKeyword, tags.moduleKeyword], color: "#c7f04a" },
  { tag: [tags.string, tags.special(tags.string)], color: "#ffb37a" },
  { tag: [tags.number, tags.bool, tags.null], color: "#8fd8ff" },
  { tag: [tags.className, tags.typeName, tags.namespace], color: "#ff8167" },
  { tag: tags.function(tags.variableName), color: "#e7e6df" },
  { tag: tags.definition(tags.variableName), color: "#f2efe6" },
  { tag: tags.propertyName, color: "#cfe8a0" },
  { tag: [tags.operator, tags.punctuation, tags.separator], color: "#9aa294" },
  { tag: tags.self, color: "#ff8167", fontStyle: "italic" },
]);

const theme = EditorView.theme(
  {
    "&": { height: "100%", fontSize: "13px", backgroundColor: "var(--panel)" },
    ".cm-scroller": { fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace" },
    ".cm-content": { caretColor: "var(--lime)" },
    ".cm-gutters": { backgroundColor: "var(--panel)", border: "none", color: "#5d615a" },
    ".cm-activeLine": { backgroundColor: "rgba(255,255,255,0.03)" },
    ".cm-activeLineGutter": { backgroundColor: "transparent" },
    ".cm-vertex-highlight": {
      backgroundColor: "rgba(217,255,87,0.28)",
      outline: "1px solid rgba(217,255,87,0.75)",
      borderRadius: "3px",
    },
    "&.cm-focused": { outline: "none" },
  },
  { dark: true },
);

export interface EditorPaneProps {
  /** Character span to highlight, or null to clear. */
  highlight: { from: number; to: number } | null;
  onRun: () => void;
}

export function EditorPane(props: EditorPaneProps) {
  let host!: HTMLDivElement;
  let view: EditorView | undefined;

  onMount(() => {
    const extensions: Extension[] = [
      lineNumbers(),
      history(),
      python(),
      syntaxHighlighting(highlightStyle),
      highlightField,
      theme,
      keymap.of([
        {
          key: "Mod-Enter",
          preventDefault: true,
          run: () => {
            props.onRun();
            return true;
          },
        },
        indentWithTab,
        ...defaultKeymap,
        ...historyKeymap,
      ]),
      EditorView.updateListener.of((update) => {
        if (!update.docChanged) return;
        const text = update.state.doc.toString();
        // Guard against feeding our own programmatic updates back as edits.
        if (text !== source()) {
          setSource(text);
          setDirty(true);
        }
      }),
    ];

    view = new EditorView({
      state: EditorState.create({ doc: source(), extensions }),
      parent: host,
    });

    onCleanup(() => view?.destroy());
  });

  // Adopt source changes that came from elsewhere (session start, /patch).
  createEffect(() => {
    const text = source();
    if (!view || view.state.doc.toString() === text) return;
    view.dispatch({
      changes: { from: 0, to: view.state.doc.length, insert: text },
    });
  });

  createEffect(() => {
    const span = props.highlight;
    if (!view) return;
    const limit = view.state.doc.length;
    const valid = span && span.from >= 0 && span.to <= limit && span.from < span.to;
    view.dispatch({
      effects: [
        setHighlight.of(valid ? span : null),
        ...(valid ? [EditorView.scrollIntoView(span.from, { y: "center" })] : []),
      ],
    });
  });

  return (
    <section class="pane editor-pane">
      <header class="pane-head">
        <span class="pane-title">scene.py</span>
        <span class="pane-hint">Ctrl/⌘ + Enter to run</span>
      </header>
      <div class="editor-host" ref={host} data-testid="editor" />
      {consoleText() && <pre class="console">{consoleText()}</pre>}
    </section>
  );
}
