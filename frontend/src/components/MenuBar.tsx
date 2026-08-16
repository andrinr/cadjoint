/**
 * Slim application menu bar: File / Edit / Window / Help.
 *
 * Menus follow the native pattern — click (or ArrowDown) opens, arrows move,
 * Escape or a click elsewhere closes. Everything acts on the shared state
 * signals; file operations go through the server's `scenes` workspace.
 */

import { For, Show, createSignal, onCleanup } from "solid-js";
import * as api from "../api";
import { sanitizeSceneName } from "../scenes";
import { SHORTCUT_GROUPS } from "../shortcuts";
import {
  dirty,
  editingMode,
  panels,
  sceneName,
  setEditingMode,
  setPanelVisible,
  setSceneName,
  setStatus,
  source,
  type PanelVisibility,
} from "../state";
import { EDITING_MODES } from "../editingMode";

export interface MenuBarProps {
  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => void;
  onRedo: () => void;
  /** Reset to the starter example (App confirms when the buffer is dirty). */
  onNew: () => void;
  /** Adopt a loaded scene file as the current program and recompile. */
  onAdoptScene: (name: string, source: string) => void;
}

type MenuId = "file" | "edit" | "window" | "help";
type DialogId = "open" | "saveAs" | "help";

const PANEL_ITEMS: { key: keyof PanelVisibility; label: string }[] = [
  { key: "editor", label: "Editor" },
  { key: "objectTree", label: "Object tree" },
  { key: "materials", label: "Materials" },
  { key: "sketch", label: "Sketch panel" },
];

const REPO_README_URL = "https://github.com/andrinr/jaxcad#readme";

export function MenuBar(props: MenuBarProps) {
  const [openMenu, setOpenMenu] = createSignal<MenuId | null>(null);
  const [dialog, setDialog] = createSignal<DialogId | null>(null);
  /** null while the listing request is in flight. */
  const [sceneFiles, setSceneFiles] = createSignal<string[] | null>(null);
  const [saveAsName, setSaveAsName] = createSignal("");
  const [saveAsError, setSaveAsError] = createSignal("");

  const closeOnOutside = (event: MouseEvent) => {
    if (!(event.target as HTMLElement).closest(".menu-bar")) setOpenMenu(null);
  };
  document.addEventListener("click", closeOnOutside);
  onCleanup(() => document.removeEventListener("click", closeOnOutside));

  const toggleMenu = (id: MenuId) => setOpenMenu(openMenu() === id ? null : id);
  /** An open menubar behaves like hover navigation between its menus. */
  const glideTo = (id: MenuId) => {
    if (openMenu() !== null && openMenu() !== id) setOpenMenu(id);
  };

  const confirmDiscard = () =>
    !dirty() ||
    window.confirm("Discard unsaved changes to the current scene?");

  const showOpenDialog = () => {
    setOpenMenu(null);
    setDialog("open");
    setSceneFiles(null);
    void api
      .listScenes()
      .then((result) => setSceneFiles(result.files ?? []))
      .catch(() => {
        setSceneFiles([]);
        setStatus({ kind: "error", text: "Could not list saved scenes." });
      });
  };

  const openScene = async (name: string) => {
    if (!confirmDiscard()) return;
    try {
      const result = await api.loadScene(name);
      if (!result.ok || typeof result.source !== "string") {
        setStatus({ kind: "error", text: result.error ?? `Could not open ${name}.` });
        return;
      }
      setDialog(null);
      props.onAdoptScene(name, result.source);
    } catch (error) {
      setStatus({
        kind: "error",
        text: error instanceof Error ? error.message : String(error),
      });
    }
  };

  const saveAs = async (rawName: string): Promise<void> => {
    const name = sanitizeSceneName(rawName);
    if (name === null) {
      setSaveAsError("Use a plain file name such as bracket.py — no folders.");
      return;
    }
    try {
      const result = await api.saveScene(name, source());
      if (!result.ok) {
        const message = result.error ?? `Could not save ${name}.`;
        setSaveAsError(message);
        setStatus({ kind: "error", text: message });
        return;
      }
      setSceneName(name);
      setDialog(null);
      setStatus({ kind: "ready", text: `Saved ${name}` });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setSaveAsError(message);
      setStatus({ kind: "error", text: message });
    }
  };

  const save = () => {
    setOpenMenu(null);
    const name = sceneName();
    if (name === null) {
      showSaveAsDialog();
      return;
    }
    void saveAs(name);
  };

  const showSaveAsDialog = () => {
    setOpenMenu(null);
    setSaveAsName(sceneName() ?? "scene.py");
    setSaveAsError("");
    setDialog("saveAs");
  };

  const download = () => {
    setOpenMenu(null);
    const blob = new Blob([source()], { type: "text/x-python" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = sceneName() ?? "scene.py";
    anchor.click();
    URL.revokeObjectURL(url);
  };

  /** Arrow-key navigation between the items of an open dropdown. */
  const onMenuKeyDown = (event: KeyboardEvent) => {
    const menu = (event.currentTarget as HTMLElement).closest(".menu");
    if (!menu) return;
    const items = [
      ...menu.querySelectorAll<HTMLElement>(
        ".menu-dropdown button:not(:disabled), .menu-dropdown a",
      ),
    ];
    const index = items.indexOf(document.activeElement as HTMLElement);
    if (event.key === "ArrowDown") {
      event.preventDefault();
      (items[(index + 1) % items.length] ?? items[0])?.focus();
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      (items[(index - 1 + items.length) % items.length] ?? items.at(-1))?.focus();
    } else if (event.key === "Escape") {
      event.preventDefault();
      setOpenMenu(null);
      menu.querySelector<HTMLElement>(".menu-title")?.focus();
    } else if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
      const order: MenuId[] = ["file", "edit", "window", "help"];
      const active = openMenu();
      if (!active) return;
      event.preventDefault();
      const step = event.key === "ArrowRight" ? 1 : -1;
      const next = order[(order.indexOf(active) + step + order.length) % order.length];
      setOpenMenu(next);
      document
        .querySelector<HTMLElement>(`[data-testid=menu-${next}]`)
        ?.focus();
    }
  };

  const menuTitle = (id: MenuId, label: string) => (
    <button
      type="button"
      class="menu-title"
      classList={{ active: openMenu() === id }}
      aria-haspopup="menu"
      aria-expanded={openMenu() === id}
      onClick={() => toggleMenu(id)}
      onPointerEnter={() => glideTo(id)}
      onKeyDown={(event) => {
        if (event.key === "ArrowDown" && openMenu() !== id) {
          event.preventDefault();
          setOpenMenu(id);
        } else {
          onMenuKeyDown(event);
        }
      }}
      data-testid={`menu-${id}`}
    >
      {label}
    </button>
  );

  const item = (
    label: string,
    action: () => void,
    options: { shortcut?: string; disabled?: boolean; testid?: string } = {},
  ) => (
    <button
      type="button"
      role="menuitem"
      disabled={options.disabled}
      onClick={action}
      onKeyDown={onMenuKeyDown}
      data-testid={options.testid}
    >
      <span>{label}</span>
      <Show when={options.shortcut}>
        <small>{options.shortcut}</small>
      </Show>
    </button>
  );

  return (
    <>
      <nav class="menu-bar" aria-label="Application menu" data-testid="menu-bar">
        <div class="menu">
          {menuTitle("file", "File")}
          <Show when={openMenu() === "file"}>
            <div class="menu-dropdown" role="menu">
              {item("New", () => (setOpenMenu(null), props.onNew()), {
                testid: "menu-file-new",
              })}
              {item("Open…", showOpenDialog, { testid: "menu-file-open" })}
              <hr />
              {item("Save", save, { testid: "menu-file-save" })}
              {item("Save As…", showSaveAsDialog, { testid: "menu-file-save-as" })}
              <hr />
              {item("Download scene.py", download, { testid: "menu-file-download" })}
            </div>
          </Show>
        </div>

        <div class="menu">
          {menuTitle("edit", "Edit")}
          <Show when={openMenu() === "edit"}>
            <div class="menu-dropdown" role="menu">
              {item("Undo", () => (setOpenMenu(null), props.onUndo()), {
                shortcut: "⌘Z",
                disabled: !props.canUndo,
                testid: "menu-edit-undo",
              })}
              {item("Redo", () => (setOpenMenu(null), props.onRedo()), {
                shortcut: "⇧⌘Z",
                disabled: !props.canRedo,
                testid: "menu-edit-redo",
              })}
            </div>
          </Show>
        </div>

        <div class="menu">
          {menuTitle("window", "Window")}
          <Show when={openMenu() === "window"}>
            <div class="menu-dropdown" role="menu">
              <For each={EDITING_MODES}>
                {(mode) => (
                  <button
                    type="button"
                    role="menuitemradio"
                    aria-checked={editingMode() === mode}
                    onClick={() => {
                      setEditingMode(mode);
                      setOpenMenu(null);
                    }}
                    onKeyDown={onMenuKeyDown}
                    data-testid={`menu-window-mode-${mode}`}
                  >
                    <i class="menu-check">{editingMode() === mode ? "•" : ""}</i>
                    <span>
                      {mode.charAt(0).toUpperCase() + mode.slice(1)} mode
                    </span>
                    <Show when={mode === "model"}>
                      <small>Esc</small>
                    </Show>
                  </button>
                )}
              </For>
              <hr />
              <For each={PANEL_ITEMS}>
                {(panel) => (
                  <button
                    type="button"
                    role="menuitemcheckbox"
                    aria-checked={panels()[panel.key]}
                    onClick={() => setPanelVisible(panel.key, !panels()[panel.key])}
                    onKeyDown={onMenuKeyDown}
                    data-testid={`menu-window-${panel.key}`}
                  >
                    <i class="menu-check">{panels()[panel.key] ? "✓" : ""}</i>
                    <span>{panel.label}</span>
                  </button>
                )}
              </For>
            </div>
          </Show>
        </div>

        <div class="menu">
          {menuTitle("help", "Help")}
          <Show when={openMenu() === "help"}>
            <div class="menu-dropdown" role="menu">
              {item(
                "Keyboard shortcuts",
                () => (setOpenMenu(null), setDialog("help")),
                { testid: "menu-help-shortcuts" },
              )}
              <a
                href={REPO_README_URL}
                target="_blank"
                rel="noreferrer"
                role="menuitem"
                onKeyDown={onMenuKeyDown}
                data-testid="menu-help-readme"
              >
                <span>JAXCAD README</span>
                <small>↗</small>
              </a>
            </div>
          </Show>
        </div>

        <span class="menu-scene" data-testid="menu-scene-name">
          {sceneName() ?? "unsaved scene"}
          {dirty() ? " •" : ""}
        </span>
      </nav>

      <Show when={dialog() === "open"}>
        <div class="dialog-backdrop" onClick={() => setDialog(null)}>
          <div
            class="dialog dialog-compact"
            role="dialog"
            aria-label="Open a saved scene"
            onClick={(event) => event.stopPropagation()}
          >
            <header>
              <span>Open a saved scene</span>
              <button type="button" onClick={() => setDialog(null)}>
                Close
              </button>
            </header>
            <div class="dialog-body" data-testid="open-scene-list">
              <Show
                when={sceneFiles() !== null}
                fallback={<p class="dialog-note">Listing scenes…</p>}
              >
                <Show
                  when={sceneFiles()!.length > 0}
                  fallback={
                    <p class="dialog-note">
                      No saved scenes yet. Use File → Save As… to create
                      <code> scenes/*.py</code> next to the server.
                    </p>
                  }
                >
                  <ul class="scene-list">
                    <For each={sceneFiles()!}>
                      {(name) => (
                        <li>
                          <button
                            type="button"
                            onClick={() => void openScene(name)}
                            data-testid={`open-scene-${name}`}
                          >
                            {name}
                          </button>
                        </li>
                      )}
                    </For>
                  </ul>
                </Show>
              </Show>
            </div>
          </div>
        </div>
      </Show>

      <Show when={dialog() === "saveAs"}>
        <div class="dialog-backdrop" onClick={() => setDialog(null)}>
          <div
            class="dialog dialog-compact"
            role="dialog"
            aria-label="Save the scene as"
            onClick={(event) => event.stopPropagation()}
          >
            <header>
              <span>Save scene as</span>
              <button type="button" onClick={() => setDialog(null)}>
                Close
              </button>
            </header>
            <form
              class="dialog-body save-as-form"
              onSubmit={(event) => {
                event.preventDefault();
                void saveAs(saveAsName());
              }}
            >
              <label>
                File name
                <input
                  type="text"
                  value={saveAsName()}
                  onInput={(event) => {
                    setSaveAsName(event.currentTarget.value);
                    setSaveAsError("");
                  }}
                  data-testid="save-as-name"
                />
              </label>
              <Show when={saveAsError()}>
                <p class="dialog-error" data-testid="save-as-error">
                  {saveAsError()}
                </p>
              </Show>
              <p class="dialog-note">
                Written to <code>scenes/</code> in the playground server's
                working directory.
              </p>
              <button type="submit" class="primary" data-testid="save-as-confirm">
                Save
              </button>
            </form>
          </div>
        </div>
      </Show>

      <Show when={dialog() === "help"}>
        <div class="dialog-backdrop" onClick={() => setDialog(null)}>
          <div
            class="dialog dialog-compact"
            role="dialog"
            aria-label="Keyboard shortcuts"
            onClick={(event) => event.stopPropagation()}
          >
            <header>
              <span>Keyboard shortcuts</span>
              <button type="button" onClick={() => setDialog(null)}>
                Close
              </button>
            </header>
            <div class="dialog-body shortcut-groups" data-testid="shortcut-dialog">
              <For each={SHORTCUT_GROUPS}>
                {(group) => (
                  <section>
                    <h4>{group.title}</h4>
                    <ul>
                      <For each={group.items}>
                        {(shortcut) => (
                          <li>
                            <span>{shortcut.action}</span>
                            <kbd>{shortcut.keys}</kbd>
                          </li>
                        )}
                      </For>
                    </ul>
                  </section>
                )}
              </For>
            </div>
          </div>
        </div>
      </Show>
    </>
  );
}
