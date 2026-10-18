import { For } from "solid-js";
import { GRID_CONTROL_HEIGHTS, UNIT } from "../../tokens";
import { Plate, Rule, v, type SectionDef } from "../kit";

/** Each control height, the button class that takes it, and its job. */
const HEIGHTS: Record<string, [cls: string, job: string]> = {
  "control-tag": ["", "non-interactive tags"],
  "control-sm": ["h-sm", "inputs, buttons in panels, list rows"],
  "control-lg": ["", "the menu bar, tool-rail cells"],
  "control-xl": ["h-xl", "the toolbar"],
};

export const space: SectionDef = {
  id: "space",
  title: "Space and size",
  intro: `Every length is a multiple of ${UNIT}px (A6).`,
  body: () => (
    <>
      <Rule
        id="S0"
        why="A distance named by its job is the same wherever the job recurs. View › Show › Layout grid (⇧G) outlines whatever is off the unit."
      >
        Layout uses Y3's three distances and no others. The spacing scale they are cut from is not used by components;
        the test fails on it.
      </Rule>
      <Rule
        id="S2"
        why="Labels, inputs and icons in a row then share a baseline."
        examples={
          <Plate label="control heights, four to ten units">
            <div class="dl-ruler">
              <For each={Object.entries(GRID_CONTROL_HEIGHTS) as [string, number][]}>
                {([token, px]) => (
                  <div>
                    <code>{token}</code>
                    <b>{px}px</b>
                    <div class="dl-row-of">
                      {token === "control-tag" ? (
                        <>
                          <span class="dl-tag">tag</span>
                          <span style={{ color: v("ink-3") }}>{HEIGHTS[token][1]}</span>
                        </>
                      ) : (
                        <button type="button" class={`dl-btn ${HEIGHTS[token][0]}`}>
                          {HEIGHTS[token][1]}
                        </button>
                      )}
                    </div>
                  </div>
                )}
              </For>
            </div>
          </Plate>
        }
      >
        Every control in a row takes one of Y2's heights: a tag, a row control, a bar control, a bar.
      </Rule>
      <Rule
        id="S3"
        why="A 2px line used for emphasis would be read as state."
        examples={
          <>
            <Plate kind="do" label="2px means active">
              <div class="dl-panel">
                <nav>
                  <button type="button" class="active">
                    Studies
                  </button>
                  <button type="button">Meshes</button>
                </nav>
                <div>
                  <input
                    class="dl-input"
                    value="0.75"
                    style={{ outline: "var(--stroke-active) solid var(--ink)", "outline-offset": "calc(var(--hairline) * -1)" }}
                  />
                </div>
              </div>
            </Plate>
            <Plate kind="dont" label="2px for emphasis">
              <div style={{ padding: "var(--inset)", border: "var(--stroke-active) solid var(--rule-heavy)" }}>
                Important settings
              </div>
            </Plate>
          </>
        }
      >
        The hairline is the only border weight. The active line (B1) marks only the active one: the active tab, the
        focused input.
      </Rule>
    </>
  ),
};
