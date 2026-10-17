/**
 * The design language, as a page of the playground: the shell, its index,
 * and the order of the sections. Each section lives in sections/ and is
 * drawn with the tokens and primitives its rules define.
 */

import { For, createSignal, onCleanup, onMount } from "solid-js";
import { Section, type SectionDef } from "./kit";
import { Variations } from "./Variations";
import { blocks } from "./sections/Blocks";
import { behaviour } from "./sections/Behaviour";
import { colour } from "./sections/Colour";
import { components } from "./sections/Components";
import { features } from "./sections/Features";
import { icons } from "./sections/Icons";
import { language } from "./sections/Language";
import { layout } from "./sections/Layout";
import { motion } from "./sections/Motion";
import { navigation } from "./sections/Navigation";
import { primitives } from "./sections/Primitives";
import { principles } from "./sections/Principles";
import { space } from "./sections/Space";
import { system } from "./sections/System";
import { type } from "./sections/Type";
import { viewport } from "./sections/Viewport";

/** The page, in reading order: the system up, then behaviour, then the reasons. */
const SECTIONS: SectionDef[] = [
  system,
  blocks,
  primitives,
  layout,
  components,
  colour,
  type,
  language,
  space,
  icons,
  motion,
  viewport,
  navigation,
  features,
  behaviour,
  principles,
];

function Nav() {
  const [current, setCurrent] = createSignal(SECTIONS[0].id);
  onMount(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((entry) => entry.isIntersecting);
        if (visible.length) setCurrent(visible[0].target.id);
      },
      { rootMargin: "-40% 0px -55% 0px" },
    );
    for (const section of SECTIONS) {
      const node = document.getElementById(section.id);
      if (node) observer.observe(node);
    }
    onCleanup(() => observer.disconnect());
  });
  return (
    <nav class="dl-nav" aria-label="Sections">
      <ol>
        <For each={SECTIONS}>
          {(section, index) => (
            <li>
              <a href={`#${section.id}`} class={current() === section.id ? "current" : ""}>
                <small>{index() + 1}</small>
                {section.title}
              </a>
            </li>
          )}
        </For>
      </ol>
    </nav>
  );
}

export function DesignPage() {
  return (
    <>
      <header class="dl-top">
        <b>CJ</b>
        <strong>cadjoint</strong>
        <span>Design language</span>
        <span class="dl-top-end">
          <Variations />
          <a class="dl-btn quiet h-sm" href="/">
            Playground
          </a>
        </span>
      </header>
      <div class="dl">
        <Nav />
        <main class="dl-main">
          <header>
            <h1>Design language</h1>
            <p>
              The rules the playground is built by, from the unit up to the principles behind them, drawn with the app's
              own tokens. A screen that contradicts a rule is a bug in the screen; a rule that traces to no principle is a
              bug in this page. The measurements behind the choices are in <code>research/design-language.md</code>.
            </p>
          </header>
          <For each={SECTIONS}>
            {(section, index) => (
              <Section id={section.id} num={String(index() + 1)} title={section.title} intro={section.intro}>
                {section.body()}
              </Section>
            )}
          </For>
        </main>
      </div>
    </>
  );
}
