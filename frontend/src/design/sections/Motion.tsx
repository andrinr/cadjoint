import { DURATIONS } from "../../tokens";
import { DurationDemo, TweenDemo } from "../demos";
import { Plate, Rule, type SectionDef } from "../kit";

export const motion: SectionDef = {
  id: "motion",
  title: "Motion",
  intro: "Derived from A9: three durations, named by what they mean, and one easing family.",
  body: () => (
    <>
      <Rule
        id="M1"
        why={`${DURATIONS["dur-fast"]} ms acknowledges the pointer, ${DURATIONS["dur-base"]} ms shows a state you caused, ${DURATIONS["dur-slow"]} ms shows something that changed because something else did.`}
        examples={
          <Plate kind="try" label="the three durations">
            <DurationDemo />
          </Plate>
        }
      >
        Acknowledgement, state, propagation.
      </Rule>
      <Rule
        id="M2"
        why="Simultaneity teaches that panel and viewport are two views; the stagger teaches that the panel edits the code and the code makes the geometry (try it in A2)."
      >
        Causal pairs are sequenced, never simultaneous: the changed source span lights first, the geometry follows.
      </Rule>
      <Rule
        id="M3"
        why="A tweened number shows values that were never true."
        examples={
          <>
            <Plate kind="do" label="digits replace: press New result">
              <TweenDemo tween={false} />
            </Plate>
            <Plate kind="dont" label="digits roll">
              <TweenDemo tween />
            </Plate>
          </>
        }
      >
        Never animated: digits, the camera under direct manipulation, layout when data arrives. Replaying a recorded
        optimisation is not animation: every frame is a state the optimiser held.
      </Rule>
      <Rule id="M4" why="Causality is information, so it survives in a form that does not move.">
        Under <code>prefers-reduced-motion</code>, acknowledgement and state become instant and propagation becomes a
        two-frame highlight.
      </Rule>
    </>
  ),
};
