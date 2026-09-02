/**
 * The shared UI layer: the small set of controls every panel is built from.
 *
 * Panels in this app are all the same shape — a list of declaration cards
 * whose fields edit Python literals — so they share a card head, a section
 * head, a numeric field, a triplet row, a toggle and a segmented control.
 * Collecting them here gives the look one home: restyle a primitive and
 * every panel follows, instead of hunting six hand-written copies.
 *
 * Rules for anything added here: no domain imports (no meshes, studies,
 * render settings), props passed as props (never destructured, so Solid's
 * reactivity survives the extraction), and DOM identical to what the panels
 * emitted before — the e2e suite and the screenshot audit both key off it.
 */

export {
  AXIS_LABELS,
  NumberField,
  NumberInput,
  ToggleSwitch,
  VectorField,
  parseNumber,
  type NumberFieldProps,
  type NumberInputProps,
  type ToggleSwitchProps,
  type VectorFieldProps,
} from "./form";
export {
  Card,
  CardHeader,
  CardList,
  SectionHead,
  Stat,
  StatRow,
  type CardHeaderProps,
  type CardListProps,
  type CardProps,
  type SectionHeadProps,
  type StatProps,
  type StatRowProps,
} from "./Card";
export {
  Section,
  SectionHeading,
  type SectionHeadingProps,
  type SectionProps,
} from "./Section";
export { Segmented, type SegmentedOption, type SegmentedProps } from "./Segmented";
export { SPARK_HEIGHT, SPARK_WIDTH, Sparkline, type SparklineProps } from "./Sparkline";
