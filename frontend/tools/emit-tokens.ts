/**
 * Write the generated token files from src/tokens.ts, the one source:
 *
 *   src/tokens.css            the stylesheet's :root custom properties
 *   ../docs/assets/tokens.scss the same values for the Quarto docs theme
 *
 * Run with `npm run tokens` (Node strips the types). test/tokens.test.ts
 * fails when either file is out of date.
 */
import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { tokensCss, tokensScss } from "../src/tokens.ts";

const here = dirname(fileURLToPath(import.meta.url));
writeFileSync(join(here, "../src/tokens.css"), tokensCss());
writeFileSync(join(here, "../../docs/assets/tokens.scss"), tokensScss());
console.log("wrote src/tokens.css and docs/assets/tokens.scss");
