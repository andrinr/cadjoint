/**
 * The public demo (`static.html`): the shipped scenes, drawn, edited and
 * exported in the browser with no Python server, so the build is plain files
 * (`npm run build:static`). See `README.md` here.
 */
import { render } from "solid-js/web";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import "@fontsource/ibm-plex-mono/600.css";
import "@fontsource/ibm-plex-mono/400-italic.css";
import "../tokens.css";
import "../styles.css";
import { DemoApp } from "./DemoApp";

const root = document.getElementById("root");
if (!root) throw new Error("Missing #root element");
render(() => <DemoApp />, root);
