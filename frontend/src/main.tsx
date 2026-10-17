import { render } from "solid-js/web";
import { App } from "./App";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import "@fontsource/ibm-plex-mono/600.css";
import "@fontsource/ibm-plex-mono/400-italic.css";
import "./tokens.css";
import "./styles.css";

const root = document.getElementById("root");
if (!root) throw new Error("Missing #root element");
render(() => <App />, root);
