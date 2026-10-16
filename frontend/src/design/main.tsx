import { render } from "solid-js/web";
import "../styles.css";
import "./design.css";
import { DesignPage } from "./DesignPage";

const root = document.getElementById("root");
if (!root) throw new Error("Missing #root element");
render(() => <DesignPage />, root);
