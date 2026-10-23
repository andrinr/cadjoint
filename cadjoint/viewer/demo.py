"""Write the static demo's bundle from the shipped scenes.

``python -m cadjoint.viewer.demo`` runs every scene in ``scenes/`` through the
compile worker once and keeps what the browser needs to draw, edit and export
it without Python (:class:`~cadjoint.viewer.schema.demo.DemoScene`), plus the
public tier's capabilities, which the page shows beside the features the
private tier would make exact.  ``--check`` compares instead of writing, for
a test that keeps the committed bundle the build of the committed scenes.

A scene whose ``scene`` has no ``cadjoint-scene`` graph cannot be drawn by the
browser and is left out, with a note.

Every run of a scene gets a fresh process, as it does in the playground,
whose worker is one process per compile.  Some names depend on what the
process ran before: a primitive declared without a name is numbered by a
per-process counter (``cylinder``, ``cylinder_2``), and its parameters are
named after it, so a second run in the same process names them differently.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import multiprocessing
import re
import sys
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from cadjoint import extract_parameters
from cadjoint.plugins import PluginRegistry
from cadjoint.tier import label, message, status
from cadjoint.viewer.schema.api import CapabilitiesResponse
from cadjoint.viewer.schema.demo import DEMO_FORMAT, DEMO_VERSION, DemoParameter, DemoScene
from cadjoint.viewer.worker.main import HANDLERS
from cadjoint.viewer.worker.scene import _execute_scene

ROOT = Path(__file__).resolve().parents[2]
SCENES = ROOT / "scenes"
BUNDLE = ROOT / "frontend" / "src" / "static" / "bundle"
TIER_FILE = "public-tier.json"


def _docstring(source: str) -> tuple[str, str]:
    """The module docstring's first line, and its first paragraph after that."""
    text = ast.get_docstring(ast.parse(source)) or ""
    head, _, rest = text.partition("\n")
    paragraph = rest.strip().split("\n\n")[0]

    def plain(text: str) -> str:
        """Prose without reST literals and emphasis."""
        return re.sub(r"``(.+?)``|\*(.+?)\*", lambda m: m[1] or m[2], " ".join(text.split()))

    return plain(head).rstrip("."), plain(paragraph)


def _points(construction: Iterable[dict[str, Any]]) -> list[list[float]]:
    points: list[list[float]] = []
    for node in construction:
        points.extend(point for edge in node["edges"] for point in edge)
        points.extend(vertex["world"] for vertex in node["vertices"])
    return points


def _extent(construction: list[dict[str, Any]]) -> float:
    """The diagonal of the construction's bounding box, or 1 for an empty one."""
    points = _points(construction)
    if not points:
        return 1.0
    return math.dist([min(c) for c in zip(*points)], [max(c) for c in zip(*points)]) or 1.0


def _step(span: float) -> float:
    """A power of ten near a hundredth of ``span``: what a slider moves by."""
    return 10.0 ** math.floor(math.log10(span / 100.0)) if span > 0 else 0.01


def _snap(value: float, step: float, up: bool) -> float:
    # Rounded first: the values are float32, and 2 x 0.2f must not snap up to 0.401.
    snapped = (math.ceil if up else math.floor)(round(value / step, 4)) * step
    return round(snapped, 12)


def _range(value: list[float], extent: float) -> tuple[list[float], list[float], float]:
    """A slider range around a shipped value that declared no bounds.

    A positive scalar is a length (a thickness, a radius): it may shrink to a
    quarter and grow to twice.  Anything else is a coordinate, and moves half
    the scene's size either way.
    """
    if len(value) == 1 and value[0] > 0:
        lower, upper = [value[0] / 4], [value[0] * 2]
    else:
        lower = [v - extent / 2 for v in value]
        upper = [v + extent / 2 for v in value]
    step = _step(max(u - lo for lo, u in zip(lower, upper)))
    lower = [min(_snap(lo, step, False), v) for lo, v in zip(lower, value)]
    upper = [max(_snap(u, step, True), v) for u, v in zip(upper, value)]
    return lower, upper, step


def _constraint_text(constraint: Any) -> str:
    names = [p.name or f"{float(p.value):g}" for p in constraint.get_parameters()]
    return f"{type(constraint).__name__}({', '.join(names)})"


def _owners(construction: list[dict[str, Any]]) -> dict[str, str]:
    """Parameter name → the construction object whose handle moves it."""
    owners: dict[str, str] = {}
    for node in construction:
        for vertex in node["vertices"]:
            if vertex.get("binding"):
                owners.setdefault(vertex["binding"]["name"], node["id"])
        for bindings in ((node.get("transform") or {}).get("bindings") or {}).values():
            for binding in bindings:
                owners.setdefault(binding["name"], node["id"])
    return owners


def _parameters(
    program: dict[str, Any], declared: dict[str, dict[str, Any]], construction: list[dict[str, Any]]
) -> list[DemoParameter]:
    """Every free slot of the shader, with its range and what holds it."""
    extent = _extent(construction)
    owners = _owners(construction)
    entries: list[DemoParameter] = []
    for slot in program["parameters"]:
        if not slot["free"] or any(v is None for v in slot["value"]):
            continue
        value = [float(v) for v in slot["value"]]
        facts = declared.get(slot["name"], {})
        bounds = facts.get("bounds")
        if bounds is not None:
            lower = [float(bounds[0])] * len(value)
            upper = [float(bounds[1])] * len(value)
            step = _step(max(u - lo for lo, u in zip(lower, upper)))
        else:
            lower, upper, step = _range(value, extent)
        entries.append(
            DemoParameter(
                name=slot["name"],
                components=slot["components"],
                value=value,
                lower=lower,
                upper=upper,
                step=step,
                bounded=bounds is not None,
                owner=owners.get(slot["name"]),
                drivenBy=facts.get("drivenBy"),
            )
        )
    return entries


def _compiled(source: str) -> dict[str, Any]:
    """The compile payload of ``source``: run in a fresh process."""
    return HANDLERS["compile"]({"source": source})


def _declared(source: str) -> dict[str, dict[str, Any]]:
    """Per free parameter, its declared bounds and the constraints holding it.

    The compile payload names the parameters; these facts are on the
    Parameter objects, which only a run of the scene has.  Run in a fresh
    process.
    """
    _, _, metadata = extract_parameters(_execute_scene(source)["scene"])
    return {
        name: {
            "bounds": [float(b) for b in parameter.bounds] if parameter.bounds else None,
            "drivenBy": "; ".join(_constraint_text(c) for c in parameter.get_constraints()) or None,
        }
        for name, parameter in metadata.items()
    }


def _record(
    path: Path, payload: dict[str, Any], declared: dict[str, dict[str, Any]]
) -> DemoScene | None:
    """The demo's record of one scene, from its two runs."""
    if not payload.get("ok"):
        raise RuntimeError(f"{path.name} did not compile:\n{payload.get('error')}")
    if payload["scene_graph"] is None or payload["program"] is None:
        print(f"note: {path.name} has no scene graph; the demo leaves it out", file=sys.stderr)
        return None
    source = path.read_text()
    title, description = _docstring(source)
    # Faces are what the sketch-on-face tool picks from, and the demo has no
    # such tool; they are most of a construction payload's size.
    construction = [{**node, "faces": []} for node in payload["construction"]]
    return DemoScene(
        format=DEMO_FORMAT,
        version=DEMO_VERSION,
        name=path.stem,
        title=title,
        description=description,
        scene_graph=payload["scene_graph"],
        parameters=_parameters(payload["program"], declared, construction),
        construction=construction,
        elements=payload["elements"],
        scene_tree=payload["scene_tree"],
        relations=payload["relations"],
    )


def demo_scenes(paths: list[Path], workers: int = 4) -> dict[Path, DemoScene | None]:
    """Each scene's record, every run of a scene in a process of its own."""
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        max_tasks_per_child=1,
    ) as pool:
        runs = {
            path: (
                pool.submit(_compiled, path.read_text()),
                pool.submit(_declared, path.read_text()),
            )
            for path in paths
        }
        return {
            path: _record(path, compiled.result(), declared.result())
            for path, (compiled, declared) in runs.items()
        }


def demo_scene(path: Path) -> DemoScene | None:
    """The demo's record of one scene file, or ``None`` if the browser cannot draw it."""
    return demo_scenes([path], workers=2)[path]


def public_tier() -> CapabilitiesResponse:
    """What :mod:`cadjoint.tier` reports with nothing filling the private kinds."""
    report = status(PluginRegistry()).as_dict()
    for kind, entry in report["kinds"].items():
        entry["label"] = label(kind, False)
        entry["message"] = message(kind, entry["reason"])
    return CapabilitiesResponse(ok=True, **report)


def _dump(model: Any) -> str:
    return json.dumps(model.model_dump(mode="json"), indent=None, separators=(",", ":")) + "\n"


def bundle_files(scenes: Path = SCENES) -> dict[str, str]:
    """File name → contents of the whole bundle."""
    files = {TIER_FILE: _dump(public_tier())}
    for path, scene in demo_scenes(sorted(scenes.glob("*.py"))).items():
        if scene is not None:
            files[f"{path.stem}.demo.json"] = _dump(scene)
    return files


def same(a: Any, b: Any) -> bool:
    """Whether two bundle documents agree: in shape exactly, in floats to float32 round-off.

    A scene's numbers come out of JAX, and XLA may fuse a multiply-add
    differently from one process to the next (a warm compilation cache, a
    test worker that compiled the program first), which moves the last bits
    of a float32 and nothing else.
    """
    if isinstance(a, float) or isinstance(b, float):
        return (
            isinstance(a, int | float)
            and isinstance(b, int | float)
            and math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-6)
        )
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cadjoint.viewer.demo", description=__doc__)
    parser.add_argument("--out", type=Path, default=BUNDLE, help="bundle directory")
    parser.add_argument("--check", action="store_true", help="fail if the bundle is stale")
    args = parser.parse_args(argv)
    files = bundle_files()
    existing = {p.name for p in args.out.glob("*.json")} if args.out.exists() else set()
    if args.check:
        stale = sorted(
            name
            for name in set(files) | existing
            if name not in files
            or not (args.out / name).exists()
            or not same(json.loads((args.out / name).read_text()), json.loads(files[name]))
        )
        for name in stale:
            print(f"stale: {args.out / name}", file=sys.stderr)
        return 1 if stale else 0
    args.out.mkdir(parents=True, exist_ok=True)
    for name in existing - set(files):
        (args.out / name).unlink()
    for name, text in files.items():
        (args.out / name).write_text(text)
    print(f"wrote {len(files)} files to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - a developer command
    raise SystemExit(main())
