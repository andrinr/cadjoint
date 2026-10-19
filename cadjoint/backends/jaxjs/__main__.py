"""``python -m cadjoint.backends.jaxjs kernels OUT.ts`` / ``scene SCENE.py OUT.json [--version 2]``."""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

from . import kernels_ts, scene_graph


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m cadjoint.backends.jaxjs")
    sub = parser.add_subparsers(dest="command", required=True)
    kernels = sub.add_parser("kernels", help="emit the generated jax-js kernels module")
    kernels.add_argument("out", type=Path)
    scene = sub.add_parser("scene", help="export a scene file's `scene` as a JSON scene graph")
    scene.add_argument("scene", type=Path)
    scene.add_argument("out", type=Path)
    scene.add_argument("--version", type=int, default=1, help="scene graph format version")
    args = parser.parse_args(argv)

    if args.command == "kernels":
        source = kernels_ts()
        args.out.write_text(source)
        print(f"wrote {args.out} ({source.count(chr(10))} lines)")
    else:
        sdf = runpy.run_path(str(args.scene), run_name="__export__")["scene"]
        args.out.write_text(json.dumps(scene_graph(sdf, version=args.version)))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
