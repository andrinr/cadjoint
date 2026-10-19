"""The viewer reaches core only through core's public API.

The viewer used to import private helpers (``_solve_mesh``, ``_BOUNDS``),
call private methods (``_study_target``) and keep its own copy of the
projection kernel, which drifted from the original once.  Each now has a
public, documented, tested counterpart in core, and this scan keeps it that
way: no module under ``cadjoint/viewer`` may import an underscore-prefixed
name from a core module, import a private core module, read an
underscore-prefixed attribute off an imported core name, or swap out a
core class's methods.

A name that genuinely has to cross the boundary privately goes in
``ALLOWED`` with the reason next to it — better, it becomes public in core.
"""

from __future__ import annotations

import ast
from pathlib import Path

import cadjoint

VIEWER = Path(cadjoint.__file__).resolve().parent / "viewer"

#: ``(module, name)`` pairs the viewer may use although private.  Empty.
ALLOWED: frozenset[tuple[str, str]] = frozenset()


#: Methods whose reassignment on a class is monkeypatching, whatever the name.
_SWAPPED = frozenset({"__init__", "__new__", "__call__", "__init_subclass__"})


def _is_core(module: str | None) -> bool:
    if not module or not (module == "cadjoint" or module.startswith("cadjoint.")):
        return False
    return not (module == "cadjoint.viewer" or module.startswith("cadjoint.viewer."))


def _private(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    where = path.relative_to(VIEWER.parent.parent)
    found: list[str] = []
    core_names: dict[str, str] = {}  # local name -> the core module or object it binds
    viewer_names: set[str] = set()  # local names bound to the viewer's own modules

    def flag(line: int, module: str, name: str, what: str) -> None:
        if (module, name) not in ALLOWED:
            found.append(f"{where}:{line}: {what} {module}.{name}")

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and _is_core(node.module):
            for part in node.module.split("."):
                if _private(part):
                    flag(node.lineno, node.module, part, "imports private core module")
            for alias in node.names:
                if _private(alias.name):
                    flag(node.lineno, node.module, alias.name, "imports private name")
                core_names[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        elif isinstance(node, ast.ImportFrom) and (node.level or "viewer" in (node.module or "")):
            viewer_names.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if _is_core(alias.name):
                    for part in alias.name.split("."):
                        if _private(part):
                            flag(node.lineno, alias.name, part, "imports private core module")
                    core_names[alias.asname or alias.name.split(".")[0]] = alias.name

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and _private(node.attr):
            root = node.value
            while isinstance(root, (ast.Attribute, ast.Call, ast.Subscript)):
                root = root.func if isinstance(root, ast.Call) else root.value
            name = root.id if isinstance(root, ast.Name) else None
            if name in core_names:
                flag(node.lineno, core_names[name], node.attr, "reads private attribute of")
            elif name not in ("self", "cls") and name not in viewer_names:
                # Not provably core — but not provably the viewer's own either:
                # a core object held in a local (``optimization._study_target``)
                # looks exactly like this, so the viewer keeps such reads to
                # its own modules and its own ``self``.
                flag(node.lineno, ast.unparse(node.value), node.attr, "reads private attribute of")
        # Monkeypatching: assigning any attribute of an imported core name.
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if not isinstance(target, ast.Attribute) or not isinstance(target.value, ast.Name):
                    continue
                if target.value.id in core_names:
                    flag(node.lineno, core_names[target.value.id], target.attr, "assigns to")
                elif target.value.id != "self" and target.attr in _SWAPPED:
                    # ``klass.__init__ = wrapper`` in a loop over core classes.
                    flag(node.lineno, target.value.id, target.attr, "assigns to")
    return found


def test_the_viewer_uses_no_private_core_names():
    files = sorted(VIEWER.rglob("*.py"))
    assert files, f"no viewer sources under {VIEWER}"
    violations = [line for path in files for line in _violations(path)]
    assert not violations, "viewer reaches into core privately:\n" + "\n".join(violations)


def test_the_scan_catches_each_kind_of_reach(monkeypatch):
    """The scan itself works: every pattern it exists for is flagged."""
    sample = VIEWER / "_boundary_probe.py"
    source = "\n".join(
        [
            "from cadjoint.fem.study import _solve_mesh",
            "from cadjoint._compiled import call_jit",
            "import cadjoint.optimize as optimize",
            "from cadjoint.sdf.boolean import Union",
            "optimize.Optimization._study_target",
            "optimize._private_helper()",
            "Union.__init__ = None",
            "for klass in (Union,):",
            "    klass.__init__ = None",
            "def run(optimization):",
            "    return optimization._study_target(None)",
            "class Mine:",
            "    def ok(self):",
            "        return self._own",
        ]
    )
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: source)
    found = _violations(sample)
    assert len(found) == 7, found
    assert any("optimization._study_target" in line for line in found)
    assert any("_solve_mesh" in line for line in found)
    assert any("private core module cadjoint._compiled" in line for line in found)
    assert any("_private_helper" in line for line in found)
    assert any("assigns to cadjoint.sdf.boolean.Union.__init__" in line for line in found)
