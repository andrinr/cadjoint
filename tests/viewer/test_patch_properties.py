"""Properties every editing operation keeps, over generated requests.

``research/editing-operations.md`` is the spec of record for the 28 patch
operations; this module is the half of it that runs.  Where the per-operation
tests pin one example each, these properties hold for *every* operation over
the three shipped scenes — the starter heat sink, the gearbox end cap and the
bracket — and over sequences of them:

1. **Compile.** An accepted operation on a compiling program yields a
   compiling program.  Checked by executing the patched text the way the
   compile worker does, after every accepted step of every sequence.
2. **Refusals are pure.** A refused request returns no ``source`` and leaves
   the request object untouched; the program text is never modified because
   the operation is a pure function of it.
3. **Identity.** Every stable id an operation did not target survives it, and
   an operation that sets a value creates no new id.  The ordinal families
   an operation renumbers — a sketch's vertices and constraints, a study's
   boundary conditions — are the only ids it may take away.
4. **Span discipline.** An operation changes only the statements it targets:
   the edited declaration, the parameter it follows a name to, the scene
   assignment it extends, the imports it adds to.  Checked at statement
   level against a per-operation budget of which *kinds* of statement may
   change and how many.
5. **Inverses.** Every operation with an inverse — add/delete, set/set-back,
   insert/delete — restores the text byte for byte, or where a rewrite
   cannot restore formatting (``repr`` of a float the user wrote as ``2e-3``,
   an import that stays extended) restores the program's AST.
6. **Idempotence.** Every state-setting operation applied to its own output
   changes nothing.
7. **Malformed requests** — stray fields, wrong types, values out of range,
   non-finite numbers — are refused with the documented message.

``hypothesis`` is not in the environment; a seeded generator over the
identity table of the *current* text plays the same role and keeps every
failure reproducible.  The whole suite has to stay well under three minutes,
so each sequence is one pass over the operations in a shuffled order rather
than an open-ended walk.
"""

from __future__ import annotations

import ast
import copy
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import pytest

from cadjoint.viewer._patch import OPERATIONS
from cadjoint.viewer._patch_requests import patch_source
from cadjoint.viewer._source_map import PLAYGROUND_FILENAME, capture_profiles, identity_index
from cadjoint.viewer._worker_scene import _execute_scene
from cadjoint.viewer.patch.geometry import EDITABLE_CALLS, PRIMITIVE_DIMENSIONS
from cadjoint.viewer.patch.materials import EDITABLE_PROPERTIES, PROPERTY_BOUNDS
from cadjoint.viewer.source_map.features import FEATURE_CALL_KINDS, PRIMITIVE_CALL_KINDS
from cadjoint.viewer.source_map.identity import Identity
from cadjoint.viewer.source_map.nodes import _called_name

SEED = 20260902
SCENES_DIR = Path(__file__).resolve().parents[2] / "scenes"
SCENES = {
    name: (SCENES_DIR / f"{name}.py").read_text() for name in ("starter", "end_cap", "bracket")
}

# ── The program as the compile worker sees it ───────────────────────────────


def compiles(source: str) -> None:
    """Execute *source* the way ``/compile`` does; raise if it cannot run."""
    _execute_scene(
        source, capture=(("__profiles__", lambda: capture_profiles(PLAYGROUND_FILENAME)),)
    )


def normalized(source: str, *, ignore_imports: bool = False) -> list[str]:
    """The program's statements as AST dumps, numbers compared by value.

    ``2e-3`` and ``0.002`` are the same number; ``0`` and ``0.0`` are too for
    every argument the viewer writes, and a tuple of numbers is the list the
    patch layer writes back.  Imports are optionally dropped: an operation
    that imports a symbol never takes the import back out.
    """

    class Numbers(ast.NodeTransformer):
        def visit_Constant(self, node: ast.Constant) -> ast.AST:
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return ast.Constant(value=float(node.value))
            return node

        def visit_Tuple(self, node: ast.Tuple) -> ast.AST:
            # ``(24, 17, 13)`` and ``[24, 17, 13]`` are one vector to every
            # constructor the viewer writes to; the patch layer writes lists.
            self.generic_visit(node)
            return ast.List(elts=node.elts, ctx=node.ctx)

    tree = Numbers().visit(ast.parse(source))
    statements = [
        statement
        for statement in tree.body
        if not (ignore_imports and isinstance(statement, (ast.Import, ast.ImportFrom)))
    ]
    return [ast.dump(statement) for statement in statements]


# ── Statement kinds, for the span-discipline budget ─────────────────────────

_DECLARATION_CALLS = {"ThermalStudy", "ElasticStudy", "SimMesh", "Optimization"}
_PARAMETER_CALLS = {"Scalar", "Vector", "Vector2"}
_BOOLEAN_CALLS = {"Union", "Difference", "Intersection"}


def statement_kind(statement: ast.stmt) -> str:
    """Which kind of statement this is, in the vocabulary the budgets use."""
    if isinstance(statement, (ast.Import, ast.ImportFrom)):
        return "import"
    if isinstance(statement, ast.Assign):
        targets = [target.id for target in statement.targets if isinstance(target, ast.Name)]
        called = _called_name(statement.value)
        if "scene" in targets:
            return "scene"
        if called in _BOOLEAN_CALLS:
            return "union"
        if called == "PolygonProfile":
            return "profile"
        if called in PRIMITIVE_CALL_KINDS:
            return "primitive"
        if called in FEATURE_CALL_KINDS:
            return "feature"
        if called == "Material":
            return "material"
        if called in _PARAMETER_CALLS:
            return "parameter"
        if called in _DECLARATION_CALLS:
            return "declaration"
        return "other"
    if isinstance(statement, ast.Expr):
        called = _called_name(statement.value) or ""
        if called.endswith("Constraint"):
            return "constraint"
        if called == "satisfy_constraints":
            return "solve"
    return "other"


def statement_kinds(source: str) -> Counter[tuple[str, str]]:
    """Multiset of ``(dump, kind)`` for every top-level statement."""
    return Counter(
        (dump, statement_kind(statement))
        for dump, statement in zip(normalized(source), ast.parse(source).body)
    )


#: ``op -> (kinds that may change, max non-import statements removed,
#: max non-import statements inserted)``.  "Removed" counts every statement
#: of the old text that is not in the new one, so a rewritten statement is
#: one removed and one inserted.  None means any number *of the allowed
#: kinds* — a vertex deletion takes every constraint on that vertex with it.
BUDGET: dict[str, tuple[frozenset[str], int | None, int | None]] = {
    "set_vertex": (frozenset({"profile", "parameter"}), 1, 1),
    "insert_vertex": (frozenset({"profile", "constraint"}), None, None),
    "delete_vertex": (frozenset({"profile", "constraint"}), None, None),
    "set_value": (
        frozenset({"profile", "parameter", "primitive", "feature", "material", "import"}),
        1,
        1,
    ),
    "add_primitive": (frozenset({"scene", "import", "primitive"}), 1, 2),
    "add_material": (frozenset({"import", "material"}), 0, 1),
    "assign_material": (frozenset({"primitive", "feature"}), 1, 1),
    "set_material_property": (frozenset({"material"}), 1, 1),
    "add_sketch": (frozenset({"import", "profile"}), 0, 1),
    "set_sketch_plane": (frozenset({"profile", "import"}), 1, 1),
    "add_extrusion": (frozenset({"scene", "import", "feature"}), 1, 2),
    "add_revolution": (frozenset({"scene", "import", "feature"}), 1, 2),
    "add_loft": (frozenset({"scene", "import", "feature"}), 1, 2),
    "add_constraint": (frozenset({"import", "constraint"}), 0, 1),
    "delete_constraint": (frozenset({"constraint"}), 1, 0),
    "set_constraint_value": (frozenset({"constraint", "parameter"}), 1, 1),
    "solve_sketch": (frozenset({"solve", "import"}), 1, 1),
    "delete_object": (
        frozenset({"profile", "primitive", "feature", "union", "scene", "constraint"}),
        None,
        2,
    ),
    "add_study": (frozenset({"import", "declaration"}), 0, 1),
    "delete_study": (frozenset({"declaration"}), 1, 0),
    "add_study_bc": (frozenset({"declaration", "import"}), 1, 1),
    "delete_study_bc": (frozenset({"declaration"}), 1, 1),
    "set_study_value": (frozenset({"declaration"}), 1, 1),
    "add_mesh": (frozenset({"import", "declaration"}), 0, 1),
    "delete_mesh": (frozenset({"declaration"}), 1, 0),
    "set_mesh_value": (frozenset({"declaration"}), 1, 1),
    "delete_optimization": (frozenset({"declaration"}), 1, 0),
    "set_optimization_value": (frozenset({"declaration"}), 1, 1),
}


def assert_within_budget(op: str, before: str, after: str) -> None:
    kinds, max_removed, max_inserted = BUDGET[op]
    old, new = statement_kinds(before), statement_kinds(after)
    removed = list((old - new).elements())
    inserted = list((new - old).elements())
    stray = [kind for _, kind in removed + inserted if kind not in kinds]
    assert not stray, f"{op} touched a {stray[0]} statement it has no business with"
    removed_count = sum(1 for _, kind in removed if kind != "import")
    inserted_count = sum(1 for _, kind in inserted if kind != "import")
    if max_removed is not None:
        assert removed_count <= max_removed, f"{op} removed {removed_count} statements"
    if max_inserted is not None:
        assert inserted_count <= max_inserted, f"{op} inserted {inserted_count} statements"


# ── Identity: what an operation may take away ───────────────────────────────

#: Operations after which ids may appear.  Everything else sets state.
ADDING = frozenset(
    {
        "insert_vertex",
        "add_primitive",
        "add_material",
        "add_sketch",
        "add_extrusion",
        "add_revolution",
        "add_loft",
        "add_constraint",
        "add_study",
        "add_study_bc",
        "add_mesh",
        "solve_sketch",
    }
)


def _children(index: dict[str, Identity], owner: Identity) -> set[str]:
    """Ids that live inside *owner*: its plane, vertices, constraints, BCs."""
    return {
        key
        for key, item in index.items()
        if item.owner == owner.id or (item.kind == "plane" and item.owner == owner.id)
    }


def _family(index: dict[str, Identity], member: Identity, kind: str) -> set[str]:
    """Every ordinal id of *kind* that shares *member*'s owner."""
    return {key for key, item in index.items() if item.kind == kind and item.owner == member.owner}


def expendable(op: str, request: dict[str, Any], index: dict[str, Identity]) -> set[str]:
    """The ids *op* is allowed to take out of the program for this request."""
    target = index.get(request.get("id", ""))
    if target is None:
        return set()
    if op in {"insert_vertex", "delete_vertex"}:
        owner = index[target.owner] if target.kind == "vertex" else target
        return _children(index, owner) - {f"plane:{owner.token}"}
    if op == "delete_object":
        return {target.id} | _children(index, target)
    if op == "delete_constraint":
        return _family(index, target, "constraint")
    if op in {"delete_study", "delete_mesh", "delete_optimization"}:
        return {target.id} | _children(index, target)
    if op == "delete_study_bc":
        return _family(index, target, "bc")
    return set()


# ── The generator ───────────────────────────────────────────────────────────


class Generator:
    """Requests for every operation, drawn from the identity table of one text."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def number(self, low: float = -1.5, high: float = 1.5) -> float:
        return round(self.rng.uniform(low, high), 3)

    def point(self, count: int = 3, low: float = -1.5, high: float = 1.5) -> list[float]:
        return [self.number(low, high) for _ in range(count)]

    def unit(self) -> float:
        return round(self.rng.uniform(0.05, 0.95), 3)

    def pick(self, index: dict[str, Identity], kind: str, **where) -> Identity | None:
        candidates = [
            item
            for item in index.values()
            if item.kind == kind
            and all(getattr(item, key) == value for key, value in where.items())
        ]
        return self.rng.choice(candidates) if candidates else None

    def vertex_count(self, index: dict[str, Identity], sketch: Identity) -> int:
        return sum(
            1 for item in index.values() if item.kind == "vertex" and item.owner == sketch.id
        )

    def request(self, op: str, source: str) -> dict[str, Any] | None:
        """One plausible request for *op* against *source*, or None if nothing fits."""
        index = identity_index(source)
        builder = getattr(self, f"build_{op}")
        return builder(index, source)

    # ── sketch vertices ─────────────────────────────────────────────────────

    def build_set_vertex(self, index, source):
        vertex = self.pick(index, "vertex")
        return vertex and {"op": "set_vertex", "id": vertex.id, "xy": self.point(2)}

    def build_insert_vertex(self, index, source):
        sketch = self.pick(index, "sketch")
        if sketch is None:
            return None
        count = self.vertex_count(index, sketch)
        if count == 0:
            return None
        position = self.rng.randint(0, count)
        body = {"op": "insert_vertex", "xy": self.point(2)}
        if position < count:
            return {**body, "id": f"vertex:{sketch.token}[{position}]"}
        return {**body, "id": sketch.id, "index": position}

    def build_delete_vertex(self, index, source):
        vertex = self.pick(index, "vertex")
        return vertex and {"op": "delete_vertex", "id": vertex.id}

    # ── values ──────────────────────────────────────────────────────────────

    def value_for(self, name: str, argument: str):
        size = EDITABLE_CALLS[name][argument]
        if argument in {"size", "radius", "height", "depth"}:
            return self.point(size, 0.1, 1.2) if size > 1 else self.number(0.1, 1.2)
        if argument == "color":
            return [self.unit() for _ in range(3)]
        if name == "Material" and argument in PROPERTY_BOUNDS:
            low, high = PROPERTY_BOUNDS[argument]
            return round(low + (high - low) * self.unit(), 6)
        return self.point(size) if size > 1 else self.number()

    def build_set_value(self, index, source):
        kind = self.rng.choice(["primitive", "feature", "material", "sketch"])
        target = self.pick(index, kind)
        if target is None:
            return None
        name = "PolygonProfile" if kind == "sketch" else target.call
        if name not in EDITABLE_CALLS:
            return None
        argument = self.rng.choice(sorted(EDITABLE_CALLS[name]))
        return {
            "op": "set_value",
            "id": target.id,
            "name": name,
            "argument": argument,
            "value": self.value_for(name, argument),
        }

    def build_add_primitive(self, index, source):
        kind = self.rng.choice(sorted(PRIMITIVE_DIMENSIONS))
        dimensions = {
            key: self.point(3, 0.1, 0.6) if size > 1 else self.number(0.1, 0.6)
            for key, size in PRIMITIVE_DIMENSIONS[kind].items()
        }
        return {
            "op": "add_primitive",
            "kind": kind,
            "position": self.point(),
            "dimensions": dimensions,
        }

    def build_add_material(self, index, source):
        return {
            "op": "add_material",
            "color": [self.unit() for _ in range(3)],
            "roughness": self.unit(),
        }

    def build_assign_material(self, index, source):
        target = self.pick(index, self.rng.choice(["sketch", "primitive", "feature"]))
        material = self.pick(index, "material")
        if target is None or material is None:
            return None
        return {"op": "assign_material", "id": target.id, "material": material.variable}

    def build_set_material_property(self, index, source):
        material = self.pick(index, "material")
        if material is None:
            return None
        key = self.rng.choice(EDITABLE_PROPERTIES)
        low, high = PROPERTY_BOUNDS[key]
        value = None if self.rng.random() < 0.2 else round(low + (high - low) * self.unit(), 6)
        return {"op": "set_material_property", "id": material.id, "property": key, "value": value}

    # ── sketches and operators ──────────────────────────────────────────────

    def build_add_sketch(self, index, source):
        return {"op": "add_sketch", "origin": self.point()}

    def build_set_sketch_plane(self, index, source):
        sketch = self.pick(index, "sketch")
        if sketch is None:
            return None
        kind = self.rng.choice(["world", "cap", "side", "face", "tangent"])
        if kind == "world":
            normal = self.rng.choice([[0, 0, 1], [0, 1, 0], [1, 0, 0], self.point()])
            reference = {"kind": "world", "origin": self.point(), "normal": normal}
        else:
            owners = [
                item
                for item in index.values()
                if item.kind in {"feature", "primitive"} and item.line < (sketch.line or 0)
            ]
            if not owners:
                return None
            owner = self.rng.choice(owners)
            extra = {
                "cap": {"sign": self.rng.choice(["+", "-"])},
                "side": {"edge": self.rng.randint(0, 3)},
                "face": {"key": self.rng.choice(["+x", "-x", "+y", "-y", "+z", "-z"])},
                # A point the viewer would pick: on the part's upper surface,
                # where the field's gradient is a normal a plane can use.
                "tangent": {"near": [self.number(-0.3, 0.3), self.number(-0.3, 0.3), 2.0]},
            }[kind]
            reference = {"kind": kind, "owner": owner.id, **extra}
        body = {"op": "set_sketch_plane", "id": sketch.id, "reference": reference}
        if self.rng.random() < 0.3:
            body["offset"] = self.number(0.0, 0.5)
        return body

    def build_add_extrusion(self, index, source):
        sketch = self.pick(index, "sketch")
        return sketch and {"op": "add_extrusion", "id": sketch.id, "depth": self.number(0.1, 1.0)}

    def build_add_revolution(self, index, source):
        sketch = self.pick(index, "sketch")
        return sketch and {"op": "add_revolution", "id": sketch.id, "offset": self.number(0.0, 0.5)}

    def build_add_loft(self, index, source):
        sketches = [item for item in index.values() if item.kind == "sketch"]
        if len(sketches) < 2:
            return None
        first, second = self.rng.sample(sketches, 2)
        return {
            "op": "add_loft",
            "id_a": first.id,
            "id_b": second.id,
            "height": self.number(0.2, 1.0),
        }

    # ── constraints ─────────────────────────────────────────────────────────

    def build_add_constraint(self, index, source):
        sketch = self.pick(index, "sketch")
        if sketch is None:
            return None
        count = self.vertex_count(index, sketch)
        if count < 4:
            return None
        kind = self.rng.choice(
            [
                "fixed",
                "distance",
                "horizontal",
                "vertical",
                "coincident",
                "parallel",
                "perpendicular",
            ]
        )
        arity = {"fixed": 1, "distance": 2, "parallel": 4, "perpendicular": 4}.get(kind, 2)
        indices = self.rng.sample(range(count), arity)
        body = {"op": "add_constraint", "id": sketch.id, "kind": kind, "indices": indices}
        if kind == "fixed":
            body["value"] = self.point(2)
        elif kind == "distance":
            body["value"] = self.number(0.1, 1.0)
        return body

    def build_delete_constraint(self, index, source):
        constraint = self.pick(index, "constraint")
        return constraint and {"op": "delete_constraint", "id": constraint.id}

    def build_set_constraint_value(self, index, source):
        # Only pins and distances carry a value; the relational kinds refuse.
        valued = [
            item
            for item in index.values()
            if item.kind == "constraint" and item.call in {"FixedConstraint", "DistanceConstraint"}
        ]
        if not valued:
            return None
        constraint = self.rng.choice(valued)
        value = self.point(2) if constraint.call == "FixedConstraint" else self.number(0.1, 1.0)
        return {"op": "set_constraint_value", "id": constraint.id, "value": value}

    def build_solve_sketch(self, index, source):
        sketch = self.pick(index, "sketch")
        return sketch and {
            "op": "solve_sketch",
            "id": sketch.id,
            "method": self.rng.choice(["newton", "adam", "sgd"]),
            "iterations": self.rng.randint(1, 12),
        }

    def build_delete_object(self, index, source):
        target = self.pick(index, self.rng.choice(["sketch", "primitive", "feature"]))
        return target and {"op": "delete_object", "id": target.id}

    # ── studies ─────────────────────────────────────────────────────────────

    def build_add_study(self, index, source):
        body = {"op": "add_study", "kind": self.rng.choice(["thermal", "elastic"])}
        if self.rng.random() < 0.5:
            body["name"] = f"study-{self.rng.randint(1, 999)}"
        return body

    def build_delete_study(self, index, source):
        study = self.pick(index, "study")
        return study and {"op": "delete_study", "id": study.id}

    def selection(self) -> dict[str, Any]:
        if self.rng.random() < 0.5:
            return {
                "kind": "side",
                "side": self.rng.choice(["+x", "-x", "+y", "-y", "+z", "-z"]),
                "tol": None,
            }
        low = self.point()
        return {"kind": "box", "min_corner": low, "max_corner": [value + 0.5 for value in low]}

    def build_add_study_bc(self, index, source):
        study = self.pick(index, "study")
        if study is None:
            return None
        thermal = study.call == "ThermalStudy"
        bc_type = self.rng.choice(["dirichlet", "heat_flux"] if thermal else ["fixed", "traction"])
        body = {
            "op": "add_study_bc",
            "id": study.id,
            "bc_type": bc_type,
            "selection": self.selection(),
        }
        if bc_type == "traction":
            body["value"] = self.point()
        elif bc_type != "fixed":
            body["value"] = self.number(0.0, 100.0)
        return body

    def build_delete_study_bc(self, index, source):
        bc = self.pick(index, "bc")
        return bc and {"op": "delete_study_bc", "id": bc.id}

    def build_set_study_value(self, index, source):
        if self.rng.random() < 0.5:
            bc = self.pick(index, "bc")
            if bc is None:
                return None
            value = self.point() if bc.call == "Traction" else self.number(0.0, 100.0)
            return {"op": "set_study_value", "id": bc.id, "value": value}
        study = self.pick(index, "study")
        if study is None:
            return None
        thermal = study.call == "ThermalStudy"
        argument = self.rng.choice(
            ["conductivity", "source", "resolution", "bounds", "size"]
            if thermal
            else ["youngs", "poisson", "resolution", "bounds", "size"]
        )
        value = {
            "resolution": [self.rng.randint(4, 12) for _ in range(3)],
            "bounds": self.point(),
            "size": self.point(3, 0.5, 2.0),
            "poisson": self.number(0.1, 0.45),
        }.get(argument, self.number(0.5, 50.0))
        return {"op": "set_study_value", "id": study.id, "argument": argument, "value": value}

    # ── meshes and optimizations ────────────────────────────────────────────

    def build_add_mesh(self, index, source):
        body: dict[str, Any] = {"op": "add_mesh"}
        if self.rng.random() < 0.5:
            body["name"] = f"mesh-{self.rng.randint(1, 999)}"
        return body

    def build_delete_mesh(self, index, source):
        mesh = self.pick(index, "mesh")
        return mesh and {"op": "delete_mesh", "id": mesh.id}

    def build_set_mesh_value(self, index, source):
        mesh = self.pick(index, "mesh")
        if mesh is None:
            return None
        argument = self.rng.choice(["resolution", "bounds", "size", "padding", "method", "domain"])
        if argument == "domain":
            owners = [
                item.variable
                for item in index.values()
                if item.kind in {"primitive", "feature"}
                and item.variable
                and item.line < (mesh.line or 0)
            ]
            if not owners:
                return None
            value: Any = self.rng.choice(owners)
        elif argument == "method":
            value = self.rng.choice(["hex", "tet4", "tet10"])
        elif argument == "resolution":
            value = [self.rng.randint(4, 12) for _ in range(3)]
        elif argument == "padding":
            value = self.number(0.0, 0.3)
        elif argument == "bounds":
            value = self.point()
        else:
            value = self.point(3, 0.5, 2.0)
        return {"op": "set_mesh_value", "id": mesh.id, "argument": argument, "value": value}

    def build_delete_optimization(self, index, source):
        optimization = self.pick(index, "optimization")
        return optimization and {"op": "delete_optimization", "id": optimization.id}

    def build_set_optimization_value(self, index, source):
        optimization = self.pick(index, "optimization")
        if optimization is None:
            return None
        if self.rng.random() < 0.5:
            return {
                "op": "set_optimization_value",
                "id": optimization.id,
                "argument": "steps",
                "value": self.rng.randint(1, 20),
            }
        return {
            "op": "set_optimization_value",
            "id": optimization.id,
            "argument": "learning_rate",
            "value": self.number(0.001, 0.1),
        }


# ── 1–4: sequences of accepted operations ───────────────────────────────────


def _step(source: str, request: dict[str, Any]) -> tuple[bool, str]:
    """Apply one request; check the invariants of whichever outcome it has."""
    op = request["op"]
    sent = copy.deepcopy(request)
    result = patch_source({"source": source, **request})
    assert request == sent, f"{op} mutated the request it was given"
    if not result["ok"]:
        assert set(result) == {"ok", "error"}, f"{op} refused but answered with {sorted(result)}"
        assert isinstance(result["error"], str) and result["error"]
        return False, source
    patched = result["source"]
    ast.parse(patched)
    try:
        compiles(patched)
    except Exception as failure:
        raise AssertionError(
            f"{op} left a program that does not run: {failure!r}\n{request}"
        ) from failure
    before, after = identity_index(source), identity_index(patched)
    lost = set(before) - set(after)
    allowed = expendable(op, request, before)
    assert lost <= allowed, f"{op} lost {sorted(lost - allowed)}"
    if op not in ADDING:
        gained = set(after) - set(before)
        assert not gained, f"{op} sets a value yet created {sorted(gained)}"
    assert_within_budget(op, source, patched)
    return True, patched


@pytest.mark.parametrize("scene", sorted(SCENES))
@pytest.mark.parametrize("seed", [SEED, SEED + 1])
def test_a_sequence_of_operations_keeps_every_invariant(scene: str, seed: int) -> None:
    """One pass over every operation in a shuffled order, on the live text.

    Each accepted step leaves a program that compiles, keeps every id it did
    not target, and changed only the statements its budget allows; each
    refused step is pure.  The text carries forward, so later operations see
    what earlier ones wrote — the new sketch gets extruded, the new study
    gets a boundary condition, the inserted vertex gets constrained.
    """
    rng = random.Random(f"{scene}:{seed}")
    generator = Generator(rng)
    source = SCENES[scene]
    compiles(source)
    order = sorted(OPERATIONS)
    rng.shuffle(order)
    attempted = accepted = 0
    for op in order + order[: len(order) // 2]:
        request = generator.request(op, source)
        if request is None:
            continue
        attempted += 1
        applied, source = _step(source, request)
        accepted += applied
    assert (
        attempted >= len(OPERATIONS) - 4
    ), "the generator found targets for almost every operation"
    assert accepted >= attempted // 3, "most generated requests should be accepted"


# ── 5: inverses ─────────────────────────────────────────────────────────────


def apply(source: str, **request) -> str:
    result = patch_source({"source": source, **request})
    assert result["ok"] is True, result.get("error")
    return result["source"]


def _literal_at(source: str, node) -> Any:
    return ast.literal_eval(node)


def _same_ast(first: str, second: str, *, ignore_imports: bool = False) -> None:
    assert normalized(first, ignore_imports=ignore_imports) == normalized(
        second, ignore_imports=ignore_imports
    )


Inverse = Callable[[str, random.Random], None]


def _inverse_set_vertex(source: str, rng: random.Random) -> None:
    from cadjoint.viewer._source_map import locate_profile_call

    index = identity_index(source)
    vertex = Generator(rng).pick(index, "vertex")
    call = locate_profile_call(source, vertex.line)
    start, end = call.element_spans[vertex.index]
    original = ast.literal_eval(source[start:end])
    moved = apply(source, op="set_vertex", id=vertex.id, xy=[9.5, -9.5])
    assert moved != source
    back = apply(moved, op="set_vertex", id=vertex.id, xy=original)
    _same_ast(back, source)


def _inverse_insert_vertex(source: str, rng: random.Random) -> None:
    index = identity_index(source)
    sketch = Generator(rng).pick(index, "sketch")
    count = sum(1 for item in index.values() if item.kind == "vertex" and item.owner == sketch.id)
    position = rng.randint(0, count)
    if position < count:
        grown = apply(
            source, op="insert_vertex", id=f"vertex:{sketch.token}[{position}]", xy=[0.5, 0.5]
        )
    else:
        grown = apply(source, op="insert_vertex", id=sketch.id, index=position, xy=[0.5, 0.5])
    back = apply(grown, op="delete_vertex", id=f"vertex:{sketch.token}[{position}]")
    assert back == source


def _inverse_add_primitive(source: str, rng: random.Random) -> None:
    kind = rng.choice(sorted(PRIMITIVE_DIMENSIONS))
    dimensions = {
        key: [0.2, 0.3, 0.4] if size > 1 else 0.25
        for key, size in PRIMITIVE_DIMENSIONS[kind].items()
    }
    grown = apply(source, op="add_primitive", kind=kind, position=[0, 1, 2], dimensions=dimensions)
    new = (set(identity_index(grown)) - set(identity_index(source))).pop()
    back = apply(grown, op="delete_object", id=new)
    _same_ast(back, source, ignore_imports=True)


def _inverse_add_sketch(source: str, rng: random.Random) -> None:
    grown = apply(source, op="add_sketch", origin=[0, 1, 2])
    new = next(
        key
        for key in set(identity_index(grown)) - set(identity_index(source))
        if key.startswith("assign:")
    )
    back = apply(grown, op="delete_object", id=new)
    _same_ast(back, source, ignore_imports=True)


def _inverse_assign_material(source: str, rng: random.Random) -> None:
    index = identity_index(source)
    generator = Generator(rng)
    tree = ast.parse(source)
    targets = [
        item
        for item in index.values()
        if item.kind in {"primitive", "feature"}
        and item.variable
        and any(
            isinstance(node, ast.Call)
            and _called_name(node) == item.call
            and node.lineno == item.line
            and any(k.arg == "material" and isinstance(k.value, ast.Name) for k in node.keywords)
            for node in ast.walk(tree)
        )
    ]
    target = rng.choice(targets)
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _called_name(node) == target.call
        and node.lineno == target.line
    )
    original = next(k.value.id for k in call.keywords if k.arg == "material")
    other = generator.pick(index, "material")
    changed = apply(source, op="assign_material", id=target.id, material=other.variable)
    back = apply(changed, op="assign_material", id=target.id, material=original)
    assert back == source


def _inverse_set_material_property(source: str, rng: random.Random) -> None:
    index = identity_index(source)
    material = Generator(rng).pick(index, "material")
    key = rng.choice(EDITABLE_PROPERTIES)
    call = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and _called_name(node) == "Material"
        and node.lineno == material.line
    )
    existing = next((k for k in call.keywords if k.arg == key), None)
    low, high = PROPERTY_BOUNDS[key]
    value = round(low + (high - low) * 0.37, 6)
    changed = apply(source, op="set_material_property", id=material.id, property=key, value=value)
    if existing is None:
        # Added, then removed: the text is exactly what it was.
        back = apply(changed, op="set_material_property", id=material.id, property=key, value=None)
        assert back == source
    else:
        original = ast.literal_eval(existing.value)
        back = apply(
            changed, op="set_material_property", id=material.id, property=key, value=original
        )
        _same_ast(back, source)


def _plain_plane(source: str, sketch: Identity) -> tuple[list[float], list[float]] | None:
    """``(origin, normal)`` of a sketch written on a literal ``SketchPlane``."""
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and _called_name(node) == "PolygonProfile"
            and node.lineno == sketch.line
        ):
            plane = next((k.value for k in node.keywords if k.arg == "plane"), None)
            if (
                isinstance(plane, ast.Call)
                and _called_name(plane) == "SketchPlane"
                and not plane.args
                and {k.arg for k in plane.keywords} == {"origin", "normal"}
            ):
                values = {k.arg: ast.literal_eval(k.value) for k in plane.keywords}
                return values["origin"], values["normal"]
    return None


def _inverse_set_sketch_plane(source: str, rng: random.Random) -> None:
    index = identity_index(source)
    sketches = [
        item for item in index.values() if item.kind == "sketch" and _plain_plane(source, item)
    ]
    sketch = rng.choice(sketches)
    origin, normal = _plain_plane(source, sketch)
    owners = [item for item in index.values() if item.kind == "feature" and item.line < sketch.line]
    if owners:
        planted = apply(
            source,
            op="set_sketch_plane",
            id=sketch.id,
            reference={"kind": "cap", "owner": rng.choice(owners).id, "sign": "+"},
        )
    else:
        planted = apply(
            source,
            op="set_sketch_plane",
            id=sketch.id,
            reference={"kind": "world", "origin": [3, 3, 3], "normal": [1, 0, 0]},
        )
    back = apply(
        planted,
        op="set_sketch_plane",
        id=sketch.id,
        reference={"kind": "world", "origin": origin, "normal": normal},
    )
    _same_ast(back, source, ignore_imports=True)


def _inverse_operator(op: str) -> Inverse:
    def inverse(source: str, rng: random.Random) -> None:
        grown = apply(source, op="add_sketch", origin=[0, 0, 0])
        sketch = next(
            key
            for key in set(identity_index(grown)) - set(identity_index(source))
            if key.startswith("assign:")
        )
        if op == "add_loft":
            twice = apply(grown, op="add_sketch", origin=[0, 0, 1])
            other = next(
                key
                for key in set(identity_index(twice)) - set(identity_index(grown))
                if key.startswith("assign:")
            )
            solid = apply(twice, op="add_loft", id_a=sketch, id_b=other)
            body = f"{sketch}_body"
        else:
            twice = grown
            solid = apply(grown, op=op, id=sketch)
            body = f"{sketch}_body"
        back = apply(solid, op="delete_object", id=body)
        _same_ast(back, twice, ignore_imports=True)

    return inverse


def _inverse_add_constraint(source: str, rng: random.Random) -> None:
    index = identity_index(source)
    generator = Generator(rng)
    sketches = [
        item
        for item in index.values()
        if item.kind == "sketch" and generator.vertex_count(index, item) >= 4
    ]
    sketch = rng.choice(sketches)
    grown = apply(source, op="add_constraint", id=sketch.id, kind="horizontal", indices=[0, 1])
    last = max(
        int(key.rsplit("[", 1)[1][:-1])
        for key in identity_index(grown)
        if key.startswith(f"constraint:{sketch.token}[")
    )
    back = apply(grown, op="delete_constraint", id=f"constraint:{sketch.token}[{last}]")
    _same_ast(back, source, ignore_imports=True)


def _inverse_set_constraint_value(source: str, rng: random.Random) -> None:
    from cadjoint.viewer._source_map import locate_constraint_statements
    from cadjoint.viewer.source_map.nodes import _editable_value_node

    index = identity_index(source)
    valued = [
        item
        for item in index.values()
        if item.kind == "constraint" and item.call in {"FixedConstraint", "DistanceConstraint"}
    ]
    constraint = rng.choice(valued)
    owner = index[constraint.owner]
    located = locate_constraint_statements(source, owner.line)[constraint.index]
    target = located.call.args[1 if located.kind == "fixed" else 2]
    original = ast.literal_eval(_editable_value_node(target, ast.parse(source)))
    value = [0.25, 0.75] if located.kind == "fixed" else 0.625
    changed = apply(source, op="set_constraint_value", id=constraint.id, value=value)
    assert changed != source
    back = apply(changed, op="set_constraint_value", id=constraint.id, value=original)
    _same_ast(back, source)


def _inverse_add_study(source: str, rng: random.Random) -> None:
    grown = apply(source, op="add_study", kind=rng.choice(["thermal", "elastic"]), name="probe")
    back = apply(grown, op="delete_study", study="probe")
    _same_ast(back, source, ignore_imports=True)


def _inverse_add_mesh(source: str, rng: random.Random) -> None:
    grown = apply(source, op="add_mesh", name="probe")
    back = apply(grown, op="delete_mesh", mesh="probe")
    _same_ast(back, source, ignore_imports=True)


def _inverse_add_study_bc(source: str, rng: random.Random) -> None:
    index = identity_index(source)
    study = Generator(rng).pick(index, "study")
    thermal = study.call == "ThermalStudy"
    grown = apply(
        source,
        op="add_study_bc",
        id=study.id,
        bc_type="dirichlet" if thermal else "fixed",
        selection={"kind": "side", "side": "+z", "tol": None},
        **({"value": 12.5} if thermal else {}),
    )
    last = max(
        int(key.rsplit("[", 1)[1][:-1])
        for key in identity_index(grown)
        if key.startswith(f"bc:{study.token}[")
    )
    back = apply(grown, op="delete_study_bc", id=f"bc:{study.token}[{last}]")
    _same_ast(back, source, ignore_imports=True)


def _inverse_set_study_value(source: str, rng: random.Random) -> None:
    from cadjoint.viewer._source_map import locate_study_statements

    index = identity_index(source)
    study = Generator(rng).pick(index, "study")
    located = locate_study_statements(source)[study.index]
    argument = "conductivity" if study.call == "ThermalStudy" else "youngs"
    original = ast.literal_eval(next(k.value for k in located.call.keywords if k.arg == argument))
    changed = apply(
        source, op="set_study_value", id=study.id, argument=argument, value=original * 2 + 1
    )
    back = apply(changed, op="set_study_value", id=study.id, argument=argument, value=original)
    _same_ast(back, source)


def _inverse_set_mesh_value(source: str, rng: random.Random) -> None:
    from cadjoint.viewer._source_map import locate_mesh_statements

    index = identity_index(source)
    mesh = Generator(rng).pick(index, "mesh")
    located = locate_mesh_statements(source)[mesh.index]
    keywords = {k.arg: k.value for k in located.call.keywords}
    argument = rng.choice(
        [key for key in ("resolution", "bounds", "size", "domain") if key in keywords]
    )
    if argument == "domain":
        original = keywords["domain"].id
        other = next(
            item.variable
            for item in index.values()
            if item.kind in {"primitive", "feature"} and item.variable and item.line < mesh.line
        )
        changed = apply(source, op="set_mesh_value", id=mesh.id, argument="domain", value=other)
        back = apply(changed, op="set_mesh_value", id=mesh.id, argument="domain", value=original)
        assert back == source
        return
    original = list(ast.literal_eval(keywords[argument]))
    value = (
        [item + 1 for item in original]
        if argument == "resolution"
        else [item + 0.5 for item in original]
    )
    changed = apply(source, op="set_mesh_value", id=mesh.id, argument=argument, value=value)
    back = apply(changed, op="set_mesh_value", id=mesh.id, argument=argument, value=original)
    _same_ast(back, source)


def _inverse_set_optimization_value(source: str, rng: random.Random) -> None:
    from cadjoint.viewer._source_map import locate_optimization_statements

    index = identity_index(source)
    optimization = Generator(rng).pick(index, "optimization")
    located = locate_optimization_statements(source)[optimization.index]
    argument = rng.choice(["steps", "learning_rate"])
    original = ast.literal_eval(next(k.value for k in located.call.keywords if k.arg == argument))
    value = original + 1 if argument == "steps" else original * 3
    changed = apply(
        source, op="set_optimization_value", id=optimization.id, argument=argument, value=value
    )
    back = apply(
        changed, op="set_optimization_value", id=optimization.id, argument=argument, value=original
    )
    _same_ast(back, source)


INVERSES: dict[str, Inverse] = {
    "set_vertex": _inverse_set_vertex,
    "insert_vertex": _inverse_insert_vertex,
    "add_primitive": _inverse_add_primitive,
    "add_sketch": _inverse_add_sketch,
    "assign_material": _inverse_assign_material,
    "set_material_property": _inverse_set_material_property,
    "set_sketch_plane": _inverse_set_sketch_plane,
    "add_extrusion": _inverse_operator("add_extrusion"),
    "add_revolution": _inverse_operator("add_revolution"),
    "add_loft": _inverse_operator("add_loft"),
    "add_constraint": _inverse_add_constraint,
    "set_constraint_value": _inverse_set_constraint_value,
    "add_study": _inverse_add_study,
    "add_mesh": _inverse_add_mesh,
    "add_study_bc": _inverse_add_study_bc,
    "set_study_value": _inverse_set_study_value,
    "set_mesh_value": _inverse_set_mesh_value,
    "set_optimization_value": _inverse_set_optimization_value,
}

#: Operations with no inverse in the vocabulary, and why.
NO_INVERSE = {
    "delete_vertex": "its inverse is insert_vertex, covered from that side",
    "set_value": "covered by the idempotence property; the inverse is set_value itself",
    "add_material": "there is no delete_material operation",
    "delete_constraint": "its inverse is add_constraint, covered from that side",
    "solve_sketch": "a solve step is never removed by the viewer",
    "delete_object": "its inverse is add_primitive/add_sketch/an operator, covered from that side",
    "delete_study": "its inverse is add_study, covered from that side",
    "delete_study_bc": "its inverse is add_study_bc, covered from that side",
    "delete_mesh": "its inverse is add_mesh, covered from that side",
    "delete_optimization": "there is no add_optimization operation",
}


def test_every_operation_has_an_inverse_or_a_reason_not_to() -> None:
    assert set(INVERSES) | set(NO_INVERSE) == set(OPERATIONS)
    assert not set(INVERSES) & set(NO_INVERSE)


def _declares(source: str, kind: str) -> bool:
    return any(item.kind == kind for item in identity_index(source).values())


#: Operations whose inverse needs a target the scene may not declare.
NEEDS = {"set_optimization_value": "optimization"}


@pytest.mark.parametrize("scene", sorted(SCENES))
@pytest.mark.parametrize("op", sorted(INVERSES))
def test_an_operation_followed_by_its_inverse_restores_the_program(scene: str, op: str) -> None:
    if op in NEEDS and not _declares(SCENES[scene], NEEDS[op]):
        pytest.skip(f"{scene} declares no {NEEDS[op]}")
    INVERSES[op](SCENES[scene], random.Random(f"{scene}:{op}:{SEED}"))


def test_set_value_and_back_restores_the_program() -> None:
    source = SCENES["starter"]
    index = identity_index(source)
    board = index["assign:board"]
    changed = apply(
        source, op="set_value", id=board.id, name="box", argument="size", value=[0.5, 0.5, 0.5]
    )
    back = apply(
        changed, op="set_value", id=board.id, name="box", argument="size", value=[1.2, 0.78, 0.015]
    )
    assert back == source
    # A value written through a named parameter comes back the same way.
    sink = index["assign:sink"]
    deeper = apply(source, op="set_value", id=sink.id, name="extrude", argument="depth", value=2.0)
    assert "fin_depth = Scalar(2, free=True" in deeper
    back = apply(deeper, op="set_value", id=sink.id, name="extrude", argument="depth", value=1.2)
    assert back == source


# ── 6: idempotence ──────────────────────────────────────────────────────────

SETTERS = (
    "set_vertex",
    "set_value",
    "assign_material",
    "set_material_property",
    "set_sketch_plane",
    "set_constraint_value",
    "solve_sketch",
    "set_study_value",
    "set_mesh_value",
    "set_optimization_value",
)


@pytest.mark.parametrize("scene", sorted(SCENES))
@pytest.mark.parametrize("op", SETTERS)
def test_a_setter_applied_to_its_own_output_changes_nothing(scene: str, op: str) -> None:
    if op in NEEDS and not _declares(SCENES[scene], NEEDS[op]):
        pytest.skip(f"{scene} declares no {NEEDS[op]}")
    rng = random.Random(f"{scene}:{op}:{SEED}")
    generator = Generator(rng)
    source = SCENES[scene]
    accepted = 0
    for _ in range(12):
        request = generator.request(op, source)
        if request is None:
            continue
        result = patch_source({"source": source, **request})
        if not result["ok"]:
            continue
        once = result["source"]
        twice = patch_source({"source": once, **request})
        assert twice["ok"] is True, twice.get("error")
        assert twice["source"] == once, f"{op} is not idempotent for {request}"
        accepted += 1
    assert accepted, f"no generated {op} request was accepted on {scene}"


# ── 7: malformed requests ───────────────────────────────────────────────────

STARTER = SCENES["starter"]
_VERTEX = {"op": "set_vertex", "id": "vertex:comb_profile[0]", "xy": [0.1, 0.2]}

MALFORMED: list[tuple[str, dict[str, Any], str]] = [
    # Stray fields — the models forbid them and so does the server.
    (
        "stray field",
        {**_VERTEX, "bogus": 1},
        "The patch operation `set_vertex` does not take `bogus`. "
        "If you updated cadjoint, restart the playground server.",
    ),
    (
        "stray fields are all named",
        {**_VERTEX, "zeta": 1, "alpha": 2},
        "The patch operation `set_vertex` does not take `alpha`, `zeta`. "
        "If you updated cadjoint, restart the playground server.",
    ),
    (
        "an id on an operation that creates",
        {"op": "add_material", "id": "assign:steel", "color": [0, 0, 0]},
        "The patch operation `add_material` creates a new object, so it takes no `id`.",
    ),
    # Unknown operation.
    (
        "unknown op",
        {"op": "teleport", "id": "assign:board"},
        "This server does not support the patch operation 'teleport'. "
        "If you updated cadjoint, restart the playground server.",
    ),
    # Wrong types.
    (
        "xy as a string",
        {**_VERTEX, "xy": "0.1, 0.2"},
        "The patch request needs `xy` as two numbers.",
    ),
    (
        "xy with a bool",
        {**_VERTEX, "xy": [True, 0.2]},
        "The patch request needs `xy` as two numbers.",
    ),
    (
        "xy of three",
        {**_VERTEX, "xy": [0.1, 0.2, 0.3]},
        "The patch request needs `xy` as two numbers.",
    ),
    (
        "line as a string",
        {"op": "set_vertex", "line": "132", "index": 0, "xy": [0, 0]},
        "The patch request needs an integer `line`.",
    ),
    (
        "index as a float",
        {"op": "set_vertex", "line": 132, "index": 1.5, "xy": [0, 0]},
        "The patch request needs an integer `index`.",
    ),
    (
        "id as a number",
        {"op": "delete_object", "id": 7},
        "The patch request needs `id` as a non-empty string.",
    ),
    (
        "kind as a list",
        {"op": "add_constraint", "id": "assign:comb_profile", "kind": ["fixed"], "indices": [0]},
        "Constraint `kind` must be one of: coincident, distance, fixed, horizontal, parallel, "
        "perpendicular, vertical.",
    ),
    (
        "indices as a string",
        {
            "op": "add_constraint",
            "id": "assign:comb_profile",
            "kind": "horizontal",
            "indices": "0,1",
        },
        "`horizontal` takes exactly 2 integer `indices`.",
    ),
    (
        "value as a string",
        {
            "op": "set_value",
            "id": "assign:board",
            "name": "box",
            "argument": "size",
            "value": "big",
        },
        "The patch request needs `value` as a number or numbers.",
    ),
    (
        "material as a number",
        {"op": "assign_material", "id": "assign:board", "material": 3},
        "The patch request needs `material` as a Python identifier.",
    ),
    (
        "study kind unknown",
        {"op": "add_study", "kind": "magnetic"},
        "Study `kind` must be `thermal` or `elastic`.",
    ),
    (
        "bc type unknown",
        {"op": "add_study_bc", "id": "assign:heat_study", "bc_type": "convection", "selection": {}},
        "`bc_type` must be one of: dirichlet, heat_flux, fixed, traction.",
    ),
    (
        "mesh method unknown",
        {"op": "set_mesh_value", "id": "assign:sink_mesh", "argument": "method", "value": "voxel"},
        "Mesh `method` must be one of: hex, tet4, tet10.",
    ),
    (
        "solver method unknown",
        {"op": "solve_sketch", "id": "assign:comb_profile", "method": "bfgs"},
        "Solver `method` must be `newton`, `adam`, or `sgd`.",
    ),
    (
        "optimization argument unknown",
        {
            "op": "set_optimization_value",
            "id": "assign:cool_sink",
            "argument": "metric",
            "value": 1,
        },
        "Optimization `argument` must be `steps` or `learning_rate`.",
    ),
    (
        "primitive kind unknown",
        {
            "op": "add_primitive",
            "kind": "torus",
            "position": [0, 0, 0],
            "dimensions": {"radius": 1},
        },
        "Primitive `kind` must be one of: box, cylinder, sphere.",
    ),
    (
        "primitive dimensions of another kind",
        {"op": "add_primitive", "kind": "box", "position": [0, 0, 0], "dimensions": {"radius": 1}},
        "A `box` takes exactly these dimensions: `size`.",
    ),
    (
        "primitive dimensions missing",
        {
            "op": "add_primitive",
            "kind": "cylinder",
            "position": [0, 0, 0],
            "dimensions": {"radius": 1},
        },
        "A `cylinder` takes exactly these dimensions: `radius`, `height`.",
    ),
    (
        "set_value on an unknown call",
        {
            "op": "set_value",
            "id": "assign:board",
            "name": "Union",
            "argument": "smoothness",
            "value": 1,
        },
        "`set_value` edits one of these calls: Material, PolygonProfile, SketchPlane, box, cylinder, extrude, loft, "
        "revolve, sphere.",
    ),
    (
        "set_value on an argument the call lacks",
        {"op": "set_value", "id": "assign:board", "name": "box", "argument": "radius", "value": 1},
        "`box` has no editable argument `radius`; expected: position, rotation, size.",
    ),
    (
        "set_value with a vector where a number goes",
        {
            "op": "set_value",
            "id": "assign:bush_a",
            "name": "cylinder",
            "argument": "radius",
            "value": [1, 2],
        },
        "`radius` needs one number.",
    ),
    (
        "set_value with a number where a vector goes",
        {"op": "set_value", "id": "assign:board", "name": "box", "argument": "size", "value": 1},
        "`size` needs 3 numbers.",
    ),
    # Out of range.
    (
        "iterations too low",
        {"op": "solve_sketch", "id": "assign:comb_profile", "iterations": 0},
        "Solver `iterations` must be an integer from 1 to 512.",
    ),
    (
        "iterations too high",
        {"op": "solve_sketch", "id": "assign:comb_profile", "iterations": 513},
        "Solver `iterations` must be an integer from 1 to 512.",
    ),
    (
        "negative constraint index",
        {"op": "delete_constraint", "line": 132, "index": -1},
        "The patch request needs a non-negative `index`.",
    ),
    (
        "negative bc index",
        {"op": "delete_study_bc", "study": 0, "bc": -1},
        "The patch request needs a non-negative `bc` index.",
    ),
    (
        "negative study index",
        {"op": "delete_study", "study": -1},
        "The patch request needs `study` as a name or a non-negative index.",
    ),
    (
        "roughness above one",
        {"op": "add_material", "color": [0, 0, 0], "roughness": 2},
        "The patch request needs `roughness` from 0 to 1.",
    ),
    (
        "color above one",
        {"op": "add_material", "color": [0, 2, 0]},
        "The patch request needs `color` as three numbers from 0 to 1.",
    ),
    (
        "density below zero",
        {"op": "set_material_property", "id": "assign:steel", "property": "density", "value": -1},
        "`density` must be a number from 1 to 25000 kg/m^3.",
    ),
    (
        "ior below one through set_value",
        {
            "op": "set_value",
            "id": "assign:steel",
            "name": "Material",
            "argument": "ior",
            "value": 0.5,
        },
        "`ior` must be a number from 1 to 3 (dimensionless).",
    ),
    (
        "a zero-size box",
        {
            "op": "set_value",
            "id": "assign:board",
            "name": "box",
            "argument": "size",
            "value": [0, 1, 1],
        },
        "`size` needs 3 positive numbers.",
    ),
    (
        "a zero radius primitive",
        {
            "op": "add_primitive",
            "kind": "sphere",
            "position": [0, 0, 0],
            "dimensions": {"radius": 0},
        },
        "Dimension `radius` must be a positive number.",
    ),
    (
        "a zero normal",
        {
            "op": "set_sketch_plane",
            "id": "assign:comb_profile",
            "reference": {"kind": "world", "origin": [0, 0, 0], "normal": [0, 0, 0]},
        },
        "A sketch-plane normal must not be zero.",
    ),
    (
        "a fixed constraint with one number",
        {
            "op": "add_constraint",
            "id": "assign:comb_profile",
            "kind": "fixed",
            "indices": [0],
            "value": 0.5,
        },
        "A `fixed` constraint needs `value` as two numbers.",
    ),
    (
        "a negative distance",
        {
            "op": "add_constraint",
            "id": "assign:comb_profile",
            "kind": "distance",
            "indices": [0, 1],
            "value": -1,
        },
        "A `distance` constraint needs `value` as a non-negative number.",
    ),
    (
        "a vertex the sketch does not have",
        {
            "op": "add_constraint",
            "id": "assign:comb_profile",
            "kind": "horizontal",
            "indices": [0, 99],
        },
        "Vertex index 99 is out of range; the sketch has 16 vertices.",
    ),
    (
        "an edge from a vertex to itself",
        {
            "op": "add_constraint",
            "id": "assign:comb_profile",
            "kind": "vertical",
            "indices": [4, 4],
        },
        "A constraint edge needs two different vertices, not 4 twice.",
    ),
    (
        "steps of zero",
        {"op": "set_optimization_value", "id": "assign:cool_sink", "argument": "steps", "value": 0},
        "`steps` must be a positive whole number.",
    ),
    (
        "a fractional resolution",
        {
            "op": "set_mesh_value",
            "id": "assign:sink_mesh",
            "argument": "resolution",
            "value": [4.5, 4, 4],
        },
        "`resolution` must be positive whole numbers.",
    ),
    # Non-finite numbers: Python's JSON decoder lets them through.
    ("nan", {**_VERTEX, "xy": [float("nan"), 0]}, "The patch request needs `xy` as two numbers."),
    (
        "inf",
        {
            "op": "set_value",
            "id": "assign:board",
            "name": "box",
            "argument": "size",
            "value": [float("inf"), 1, 1],
        },
        "The patch request needs `value` as a number or numbers.",
    ),
    (
        "an empty vector",
        {"op": "set_value", "id": "assign:board", "name": "box", "argument": "size", "value": []},
        "The patch request needs `value` as a number or numbers.",
    ),
    # Targets that do not exist or are of the wrong kind.
    (
        "an unknown id",
        {**_VERTEX, "id": "vertex:nowhere[0]"},
        "No statement in this program has the id 'vertex:nowhere[0]'.",
    ),
    (
        "an id of the wrong kind",
        {"op": "delete_mesh", "id": "assign:board"},
        "The id 'assign:board' names a primitive, which `delete_mesh` cannot address.",
    ),
    (
        "a vertex past the end",
        {"op": "set_vertex", "line": 132, "index": 16, "xy": [0, 0]},
        "Vertex index 16 is out of range for the sketch at line 132 (16 vertices).",
    ),
    (
        "a source that is not a string",
        {"op": "set_vertex", "line": 132, "index": 0, "xy": [0, 0], "source": None},
        "The patch request must contain a string `source` field.",
    ),
]


@pytest.mark.parametrize(
    ("name", "request_body", "message"), MALFORMED, ids=[case[0] for case in MALFORMED]
)
def test_a_malformed_request_is_refused_with_its_documented_message(
    name: str, request_body: dict[str, Any], message: str
) -> None:
    sent = copy.deepcopy(request_body)
    result = patch_source({"source": STARTER, **request_body})
    assert result == {"ok": False, "error": message}
    assert request_body == sent


def test_the_starter_line_the_malformed_cases_rely_on_is_still_the_comb() -> None:
    assert identity_index(STARTER)["assign:comb_profile"].line == 132
