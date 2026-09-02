# Editing operations — the spec of record

The scene is a Python file and the file is the truth. Every gesture in the
viewer is a `POST /patch` that rewrites a span of that file and nothing else;
the compile that follows is what the user sees. This document states, for
each of the 28 operations, what a request looks like, what must already be in
the source, exactly what is rewritten, what the next compile must show, and
every refusal with its message. The invariants at the end hold for all of
them and are executed by `tests/viewer/test_patch_properties.py`; the
per-operation findings that fixed the code to match this document are in
`tests/viewer/test_patch_contract.py`.

The implementation this describes: `cadjoint/viewer/_patch_requests.py` (the
gate), `cadjoint/viewer/schema/requests.py` (the wire, emitted to
`payloads.d.ts`), `cadjoint/viewer/patch/**` (the rewrites), and
`cadjoint/viewer/source_map/**` (locating things in the text; `identity.py`
gives the stable ids).

## 1. The request pipeline

Every request is `{"source", "op", ...fields}`. `patch_source` runs these
checks in this order, and the first one that fails is the answer:

| # | Check | Refusal |
| --- | --- | --- |
| 1 | `source` is a string | `The patch request must contain a string `source` field.` |
| 2 | `source` is under the size limit | `OVERSIZED_SOURCE_ERROR` (`_limits.py`) |
| 3 | `op` is a string | `The patch request must contain a string `op` field.` |
| 4 | `op` is in `OPERATIONS` | `This server does not support the patch operation '<op>'. If you updated cadjoint, restart the playground server.` |
| 5 | stable ids resolve (§2) | see §2 |
| 6 | every field is one the operation's request model names | `The patch operation `<op>` does not take `<a>`, `<b>`. If you updated cadjoint, restart the playground server.` (fields sorted) |
| 7 | the operation's validator (types, ranges, enums) | per operation |
| 8 | the operation itself (the source supports the edit) | per operation, a `PatchError` |

A refused request answers `{"ok": false, "error": <message>}` and nothing
else; an accepted one `{"ok": true, "source": <patched text>}`. The gate never
executes the program.

**Decision (stray fields).** The server used to read the fields it needed and
ignore the rest, on the argument that a browser running newer assets should
keep working against an older server. That kept it *answering* — by silently
dropping whatever the new field asked for and applying half an edit. A field
none of the request models names is the same skew the unknown-operation check
already refuses, so it is refused the same way, with the same advice. The
pydantic models (`extra="forbid"`) and the gate now agree; the pinned test in
`test_parity_schema.py` says so.

## 2. Addressing a target

Every operation that edits something existing accepts the **stable id** the
last compile published (`identity.py`), or the legacy position the payload
reported. When both are given the id wins: it is resolved against the text in
*this* request, so a line that moved since the compile cannot misdirect the
edit.

Ids are derived from the AST path and the name the program gave the thing:

| Form | Names | Resolves to |
| --- | --- | --- |
| `assign:<variable>` | any call bound to a module-level variable: sketch, primitive, feature, material, study, mesh, optimization | `line` (sketches, primitives, features, materials, planes) or the declaration's source-order `index` (study, mesh, optimization) |
| `sketch:<name>`, `box:<name>`, … , `sketch:#<n>` | an unbound call, by its literal `name=` or its ordinal among anonymous ones | `line` |
| `call:extrude@<sketch>` / `call:<kind>:<token>` | an unbound feature, keyed by the sketch it consumes | `line` |
| `plane:<sketch>` | a sketch's plane | the sketch's `line` |
| `vertex:<sketch>[<i>]` | a sketch vertex | `line` + `index` |
| `constraint:<sketch>[<i>]` | the i-th constraint statement of that sketch, in source order | the sketch's `line` + `index` |
| `bc:<study>[<i>]` | the i-th element of a study's literal `bcs` list | `study` index + `bc` |

An id is a **non-empty string** (`The patch request needs `id` as a non-empty
string.`), must exist (`No statement in this program has the id '<id>'.`),
and must be of a kind the operation can act on (`The id '<id>' names a
<kind>, which `<op>` cannot address.`). An operation that creates something
takes no id at all (`The patch operation `<op>` creates a new object, so it
takes no `id`.`); `add_loft` takes `id_a`/`id_b` (``add_loft` names its two
sketches with `id_a` and `id_b`, not `id`.`). A duplicated id — two
module-level assignments to the same variable — resolves to nothing rather
than to a guess.

What each operation's `id` may name (`_ID_TARGETS`); every entry is a kind
the operation can actually edit:

| Operation | id kinds |
| --- | --- |
| set_vertex, insert_vertex, delete_vertex | sketch, vertex |
| set_value | sketch, primitive, feature, material, plane |
| assign_material | sketch, primitive, feature |
| set_material_property | material |
| set_sketch_plane | sketch, plane |
| add_extrusion, add_revolution, add_constraint, solve_sketch | sketch |
| delete_constraint, set_constraint_value | sketch, constraint |
| delete_object | sketch, primitive, feature |
| delete_study, add_study_bc | study |
| delete_study_bc, set_study_value | study, bc |
| delete_mesh, set_mesh_value | mesh |
| delete_optimization, set_optimization_value | optimization |
| add_primitive, add_material, add_sketch, add_loft, add_study, add_mesh | none |

Legacy addressing: `line` (1-based, the line the call was constructed on; a
multi-line call is matched by its range), `index` (vertex or constraint
ordinal), `study`/`mesh`/`optimization` (source-order index among top-level
declarations, or the declared `name=`, or the bound variable), `material`
(name or index among literal `Material(...)` definitions), `bc`.

Resolution refuses ambiguity everywhere: two candidate calls on a line, two
declarations with one name, a parameter declared twice — each is an error,
never a choice.

## 3. Formatting rules

- **Viewer-generated numbers** (drag coordinates, ray hits, dimensions, the
  world plane a sketch is planted on) are written compactly: `%.4g`, with
  `|x| < 1e-9` snapped to `0`. So `[0.30000000000000004, -0.0]` is written
  `[0.3, 0]`.
- **Typed numbers** (constraint values, study/mesh/optimization arguments,
  material properties) are written exactly, `repr(float(x))`, so they round
  trip: `2.5` stays `2.5`; `68.9e9` becomes `68900000000.0`.
- `resolution` and `steps` are written as integers; `method` on a mesh and on
  a solve step as a single-quoted string literal.
- Vectors are written as lists `[a, b, c]`; a tuple the user wrote is
  replaced by a list.
- Node selections are rendered through the runtime (`selection_from_description`
  → `describe()`) so only a selection the runtime can rebuild is ever written.

**Where a new declaration goes, and why**

| Operation | Placement | Reason |
| --- | --- | --- |
| add_primitive, add_sketch, add_extrusion, add_revolution, add_loft | immediately above the `scene = …` assignment | the scene expression is extended in the same edit, and a name must be bound before the scene reads it |
| add_material | directly after the last import | a drag can assign the material to an object declared anywhere, so the name must be bound above every object |
| add_constraint | after the sketch's last existing constraint statement, else after the sketch | source order is chip order; prepending would renumber every chip |
| solve_sketch | after the sketch's constraint block, imports beside it included | the solve has to see every constraint |
| add_study | after the last study, else after `scene = …` | studies come last |
| add_mesh | after the last mesh, else before the first study, else after `scene = …` | a study can only reference a mesh declared above it |
| add_study_bc | at the end of the study's literal `bcs` list | chip order |

**Imports.** `_ensure_import` binds a symbol the edit needs. If the symbol is
already bound (any import or assignment of that name) nothing happens. Else an
existing single-line `from <module> import …` for that module is extended in
place — so no line number moves. Else a new line `from <module> import
<symbol>` is inserted: beside the edited statement for constraints, studies,
meshes and boundary conditions (`prefer_offset`, keeping everything above the
target untouched), and otherwise after the last import. A parenthesised
multi-line import is never extended; that is why the starter scene, whose
constraint import is multi-line, gains a second `from cadjoint.constraints
import …` line beside the first constraint the viewer adds. An import is never
removed by the operation's inverse.

## 4. The operations

Each section: **Request** (fields beyond `source`/`op` and the addressing of
§2; types as the models state them), **Preconditions**, **Rewrite**,
**Postconditions** (what the compile payload shows; which ids survive),
**Refusals** (validator messages first, then the operation's), and
**Interactions**.

### 4.1 set_vertex

**Request.** `id` (vertex) or `line`+`index`; `xy: [x, y]`.

**Preconditions.** The line holds exactly one `PolygonProfile(...)` whose
vertex list resolves to a literal list; every element is a two-number literal
or a name bound to `Vector2(value=[x, y], …)`; `index < len(vertices)`.

**Rewrite.** Exactly the characters of the vertex's coordinate literal —
`[x, y]` inside the list, or inside the `Vector2(value=…)` declaration it
names — become `_format_vertex(xy)`. Nothing else changes; line count is
preserved.

**Postconditions.** The sketch's vertex `index` reports `uv ≈ xy` (float32).
Every id survives; none is created. Idempotent.

**Refusals.** `The patch request needs an integer `line`.` / `…`index`.`;
`The patch request needs `xy` as two numbers.`; `No editable PolygonProfile
literal found at line N. Sketches built in a loop or from variables cannot
be edited from the viewer.`; `Vertex index i is out of range for the sketch
at line N (n vertices).`

**Interactions.** A `Vector2` shared by two sketches moves in both — the
declaration is the truth. Constraints are untouched: they name the vertex by
parameter or by position, and neither changed.

### 4.2 insert_vertex

**Request.** `id` (vertex, meaning "before this one") or `line`+`index`
(`index == len(vertices)` appends); `xy`.

**Preconditions.** As set_vertex, with `0 ≤ index ≤ len(vertices)`.

**Rewrite.** `[x, y], ` is inserted before the list element at `index`
(structurally, into the profile's own list — never inside a `Vector2`
constructor), or `, [x, y]` after the last. Then every constraint of the
sketch written as `profile.vertices[j]` with `j ≥ index` is renumbered to
`j + 1`, in place. Parameter-named constraints do not move.

**Postconditions.** Vertex count +1, vertex `index` reports `xy`, the vertex
ids after it are renumbered (`vertex:S[k]` for `k ≥ index` now name the next
point), constraints keep naming the points they were written for. Adding
twice adds twice.

**Refusals.** As set_vertex, with `Insert index i is out of range …`.

**Interactions.** Followed by `delete_vertex` at the same index the text is
restored byte for byte, constraints included.

### 4.3 delete_vertex

**Request.** `id` (vertex) or `line`+`index`.

**Preconditions.** As set_vertex, and the sketch has more than three
vertices.

**Rewrite.** The list element and its separator are cut (a middle element
takes the comma after it, the last takes the comma before). Every constraint
statement of the sketch that references the deleted vertex — by
`profile.vertices[index]` or by the parameter name that was in the list — is
deleted whole; every remaining `profile.vertices[j]` with `j > index` becomes
`j - 1`. A `Vector2` declaration the list named stays (it is a parameter
declaration, not a use).

**Postconditions.** Vertex count −1; `vertex:S[n-1]` is gone; the sketch's
constraint ordinals are renumbered where statements were removed; the program
compiles (the previous behaviour left constraints on a parameter the scene no
longer contained, which the next `satisfy_constraints(scene)` refused).

**Refusals.** As set_vertex, plus `A sketch profile needs at least 3
vertices.`

**Decision.** Cascading the deletion to the vertex's constraints is the
sensible contract — a relation on a point that no longer exists is not one
the program can keep — and renumbering positional references is required for
correctness: before this, deleting vertex 1 left `plate.vertices[2]` naming
what had been vertex 3.

### 4.4 set_value

**Request.** `id` (sketch/primitive/feature/material/plane) or `line`;
`name` (the called function: `box`, `sphere`, `cylinder`, `extrude`,
`revolve`, `loft`, `PolygonProfile`, `Material`); `argument`; `value`
(number or numbers).

The contract table `EDITABLE_CALLS` (geometry.py) decides what may be written
and its shape:

| name | arguments (component count) |
| --- | --- |
| box | position 3, rotation 3, size 3 |
| sphere | position 3, rotation 3, radius 1 |
| cylinder | position 3, rotation 3, radius 1, height 1 |
| extrude | depth 1 |
| revolve | offset 1 |
| loft | height 1 |
| PolygonProfile | planeOrigin 3, planeNormal 3 |
| SketchPlane | origin 3, normal 3 (the viewport's gizmo addresses the plane call directly when a sketch is dragged as an object; same rewrite as `planeOrigin`/`planeNormal`) |
| Material | color 3 (each in 0…1); roughness, metallic, opacity, ior, reflectivity, density, conductivity, specific_heat, youngs_modulus, poisson_ratio, thermal_expansion, yield_strength — 1 each, inside `PROPERTY_BOUNDS` |

`size`, `radius`, `height` must be positive; `planeNormal` non-zero.

**Preconditions.** The line holds exactly one call named `name` (matched by
exact line first, then by range). For an argument other than a plane
component: the keyword is absent, or present with a value that is a numeric
literal, a list of them, or a name/`Scalar`/`Vector` wrapper that resolves to
one. For `planeOrigin`/`planeNormal`: the sketch has no `plane=`, or its
`plane=` is a literal `SketchPlane(...)` call (directly, or through one name
bound to one).

**Rewrite.** The literal's span is replaced by `_format_value(value)` —
following the name to its declaration when the keyword is `depth=fin_depth`
or `position=bushing_a`, so every use of that parameter follows. An absent
keyword is appended after the call's last argument as `, <argument>=<value>`
(a solid written without `rotation=` is still rotatable). A plane component
is rewritten inside the `SketchPlane(...)`, appended to it, or — when the
sketch has no plane — a `plane=SketchPlane(<origin|normal>=…)` keyword is
added and `SketchPlane` imported.

**Postconditions.** The payload reports the new value on that node (transform
dimensions, plane origin/normal, material property). Ids survive; none is
created. Idempotent.

**Refusals.** `The patch request needs a string `name`.` / `…`argument`.`;
``set_value` edits one of these calls: Material, PolygonProfile, SketchPlane, box,
cylinder, extrude, loft, revolve, sphere.`; `The patch request needs an
integer `line`.`; `The patch request needs `value` as a number or numbers.`
(also for NaN/inf and an empty list); `A sketch-plane edit needs `value` as
three numbers.`; `A sketch-plane normal must not be zero.`; then from the
operation: ``<name>` has no editable argument `<argument>`; expected: …`;
``<argument>` needs one number.` / ``<argument>` needs 3 numbers.` /
``<argument>` needs 3 positive numbers.` / ``<argument>` needs a positive
number.` / ``color` needs 3 numbers from 0 to 1.` / the material bracket
message of §4.8; `No editable <name>() call found at line N.`; `The <name>'s
`<argument>` is not an editable literal; edit it in the code.`; `The
sketch's `plane` is an expression over other geometry, not a literal
`SketchPlane(...)`; re-plant it with `set_sketch_plane` or edit it in the
code.`

**Defects fixed.** A keyword bound to an expression (`position=[_corner,
_corner, 0.215]`) or a face-derived plane (`plane=seal_land.cap("+")…`) was
*appended* as a second keyword, which `ast.parse` accepts and `compile`
rejects — the program stopped running. Any `argument` and any `name` were
written verbatim (`Solid.box(foo=1)`). Material scalars were unbounded
through this path while bounded through set_material_property. NaN reached
the file as the bare name `nan`.

### 4.5 add_primitive

**Request.** `kind ∈ {box, sphere, cylinder}`; `position: [x, y, z]`;
`dimensions`: exactly the kind's — box `{size: [3 positive]}`, sphere
`{radius > 0}`, cylinder `{radius > 0, height > 0}` (read off the runtime's
`DIMENSIONS`).

**Preconditions.** A module-level `scene = …` assignment exists.

**Rewrite.** The scene expression is extended first (§4.18's union rule:
`, <var>` after the last positional operand of a `Union(...)`, or the whole
value wrapped as `Union(<value>, <var>)` with `Union` imported); then
`<var> = Solid.<kind>(<dims>, position=[…], name="<var>")` is inserted on
its own line above `scene = …`, and `Solid` imported. `<var>` is
`<kind>1`, `<kind>2`, … — the first not bound anywhere in the module.

**Postconditions.** One more construction node, `assign:<var>`, listed in
the scene. Every prior id survives. Adding twice adds `<kind>2`.

**Refusals.** `Primitive `kind` must be one of: box, cylinder, sphere.`;
`The patch request needs `position` as three numbers.`; `The patch request
needs a `dimensions` object.`; `A `<kind>` takes exactly these dimensions:
`size`.` (or `` `radius` ``, `` `radius`, `height` ``); `Dimension `<key>`
must be a positive number.` / `… must be 3 positive numbers.`; `Add a `scene
= ...` assignment before placing solids from the viewer.`; `The scene Union
has no operands to extend.`

**Defects fixed.** `kind` was any string (`Solid.torus` compiled to an
`AttributeError`); `dimensions` any keys (a `ValueError` on compile);
`dimensions={}` produced `Solid.box(, position=…)` and an *uncaught*
`SyntaxError` — a crashed request rather than a refusal.

### 4.6 add_material

**Request.** `color: [r, g, b]` each in 0…1; optional `roughness`,
`metallic`, `opacity`, `reflectivity` (0…1; defaults 0.4, 0, 1, 0), `ior`
(1…3; default 1.45).

**Preconditions.** A `scene = …` assignment exists.

**Rewrite.** `<var> = Material(name='<var>', color=[…], roughness=…,
metallic=…, opacity=…, ior=…, reflectivity=…)` inserted directly after the
last import (§3); `Material` imported. `<var>` is `material1`, `material2`, …

**Postconditions.** One more material, `assign:<var>`, assigned to nothing.
No inverse exists (there is no delete_material); documented as such.

**Refusals.** `The patch request needs `color` as three numbers from 0 to
1.`; `The patch request needs `<key>` from <low> to <high>.`; `Add a `scene
= ...` assignment before creating a material.`

### 4.7 assign_material

**Request.** `id` (sketch/primitive/feature) or `line`; `material`: a Python
identifier.

**Preconditions.** Exactly one module-level `<material> = Material(...)`.
The line holds exactly one construction or feature call. A sketch must be
consumed by exactly one operator (extrude, revolve or loft); the keyword goes
on that operator, because a profile carries no material.

**Rewrite.** `material=<name>` replaces the existing keyword's value, or is
appended after the call's last argument.

**Postconditions.** The node reports `material == <name>`. Idempotent;
assigning the previous name back restores the text byte for byte.

**Refusals.** `The patch request needs an integer `line`.`; `The patch
request needs `material` as a Python identifier.`; ``<material>` is not a
named Material definition.`; `No single construction call found at line
N.`; ``<profile>` needs one operator (extrude, revolve or loft) before a
material can be assigned.`; `Name the sketch before adding constraints or
operators from the viewer.`

**Defects fixed.** `_ID_TARGETS` promised a feature could be addressed but
the operation only knew primitives and sketches, and a sketch could only
be given a material through an `extrude` — a revolved or lofted sketch had no
way to a material from the viewer.

### 4.8 set_material_property

**Request.** `id` (material) or `line` or `material` (name/variable, or
index among literal definitions); `property ∈ EDITABLE_PROPERTIES`
(roughness, metallic, opacity, ior, reflectivity, density, conductivity,
specific_heat, youngs_modulus, poisson_ratio, thermal_expansion,
yield_strength); `value`: a number inside the property's bracket, or `null`;
`expand: bool` (default false).

**Preconditions.** The material is a module-level `x = Material(...)` — or,
with `expand`, a bare argument-free catalogue factory `x = copper_c11000()`,
which is first rewritten as the literal it builds (one keyword per line).
An existing keyword's value must be a numeric literal (or resolve to one).

**Rewrite.** Existing keyword: its literal's span becomes `repr(float(v))`.
Absent keyword: `, <property>=<v>` appended after the last argument on the
same line, or — if that would push the line past 100 columns — as
`,\n<indent><property>=<v>` (the one case that adds a line, by exactly one).
`value: null`: the keyword is removed with its separator; a keyword that
owns its line takes the line, and a keyword that was wrapped onto its own
line as the last argument takes the line break and the comma before it.
Removing what is absent is a no-op that answers the same text.

**Postconditions.** The material payload reports the value under `physical`
or the optical key; `spans` gains/loses the property. Idempotent; add then
remove is byte-identical; set then set-back is AST-identical.

**Refusals.** `The patch request needs an integer `line`.`; `The patch
request needs `material` as a name or a non-negative index.`; `Material
`property` must be one of: …`; the bracket: ``<property>` must be a number
from <low> to <high> <unit>.` (e.g. ``density` must be a number from 1 to
25000 kg/m^3.`, ``roughness` must be a number from 0 to 1
(dimensionless).`); `The patch request needs `expand` as a boolean.`; `No
single material definition found at line N.`; `Material index i is out of
range; the program declares n.`; `No single material named '<x>'; the
program declares: …`; ``<x>` is built by the catalogue factory `<f>()`, which
has no property keyword to edit. Convert it to a literal `Material(...)`
first — send this request again with `expand: true` to have that done for
you.`; ``<f>()` is called with arguments, so it cannot be expanded
automatically. Write the `Material(...)` you want by hand.`; ``<x>`'s
`<property>` is not an editable literal; edit it directly in the code.`;
`Cannot add a property to a material call with no arguments.`

**Defect fixed.** Removing a wrapped last keyword left `metallic=0.85,\n)`
behind — AST-equal, not the text that was there.

### 4.9 add_sketch

**Request.** `origin: [x, y, z]`.

**Preconditions.** A `scene = …` assignment exists.

**Rewrite.** `<var> = PolygonProfile([[-0.6, -0.6], [0.6, -0.6], [0.6, 0.6],
[-0.6, 0.6]], plane=SketchPlane(origin=[…]), name='<var>')` above `scene =
…`; `PolygonProfile` and `SketchPlane` imported. `<var>` is `sketch1`,
`sketch2`, … The scene is *not* extended — a sketch is not a solid until an
operator consumes it.

**Postconditions.** One more construction node with four vertices, a plane
and no operators; ids `assign:<var>`, `plane:<var>`, `vertex:<var>[0..3]`.

**Refusals.** `The patch request needs `origin` as three numbers.`; `Add a
`scene = ...` assignment before creating a sketch.`

### 4.10 set_sketch_plane

**Request.** `id` (sketch/plane) or `line`; `reference`, one of
`{kind: "world", origin: [3], normal: [3] ≠ 0}`, `{kind: "cap", owner,
sign: "+"|"-"}`, `{kind: "side", owner, edge ≥ 0}`, `{kind: "face", owner,
key}`, `{kind: "tangent", owner, near: [3]}` where `owner` is a feature or
primitive id, or its line; optional `x_axis: [3] ≠ 0`, `flip: bool` (face
references only), `offset: number`.

**Preconditions.** The line holds exactly one `PolygonProfile(...)` (bound or
not). For a reference: the owner line holds exactly one feature/primitive
call bound to a plain variable, whose statement is *above* the sketch's. The
face must be one that kind of solid declares — decided statically from the
kind, the way the runtime's `FaceSet` builds them:

| owner kind | cap | side | face keys |
| --- | --- | --- | --- |
| extrude, loft | `+`/`-` | `0 … n-1`, n = the consumed sketch's literal vertex count (unchecked when the sketch is generated) | `cap+`, `cap-`, `side<i>` |
| cylinder | `+`/`-` | none | `cap+`, `cap-` |
| box | none | none | `+x -x +y -y +z -z` |
| sphere, revolve | none | none | none (tangent only) |

**Rewrite.** The sketch's `plane=` value span becomes one of
`SketchPlane(origin=[…], normal=[…][, x_axis=[…]])`,
`SketchPlane.on(<owner>.cap('+')[, x_axis=…][, flip=True])`,
`SketchPlane.on(<owner>.side(2)…)`, `SketchPlane.on(<owner>.face('+z')…)`,
`SketchPlane.tangent(<owner>, near=[…][, x_axis=…])`, each wrapped as
`SketchPlane.offset(<plane>, d)` when `offset` is non-zero; or `,
plane=<expr>` is appended when the sketch had none. `SketchPlane` is
imported. The reference is written as *source*, never as coordinates, so
the plane follows the parent's parameters.

**Postconditions.** The sketch's plane reference reports the constructor and
owner. Idempotent. Planting a plain-plane sketch on a face and then writing
its original origin/normal back as a `world` reference restores the AST.

**Refusals.** `The patch request needs an integer `line`.`; `The patch
request needs `reference` as an object.`; `A `world` plane needs `origin`
and `normal` as three numbers.`; `A sketch-plane normal must not be
zero.`; `Plane `reference.kind` must be one of: cap, face, side, tangent,
world.`; `The plane reference needs an integer `owner` line.`; `A cap
reference needs `sign` as `+` or `-`.`; `A side reference needs a
non-negative `edge` index.`; `A face reference needs a non-empty `key`.`;
`A tangent reference needs `near` as three numbers.`; ``x_axis` must be
three numbers and must not be zero.`; `The patch request needs `flip` as a
boolean.`; `The patch request needs a numeric `offset`.`; then `No single
PolygonProfile() call found at line N.`; `No single extrude/revolve/loft or
primitive call found at line N.`; `The <kind> at line N is not assigned to a
variable, so its faces cannot be referenced. Name it first.`; ``<owner>` is
defined at line M, at or after the sketch at line N; a sketch can only sit
on geometry built before it.`; `A <kind> declares no cap faces; `<owner>`
has no `cap`.`; `A <kind> declares no side faces; `<owner>` has no
`side`.`; ``<owner>` has n sides, so `edge` must be from 0 to n-1.`; `A
<kind> has no face '<key>'; `<owner>` declares: ….`

**Interactions.** The ordering rule also stops a sketch from sitting on the
feature it feeds. A sketch planted on `body.cap("+")` is a *use* of `body`
outside any union, so `delete_object` on `body` is refused (§4.18).

**Defect fixed.** `SketchPlane.on(plate.face('+z'))` and
`seat_cut.side(0)` were written and raised `KeyError` on compile.

**Known limit (runtime, deferred).** A tangent plane whose normal is ±X
fails in `SketchPlane.tangent` with `Cannot normalize zero-length vector`:
the runtime picks world X as the default in-plane axis and does not fall
back. It cannot be known without evaluating the field at `near`; the fix
belongs in `cadjoint.construction`, not here.

### 4.11 add_extrusion / 4.12 add_revolution / 4.13 add_loft

**Request.** extrusion: `id`/`line`, `depth` (default 0.5); revolution:
`id`/`line`, `offset` (default 0); loft: `id_a`/`id_b` or `line_a`/`line_b`,
`height` (default 1).

**Preconditions.** Each sketch is a `PolygonProfile(...)` bound to a plain
variable, and feeds **no** operator yet — one sketch, one solid. A loft needs
two different sketches with equal, statically countable vertex counts. A
`scene = …` assignment exists.

**Rewrite.** The scene is extended with `<profile>_body` (or `_body2`, …;
§4.18's union rule, `Union` imported if the scene was wrapped), then
`<profile>_body = extrude(<profile>, depth=<d>)` /
`revolve(<profile>, offset=<o>)` / `loft(<a>, <b>, height=<h>)` is
inserted above `scene = …` and the operator imported. `depth`/`offset`/
`height` are viewer-formatted.

**Postconditions.** The sketch node reports one operator of that kind; a new
feature `assign:<profile>_body`; the scene lists it. Deleting the body
(§4.18) restores the text modulo the import.

**Refusals.** `The patch request needs an integer `line`.` / `… integer
`line_a` and `line_b`.`; `The patch request needs a numeric `depth`.` /
``offset`` / ``height``; `No single PolygonProfile() call found at line
N.`; `Name the sketch before adding constraints or operators from the
viewer.`; ``<profile>` already has an operator.`; `Loft needs two different
sketches.`; `Could not count the vertices of `<profile>`.`; `Loft needs
equal vertex counts; `<a>` has n and `<b>` has m.`; `Add a `scene = ...`
assignment before adding an extrusion.`; `The scene Union has no operands
to extend.`

**Defect fixed.** `add_extrusion` only looked for an existing *extrude* and
happily extruded a sketch a revolve already consumed; the three operators
now share one check and one message.

### 4.14 add_constraint

**Request.** `id` (sketch) or `line`; `kind ∈ {fixed, distance, horizontal,
vertical, coincident, parallel, perpendicular}`; `indices`: exactly 1
(fixed), 2 (distance, horizontal, vertical, coincident) or 4 (parallel,
perpendicular) non-negative integers, each below the sketch's vertex count
when that is literal, and each edge's two endpoints different; `value`:
`[x, y]` for fixed, a non-negative number for distance, ignored otherwise.

**Preconditions.** A named sketch.

**Rewrite.** One statement — `FixedConstraint(<p>.vertices[i], [x, y])`,
`DistanceConstraint(<p>.vertices[i], <p>.vertices[j], d)`,
`HorizontalConstraint(…)`, `VerticalConstraint(…)`,
`CoincidentConstraint(…)`, `ParallelEdgesConstraint(a, b, c, d)`,
`PerpendicularEdgesConstraint(a, b, c, d)` — inserted after the sketch's
last constraint statement (else after the sketch), its class imported beside
it.

**Postconditions.** The sketch's last constraint reports `kind` and
`vertices == indices`; id `constraint:<sketch>[n]`. Adding twice adds twice.

**Refusals.** `The patch request needs an integer `line`.`; `Constraint
`kind` must be one of: coincident, distance, fixed, horizontal, parallel,
perpendicular, vertical.`; ``<kind>` takes exactly <n> integer `indices`.`;
`A `fixed` constraint needs `value` as two numbers.`; `A `distance`
constraint needs `value` as a non-negative number.`; `Constraint `indices`
must be non-negative integers.`; `Vertex index i is out of range; the
sketch has n vertices.`; `A constraint edge needs two different vertices,
not i twice.`; `Name the sketch before adding constraints or operators from
the viewer.`

**Defects fixed.** `indices` were unchecked (`plate.vertices[99]` → an
`IndexError` on compile); `fixed` accepted a scalar (`FixedConstraint target
shape () does not match parameter shape (2,)`); a list-valued `kind`
crashed the validator with `TypeError` instead of answering.

### 4.15 delete_constraint

**Request.** `id` (constraint) or `line`+`index ≥ 0`.

**Preconditions.** A named sketch; `index` below its constraint count (the
ordinal among top-level statements that name the sketch's vertices, in
source order — the same the payload publishes).

**Rewrite.** The statement's lines are removed. Its import stays.

**Postconditions.** Constraint count −1; `constraint:<sketch>[n-1]` gone;
later ordinals shift down.

**Refusals.** `The patch request needs an integer `line`.`; `The patch
request needs a non-negative `index`.`; `Name the sketch before editing
constraints from the viewer.`; `Constraint index i is out of range; the
sketch has n.`

### 4.16 set_constraint_value

**Request.** `id`/`line`+`index`; `value`: `[x, y]` for a fixed
constraint, a non-negative number for a distance.

**Preconditions.** The constraint is `fixed` or `distance` and its value
argument is a numeric literal or resolves to one (a `Scalar` name is
followed to its declaration, so `DistanceConstraint(a, b, base_width)`
rewrites `base_width = Scalar(1.8, …)`).

**Rewrite.** The literal's span becomes the exact repr.

**Postconditions.** The constraint reports `value`. Idempotent; set-back is
AST-identical.

**Refusals.** As delete_constraint, plus `The constraint needs a numeric
`value`.` (validator), `Only `fixed` and `distance` constraints carry an
editable value.`, `A `fixed` constraint needs `value` as two numbers.`, `A
`distance` constraint needs `value` as a non-negative number.`, `The
constraint statement has no value argument to rewrite.`, `The constraint
value is not an editable literal.`

### 4.17 solve_sketch

**Request.** `id`/`line`; `method ∈ {newton, adam, sgd}` (default newton);
`iterations` 1…512 (default 8).

**Preconditions.** A named sketch.

**Rewrite.** If a `satisfy_constraints(<profile>, …)` call exists its
`method=` and `steps=` keywords are rewritten or appended; else
`satisfy_constraints(<profile>, method='<m>', steps=<n>)` is inserted after
the sketch's constraint block (constraint statements and the imports beside
them) and `satisfy_constraints` imported.

**Postconditions.** Exactly one solve call for the sketch. Idempotent.

**Refusals.** `The patch request needs an integer `line`.`; `Solver
`method` must be `newton`, `adam`, or `sgd`.`; `Solver `iterations` must
be an integer from 1 to 512.`; `Name the sketch …`.

### 4.18 delete_object

**Request.** `id` (sketch/primitive/feature) or `line`.

**Preconditions.** The line holds exactly one `PolygonProfile`, `box`,
`sphere`, `cylinder`, `extrude`, `revolve` or `loft` call. Either it is the
value of a module-level `<var> = …` whose every other load of `<var>` is a
positional operand of some module-level `<name> = Union(...)` (the scene's
or a sub-assembly's — the **union rule**), and no such union has it as its
*only* operand; or it is itself a direct positional operand of the scene
Union (built inline), and not the only one.

**Rewrite.** All in one pass, back to front: each union operand removed with
its separator (a middle operand takes the comma after it, the last operand
the comma before it, so `Union(a, b)` → `Union(a)`); every top-level
constraint statement referencing a parameter that only this object's
keywords named (`position=bolt_left`) removed — the parameter would no
longer be in the scene tree and `satisfy_constraints(scene)` would refuse
it; and the object's own statement removed. An inline operand loses only
its argument.

**Postconditions.** Construction count −1; `assign:<var>` and its children
(`plane:`, `vertex:`, `constraint:`) gone; nothing else. A feature's sketch
stays as a bare sketch.

**Refusals.** `The patch request needs an integer `line`.`; `No single
construction call found at line N.`; `Could not find the statement that
builds this object.`; ``<var>` is used elsewhere in the program, so it
cannot be deleted from the viewer. Remove those uses first.`; ``<var>` is
the last operand of `<name> = Union(...)`, so deleting it would leave an
empty union. Remove that union in the code first.`; `This object is not a
direct operand of the scene Union, so it cannot be deleted from the
viewer.`; `This object is the last operand of the scene Union, so deleting
it would leave an empty scene. Remove the union in the code first.`

**Interactions.** A sketch consumed by an operator, a solid a `Difference`
subtracts, a feature `extract_parameters(sink)` reads, a body a sketch is
planted on — all "used elsewhere". `Union` is recognised bare or qualified
(`boolean.Union`).

**Defects fixed.** Features were promised by `_ID_TARGETS` and refused by
the operation with `No single construction call found` — deleting
`plate_body` after extruding a sketch was impossible. Deleting the last
operand left `Union(smoothness=0.05)`. Deleting a trailing operand left
`Union(a, )`.

### 4.19 add_study

**Request.** `kind ∈ {thermal, elastic}`; optional `name` (non-empty; not
already a study's name).

**Rewrite.** `<var> = ThermalStudy(name='<n>', resolution=20,
conductivity=1.0, bcs=[])` or `ElasticStudy(name='<n>', resolution=20,
youngs=200.0, poisson=0.3, bcs=[])` after the last study, else after
`scene = …`; the class imported beside it. `<var>` = `study1`, …; `<n>` =
`name` or `<var>`.

**Postconditions.** Studies +1, `assign:<var>`, no BCs. Deleting it restores
the text modulo the import.

**Refusals.** `Study `kind` must be `thermal` or `elastic`.`; `Study `name`
must be a non-empty string.`; `A study named '<n>' already exists.`; `Add a
`scene = ...` assignment before declaring studies from the viewer.`

### 4.20 delete_study

**Request.** `id` (study) or `study` (index/name/variable).

**Preconditions.** The study's variable is loaded nowhere else, and its
literal `name` is not any call's `study="…"` argument.

**Rewrite.** The statement's lines are removed; its BCs go with it.

**Postconditions.** Studies −1; `assign:<var>` and `bc:<var>[*]` gone.

**Refusals.** `The patch request needs `study` as a name or a non-negative
index.`; `Study index i is out of range; the program declares n.`; `No
single study named '<x>'; the program declares: ….`; ``<var>` is used
elsewhere in the program, so it cannot be deleted from the viewer. Remove
those uses first.`; `Study '<name>' is referenced by an optimization, so it
cannot be deleted from the viewer. Point the optimization at another study
first.`

**Interactions.** Both reference forms an `Optimization` can use are
guarded: `study=pry_study` (variable) and `study="sink-conduction"`
(literal). The mirror image — a mesh named by `mesh="…"` — is §4.25.

### 4.21 add_study_bc

**Request.** `id`/`study`; `bc_type ∈ {dirichlet, heat_flux, fixed,
traction}`; `selection` (a serialised node selection: `box`, `sphere`,
`halfspace`, `side`, `and`/`or`/`not` — never `predicate`); `value`: absent
for fixed, three numbers for traction, a number otherwise.

**Preconditions.** The BC type fits the study's kind (thermal: dirichlet,
heat_flux; elastic: fixed, traction). The study's `bcs=` is absent or a
literal list.

**Rewrite.** `Dirichlet(Nodes.…, value=v)`, `HeatFlux(…, flux=v)`,
`Fixed(…)`, `Traction(…, vector=[…])` appended after the last element (`,
<bc>`), placed inside an empty list, or added as `bcs=[<bc>]`; the class and
`Nodes` imported beside the study.

**Postconditions.** The study's last BC reports `type`; id
`bc:<study>[n]`. Deleting it restores the text modulo imports.

**Refusals.** ``bc_type` must be one of: dirichlet, heat_flux, fixed,
traction.`; `The patch request needs `selection` as a description
object.`; `A `fixed` boundary condition takes no value.`; `A `traction`
boundary condition needs `value` as three numbers.`; `A `<type>` boundary
condition needs a numeric `value`.`; `A <kind> study accepts <types>
boundary conditions, not `<type>`.`; `Invalid node selection: …`; `The
study's `bcs` argument is not an editable literal list.`

### 4.22 delete_study_bc

**Request.** `id` (bc) or `study`+`bc ≥ 0`.

**Preconditions.** A literal `bcs` list; `bc` in range; the element is not
built on `Nodes.predicate`.

**Rewrite.** The element and its separator are cut; the only element leaves
`bcs=[]` (the list span is rewritten, so a multi-line list keeps no stray
comma).

**Postconditions.** BC count −1; `bc:<study>[n-1]` gone; later ordinals
shift.

**Refusals.** `The patch request needs a non-negative `bc` index.`; `The
study's `bcs` argument is not an editable literal list.`;
`Boundary-condition index i is out of range; the study has n.`; `This
boundary condition uses a `Nodes.predicate` selection, which is not
serializable; edit it directly in the code.`

**Defect fixed.** Deleting the last BC of a multi-line list wrote
`bcs=[\n        ,\n    ]` — invalid Python, caught by `_validate` as `Patched
source is not valid Python` instead of doing the edit.

### 4.23 set_study_value

**Request.** `id` (study or bc) / `study`; exactly one of `bc ≥ 0` or
`argument`; `value`.

- `bc`: the BC's value — a number for Dirichlet/HeatFlux, three for
  Traction; Fixed has none. The literal (keyword or second positional) is
  rewritten exactly.
- `argument` ∈ thermal `resolution, conductivity, source, bounds, size` /
  elastic `resolution, youngs, poisson, bounds, size`, plus `mesh` and
  `domain`: the keyword (or positional slot, by the constructor's field
  order) is rewritten exactly, or appended. `resolution` is one or three
  positive integers; `bounds`/`size` three numbers, and they are stated
  together or not at all.
- `argument: "mesh"`, `value`: a declared SimMesh's name (written as a
  string literal) or variable (bare name); the study's `resolution`,
  `bounds`, `size`, `domain` are removed in the same edit — meshing intent
  lives on the SimMesh.
- `argument: "domain"`, `value`: the variable of an object assigned above
  the study; refused when the study has `mesh=`.
- `resolution`/`bounds`/`size` on a study with `mesh=` are refused — the
  runtime refuses a study that states both.

**Refusals.** `The patch request needs exactly one of `bc` or
`argument`.`; `The patch request needs a non-negative `bc` index.`; `The
patch request needs a string `argument`.`; `The patch request needs `value`
as a `mesh` name.` / `… `domain` name.`; `The patch request needs `value`
as a number or numbers.`; `A <kind> study's editable arguments are: …,
mesh, domain.`; `This study solves on a SimMesh; set the mesh's
`<argument>` instead (set_mesh_value).`; ``bounds` and `size` are stated
together or not at all; this study states neither, so add both in the code
first.`; ``resolution` must be an integer or three integers.` /
``resolution` must be positive whole numbers.` / ``<argument>` must be
three numbers.` / ``<argument>` must be a number.`; `The study's
`<argument>` value is not an editable literal.`; `This boundary condition
is not an editable constructor call.`; `A `Fixed` boundary condition has no
value to edit.`; `A `Traction` boundary condition needs `value` as three
numbers.`; `A `<class>` boundary condition needs a numeric `value`.`; `The
boundary condition has no value argument to rewrite.`; `The
boundary-condition value is not an editable literal.`; ``mesh` needs the
name of a declared SimMesh.`; `No single SimMesh named '<x>'; the program
declares: ….`; `This study solves on a SimMesh; set the mesh's `domain`
instead (set_mesh_value).`; ``domain` needs the variable name of a named
scene object.`; ``<x>` is not assigned before the study; `domain` must name
an existing construction object.`

**Defects fixed.** `resolution=10` was added to a mesh-backed study (the
runtime: `Thermal study got mesh= and resolution`); `bounds` alone to a
study with neither (`bounds and size must be given together`).

### 4.24 add_mesh

**Request.** optional `name`.

**Rewrite.** `<var> = SimMesh(name='<n>', resolution=20)` after the last
mesh, else before the first study, else after `scene = …`; `SimMesh`
imported beside it.

**Refusals.** `Mesh `name` must be a non-empty string.`; `A mesh named '<n>'
already exists.`; `Add a `scene = ...` assignment before declaring meshes
from the viewer.`

### 4.25 delete_mesh

As delete_study, guarding both reference forms: ``<var>` is used elsewhere
…` and `Mesh '<name>' is referenced by a study, so it cannot be deleted
from the viewer. Point the study at another mesh first.`

### 4.26 set_mesh_value

**Request.** `id`/`mesh`; `argument` ∈ `resolution, bounds, size, padding,
domain, method`; `value`.

- numeric arguments as §4.23 (`padding ≥ 0`, exact repr); `bounds`/`size`
  are a pair.
- `domain`: the variable of an object assigned above the mesh; the existing
  `domain=` value span is replaced whatever expression it holds (a domain
  is a name, not a literal), or the keyword is appended.
- `method ∈ {hex, tet4, tet10}` written as `'<m>'`.

**Refusals.** `The patch request needs `mesh` as a name or a non-negative
index.`; `The patch request needs a string `argument`.`; `The patch request
needs `value` as a `domain` name.`; `Mesh `method` must be one of: hex,
tet4, tet10.`; `The patch request needs `value` as a number or numbers.`;
`A mesh's editable arguments are: resolution, bounds, size, padding,
domain, method.`; ``padding` must be a non-negative number.`; ``bounds` and
`size` are stated together or not at all; this mesh states neither, so add
both in the code first.`; the `resolution`/`bounds`/`size` shape messages;
`The mesh's `<argument>` value is not an editable literal.`; ``domain`
needs the variable name of a named scene object.`; ``<x>` is not assigned
before the mesh; `domain` must name an existing construction object.`

**Defect fixed.** A mesh that already had `domain=thermal_body` could not
be pointed at anything else: the rewrite went through the literal-only
path and answered `not an editable literal`. Every shipped scene has a
domain, so the row in the mesh card was dead.

### 4.27 delete_optimization

**Request.** `id`/`optimization`. Refuses while the variable is loaded
elsewhere. There is no `add_optimization` — an objective is code — so this
has no inverse.

### 4.28 set_optimization_value

**Request.** `id`/`optimization`; `argument ∈ {steps, learning_rate}`;
`value`: a positive whole number for steps, a positive number for
learning_rate. Rewrites the keyword exactly (or appends it).

**Refusals.** `The patch request needs `optimization` as a name or a
non-negative index.`; `Optimization `argument` must be `steps` or
`learning_rate`.`; `The patch request needs a numeric `value`.`; ``steps`
must be a positive whole number.`; ``learning_rate` must be a positive
number.`; `The optimization's `<argument>` value is not an editable
literal.`

## 5. Interactions, stated once

- **The union rule.** Any module-level `<name> = Union(...)` — the scene's or
  a sub-assembly's, bare or qualified — is a place an object's name may be
  dropped from. Any other use (an operator's operand, a `Difference`, an
  `extract_parameters`, a face reference, a bare `probe = a`) is "used
  elsewhere" and refuses the deletion. No union is ever emptied. Adding goes
  through the *scene's* union only.
- **Deleting a study an Optimization names.** Refused, in both spellings
  (`study=pry_study`, `study="sink-conduction"`). Delete or repoint the
  optimization first; then the study deletes.
- **Deleting an object a face-derived sketch plane references.**
  `pad_profile`'s plane is `seal_land.cap("+").plane(...)`: that is a load of
  `seal_land` outside any union, so `delete_object` on `seal_land` is refused
  as used elsewhere. The sketch keeps its parent.
- **Renaming.** There is no rename operation. Ids are derived from the
  variable name, so renaming in the editor changes the id; the next compile
  publishes the new one and a request carrying the old id is refused with
  `No statement in this program has the id …`. Renaming does not move the
  legacy `line`, which is why both are sent.
- **Ordinal ids.** `vertex:S[i]`, `constraint:S[i]`, `bc:T[i]` and `#n`
  fallbacks are positions inside their owner. Inserting or deleting before
  them moves them — by design, and the only ids an operation may take away
  (§6, invariant 3).
- **A shared `Vector2`.** Two sketches naming one parameter move together
  under set_vertex; deleting the vertex from one sketch removes only that
  sketch's use and the constraints naming the parameter.

## 6. Invariants — every operation, every scene

Executed by `tests/viewer/test_patch_properties.py` over `scenes/starter.py`,
`scenes/end_cap.py` and `scenes/bracket.py` with a seeded generator (no
`hypothesis` in the environment); counts from the current run.

1. **Compile.** An accepted operation on a compiling program yields a
   compiling program. Checked by executing every accepted step of every
   sequence the way the compile worker does — 6 sequences (3 scenes × 2
   seeds) of ~40 generated requests, each covering all 28 operations at
   least once.
2. **Refusals are pure.** A refused request answers only `ok`/`error` and
   leaves the request object untouched; the operation is a pure function of
   the text, so nothing else can change.
3. **Identity.** Every stable id an operation did not target survives it; a
   value-setting operation creates no id. The only ids an operation may
   remove are the target and its children, or the ordinal family it
   renumbers (a sketch's vertices and constraints, a study's BCs, a sketch's
   constraint ordinals).
4. **Span discipline.** An operation changes only statements of the kinds
   its budget allows, and at most as many: the edited declaration, a
   parameter it followed a name to, the scene assignment it extends, the
   imports it adds to, the constraints a vertex deletion cascades to.
   Checked at statement level (AST dump multisets, numbers by value).
5. **Inverses.** 18 operations have an inverse and it restores the program:
   add/delete pairs (primitive, sketch, extrusion, revolution, loft,
   constraint, study, mesh, boundary condition, inserted vertex) byte for
   byte modulo an import that stays extended; set/set-back pairs (vertex,
   value, material, material property, sketch plane, constraint value,
   study/mesh/optimization values) byte for byte where the formatter agrees
   with what the user wrote, else AST-identical. The 10 without one say why
   (`NO_INVERSE`).
6. **Idempotence.** The 10 state-setting operations applied to their own
   output change nothing (12 generated requests each, per scene).
7. **Malformed requests** — 51 cases of stray fields, wrong types, wrong
   arity, values out of range, non-finite numbers, empty vectors, unknown
   or mis-kinded ids — are refused with exactly the documented message.

The formatter round trip stated in the brief — "source round-trips unchanged
when nothing else changes" — is invariant 6 for setters and, for the text
itself, the fact that no operation touches a span outside its target
(invariant 4): a scene that is not edited is not reformatted.

## 7. Where the code and the contract disagreed

| Operation | Finding | Status |
| --- | --- | --- |
| set_value | second keyword appended for an expression-valued argument or a face-derived plane → program stops compiling | fixed: refused with a message naming the fix |
| set_value | any `name`/`argument`/shape written verbatim; Material scalars unbounded; NaN/inf/`[]` accepted | fixed: `EDITABLE_CALLS` contract, bounds, finite non-empty numbers |
| add_primitive | unknown `kind`, foreign/missing dimensions → non-compiling program; `{}` → uncaught `SyntaxError` | fixed: kind and dimensions validated against the runtime's table |
| add_constraint | indices unchecked (IndexError on compile); `fixed` took a scalar; list-valued `kind` crashed | fixed: range, distinct edge endpoints, value shape, hashability |
| set_constraint_value | no shape check | fixed: same shapes as add_constraint |
| delete_vertex / insert_vertex | positional constraint references not renumbered; constraints on a deleted parameter left dangling (`Parameter values must include …`) | fixed: renumber and cascade |
| delete_object | features promised by `_ID_TARGETS`, refused by the operation; last union operand emptied the union; trailing operand left `Union(a, )` | fixed |
| assign_material | features refused; only `extrude` accepted as a sketch's operator | fixed: any single operator, or the feature itself |
| add_extrusion | did not see a revolve/loft already consuming the sketch | fixed: one check for the three operators, one message |
| set_mesh_value(domain) | a mesh with a domain could not change it | fixed: the value span is replaced whatever it holds |
| set_study_value | `resolution`/`bounds`/`size` added to a mesh-backed study; `bounds` without `size` | fixed: refused |
| set_mesh_value | `bounds` without `size` | fixed: refused |
| delete_study_bc | last element of a multi-line list left a stray comma | fixed: `bcs=[]` |
| set_material_property | removing a wrapped last keyword left `,\n)` | fixed: byte-identical |
| set_sketch_plane | face keys not checked against the owner's kind (`extrude.face('+z')`, `revolve.side(0)`) | fixed: static face vocabulary |
| patch_source | stray fields silently ignored, against the models | changed: refused; pinned test updated (§1) |
| SketchPlane.tangent | ±X normal → `Cannot normalize zero-length vector` at runtime | deferred: `cadjoint.construction` picks world X as the default in-plane axis; not knowable statically |
| schema models | `AddPrimitiveRequest.kind` and `SetValueRequest.name` are `str`, narrower on the server | deferred: narrowing them changes `payloads.d.ts` and the frontend's types; the validator is the gate |
| _ensure_import | a parenthesised multi-line import is never extended, so a new line appears beside the statement | by design (line stability for legacy addressing); documented |

Messages changed on the way, all previously unpinned except where noted:
``<profile>` already has an extrusion.` and `… extrusion or revolution.` →
``<profile>` already has an operator.`; ``<profile>` needs one named
extrusion before a material can be assigned.` → `… needs one operator
(extrude, revolve or loft) …`; `The constraint needs a numeric `value`.` on
add_constraint → the two shape messages of §4.14 (it stays for
set_constraint_value's validator); the stray-field behaviour of §1 (pinned
in `test_parity_schema.py`, updated with the reasoning).
