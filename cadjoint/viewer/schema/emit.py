"""Turn the pydantic models into the TypeScript the frontend imports.

The emitter is deliberately small and offline. ``json-schema-to-typescript``
would be a second toolchain to install and a network fetch to run in CI, and
the schemas here use a narrow slice of JSON Schema — objects, arrays, tuples,
unions, literals, and ``Record``-shaped ``additionalProperties``. Emitting
that slice directly is about a hundred lines and has no way to be
unavailable.

The output is deterministic: models are emitted in a fixed order and each
one's properties in declaration order, so
:func:`typescript_source` regenerated on an unchanged model set is
byte-identical to :data:`TYPESCRIPT_PATH`. A test asserts exactly that,
which is what stops the two sides of the contract from drifting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic.json_schema import models_json_schema

from cadjoint.viewer.schema import payloads, requests

TYPESCRIPT_PATH = Path(__file__).with_name("payloads.d.ts")
"""The generated file the frontend imports."""

_HEADER = """\
/**
 * Generated from the pydantic models in cadjoint/viewer/schema — do not edit.
 *
 * Regenerate with:
 *
 *     python -m cadjoint.viewer.schema.emit
 *
 * The compile worker validates every payload it sends against those same
 * models, so a type here is a guarantee, not a hope. `tests/viewer/
 * test_parity_schema.py` fails when this file and the models disagree.
 */
"""

#: Emitted in this order, so the file is stable and reads top-down.
_MODELS: tuple[type[BaseModel], ...] = (
    payloads.CompilePayload,
    payloads.WorkerFailure,
    payloads.IdentityEntry,
    payloads.ConstructionNode,
    payloads.ConstructionPlane,
    payloads.PlaneReference,
    payloads.ConstructionFace,
    payloads.FaceAccessor,
    payloads.FaceOwner,
    payloads.ConstructionVertex,
    payloads.ConstructionConstraint,
    payloads.ConstructionRelation,
    payloads.ConstructionOperator,
    payloads.ConstructionTransform,
    payloads.ParameterBinding,
    payloads.ConstraintSolverRun,
    payloads.MaterialDefinition,
    payloads.StudyPayload,
    payloads.StudyBc,
    payloads.StudySelection,
    payloads.DomainEntry,
    payloads.SimMeshPayload,
    payloads.OptimizationPayload,
    payloads.MeshEdgePayload,
    requests.PatchResponse,
    requests.ExportRequest,
    *requests.PATCH_REQUEST_MODELS.values(),
    requests.WorldPlaneReference,
    requests.CapPlaneReference,
    requests.SidePlaneReference,
    requests.FacePlaneReference,
    requests.TangentPlaneReference,
)


def _schema_document() -> dict[str, Any]:
    """One JSON Schema document holding every model as a named definition."""
    _, document = models_json_schema(
        [(model, "validation") for model in _MODELS],
        ref_template="#/$defs/{model}",
    )
    return document.get("$defs", {})


def _quoted(value: Any) -> str:
    """A JSON Schema constant as a TypeScript literal type."""
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def _tuple_type(schema: dict[str, Any]) -> str | None:
    """A fixed-length array — ``[number, number]`` — or None."""
    items = schema.get("prefixItems")
    if items:
        return "[" + ", ".join(_type(item) for item in items) + "]"
    low, high = schema.get("minItems"), schema.get("maxItems")
    if low is not None and low == high:
        inner = _type(schema.get("items", {}))
        return "[" + ", ".join([inner] * low) + "]"
    return None


def _union(options: list[dict[str, Any]]) -> str:
    """A TypeScript union, with duplicates collapsed and order preserved."""
    seen: list[str] = []
    for option in options:
        rendered = _type(option)
        if rendered not in seen:
            seen.append(rendered)
    return " | ".join(seen) if seen else "unknown"


def _type(schema: dict[str, Any]) -> str:
    """One JSON Schema node as a TypeScript type expression."""
    if "$ref" in schema:
        return schema["$ref"].rsplit("/", 1)[-1]
    if "const" in schema:
        return _quoted(schema["const"])
    if "enum" in schema:
        return " | ".join(_quoted(value) for value in schema["enum"])
    for key in ("anyOf", "oneOf"):
        if key in schema:
            return _union(schema[key])
    if "allOf" in schema and len(schema["allOf"]) == 1:
        return _type(schema["allOf"][0])

    kind = schema.get("type")
    if kind == "array":
        fixed = _tuple_type(schema)
        if fixed is not None:
            return fixed
        item = _type(schema.get("items", {}))
        return f"{item}[]" if item.isidentifier() or item.endswith("]") else f"({item})[]"
    if kind == "object":
        extra = schema.get("additionalProperties")
        if isinstance(extra, dict):
            return f"Record<string, {_type(extra)}>"
        if schema.get("properties"):
            body = ", ".join(
                f"{name}: {_type(value)}" for name, value in schema["properties"].items()
            )
            return "{ " + body + " }"
        return "Record<string, unknown>"
    return {
        "string": "string",
        "integer": "number",
        "number": "number",
        "boolean": "boolean",
        "null": "null",
    }.get(kind or "", "unknown")


def _documentation(schema: dict[str, Any], indent: str) -> str:
    """A model's or property's description as a JSDoc block, or nothing."""
    text = (schema.get("description") or "").strip()
    if not text:
        return ""
    lines = [line.rstrip() for line in text.splitlines()]
    if len(lines) == 1:
        return f"{indent}/** {lines[0]} */\n"
    body = "".join(f"{indent} *{(' ' + line).rstrip()}\n" for line in lines)
    return f"{indent}/**\n{body}{indent} */\n"


def _interface(name: str, schema: dict[str, Any]) -> str:
    """One named object schema as an exported TypeScript interface."""
    required = set(schema.get("required", []))
    lines = [_documentation(schema, ""), f"export interface {name} {{\n"]
    for field, definition in (schema.get("properties") or {}).items():
        lines.append(_documentation(definition, "  "))
        optional = "" if field in required else "?"
        lines.append(f"  {field}{optional}: {_type(definition)};\n")
    if schema.get("additionalProperties") is True:
        lines.append("  /** Fields the object's own describe() may add. */\n")
        lines.append("  [key: string]: unknown;\n")
    lines.append("}\n")
    return "".join(lines)


def _alias(name: str, schema: dict[str, Any]) -> str:
    """One named non-object schema as an exported type alias."""
    return f"{_documentation(schema, '')}export type {name} = {_type(schema)};\n"


def typescript_source() -> str:
    """The whole generated module, ready to write to :data:`TYPESCRIPT_PATH`.

    Returns:
        TypeScript declaring one interface per model, a discriminated union
        of every patch request, and the union of every compile response.
    """
    definitions = _schema_document()
    ordered = [model.__name__ for model in _MODELS]
    chunks = [_HEADER]
    for name in ordered:
        schema = definitions.get(name)
        if schema is None:  # pragma: no cover - every model is registered
            continue
        chunks.append(
            _interface(name, schema) if schema.get("type") == "object" else _alias(name, schema)
        )
    # Anything pydantic pulled in that is not one of the listed models —
    # ``Value``-style aliases, nested enums — still has to be declared.
    for name, schema in definitions.items():
        if name in ordered:
            continue
        chunks.append(
            _interface(name, schema) if schema.get("type") == "object" else _alias(name, schema)
        )
    members = "".join(f"\n  | {model.__name__}" for model in requests.PATCH_REQUEST_MODELS.values())
    chunks.append(
        "/** Every accepted `/patch` request, discriminated on `op`. */\n"
        f"export type PatchRequest ={members};\n"
    )
    chunks.append(
        "/** The operation names the server accepts. */\n"
        'export type PatchOperation = PatchRequest["op"];\n'
    )
    chunks.append(
        "/** The plane a `set_sketch_plane` request plants a sketch on. */\n"
        "export type SketchPlaneReference =\n"
        "  | WorldPlaneReference\n"
        "  | CapPlaneReference\n"
        "  | SidePlaneReference\n"
        "  | FacePlaneReference\n"
        "  | TangentPlaneReference;\n"
    )
    chunks.append(
        '/** What `mode: "compile"` answers with. */\n'
        "export type CompileResponse = CompilePayload | WorkerFailure;\n"
    )
    return "\n".join(chunk for chunk in chunks if chunk)


def write_typescript(path: Path | None = None) -> Path:
    """Write :func:`typescript_source` to disk and return where it went."""
    destination = path or TYPESCRIPT_PATH
    destination.write_text(typescript_source())
    return destination


if __name__ == "__main__":  # pragma: no cover - a developer command
    print(write_typescript())
