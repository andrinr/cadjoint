"""The static demo's bundle: what a page with no Python server reads.

The public demo (``frontend/static.html``) renders, edits and exports the
shipped scenes in the browser alone.  It cannot run a scene, so everything a
compile would have told it is written at build time by
:mod:`cadjoint.viewer.demo`, one :class:`DemoScene` per scene, and checked
against these models on the way out.  The frontend reads the TypeScript
emitted from them (``payloads.d.ts``), like every other payload.

A demo scene is a slice of the compile payload — the graph the browser
builds its WGSL from, and the construction data the overlay draws — plus the
one thing a compile never had to say: which free parameters the page may
move on its own.  A parameter some constraint holds needs the constraint
solver to move, and the solver is Python; such a parameter is listed with
the constraint that holds it and stays at its shipped value.

The construction nodes are the compile payload's with their ``faces`` left
empty: faces are what the sketch-on-face tool picks from, and the demo has
no such tool.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from cadjoint.viewer.schema.payloads import (
    ConstructionElement,
    ConstructionNode,
    ConstructionRelation,
    SceneGraph,
    SceneTree,
    Strict,
)

DEMO_FORMAT = "cadjoint-demo"
"""The ``format`` every demo scene file carries."""

DEMO_VERSION = 1
"""The version :mod:`cadjoint.viewer.demo` writes and the page reads."""


class DemoParameter(Strict):
    """One free design parameter, as the demo's parameter panel offers it.

    ``name``, ``components`` and ``value`` are the shader slot's
    (:class:`~cadjoint.viewer.schema.payloads.ShaderParameter`), so a value
    the panel sets is a buffer write under the same name.  ``lower`` and
    ``upper`` bound each component; ``bounded`` says whether they are the
    parameter's declared ``bounds`` or a range the exporter chose around the
    shipped value.
    """

    name: str
    components: int = Field(ge=1, le=4)
    value: list[float]
    lower: list[float]
    upper: list[float]
    step: float = Field(gt=0)
    bounded: bool
    # The unit the source states, when it states one.  cadjoint parameters
    # carry none today, so this is null for every shipped scene.
    unit: str | None = None
    # The construction object whose handle moves this value, when one does.
    owner: str | None
    # The constraint holding this parameter, written out (``"DistanceConstraint
    # (rib_web_corner, rib_tip, rib_height)"``); ``None`` when nothing holds it
    # and the page may move it.
    drivenBy: str | None


class DemoScene(Strict):
    """Everything the demo needs about one shipped scene."""

    format: Literal["cadjoint-demo"]
    version: Literal[1]
    name: str
    title: str
    description: str
    scene_graph: SceneGraph
    parameters: list[DemoParameter]
    construction: list[ConstructionNode]
    elements: list[ConstructionElement]
    scene_tree: SceneTree | None
    relations: list[ConstructionRelation]
