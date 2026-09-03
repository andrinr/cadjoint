"""Simulation results as first-class, re-inspectable objects.

:meth:`~cadjoint.fem.study.ThermalStudy.solve` returns a
:class:`SimulationResult`: the study's name and kind, the display field
name, the solved field arrays, and a reference back to the
:class:`~cadjoint.fem.simmesh.SimMesh` it ran on.  The instance is also
stored on the study as ``last_result`` so a program (or the viewer) can
re-inspect a solve without repeating it.

A result also carries what the scene's *materials* make computable: the
domain's :attr:`~SimulationResult.mass` (whenever the materials state a
density) and, for elastic results whose materials state a yield strength, the
:attr:`~SimulationResult.safety_factor` — the smallest ratio of yield strength
to von Mises stress over the elements, i.e. the factor by which the whole load
case could be scaled before the first element yields.  Both are None when the
scene never said enough to compute them; neither invents a value.

Differentiability contract: the field arrays (``temperature``,
``displacement``), the domain mass, and the objective helpers
:meth:`SimulationResult.mean` / :meth:`SimulationResult.max` stay JAX-traced
whenever the solve was traced
(``points=recompute_points(...)``), so ``jax.grad`` flows through them
exactly as through the underlying solver results.  Everything that needs
concrete numbers — :meth:`SimulationResult.describe`,
:meth:`SimulationResult.nodal_scalar`, :meth:`SimulationResult.von_mises`,
:meth:`SimulationResult.to_vtk` — is only valid on an untraced result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cadjoint.fem.hexmesh import HexMesh
from cadjoint.fem.render_payload import cell_to_node_scalar
from cadjoint.fem.simulate import ElasticResult, ThermalResult

__all__ = ["SimulationResult"]


@dataclass(frozen=True)
class SimulationResult:
    """One solved study: named, typed, inspectable, exportable.

    Attributes:
        name: The study's name.
        kind: ``"thermal"`` or ``"elastic"``.
        field: Display scalar name — ``"temperature"`` or ``"von_mises"``.
        solution: The underlying solver result
            (:class:`~cadjoint.fem.simulate.ThermalResult` or
            :class:`~cadjoint.fem.simulate.ElasticResult`).
        sim_mesh: The :class:`~cadjoint.fem.simmesh.SimMesh` the study
            solved on (None when a raw ``HexMesh`` was passed to ``solve``).
        mass: Mass of the solved domain in kg — ``sum(rho_e * V_e)`` over the
            elements — or None when the scene's materials state no density.
            Traced whenever the solve was, so it can be optimized against.
        yield_strength: Per-element yield strength ``(C,)`` in Pa, or None
            when the materials state none; drives :attr:`safety_factor`.
    """

    name: str
    kind: str
    field: str
    solution: ThermalResult | ElasticResult
    sim_mesh: Any = None
    mass: Any = None
    yield_strength: Any = None

    # ── field access ────────────────────────────────────────────────────────

    @property
    def mesh(self) -> HexMesh:
        """The hex mesh the fields live on."""
        return self.solution.mesh

    @property
    def temperature(self) -> Any:
        """Per-node temperature ``(N,)`` (thermal results; possibly traced)."""
        if self.kind != "thermal":
            raise AttributeError(f"{self.kind} result has no temperature field.")
        return self.solution.temperature

    @property
    def displacement(self) -> Any:
        """Per-node displacement ``(N, 3)`` (elastic results; possibly traced)."""
        if self.kind != "elastic":
            raise AttributeError(f"{self.kind} result has no displacement field.")
        return self.solution.displacement

    def von_mises(self) -> np.ndarray:
        """Per-cell von Mises stress ``(C,)`` (elastic results, concrete only)."""
        if self.kind != "elastic":
            raise AttributeError(f"{self.kind} result has no von Mises stress.")
        return self.solution.von_mises()

    @property
    def safety_factor(self) -> float | None:
        """Smallest ``yield_strength / von_Mises`` over the elements, or None.

        The factor the whole (linear) load case could be scaled by before the
        first element reaches yield.  None unless this is an elastic result
        whose materials state a yield strength.  Concrete results only, like
        :meth:`von_mises`.

        Returns:
            The minimum safety factor as a float, or None when it is not
            defined for this result.
        """
        if self.kind != "elastic" or self.yield_strength is None:
            return None
        stress = np.asarray(self.von_mises(), dtype=np.float64)
        strength = np.asarray(self.yield_strength, dtype=np.float64)
        # An unloaded element has zero stress and an infinite margin; it must
        # not become the reported minimum, nor a division warning.
        return float(np.min(strength / np.maximum(stress, 1e-30)))

    @property
    def refinement(self) -> dict[str, Any] | None:
        """What automatic grid refinement the tet mesher had to do, or None.

        Tet meshes need a finer grid than hexes on thin features, so
        :func:`~cadjoint.fem.tetmesh.sdf_to_tet_mesh` may have meshed this
        result on a finer grid than the SimMesh declared (see that
        function for the record's shape).  None for hex results and for
        tet meshes that did not come through the ladder.
        """
        return getattr(self.mesh, "refinement", None)

    def nodal_scalar(self) -> np.ndarray:
        """The concrete per-node display field named by :attr:`field`.

        Temperature for thermal results; the cell-centered von Mises stress
        averaged onto nodes for elastic results.  Concrete only (this is
        the array the viewer colors the surface with).
        """
        if self.kind == "thermal":
            return np.asarray(self.solution.temperature, dtype=np.float64)
        return cell_to_node_scalar(self.mesh, self.solution.von_mises())

    # ── differentiable objective helpers ────────────────────────────────────

    def _objective_scalar(self) -> Any:
        """Per-node objective scalar, traced-safe.

        Temperature for thermal results; the (guarded) displacement
        magnitude for elastic ones.  The guard keeps the gradient finite at
        exactly-zero displacements (clamped nodes), where a bare norm's
        derivative is 0/0.
        """
        import jax.numpy as jnp

        if self.kind == "thermal":
            return jnp.asarray(self.solution.temperature)
        displacement = jnp.asarray(self.solution.displacement)
        return jnp.sqrt(jnp.sum(displacement * displacement, axis=-1) + 1e-30)

    def mean(self) -> Any:
        """Mean of the objective scalar (temperature / displacement magnitude).

        A JAX scalar; differentiable through a traced solve.
        """
        import jax.numpy as jnp

        return jnp.mean(self._objective_scalar())

    def max(self) -> Any:
        """Max of the objective scalar (temperature / displacement magnitude).

        A JAX scalar; differentiable through a traced solve (subgradient at
        ties, as usual for ``max``).
        """
        import jax.numpy as jnp

        return jnp.max(self._objective_scalar())

    # ── inspection / export ─────────────────────────────────────────────────

    def describe(self) -> dict[str, Any]:
        """JSON-ready summary of the solved result (concrete results only).

        Returns:
            ``{"name", "kind", "field", "mesh", "nodes", "elements",
            "range", "fields", "mass", "safety_factor", "refinement"}``
            where ``mesh`` is
            the SimMesh name (or None), ``range`` is the ``[min, max]`` of
            the display field of :meth:`nodal_scalar`, ``fields`` maps each
            solved field to a ``{"min", "mean", "max"}`` summary
            (displacement summarized by magnitude, von Mises per cell), and
            ``mass`` (kg) / ``safety_factor`` are None when the scene's
            materials do not make them computable.  ``refinement`` is None
            unless the tet mesher had to re-dice the declared grid, in
            which case it is ``{"declared", "used", "attempts"}`` — the
            declared and actually-used cell counts and how many
            extractions it took (:attr:`refinement` holds the full record).
        """
        scalar = self.nodal_scalar()
        fields: dict[str, dict[str, float]] = {}
        if self.kind == "thermal":
            fields["temperature"] = _summary(scalar)
        else:
            magnitude = np.linalg.norm(
                np.asarray(self.solution.displacement, dtype=np.float64), axis=-1
            )
            fields["displacement"] = _summary(magnitude)
            fields["von_mises"] = _summary(np.asarray(self.von_mises(), dtype=np.float64))
        return {
            "name": self.name,
            "kind": self.kind,
            "field": self.field,
            "mesh": self.sim_mesh.name if self.sim_mesh is not None else None,
            "nodes": self.mesh.num_points,
            "elements": self.mesh.num_cells,
            "range": [round(float(scalar.min()), 6), round(float(scalar.max()), 6)],
            "fields": fields,
            "mass": None if self.mass is None else round(float(self.mass), 9),
            "safety_factor": (None if self.safety_factor is None else round(self.safety_factor, 6)),
            "refinement": _refinement_summary(self.refinement),
        }

    def to_vtk(self, path: str) -> None:
        """Write mesh + fields as a VTK file for ParaView (concrete only).

        Reuses the underlying result's writer: temperature for thermal,
        displacement + von Mises for elastic.
        """
        self.solution.vtk_export(path)


def _refinement_summary(record: dict[str, Any] | None) -> dict[str, Any] | None:
    """JSON-ready digest of a tet refinement record (lists, not tuples).

    Args:
        record: The mesh's ``refinement`` record, or None.

    Returns:
        None when nothing was refined, else ``{"declared", "used",
        "attempts"}`` with the cell counts as lists so the payload
        round-trips through JSON.
    """
    if not record or not record.get("refined"):
        return None
    return {
        "declared": [int(count) for count in record["declared"]],
        "used": [int(count) for count in record["used"]],
        "attempts": len(record["attempts"]),
    }


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "min": round(float(values.min()), 6),
        "mean": round(float(values.mean()), 6),
        "max": round(float(values.max()), 6),
    }
