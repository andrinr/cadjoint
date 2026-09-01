"""Simulation results as first-class, re-inspectable objects.

:meth:`~cadjoint.fem.study.ThermalStudy.solve` returns a
:class:`SimulationResult`: the study's name and kind, the display field
name, the solved field arrays, and a reference back to the
:class:`~cadjoint.fem.simmesh.SimMesh` it ran on.  The instance is also
stored on the study as ``last_result`` so a program (or the viewer) can
re-inspect a solve without repeating it.

Differentiability contract: the field arrays (``temperature``,
``displacement``) and the objective helpers :meth:`SimulationResult.mean` /
:meth:`SimulationResult.max` stay JAX-traced whenever the solve was traced
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
    """

    name: str
    kind: str
    field: str
    solution: ThermalResult | ElasticResult
    sim_mesh: Any = None

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
            "range", "fields"}`` where ``mesh`` is the SimMesh name (or
            None), ``range`` is the ``[min, max]`` of the display field of
            :meth:`nodal_scalar`, and ``fields`` maps each solved field to
            a ``{"min", "mean", "max"}`` summary (displacement summarized
            by magnitude, von Mises per cell).
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
        }

    def to_vtk(self, path: str) -> None:
        """Write mesh + fields as a VTK file for ParaView (concrete only).

        Reuses the underlying result's writer: temperature for thermal,
        displacement + von Mises for elastic.
        """
        self.solution.vtk_export(path)


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "min": round(float(values.min()), 6),
        "mean": round(float(values.mean()), 6),
        "max": round(float(values.max()), 6),
    }
