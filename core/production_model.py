"""Production-side Markov model for modular IoT manufacturing simulations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


STATE_NAMES = ("failed", "degraded", "nominal", "boost")


@dataclass
class ProductionUnit:
    """A production unit with four Markov states and state-dependent capacity.

    States are encoded as:
    0 = failed/offline, 1 = degraded, 2 = nominal, 3 = boost/high-output.
    The transition model is intentionally stress-aware: high production commands
    and network congestion increase the chance of dropping to a worse state.
    """

    unit_id: int
    initial_state: int = 2
    capacities: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.55, 1.0, 1.35], dtype=float)
    )
    base_transition_matrix: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [0.62, 0.30, 0.08, 0.00],
                [0.10, 0.68, 0.20, 0.02],
                [0.02, 0.14, 0.70, 0.14],
                [0.04, 0.22, 0.34, 0.40],
            ],
            dtype=float,
        )
    )

    def __post_init__(self) -> None:
        self.state = int(self.initial_state)
        self.degradation = 0.0

    @property
    def state_name(self) -> str:
        return STATE_NAMES[self.state]

    @property
    def capacity(self) -> float:
        return float(self.capacities[self.state])

    def reset(self) -> None:
        self.state = int(self.initial_state)
        self.degradation = 0.0

    def transition(
        self,
        rng: np.random.Generator,
        production_command: float,
        overload_ratio: float,
        maintenance_relief: float = 0.0,
    ) -> int:
        """Advance one time step using a stress-adjusted Markov transition."""

        command_stress = float(np.clip(production_command, 0.0, 1.25))
        overload_stress = float(np.clip(overload_ratio - 1.0, 0.0, 2.0))
        stress = 0.55 * command_stress + 0.45 * overload_stress

        self.degradation = float(
            np.clip(
                0.88 * self.degradation
                + 0.16 * stress
                - 0.10 * maintenance_relief,
                0.0,
                1.0,
            )
        )

        probs = self.base_transition_matrix[self.state].astype(float).copy()

        if self.state > 0:
            downward_shift = min(0.30, 0.10 * stress + 0.16 * self.degradation)
            probs[self.state] -= downward_shift
            probs[self.state - 1] += downward_shift

        if self.state < 3 and stress < 0.75:
            upward_shift = min(0.12, 0.05 * (0.75 - stress) + 0.04 * maintenance_relief)
            probs[self.state] -= upward_shift
            probs[self.state + 1] += upward_shift

        probs = np.clip(probs, 0.0, None)
        probs = probs / probs.sum()
        self.state = int(rng.choice(np.arange(4), p=probs))
        return self.state

    def planned_output(self, production_command: float) -> float:
        """Return commanded output bounded by physical state capacity."""

        command = float(np.clip(production_command, 0.0, 1.25))
        return self.capacity * command
