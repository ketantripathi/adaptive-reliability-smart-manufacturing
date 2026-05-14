"""Information-network model for IoT smart manufacturing reliability studies."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Node:
    """A network node with finite capacity and smoothed congestion memory."""

    name: str
    capacity: float
    load: float = 0.0
    congestion_memory: float = 0.0

    def reset(self) -> None:
        self.load = 0.0
        self.congestion_memory = 0.0

    @property
    def utilization(self) -> float:
        return float(self.load / self.capacity) if self.capacity > 0 else 0.0

    @property
    def overload(self) -> float:
        return float(max(0.0, self.utilization - 1.0))

    def update_memory(self) -> None:
        self.congestion_memory = float(
            np.clip(0.82 * self.congestion_memory + 0.18 * self.overload, 0.0, 2.0)
        )


@dataclass
class EndNode(Node):
    """End node attached to a production unit."""

    unit_id: int = 0

    def update_load(self, unit_state: int, production_command: float, noise: float) -> None:
        state_component = 0.18 + 0.32 * unit_state
        command_component = 0.65 * float(np.clip(production_command, 0.0, 1.25))
        self.load = max(0.0, state_component + command_component + noise)
        self.update_memory()


@dataclass
class RouteNode(Node):
    """Route node aggregating traffic from a set of end nodes."""

    children: list[EndNode] = field(default_factory=list)

    def update_load(self, noise: float) -> None:
        child_load = sum(node.load for node in self.children)
        self.load = max(0.0, 0.10 + child_load * (1.0 + 0.05 * len(self.children)) + noise)
        self.update_memory()


@dataclass
class Gateway(Node):
    """Gateway node aggregating all route traffic."""

    routes: list[RouteNode] = field(default_factory=list)

    def update_load(self, noise: float) -> None:
        route_load = sum(node.load for node in self.routes)
        self.load = max(0.0, 0.20 + 1.08 * route_load + noise)
        self.update_memory()


class InformationNetwork:
    """Tree network: production end nodes -> route nodes -> gateway."""

    def __init__(
        self,
        num_units: int,
        route_capacity: float = 3.10,
        gateway_capacity: float = 6.10,
    ) -> None:
        self.end_nodes = [
            EndNode(name=f"end_{idx}", capacity=1.70, unit_id=idx)
            for idx in range(num_units)
        ]
        midpoint = int(np.ceil(num_units / 2))
        self.route_nodes = [
            RouteNode(name="route_0", capacity=route_capacity, children=self.end_nodes[:midpoint]),
            RouteNode(name="route_1", capacity=route_capacity, children=self.end_nodes[midpoint:]),
        ]
        self.gateway = Gateway(name="gateway", capacity=gateway_capacity, routes=self.route_nodes)

    def reset(self) -> None:
        for node in self.all_nodes:
            node.reset()

    @property
    def all_nodes(self) -> list[Node]:
        return [*self.end_nodes, *self.route_nodes, self.gateway]

    def propagate(
        self,
        unit_states: list[int],
        production_command: float,
        rng: np.random.Generator,
        noise_scale: float = 0.05,
    ) -> dict[str, float]:
        """Update all node loads and return aggregate network metrics."""

        for node, state in zip(self.end_nodes, unit_states):
            node.update_load(
                unit_state=state,
                production_command=production_command,
                noise=float(rng.normal(0.0, noise_scale)),
            )

        for route in self.route_nodes:
            route.update_load(noise=float(rng.normal(0.0, noise_scale)))

        self.gateway.update_load(noise=float(rng.normal(0.0, noise_scale)))

        loads = np.array([node.load for node in self.all_nodes], dtype=float)
        utilizations = np.array([node.utilization for node in self.all_nodes], dtype=float)
        overloads = np.array([node.overload for node in self.all_nodes], dtype=float)
        return {
            "avg_load": float(loads.mean()),
            "max_load": float(loads.max()),
            "gateway_load": float(self.gateway.load),
            "gateway_utilization": float(self.gateway.utilization),
            "avg_utilization": float(utilizations.mean()),
            "max_utilization": float(utilizations.max()),
            "overload_risk": float(np.clip(overloads.max(), 0.0, 2.0)),
            "congestion_memory": float(
                np.mean([node.congestion_memory for node in self.all_nodes])
            ),
        }
