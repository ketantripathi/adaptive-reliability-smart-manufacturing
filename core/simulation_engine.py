"""Integrated simulation engine for baseline and improved IoT reliability models."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

try:
    from .network_model import InformationNetwork
    from .production_model import ProductionUnit
    from .reliability_model import (
        effective_production,
        system_reliability,
        transmission_accuracy,
    )
except ImportError:  # Allows direct execution from notebooks in loose folders.
    from network_model import InformationNetwork
    from production_model import ProductionUnit
    from reliability_model import (
        effective_production,
        system_reliability,
        transmission_accuracy,
    )


@dataclass
class SimulationConfig:
    num_units: int = 4
    demand: float = 0.90
    route_capacity: float = 3.10
    gateway_capacity: float = 6.10
    baseline_gamma: float = 1.25
    noise_scale: float = 0.06
    lambda_degradation: float = 0.42
    lambda_overload: float = 0.62
    lambda_smoothing: float = 0.18


class AdaptivePredictiveController:
    """Adaptive predictive reliability-aware production controller."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.candidate_gammas = np.array([0.62, 0.72, 0.82, 0.92, 1.00, 1.08], dtype=float)

    def predict_reliability(
        self,
        planned_output: float,
        load: float,
        capacity: float,
        congestion_memory: float,
        overload_risk: float,
    ) -> float:
        future_accuracy = transmission_accuracy(
            load=load * (1.0 + 0.18 * overload_risk),
            capacity=capacity,
            congestion_memory=congestion_memory,
        )
        expected_output = effective_production(planned_output, future_accuracy, overload_risk)
        margin = expected_output - self.config.demand
        return float(1.0 / (1.0 + np.exp(-4.0 * margin)))

    def choose_gamma(
        self,
        units: list[ProductionUnit],
        network_metrics: dict[str, float],
        previous_gamma: float,
        mode: str,
    ) -> float:
        if mode == "baseline":
            return self.config.baseline_gamma

        if mode == "load_aware":
            utilization = network_metrics.get("gateway_utilization", 0.0)
            return float(np.clip(1.08 - 0.30 * max(0.0, utilization - 0.82), 0.72, 1.08))

        if mode == "prediction_only":
            return self._maximize_cost(
                units=units,
                network_metrics=network_metrics,
                previous_gamma=previous_gamma,
                use_overload_penalty=False,
                use_smoothing=False,
            )

        if mode == "improved":
            return self._maximize_cost(
                units=units,
                network_metrics=network_metrics,
                previous_gamma=previous_gamma,
                use_overload_penalty=True,
                use_smoothing=True,
            )

        raise ValueError(f"Unknown simulation mode: {mode}")

    def _maximize_cost(
        self,
        units: list[ProductionUnit],
        network_metrics: dict[str, float],
        previous_gamma: float,
        use_overload_penalty: bool,
        use_smoothing: bool,
    ) -> float:
        base_capacity = sum(unit.capacity for unit in units)
        best_gamma = float(self.candidate_gammas[0])
        best_cost = -np.inf

        for gamma in self.candidate_gammas:
            planned_output = base_capacity * gamma
            projected_load = network_metrics["gateway_load"] * (0.40 + 0.62 * gamma)
            projected_overload = max(0.0, projected_load / self.config.gateway_capacity - 1.0)
            degradation_risk = float(np.mean([unit.degradation for unit in units]))
            r_hat = self.predict_reliability(
                planned_output=planned_output,
                load=projected_load,
                capacity=self.config.gateway_capacity,
                congestion_memory=network_metrics["congestion_memory"],
                overload_risk=projected_overload,
            )
            overload_penalty = self.config.lambda_overload * projected_overload
            smoothing_penalty = self.config.lambda_smoothing * abs(gamma - previous_gamma)
            cost = (
                r_hat
                - self.config.lambda_degradation * degradation_risk
                - (overload_penalty if use_overload_penalty else 0.0)
                - (smoothing_penalty if use_smoothing else 0.0)
            )
            if cost > best_cost:
                best_cost = cost
                best_gamma = float(gamma)

        return best_gamma


class SimulationEngine:
    """Runs Monte Carlo simulations for baseline, ablations, and proposed model."""

    def __init__(self, config: SimulationConfig | None = None, seed: int | None = None) -> None:
        self.config = config or SimulationConfig()
        self.rng = np.random.default_rng(seed)
        self.units = [ProductionUnit(unit_id=idx) for idx in range(self.config.num_units)]
        self.network = InformationNetwork(
            num_units=self.config.num_units,
            route_capacity=self.config.route_capacity,
            gateway_capacity=self.config.gateway_capacity,
        )
        self.controller = AdaptivePredictiveController(self.config)
        self.previous_gamma = self.config.baseline_gamma

    def reset(self) -> None:
        for unit in self.units:
            unit.reset()
        self.network.reset()
        self.previous_gamma = self.config.baseline_gamma

    def _current_network_metrics(self) -> dict[str, float]:
        return {
            "avg_load": 0.0,
            "max_load": 0.0,
            "gateway_load": max(0.01, self.network.gateway.load),
            "gateway_utilization": self.network.gateway.utilization,
            "avg_utilization": 0.0,
            "max_utilization": 0.0,
            "overload_risk": self.network.gateway.overload,
            "congestion_memory": self.network.gateway.congestion_memory,
        }

    def step(self, mode: str = "baseline") -> dict[str, object]:
        pre_metrics = self._current_network_metrics()
        gamma = self.controller.choose_gamma(
            units=self.units,
            network_metrics=pre_metrics,
            previous_gamma=self.previous_gamma,
            mode=mode,
        )

        overload_ratio = max(1.0, pre_metrics["gateway_utilization"])
        maintenance_relief = 0.18 if mode in {"load_aware", "improved"} and gamma < 0.86 else 0.0

        for unit in self.units:
            unit.transition(
                rng=self.rng,
                production_command=gamma,
                overload_ratio=overload_ratio,
                maintenance_relief=maintenance_relief,
            )

        unit_states = [unit.state for unit in self.units]
        network_metrics = self.network.propagate(
            unit_states=unit_states,
            production_command=gamma,
            rng=self.rng,
            noise_scale=self.config.noise_scale,
        )

        planned = float(sum(unit.planned_output(gamma) for unit in self.units))
        accuracy = transmission_accuracy(
            self.network.gateway.load,
            self.network.gateway.capacity,
            network_metrics["congestion_memory"],
        )
        production_rate = effective_production(
            planned_production=planned,
            accuracy=accuracy,
            overload_risk=network_metrics["overload_risk"],
        )
        reliability = system_reliability(production_rate, self.config.demand)
        throughput = production_rate * accuracy

        self.previous_gamma = gamma

        return {
            "unit_state_vector": unit_states,
            "unit_state_mean": float(np.mean(unit_states)),
            "gamma": float(gamma),
            "planned_production": planned,
            "production_rate": production_rate,
            "throughput": float(throughput),
            "transmission_accuracy": accuracy,
            "reliability": reliability,
            "failure": float(1.0 - reliability),
            **network_metrics,
        }

    def run(self, steps: int = 600, runs: int = 200, mode: str = "baseline", seed: int = 42) -> pd.DataFrame:
        records: list[dict[str, object]] = []
        master_rng = np.random.default_rng(seed)

        for run_id in range(runs):
            self.rng = np.random.default_rng(int(master_rng.integers(0, 2**31 - 1)))
            self.reset()
            for time_index in range(steps):
                row = self.step(mode=mode)
                row["time"] = time_index
                row["run"] = run_id
                row["mode"] = mode
                records.append(row)

        return pd.DataFrame.from_records(records)

    def sample_state_grid(self) -> pd.DataFrame:
        rows = []
        for states in product(range(4), repeat=self.config.num_units):
            planned = sum(self.units[idx].capacities[state] for idx, state in enumerate(states))
            rows.append(
                {
                    "unit_state_vector": list(states),
                    "nominal_capacity": float(planned),
                    "meets_demand_without_network_loss": float(planned >= self.config.demand),
                }
            )
        return pd.DataFrame(rows)


def summarize_by_time(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Monte Carlo records into time-series means."""

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    keep = [col for col in numeric_cols if col not in {"run"}]
    return df.groupby("time", as_index=False)[keep].mean()


def performance_summary(df: pd.DataFrame) -> dict[str, float]:
    """Research metrics used by comparison and ablation notebooks."""

    return {
        "mean_reliability": float(df["reliability"].mean()),
        "failure_rate": float(df["failure"].mean()),
        "mean_production": float(df["production_rate"].mean()),
        "production_variance": float(df["production_rate"].var()),
        "mean_accuracy": float(df["transmission_accuracy"].mean()),
        "mean_load": float(df["gateway_load"].mean()),
        "mean_throughput": float(df["throughput"].mean()),
    }
