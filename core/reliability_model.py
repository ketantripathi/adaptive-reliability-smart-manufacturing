"""Reliability and accuracy functions for the IoT manufacturing model."""

from __future__ import annotations

import numpy as np


def transmission_accuracy(
    load: float,
    capacity: float,
    congestion_memory: float = 0.0,
    shape: float = 2.35,
) -> float:
    """Nonlinear transmission accuracy degradation under load.

    Accuracy remains high below capacity, then drops sharply as the gateway
    approaches overload. Congestion memory captures lingering packet loss and
    retransmission effects from previous overload periods.
    """

    utilization = max(0.0, float(load) / float(capacity))
    congestion_penalty = 1.0 + 0.55 * max(0.0, congestion_memory)
    accuracy = 0.985 / (1.0 + (utilization * congestion_penalty) ** shape)
    return float(np.clip(accuracy, 0.05, 0.985))


def overload_loss(overload_risk: float) -> float:
    """Fraction of effective production lost to network overload."""

    risk = max(0.0, float(overload_risk))
    return float(np.clip(0.08 + 0.52 * (1.0 - np.exp(-1.6 * risk)), 0.0, 0.72))


def effective_production(
    planned_production: float,
    accuracy: float,
    overload_risk: float,
) -> float:
    """Production delivered after information-network degradation."""

    loss = overload_loss(overload_risk)
    return float(planned_production * accuracy * (1.0 - loss))


def system_reliability(production_rate: float, demand: float) -> float:
    """Binary reliability indicator Rs(t)."""

    return float(production_rate >= demand)


def rolling_failure_rate(reliability: np.ndarray, window: int = 25) -> np.ndarray:
    """Rolling mean of failures where failure = 1 - reliability."""

    rel = np.asarray(reliability, dtype=float)
    failures = 1.0 - rel
    if len(failures) == 0:
        return failures
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(failures, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")
