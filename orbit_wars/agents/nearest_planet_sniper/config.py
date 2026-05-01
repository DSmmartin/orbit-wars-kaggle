"""Configuration knobs for the nearest-planet sniper policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SniperPolicyConfig:
    """Tunables used to select launch size and planning horizons."""

    minimum_ships_to_send: int = 20
    max_speed: float = 6.0
    max_iterations: int = 100
    step_tolerance: float = 0.05
    evaluation_horizon: int = 220
