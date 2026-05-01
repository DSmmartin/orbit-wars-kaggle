"""Data models for fleet-level tactical analysis."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FleetEnemy:
    """Predicted enemy fleet impact against a future target planet."""

    id: int
    target_planet: int
    time_arrival: int
    ships: int
    angle_rad: float
