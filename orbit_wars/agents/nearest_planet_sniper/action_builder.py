"""Helpers that convert policy decisions into environment move payloads."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ShotDecision:
    """A selected launch decision for one source/target pair."""

    source_planet_id: int
    target_planet_id: int
    angle_rad: float
    ships: int


def build_moves(decisions: list[ShotDecision]) -> list[list[float | int]]:
    """Convert decisions to Kaggle move format `[from_id, angle, ships]`."""
    return [[d.source_planet_id, d.angle_rad, d.ships] for d in decisions]
