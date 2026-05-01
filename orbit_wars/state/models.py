"""Typed state models used by strategies and tactical policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from orbit_wars.fleets import FleetEnemy

if TYPE_CHECKING:
    from orbit_wars.astronomy import AstronomyForecast


@dataclass(slots=True)
class GameState:
    """Normalized game snapshot derived from a raw Kaggle observation."""

    player: int
    current_step: int
    angular_velocity: float
    planets: list[Planet]
    initial_by_id: dict[int, Planet]
    raw_comets: list[Any]
    comet_planet_ids: list[int]
    enemy_fleets: list[FleetEnemy]
    # Carried between turns to enable incremental updates — not part of strategy API.
    astronomy_forecast: AstronomyForecast | None = field(default=None, repr=False)

    def my_planets(self) -> list[Planet]:
        """Return planets currently owned by the active player."""
        return [planet for planet in self.planets if planet.owner == self.player]

    def target_planets(self) -> list[Planet]:
        """Return non-friendly planets as candidate targets."""
        return [planet for planet in self.planets if planet.owner != self.player]
