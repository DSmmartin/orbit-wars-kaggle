"""Battle records — typed game-state structures parsed from raw Kaggle observations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PlanetState:
    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: int
    production: int


@dataclass(slots=True)
class FleetState:
    id: int
    owner: int
    x: float
    y: float
    angle: float
    from_planet_id: int
    ships: int


@dataclass(slots=True)
class GameState:
    step: int
    player: int
    angular_velocity: float
    planets: list[PlanetState]
    fleets: list[FleetState]
    initial_by_id: dict[int, PlanetState] = field(default_factory=dict)
    raw_comets: list = field(default_factory=list)
    comet_planet_ids: list[int] = field(default_factory=list)


def parse_observation(observation: Any) -> GameState:
    def obs_get(key: str, default: Any) -> Any:
        if isinstance(observation, dict):
            return observation.get(key, default)
        return getattr(observation, key, default)

    planets = [
        PlanetState(
            id=int(row[0]),
            owner=int(row[1]),
            x=float(row[2]),
            y=float(row[3]),
            radius=float(row[4]),
            ships=int(row[5]),
            production=int(row[6]),
        )
        for row in obs_get("planets", [])
    ]
    fleets = [
        FleetState(
            id=int(row[0]),
            owner=int(row[1]),
            x=float(row[2]),
            y=float(row[3]),
            angle=float(row[4]),
            from_planet_id=int(row[5]),
            ships=int(row[6]),
        )
        for row in obs_get("fleets", [])
    ]
    initial_by_id = {
        int(row[0]): PlanetState(
            id=int(row[0]),
            owner=int(row[1]),
            x=float(row[2]),
            y=float(row[3]),
            radius=float(row[4]),
            ships=int(row[5]),
            production=int(row[6]),
        )
        for row in obs_get("initial_planets", [])
    }
    return GameState(
        step=int(obs_get("step", 0)),
        player=int(obs_get("player", 0)),
        angular_velocity=float(obs_get("angular_velocity", 0.0)),
        planets=planets,
        fleets=fleets,
        initial_by_id=initial_by_id,
        raw_comets=list(obs_get("comets", [])),
        comet_planet_ids=[int(pid) for pid in obs_get("comet_planet_ids", [])],
    )
