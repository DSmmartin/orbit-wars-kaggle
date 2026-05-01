"""Trajectory evaluation for Orbit Wars shots."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from orbit_wars.army.physics import (
    BOARD_SIZE,
    SUN_RADIUS,
    fleet_speed,
    is_rotating,
    planet_position_at_step,
    point_to_segment_distance,
)


@dataclass(slots=True)
class ShotEvaluation:
    valid: bool
    reason: str | None = None
    hit_target_step_offset: int | None = None


def _value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _comet_state_by_id(comets: Sequence[Any] | None) -> dict[int, tuple[list[list[float]], int]]:
    """Map comet planet id -> (path, current_path_index)."""
    result: dict[int, tuple[list[list[float]], int]] = {}
    if not comets:
        return result

    for group in comets:
        pids = _value(group, "planet_ids", [])
        paths = _value(group, "paths", [])
        path_index = int(_value(group, "path_index", -1))
        for i, pid in enumerate(pids):
            if i < len(paths):
                result[pid] = (paths[i], path_index)
    return result


def _planet_pos_at_offset(
    *,
    planet: Planet,
    step_offset: int,
    current_step: int,
    angular_velocity: float,
    initial_by_id: Mapping[int, Planet],
    comet_state: Mapping[int, tuple[list[list[float]], int]],
) -> tuple[float, float] | None:
    """
    Planet position at the start of a future fleet-movement phase.

    Returns None if the body no longer exists at that offset (expired comet).
    """
    if planet.id in comet_state:
        path, path_index = comet_state[planet.id]
        idx = path_index + step_offset
        if idx < 0 or idx >= len(path):
            return None
        return float(path[idx][0]), float(path[idx][1])

    initial = initial_by_id.get(planet.id, planet)
    if is_rotating(initial):
        absolute_step = current_step + step_offset
        return planet_position_at_step(initial, absolute_step, angular_velocity)

    return initial.x, initial.y


def evaluate_shot(
    *,
    from_planet: Planet,
    target_id: int,
    angle: float,
    ships: int,
    current_step: int,
    angular_velocity: float,
    planets: Sequence[Planet],
    initial_by_id: Mapping[int, Planet],
    comets: Sequence[Any] | None,
    max_speed: float = 6.0,
    max_steps: int = 220,
) -> ShotEvaluation:
    """Simulate a shot and report if the target is reached before invalidation."""
    speed = fleet_speed(ships, max_speed)

    start_x = from_planet.x + math.cos(angle) * (from_planet.radius + 0.1)
    start_y = from_planet.y + math.sin(angle) * (from_planet.radius + 0.1)

    fleet_x = start_x
    fleet_y = start_y
    comet_state = _comet_state_by_id(comets)

    for step_offset in range(max_steps + 1):
        old_pos = (fleet_x, fleet_y)
        new_pos = (
            fleet_x + math.cos(angle) * speed,
            fleet_y + math.sin(angle) * speed,
        )

        # 1) Planet collisions first (matches env ordering precedence).
        for planet in planets:
            planet_pos = _planet_pos_at_offset(
                planet=planet,
                step_offset=step_offset,
                current_step=current_step,
                angular_velocity=angular_velocity,
                initial_by_id=initial_by_id,
                comet_state=comet_state,
            )
            if planet_pos is None:
                continue
            if point_to_segment_distance(planet_pos, old_pos, new_pos) < planet.radius:
                if planet.id == target_id:
                    return ShotEvaluation(True, hit_target_step_offset=step_offset)
                return ShotEvaluation(False, reason=f"blocked_by_planet_{planet.id}")

        # 2) Bounds and sun checks (same order as env after planet checks).
        if not (0.0 <= new_pos[0] <= BOARD_SIZE and 0.0 <= new_pos[1] <= BOARD_SIZE):
            return ShotEvaluation(False, reason="out_of_bounds")

        if point_to_segment_distance((50.0, 50.0), old_pos, new_pos) < SUN_RADIUS:
            return ShotEvaluation(False, reason="crosses_sun")

        fleet_x, fleet_y = new_pos

        # 3) Planet/comet sweep during movement phase.
        for planet in planets:
            old_planet_pos = _planet_pos_at_offset(
                planet=planet,
                step_offset=step_offset,
                current_step=current_step,
                angular_velocity=angular_velocity,
                initial_by_id=initial_by_id,
                comet_state=comet_state,
            )
            new_planet_pos = _planet_pos_at_offset(
                planet=planet,
                step_offset=step_offset + 1,
                current_step=current_step,
                angular_velocity=angular_velocity,
                initial_by_id=initial_by_id,
                comet_state=comet_state,
            )
            if old_planet_pos is None or new_planet_pos is None or old_planet_pos == new_planet_pos:
                continue

            if point_to_segment_distance((fleet_x, fleet_y), old_planet_pos, new_planet_pos) < planet.radius:
                if planet.id == target_id:
                    return ShotEvaluation(True, hit_target_step_offset=step_offset)
                return ShotEvaluation(False, reason=f"swept_by_planet_{planet.id}")

    return ShotEvaluation(False, reason="target_not_reached_in_horizon")
