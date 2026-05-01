"""Comet ballistic solver."""

from __future__ import annotations

import math
from typing import Any

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from orbit_wars.army.physics import fleet_speed


def _value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _find_comet_path(
    target_id: int,
    comets: list[Any] | None,
) -> tuple[list[list[float]] | None, int | None]:
    if not comets:
        return None, None

    for group in comets:
        planet_ids = _value(group, "planet_ids", [])
        if target_id not in planet_ids:
            continue
        idx = planet_ids.index(target_id)
        paths = _value(group, "paths", [])
        path_index = int(_value(group, "path_index", -1))
        if idx < len(paths):
            return paths[idx], path_index
    return None, None


def solve_comet_intercept(
    from_planet: Planet,
    to_planet: Planet,
    ships: int,
    *,
    comets: list[Any] | None,
    max_speed: float = 6.0,
    max_future_steps: int = 80,
    time_tolerance: float = 0.65,
) -> tuple[float | None, float | None, str | None]:
    """
    Return angle, ETA and optional invalid reason for a comet target.

    The solver searches future path indices and finds the earliest step where the
    fleet travel time approximately matches the comet's future position index.
    """
    path, path_index = _find_comet_path(to_planet.id, comets)
    if not path or path_index is None:
        return None, None, "comet_path_not_found"

    speed = fleet_speed(ships, max_speed)
    launch_offset = from_planet.radius + 0.1
    fx, fy = from_planet.x, from_planet.y

    best: tuple[float, float, float] | None = None  # err, angle, eta

    for future in range(0, max_future_steps + 1):
        idx = path_index + future
        if idx < 0 or idx >= len(path):
            break

        tx, ty = path[idx]
        d = math.hypot(tx - fx, ty - fy)
        eta = max(0.0, d - launch_offset) / speed
        err = abs(eta - future)
        angle = math.atan2(ty - fy, tx - fx)

        if best is None or err < best[0]:
            best = (err, angle, eta)

        if err <= time_tolerance:
            return angle, eta, None

    if best is None:
        return None, None, "comet_expires_before_reachable"

    if best[0] > 2.0:
        return None, None, "comet_intercept_unstable"

    return best[1], best[2], None
