"""Static planet ballistic solver."""

from __future__ import annotations

import math

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from orbit_wars.army.physics import fleet_speed


def solve_static_planet_intercept(
    from_planet: Planet,
    to_planet: Planet,
    ships: int,
    *,
    max_speed: float = 6.0,
) -> tuple[float, float]:
    """Return angle and ETA (steps) for a static target."""
    dx = to_planet.x - from_planet.x
    dy = to_planet.y - from_planet.y
    angle = math.atan2(dy, dx)
    launch_offset = from_planet.radius + 0.1
    eta = max(0.0, math.hypot(dx, dy) - launch_offset) / fleet_speed(ships, max_speed)
    return angle, eta
