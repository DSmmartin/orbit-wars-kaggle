"""Rotating planet ballistic solver."""

from __future__ import annotations

import math

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from orbit_wars.army.physics import fleet_speed, planet_position_at_step


def solve_rotating_planet_intercept(
    from_planet: Planet,
    current_to_planet: Planet,
    ships: int,
    current_step: int,
    angular_velocity: float,
    *,
    initial_to_planet: Planet,
    max_speed: float = 6.0,
    max_iterations: int = 100,
    step_tolerance: float = 0.05,
) -> tuple[float, float]:
    """Return angle and ETA (steps) for an orbiting target."""
    speed = fleet_speed(ships, max_speed)
    fx, fy = from_planet.x, from_planet.y
    launch_offset = from_planet.radius + 0.1

    d0 = math.hypot(current_to_planet.x - fx, current_to_planet.y - fy)
    t_est = max(0.0, d0 - launch_offset) / speed

    tx, ty = current_to_planet.x, current_to_planet.y
    for _ in range(max_iterations):
        arrival_step = current_step + t_est
        tx, ty = planet_position_at_step(initial_to_planet, arrival_step, angular_velocity)

        d_new = math.hypot(tx - fx, ty - fy)
        t_new = max(0.0, d_new - launch_offset) / speed
        if abs(t_new - t_est) < step_tolerance:
            t_est = t_new
            break
        t_est = t_new

    angle = math.atan2(ty - fy, tx - fx)
    return angle, t_est
