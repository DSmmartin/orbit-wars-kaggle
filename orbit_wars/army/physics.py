"""Shared physics helpers for Orbit Wars ballistic planning."""

from __future__ import annotations

import math

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

CENTER = 50.0
BOARD_SIZE = 100.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_to_segment_distance(
    point: tuple[float, float],
    segment_start: tuple[float, float],
    segment_end: tuple[float, float],
) -> float:
    """Minimum distance from *point* to the segment [segment_start, segment_end]."""
    vx = segment_end[0] - segment_start[0]
    vy = segment_end[1] - segment_start[1]
    l2 = vx * vx + vy * vy
    if l2 == 0.0:
        return distance(point, segment_start)

    px = point[0] - segment_start[0]
    py = point[1] - segment_start[1]
    t = max(0.0, min(1.0, (px * vx + py * vy) / l2))
    projection = (segment_start[0] + t * vx, segment_start[1] + t * vy)
    return distance(point, projection)


def fleet_speed(ships: int, max_speed: float = 6.0) -> float:
    """Return fleet speed (units/step) using env scaling."""
    if ships <= 1:
        return 1.0
    speed = 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
    return min(speed, max_speed)


def is_rotating(planet: Planet) -> bool:
    """Return True when a planet rotates around the sun."""
    orbital_radius = math.hypot(planet.x - CENTER, planet.y - CENTER)
    return orbital_radius + planet.radius < ROTATION_RADIUS_LIMIT


def planet_position_at_step(
    initial_planet: Planet,
    step: float,
    angular_velocity: float,
) -> tuple[float, float]:
    """
    Planet position for an absolute game step (supports float steps).

    Matches the environment's step indexing where observation step N corresponds
    to angle = initial_angle + angular_velocity * max(0, N - 1).
    """
    dx = initial_planet.x - CENTER
    dy = initial_planet.y - CENTER
    orbital_radius = math.hypot(dx, dy)

    if orbital_radius + initial_planet.radius >= ROTATION_RADIUS_LIMIT:
        return initial_planet.x, initial_planet.y

    initial_angle = math.atan2(dy, dx)
    angle = initial_angle + angular_velocity * max(0.0, step - 1.0)
    return (
        CENTER + orbital_radius * math.cos(angle),
        CENTER + orbital_radius * math.sin(angle),
    )
