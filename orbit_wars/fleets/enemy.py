from __future__ import annotations

import math

import numpy as np
from kaggle_environments.envs.orbit_wars.orbit_wars import CENTER, SUN_RADIUS, Fleet

from orbit_wars.army.physics import fleet_speed, point_to_segment_distance
from orbit_wars.astronomy import (
    build_astronomy_forecast,
    point_to_segments_distance,
    segment_to_points_distance,
)
from orbit_wars.fleets.models import FleetEnemy
from orbit_wars.observatory import timed_calc, timing_logger


def _get(obs: dict | object, key: str, default):
    """Read observation fields from either dict or object payloads."""
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _predict_fleet_hit(
    *,
    fleet: Fleet,
    max_lookahead: int,
    forecast,
) -> tuple[int | None, int | None]:
    """Predict the first planet hit by a fleet within the lookahead horizon."""
    fleet_x = float(fleet.x)
    fleet_y = float(fleet.y)
    speed = fleet_speed(int(fleet.ships))

    for dt in range(1, max_lookahead + 1):
        old_pos = (fleet_x, fleet_y)
        fleet_x += math.cos(fleet.angle) * speed
        fleet_y += math.sin(fleet.angle) * speed
        new_pos = (fleet_x, fleet_y)

        step_forecast = forecast.steps[dt]
        if step_forecast is None:
            continue

        direct_distances = segment_to_points_distance(step_forecast.centers, old_pos, new_pos)
        direct_hits = np.flatnonzero(direct_distances < step_forecast.radii)
        if direct_hits.size > 0:
            hit_idx = int(direct_hits[0])
            return int(step_forecast.planet_ids[hit_idx]), dt

        # Fleet dies before any future arrival.
        if not (0.0 <= fleet_x <= 100.0 and 0.0 <= fleet_y <= 100.0):
            return None, None

        if point_to_segment_distance((CENTER, CENTER), old_pos, new_pos) < SUN_RADIUS:
            return None, None

        moving_mask = np.any(step_forecast.segment_starts != step_forecast.segment_ends, axis=1)
        if np.any(moving_mask):
            distances = point_to_segments_distance(
                (fleet_x, fleet_y),
                step_forecast.segment_starts[moving_mask],
                step_forecast.segment_ends[moving_mask],
            )
            sweep_hits = np.flatnonzero(distances < step_forecast.segment_radii[moving_mask])
            if sweep_hits.size > 0:
                hit_idx = int(sweep_hits[0])
                hit_ids = step_forecast.segment_planet_ids[moving_mask]
                return int(hit_ids[hit_idx]), dt

    return None, None


def predict_enemy_fleets(
    obs: dict | object,
    forecast,
    *,
    max_lookahead: int = 220,
) -> list[FleetEnemy]:
    """Predict all enemy fleet arrivals given a pre-built astronomy forecast."""
    player = int(_get(obs, "player", 0))
    raw_fleets = _get(obs, "fleets", [])
    fleets = [Fleet(*f) for f in raw_fleets]

    results: list[FleetEnemy] = []
    for fleet in fleets:
        if fleet.owner == player:
            continue
        target_planet, time_arrival = _predict_fleet_hit(
            fleet=fleet, max_lookahead=max_lookahead, forecast=forecast,
        )
        if target_planet is None or time_arrival is None:
            continue
        results.append(FleetEnemy(
            id=int(fleet.id),
            target_planet=int(target_planet),
            time_arrival=int(time_arrival),
            ships=int(fleet.ships),
            angle_rad=float(fleet.angle),
        ))

    timing_logger.info(
        "fleets.predict.summary | fleets_total={total} | predicted={predicted}",
        total=len(fleets),
        predicted=len(results),
    )
    return results


def enemy_fleets_by_step(obs: dict | object, *, max_lookahead: int = 220) -> list[FleetEnemy]:
    """Return predicted enemy arrivals as `FleetEnemy` records."""
    with timed_calc("fleets.enemy_fleets_by_step", max_lookahead=max_lookahead):
        forecast = build_astronomy_forecast(obs, max_lookahead=max_lookahead)
        return predict_enemy_fleets(obs, forecast, max_lookahead=max_lookahead)


def incremental_enemy_fleets(
    obs: dict | object,
    previous_fleets: list[FleetEnemy],
    forecast,
    *,
    max_lookahead: int = 220,
) -> list[FleetEnemy]:
    """Update enemy fleet predictions incrementally.

    Carries over previously known fleets (decrementing time_arrival by 1) and runs
    trajectory prediction only for fleet IDs that did not exist in the previous step.
    """
    player = int(_get(obs, "player", 0))
    raw_fleets = _get(obs, "fleets", [])
    fleets = [Fleet(*f) for f in raw_fleets]

    enemy_fleet_by_id: dict[int, Fleet] = {int(f.id): f for f in fleets if f.owner != player}
    current_enemy_ids = set(enemy_fleet_by_id.keys())
    previous_ids: set[int] = {fe.id for fe in previous_fleets}

    results: list[FleetEnemy] = []

    for fe in previous_fleets:
        if fe.id not in current_enemy_ids:
            continue
        remaining = fe.time_arrival - 1
        if remaining <= 0:
            continue
        results.append(FleetEnemy(
            id=fe.id,
            target_planet=fe.target_planet,
            time_arrival=remaining,
            ships=int(enemy_fleet_by_id[fe.id].ships),
            angle_rad=fe.angle_rad,
        ))

    n_carried = len(results)
    new_ids = current_enemy_ids - previous_ids
    for fid in new_ids:
        fleet = enemy_fleet_by_id[fid]
        target_planet, time_arrival = _predict_fleet_hit(
            fleet=fleet, max_lookahead=max_lookahead, forecast=forecast,
        )
        if target_planet is not None and time_arrival is not None:
            results.append(FleetEnemy(
                id=int(fleet.id),
                target_planet=int(target_planet),
                time_arrival=int(time_arrival),
                ships=int(fleet.ships),
                angle_rad=float(fleet.angle),
            ))

    timing_logger.info(
        "fleets.incremental.summary | carried={carried} | new_ids={new_ids} | new_predicted={new_predicted} | total={total}",
        carried=n_carried,
        new_ids=len(new_ids),
        new_predicted=len(results) - n_carried,
        total=len(results),
    )
    return results


__all__ = ["enemy_fleets_by_step", "incremental_enemy_fleets", "predict_enemy_fleets"]
