from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from orbit_wars.army.physics import CENTER, ROTATION_RADIUS_LIMIT
from orbit_wars.observatory import timed_calc, timing_logger


@dataclass(frozen=True)
class StepForecast:
    """Planet centers and swept segments for one forecasted future step."""

    planet_ids: np.ndarray
    centers: np.ndarray
    radii: np.ndarray
    segment_planet_ids: np.ndarray
    segment_starts: np.ndarray
    segment_ends: np.ndarray
    segment_radii: np.ndarray


@dataclass(frozen=True)
class AstronomyForecast:
    """Cached multi-step forecast used by fleet interception predictions."""

    horizon: int
    steps: list[StepForecast | None]
    signature: tuple


_FORECAST_CACHE: dict[tuple, AstronomyForecast] = {}


def _get(obs: dict | object, key: str, default):
    """Read observation fields from either dict or object payloads."""
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def segment_to_points_distance(
    points: np.ndarray,
    segment_start: tuple[float, float],
    segment_end: tuple[float, float],
) -> np.ndarray:
    """Vectorized point-to-segment distance for many points."""
    if points.size == 0:
        return np.empty((0,), dtype=np.float64)

    start = np.asarray(segment_start, dtype=np.float64)
    end = np.asarray(segment_end, dtype=np.float64)
    seg = end - start
    l2 = float(np.dot(seg, seg))

    if l2 == 0.0:
        return np.linalg.norm(points - start, axis=1)

    t = np.clip(np.sum((points - start) * seg, axis=1) / l2, 0.0, 1.0)
    projection = start + t[:, None] * seg
    return np.linalg.norm(points - projection, axis=1)


def point_to_segments_distance(
    point: tuple[float, float],
    segment_starts: np.ndarray,
    segment_ends: np.ndarray,
) -> np.ndarray:
    """Vectorized point-to-segments distance for many segments."""
    if segment_starts.size == 0:
        return np.empty((0,), dtype=np.float64)

    p = np.asarray(point, dtype=np.float64)
    seg = segment_ends - segment_starts
    l2 = np.sum(seg * seg, axis=1)

    t = np.zeros_like(l2, dtype=np.float64)
    non_zero = l2 > 0.0
    if np.any(non_zero):
        rel = p - segment_starts[non_zero]
        dot = np.sum(rel * seg[non_zero], axis=1)
        t[non_zero] = np.clip(dot / l2[non_zero], 0.0, 1.0)

    projection = segment_starts + t[:, None] * seg
    return np.linalg.norm(projection - p, axis=1)


def _forecast_signature(obs: dict | object, max_lookahead: int) -> tuple:
    """Build a cache key capturing observation fields relevant to motion."""
    step = int(_get(obs, "step", 0))
    angular_velocity = round(float(_get(obs, "angular_velocity", 0.0)), 10)
    raw_planets = _get(obs, "planets", [])
    raw_comets = _get(obs, "comets", [])
    comet_planet_ids = tuple(int(pid) for pid in _get(obs, "comet_planet_ids", []))

    planets_sig = tuple(
        (
            int(p[0]),
            int(p[1]),
            round(float(p[2]), 6),
            round(float(p[3]), 6),
            round(float(p[4]), 6),
        )
        for p in raw_planets
    )
    comets_sig = tuple(
        (
            int(c.get("path_index", 0)),
            tuple(int(pid) for pid in c.get("planet_ids", [])),
        )
        for c in raw_comets
    )
    return (step, angular_velocity, comet_planet_ids, planets_sig, comets_sig, max_lookahead)


def build_astronomy_forecast(
    obs: dict | object,
    *,
    max_lookahead: int = 220,
) -> AstronomyForecast:
    """Compute or reuse a cached astronomy forecast for future movement."""
    signature = _forecast_signature(obs, max_lookahead)
    cached = _FORECAST_CACHE.get(signature)
    if cached is not None:
        timing_logger.info("astronomy.forecast.cache_hit | horizon={horizon}", horizon=max_lookahead)
        return cached

    with timed_calc("astronomy.build_forecast", horizon=max_lookahead):
        raw_planets = _get(obs, "planets", [])
        raw_initial_planets = _get(obs, "initial_planets", [])
        raw_comets = deepcopy(_get(obs, "comets", []))
        angular_velocity = float(_get(obs, "angular_velocity", 0.0))
        current_step = int(_get(obs, "step", 0))

        initial_by_id = {int(p[0]): p for p in raw_initial_planets}
        radius_by_id = {int(p[0]): float(p[4]) for p in raw_planets}
        active_ids = [int(p[0]) for p in raw_planets]
        comet_ids = {int(pid) for pid in _get(obs, "comet_planet_ids", [])}

        positions: dict[int, np.ndarray] = {
            int(p[0]): np.asarray((float(p[2]), float(p[3])), dtype=np.float64) for p in raw_planets
        }

        orbit_params: dict[int, tuple[bool, float, float, float]] = {}
        for pid, initial in initial_by_id.items():
            dx = float(initial[2]) - CENTER
            dy = float(initial[3]) - CENTER
            orbital_radius = math.hypot(dx, dy)
            initial_angle = math.atan2(dy, dx)
            rotating = orbital_radius + radius_by_id.get(pid, float(initial[4])) < ROTATION_RADIUS_LIMIT
            orbit_params[pid] = (rotating, orbital_radius, initial_angle, float(initial[4]))

        steps: list[StepForecast | None] = [None] * (max_lookahead + 1)
        for dt in range(1, max_lookahead + 1):
            start_ids = list(active_ids)
            start_positions = {pid: positions[pid].copy() for pid in start_ids}

            if start_ids:
                centers = np.vstack([start_positions[pid] for pid in start_ids]).astype(np.float64)
                radii = np.asarray([radius_by_id[pid] for pid in start_ids], dtype=np.float64)
                planet_ids = np.asarray(start_ids, dtype=np.int64)
            else:
                centers = np.empty((0, 2), dtype=np.float64)
                radii = np.empty((0,), dtype=np.float64)
                planet_ids = np.empty((0,), dtype=np.int64)

            for pid in start_ids:
                if pid in comet_ids:
                    continue
                params = orbit_params.get(pid)
                if params is None:
                    continue
                rotating, orbital_radius, initial_angle, _ = params
                if not rotating:
                    continue
                angle = initial_angle + angular_velocity * (current_step + dt - 1)
                positions[pid] = np.asarray(
                    (
                        CENTER + orbital_radius * math.cos(angle),
                        CENTER + orbital_radius * math.sin(angle),
                    ),
                    dtype=np.float64,
                )

            expired_ids: set[int] = set()
            for group in raw_comets:
                group["path_index"] += 1
                idx = int(group["path_index"])
                updated_ids: list[int] = []
                for i, pid_value in enumerate(group.get("planet_ids", [])):
                    pid = int(pid_value)
                    if pid not in positions:
                        continue
                    path = group["paths"][i]
                    if idx >= len(path):
                        expired_ids.add(pid)
                        continue
                    point = path[idx]
                    positions[pid] = np.asarray((float(point[0]), float(point[1])), dtype=np.float64)
                    updated_ids.append(pid)
                group["planet_ids"] = updated_ids

            segment_ids: list[int] = []
            segment_starts: list[np.ndarray] = []
            segment_ends: list[np.ndarray] = []
            segment_radii: list[float] = []
            for pid in start_ids:
                if pid in expired_ids:
                    continue
                new_pos = positions.get(pid)
                if new_pos is None:
                    continue
                segment_ids.append(pid)
                segment_starts.append(start_positions[pid])
                segment_ends.append(new_pos.copy())
                segment_radii.append(radius_by_id[pid])

            if segment_ids:
                seg_ids_arr = np.asarray(segment_ids, dtype=np.int64)
                seg_starts_arr = np.vstack(segment_starts).astype(np.float64)
                seg_ends_arr = np.vstack(segment_ends).astype(np.float64)
                seg_radii_arr = np.asarray(segment_radii, dtype=np.float64)
            else:
                seg_ids_arr = np.empty((0,), dtype=np.int64)
                seg_starts_arr = np.empty((0, 2), dtype=np.float64)
                seg_ends_arr = np.empty((0, 2), dtype=np.float64)
                seg_radii_arr = np.empty((0,), dtype=np.float64)

            steps[dt] = StepForecast(
                planet_ids=planet_ids,
                centers=centers,
                radii=radii,
                segment_planet_ids=seg_ids_arr,
                segment_starts=seg_starts_arr,
                segment_ends=seg_ends_arr,
                segment_radii=seg_radii_arr,
            )

            if expired_ids:
                active_ids = [pid for pid in active_ids if pid not in expired_ids]
                for pid in expired_ids:
                    positions.pop(pid, None)
                    comet_ids.discard(pid)

        forecast = AstronomyForecast(horizon=max_lookahead, steps=steps, signature=signature)
        _FORECAST_CACHE[signature] = forecast
        if len(_FORECAST_CACHE) > 8:
            oldest_key = next(iter(_FORECAST_CACHE))
            _FORECAST_CACHE.pop(oldest_key, None)
        return forecast


def _comet_position_at_dt(pid: int, dt: int, raw_comets: list) -> np.ndarray | None:
    """Return a comet's position at path_index + dt, or None if the comet has expired."""
    for group in raw_comets:
        for i, comet_pid in enumerate(group.get("planet_ids", [])):
            if int(comet_pid) == pid:
                idx = int(group["path_index"]) + dt
                path = group["paths"][i]
                if idx >= len(path):
                    return None
                pt = path[idx]
                return np.asarray((float(pt[0]), float(pt[1])), dtype=np.float64)
    return None


def _compute_terminal_step(
    obs: dict | object,
    prev_step: StepForecast,
    dt: int,
) -> StepForecast | None:
    """Compute the single new terminal StepForecast for a shifted forecast.

    prev_step is old_steps[max_lookahead]; its segment_ends are the start positions
    for the new terminal step, and its segment_planet_ids are the still-alive planets.
    """
    if prev_step.segment_planet_ids.size == 0:
        return None

    raw_planets = _get(obs, "planets", [])
    raw_initial_planets = _get(obs, "initial_planets", [])
    raw_comets = _get(obs, "comets", [])
    angular_velocity = float(_get(obs, "angular_velocity", 0.0))
    current_step = int(_get(obs, "step", 0))

    initial_by_id = {int(p[0]): p for p in raw_initial_planets}
    radius_by_id = {int(p[0]): float(p[4]) for p in raw_planets}
    comet_ids = {int(pid) for pid in _get(obs, "comet_planet_ids", [])}

    orbit_params: dict[int, tuple[bool, float, float]] = {}
    for pid, initial in initial_by_id.items():
        r = radius_by_id.get(pid, float(initial[4]))
        dx = float(initial[2]) - CENTER
        dy = float(initial[3]) - CENTER
        orbital_radius = math.hypot(dx, dy)
        initial_angle = math.atan2(dy, dx)
        orbit_params[pid] = (orbital_radius + r < ROTATION_RADIUS_LIMIT, orbital_radius, initial_angle)

    active_ids = [int(pid) for pid in prev_step.segment_planet_ids]
    start_pos_by_id: dict[int, np.ndarray] = {
        int(pid): prev_step.segment_ends[i]
        for i, pid in enumerate(prev_step.segment_planet_ids)
    }

    segment_ids: list[int] = []
    segment_starts_list: list[np.ndarray] = []
    segment_ends_list: list[np.ndarray] = []
    segment_radii_list: list[float] = []
    expired_ids: set[int] = set()

    for pid in active_ids:
        start = start_pos_by_id.get(pid)
        if start is None:
            continue
        if pid in comet_ids:
            end = _comet_position_at_dt(pid, dt, raw_comets)
            if end is None:
                expired_ids.add(pid)
                continue
        else:
            params = orbit_params.get(pid)
            if params is None:
                continue
            rotating, orbital_radius, initial_angle = params
            if not rotating:
                end = start
            else:
                angle = initial_angle + angular_velocity * (current_step + dt - 1)
                end = np.asarray(
                    (CENTER + orbital_radius * math.cos(angle), CENTER + orbital_radius * math.sin(angle)),
                    dtype=np.float64,
                )
        segment_ids.append(pid)
        segment_starts_list.append(start)
        segment_ends_list.append(end)
        segment_radii_list.append(radius_by_id.get(pid, 1.0))

    centers_arr = np.vstack([start_pos_by_id[pid] for pid in active_ids]).astype(np.float64)
    radii_arr = np.asarray([radius_by_id.get(pid, 1.0) for pid in active_ids], dtype=np.float64)
    planet_ids_arr = np.asarray(active_ids, dtype=np.int64)

    if segment_ids:
        seg_ids = np.asarray(segment_ids, dtype=np.int64)
        seg_starts = np.vstack(segment_starts_list).astype(np.float64)
        seg_ends = np.vstack(segment_ends_list).astype(np.float64)
        seg_radii = np.asarray(segment_radii_list, dtype=np.float64)
    else:
        seg_ids = np.empty((0,), dtype=np.int64)
        seg_starts = np.empty((0, 2), dtype=np.float64)
        seg_ends = np.empty((0, 2), dtype=np.float64)
        seg_radii = np.empty((0,), dtype=np.float64)

    return StepForecast(
        planet_ids=planet_ids_arr,
        centers=centers_arr,
        radii=radii_arr,
        segment_planet_ids=seg_ids,
        segment_starts=seg_starts,
        segment_ends=seg_ends,
        segment_radii=seg_radii,
    )


def shift_astronomy_forecast(
    previous_forecast: AstronomyForecast,
    obs: dict | object,
    *,
    max_lookahead: int = 220,
) -> AstronomyForecast:
    """Shift a forecast forward by 1 step.

    Reuses steps[2..max_lookahead] from the previous forecast as steps[1..max_lookahead-1],
    then computes only the single new terminal step. This replaces a full 220-step recomputation
    with a single-step calculation.
    """
    old_steps = previous_forecast.steps
    new_steps: list[StepForecast | None] = [None] * (max_lookahead + 1)

    for k in range(1, max_lookahead):
        new_steps[k] = old_steps[k + 1] if k + 1 <= max_lookahead else None

    prev_terminal = old_steps[max_lookahead]
    if prev_terminal is not None:
        new_steps[max_lookahead] = _compute_terminal_step(obs, prev_terminal, dt=max_lookahead)

    signature = _forecast_signature(obs, max_lookahead)
    forecast = AstronomyForecast(horizon=max_lookahead, steps=new_steps, signature=signature)
    _FORECAST_CACHE[signature] = forecast
    if len(_FORECAST_CACHE) > 8:
        oldest_key = next(iter(_FORECAST_CACHE))
        _FORECAST_CACHE.pop(oldest_key, None)
    return forecast


__all__ = [
    "AstronomyForecast",
    "StepForecast",
    "build_astronomy_forecast",
    "shift_astronomy_forecast",
    "segment_to_points_distance",
    "point_to_segments_distance",
]
