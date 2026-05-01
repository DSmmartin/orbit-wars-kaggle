"""Adapters that convert raw environment observations into typed state models."""

from __future__ import annotations

from typing import Any

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from orbit_wars.astronomy import build_astronomy_forecast, shift_astronomy_forecast
from orbit_wars.fleets import incremental_enemy_fleets, predict_enemy_fleets
from orbit_wars.observatory import timed_calc
from orbit_wars.state.models import GameState


def _get(obs: dict[str, Any] | object, key: str, default: Any) -> Any:
    """Read observation data from dict-like or object-like payloads."""
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def build_game_state(
    obs: dict[str, Any] | object,
    previous_state: GameState | None = None,
) -> GameState:
    """Build a `GameState` from a raw observation.

    When `previous_state` is provided and is from the immediately preceding step, the
    astronomy forecast is shifted by 1 step (reusing 219/220 pre-computed steps) and
    enemy fleets are updated incrementally — only new fleet IDs are predicted.
    """
    player = int(_get(obs, "player", 0))
    current_step = int(_get(obs, "step", 0))

    raw_planets = _get(obs, "planets", [])
    planets = [Planet(*planet) for planet in raw_planets]

    raw_initial = _get(obs, "initial_planets", [])
    initial_by_id = {int(planet[0]): Planet(*planet) for planet in raw_initial}

    raw_comets = list(_get(obs, "comets", []))
    comet_planet_ids = [int(planet_id) for planet_id in _get(obs, "comet_planet_ids", [])]
    angular_velocity = float(_get(obs, "angular_velocity", 0.0))

    prev = previous_state
    # Invalidate previous state if it is not from the immediately preceding step.
    if prev is not None and prev.current_step + 1 != current_step:
        prev = None

    if prev is not None and prev.astronomy_forecast is not None:
        with timed_calc("state.build_game_state.incremental", step=current_step):
            forecast = shift_astronomy_forecast(prev.astronomy_forecast, obs)
            enemy_fleets = incremental_enemy_fleets(obs, prev.enemy_fleets, forecast)
    else:
        with timed_calc("state.build_game_state.full", step=current_step):
            forecast = build_astronomy_forecast(obs)
            enemy_fleets = predict_enemy_fleets(obs, forecast)

    return GameState(
        player=player,
        current_step=current_step,
        angular_velocity=angular_velocity,
        planets=planets,
        initial_by_id=initial_by_id,
        raw_comets=raw_comets,
        comet_planet_ids=comet_planet_ids,
        enemy_fleets=enemy_fleets,
        astronomy_forecast=forecast,
    )
