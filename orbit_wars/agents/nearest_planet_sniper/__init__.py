"""Nearest-planet sniper agent package."""

from __future__ import annotations

from orbit_wars.agents.nearest_planet_sniper.action_builder import ShotDecision, build_moves
from orbit_wars.agents.nearest_planet_sniper.config import SniperPolicyConfig
from orbit_wars.agents.nearest_planet_sniper.policy import choose_shot_decisions
from orbit_wars.observatory import app_logger, record_decision, record_enemy_fleet
from orbit_wars.state import GameState, build_game_state

_previous_state: GameState | None = None


def nearest_planet_sniper_agent(obs) -> list[list[float | int]]:
    """Entry point compatible with Kaggle environment agent callbacks."""
    global _previous_state
    state = build_game_state(obs, previous_state=_previous_state)
    _previous_state = state
    decisions = choose_shot_decisions(state, config=SniperPolicyConfig())

    app_logger.debug(
        "Step {step} | player={player} | my_planets={my_planets} | targets={targets} | enemy_fleets={enemy_fleets} | decisions={decisions}",
        step=state.current_step,
        player=state.player,
        my_planets=len(state.my_planets()),
        targets=len(state.target_planets()),
        enemy_fleets=len(state.enemy_fleets),
        decisions=len(decisions),
    )

    for d in decisions:
        record_decision(
            step=state.current_step,
            player=state.player,
            source_planet_id=d.source_planet_id,
            target_planet_id=d.target_planet_id,
            angle_rad=d.angle_rad,
            ships=d.ships,
        )

    for fleet in state.enemy_fleets:
        record_enemy_fleet(
            step=state.current_step,
            fleet_id=fleet.id,
            target_planet_id=fleet.target_planet,
            time_arrival=fleet.time_arrival,
            ships=fleet.ships,
            angle_rad=fleet.angle_rad,
        )

    return build_moves(decisions)


__all__ = [
    "ShotDecision",
    "SniperPolicyConfig",
    "build_moves",
    "choose_shot_decisions",
    "nearest_planet_sniper_agent",
]
