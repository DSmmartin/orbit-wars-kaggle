"""Policy logic for selecting nearest valid target shots."""

from __future__ import annotations

import math

from orbit_wars.agents.nearest_planet_sniper.action_builder import ShotDecision
from orbit_wars.agents.nearest_planet_sniper.config import SniperPolicyConfig
from orbit_wars.army.ballistics import plan_shot
from orbit_wars.state import GameState


def _ships_needed_for_target(target_ships: int, config: SniperPolicyConfig) -> int:
    """Compute ships required to reliably capture a target."""
    return max(int(target_ships) + 1, config.minimum_ships_to_send)


def choose_shot_decisions(
    state: GameState,
    *,
    config: SniperPolicyConfig,
) -> list[ShotDecision]:
    """Select at most one valid shot per owned planet using nearest-target ranking."""
    decisions: list[ShotDecision] = []
    targets = state.target_planets()

    if not targets:
        return decisions

    for mine in state.my_planets():
        ordered_targets = sorted(
            targets,
            key=lambda target: math.hypot(mine.x - target.x, mine.y - target.y),
        )

        for target in ordered_targets:
            ships_needed = _ships_needed_for_target(int(target.ships), config)
            if mine.ships < ships_needed:
                continue

            shot = plan_shot(
                mine,
                target,
                ships=ships_needed,
                current_step=state.current_step,
                angular_velocity=state.angular_velocity,
                initial_to_planet=state.initial_by_id.get(target.id),
                planets=state.planets,
                initial_planets=state.initial_by_id,
                comets=state.raw_comets,
                comet_planet_ids=state.comet_planet_ids,
                max_speed=config.max_speed,
                max_iterations=config.max_iterations,
                step_tolerance=config.step_tolerance,
                evaluation_horizon=config.evaluation_horizon,
            )
            if not shot.valid:
                continue

            decisions.append(
                ShotDecision(
                    source_planet_id=mine.id,
                    target_planet_id=target.id,
                    angle_rad=shot.angle,
                    ships=ships_needed,
                )
            )
            break

    return decisions
