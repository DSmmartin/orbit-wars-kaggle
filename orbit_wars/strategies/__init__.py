"""Public strategy entry points."""

from __future__ import annotations

import math
import random

from .nearest_planet_sniper import nearest_planet_sniper
from .ppo_checkpoint import build_ppo_agent

SNIPER_TIERS: dict[str, float] = {
    "sniper":    1.00,
    "sniper_95": 0.95,
    "sniper_90": 0.90,
    "sniper_80": 0.80,
    "sniper_50": 0.50,
}


def build_tiered_sniper_agent(tier: str):
    """Return a Kaggle-compatible agent callable for the given sniper tier.

    For 100% accuracy returns the standard sniper directly.
    For lower tiers, each shot has a (1 - accuracy) chance of firing a random angle.
    """
    accuracy = SNIPER_TIERS.get(tier)
    if accuracy is None:
        valid = list(SNIPER_TIERS)
        raise ValueError(f"Unknown sniper tier: {tier!r}. Valid: {valid}")

    if accuracy >= 1.0:
        return nearest_planet_sniper

    from orbit_wars.agents.nearest_planet_sniper import (
        ShotDecision, SniperPolicyConfig, build_moves, choose_shot_decisions,
    )
    from orbit_wars.state import build_game_state

    _state = [None]

    def _agent(obs):
        state = build_game_state(obs, previous_state=_state[0])
        _state[0] = state
        decisions = choose_shot_decisions(state, config=SniperPolicyConfig())
        corrupted: list[ShotDecision] = []
        for d in decisions:
            if random.random() > accuracy:
                d = ShotDecision(
                    source_planet_id=d.source_planet_id,
                    target_planet_id=d.target_planet_id,
                    angle_rad=random.uniform(-math.pi, math.pi),
                    ships=d.ships,
                )
            corrupted.append(d)
        return build_moves(corrupted)

    return _agent


__all__ = ["nearest_planet_sniper", "build_ppo_agent", "build_tiered_sniper_agent", "SNIPER_TIERS"]
