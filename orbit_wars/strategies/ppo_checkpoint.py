"""PPO checkpoint strategy — loads a trained policy and returns a Kaggle-compatible agent."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


def _register_module_aliases() -> None:
    """Register sys.modules aliases so both academy and rl_solution checkpoints deserialise.

    Academy is the canonical source. Old rl_solution checkpoints pickle their config as
    src.config.TrainConfig — aliasing src.* → orbit_wars.academy.* lets torch.load find
    the class without an ImportError.
    """
    from orbit_wars.academy import arena, campaign, cartography, chronicle, crucible, doctrine, rivals, tactician

    _alias_map = {
        "config": doctrine,
        "features": cartography,
        "policy": tactician,
        "ppo": crucible,
        "game_types": chronicle,
        "opponents": rivals,
        "env": arena,
        "train": campaign,
    }
    for short, mod in _alias_map.items():
        sys.modules.setdefault(f"src.{short}", mod)
        sys.modules.setdefault(f"src.rl_template.{short}", mod)


def build_ppo_agent(checkpoint_path: str | Path):
    """Return a Kaggle-compatible agent callable backed by a PPO checkpoint.

    Accepts checkpoints from both the academy campaign (outputs/rl_checkpoints/)
    and the legacy rl_solution training scripts.

    Args:
        checkpoint_path: Path to a .pt checkpoint file.

    Returns:
        A function ``agent(obs) -> list[list]`` ready for ``env.run()``.
    """
    _register_module_aliases()

    from orbit_wars.academy.cartography import (
        candidate_feature_dim,
        encode_turn,
        global_feature_dim,
        self_feature_dim,
    )
    from orbit_wars.academy.crucible import sample_actions
    from orbit_wars.academy.doctrine import default_train_config_path, load_train_config
    from orbit_wars.academy.tactician import PlanetPolicy

    cfg = load_train_config(default_train_config_path())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = PlanetPolicy(
        self_dim=self_feature_dim(),
        candidate_dim=candidate_feature_dim(),
        global_dim=global_feature_dim(),
        candidate_count=cfg.env.candidate_count,
        hidden_size=cfg.model.hidden_size,
    ).to(device)

    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    policy.load_state_dict(ckpt.get("policy", ckpt))
    policy.eval()

    def _agent(obs) -> list[list]:
        batch = encode_turn(obs, cfg.env, env_index=0)
        if batch.self_features.shape[0] == 0:
            return []
        with torch.inference_mode():
            outputs = policy(
                torch.from_numpy(batch.self_features).to(device),
                torch.from_numpy(batch.candidate_features).to(device),
                torch.from_numpy(batch.global_features).to(device),
                torch.from_numpy(batch.candidate_mask).to(device).bool(),
            )
            sampled = sample_actions(outputs, deterministic=False)
        target_indices = sampled.target_index.detach().cpu().numpy()
        moves = []
        for row_idx, context in enumerate(batch.contexts):
            target_idx = int(target_indices[row_idx])
            if target_idx == 0 or target_idx >= len(context.candidate_ids):
                continue
            if not context.candidate_mask[target_idx]:
                continue
            ships = int(context.ship_counts[target_idx])
            if ships <= 0:
                continue
            moves.append(
                [context.source_id, float(context.target_angles[target_idx]), ships]
            )
        return moves

    return _agent


__all__ = ["build_ppo_agent"]
