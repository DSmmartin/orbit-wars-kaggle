"""Academy — PPO training infrastructure for Orbit Wars agents.

Lore module map:
  chronicle   → game state types (PlanetState, FleetState, GameState)
  doctrine    → training configuration (TrainConfig and loaders)
  cartography → feature encoding (encode_turn, TurnBatch)
  tactician   → neural policy network (PlanetPolicy)
  crucible    → PPO algorithm (ppo_update, sample_actions)
  arena       → Kaggle env wrapper (OrbitWarsEnv)
  rivals      → opponent policies (SelfPlayOpponent, KaggleRandomOpponent)
  campaign    → training loop entry point (main)
"""

from .doctrine import TrainConfig, default_train_config_path, load_train_config
from .tactician import PlanetPolicy
from .campaign import main as run_campaign

__all__ = [
    "TrainConfig",
    "default_train_config_path",
    "load_train_config",
    "PlanetPolicy",
    "run_campaign",
]
