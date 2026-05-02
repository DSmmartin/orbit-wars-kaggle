# Academy

PPO training infrastructure for Orbit Wars agents, ported from `rl_solution/src/` with lore-appropriate module names.

## Module Map

| Module | Role |
|---|---|
| `chronicle.py` | Battle record types (`PlanetState`, `FleetState`, `GameState`) parsed from raw Kaggle observations |
| `doctrine.py` | Training configuration (`TrainConfig`, YAML loader) |
| `cartography.py` | Battlefield feature encoder — converts observations into tensors (`encode_turn`, `TurnBatch`); launch angles computed via the ballistic solver (`aim_angle`) |
| `tactician.py` | Neural policy network (`PlanetPolicy`) — actor-critic with per-planet action heads |
| `crucible.py` | PPO algorithm (`ppo_update`, `sample_actions`, `TransitionBatch`) |
| `arena.py` | Kaggle env wrapper (`OrbitWarsEnv`) — handles reset/step, player alternation, terminal reward |
| `rivals.py` | Opponent policies — `NearestPlanetSniperOpponent`, `KaggleRandomOpponent`, `SelfPlayOpponent` |
| `campaign.py` | Training loop entry point (`main`) — rollout collection, PPO updates, checkpoint saving |

## Configuration

Training is controlled by `configs/default.yaml`. Key fields:

```yaml
opponent: sniper        # rivals: sniper | random | self
ppo:
  total_updates: 500
  num_envs: 2
  rollout_steps: 64
save_dir: outputs/rl_checkpoints
```

## Run Training

```bash
uv run python orbit_wars/academy/campaign.py
# custom config:
uv run python orbit_wars/academy/campaign.py --config orbit_wars/academy/configs/default.yaml
```

Checkpoints are saved to `outputs/rl_checkpoints/<run_name>/ckpt_<update>.pt`.

## Use a Checkpoint

```bash
uv run python main.py --checkpoint outputs/rl_checkpoints/orbit_wars_ppo/ckpt_000500.pt --turns 200
```

Player 0 will use the PPO policy; player 1 uses the nearest-planet sniper.

## Action Space

The policy predicts a **target planet index** (one of up to `candidate_count` slots per owned planet). The launch angle and ship count are **not learned** — they are computed deterministically:

- **Angle**: `orbit_wars.army.ballistics.aim_angle(src, tgt, ships, step, angular_velocity)` — accounts for orbital movement of rotating planets, giving a ballistically-correct intercept trajectory.
- **Ships**: `fixed_ship_count(src, tgt)` → `max(tgt.ships + 1, 20)` — always enough to capture.

This means the neural network only needs to decide *who to attack*, not *how to aim*.
