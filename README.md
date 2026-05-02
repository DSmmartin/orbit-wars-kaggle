# Orbit Wars Kaggle Agent

Local project to build and evaluate an agent for Kaggle's `orbit_wars` environment.

## Challenge Context

`orbit_wars` is a strategy game played on a continuous `100 x 100` map. Players launch fleets between planets and try to finish with the highest total number of ships.

Final score is:

- Ships on owned planets/comets
- Ships in active fleets

A game ends when:

- The step limit is reached (typically `500`), or
- Only one player remains with planets/fleets

## Core Mechanics (Why This Is Hard)

- Planets can be static or orbiting, so target positions change over time.
- A central sun destroys fleets that cross it.
- Fleet speed scales with fleet size (larger fleets travel faster).
- Comets periodically appear, can be captured, produce ships, and later leave the map.
- Combat depends on grouped fleet arrivals and defender garrison strength.

Each turn, an agent returns moves in this format:

```python
[[from_planet_id, direction_angle, num_ships], ...]
```

The `direction_angle` is the launch heading in radians. For the PPO agent, this is **not learned** — it is computed by `orbit_wars.army.ballistics.aim_angle`, which solves the ballistic intercept trajectory accounting for orbital movement. The policy only decides *which planet to attack*.

The strategic goal is to balance expansion, defense, and attacks while timing launches against moving targets.

## Refactor Overview

The strategy is split into clear layers:

- `orbit_wars/state`: observation adapters and normalized game state
- `orbit_wars/agents/nearest_planet_sniper`: policy and action-builder logic
- `orbit_wars/fleets`: enemy fleet prediction utilities
- `orbit_wars/astronomy`: planet/comet trajectory forecast support
- `orbit_wars/army`: low-level ballistics and physics helpers
- `orbit_wars/strategies`: Kaggle-compatible wrapper entrypoints
- `orbit_wars/observatory`: app/timing logging utilities

## Run The Project

Install dependencies and run a quick heuristic match:

```bash
uv sync
uv run python main.py --turns 30
```

Notes:

- `--turns` sets `episodeSteps` for the run.
- Use up to `--turns 500` to simulate full-length matches.
- Default mode pits `nearest_planet_sniper` vs itself.

## Training a PPO Agent

The `orbit_wars/academy/` module contains the full PPO training pipeline.

### 1. Configure

Edit `orbit_wars/academy/configs/default.yaml` to tune training:

```yaml
opponent: sniper          # rivals: sniper | random | self
ppo:
  total_updates: 500      # increase for serious training (2000+)
  num_envs: 2
  rollout_steps: 64
save_dir: outputs/rl_checkpoints
```

### 2. Train

```bash
uv run python train.py
```

Checkpoints are saved to `outputs/rl_checkpoints/<run_name>/` at the interval set by `checkpoint_every`. `ckpt_last.pt` is always overwritten with the most recent update.

### 3. Evaluate

Pit the trained policy (player 0) against the nearest-planet sniper (player 1):

```bash
uv run python main.py --checkpoint outputs/rl_checkpoints/orbit_wars_ppo/ckpt_last.pt --turns 500
```

Pass a specific checkpoint to compare snapshots from different stages of training:

```bash
uv run python main.py --checkpoint outputs/rl_checkpoints/orbit_wars_ppo/ckpt_000200.pt --turns 500
```

## Outputs

After execution, logs are written to `outputs/logs`:

- `app.log`
- `timing.log`

Console output includes per-player final `reward` and `status`.

## Project Structure

```
orbit_wars/
  academy/        # PPO training pipeline (campaign, arena, rivals, tactician, …)
  agents/         # Heuristic agent logic (nearest_planet_sniper)
  army/           # Ballistics and physics helpers
  astronomy/      # Planet/comet trajectory forecasting
  fleets/         # Enemy fleet prediction
  observatory/    # Logging and timing utilities
  state/          # Observation adapters and game state models
  strategies/     # Kaggle-compatible agent entrypoints
rl_solution/      # Original reference notebook code (kept for reference)
outputs/
  rl_checkpoints/ # PPO checkpoints produced by academy training
  logs/           # app.log, timing.log
```

## Orbit Wars Source Reference

Environment source in your local venv:

- `.venv/lib/python3.14/site-packages/kaggle_environments/envs/orbit_wars`

Main rules reference used for this README:

- `.venv/lib/python3.14/site-packages/kaggle_environments/envs/orbit_wars/README.md`
