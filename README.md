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

From repository root:

```bash
uv sync
uv run python main.py --turns 30
```

Notes:

- `--turns` sets `episodeSteps` for the run.
- Use up to `--turns 500` to simulate full-length matches.

## Outputs

After execution, logs are written to `outputs/logs`:

- `app.log`
- `timing.log`

Console output includes per-player final `reward` and `status`.

## Orbit Wars Source Reference

Environment source in your local venv:

- `.venv/lib/python3.14/site-packages/kaggle_environments/envs/orbit_wars`

Main rules reference used for this README:

- `.venv/lib/python3.14/site-packages/kaggle_environments/envs/orbit_wars/README.md`
