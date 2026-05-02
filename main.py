"""Run a local Orbit Wars match and emit structured logs."""

import io
import logging
import sys
from dataclasses import dataclass

import draccus

_real_stdout = sys.stdout
sys.stdout = io.StringIO()
try:
    from kaggle_environments import make
finally:
    sys.stdout = _real_stdout
logging.getLogger("kaggle_environments.envs.open_spiel_env.open_spiel_env").setLevel(
    logging.WARNING
)

from orbit_wars.observatory import (
    app_logger,
    clear_decisions,
    export_run_artifacts,
    setup_logging,
    timed_calc,
)
from orbit_wars.strategies import build_ppo_agent, nearest_planet_sniper


@dataclass
class RunConfig:
    turns: int = 200
    """Number of steps per episode."""

    episodes: int = 1
    """Number of episodes to run."""

    checkpoint: str = ""
    """Path to a PPO .pt checkpoint (e.g. outputs/rl_checkpoints/orbit_wars_ppo/ckpt_last.pt).
    When set, player 0 uses the PPO policy; player 1 uses nearest_planet_sniper."""


@draccus.wrap()
def main(cfg: RunConfig) -> None:
    setup_logging()

    if cfg.checkpoint:
        agents = [build_ppo_agent(cfg.checkpoint), nearest_planet_sniper]
        app_logger.info(
            "Starting Orbit Wars | mode=ppo_vs_sniper checkpoint={ckpt} turns={turns}",
            ckpt=cfg.checkpoint,
            turns=cfg.turns,
        )
    else:
        agents = [nearest_planet_sniper, nearest_planet_sniper]
        app_logger.info(
            "Starting Orbit Wars | mode=sniper_vs_sniper turns={turns}", turns=cfg.turns
        )

    with timed_calc("main.make_env", turns=cfg.turns):
        env = make("orbit_wars", configuration={"episodeSteps": cfg.turns}, debug=True)

    wins = [0, 0]
    for ep in range(1, cfg.episodes + 1):
        env.reset()
        clear_decisions()
        with timed_calc("main.episode", episode=ep, turns=cfg.turns):
            env.run(agents)

        final = env.steps[-1]
        rewards = [s.reward for s in final]
        statuses = [s.status for s in final]
        for i, (r, st) in enumerate(zip(rewards, statuses)):
            app_logger.info(
                "Episode {ep}/{total} player={player} reward={reward} status={status}",
                ep=ep, total=cfg.episodes, player=i, reward=r, status=st,
            )
        if rewards[0] is not None and rewards[1] is not None:
            if rewards[0] > rewards[1]:
                wins[0] += 1
            elif rewards[1] > rewards[0]:
                wins[1] += 1

    app_logger.info(
        "All episodes done | wins_p0={w0} wins_p1={w1}", w0=wins[0], w1=wins[1]
    )
    out = export_run_artifacts(env)
    app_logger.info("Artifacts exported | path={path}", path=out.resolve())


if __name__ == "__main__":
    main()
