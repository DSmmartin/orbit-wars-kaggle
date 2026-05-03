"""Run a local Orbit Wars match and emit structured logs."""

import io
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

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
from orbit_wars.strategies import (
    SNIPER_TIERS,
    build_ppo_agent,
    build_tiered_sniper_agent,
    nearest_planet_sniper,
)

_VALID_OPPONENTS = list(SNIPER_TIERS)


@dataclass
class RunConfig:
    turns: int = 200
    """Number of steps per episode."""

    episodes: int = 1
    """Number of episodes to run."""

    checkpoint: str = ""
    """Path to a PPO .pt checkpoint. When set, player 0 uses the PPO policy."""

    opponent: str = "sniper"
    """Player 1 strategy: sniper | sniper_95 | sniper_90 | sniper_80 | sniper_50"""

    replays_dir: str = "outputs/replays"
    """Directory where per-episode replay folders are saved."""


@dataclass
class _EpisodeResult:
    ep: int
    reward_p0: float | None
    reward_p1: float | None
    status_p0: str
    status_p1: str

    @property
    def outcome(self) -> str:
        if self.reward_p0 is None or self.reward_p1 is None:
            return "?"
        if self.reward_p0 > self.reward_p1:
            return "p0"
        if self.reward_p1 > self.reward_p0:
            return "p1"
        return "draw"


def _print_summary(results: list[_EpisodeResult], p0_label: str, p1_label: str, replays_dir: Path) -> None:
    wins_p0 = sum(1 for r in results if r.outcome == "p0")
    wins_p1 = sum(1 for r in results if r.outcome == "p1")
    draws = sum(1 for r in results if r.outcome == "draw")

    w = 68
    print()
    print("═" * w)
    print(f"{'BATTLE SUMMARY':^{w}}")
    print("═" * w)
    print(f"  p0={p0_label}  vs  p1={p1_label}")
    print("  " + "─" * (w - 2))
    print(f"  {'ep':<5} {'p0 reward':>10} {'p1 reward':>10}  {'result':<10}  status")
    print("  " + "─" * (w - 2))
    for r in results:
        r0 = f"{r.reward_p0}" if r.reward_p0 is not None else "—"
        r1 = f"{r.reward_p1}" if r.reward_p1 is not None else "—"
        outcome_label = {"p0": "p0 wins", "p1": "p1 wins", "draw": "draw", "?": "?"}.get(r.outcome, "?")
        print(
            f"  {r.ep:<5} {r0:>10} {r1:>10}  {outcome_label:<10}"
            f"  {r.status_p0} / {r.status_p1}"
        )
    print("  " + "─" * (w - 2))
    print(f"  wins p0={wins_p0}  wins p1={wins_p1}  draws={draws}  total={len(results)}")
    print("═" * w)
    print(f"  replays → {replays_dir.resolve()}")
    print("═" * w)
    print()


@draccus.wrap()
def main(cfg: RunConfig) -> None:
    setup_logging()

    if cfg.opponent not in _VALID_OPPONENTS:
        raise SystemExit(f"Unknown opponent {cfg.opponent!r}. Valid: {_VALID_OPPONENTS}")

    opponent_agent = build_tiered_sniper_agent(cfg.opponent)
    accuracy_pct = int(SNIPER_TIERS[cfg.opponent] * 100)
    p1_label = f"{cfg.opponent} ({accuracy_pct}%)"

    if cfg.checkpoint:
        agents = [build_ppo_agent(cfg.checkpoint), opponent_agent]
        p0_label = "ppo"
        app_logger.info(
            "Starting Orbit Wars | p0=ppo p1={opponent} checkpoint={ckpt} turns={turns} episodes={ep}",
            opponent=cfg.opponent,
            ckpt=cfg.checkpoint,
            turns=cfg.turns,
            ep=cfg.episodes,
        )
    else:
        agents = [nearest_planet_sniper, opponent_agent]
        p0_label = "sniper (100%)"
        app_logger.info(
            "Starting Orbit Wars | p0=sniper p1={opponent} turns={turns} episodes={ep}",
            opponent=cfg.opponent,
            turns=cfg.turns,
            ep=cfg.episodes,
        )

    with timed_calc("combat.make_env", turns=cfg.turns):
        env = make("orbit_wars", configuration={"episodeSteps": cfg.turns}, debug=True)

    replays_dir = Path(cfg.replays_dir)
    results: list[_EpisodeResult] = []

    for ep in range(1, cfg.episodes + 1):
        env.reset()
        clear_decisions()
        with timed_calc("combat.episode", episode=ep, turns=cfg.turns):
            env.run(agents)

        final = env.steps[-1]
        rewards = [s.reward for s in final]
        statuses = [s.status for s in final]

        result = _EpisodeResult(
            ep=ep,
            reward_p0=rewards[0],
            reward_p1=rewards[1],
            status_p0=statuses[0],
            status_p1=statuses[1],
        )
        results.append(result)

        for i, (r, st) in enumerate(zip(rewards, statuses)):
            app_logger.info(
                "Episode {ep}/{total} player={player} reward={reward} status={status}",
                ep=ep,
                total=cfg.episodes,
                player=i,
                reward=r,
                status=st,
            )

        episode_dir = replays_dir / f"episode_{ep:03d}"
        export_run_artifacts(env, output_dir=episode_dir)
        app_logger.info(
            "Episode {ep} artifacts saved | path={path}",
            ep=ep,
            path=episode_dir.resolve(),
        )

    _print_summary(results, p0_label, p1_label, replays_dir)


if __name__ == "__main__":
    main()
