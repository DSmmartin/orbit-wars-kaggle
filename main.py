"""Run a local Orbit Wars match and emit structured logs."""

import argparse

from kaggle_environments import make
from orbit_wars.observatory import app_logger, clear_decisions, export_run_artifacts, setup_logging, timed_calc
from orbit_wars.strategies import nearest_planet_sniper


parser = argparse.ArgumentParser()
parser.add_argument(
    "--turns", type=int, default=200, help="Number of steps per episode (default: 200)"
)
args = parser.parse_args()
setup_logging()
app_logger.info(
    "Starting Orbit Wars | episodes=30 turns={turns}",
    turns=args.turns,
)

with timed_calc("main.make_env", turns=args.turns):
    env = make("orbit_wars", configuration={"episodeSteps": args.turns}, debug=True)

EPISODES = 1
wins = [0, 0]
for ep in range(1, EPISODES + 1):
    env.reset()
    clear_decisions()
    with timed_calc("main.episode", episode=ep, turns=args.turns):
        env.run([nearest_planet_sniper, "random"])

    final = env.steps[-1]
    rewards = [s.reward for s in final]
    statuses = [s.status for s in final]
    for i, (r, st) in enumerate(zip(rewards, statuses)):
        app_logger.info(
            "Episode {ep}/{total} player={player} reward={reward} status={status}",
            ep=ep,
            total=EPISODES,
            player=i,
            reward=r,
            status=st,
        )
    if rewards[0] is not None and rewards[1] is not None:
        if rewards[0] > rewards[1]:
            wins[0] += 1
        elif rewards[1] > rewards[0]:
            wins[1] += 1

app_logger.info(
    "All episodes done | episodes=30 wins_p0={w0} wins_p1={w1}",
    w0=wins[0],
    w1=wins[1],
)

out = export_run_artifacts(env)
app_logger.info("Artifacts exported | path={path}", path=out.resolve())
