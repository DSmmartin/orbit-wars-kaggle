import argparse

from kaggle_environments import make
from orbit_wars.observatory import clear_decisions, export_run_artifacts
from orbit_wars.strategies import nearest_planet_sniper


parser = argparse.ArgumentParser()
parser.add_argument(
    "--turns", type=int, default=30, help="Number of episode steps (default: 500)"
)
args = parser.parse_args()

env = make("orbit_wars", configuration={"episodeSteps": args.turns}, debug=True)

clear_decisions()
env.run([nearest_planet_sniper, "random"])

final = env.steps[-1]
for i, s in enumerate(final):
    print(f"Player {i}: reward={s.reward}, status={s.status}")

out = export_run_artifacts(env)
print(f"Artifacts exported to: {out.resolve()}")
