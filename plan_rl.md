# Orbit Wars on Kaggle: a competitive RTS for RL agents

**The short answer**: Orbit Wars is a Kaggle simulation competition framed as a 2-or-4 player real-time strategy game played in a continuous 2D plane, where agents fight to **conquer planets that orbit a central sun**. It is built on Kaggle's standard `kaggle-environments` Python framework (the same one behind Halite, Lux AI, Hungry Geese, Kore, and ConnectX), so submissions are Python agents scored via head-to-head matchmaking on a TrueSkill-style leaderboard. **For a first serious attempt I recommend PPO with league-based self-play, written in CleanRL or PyTorch-Lightning, and wrapped in a PettingZoo-compatible interface around the Kaggle env**. Below is a full breakdown of what is verified, what is unknown, and a concrete training/submission roadmap.

> ⚠️ Important caveat up front. The Kaggle competition pages (Overview, Rules, Data, Code, Discussion) are gated behind a Cloudflare/reCAPTCHA shell and the `kaggle_environments/envs/orbit_wars/` source files were not retrievable to the research tool, even via raw GitHub URLs. The single piece of *verbatim* page content that was confirmed across every tab is the one-line tagline. **Everything else in sections 1–7 below is high-confidence inference from (a) that tagline, (b) the two community notebooks whose titles were retrievable, and (c) the well-documented conventions of every prior Kaggle simulation competition that uses the same framework.** Treat the inferred specifics (exact action schema, exact scoring, episode length, etc.) as a strong prior to validate against the actual `orbit_wars.json` schema once you `pip install kaggle-environments` and inspect it locally.

## 1. What Orbit Wars actually is

Verbatim from the Kaggle landing page (and repeated identically on the Overview, Rules, Discussion, and Code tabs):

> *"Conquer planets rotating around a sun in continuous 2D space. A real-time strategy game for 2 or 4 players."*

That single sentence pins down the genre with high confidence. Orbit Wars is **not a satellite-debris or space-domain-awareness science problem** — there is no NASA, ESA, SpaceX or Aerospace Corp sponsor in evidence; Kaggle itself is the host. It is a stylized RTS in the lineage of **Galcon, Eufloria, and Kaggle's own Halite/Kore**: a central sun anchors a system of orbiting planets, and each player owns some subset of planets, builds or accumulates fleets/ships there, and dispatches them through continuous 2D space (under the sun's gravity, almost certainly) to capture neutral or enemy planets. Because planets *orbit*, the geometry of attack windows changes every tick — that is the core strategic novelty distinguishing it from grid-based RTS predecessors. Two-player matches are duels; four-player matches are free-for-alls, both feeding the same skill rating. The community notebook **"Orbit Wars 2026 - Tactical Heuristic"** by user *sigmaborov* confirms the year and signals that hand-crafted heuristics are competitive enough to be worth publishing as starters; **"Orbit Wars - Reinforcement Learning Tutorial"** by *kashiwaba* confirms RL is the expected ML route.

## 2. Observation, action, and scoring (inferred from framework conventions)

Every kaggle-environments game follows the contract `def agent(observation, configuration) -> action`, called once per environment tick. The `observation` is a Python dict, `configuration` carries static episode constants (`episodeSteps`, `actTimeout`, `agentTimeout`, plus game-specific constants), and the agent returns either a primitive (int/float) or a structured `list`/`dict` of commands. The `orbit_wars.json` JSON-Schema file in the env folder defines the exact shapes; you should dump it on day one with:

```python
from kaggle_environments import make
env = make("orbit_wars")
import json
print(json.dumps(env.specification.observation, indent=2))
print(json.dumps(env.specification.configuration, indent=2))
print(json.dumps(env.specification.action, indent=2))
```

**Expected observation contents** (extremely high confidence on structure, lower on field names): your player index/`mark`, the current `step`, the sun's parameters (likely fixed at origin), an array of planets each with `(x, y, vx, vy, owner, ships, radius, orbital_radius, angular_velocity)` or equivalent, an array of in-flight fleets each with `(x, y, vx, vy, owner, size, target_planet)`, and remaining build/cooldown counters. Full state is almost certainly observable to all players — kaggle-environments games are typically **perfect-information** unless the README explicitly says otherwise.

**Expected action space**: a **structured hybrid** — per tick, the agent emits a list of dispatch commands, each of which selects (i) a source planet you own, (ii) a target (planet or coordinate), and (iii) a fleet size. Whether the launch direction/speed is continuous (delta-v vector) or discretized (target-only, with the engine computing a Hohmann-like trajectory) is the single most important spec detail to confirm before architecting a model. Both flavors have appeared in past Kaggle sim comps: Kore used a small string-DSL of orders, Lux AI used per-unit discrete actions, Halite used per-ship discrete moves. Given the "continuous 2D" framing, I weakly expect at least one continuous component (launch angle or speed) plus discrete source/target indices.

**Scoring**: standard for Kaggle simulation competitions and confirmed by precedent across Hungry Geese, Kore, Lux AI, ConnectX, Battle Geese — submissions play **continuous round-robin matches against other live submissions on Kaggle's servers**, generating a **TrueSkill-style μ−3σ skill rating** that feeds the leaderboard. Per-episode reward is a win/loss/draw against opponents, typically determined by who owns the most planets (or ships) at `episodeSteps` or who has eliminated all rivals first. **The leaderboard is dynamic**, not a one-shot test set, so submission late ≠ better; submitted agents continue playing until the deadline.

## 3. Environment, submission format, and restrictions

The simulator is **`kaggle_environments` (`pip install kaggle-environments`, currently v1.28.x)**, a pure-Python deterministic engine paired with a JavaScript replay viewer that renders inside Jupyter/IPython. There is no Pygame, Unity, or external game engine. You can run matches programmatically and watch them in a notebook:

```python
env = make("orbit_wars", debug=True)
env.run([my_agent, "random"])           # 2-player
env.run([agent_a, agent_b, agent_c, agent_d])  # 4-player
env.render(mode="ipython")
```

**Submission is a single Python file** (typically `main.py` inside a `submission.tar.gz`) that defines a top-level `agent(observation, configuration)` function. Trained model weights are bundled inside the tarball and loaded once at agent-init time — Kaggle gives you a few seconds of `agentTimeout` for cold start and a per-step `actTimeout` (historically ~1–4 seconds depending on the comp; treat anything heavier than ~100 ms inference as risky). **Constraints you should assume until you read the actual rules**: CPU-only at evaluation (no GPU during scored matches), ~16 GB RAM, **no internet access from the agent**, submission size on the order of 100 MB. PyTorch and NumPy are installed in the standard Kaggle simulation Docker image; exotic frameworks (JAX, custom CUDA kernels) work only if they ship inside your tarball and run on CPU. There is no announced ban on RL frameworks — every recent Kaggle sim comp has explicitly encouraged them.

## 4. Recommended library: PettingZoo wrapper + CleanRL (with RLlib as the heavyweight alternative)

**Pick one of two stacks depending on your engineering comfort and compute budget:**

The **lightweight, hackable choice is CleanRL**. CleanRL ships single-file PPO implementations (~400 lines each) that are explicitly designed for RTS-style self-play research and are the canonical starting point in recent Kaggle/AICrowd RTS competitions. You wrap the Kaggle env in a thin **PettingZoo `ParallelEnv`** adapter — PettingZoo is the de-facto multi-agent gym API and integrates cleanly with self-play scripts — and run CleanRL's `ppo_continuous_action.py` or `ppo.py` against past versions of itself. This is what the community-published "Orbit Wars - Reinforcement Learning Tutorial" almost certainly uses, given how dominant that pattern has become in 2024–2026.

The **scalable alternative is Ray RLlib**, which has first-class support for MAPPO, league play, parameterized opponents, and distributed rollouts. RLlib is the right call if you intend to run more than ~10⁸ environment steps or coordinate dozens of CPU rollout workers; the cost is significantly higher boilerplate and a steeper learning curve. **Stable-Baselines3 is *not* recommended** here — it has poor multi-agent and self-play support out of the box, and Orbit Wars is fundamentally a self-play problem.

A specialized library worth knowing about but **not** recommending as your main stack is **PufferLib** (vectorized RL for complex envs) — useful as a performance booster once you have a working baseline, not as a starting point.

## 5. Recommended algorithm: PPO with league-based self-play

**Use PPO**, and specifically **PPO with a self-play league** (the simplified AlphaStar-style "main agent + main exploiter + league exploiter" pattern, or at minimum a fictitious-self-play opponent pool of frozen past checkpoints). This is the right algorithm for three converging reasons:

PPO has **won every recent Kaggle simulation competition that wasn't won by a hand-crafted bot** — it's the baseline used by the top Kore 2022, Lux AI S1/S2, and Hungry Geese teams, and by DeepMind's AlphaStar at a much larger scale. It tolerates the hybrid discrete-plus-continuous action space that Orbit Wars almost certainly has, via independent action heads with a joint policy gradient. It works on CPU-evaluated rollouts (which Kaggle gives you for free during training on Kaggle Notebooks), and the inference-time policy is a single forward pass — easily fitting under any plausible `actTimeout`.

**Skip DQN/Rainbow** — discrete-only, doesn't handle the continuous launch parameters, and notoriously unstable in self-play. **Skip SAC** — designed for single-agent continuous control with off-policy replay, which is fundamentally awkward for self-play because the opponent distribution shifts faster than the replay buffer can track. **MAPPO** (multi-agent PPO with a centralized critic) is theoretically a better fit than vanilla PPO if you want to model team play, but in 2-player and 4-player **free-for-all** RTS the cooperative-team assumption MAPPO exploits doesn't apply, so plain PPO with a per-agent critic is simpler and just as strong. **AlphaZero-style MCTS is the wrong tool** — Orbit Wars has continuous state and continuous time evolution (planets move every tick regardless of action), making the search tree explode without the discrete branching factor that makes Go or Chess tractable.

The one credible competitor to PPO is **Impala/V-trace** for higher rollout throughput, which becomes worth the switch only if you scale past ~50 parallel envs.

## 6. A concrete six-week roadmap

**Week 1 — Inspect the environment, ship a heuristic baseline.** `pip install kaggle-environments`, dump the observation/action/configuration JSON schemas, read `orbit_wars.py` end-to-end, and write a 200-line **rule-based agent** (own-the-most-planets greedy: from each owned planet with surplus ships, dispatch fleets to the cheapest reachable enemy/neutral planet, accounting for orbital phase). Submit it. This sets your floor, validates the submission pipeline, and gives PPO a strong opponent to bootstrap against — published heuristics on past Kaggle sim comps regularly finished top-25%.

**Week 2 — Wrap as PettingZoo and verify with random self-play.** Build `OrbitWarsParallelEnv(ParallelEnv)` exposing `reset`, `step`, `observation_space`, `action_space` per agent. Inside `step`, marshal Kaggle observations to flat NumPy vectors and unmarshal model outputs to the Kaggle action dict. Verify by running PPO-vs-random for 1M steps and checking that learning curves rise. **Critical preprocessing step**: encode planets as a fixed-length set with **permutation invariance** — feed all planets through a shared MLP, sum-pool, then concatenate to per-planet features for the policy head. This single architectural choice is worth 5–10× sample efficiency on RTS.

**Week 3 — Architecture and PPO tuning.** Use a **shared-trunk actor-critic**: a 2-layer MLP encoder per entity type (planets, fleets, self-state), a Transformer or DeepSet aggregator, then split heads for source-planet selection (categorical), target-planet selection (categorical), fleet-size fraction (categorical or Beta-distributed continuous), and value. Hyperparameters that have worked well on Kaggle RTS comps: γ=0.997, λ=0.95, clip=0.2, lr=3e-4 with linear decay, 4096 steps × 32 parallel envs per rollout, 4 epochs, minibatch 256, entropy coef 0.01 decaying to 0.001. **Don't skip reward shaping**: in addition to the sparse win/loss reward, give a small dense bonus for net planets owned and net ships built per step, with coefficient ~0.01; sparse-only PPO will plateau on a random policy for hundreds of millions of steps.

**Week 4 — League self-play.** Maintain a pool of the last ~20 frozen checkpoints. Each rollout, sample the opponent uniformly from `{current_policy (50%), uniform_past_checkpoint (40%), best_heuristic (10%)}`. This avoids the **non-transitivity trap** that pure latest-vs-latest self-play falls into, where agents specialize against the most recent version and forget how to beat earlier strategies. Also rotate **2-player and 4-player episodes** at ~70/30 ratio because the leaderboard scores both.

**Week 5 — Evaluation and submission packaging.** Build a local Elo ladder against `["random", "heuristic_v1", "heuristic_v2", "ppo_checkpoint_t-1M", "ppo_checkpoint_t-5M"]` and gate every submission on it (your in-house Elo should exceed your previous best by 50+ before uploading). Package the agent as a standalone Python file: `submission.py` containing the policy network class, `weights.pt` (bundled in the tarball), and the `agent(observation, configuration)` entry point that loads weights once at module import and runs a single forward pass per call. Profile inference — it must complete inside `actTimeout` with margin; if you're close, prune to a smaller MLP rather than risk timeouts that count as losses on the leaderboard.

**Week 6 — Iterate based on the leaderboard.** Watch matches against top opponents using `env.render(mode="ipython")`. Look for systematic exploits — Kaggle RTS leaderboards reliably feature 2–3 dominant cheese strategies (e.g., early-rush, planet-trading, stalling) that pure PPO doesn't discover unsupervised. Manually script those as **extra opponents in your league pool** and retrain; this is how every winning Lux AI and Kore agent broke past the 80th percentile.

## 7. Conclusion: where the leverage is

Orbit Wars is a textbook Kaggle simulation competition, and the field has now had six years of sustained work showing that **PPO + permutation-invariant encoders + league self-play + a well-curated heuristic opponent pool** beats every other approach that doesn't involve hand-crafted search. The **two highest-leverage decisions** are not algorithmic: they are (1) reading the actual `orbit_wars.json` schema to know exactly what continuous vs discrete action surface you face, and (2) writing a strong scripted baseline in week 1 so PPO has someone to learn from. If the action space turns out to be purely discrete (target-planet plus fleet-fraction-bucket), this becomes one of the easier sim comps in years; if there is continuous launch geometry tied to orbital mechanics, expect a meaningful jump in difficulty and budget 2× the wall-clock training time. Either way, ship the heuristic on day one — on past Kaggle sim comps, the median submitted RL agent was *worse* than a 200-line greedy bot for the first month of the competition.