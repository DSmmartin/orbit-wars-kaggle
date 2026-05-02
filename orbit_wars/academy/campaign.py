"""Campaign — runs the PPO training loop, collecting rollouts and forging checkpoints."""

import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import draccus
import numpy as np
import torch

from .arena import OrbitWarsEnv
from .cartography import TurnBatch, candidate_feature_dim, global_feature_dim, self_feature_dim
from .chronicle import PlanetState
from .crucible import TransitionBatch, ppo_update, sample_actions
from .doctrine import TrainConfig
from .rivals import SelfPlayOpponent, build_opponent
from .tactician import PlanetPolicy

_VALID_OPPONENTS = {"sniper", "random", "self"}


@dataclass(slots=True)
class _PeriodAccumulator:
    """Collects stats between two summary prints."""
    updates: int = 0
    episodes: int = 0
    episode_rewards: list = field(default_factory=list)
    losses: list = field(default_factory=list)
    samples: int = 0


def _fmt_time(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m {sec:02d}s"


def _print_summary(
    cfg: TrainConfig,
    update: int,
    acc: _PeriodAccumulator,
    period_secs: float,
    elapsed_total: float,
) -> None:
    remaining_updates = cfg.ppo.total_updates - update
    secs_per_update = elapsed_total / max(update, 1)
    eta_secs = secs_per_update * remaining_updates

    updates_per_min = acc.updates / max(period_secs / 60, 1e-6)
    samples_per_min = acc.samples / max(period_secs / 60, 1e-6)

    w = 54
    sep = "─" * w
    print(f"\n{'═' * w}")
    print(
        f" Summary  [update {update}/{cfg.ppo.total_updates}"
        f" | elapsed {_fmt_time(elapsed_total)}"
        f" | eta {_fmt_time(eta_secs)}]"
    )
    print(f"{'═' * w}")
    print(f"  {'period':<18}: {_fmt_time(period_secs)}")
    print(f"  {'updates':<18}: {acc.updates}")
    print(
        f"  {'throughput':<18}: {updates_per_min:.1f} updates/min"
        f"  |  {samples_per_min:.0f} samples/min"
    )
    print(f"  {'episodes':<18}: {acc.episodes}")
    if acc.episode_rewards:
        r_mean = sum(acc.episode_rewards) / len(acc.episode_rewards)
        r_min = min(acc.episode_rewards)
        r_max = max(acc.episode_rewards)
        print(f"  {'reward':<18}: mean {r_mean:.4f}  min {r_min:.4f}  max {r_max:.4f}")
    else:
        print(f"  {'reward':<18}: —  (no episodes finished)")
    if acc.losses:
        print(f"  {'loss':<18}: mean {sum(acc.losses) / len(acc.losses):.4f}")
    print(f"{'═' * w}\n")


@dataclass(slots=True)
class StepGroup:
    indices: list[int]
    reward: float
    done: bool


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _device_label(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda  ({torch.cuda.get_device_name(0)})"
    if device.type == "mps":
        return "mps  (Apple Silicon)"
    return "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _print_config_status(cfg: TrainConfig, device: torch.device, start_update: int) -> None:
    w = 54
    sep = "=" * w
    print(sep)
    print(f"{'ORBIT WARS ACADEMY — Training Campaign':^{w}}")
    print(sep)
    print(f"  {'run_name':<18}: {cfg.run_name}")
    print(f"  {'device':<18}: {_device_label(device)}")
    print(f"  {'opponent':<18}: {cfg.opponent}")
    print(f"  {'total_updates':<18}: {cfg.ppo.total_updates}")
    print(f"  {'start_update':<18}: {start_update}")
    print(f"  {'num_envs':<18}: {cfg.ppo.num_envs}")
    print(f"  {'rollout_steps':<18}: {cfg.ppo.rollout_steps}")
    print(f"  {'lr':<18}: {cfg.ppo.lr}")
    print(f"  {'gamma':<18}: {cfg.ppo.gamma}")
    print(f"  {'hidden_size':<18}: {cfg.model.hidden_size}")
    print(f"  {'save_dir':<18}: {cfg.save_dir}")
    print(f"  {'checkpoint_every':<18}: {cfg.checkpoint_every}")
    print(f"  {'resume':<18}: {cfg.resume}")
    print(sep)
    print()


def collect_rollout(
    envs: list[OrbitWarsEnv],
    batches: list[TurnBatch],
    policy: PlanetPolicy,
    cfg: TrainConfig,
    device: torch.device,
    next_seed: int,
) -> tuple[TransitionBatch, list[TurnBatch], int, dict[str, float]]:
    empty_candidate = (cfg.env.candidate_count, candidate_feature_dim())
    self_rows: list[np.ndarray] = []
    candidate_rows: list[np.ndarray] = []
    global_rows: list[np.ndarray] = []
    candidate_masks: list[np.ndarray] = []
    target_indices: list[int] = []
    log_probs: list[float] = []
    values: list[float] = []
    groups_per_env: list[list[StepGroup]] = [[] for _ in envs]
    episode_rewards: list[float] = []
    running_episode_rewards = [0.0 for _ in envs]

    for _ in range(cfg.ppo.rollout_steps):
        offsets = np.cumsum([0] + [batch.self_features.shape[0] for batch in batches[:-1]])
        merged = merge_batches(batches)
        row_values = np.zeros((merged.self_features.shape[0],), dtype=np.float32)
        if merged.self_features.shape[0] > 0:
            with torch.inference_mode():
                outputs = policy(
                    torch.from_numpy(merged.self_features).to(device),
                    torch.from_numpy(merged.candidate_features).to(device),
                    torch.from_numpy(merged.global_features).to(device),
                    torch.from_numpy(merged.candidate_mask).to(device).bool(),
                )
                sampled = sample_actions(outputs, deterministic=False)
                row_values = outputs.value.detach().cpu().numpy()
                sampled_target_index = sampled.target_index.detach().cpu().numpy()
                sampled_log_prob = sampled.log_prob.detach().cpu().numpy()
        else:
            sampled_target_index = np.zeros((0,), dtype=np.int64)
            sampled_log_prob = np.zeros((0,), dtype=np.float32)

        next_batches: list[TurnBatch] = []
        for env_idx, env in enumerate(envs):
            batch = batches[env_idx]
            start = int(offsets[env_idx])
            moves = []
            group_indices: list[int] = []
            for local_idx, context in enumerate(batch.contexts):
                global_idx = start + local_idx
                self_rows.append(batch.self_features[local_idx])
                candidate_rows.append(batch.candidate_features[local_idx])
                global_rows.append(batch.global_features[local_idx])
                candidate_masks.append(batch.candidate_mask[local_idx])
                values.append(float(row_values[global_idx]))
                tgt_idx = int(sampled_target_index[global_idx]) if batch.self_features.shape[0] > 0 else 0
                is_valid_send = (
                    tgt_idx > 0
                    and tgt_idx < len(context.candidate_ids)
                    and context.candidate_mask[tgt_idx]
                    and int(context.ship_counts[tgt_idx]) > 0
                )
                target_indices.append(tgt_idx)
                log_probs.append(
                    float(sampled_log_prob[global_idx]) if batch.self_features.shape[0] > 0 else 0.0
                )
                group_indices.append(len(values) - 1)
                if not is_valid_send:
                    continue
                ships = int(context.ship_counts[tgt_idx])
                src_planet = _find_planet(batch.state.planets, context.source_id)
                if src_planet is None or src_planet.ships < ships:
                    continue
                moves.append([context.source_id, float(context.target_angles[tgt_idx]), ships])
            result = env.step(moves)
            running_episode_rewards[env_idx] += float(result.reward)
            groups_per_env[env_idx].append(
                StepGroup(indices=group_indices, reward=float(result.reward), done=result.done)
            )
            if result.done:
                episode_rewards.append(running_episode_rewards[env_idx])
                running_episode_rewards[env_idx] = 0.0
                next_seed += 1
                next_batch = env.reset(seed=next_seed)
            else:
                next_batch = result.batch
            next_batches.append(next_batch)
        batches = next_batches

    returns: list[float] = [0.0] * len(values)
    advantages: list[float] = [0.0] * len(values)
    next_state_values = _bootstrap_values(policy, batches, device)
    for env_idx, groups in enumerate(groups_per_env):
        future_return = next_state_values[env_idx]
        for group in reversed(groups):
            future_return = group.reward + cfg.ppo.gamma * future_return * (1.0 - float(group.done))
            for idx in group.indices:
                returns[idx] = future_return
                advantages[idx] = future_return - values[idx]
    batch = TransitionBatch(
        self_features=torch.from_numpy(np.asarray(self_rows, dtype=np.float32).reshape(-1, self_feature_dim())),
        candidate_features=torch.from_numpy(
            np.asarray(candidate_rows, dtype=np.float32).reshape(-1, empty_candidate[0], empty_candidate[1])
        ),
        global_features=torch.from_numpy(np.asarray(global_rows, dtype=np.float32).reshape(-1, global_feature_dim())),
        candidate_mask=torch.from_numpy(
            np.asarray(candidate_masks, dtype=bool).reshape(-1, cfg.env.candidate_count)
        ),
        target_index=torch.tensor(target_indices, dtype=torch.long),
        log_prob=torch.tensor(log_probs, dtype=torch.float32),
        returns=torch.tensor(returns, dtype=torch.float32),
        advantages=torch.tensor(advantages, dtype=torch.float32),
    )
    stats = {
        "episode_reward_mean": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "episode_rewards": episode_rewards,
        "episodes_finished": float(len(episode_rewards)),
        "samples": float(len(values)),
    }
    return batch, batches, next_seed, stats


def _bootstrap_values(policy: PlanetPolicy, batches: list[TurnBatch], device: torch.device) -> list[float]:
    merged = merge_batches(batches)
    if merged.self_features.shape[0] == 0:
        return [0.0 for _ in batches]
    offsets = np.cumsum([0] + [batch.self_features.shape[0] for batch in batches[:-1]])
    with torch.inference_mode():
        outputs = policy(
            torch.from_numpy(merged.self_features).to(device),
            torch.from_numpy(merged.candidate_features).to(device),
            torch.from_numpy(merged.global_features).to(device),
            torch.from_numpy(merged.candidate_mask).to(device).bool(),
        )
    values = outputs.value.detach().cpu().numpy()
    per_env = []
    for env_idx, batch in enumerate(batches):
        start = int(offsets[env_idx])
        count = batch.self_features.shape[0]
        per_env.append(0.0 if count == 0 else float(values[start : start + count].mean()))
    return per_env


def merge_batches(batches: list[TurnBatch]) -> TurnBatch:
    if not batches:
        raise ValueError("batches must not be empty")
    has_rows = any(batch.self_features.shape[0] > 0 for batch in batches)
    self_rows = (
        np.concatenate([batch.self_features for batch in batches], axis=0)
        if has_rows
        else np.zeros((0, self_feature_dim()), dtype=np.float32)
    )
    candidate_rows = (
        np.concatenate([batch.candidate_features for batch in batches], axis=0)
        if has_rows
        else np.zeros((0, batches[0].candidate_features.shape[1], candidate_feature_dim()), dtype=np.float32)
    )
    global_rows = (
        np.concatenate([batch.global_features for batch in batches], axis=0)
        if has_rows
        else np.zeros((0, global_feature_dim()), dtype=np.float32)
    )
    candidate_masks = (
        np.concatenate([batch.candidate_mask for batch in batches], axis=0)
        if has_rows
        else np.zeros((0, batches[0].candidate_mask.shape[1]), dtype=bool)
    )
    return TurnBatch(
        self_features=self_rows,
        candidate_features=candidate_rows,
        global_features=global_rows,
        candidate_mask=candidate_masks,
        contexts=[context for batch in batches for context in batch.contexts],
        state=batches[0].state,
    )


def save_checkpoint(
    save_dir: Path,
    run_name: str,
    update: int,
    policy: PlanetPolicy,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
) -> None:
    run_dir = save_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "update": update,
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
    }
    torch.save(payload, run_dir / "ckpt_last.pt")
    torch.save(payload, run_dir / f"ckpt_{update:06d}.pt")


def _find_planet(planets: list[PlanetState], planet_id: int) -> PlanetState | None:
    for planet in planets:
        if planet.id == planet_id:
            return planet
    return None


def _print_resume_summary(cfg: TrainConfig, resumed_from: int) -> None:
    ckpt_path = Path(cfg.save_dir) / cfg.run_name / "ckpt_last.pt"
    remaining = cfg.ppo.total_updates - resumed_from
    progress_pct = resumed_from / cfg.ppo.total_updates * 100
    w = 54
    print(f"{'═' * w}")
    print(f"{'ORBIT WARS ACADEMY — Resuming Campaign':^{w}}")
    print(f"{'═' * w}")
    print(f"  {'checkpoint':<18}: {ckpt_path}")
    print(f"  {'resumed from':<18}: update {resumed_from}/{cfg.ppo.total_updates}  ({progress_pct:.1f}% done)")
    print(f"  {'remaining':<18}: {remaining} updates")
    print(f"{'═' * w}\n")


def _load_resume_checkpoint(
    cfg: TrainConfig,
    policy: PlanetPolicy,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    """Load ckpt_last.pt and return the next update index to start from."""
    ckpt_path = Path(cfg.save_dir) / cfg.run_name / "ckpt_last.pt"
    if not ckpt_path.exists():
        print(f"[resume] No checkpoint found at {ckpt_path} — starting from scratch.")
        return 1
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["policy"])
    optimizer.load_state_dict(ckpt["optimizer"])
    resumed_from = int(ckpt["update"])
    _print_resume_summary(cfg, resumed_from)
    return resumed_from + 1


@draccus.wrap()
def main(cfg: TrainConfig) -> None:
    if cfg.opponent not in _VALID_OPPONENTS:
        raise ValueError(f"Unknown opponent '{cfg.opponent}'. Valid choices: {_VALID_OPPONENTS}")

    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    opponent = build_opponent(cfg.opponent, cfg=cfg, device=device)
    envs = [OrbitWarsEnv(cfg, opponent, env_index=idx) for idx in range(cfg.ppo.num_envs)]

    policy = PlanetPolicy(
        self_dim=self_feature_dim(),
        candidate_dim=candidate_feature_dim(),
        global_dim=global_feature_dim(),
        candidate_count=cfg.env.candidate_count,
        hidden_size=cfg.model.hidden_size,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.ppo.lr)

    start_update = 1
    if cfg.resume:
        start_update = _load_resume_checkpoint(cfg, policy, optimizer, device)

    if isinstance(opponent, SelfPlayOpponent):
        opponent.sync_from(policy)

    _print_config_status(cfg, device, start_update)

    next_seed = cfg.seed + (start_update - 1) * cfg.ppo.num_envs
    batches = []
    for env in envs:
        batches.append(env.reset(seed=next_seed))
        next_seed += 1

    save_dir = Path(cfg.save_dir)
    train_start = time.monotonic()
    last_summary = train_start
    period = _PeriodAccumulator()

    for update in range(start_update, cfg.ppo.total_updates + 1):
        batch, batches, next_seed, stats = collect_rollout(envs, batches, policy, cfg, device, next_seed)
        metrics = ppo_update(
            policy,
            optimizer,
            batch,
            clip_coef=cfg.ppo.clip_coef,
            ent_coef=cfg.ppo.ent_coef,
            vf_coef=cfg.ppo.vf_coef,
            max_grad_norm=cfg.ppo.max_grad_norm,
            epochs=cfg.ppo.epochs,
            minibatch_size=cfg.ppo.minibatch_size,
            device=device,
        )
        if isinstance(opponent, SelfPlayOpponent) and update % cfg.self_play_update_interval == 0:
            opponent.sync_from(policy)

        period.updates += 1
        period.episodes += int(stats["episodes_finished"])
        period.episode_rewards.extend(stats["episode_rewards"])
        period.losses.append(metrics["loss"])
        period.samples += int(stats["samples"])

        if update % cfg.log_every == 0:
            print(
                f"update={update}/{cfg.ppo.total_updates}"
                f"  reward={stats['episode_reward_mean']:.4f}"
                f"  episodes={int(stats['episodes_finished'])}"
                f"  samples={int(stats['samples'])}"
                f"  loss={metrics['loss']:.4f}"
            )

        now = time.monotonic()
        if cfg.summary_freq > 0 and (now - last_summary) >= cfg.summary_freq * 60:
            _print_summary(cfg, update, period, now - last_summary, now - train_start)
            period = _PeriodAccumulator()
            last_summary = now

        if update % cfg.checkpoint_every == 0 or update == cfg.ppo.total_updates:
            save_checkpoint(save_dir, cfg.run_name, update, policy, optimizer, cfg)


if __name__ == "__main__":
    main()
