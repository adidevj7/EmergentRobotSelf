#!/usr/bin/env python3
# save as: collect_cw_rollout_obs.py

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
from pathlib import Path
from typing import List, Optional

# Optional device pinning before TensorFlow import.
# This keeps rollout collection from grabbing all GPUs unless requested.
def _early_device_bootstrap(argv: List[str]) -> None:
    mini = argparse.ArgumentParser(add_help=False)
    mini.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"])
    mini.add_argument("--gpu_id", type=int, default=0)
    ns, _ = mini.parse_known_args(argv)

    if ns.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        if "CUDA_VISIBLE_DEVICES" not in os.environ or not os.environ["CUDA_VISIBLE_DEVICES"].strip():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(ns.gpu_id)


_early_device_bootstrap(os.sys.argv[1:])

import numpy as np
import tensorflow as tf

import continualworld.gym_compat
from continualworld.envs import get_single_env
from continualworld.sac.models import MlpActor


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Collect real Continual World rollout observations from actor checkpoints."
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--accepted_root",
        type=str,
        default=None,
        help="Path to checkpoints/accepted directory. Script will find checkpoint dirs inside it."
    )
    src.add_argument(
        "--ckpt_list_txt",
        type=str,
        default=None,
        help="Text file containing checkpoint directory paths."
    )

    p.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Folder to save per-checkpoint observation files."
    )
    p.add_argument(
        "--max_checkpoints",
        type=int,
        default=24,
        help="Max number of checkpoint dirs to process after sorting/filtering."
    )
    p.add_argument(
        "--target_obs_total",
        type=int,
        default=100_000,
        help="Total number of observations to collect across all checkpoints."
    )

    p.add_argument(
        "--hidden_sizes",
        type=int,
        nargs="+",
        default=[256, 256, 256, 256],
        help="Actor hidden sizes. Must match training."
    )
    p.add_argument(
        "--activation",
        type=str,
        default="lrelu",
        choices=["relu", "tanh", "elu", "lrelu"],
        help="Actor activation. Must match training."
    )

    p.set_defaults(use_layer_norm=True)
    p.add_argument(
        "--use_layer_norm",
        dest="use_layer_norm",
        action="store_true",
        help="Use layer norm in actor."
    )
    p.add_argument(
        "--no_layer_norm",
        dest="use_layer_norm",
        action="store_false",
        help="Disable layer norm in actor."
    )

    p.set_defaults(hide_task_id=False)
    p.add_argument(
        "--hide_task_id",
        dest="hide_task_id",
        action="store_true",
        help="Manual override hint. Usually not needed because the script auto-detects checkpoint input width."
    )
    p.add_argument(
        "--show_task_id",
        dest="hide_task_id",
        action="store_false",
        help="Manual override hint."
    )

    p.add_argument(
        "--policy_mode",
        type=str,
        default="deterministic",
        choices=["deterministic", "stochastic"],
        help="Use actor mean action or sampled SAC action."
    )
    p.add_argument(
        "--seed_base",
        type=int,
        default=0,
        help="Base seed; each checkpoint/episode derives a unique seed from this."
    )

    p.add_argument(
        "--save_csv",
        action="store_true",
        help="Also save a CSV alongside the .npy file."
    )
    p.add_argument(
        "--float_dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
        help="Storage dtype for saved observations."
    )
    p.add_argument(
        "--randomization",
        type=str,
        default="deterministic",
        help="Env randomization mode passed to get_single_env."
    )

    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="TensorFlow device preference."
    )
    p.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="Visible GPU id if --device gpu is used."
    )

    p.add_argument(
        "--allow_nonempty_out_dir",
        action="store_true",
        help="Allow writing into a non-empty output directory. Default is to refuse."
    )

    return p.parse_args()


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_activation(name: str):
    name = name.lower()
    if name == "relu":
        return tf.nn.relu
    if name == "tanh":
        return tf.nn.tanh
    if name == "elu":
        return tf.nn.elu
    if name == "lrelu":
        return tf.nn.leaky_relu
    raise ValueError(f"Unsupported activation: {name}")


def reset_compat(env, seed=None):
    try:
        out = env.reset(seed=seed)
    except TypeError:
        out = env.reset()

    if isinstance(out, tuple):
        return out[0]
    return out


def step_compat(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, rew, terminated, truncated, info = out
        done = bool(terminated) or bool(truncated)
        return obs, rew, done, info
    obs, rew, done, info = out
    return obs, rew, bool(done), info


def is_checkpoint_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "actor.index").exists()
        and (path / "actor.data-00000-of-00001").exists()
    )


def discover_checkpoint_dirs_from_root(accepted_root: Path) -> List[Path]:
    if not accepted_root.exists():
        raise FileNotFoundError(f"accepted_root does not exist: {accepted_root}")
    ckpts = [p for p in accepted_root.iterdir() if is_checkpoint_dir(p)]
    ckpts = sorted(ckpts, key=lambda p: p.name)
    return ckpts


def discover_checkpoint_dirs_from_txt(txt_path: Path) -> List[Path]:
    if not txt_path.exists():
        raise FileNotFoundError(f"ckpt_list_txt does not exist: {txt_path}")

    ckpts = []
    seen = set()

    with txt_path.open("r") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue

            p = Path(s).expanduser()
            if p.name in {"actor.index", "actor.data-00000-of-00001"}:
                p = p.parent

            try:
                p = p.resolve()
            except Exception:
                continue

            if str(p) in seen:
                continue

            if is_checkpoint_dir(p):
                ckpts.append(p)
                seen.add(str(p))

    ckpts = sorted(ckpts, key=lambda p: p.name)
    return ckpts


def parse_task_name_from_ckpt_dirname(dirname: str) -> str:
    """
    Expected format:
      c01_b02_r00_faucet-close-v1_2026-03-14_00-36-12
    Extract:
      faucet-close-v1
    """
    m = re.match(
        r"^c\d+_b\d+_r\d+_(.+?)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$",
        dirname
    )
    if m:
        return m.group(1)
    raise ValueError(f"Could not parse task name from checkpoint dir: {dirname}")


def find_run_root_from_ckpt_dir(ckpt_dir: Path) -> Optional[Path]:
    """
    Expected structure:
      run_root/checkpoints/accepted/ckpt_dir
    """
    cur = ckpt_dir.resolve()
    for parent in [cur] + list(cur.parents):
        if (parent / "config.json").exists() and (parent / "checkpoints").exists():
            return parent
    return None


def try_read_run_config(ckpt_dir: Path) -> dict:
    run_root = find_run_root_from_ckpt_dir(ckpt_dir)
    if run_root is None:
        return {}
    config_path = run_root / "config.json"
    if not config_path.exists():
        return {}
    try:
        with config_path.open("r") as f:
            return json.load(f)
    except Exception:
        return {}


def infer_ckpt_input_dim(actor_prefix: str, first_hidden: int) -> int:
    vars_meta = tf.train.list_variables(actor_prefix)
    candidates = []
    for name, shape in vars_meta:
        if len(shape) == 2 and int(shape[1]) == int(first_hidden):
            candidates.append((name, shape))

    if not candidates:
        raise RuntimeError(f"Could not infer input dim from checkpoint: {actor_prefix}")

    _, shape = sorted(candidates, key=lambda x: int(x[1][0]))[0]
    return int(shape[0])


def get_action(actor: MlpActor, obs: np.ndarray, policy_mode: str) -> np.ndarray:
    obs_batch = tf.convert_to_tensor(obs[None, :], dtype=tf.float32)
    mu, _, pi, _ = actor(obs_batch)
    if policy_mode == "deterministic":
        return mu.numpy()[0]
    return pi.numpy()[0]


def safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", s)


def list_dir_entries(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return sorted([p for p in path.iterdir()], key=lambda x: x.name)


def ensure_out_dir_ready(out_dir: Path, allow_nonempty: bool) -> None:
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        return

    entries = list_dir_entries(out_dir)
    if entries and not allow_nonempty:
        preview = "\n".join(str(p) for p in entries[:20])
        raise RuntimeError(
            f"Output directory is not empty: {out_dir}\n"
            f"Refusing to write into it.\n"
            f"First entries:\n{preview}\n"
            f"\nClean it first, or rerun with --allow_nonempty_out_dir if you really want to reuse it."
        )


# ------------------------------------------------------------
# Collection
# ------------------------------------------------------------

def collect_obs_for_checkpoint(
    ckpt_dir: Path,
    out_dir: Path,
    args,
    ckpt_idx: int,
    target_n: int,
) -> dict:
    task_name = parse_task_name_from_ckpt_dirname(ckpt_dir.name)
    config = try_read_run_config(ckpt_dir)
    max_episode_len = int(config.get("max_episode_len", 200))

    env_seed_offset = args.seed_base + ckpt_idx * 1_000_000
    set_global_seed(env_seed_offset)

    env = get_single_env(task_name, randomization=args.randomization)

    env_obs_dim = int(env.observation_space.shape[0])
    ckpt_input_dim = infer_ckpt_input_dim(
        str(ckpt_dir / "actor"),
        first_hidden=args.hidden_sizes[0],
    )

    if ckpt_input_dim == env_obs_dim:
        hide_task_id_effective = False
    elif ckpt_input_dim == env_obs_dim - 1:
        hide_task_id_effective = True
    else:
        raise RuntimeError(
            f"Checkpoint/env obs mismatch for {ckpt_dir.name}: "
            f"checkpoint expects {ckpt_input_dim}, env gives {env_obs_dim}."
        )

    actor = MlpActor(
        input_dim=env_obs_dim,
        action_space=env.action_space,
        hidden_sizes=args.hidden_sizes,
        activation=get_activation(args.activation),
        use_layer_norm=args.use_layer_norm,
        num_heads=1,
        hide_task_id=hide_task_id_effective,
    )

    dummy = tf.convert_to_tensor(np.zeros((1, env_obs_dim), dtype=np.float32))
    actor(dummy)
    actor.load_weights(str(ckpt_dir / "actor"))

    save_dtype = np.float32 if args.float_dtype == "float32" else np.float64
    saved_obs_dim = ckpt_input_dim
    obs_store = np.empty((target_n, saved_obs_dim), dtype=save_dtype)

    n_saved = 0
    episode_count = 0
    episode_lengths = []
    episode_returns = []
    episode_seeds = []

    min_episodes_lower_bound = int(math.ceil(target_n / max_episode_len))

    print("=" * 100)
    print(f"[{ckpt_idx+1}] checkpoint: {ckpt_dir}")
    print(f"[{ckpt_idx+1}] task: {task_name}")
    print(f"[{ckpt_idx+1}] env_obs_dim: {env_obs_dim}")
    print(f"[{ckpt_idx+1}] ckpt_input_dim: {ckpt_input_dim}")
    print(f"[{ckpt_idx+1}] hide_task_id_effective: {hide_task_id_effective}")
    print(f"[{ckpt_idx+1}] target_obs_for_this_checkpoint: {target_n}")
    print(f"[{ckpt_idx+1}] max_episode_len(from config or fallback): {max_episode_len}")
    print(f"[{ckpt_idx+1}] lower-bound episodes needed: {min_episodes_lower_bound}")
    print("=" * 100)

    while n_saved < target_n:
        ep_seed = env_seed_offset + episode_count
        episode_seeds.append(ep_seed)

        obs = reset_compat(env, seed=ep_seed)
        obs = np.asarray(obs, dtype=np.float32)

        done = False
        ep_ret = 0.0
        ep_len = 0

        while not done and n_saved < target_n:
            obs_for_save = obs[:-1] if hide_task_id_effective else obs
            obs_store[n_saved] = obs_for_save.astype(save_dtype, copy=False)
            n_saved += 1

            action = get_action(actor, obs, args.policy_mode)
            obs, rew, done, info = step_compat(env, action)
            obs = np.asarray(obs, dtype=np.float32)

            ep_ret += float(rew)
            ep_len += 1

        episode_lengths.append(ep_len)
        episode_returns.append(ep_ret)
        episode_count += 1

        if episode_count % 25 == 0 or n_saved >= target_n:
            print(
                f"[{ckpt_idx+1}] episodes={episode_count} "
                f"saved={n_saved}/{target_n} "
                f"last_ep_len={ep_len} "
                f"last_ep_ret={ep_ret:.3f}"
            )

    env.close()

    ckpt_safe = safe_name(ckpt_dir.name)
    task_safe = safe_name(task_name)
    stem = f"{ckpt_idx+1:02d}_{task_safe}_{ckpt_safe}"

    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / f"{stem}.npy"
    meta_path = out_dir / f"{stem}.json"

    np.save(npy_path, obs_store)

    if args.save_csv:
        csv_path = out_dir / f"{stem}.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            header = [f"obs_{i}" for i in range(obs_store.shape[1])]
            writer.writerow(header)
            writer.writerows(obs_store)

    meta = {
        "checkpoint_dir": str(ckpt_dir),
        "actor_ckpt_prefix": str(ckpt_dir / "actor"),
        "task_name": task_name,
        "n_observations": int(n_saved),
        "env_obs_dim": int(env_obs_dim),
        "saved_obs_dim": int(saved_obs_dim),
        "ckpt_input_dim": int(ckpt_input_dim),
        "hide_task_id_effective": bool(hide_task_id_effective),
        "episodes_recorded": int(episode_count),
        "episode_lengths_mean": float(np.mean(episode_lengths)) if episode_lengths else None,
        "episode_lengths_min": int(np.min(episode_lengths)) if episode_lengths else None,
        "episode_lengths_max": int(np.max(episode_lengths)) if episode_lengths else None,
        "episode_returns_mean": float(np.mean(episode_returns)) if episode_returns else None,
        "policy_mode": args.policy_mode,
        "seed_base_for_checkpoint": int(env_seed_offset),
        "episode_seeds": episode_seeds,
        "hidden_sizes": list(args.hidden_sizes),
        "activation": args.activation,
        "use_layer_norm": bool(args.use_layer_norm),
        "randomization": args.randomization,
        "target_obs_for_this_checkpoint": int(target_n),
        "max_episode_len_from_config_or_fallback": int(max_episode_len),
        "min_episode_lower_bound": int(min_episodes_lower_bound),
        "dtype": args.float_dtype,
        "saved_npy": str(npy_path),
        "run_config_found": bool(config),
    }

    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"[{ckpt_idx+1}] saved: {npy_path}")
    print(f"[{ckpt_idx+1}] meta : {meta_path}")

    return meta


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    args = parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    ensure_out_dir_ready(out_dir, allow_nonempty=args.allow_nonempty_out_dir)

    if args.accepted_root is not None:
        ckpts = discover_checkpoint_dirs_from_root(
            Path(args.accepted_root).expanduser().resolve()
        )
    else:
        ckpts = discover_checkpoint_dirs_from_txt(
            Path(args.ckpt_list_txt).expanduser().resolve()
        )

    if not ckpts:
        raise RuntimeError("No valid checkpoint directories found.")

    ckpts = ckpts[: args.max_checkpoints]
    n_ckpts = len(ckpts)

    total_obs = int(args.target_obs_total)
    if total_obs <= 0:
        raise ValueError("--target_obs_total must be > 0")

    base = total_obs // n_ckpts
    rem = total_obs % n_ckpts
    per_ckpt_targets = [base + (1 if i < rem else 0) for i in range(n_ckpts)]

    print(f"Found {n_ckpts} checkpoint(s) to process.")
    print(f"Total observation budget: {total_obs}")
    print(f"Base per checkpoint: {base}")
    print(f"Remainder checkpoints getting +1: {rem}")
    print(f"Sum check: {sum(per_ckpt_targets)}")

    for i, (p, t) in enumerate(zip(ckpts, per_ckpt_targets), start=1):
        print(f"  {i:02d}. {p}  -> target_obs={t}")

    summary_rows = []
    for ckpt_idx, (ckpt_dir, target_n) in enumerate(zip(ckpts, per_ckpt_targets)):
        meta = collect_obs_for_checkpoint(
            ckpt_dir=ckpt_dir,
            out_dir=out_dir,
            args=args,
            ckpt_idx=ckpt_idx,
            target_n=target_n,
        )
        summary_rows.append(meta)

    summary_json = out_dir / "collection_summary.json"
    summary_csv = out_dir / "collection_summary.csv"

    with summary_json.open("w") as f:
        json.dump(summary_rows, f, indent=2)

    fieldnames = [
        "checkpoint_dir",
        "task_name",
        "n_observations",
        "env_obs_dim",
        "saved_obs_dim",
        "ckpt_input_dim",
        "hide_task_id_effective",
        "episodes_recorded",
        "episode_lengths_mean",
        "episode_lengths_min",
        "episode_lengths_max",
        "episode_returns_mean",
        "policy_mode",
        "seed_base_for_checkpoint",
        "target_obs_for_this_checkpoint",
        "max_episode_len_from_config_or_fallback",
        "min_episode_lower_bound",
        "saved_npy",
    ]

    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    print("\nDone.")
    print(f"Summary JSON: {summary_json}")
    print(f"Summary CSV : {summary_csv}")


if __name__ == "__main__":
    main()