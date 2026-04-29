#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
import gym
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Union


def _early_gpu_bootstrap(argv: List[str]) -> None:
    """
    Parse the minimum GPU-selection args BEFORE importing TensorFlow or modules
    that create TensorFlow constants at import time.
    """
    mini = argparse.ArgumentParser(add_help=False)
    mini.add_argument("--learner_device", type=str, default="gpu")
    mini.add_argument("--gpu_id", type=int, default=0)
    ns, _ = mini.parse_known_args(argv)

    learner_device = str(ns.learner_device).strip().lower()

    if learner_device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return

    # Respect an externally pinned CUDA_VISIBLE_DEVICES if already set.
    if "CUDA_VISIBLE_DEVICES" not in os.environ or not os.environ["CUDA_VISIBLE_DEVICES"].strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(ns.gpu_id)


_early_gpu_bootstrap(sys.argv[1:])

import continualworld.gym_compat
import numpy as np
import tensorflow as tf

from continualworld.envs import get_single_env
from continualworld.sac import models
from continualworld.sac.sac import SAC
from continualworld.sac.utils.logx import EpochLogger
from continualworld.utils.utils import get_activation_from_str


def _parse_tasks(tasks_csv: str) -> List[str]:
    tasks = [t.strip() for t in tasks_csv.split(",") if t.strip()]
    if not tasks:
        raise ValueError("You must provide at least one task in --tasks.")
    return tasks


def _parse_hidden_sizes(hidden_sizes_csv: str) -> List[int]:
    vals = [x.strip() for x in hidden_sizes_csv.split(",") if x.strip()]
    if not vals:
        raise ValueError("hidden_sizes cannot be empty.")
    return [int(x) for x in vals]


def _parse_logger_output(logger_output_csv: str) -> List[str]:
    vals = [x.strip() for x in logger_output_csv.split(",") if x.strip()]
    if not vals:
        raise ValueError("logger_output cannot be empty.")
    return vals


def _parse_alpha(alpha_str: str) -> Union[str, float]:
    alpha_str = alpha_str.strip().lower()
    if alpha_str == "auto":
        return "auto"
    return float(alpha_str)


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def _safe_task_name(task_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", task_name)


def _safe_pct_tag(percent: float) -> str:
    pct_str = f"{percent:.6f}".rstrip("0").rstrip(".")
    return pct_str.replace(".", "p")


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _maybe_symlink(src: Optional[Union[str, Path]], dst: Path) -> None:
    if src is None:
        return
    src = Path(src).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
    except Exception:
        pass


def _resolve_resume(resume_from: Optional[str]) -> Dict:
    if resume_from is None:
        return {
            "resume_mode": None,
            "checkpoint_dir": None,
            "actor_ckpt": None,
            "start_cycle_idx": 0,
            "start_behavior_idx": 0,
            "run_root": None,
            "resume_meta": None,
        }

    resume_path = Path(resume_from).expanduser().resolve()

    if resume_path.is_dir():
        meta_path = resume_path / "meta.json"
        required = [
            resume_path / "actor.index",
            resume_path / "critic1.index",
            resume_path / "critic2.index",
            resume_path / "target_critic1.index",
            resume_path / "target_critic2.index",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Resume checkpoint directory is missing required files:\n" + "\n".join(missing)
            )

        resume_meta = None
        start_cycle_idx = 0
        start_behavior_idx = 0
        run_root = None

        if meta_path.exists():
            with meta_path.open("r") as f:
                resume_meta = json.load(f)

            start_cycle_idx = int(resume_meta.get("next_cycle_idx", 0))
            start_behavior_idx = int(resume_meta.get("next_behavior_idx", 0))

            if resume_path.parent.name in {"accepted", "failed", "progress"}:
                if resume_path.parent.parent.name == "checkpoints":
                    run_root = resume_path.parent.parent.parent
            elif resume_path.parent.name == "checkpoints":
                run_root = resume_path.parent.parent

        return {
            "resume_mode": "full",
            "checkpoint_dir": resume_path,
            "actor_ckpt": resume_path / "actor",
            "start_cycle_idx": start_cycle_idx,
            "start_behavior_idx": start_behavior_idx,
            "run_root": run_root,
            "resume_meta": resume_meta,
        }

    actor_ckpt = resume_path
    if actor_ckpt.with_suffix(".index").exists():
        return {
            "resume_mode": "actor_only",
            "checkpoint_dir": None,
            "actor_ckpt": actor_ckpt,
            "start_cycle_idx": 0,
            "start_behavior_idx": 0,
            "run_root": None,
            "resume_meta": None,
        }

    raise FileNotFoundError(
        f"Could not resolve --resume_from={resume_from} as either a checkpoint directory or actor checkpoint prefix."
    )


def _make_env_thunk(task_name: str):
    def _thunk():
        env = get_single_env(task_name)
        env = VectorEnvAPICompatWrapper(env)
        return env

    return _thunk


def _build_parallel_env(task_name: str, num_envs: int):
    if num_envs <= 1:
        return None
    env_fns = [_make_env_thunk(task_name) for _ in range(num_envs)]
    return gym.vector.AsyncVectorEnv(env_fns, shared_memory=False)


def _configure_tf_device(learner_device: str, gpu_id: int) -> str:
    learner_device = learner_device.strip().lower()
    if learner_device not in {"auto", "cpu", "gpu"}:
        raise ValueError("--learner_device must be one of: auto, cpu, gpu")

    if learner_device == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        return "/CPU:0"

    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        # If we masked visibility early with CUDA_VISIBLE_DEVICES, TensorFlow may
        # see exactly one GPU regardless of the original physical gpu_id.
        if len(gpus) == 1:
            chosen_idx = 0
        else:
            if gpu_id < 0 or gpu_id >= len(gpus):
                raise ValueError(f"--gpu_id={gpu_id} is invalid; found {len(gpus)} visible GPU(s).")
            chosen_idx = gpu_id

        try:
            tf.config.set_visible_devices(gpus[chosen_idx], "GPU")
        except RuntimeError:
            pass

        for gpu in tf.config.get_visible_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass

        return "/GPU:0"

    if learner_device == "gpu":
        raise RuntimeError("Requested --learner_device gpu, but TensorFlow does not see any GPU.")

    return "/CPU:0"


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _load_phase_state(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_phase_state(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp.replace(path)


def _replace_or_append_cli_arg(argv: List[str], key: str, value: str) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == key:
            i += 2
            continue
        out.append(argv[i])
        i += 1
    out.extend([key, value])
    return out


def _hard_restart_self(run_root: Path) -> None:
    script_path = str(Path(__file__).resolve())
    argv = list(sys.argv[1:])
    argv = _replace_or_append_cli_arg(argv, "--run_root_override", str(run_root))
    cmd = [sys.executable, script_path] + argv
    print(f"[hard-restart] execv: {' '.join(cmd)}")
    os.execv(sys.executable, cmd)


class VectorEnvAPICompatWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        out = self.env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            return out
        return out, {}

    def step(self, action):
        out = self.env.step(action)

        if len(out) == 5:
            return out

        obs, reward, done, info = out
        truncated = bool(info.get("TimeLimit.truncated", False))
        terminated = bool(done) and not truncated
        return obs, reward, terminated, truncated, info


class BoundarySaveSAC(SAC):
    def __init__(
        self,
        *args,
        rollout_log_path: Union[str, Path],
        rollout_log_interval_s: float,
        cycle_number_1based: int,
        behavior_number_1based: int,
        retry_idx: int,
        task_name: str,
        phase_progress_every_pct: float = 0.0,
        phase_progress_callback=None,
        collector_env=None,
        num_envs: int = 1,
        updates_per_collect: Optional[int] = None,
        disable_eval: bool = False,
        plateau_min_steps: int = 250_000,
        plateau_episode_window: int = 100,
        plateau_min_return: float = 0.0,
        plateau_rel_change: float = 0.05,
        plateau_std_coeff: float = 1.0,
        restart_floor_return: float = 0.0,
        restart_min_frac: float = 1.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if updates_per_collect is not None and int(updates_per_collect) <= 0:
            raise ValueError("updates_per_collect must be > 0 when provided.")
        self.updates_per_collect = int(updates_per_collect) if updates_per_collect is not None else None

        self.rollout_log_path = Path(rollout_log_path)
        self.rollout_log_interval_s = float(rollout_log_interval_s)
        self.cycle_number_1based = int(cycle_number_1based)
        self.behavior_number_1based = int(behavior_number_1based)
        self.retry_idx = int(retry_idx)
        self.task_name = str(task_name)
        self.disable_eval = bool(disable_eval)

        self.plateau_min_steps = int(plateau_min_steps)
        self.plateau_episode_window = int(plateau_episode_window)
        self.plateau_min_return = float(plateau_min_return)
        self.plateau_rel_change = float(plateau_rel_change)
        self.plateau_std_coeff = float(plateau_std_coeff)
        self.restart_floor_return = float(restart_floor_return)
        self.restart_min_frac = float(restart_min_frac)

        self.phase_progress_every_pct = float(phase_progress_every_pct)
        self.phase_progress_callback = phase_progress_callback
        if 0.0 < self.phase_progress_every_pct < 100.0:
            self._next_phase_progress_pct: Optional[float] = self.phase_progress_every_pct
        else:
            self._next_phase_progress_pct = None

        self.collector_env = collector_env
        self.num_collect_envs = int(num_envs)
        if self.num_collect_envs < 1:
            raise ValueError("num_envs must be >= 1")

        self._episodes_done: int = 0
        self._recent_returns: Deque[float] = deque(maxlen=100)
        self._recent_lengths: Deque[int] = deque(maxlen=100)
        self._plateau_metric_history: Deque[float] = deque(maxlen=max(2 * self.plateau_episode_window, 1))
        self._last_episode_return: float = np.nan
        self._last_episode_length: int = -1
        self._last_rollout_log_ts: float = 0.0

        self._restart_floor_checked: bool = False
        self._stop_reason: str = ""
        self._phase_steps: int = 0
        self._plateau_mu_prev: float = np.nan
        self._plateau_mu_recent: float = np.nan
        self._plateau_sigma: float = np.nan
        self._final_window_mean: float = np.nan

        self._ensure_rollout_header()
        self._prepare_all_trainable_variables_and_optimizer()

    def save_model(self, current_task_idx):
        return

    def _ensure_models_built(self) -> None:
        dummy_obs = tf.zeros((1, self.obs_dim), dtype=tf.float32)
        dummy_act = tf.zeros((1, self.act_dim), dtype=tf.float32)

        _ = self.actor(dummy_obs)
        _ = self.critic1(dummy_obs, dummy_act)
        _ = self.target_critic1(dummy_obs, dummy_act)
        _ = self.critic2(dummy_obs, dummy_act)
        _ = self.target_critic2(dummy_obs, dummy_act)

    def _prepare_all_trainable_variables_and_optimizer(self) -> None:
        self._ensure_models_built()

        self.critic_variables = self.critic1.trainable_variables + self.critic2.trainable_variables
        self.all_common_variables = (
            self.actor.common_variables
            + self.critic1.common_variables
            + self.critic2.common_variables
        )

        all_vars = list(self.actor.trainable_variables) + list(self.critic_variables)
        if self.auto_alpha:
            all_vars += [self.all_log_alpha]

        try:
            self.optimizer.build(all_vars)
        except Exception:
            pass

    def load_actor_only(self, actor_ckpt_prefix: Union[str, Path]) -> None:
        self._ensure_models_built()
        self.actor.load_weights(str(actor_ckpt_prefix))

    @tf.function
    def get_action_batch(self, o, deterministic: bool = False):
        mu, log_std, pi, logp_pi = self.actor(o)
        return mu if deterministic else pi

    def load_full_checkpoint(self, checkpoint_dir: Union[str, Path]) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        self._ensure_models_built()

        self.actor.load_weights(str(checkpoint_dir / "actor"))
        self.critic1.load_weights(str(checkpoint_dir / "critic1"))
        self.target_critic1.load_weights(str(checkpoint_dir / "target_critic1"))
        self.critic2.load_weights(str(checkpoint_dir / "critic2"))
        self.target_critic2.load_weights(str(checkpoint_dir / "target_critic2"))

        alpha_path = checkpoint_dir / "all_log_alpha.npy"
        if self.auto_alpha and alpha_path.exists():
            self.all_log_alpha.assign(np.load(alpha_path))

    def save_boundary_checkpoint(self, checkpoint_dir: Union[str, Path]) -> Path:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._ensure_models_built()

        self.actor.save_weights(str(checkpoint_dir / "actor"))
        self.critic1.save_weights(str(checkpoint_dir / "critic1"))
        self.target_critic1.save_weights(str(checkpoint_dir / "target_critic1"))
        self.critic2.save_weights(str(checkpoint_dir / "critic2"))
        self.target_critic2.save_weights(str(checkpoint_dir / "target_critic2"))

        if self.auto_alpha:
            np.save(checkpoint_dir / "all_log_alpha.npy", self.all_log_alpha.numpy())

        return checkpoint_dir / "actor"

    def _ensure_rollout_header(self) -> None:
        self.rollout_log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.rollout_log_path.exists():
            return
        with self.rollout_log_path.open("w") as f:
            f.write(
                "time_s,cycle,behavior,retry_idx,task,episodes_done,total_env_steps,current_task_steps,phase_steps,"
                "avg_return_window,avg_len_window,last_episode_return,last_episode_len,stop_reason\n"
            )

    def _current_avg_return(self) -> float:
        if self._recent_returns:
            return float(np.mean(self._recent_returns))
        return float("nan")

    def _current_avg_len(self) -> float:
        if self._recent_lengths:
            return float(np.mean(self._recent_lengths))
        return float("nan")

    def _append_rollout_line(
        self,
        elapsed_s: float,
        global_timestep: int,
        current_task_timestep: int,
        stop_reason: str = "",
    ) -> None:
        avg_return = self._current_avg_return()
        avg_len = self._current_avg_len()

        if not np.isnan(avg_return):
            self._final_window_mean = avg_return

        with self.rollout_log_path.open("a") as f:
            f.write(
                f"{elapsed_s:.3f},"
                f"{self.cycle_number_1based},"
                f"{self.behavior_number_1based},"
                f"{self.retry_idx},"
                f"{self.task_name},"
                f"{self._episodes_done},"
                f"{global_timestep + 1},"
                f"{current_task_timestep + 1},"
                f"{self._phase_steps},"
                f"{avg_return:.6f},"
                f"{avg_len:.6f},"
                f"{self._last_episode_return:.6f},"
                f"{self._last_episode_length},"
                f"{stop_reason}\n"
            )
            f.flush()
            os.fsync(f.fileno())

    def _maybe_log_rollout(
        self,
        elapsed_s: float,
        global_timestep: int,
        current_task_timestep: int,
        force: bool = False,
        stop_reason: str = "",
    ) -> None:
        now = time.time()
        if force or (now - self._last_rollout_log_ts >= self.rollout_log_interval_s):
            self._append_rollout_line(
                elapsed_s=elapsed_s,
                global_timestep=global_timestep,
                current_task_timestep=current_task_timestep,
                stop_reason=stop_reason,
            )
            self._last_rollout_log_ts = now

    def _maybe_trigger_phase_progress_checkpoint(self, completed_steps: int) -> None:
        if self._next_phase_progress_pct is None or self.phase_progress_callback is None:
            return

        while self._next_phase_progress_pct is not None:
            threshold_step = int(np.ceil(self.steps * (self._next_phase_progress_pct / 100.0)))

            if threshold_step >= self.steps:
                self._next_phase_progress_pct = None
                break

            if completed_steps < threshold_step:
                break

            fired_pct = self._next_phase_progress_pct
            try:
                self.phase_progress_callback(fired_pct, completed_steps)
            except Exception as e:
                print(
                    f"[checkpoint] failed progress checkpoint at {fired_pct:.2f}% "
                    f"for task={self.task_name}: {e}"
                )

            next_pct = round(fired_pct + self.phase_progress_every_pct, 10)
            self._next_phase_progress_pct = next_pct if next_pct < 100.0 else None

    def _collector_reset(self) -> np.ndarray:
        if self.collector_env is None:
            reset_out = self.env.reset()
            if isinstance(reset_out, tuple) and len(reset_out) == 2:
                obs, _reset_info = reset_out
            else:
                obs = reset_out
            obs = np.asarray(obs, dtype=np.float32)
            return obs[None, ...]

        reset_out = self.collector_env.reset()
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs, _reset_info = reset_out
        else:
            obs = reset_out
        return np.asarray(obs, dtype=np.float32)

    def _collector_step(self, actions: np.ndarray):
        if self.collector_env is None:
            step_out = self.env.step(np.asarray(actions[0]))
            if len(step_out) == 5:
                next_obs, reward, terminated, truncated, info = step_out
            else:
                next_obs, reward, done, info = step_out
                terminated = bool(done)
                truncated = bool(info.get("TimeLimit.truncated", False))

            next_obs = np.asarray(next_obs, dtype=np.float32)[None, ...]
            rewards = np.asarray([reward], dtype=np.float32)
            terminations = np.asarray([terminated], dtype=np.bool_)
            truncations = np.asarray([truncated], dtype=np.bool_)
            return next_obs, rewards, terminations, truncations, info

        step_out = self.collector_env.step(np.asarray(actions))
        if len(step_out) == 5:
            next_obs, rewards, terminations, truncations, infos = step_out
        else:
            next_obs, rewards, dones, infos = step_out
            terminations = np.asarray(dones, dtype=np.bool_)
            truncations = np.zeros_like(terminations, dtype=np.bool_)

        next_obs = np.asarray(next_obs, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        terminations = np.asarray(terminations, dtype=np.bool_)
        truncations = np.asarray(truncations, dtype=np.bool_)
        return next_obs, rewards, terminations, truncations, infos

    def _sample_random_actions(self) -> np.ndarray:
        if self.collector_env is None:
            action = self.env.action_space.sample()
            return np.asarray(action, dtype=np.float32)[None, ...]

        actions = [
            self.collector_env.single_action_space.sample()
            for _ in range(self.num_collect_envs)
        ]
        return np.asarray(actions, dtype=np.float32)

    def _extract_store_next_obs(
        self,
        next_obs_batch: np.ndarray,
        infos,
        idx: int,
    ) -> np.ndarray:
        if not isinstance(infos, dict):
            return np.asarray(next_obs_batch[idx], dtype=np.float32)

        final_obs = infos.get("final_observation", None)
        final_obs_mask = infos.get("_final_observation", None)

        if final_obs is None or final_obs_mask is None:
            return np.asarray(next_obs_batch[idx], dtype=np.float32)

        try:
            if bool(final_obs_mask[idx]):
                return np.asarray(final_obs[idx], dtype=np.float32)
        except Exception:
            pass

        return np.asarray(next_obs_batch[idx], dtype=np.float32)

    def _check_restart_floor(self, env_steps_done: int) -> bool:
        if self.restart_min_frac > 1.0:
            return False

        if self._restart_floor_checked:
            return False

        threshold_step = int(np.ceil(self.steps * self.restart_min_frac))
        if env_steps_done < threshold_step:
            return False

        self._restart_floor_checked = True
        avg_return = self._current_avg_return()
        if np.isnan(avg_return):
            return False

        return avg_return < self.restart_floor_return

    def _check_plateau(self, env_steps_done: int) -> bool:
        if env_steps_done < self.plateau_min_steps:
            return False

        if len(self._plateau_metric_history) < 2 * self.plateau_episode_window:
            return False

        arr = np.asarray(self._plateau_metric_history, dtype=np.float64)
        prev = arr[-2 * self.plateau_episode_window : -self.plateau_episode_window]
        recent = arr[-self.plateau_episode_window :]

        mu_prev = float(prev.mean())
        mu_recent = float(recent.mean())
        sigma = float(arr[-2 * self.plateau_episode_window :].std(ddof=0))
        rel_change = abs(mu_recent - mu_prev) / max(abs(mu_prev), 1e-8)

        cond_min = mu_recent >= self.plateau_min_return
        cond_rel = rel_change <= self.plateau_rel_change
        cond_std = abs(mu_recent - mu_prev) <= self.plateau_std_coeff * sigma

        self._plateau_mu_prev = mu_prev
        self._plateau_mu_recent = mu_recent
        self._plateau_sigma = sigma

        return cond_min and cond_rel and cond_std

    def close_collector_env(self) -> None:
        if self.collector_env is not None:
            try:
                self.collector_env.close()
            except Exception:
                pass

    def run(self) -> Dict[str, Union[str, float, int]]:
        self.start_time = time.time()
        self._last_rollout_log_ts = self.start_time

        obs_batch = self._collector_reset()
        episode_returns = np.zeros(self.num_collect_envs, dtype=np.float32)
        episode_lens = np.zeros(self.num_collect_envs, dtype=np.int32)

        current_task_idx = getattr(self.env, "cur_seq_idx", -1)
        self.learn_on_batch = self.get_learn_on_batch(current_task_idx)

        info = {}
        env_steps_done = 0
        next_update_step = int(self.update_after)
        next_log_step = int(self.log_every)

        while env_steps_done < self.steps:
            remaining_steps = self.steps - env_steps_done
            batch_limit = min(self.num_collect_envs, remaining_steps)

            if env_steps_done > self.start_steps or (
                self.agent_policy_exploration and current_task_idx > 0
            ):
                obs_tensor = tf.convert_to_tensor(obs_batch, dtype=tf.float32)
                actions = self.get_action_batch(obs_tensor, deterministic=False)
                if hasattr(actions, "numpy"):
                    actions = actions.numpy()
                actions = np.asarray(actions, dtype=np.float32)
            else:
                actions = self._sample_random_actions()

            next_obs_batch, rewards, terminations, truncations, info = self._collector_step(actions)
            dones = np.asarray(terminations | truncations, dtype=np.bool_)

            episode_returns[:batch_limit] += rewards[:batch_limit]
            episode_lens[:batch_limit] += 1

            for i in range(batch_limit):
                next_obs_to_store = self._extract_store_next_obs(next_obs_batch, info, i)
                done_to_store = bool(dones[i])
                if bool(truncations[i]):
                    done_to_store = False

                self.replay_buffer.store(
                    np.asarray(obs_batch[i], dtype=np.float32),
                    np.asarray(actions[i], dtype=np.float32),
                    float(rewards[i]),
                    next_obs_to_store,
                    done_to_store,
                )

                if bool(dones[i]) or int(episode_lens[i]) == self.max_episode_len:
                    ep_return = float(episode_returns[i])
                    ep_len = int(episode_lens[i])

                    self.logger.store({"train/return": ep_return, "train/ep_length": ep_len})

                    self._episodes_done += 1
                    self._last_episode_return = ep_return
                    self._last_episode_length = ep_len
                    self._recent_returns.append(ep_return)
                    self._recent_lengths.append(ep_len)

                    current_avg_return = self._current_avg_return()
                    if not np.isnan(current_avg_return):
                        self._plateau_metric_history.append(current_avg_return)
                        self._final_window_mean = current_avg_return

                    episode_returns[i] = 0.0
                    episode_lens[i] = 0

                    if self.collector_env is None:
                        reset_out = self.env.reset()
                        if isinstance(reset_out, tuple) and len(reset_out) == 2:
                            reset_obs, _reset_info = reset_out
                        else:
                            reset_obs = reset_out
                        next_obs_batch[i] = np.asarray(reset_obs, dtype=np.float32)

            obs_batch = next_obs_batch
            env_steps_done += batch_limit
            self._phase_steps = env_steps_done
            updates_per_collect = self.update_every if self.updates_per_collect is None else self.updates_per_collect

            while env_steps_done >= next_update_step:
                for _ in range(updates_per_collect):
                    batch = self.replay_buffer.sample_batch(self.batch_size)
                    episodic_batch = self.get_episodic_batch(current_task_idx)
                    results = self.learn_on_batch(
                        tf.convert_to_tensor(current_task_idx), batch, episodic_batch
                    )
                    self._log_after_update(results)
                next_update_step += self.update_every

            if env_steps_done >= next_log_step or env_steps_done >= self.steps:
                if not self.disable_eval:
                    epoch = (env_steps_done + self.log_every - 1) // self.log_every
                    self._log_after_epoch(
                        epoch=epoch,
                        current_task_timestep=max(0, env_steps_done - 1),
                        global_timestep=max(0, env_steps_done - 1),
                        info=info,
                    )

                while next_log_step <= env_steps_done:
                    next_log_step += self.log_every

            elapsed_s = time.time() - self.start_time
            self._maybe_log_rollout(
                elapsed_s=elapsed_s,
                global_timestep=max(0, env_steps_done - 1),
                current_task_timestep=max(0, env_steps_done - 1),
                force=False,
                stop_reason="",
            )

            self._maybe_trigger_phase_progress_checkpoint(env_steps_done)

            if self._check_restart_floor(env_steps_done):
                self._stop_reason = "PHASE_RESTART_FLOOR"
                break

            if self._check_plateau(env_steps_done):
                self._stop_reason = "PHASE_PLATEAU"
                break

        if not self._stop_reason:
            self._stop_reason = "STOP_AFTER_STEPS"

        elapsed_s = time.time() - self.start_time
        final_global_step = max(0, min(env_steps_done, self.steps) - 1)
        self._maybe_log_rollout(
            elapsed_s=elapsed_s,
            global_timestep=final_global_step,
            current_task_timestep=final_global_step,
            force=True,
            stop_reason=self._stop_reason,
        )

        return {
            "stop_reason": self._stop_reason,
            "phase_steps": int(env_steps_done),
            "episodes_done": int(self._episodes_done),
            "plateau_mu_prev": float(self._plateau_mu_prev),
            "plateau_mu_recent": float(self._plateau_mu_recent),
            "plateau_sigma": float(self._plateau_sigma),
            "final_window_mean": float(self._final_window_mean),
        }


def _spawn_video_recorder(
    repo_root: Path,
    task_name: str,
    actor_ckpt_prefix: Path,
    out_path: Path,
    episodes: int,
    width: int,
    height: int,
    fps: int,
    seed: int,
    camera_name: Optional[str],
    hidden_sizes: List[int],
    activation: str,
    use_layer_norm: bool,
    hide_task_id: bool,
) -> Optional[subprocess.Popen]:
    video_script = repo_root / "record_actor_video.py"
    if not video_script.exists():
        print(f"[video] Skipping video: {video_script} does not exist.")
        return None

    cmd = [
        sys.executable,
        str(video_script),
        "--task", task_name,
        "--actor_ckpt", str(actor_ckpt_prefix),
        "--out", str(out_path),
        "--episodes", str(episodes),
        "--width", str(width),
        "--height", str(height),
        "--fps", str(fps),
        "--seed", str(seed),
        "--hidden_sizes", *[str(x) for x in hidden_sizes],
        "--activation", activation,
    ]

    if camera_name:
        cmd.extend(["--camera_name", camera_name])
    if use_layer_norm:
        cmd.append("--use_layer_norm")
    if hide_task_id:
        cmd.append("--hide_task_id")

    log_path = out_path.with_suffix(".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    child_env = os.environ.copy()
    child_env["CUDA_VISIBLE_DEVICES"] = ""

    with log_path.open("wb") as f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=child_env,
        )

    print(f"[video] launched pid={proc.pid} -> {out_path.name}")
    return proc


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("run_cycles_v10.py")

    p.add_argument("--tasks", type=str, required=True,
                   help="Comma-separated task list, e.g. reach-v1,push-v1,pick-place-v1")
    p.add_argument("--num_cycles", type=int, required=True)
    p.add_argument("--run_tag", type=str, required=True)

    p.add_argument("--resume_from", type=str, default=None,
                   help="Path to a full boundary checkpoint directory. Raw actor prefix is also accepted as fallback.")
    p.add_argument("--runs_root", type=str, default="runs")
    p.add_argument("--run_root_override", type=str, default=None)
    p.add_argument("--logger_output", type=str, default="tsv,tensorboard")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps_per_task", type=int, default=300000)
    p.add_argument("--plateau_min_steps", type=int, default=250000)
    p.add_argument("--plateau_episode_window", type=int, default=100)
    p.add_argument("--plateau_rel_change", type=float, default=0.05)
    p.add_argument("--plateau_std_coeff", type=float, default=1.0)
    p.add_argument("--plateau_min_return", type=float, default=0.0)
    p.add_argument("--restart_floor_return", type=float, default=0.0)
    p.add_argument("--restart_min_frac", type=float, default=1.1)
    p.add_argument("--min_phase_mean_reward_on_switch", type=float, default=0.0)
    p.add_argument("--phase_retry_max", type=int, default=0)

    p.add_argument("--log_every", type=int, default=10000)
    p.add_argument("--rollout_log_interval_s", type=float, default=30.0)
    p.add_argument(
        "--checkpoint_video_every_pct",
        type=float,
        default=0.0,
        help="If > 0 and < 100, save a checkpoint and launch video every X percent of each phase.",
    )

    p.add_argument("--num_envs", type=int, default=1,
                   help="Number of CPU collector envs to run in parallel. 1 preserves old behavior.")
    p.add_argument("--learner_device", type=str, default="gpu", choices=["auto", "cpu", "gpu"],
                   help="Where to place the TensorFlow learner.")
    p.add_argument("--gpu_id", type=int, default=0,
                   help="GPU index to use when --learner_device is gpu/auto and a GPU is available.")

    p.add_argument("--replay_size", type=int, default=1000000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--hidden_sizes", type=str, default="256,256,256,256")
    p.add_argument("--activation", type=str, default="lrelu")
    p.add_argument("--use_layer_norm", action="store_true", default=True)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--alpha", type=str, default="auto")
    p.add_argument("--target_output_std", type=float, default=None)
    p.add_argument("--clipnorm", type=float, default=None)
    p.add_argument("--start_steps", type=int, default=10000)
    p.add_argument("--update_after", type=int, default=1000)
    p.add_argument("--update_every", type=int, default=50)
    p.add_argument("--num_test_eps_stochastic", type=int, default=0)
    p.add_argument("--num_test_eps_deterministic", type=int, default=0)
    p.add_argument("--disable_eval", action="store_true", default=False)
    p.add_argument("--max_episode_len", type=int, default=200)
    p.add_argument("--agent_policy_exploration", action="store_true", default=False)

    p.add_argument("--hide_task_id", action="store_true", default=True)

    p.add_argument("--disable_videos", action="store_true", default=False)
    p.add_argument("--video_episodes", type=int, default=1)
    p.add_argument("--video_width", type=int, default=640)
    p.add_argument("--video_height", type=int, default=480)
    p.add_argument("--video_fps", type=int, default=30)
    p.add_argument("--video_camera_name", type=str, default=None)
    p.add_argument("--updates_per_collect", type=int, default=None)

    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    tasks = _parse_tasks(args.tasks)
    hidden_sizes = _parse_hidden_sizes(args.hidden_sizes)
    logger_output = _parse_logger_output(args.logger_output)
    alpha = _parse_alpha(args.alpha)
    tf_device = _configure_tf_device(args.learner_device, args.gpu_id)
    resume_info = _resolve_resume(args.resume_from)

    disable_eval = bool(args.disable_eval or (
        args.num_test_eps_stochastic == 0 and args.num_test_eps_deterministic == 0
    ))

    if args.run_root_override is not None:
        run_root = Path(args.run_root_override).expanduser().resolve()
        run_root.mkdir(parents=True, exist_ok=True)
        print(f"[restart] reusing run root override: {run_root}")
    elif resume_info["run_root"] is not None:
        run_root = Path(resume_info["run_root"]).resolve()
        print(f"[resume] reusing existing run root: {run_root}")
    else:
        run_root = (
            Path(args.runs_root).expanduser().resolve()
            / f"{args.run_tag}_{_timestamp()}"
        )
        run_root.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = run_root / "checkpoints"
    accepted_dir = checkpoints_dir / "accepted"
    failed_dir = checkpoints_dir / "failed"
    progress_dir = checkpoints_dir / "progress"
    videos_dir = run_root / "videos"
    logs_dir = run_root / "logs"
    rollout_log_path = run_root / "rollout.csv"
    phase_state_path = run_root / "phase_state.json"
    latest_state_path = run_root / "latest_state.json"

    accepted_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)
    progress_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    config_payload = {
        "run_tag": args.run_tag,
        "tasks": tasks,
        "num_cycles": args.num_cycles,
        "seed": args.seed,
        "steps_per_task": args.steps_per_task,
        "plateau_min_steps": args.plateau_min_steps,
        "plateau_episode_window": args.plateau_episode_window,
        "plateau_rel_change": args.plateau_rel_change,
        "plateau_std_coeff": args.plateau_std_coeff,
        "plateau_min_return": args.plateau_min_return,
        "restart_floor_return": args.restart_floor_return,
        "restart_min_frac": args.restart_min_frac,
        "min_phase_mean_reward_on_switch": args.min_phase_mean_reward_on_switch,
        "phase_retry_max": args.phase_retry_max,
        "log_every": args.log_every,
        "rollout_log_interval_s": args.rollout_log_interval_s,
        "checkpoint_video_every_pct": args.checkpoint_video_every_pct,
        "num_envs": args.num_envs,
        "learner_device": args.learner_device,
        "gpu_id": args.gpu_id,
        "tf_device": tf_device,
        "replay_size": args.replay_size,
        "batch_size": args.batch_size,
        "hidden_sizes": hidden_sizes,
        "activation": args.activation,
        "use_layer_norm": args.use_layer_norm,
        "lr": args.lr,
        "gamma": args.gamma,
        "alpha": alpha,
        "target_output_std": args.target_output_std,
        "clipnorm": args.clipnorm,
        "start_steps": args.start_steps,
        "update_after": args.update_after,
        "update_every": args.update_every,
        "num_test_eps_stochastic": args.num_test_eps_stochastic,
        "num_test_eps_deterministic": args.num_test_eps_deterministic,
        "disable_eval": disable_eval,
        "max_episode_len": args.max_episode_len,
        "agent_policy_exploration": args.agent_policy_exploration,
        "hide_task_id": args.hide_task_id,
        "disable_videos": args.disable_videos,
        "video_episodes": args.video_episodes,
        "video_width": args.video_width,
        "video_height": args.video_height,
        "video_fps": args.video_fps,
        "video_camera_name": args.video_camera_name,
        "resume_from": args.resume_from,
        "run_root": str(run_root),
        "rollout_log_path": str(rollout_log_path),
        "updates_per_collect": args.updates_per_collect,
    }
    _write_json(run_root / "config.json", config_payload)

    state = _load_phase_state(phase_state_path)
    if not state:
        incoming_path = None
        if resume_info["resume_mode"] == "full" and resume_info["checkpoint_dir"] is not None:
            incoming_path = str(Path(resume_info["checkpoint_dir"]).resolve())
        elif resume_info["resume_mode"] == "actor_only" and resume_info["actor_ckpt"] is not None:
            incoming_path = str(Path(resume_info["actor_ckpt"]).resolve())

        state = {
            "cycle_idx": int(resume_info["start_cycle_idx"]),
            "behavior_idx": int(resume_info["start_behavior_idx"]),
            "retry_idx": 0,
            "incoming_resume_mode": resume_info["resume_mode"],
            "incoming_path": incoming_path,
            "done": False,
            "last_stop_reason": None,
        }
        _save_phase_state(phase_state_path, state)
        _write_json(latest_state_path, state)

    if bool(state.get("done", False)):
        print("[done] phase_state says run is complete.")
        return

    cycle_idx = int(state.get("cycle_idx", 0))
    behavior_idx = int(state.get("behavior_idx", 0))
    retry_idx = int(state.get("retry_idx", 0))
    incoming_resume_mode = state.get("incoming_resume_mode", None)
    incoming_path = state.get("incoming_path", None)

    if cycle_idx >= args.num_cycles:
        state["done"] = True
        _save_phase_state(phase_state_path, state)
        _write_json(latest_state_path, state)
        print("[done] all cycles complete")
        return

    task_name = tasks[behavior_idx]
    safe_task = _safe_task_name(task_name)
    phase_label = f"c{cycle_idx + 1:02d}_b{behavior_idx + 1:02d}_r{retry_idx:02d}_{safe_task}"
    phase_stamp = _timestamp()

    attempt_seed = int(args.seed) + 1000 * cycle_idx + 100 * behavior_idx + retry_idx
    _set_global_seed(attempt_seed)

    print("=" * 80)
    print(f"[phase] {phase_label} | task={task_name}")
    print(f"[phase] cycle_idx={cycle_idx} behavior_idx={behavior_idx} retry_idx={retry_idx} seed={attempt_seed}")
    print(f"[phase] incoming_resume_mode={incoming_resume_mode} incoming_path={incoming_path}")
    print("=" * 80)

    train_env = get_single_env(task_name)
    collector_env = _build_parallel_env(task_name, args.num_envs)
    test_envs = [] if disable_eval else [get_single_env(task_name)]

    phase_config = dict(config_payload)
    phase_config.update(
        {
            "task_name": task_name,
            "cycle_idx": cycle_idx,
            "behavior_idx": behavior_idx,
            "retry_idx": retry_idx,
            "phase_label": phase_label,
            "phase_stamp": phase_stamp,
            "attempt_seed": attempt_seed,
        }
    )

    logger = EpochLogger(
        logger_output,
        config=phase_config,
        group_id=f"{args.run_tag}/{phase_label}",
    )

    logger_output_dir = getattr(logger, "output_dir", None)
    if logger_output_dir is not None:
        _maybe_symlink(logger_output_dir, logs_dir / phase_label)

    actor_kwargs = dict(
        hidden_sizes=hidden_sizes,
        activation=get_activation_from_str(args.activation),
        use_layer_norm=args.use_layer_norm,
        num_heads=1,
        hide_task_id=args.hide_task_id,
    )
    critic_kwargs = dict(
        hidden_sizes=hidden_sizes,
        activation=get_activation_from_str(args.activation),
        use_layer_norm=args.use_layer_norm,
        num_heads=1,
        hide_task_id=args.hide_task_id,
    )

    video_procs: List[subprocess.Popen] = []

    def _phase_progress_callback(progress_pct: float, phase_step: int) -> None:
        progress_stamp = _timestamp()
        pct_tag = _safe_pct_tag(progress_pct)
        checkpoint_name = (
            f"{phase_label}_pct{pct_tag}_step{phase_step:07d}_{progress_stamp}"
        )
        checkpoint_dir = progress_dir / checkpoint_name
        actor_prefix = sac.save_boundary_checkpoint(checkpoint_dir)

        checkpoint_meta = {
            "run_tag": args.run_tag,
            "run_root": str(run_root),
            "task_name": task_name,
            "tasks": tasks,
            "cycle_idx": cycle_idx,
            "behavior_idx": behavior_idx,
            "retry_idx": retry_idx,
            "cycle_number_1based": cycle_idx + 1,
            "behavior_number_1based": behavior_idx + 1,
            "checkpoint_name": checkpoint_name,
            "checkpoint_dir": str(checkpoint_dir),
            "actor_ckpt_prefix": str(actor_prefix),
            "timestamp": progress_stamp,
            "checkpoint_kind": "progress",
            "progress_percent": progress_pct,
            "phase_step": phase_step,
            "phase_total_steps": args.steps_per_task,
            "stop_reason": None,
        }
        _write_json(checkpoint_dir / "meta.json", checkpoint_meta)

        print(
            f"[checkpoint] progress save @ {progress_pct:.2f}% "
            f"({phase_step}/{args.steps_per_task}) -> {checkpoint_dir}"
        )

        if not args.disable_videos:
            video_out = videos_dir / f"{checkpoint_name}.mp4"
            proc = _spawn_video_recorder(
                repo_root=repo_root,
                task_name=task_name,
                actor_ckpt_prefix=actor_prefix,
                out_path=video_out,
                episodes=args.video_episodes,
                width=args.video_width,
                height=args.video_height,
                fps=args.video_fps,
                seed=attempt_seed,
                camera_name=args.video_camera_name,
                hidden_sizes=hidden_sizes,
                activation=args.activation,
                use_layer_norm=args.use_layer_norm,
                hide_task_id=args.hide_task_id,
            )
            if proc is not None:
                video_procs.append(proc)

    with tf.device(tf_device):
        sac = BoundarySaveSAC(
            env=train_env,
            test_envs=test_envs,
            logger=logger,
            seed=attempt_seed,
            steps=args.steps_per_task,
            log_every=args.log_every,
            replay_size=args.replay_size,
            batch_size=args.batch_size,
            actor_cl=models.MlpActor,
            actor_kwargs=actor_kwargs,
            critic_kwargs=critic_kwargs,
            lr=args.lr,
            gamma=args.gamma,
            alpha=alpha,
            target_output_std=args.target_output_std,
            clipnorm=args.clipnorm,
            start_steps=args.start_steps,
            update_after=args.update_after,
            update_every=args.update_every,
            num_test_eps_stochastic=args.num_test_eps_stochastic,
            num_test_eps_deterministic=args.num_test_eps_deterministic,
            max_episode_len=args.max_episode_len,
            save_freq_epochs=10**18,
            agent_policy_exploration=args.agent_policy_exploration,
            rollout_log_path=rollout_log_path,
            rollout_log_interval_s=args.rollout_log_interval_s,
            cycle_number_1based=cycle_idx + 1,
            behavior_number_1based=behavior_idx + 1,
            retry_idx=retry_idx,
            task_name=task_name,
            phase_progress_every_pct=args.checkpoint_video_every_pct,
            phase_progress_callback=_phase_progress_callback,
            collector_env=collector_env,
            num_envs=args.num_envs,
            updates_per_collect=args.updates_per_collect,
            disable_eval=disable_eval,
            plateau_min_steps=args.plateau_min_steps,
            plateau_episode_window=args.plateau_episode_window,
            plateau_min_return=args.plateau_min_return,
            plateau_rel_change=args.plateau_rel_change,
            plateau_std_coeff=args.plateau_std_coeff,
            restart_floor_return=args.restart_floor_return,
            restart_min_frac=args.restart_min_frac,
        )

        if incoming_resume_mode == "full" and incoming_path is not None:
            print(f"[phase] loading full SAC weights from {incoming_path}")
            sac.load_full_checkpoint(incoming_path)
        elif incoming_resume_mode == "actor_only" and incoming_path is not None:
            print(f"[phase] loading actor-only weights from {incoming_path}")
            sac.load_actor_only(incoming_path)

        summary = sac.run()

    stop_reason = str(summary["stop_reason"])
    plateau_mu_recent = float(summary["plateau_mu_recent"])
    final_window_mean = float(summary["final_window_mean"])
    phase_steps = int(summary["phase_steps"])

    score_for_retry = plateau_mu_recent if not np.isnan(plateau_mu_recent) else final_window_mean
    need_retry = False
    if stop_reason == "PHASE_RESTART_FLOOR":
        need_retry = True
    elif stop_reason == "PHASE_PLATEAU":
        if not np.isnan(score_for_retry) and score_for_retry < float(args.min_phase_mean_reward_on_switch):
            need_retry = True

    if need_retry:
        final_parent_dir = failed_dir
        checkpoint_kind = "failed_attempt"
    else:
        final_parent_dir = accepted_dir
        checkpoint_kind = "accepted_boundary"

    checkpoint_name = f"{phase_label}_{phase_stamp}"
    checkpoint_dir = final_parent_dir / checkpoint_name
    actor_prefix = sac.save_boundary_checkpoint(checkpoint_dir)

    checkpoint_meta = {
        "run_tag": args.run_tag,
        "run_root": str(run_root),
        "task_name": task_name,
        "tasks": tasks,
        "cycle_idx": cycle_idx,
        "behavior_idx": behavior_idx,
        "retry_idx": retry_idx,
        "cycle_number_1based": cycle_idx + 1,
        "behavior_number_1based": behavior_idx + 1,
        "checkpoint_name": checkpoint_name,
        "checkpoint_dir": str(checkpoint_dir),
        "actor_ckpt_prefix": str(actor_prefix),
        "timestamp": phase_stamp,
        "checkpoint_kind": checkpoint_kind,
        "stop_reason": stop_reason,
        "phase_steps": phase_steps,
        "plateau_mu_recent": None if np.isnan(plateau_mu_recent) else plateau_mu_recent,
        "final_window_mean": None if np.isnan(final_window_mean) else final_window_mean,
    }

    _write_json(checkpoint_dir / "meta.json", checkpoint_meta)
    print(f"[checkpoint] saved -> {checkpoint_dir}")

    if not args.disable_videos:
        video_out = videos_dir / f"{checkpoint_name}.mp4"
        proc = _spawn_video_recorder(
            repo_root=repo_root,
            task_name=task_name,
            actor_ckpt_prefix=actor_prefix,
            out_path=video_out,
            episodes=args.video_episodes,
            width=args.video_width,
            height=args.video_height,
            fps=args.video_fps,
            seed=attempt_seed,
            camera_name=args.video_camera_name,
            hidden_sizes=hidden_sizes,
            activation=args.activation,
            use_layer_norm=args.use_layer_norm,
            hide_task_id=args.hide_task_id,
        )
        if proc is not None:
            video_procs.append(proc)

    if need_retry and retry_idx < int(args.phase_retry_max):
        state["retry_idx"] = retry_idx + 1
        state["last_stop_reason"] = stop_reason
        state["last_phase_steps"] = phase_steps
        state["last_checkpoint_dir"] = str(checkpoint_dir)
        print(
            f"[state] retrying same task: retry {state['retry_idx']}/{args.phase_retry_max} "
            f"reason={stop_reason} score={score_for_retry}"
        )
    else:
        if behavior_idx + 1 < len(tasks):
            next_cycle_idx = cycle_idx
            next_behavior_idx = behavior_idx + 1
        else:
            next_cycle_idx = cycle_idx + 1
            next_behavior_idx = 0

        state["cycle_idx"] = next_cycle_idx
        state["behavior_idx"] = next_behavior_idx
        state["retry_idx"] = 0
        state["incoming_resume_mode"] = "full"
        state["incoming_path"] = str(checkpoint_dir)
        state["last_stop_reason"] = stop_reason
        state["last_phase_steps"] = phase_steps
        state["last_checkpoint_dir"] = str(checkpoint_dir)
        print(
            f"[state] advancing to next phase/cycle; next_cycle_idx={next_cycle_idx} "
            f"next_behavior_idx={next_behavior_idx} incoming={checkpoint_dir}"
        )

    if int(state["cycle_idx"]) >= args.num_cycles:
        state["done"] = True
    else:
        state["done"] = False

    _save_phase_state(phase_state_path, state)
    _write_json(latest_state_path, state)

    try:
        sac.close_collector_env()
    except Exception:
        pass

    try:
        train_env.close()
    except Exception:
        pass
    for env in test_envs:
        try:
            env.close()
        except Exception:
            pass

    tf.keras.backend.clear_session()

    print("=" * 80)
    print(f"[done-attempt] stop_reason={stop_reason}")
    print(f"[done-attempt] run_root={run_root}")
    print(f"[done-attempt] rollout_log={rollout_log_path}")
    print(f"[done-attempt] checkpoint={checkpoint_dir}")
    print(f"[done-attempt] launched_video_jobs={len(video_procs)}")
    print("=" * 80)

    if bool(state.get("done", False)):
        print("[done] all cycles complete")
        return

    _hard_restart_self(run_root)


if __name__ == "__main__":
    main()