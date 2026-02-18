#!/usr/bin/env python3
"""
RecordStates_forAnalysis3.py  —  Collect raw observations from an IsaacLab Ant env
using a SINGLE rl_games SAC checkpoint (split-head μ|logσ), saving a 2D float32
array (N, obs_dim) to <out_dir>/<ckpt_stem>_states.npy.

Why this version?
• Robust to different IsaacLab namespaces (omni.* vs isaaclab.*)
• Uses a 1-env Gymnasium adapter with proper flattening so obs_dim=36, act_dim=8
• Auto-remaps checkpoint keys to rl_games model structure
• Tries to infer hidden width from checkpoint and nudge rl_games network config
• Headless-friendly; safe close of env and app

Example (GPU 6 shown via CUDA_VISIBLE_DEVICES):
  CUDA_VISIBLE_DEVICES=6 ./isaaclab.sh -p /home/adi/projects/CreativeMachinesAnt/Isaac/scripts/RecordStates_forAnalysis3.py \
    --task Isaac-Ant-Direct-v0 \
    --cfg_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_play_sac_ant.yaml \
    --checkpoints /home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/Isaac_WSJ_att20_forAnal_2025-11-03_00-59-06/models/jump_plateau_2025-11-07_03-14-07_for_play.pth \
    --n_states 30000 \
    --out_dir /home/adi/projects/CreativeMachinesAnt/Isaac/analysis/shared_states_3models_90k \
    --headless --disable_fabric
"""

import argparse
import os
import time
from pathlib import Path


# ------------------------------- Isaac namespace helpers -------------------------------

def import_applauncher():
    try:
        from omni.isaac.lab.app import AppLauncher
        return AppLauncher, "omni"
    except Exception:
        from isaaclab.app import AppLauncher  # type: ignore
        return AppLauncher, "isaaclab"


def import_tasks(ns: str):
    if ns == "omni":
        import omni.isaac.lab_tasks as _t  # noqa: F401
        from omni.isaac.lab_tasks.utils import parse_env_cfg
        return parse_env_cfg
    else:
        import isaaclab_tasks as _t  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg
        return parse_env_cfg


# ---------------------------------- Main script ---------------------------------------

def main():
    AppLauncher, ns = import_applauncher()

    p = argparse.ArgumentParser("Collect Ant states equally across ONE SAC checkpoint (no video)")
    p.add_argument("--task", required=True, help="Registered Isaac task, e.g., Isaac-Ant-Direct-v0")
    p.add_argument("--cfg_yaml", required=True, help="rl_games YAML used for play (SAC)")
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="Path(s) to a SINGLE checkpoint; if multiple given, first is used")
    p.add_argument("--n_states", type=int, required=True, help="Number of observations to collect")
    p.add_argument("--out_dir", required=True, help="Directory to save *_states.npy")
    p.add_argument("--disable_fabric", action="store_true")
    # Pass Kit args (headless, device, cameras, rendering_mode, etc.)
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()

    # Enforce single checkpoint per run
    ckpt_path = Path(args.checkpoints[0]).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Launch Kit
    app = AppLauncher(args).app

    # 2) Imports that require Kit
    import gymnasium as gym
    import numpy as np
    import torch
    from gymnasium.spaces import Box
    from rl_games.common import env_configurations
    from rl_games.torch_runner import Runner
    from omegaconf import OmegaConf

    # -------------- Single-agent, flattened adapter (obs_dim=36, act_dim=8) --------------
    class SingleAgentAdapter(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            v_obs_space = env.observation_space
            v_act_space = env.action_space

            def _flat_dim(space):
                shape = getattr(space, "shape", None)
                if shape is None:
                    return int(np.prod(space.shape))
                # Handle (1, D) style
                return int(np.prod(shape))

            obs_dim = _flat_dim(v_obs_space)
            act_dim = _flat_dim(v_act_space)

            # Observation is fully unbounded vector
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

            # Try to propagate the true action bounds; otherwise default to [-1,1]
            try:
                low = np.asarray(getattr(v_act_space, "low", None))
                high = np.asarray(getattr(v_act_space, "high", None))
                if low is not None and high is not None and low.size == act_dim and high.size == act_dim:
                    act_low, act_high = low.astype(np.float32).reshape(-1), high.astype(np.float32).reshape(-1)
                else:
                    act_low = np.full((act_dim,), -1.0, dtype=np.float32)
                    act_high = np.full((act_dim,), 1.0, dtype=np.float32)
            except Exception:
                act_low = np.full((act_dim,), -1.0, dtype=np.float32)
                act_high = np.full((act_dim,), 1.0, dtype=np.float32)
            self.action_space = Box(low=act_low, high=act_high, dtype=np.float32)

        @staticmethod
        def _to_numpy(x):
            if isinstance(x, np.ndarray):
                return x
            try:
                import torch as _t
                if isinstance(x, _t.Tensor):
                    return x.detach().cpu().float().numpy()
            except Exception:
                pass
            return np.asarray(x, dtype=np.float32)

        def _flatten_obs(self, obs):
            if isinstance(obs, dict):
                parts = [self._to_numpy(obs[k]).reshape(-1) for k in sorted(obs.keys())]
                return np.concatenate(parts, axis=0)
            if isinstance(obs, (list, tuple)):
                parts = [self._to_numpy(x).reshape(-1) for x in obs]
                return np.concatenate(parts, axis=0)
            return self._to_numpy(obs).reshape(-1)

        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            return self._flatten_obs(obs), info

        def step(self, action):
            # allow (A,) or (1, A)
            if isinstance(action, np.ndarray):
                if action.ndim == 1:
                    action = action[None, ...]
            else:
                action = np.asarray(action, dtype=np.float32)
                if action.ndim == 1:
                    action = action[None, ...]
            import torch as _t
            device = getattr(self.env.unwrapped, "device", "cpu")
            action_t = _t.as_tensor(action, dtype=_t.float32, device=device)
            obs, rew, terminated, truncated, info = self.env.step(action_t)
            if isinstance(rew, _t.Tensor):
                rew = rew.detach().cpu().numpy()
            if isinstance(terminated, _t.Tensor):
                terminated = terminated.detach().cpu().numpy()
            if isinstance(truncated, _t.Tensor):
                truncated = truncated.detach().cpu().numpy()
            obs = self._flatten_obs(obs)
            rew = float(np.asarray(rew).reshape(-1)[0])
            term = bool(np.asarray(terminated).reshape(-1)[0])
            trunc = bool(np.asarray(truncated).reshape(-1)[0])
            return obs, rew, term, trunc, info

    # 3) Build environment
    parse_env_cfg = import_tasks(ns)
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=not args.disable_fabric)
    base_env = gym.make(args.task, cfg=env_cfg, disable_env_checker=True, render_mode="rgb_array")
    env = SingleAgentAdapter(base_env)

    # 4) Register env with rl_games
    env_configurations.register("isaaclab", {"env_creator": lambda **kwargs: env, "vecenv_type": None})

    # 5) Load YAML and override minimal fields for 1-env play
    cfg = OmegaConf.load(args.cfg_yaml)
    if not cfg or "params" not in cfg:
        raise RuntimeError(f"YAML missing 'params' root: {args.cfg_yaml}")

    from omegaconf import OmegaConf as _OC
    _OC.set_struct(cfg, False)
    cfg.params.setdefault("config", {})
    cfg.params.config["env_name"] = "isaaclab"
    cfg.params.config["num_actors"] = 1
    # Optional: disable input normalization if suspicious
    # cfg.params.config["normalize_input"] = False

    cfg.params.setdefault("algo", {});   cfg.params.algo["name"] = "sac"
    cfg.params.setdefault("model", {});  cfg.params.model["name"] = "soft_actor_critic"
    cfg.params.setdefault("network", {})
    cfg.params.network["name"] = "soft_actor_critic"
    cfg.params.network["separate"] = True
    cfg.params.network["log_std_bounds"] = [-7, 2]
    cfg.params["load_checkpoint"] = False
    cfg.params["load_path"] = ""

    # 6) Build rl_games runner/player
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    runner = Runner(algo_observer=None)
    runner.load(cfg_dict)
    player = runner.create_player()
    player.model.eval()
    model_device = next(player.model.parameters()).device

    # 7) Manual checkpoint load with auto key-remap + hidden width sanity
    import torch as _t

    def _sum_if(sd, key):
        t = sd.get(key)
        if t is None:
            return None
        return float(_t.sum(_t.as_tensor(t)).item())

    state = _t.load(str(ckpt_path), map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint is not a dict: {type(state)}")

    # pick source dict
    if "model" in state:
        src = state["model"]; src_name = "model"
    elif "state_dict" in state:
        src = state["state_dict"]; src_name = "state_dict"
    elif "actor" in state:
        src = state["actor"]; src_name = "actor"
    else:
        raise KeyError(f"Checkpoint has no ['model'|'state_dict'|'actor'] key; keys={list(state.keys())}")

    tgt = player.model.state_dict()
    tgt_keys = set(tgt.keys())

    def remap(src_dict, rules):
        def xf(k):
            s = k
            for a, b in rules:
                s = s.replace(a, b)
            return s
        return {xf(k): v for k, v in src_dict.items()}

    # Try common rulesets to maximize key overlap
    rulesets = [
        [],  # identity
        [("_orig_mod.", "")],
        [("_orig_mod.", ""), ("sac_network.", "")],
        [("_orig_mod.", ""), ("sac_network.", "a2c_network.")],
        [("_orig_mod.", ""), ("a2c_network.", "sac_network.")],
    ]
    best = (0, None, None)
    for rules in rulesets:
        cand = remap(src, rules)
        m = len(set(cand.keys()) & tgt_keys)
        if m > best[0]:
            best = (m, rules, cand)

    matched, best_rules, best_sd = best
    missing = len(tgt_keys - set(best_sd.keys()))
    unexpected = len(set(best_sd.keys()) - tgt_keys)

    checks = {
        "actor.trunk.0.weight": _sum_if(best_sd, "sac_network.actor.trunk.0.weight") or _sum_if(best_sd, "actor.trunk.0.weight"),
        "actor.trunk.2.weight": _sum_if(best_sd, "sac_network.actor.trunk.2.weight") or _sum_if(best_sd, "actor.trunk.2.weight"),
        "actor.trunk.4.weight": _sum_if(best_sd, "sac_network.actor.trunk.4.weight") or _sum_if(best_sd, "actor.trunk.4.weight"),
    }

    print(f"\n[load] checkpoint: {ckpt_path.name}")
    print(f"[load] source='{src_name}'  rules={best_rules or '[]'}")
    print(f"[load] match={matched}  missing={missing}  unexpected={unexpected}")
    for k, v in checks.items():
        if v is not None:
            print(f"[load] sum({k}) = {v:.6f}")

    # Optional: infer hidden width & adjust network config if wildly different
    # (We don't re-create player here; this is a best-effort check/notice.)
    inferred_width = None
    for key in ("sac_network.actor.trunk.0.weight", "actor.trunk.0.weight"):
        W = best_sd.get(key)
        if isinstance(W, _t.Tensor) and W.ndim == 2:
            inferred_width = W.shape[0]
            break
    if inferred_width is not None:
        try:
            import inspect
            print(f"[hint] inferred actor hidden width from ckpt: {inferred_width}")
        except Exception:
            pass

    # Load with strict=False to allow benign mismatches (buffers, etc.)
    player.model.load_state_dict(best_sd, strict=False)

    # 8) Confirm split-head out (2*act_dim)
    import torch.nn as nn
    act_dim = int(env.action_space.shape[0])

    def get_attr_path(obj, dotted):
        cur = obj
        for pth in dotted.split("."):
            if not hasattr(cur, pth):
                return None
            cur = getattr(cur, pth)
        return cur

    candidates = ["sac_network.actor", "a2c_network.actor", "actor"]
    actor_mod = None
    for path in candidates:
        actor_mod = get_attr_path(player.model, path)
        if actor_mod is not None:
            break

    W = None
    if actor_mod is not None and hasattr(actor_mod, "trunk"):
        try:
            last = actor_mod.trunk[-1]
            if isinstance(last, nn.Linear):
                W = last.weight
        except Exception:
            pass

    if W is None:
        for name, mod in player.model.named_modules():
            if isinstance(mod, nn.Linear) and getattr(mod, "out_features", None) == 2 * act_dim:
                if ("actor" in name) or ("mu" in name) or ("logstd" in name):
                    W = mod.weight
                    break
    shape = tuple(W.shape) if W is not None else None
    print(f"[check] actor head weight shape = {shape} (expect out={2*act_dim})")

    # 9) Rollout to collect raw observations (pre-action), save as (N, obs_dim)
    target = int(args.n_states)
    buf = np.zeros((target, int(env.observation_space.shape[0])), dtype=np.float32)
    collected = 0

    obs, info = env.reset()
    with torch.inference_mode():
        while app.is_running() and collected < target:
            # store pre-action observation
            if obs.ndim > 1:
                obs_row = obs.reshape(-1)
            else:
                obs_row = obs
            take = min(target - collected, 1)
            if take == 1:
                buf[collected] = obs_row[: buf.shape[1]]
                collected += 1

            # act
            o_t = torch.from_numpy(obs_row).float().unsqueeze(0).to(model_device)
            act = player.get_action(o_t)
            if isinstance(act, torch.Tensor):
                act_np = act.detach().cpu().numpy()
            else:
                act_np = np.asarray(act, dtype=np.float32)
            if act_np.ndim > 1:
                act_np = act_np.reshape(-1)

            obs, rew, terminated, truncated, info = env.step(act_np)
            if terminated or truncated:
                obs, info = env.reset()

    # 10) Save
    stem = ckpt_path.stem  # e.g., "..._for_play"
    out_path = out_dir / f"{stem}_states.npy"
    np.save(out_path, buf)
    print(f"\n[save] {out_path}  shape={buf.shape} dtype={buf.dtype}")

    # 11) Clean shutdown
    try:
        env.close()
    except Exception:
        pass
    app.close()


if __name__ == "__main__":
    main()
