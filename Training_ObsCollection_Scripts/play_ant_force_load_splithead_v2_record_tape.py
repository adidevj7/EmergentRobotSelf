#!/usr/bin/env python3 - ACUTALLY WORKING!
"""
Force-load an rl_games SAC checkpoint, auto-remap keys, verify split-head (mu|logσ),
and record a uniquely named video.

ADDED: record a joint-angle "tape" alongside the video:
  - Saves <prefix>_tape.npz containing:
      dof_pos_rad: (T, 8) float32
      actions:     (T, 8) float32
      rewards:     (T,)   float32
      dones:       (T,)   bool
      obs:         (T, 36) float32  (optional but useful)
      meta:        dict (saved as np.string_ JSON)

Run:
  CUDA_VISIBLE_DEVICES=2 ./isaaclab.sh -p /home/adi/projects/CreativeMachinesAnt/Isaac/scripts/play_ant_force_load_splithead_v2_record_tape.py \
    --task Isaac-Ant-Direct-v0 \
    --cfg_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_play_sac_ant_150.yaml \
    --checkpoint /home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att49_NewSpin_2025-11-20_12-52-41/models/c009_b01_walk_plateau_2025-11-22_14-57-33_for_play.pth \
    --steps 1000 \
    --video_dir /home/adi/projects/CreativeMachinesAnt/Isaac/videos \
    --tape_dir /home/adi/projects/CreativeMachinesAnt/Isaac/tapes \
    --headless --enable_cameras --rendering_mode quality
"""

#!/usr/bin/env python3 - ACUTALLY WORKING!
"""
Force-load an rl_games SAC checkpoint, auto-remap keys, verify split-head (mu|logσ),
and record a uniquely named video.

ADDED: record a joint-angle "tape" alongside the video:
  - Saves <prefix>_tape.npz containing:
      dof_pos_rad: (T, 8) float32
      actions:     (T, 8) float32
      rewards:     (T,)   float32
      dones:       (T,)   bool
      obs:         (T, 36) float32
      meta:        dict (saved as np.uint8 bytes of JSON)

Run:
  CUDA_VISIBLE_DEVICES=2 ./isaaclab.sh -p /home/adi/projects/CreativeMachinesAnt/Isaac/scripts/play_ant_force_load_splithead_v2_record_tape.py \
    --task Isaac-Ant-Direct-v0 \
    --cfg_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_play_sac_ant.yaml \
    --checkpoint /home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/.../manual_ep_1200000_for_play.pth \
    --steps 1000 \
    --video_dir /home/adi/projects/CreativeMachinesAnt/Isaac/videos \
    --tape_dir  /home/adi/projects/CreativeMachinesAnt/Isaac/tapes \
    --headless --enable_cameras --rendering_mode quality
"""
import argparse, time
from pathlib import Path

def import_applauncher():
    try:
        from omni.isaac.lab.app import AppLauncher
        return AppLauncher, "omni"
    except Exception:
        from isaaclab.app import AppLauncher
        return AppLauncher, "isaaclab"

def import_tasks(ns):
    if ns == "omni":
        import omni.isaac.lab_tasks as _t  # noqa
        from omni.isaac.lab_tasks.utils import parse_env_cfg
        return parse_env_cfg
    else:
        import isaaclab_tasks as _t  # noqa
        from isaaclab_tasks.utils import parse_env_cfg
        return parse_env_cfg

def main():
    AppLauncher, ns = import_applauncher()

    p = argparse.ArgumentParser("Force-load rl_games SAC checkpoint (split-head) and record video + joint tape")
    p.add_argument("--task", required=True)
    p.add_argument("--cfg_yaml", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--video_dir", type=str, default="videos")
    p.add_argument("--tape_dir", type=str, default="tapes")
    p.add_argument("--disable_fabric", action="store_true")
    # pass Kit args through (headless, cameras, device, rendering_mode)
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()

    # 1) Launch Kit
    app = AppLauncher(args).app

    # 2) Imports after Kit starts
    import gymnasium as gym
    import numpy as np
    import torch
    import json
    from gymnasium.wrappers import RecordVideo
    from gymnasium.spaces import Box
    from rl_games.common import env_configurations
    from rl_games.torch_runner import Runner
    from omegaconf import OmegaConf

    # --- single-agent adapter (num_envs=1) ---
    class SingleAgentAdapter(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            v_obs_space = env.observation_space
            v_act_space = env.action_space
            obs_dim = int(np.prod(v_obs_space.shape[-1:])) if len(v_obs_space.shape) >= 2 else int(np.prod(v_obs_space.shape))
            act_dim = int(np.prod(v_act_space.shape[-1:])) if len(v_act_space.shape) >= 2 else int(np.prod(v_act_space.shape))
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            act_low  = np.full((act_dim,), -1.0, dtype=np.float32)
            act_high = np.full((act_dim,),  1.0, dtype=np.float32)
            try:
                if hasattr(v_act_space, "low") and hasattr(v_act_space, "high"):
                    low = np.asarray(v_act_space.low).reshape(-1)[-act_dim:]
                    high = np.asarray(v_act_space.high).reshape(-1)[-act_dim:]
                    if low.shape == (act_dim,) and high.shape == (act_dim,):
                        act_low, act_high = low.astype(np.float32), high.astype(np.float32)
            except Exception:
                pass
            self.action_space = Box(low=act_low, high=act_high, dtype=np.float32)

        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            return self._flatten_obs(obs), info

        def step(self, action):
            if isinstance(action, np.ndarray):
                if action.ndim == 1: action = action[None, ...]
            else:
                action = np.asarray(action, dtype=np.float32)
                if action.ndim == 1: action = action[None, ...]
            import torch as _t
            device = getattr(self.env.unwrapped, "device", "cpu")
            action_t = _t.as_tensor(action, dtype=_t.float32, device=device)
            obs, rew, terminated, truncated, info = self.env.step(action_t)
            if isinstance(rew, _t.Tensor):        rew = rew.detach().cpu().numpy()
            if isinstance(terminated, _t.Tensor): terminated = terminated.detach().cpu().numpy()
            if isinstance(truncated, _t.Tensor):  truncated = truncated.detach().cpu().numpy()
            obs = self._flatten_obs(obs)
            rew = float(np.asarray(rew).reshape(-1)[0])
            term = bool(np.asarray(terminated).reshape(-1)[0])
            trunc = bool(np.asarray(truncated).reshape(-1)[0])
            return obs, rew, term, trunc, info

        @staticmethod
        def _to_numpy(x):
            if isinstance(x, np.ndarray): return x
            try:
                import torch as _t
                if isinstance(x, _t.Tensor): return x.detach().cpu().float().numpy()
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

    # 3) Build env (1 env) and video recorder with unique prefix
    parse_env_cfg = import_tasks(ns)
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=not args.disable_fabric)
    base_env = gym.make(args.task, cfg=env_cfg, disable_env_checker=True, render_mode="rgb_array")

    ckpt_path = Path(args.checkpoint)
    prefix = f"ant_{ckpt_path.stem}_{int(time.time())}"
    video_dir = Path(args.video_dir); video_dir.mkdir(parents=True, exist_ok=True)
    tape_dir = Path(args.tape_dir); tape_dir.mkdir(parents=True, exist_ok=True)

    tape_path = tape_dir / f"{prefix}_tape.npz"
    print(f"[tape] will save to: {tape_path.resolve()}", flush=True)

    recorded_env = RecordVideo(base_env, video_folder=str(video_dir),
                               episode_trigger=lambda i: True, name_prefix=prefix)
    env = SingleAgentAdapter(recorded_env)

    # 4) Register env with rl_games
    env_configurations.register("isaaclab", {"env_creator": lambda **kwargs: env, "vecenv_type": None})

    # 5) Load YAML but DISABLE internal auto-loading; we load manually
    cfg = OmegaConf.load(args.cfg_yaml)
    if not cfg or "params" not in cfg:
        raise RuntimeError(f"YAML missing 'params' root: {args.cfg_yaml}")
    from omegaconf import OmegaConf as _OC
    _OC.set_struct(cfg, False)
    cfg.params.setdefault("config", {})
    cfg.params.config["env_name"] = "isaaclab"
    cfg.params.config["num_actors"] = 1
    # cfg.params.config["normalize_input"] = False  # optional

    cfg.params.setdefault("algo", {});   cfg.params.algo["name"] = "sac"
    cfg.params.setdefault("model", {});  cfg.params.model["name"] = "soft_actor_critic"
    cfg.params.setdefault("network", {})
    cfg.params.network["name"] = "soft_actor_critic"
    cfg.params.network["separate"] = True
    cfg.params.network["log_std_bounds"] = [-7, 2]
    cfg.params["load_checkpoint"] = False
    cfg.params["load_path"] = ""

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # 6) Build runner & player
    runner = Runner(algo_observer=None)
    runner.load(cfg_dict)
    player = runner.create_player()
    player.model.eval()
    model_device = next(player.model.parameters()).device

    # 7) Manual load with key auto-remap
    import torch as _t
    def _sum_if(sd, key):
        t = sd.get(key)
        if t is None: return None
        return float(_t.sum(_t.as_tensor(t)).item())

    state = _t.load(str(ckpt_path), map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint is not a dict: {type(state)}")
    if   "model" in state:      src = state["model"]; src_name="model"
    elif "state_dict" in state: src = state["state_dict"]; src_name="state_dict"
    elif "actor" in state:      src = state["actor"]; src_name="actor"
    else:
        raise KeyError(f"Checkpoint has no ['model'|'state_dict'|'actor'] key; keys={list(state.keys())}")

    tgt = player.model.state_dict()
    tgt_keys = set(tgt.keys())

    def remap(src_dict, rules):
        def xf(k):
            s = k
            for a,b in rules: s = s.replace(a, b)
            return s
        return {xf(k): v for k, v in src_dict.items()}

    rulesets = [
        [],
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

    print(f"\n[load] checkpoint: {ckpt_path}")
    print(f"[load] source='{src_name}'  rules={best_rules or '[]'}")
    print(f"[load] match={matched}  missing={missing}  unexpected={unexpected}")
    for k,v in checks.items():
        if v is not None:
            print(f"[load] sum({k}) = {v:.6f}")

    if matched == 0:
        print("\n[load] match=0 → sample target keys:")
        for i,k in enumerate(list(tgt_keys)[:20]):
            print("   ", k)
        raise SystemExit("No keys matched. Mapping rules need adjustment.")

    player.model.load_state_dict(best_sd, strict=False)

    # ------------------- tape buffers -------------------
    steps_T = int(args.steps)
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])

    import numpy as np
    obs_buf = np.zeros((steps_T, obs_dim), dtype=np.float32)
    act_buf = np.zeros((steps_T, act_dim), dtype=np.float32)
    rew_buf = np.zeros((steps_T,), dtype=np.float32)
    done_buf = np.zeros((steps_T,), dtype=bool)
    dof_pos_buf = np.zeros((steps_T, 8), dtype=np.float32)

    DOF_POS_SLICE = slice(12, 20)

    meta = {
        "prefix": prefix,
        "checkpoint": str(ckpt_path),
        "task": args.task,
        "cfg_yaml": args.cfg_yaml,
        "steps": steps_T,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "dof_pos_slice": [DOF_POS_SLICE.start, DOF_POS_SLICE.stop],
        "note": "dof_pos extracted from flattened obs[12:20] per user mapping; verify if task obs layout changes.",
    }
    # ----------------------------------------------------

    # 9) Rollout and record
    obs, info = env.reset()
    steps = 0
    with torch.inference_mode():
        while app.is_running() and steps < steps_T:
            o_t = torch.from_numpy(obs).float().unsqueeze(0).to(model_device)
            act = player.get_action(o_t)
            act_np = act.detach().cpu().numpy() if isinstance(act, torch.Tensor) else np.asarray(act, dtype=np.float32)
            if act_np.ndim > 1: act_np = act_np.reshape(-1)

            obs_buf[steps] = obs
            act_buf[steps] = act_np
            dof_pos_buf[steps] = obs[DOF_POS_SLICE]

            obs, rew, terminated, truncated, info = env.step(act_np)
            rew_buf[steps] = float(rew)
            done_buf[steps] = bool(terminated or truncated)

            steps += 1
            if terminated or truncated:
                obs, info = env.reset()

    # ------------------- SAVE TAPE *BEFORE* closing app/env -------------------
    meta_json = json.dumps(meta).encode("utf-8")
    np.savez_compressed(
        tape_path,
        dof_pos_rad=dof_pos_buf[:steps],
        actions=act_buf[:steps],
        rewards=rew_buf[:steps],
        dones=done_buf[:steps],
        obs=obs_buf[:steps],
        meta=np.frombuffer(meta_json, dtype=np.uint8),
    )
    print(f"[tape] saved: {tape_path.resolve()}  steps={steps}", flush=True)
    try:
        print(f"[tape] exists? {tape_path.exists()}", flush=True)
    except Exception:
        pass
    # ------------------------------------------------------------------------

    env.close()
    app.close()
    print(f"\n[OK] Saved video(s) to: {video_dir.resolve()}  (prefix='{prefix}')", flush=True)

if __name__ == "__main__":
    main()
