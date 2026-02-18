# CreativeMachinesAnt/Isaac/tasks/ant_cfg_registry.py
"""
Config-first Ant task registry for IsaacLab.

What this does
--------------
- Registers Gymnasium task IDs (e.g., "Ant-Walk-v0") that expose an
  `env_cfg_entry_point` for IsaacLab's `parse_env_cfg(...)`.
- Each ID maps to a small EnvCfg subclass that inherits the stock
  `AntEnvCfg` and then applies overrides from a YAML in your repo.

Why this exists
---------------
- Lets you use IsaacLab utilities that expect: --task <ID> + CLI overrides.
- Centralizes behavior-specific knobs (episode length, terminations, reward scales)
  in clean YAML files you can version control and Hydra-compose.

Usage (in any script BEFORE parse_env_cfg)
------------------------------------------
    from pathlib import Path
    import sys
    repo = Path.home() / "projects" / "CreativeMachinesAnt" / "Isaac" / "tasks"
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from ant_cfg_registry import register_ant_cfg_tasks
    register_ant_cfg_tasks()

    from isaaclab_tasks.utils import parse_env_cfg
    env_cfg = parse_env_cfg("Ant-Walk-v0", device="cuda:0", num_envs=36, use_fabric=True)

    import gymnasium as gym
    env = gym.make("Isaac-Ant-Direct-v0", cfg=env_cfg, disable_env_checker=True, render_mode=None)

YAML location & override
------------------------
- Default YAML path for each behavior:
    ~/projects/CreativeMachinesAnt/Isaac/cfg/tasks/ant_walk.yaml
    ~/projects/CreativeMachinesAnt/Isaac/cfg/tasks/ant_spin.yaml
    ~/projects/CreativeMachinesAnt/Isaac/cfg/tasks/ant_jump.yaml
- You can override the YAML at runtime with env var:
    CM_TASK_YAML=/path/to/custom.yaml

Notes
-----
- These are *config-only* tasks; calling gym.make("Ant-Walk-v0") will raise.
- We also register aliases without "-v0" (e.g., "Ant-Walk") for convenience.
"""

from __future__ import annotations
import os
from pathlib import Path
import gymnasium as gym
# at top (imports) – add if not already present
        # 0) (optional) propagate GPU PhysX hints from YAML if present
        #    These keys are ignored gracefully if your Isaac version lacks them.
try:
    physx = y.get("sim", {}).get("physx", {})
    if "use_gpu" in physx:
        _set_if_has(self, "sim.physx.use_gpu", bool(physx["use_gpu"]))
    if "use_gpu_pipeline" in physx:
        _set_if_has(self, "sim.physx.use_gpu_pipeline", bool(physx["use_gpu_pipeline"]))
except Exception:
    pass


# Base Isaac Ant EnvCfg (dataclass-style config)
from isaaclab_tasks.direct.ant.ant_env import AntEnvCfg  # type: ignore

# Optional YAML loader (gracefully skip if not available)
try:
    from omegaconf import OmegaConf
except Exception:
    OmegaConf = None


# ---------------------------- helpers ----------------------------

def _get(cfg, path: str):
    """Traverse dotted path on a cfg object; return None if any segment missing."""
    cur = cfg
    for p in path.split("."):
        if not hasattr(cur, p):
            return None
        cur = getattr(cur, p)
    return cur

def _set_if_has(cfg, path: str, value) -> bool:
    """Set a dotted attribute if it exists; return True if set."""
    parts = path.split(".")
    cur = cfg
    for p in parts[:-1]:
        if not hasattr(cur, p):
            return False
        cur = getattr(cur, p)
    leaf = parts[-1]
    if hasattr(cur, leaf):
        setattr(cur, leaf, value)
        return True
    return False

def _disable_term_like(cfg, name: str):
    """Try to disable a known termination by flipping its enable flag (name varies across versions)."""
    term = _get(cfg, f"terminations.{name}")
    if term is None:
        return
    for key in ("enable", "enabled", "use"):
        if hasattr(term, key):
            setattr(term, key, False)
            return

def _set_time_out_steps(cfg, steps: int):
    """Prefer step-based time limit if present."""
    if _set_if_has(cfg, "terminations.time_out.num_steps", int(steps)):
        return
    # Some versions differ; add other fallbacks here if needed.

def _set_time_out_seconds(cfg, seconds: float):
    # common spellings across IsaacLab versions — try all quietly
    for path in (
        "terminations.time_out.time_limit_s",
        "terminations.time_out.timeout_s",
        "episode_length_s",                # some envs expose this directly
    ):
        if _set_if_has(cfg, path, float(seconds)):
            return True
    return False



# -------------------- YAML-backed EnvCfg base --------------------

class _YamlBackedAntCfg(AntEnvCfg):
    """Inherit stock AntEnvCfg, then apply overrides from a repo YAML."""
    YAML_DEFAULT: str = ""  # set in subclasses, e.g., "ant_walk.yaml"

    def __init__(self):
        super().__init__()
        # --- Force PhysX GPU pipeline if available ---
        sim = getattr(self, "sim", None)
        if sim is not None:
            # IsaacLab builds vary: try both common flags quietly.
            if hasattr(sim, "use_gpu_pipeline"):
                sim.use_gpu_pipeline = True
            if hasattr(sim, "use_gpu"):
                sim.use_gpu = True

        if OmegaConf is None:
            return  # silently skip if OmegaConf isn't available

        # Allow per-run override via env var (handy for sweeps)
        default_dir = Path.home() / "projects" / "CreativeMachinesAnt" / "Isaac" / "cfg" / "tasks"
        ypath = os.environ.get("CM_TASK_YAML", str(default_dir / self.YAML_DEFAULT))
        yfile = Path(ypath)
        if not yfile.exists():
            return  # no YAML found → keep base defaults

        y = OmegaConf.load(str(yfile))

        # 1) Episode length (steps) → time_out term
        steps = int(y.get("env", {}).get("episode_length_steps", 0))
        if steps > 0:
            # Try step-based first
            _set_time_out_steps(self, steps)
            # If that field doesn't exist in this Isaac version, set seconds-based:
            # environment step is (sim_dt * decimation) if available; else fall back to 1/60
            sim_dt = getattr(getattr(self, "sim", None), "dt", None) or getattr(self, "dt", None)
            decim  = getattr(self, "decimation", 1) or 1
            env_step = (float(sim_dt) * float(decim)) if sim_dt else (1.0 / 60.0)
            _set_time_out_seconds(self, steps * env_step)

        # 2) Only timeouts (disable “health”/“fall” style terms)
        if y.get("terminations", {}).get("only_timeout", False):
            for name in (
                "unhealthy_state",
                "fall",
                "unhealthy_tilt",
                "unhealthy_height",
                "out_of_bounds",
                "termination_height",
                "termination_tilt",
            ):
                _disable_term_like(self, name)

        # 3) Reward scales (names must match fields under rewards.scales.*)
        scales = y.get("rewards", {}).get("scales", {})
        for k, v in scales.items():
            _set_if_has(self, f"rewards.scales.{k}", float(v))


# -------------------- Behavior-specific cfgs --------------------



class AntWalkEnvCfg(_YamlBackedAntCfg):
    YAML_DEFAULT = "ant_walk.yaml"

    def __init__(self):
        super().__init__()
        # Force PhysX on GPU (safe across versions)
        try:
            if getattr(self, "sim", None) is None:
                # construct a minimal SimCfg if missing (rare)
                self.sim = SimCfg(physx=PhysxCfg())
            if getattr(self.sim, "physx", None) is None:
                self.sim.physx = PhysxCfg()

            self.sim.physx.use_gpu = True          # ← critical
            self.sim.physx.solver_type = "TGS"     # good default for GPU
            # generous defaults that avoid caps at high env counts:
            self.sim.physx.gpu_max_rigid_contact_count = 524288
            self.sim.physx.gpu_max_particle_contacts = 1048576
        except Exception:
            # stay silent; we’ll probe at runtime
            pass


class AntSpinEnvCfg(_YamlBackedAntCfg):
    YAML_DEFAULT = "ant_spin.yaml"

class AntJumpEnvCfg(_YamlBackedAntCfg):
    YAML_DEFAULT = "ant_jump.yaml"


# ------------------------- Registration -------------------------

def _register_one(name: str, entry_point_str: str):
    """Register a config-only Gym spec carrying env_cfg_entry_point."""
    try:
        gym.spec(name)
        return  # already registered
    except gym.error.Error:
        pass

    def _stub(**_kw):
        raise RuntimeError(
            f"'{name}' is a config-only task. "
            f"Use parse_env_cfg('{name}', ...) to obtain an EnvCfg, then "
            f"gym.make('Isaac-Ant-Direct-v0', cfg=env_cfg, ...)."
        )

    gym.register(
        id=name,
        entry_point=lambda **kw: _stub(**kw),
        kwargs={"env_cfg_entry_point": entry_point_str},
    )

def register_ant_cfg_tasks():
    specs = [
        ("Ant-Walk-v0", "ant_cfg_registry:AntWalkEnvCfg"),
        ("Ant-Spin-v0", "ant_cfg_registry:AntSpinEnvCfg"),
        ("Ant-Jump-v0", "ant_cfg_registry:AntJumpEnvCfg"),
    ]
    for name, ep in specs:
        _register_one(name, ep)
