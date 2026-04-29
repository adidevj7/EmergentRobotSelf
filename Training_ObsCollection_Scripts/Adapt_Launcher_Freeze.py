#!/usr/bin/env python3
from __future__ import annotations

"""
Adaptation Launcher — Freeze (att71)
--------------------------------------------------------------------
Goal:
  - Launch adaptation runs from *_for_play.pth models in SelfExtractorTriplets/models_all/
  - For each model, auto-discover paired _self_freeze_idx.json and _task_freeze_idx.json
  - Run ONLY on the TWO new behaviors (never re-trains on source behavior)
    e.g. spin model → trains walk and jump only
  - Both freeze types (self + task) per new behavior
  - Seeds per job: [7, 42, 123]
  - Uses Isaac_WSJ_att71_freeze.py (NeuronFreezer via --frozen_indices_json)

JSON discovery:
  {stem}_for_play.pth  →  {stem}_self_freeze_idx.json
                           {stem}_task_freeze_idx.json
  (stem = filename without _for_play.pth)

  
How to run:
  python3 Adapt_Launcher_Freeze.py --gpus 5,6 --slots_per_gpu 2
"""

import os, sys, time, json, argparse, subprocess, shlex, re, hashlib, shutil, glob
from pathlib import Path
from datetime import datetime

# =========================
# 0) MODELS — paste *_for_play.pth paths from SelfExtractorTriplets/models_all/
# =========================
MODELS_DIR = "/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/SelfExtractorTriplets/models_all"

MODELS: list[str] = sorted(
    str(p) for p in Path(MODELS_DIR).glob("*_for_play.pth")
)

# =========================
# 1) DEFAULTS
# =========================
ISAACLAB_DIR = str(Path.home() / "projects" / "IsaacLab")
ISAACLAB_SH  = str(Path.home() / "projects" / "IsaacLab" / "isaaclab.sh")

TRAIN_SCRIPT = "/home/adi/projects/CreativeMachinesAnt/Isaac/scripts/Isaac_WSJ_att71_freeze.py"
CFG_YAML     = "/home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_walk_new_150_relu.yaml"
PLAYER_YAML  = "/home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_play_sac_ant_150_relu.yaml"

TASK       = "Ant-Walk-v0"
GYM_ENV_ID = "Isaac-Ant-Direct-v0"

NUM_ENVS              = 8192
UPDATES_PER_STEP      = 32
OVERRIDE_WARMUP_STEPS = 10000
HEADLESS              = True
LAMBDA_BACK           = 1

# Adaptation budget (same as lesion launcher)
PLATEAU_MIN_STEPS = 500_000_000
MAX_STEPS_PHASE   = 550_000_000
RESTART_MIN_FRAC  = 0.68

# Disable in-training retry/restart logic (launcher handles retries externally)
PHASE_RETRY_MAX              = 0
MIN_PHASE_MEAN_REWARD_ON_SWITCH = 0

LOG_INTERVAL_S = 15
RECORD_EVERY   = 0
VIDEO_GPU      = 6
VIDEO_WAIT_PCT = 50
VIDEO_WAIT_S   = 30

N_CYCLES = 1

SEEDS_PER_JOB = [7, 42, 123]
RETRY_MAX     = 2

ALL_BEHAVIORS = ["walk", "spin", "jump"]

CKPT_ROOT   = "/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/AdaptionTesting/Freeze_att71"
CKPT_PARENT = "/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/"


# =========================
# 2) CLI
# =========================
p = argparse.ArgumentParser("Adaptation launcher — Freeze (att71)")
p.add_argument("--gpus",          type=str, default="0,1,2,3,4,5,6")
p.add_argument("--slots_per_gpu", type=int, default=2)
p.add_argument("--concurrency",   type=int, default=None)
p.add_argument("--dryrun",        action="store_true")
p.add_argument("--name_prefix",   type=str, default="Adapt_Freeze71")
p.add_argument("--seeds",         type=str, default=",".join(map(str, SEEDS_PER_JOB)))
p.add_argument("--retries",       type=int, default=RETRY_MAX)
args_cli = p.parse_args()

GPUS = [int(x) for x in args_cli.gpus.split(",") if x.strip()]
if not GPUS:
    raise SystemExit("No GPUs specified.")

SLOTS_PER_GPU = int(args_cli.slots_per_gpu)
GLOBAL_MAX    = args_cli.concurrency or (SLOTS_PER_GPU * len(GPUS))

try:
    SEEDS_PER_JOB = [int(s.strip()) for s in args_cli.seeds.split(",") if s.strip()]
except Exception as e:
    raise SystemExit(f"Could not parse --seeds: {e}")

RETRY_MAX = int(args_cli.retries)


# =========================
# 3) Helpers
# =========================
def _shquote(s):           return shlex.quote(str(s))
def _hash6(s):             return hashlib.sha1(s.encode()).hexdigest()[:6]
def _safe_slug(s, n=220):
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", s)
    return s[:n]

def _src_behavior_from_path(path: str) -> str | None:
    """Extract walk/spin/jump from the filename."""
    fn = Path(path).name.lower()
    for b in ALL_BEHAVIORS:
        # match _b01_walk_ or _b02_spin_ etc.
        if re.search(rf"_b\d+_{b}_", fn):
            return b
    # fallback: bare behavior token
    for b in ALL_BEHAVIORS:
        if f"_{b}_" in fn:
            return b
    return None

def _new_behaviors(src_behavior: str) -> list[str]:
    """Return the two behaviors that are NOT the source."""
    return [b for b in ALL_BEHAVIORS if b != src_behavior]

def _json_path(for_play_path: str, freeze_type: str) -> Path:
    """
    Derive JSON path from for_play path.
    {stem}_for_play.pth  →  {stem}_self_freeze_idx.json
                             {stem}_task_freeze_idx.json
    """
    p = Path(for_play_path)
    stem = p.name.replace("_for_play.pth", "")
    return p.parent / f"{stem}_{freeze_type}_freeze_idx.json"

def _parse_base_info(path: str) -> dict:
    fn = Path(path).name.replace("_for_play.pth", "")
    m_run = re.match(r"(run\d+)_", fn)
    run_id = m_run.group(1) if m_run else "runXX"
    m_cyc  = re.search(r"(c\d+)_b\d+_", fn)
    src_cycle = m_cyc.group(1) if m_cyc else "cXXX"
    return {
        "run_id":    _safe_slug(run_id),
        "src_cycle": _safe_slug(src_cycle),
        "model_tag": _safe_slug(fn),
    }

def _ckpt_label(prefix, run_id, src_cycle, src_b, adapt_b, freeze_type, seed, base_model):
    h = _hash6(f"{run_id}|{src_cycle}|{src_b}|{adapt_b}|{freeze_type}|{seed}|{base_model}")
    return f"{prefix}__{adapt_b}__{freeze_type}__{h}"

def _pointer_file(ckpt_label, seed):
    return Path(CKPT_PARENT) / f".active_run__{ckpt_label}__seed{seed}.txt"

def _read_pointer(ptr: Path) -> Path | None:
    try:
        if ptr.exists():
            s = ptr.read_text().strip()
            if s and os.path.isdir(s):
                return Path(s).resolve()
    except Exception:
        pass
    return None

def _build_cmd(gpu, cli_args):
    exports = f"CUDA_VISIBLE_DEVICES={gpu}"
    isaac   = _shquote(str(Path(ISAACLAB_SH).resolve()))
    inner   = " ".join(map(_shquote, cli_args))
    return (
        f"conda deactivate >/dev/null 2>&1 || true; "
        f"cd {_shquote(str(Path(ISAACLAB_DIR).resolve()))} && "
        f"{exports} {isaac} -p {inner}"
    )

def _run_bash(cmd, stdout_path):
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w") as out:
        return subprocess.Popen(["bash", "-lc", cmd], stdout=out, stderr=subprocess.STDOUT)

def _print_tail(stdout_path, lines=160):
    try:
        content = stdout_path.read_text().splitlines()
        print("\n----- FAIL LOG TAIL -----")
        print("\n".join(content[-lines:]))
        print("----- END FAIL LOG TAIL -----\n")
    except Exception:
        pass

def _copy_file(src, dst):
    try:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
    except Exception as e:
        print(f"[copy] WARN {src} -> {dst}: {e}")

def _copy_tree(src_dir, dst_dir):
    try:
        if not Path(src_dir).exists(): return
        Path(dst_dir).mkdir(parents=True, exist_ok=True)
        for f in glob.glob(str(Path(src_dir) / "**" / "*"), recursive=True):
            if Path(f).is_file():
                rel = Path(f).relative_to(src_dir)
                _copy_file(f, Path(dst_dir) / rel)
    except Exception as e:
        print(f"[copy] WARN tree {src_dir}: {e}")

def _postprocess(run_dir, ckpt_label, seed):
    ptr = _pointer_file(ckpt_label, seed)
    root = _read_pointer(ptr)
    if root is None:
        print(f"[post] WARN no pointer root for {ckpt_label} seed={seed}")
        return
    for name in ("rollout_log.csv", "phase_state.json"):
        src = root / name
        if src.exists(): _copy_file(src, Path(run_dir) / name)
    # latest checkpoints
    models_src = root / "models"
    if models_src.exists():
        dst = Path(run_dir) / "models_copied"
        dst.mkdir(parents=True, exist_ok=True)
        for f in sorted(models_src.glob("*.pth"), key=lambda x: x.stat().st_mtime, reverse=True)[:6]:
            _copy_file(f, dst / f.name)
    _copy_tree(root / "graphs", Path(run_dir) / "graphs_copied")
    _copy_tree(root / "videos", Path(run_dir) / "videos_copied")


# =========================
# 4) Build JOBS
# =========================
if not MODELS:
    raise SystemExit("MODELS list is empty — paste your _for_play.pth paths at the top.")

ckpt_root = Path(CKPT_ROOT).resolve()
ckpt_root.mkdir(parents=True, exist_ok=True)

stamp      = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
index_csv  = ckpt_root / f"index_{stamp}.csv"
stdout_dir = ckpt_root / f"_stdout_{stamp}"
stdout_dir.mkdir(parents=True, exist_ok=True)

JOBS = []
skipped_no_json   = 0
skipped_no_behav  = 0

for base_model in MODELS:
    base_model = str(Path(base_model).resolve())
    if not os.path.isfile(base_model):
        print(f"[warn] missing on disk: {base_model}")
        continue

    src_b = _src_behavior_from_path(base_model)
    if src_b is None:
        print(f"[warn] cannot parse source behavior from: {Path(base_model).name}")
        skipped_no_behav += 1
        continue

    new_behaviors = _new_behaviors(src_b)
    info = _parse_base_info(base_model)

    for freeze_type in ("self", "task"):
        json_path = _json_path(base_model, freeze_type)
        if not json_path.exists():
            print(f"[warn] JSON not found ({freeze_type}): {json_path.name}")
            skipped_no_json += 1
            continue

        for adapt_b in new_behaviors:
            for seed in SEEDS_PER_JOB:
                lbl = _ckpt_label(
                    prefix=args_cli.name_prefix,
                    run_id=info["run_id"],
                    src_cycle=info["src_cycle"],
                    src_b=src_b,
                    adapt_b=adapt_b,
                    freeze_type=freeze_type,
                    seed=seed,
                    base_model=base_model,
                )

                run_name = _safe_slug(
                    f"{args_cli.name_prefix}__{info['run_id']}__{info['src_cycle']}"
                    f"_{src_b}__{freeze_type}__to_{adapt_b}__seed{seed}"
                )

                run_dir    = ckpt_root / run_name
                stdout_log = stdout_dir / f"{run_name}.log"
                run_dir.mkdir(parents=True, exist_ok=True)

                cli = [
                    TRAIN_SCRIPT,
                    "--task",                          TASK,
                    "--gym_env_id",                    GYM_ENV_ID,
                    "--cfg_yaml",                      CFG_YAML,
                    "--player_yaml",                   PLAYER_YAML,
                    "--num_envs",                      str(NUM_ENVS),
                    "--n_cycles",                      str(N_CYCLES),
                    "--phase_order",                   adapt_b,
                    "--updates_per_step",              str(UPDATES_PER_STEP),
                    "--plateau_min_steps",             str(PLATEAU_MIN_STEPS),
                    "--max_steps_phase",               str(MAX_STEPS_PHASE),
                    "--restart_min_frac",              str(RESTART_MIN_FRAC),
                    "--override_warmup_steps",         str(OVERRIDE_WARMUP_STEPS),
                    "--phase_retry_max",               str(PHASE_RETRY_MAX),
                    "--min_phase_mean_reward_on_switch", str(MIN_PHASE_MEAN_REWARD_ON_SWITCH),
                    "--log_interval_s",                str(LOG_INTERVAL_S),
                    "--lambda_back",                   str(LAMBDA_BACK),
                    "--gpu",                           str(-1),  # filled in at launch
                    "--record_every",                  str(RECORD_EVERY),
                    "--video_gpu",                     str(VIDEO_GPU),
                    "--video_wait_pct",                str(VIDEO_WAIT_PCT),
                    "--video_wait_s",                  str(VIDEO_WAIT_S),
                    "--run_tag",                       f"{args_cli.name_prefix}_{freeze_type}_to_{adapt_b}",
                    "--ckpt_label",                    lbl,
                    "--seed",                          str(seed),
                    "--resume_from",                   base_model,
                    "--frozen_indices_json",           str(json_path),
                ]
                if HEADLESS:
                    cli.append("--headless")

                JOBS.append({
                    "run_name":     run_name,
                    "adapt_b":      adapt_b,
                    "freeze_type":  freeze_type,
                    "src_behavior": src_b,
                    "seed":         seed,
                    "gpu":          None,
                    "base_model":   base_model,
                    "json_path":    str(json_path),
                    "ckpt_label":   lbl,
                    "stdout_log":   str(stdout_log),
                    "run_dir":      str(run_dir),
                    "cli_template": cli,
                    "attempt":      0,
                })

print(f"[freeze] Models processed : {len(MODELS)}")
print(f"[freeze] Skipped (no behavior parse) : {skipped_no_behav}")
print(f"[freeze] Skipped (JSON missing)      : {skipped_no_json}")
print(f"[freeze] Jobs built : {len(JOBS)}")
print(f"           = {len(MODELS)} models"
      f" × 2 freeze types"
      f" × 2 new behaviors"
      f" × {len(SEEDS_PER_JOB)} seeds"
      f"  (minus skipped)")
print(f"[freeze] Output root : {ckpt_root}")
print(f"[freeze] GPUs={GPUS}  slots_per_gpu={SLOTS_PER_GPU}  global_max={GLOBAL_MAX}")

with index_csv.open("w") as f:
    f.write("run_name,adapt_b,freeze_type,src_behavior,seed,gpu,base_model,json_path,ckpt_label,run_dir,stdout_log,start_ts,attempt\n")

if args_cli.dryrun:
    for j in JOBS[:16]:
        print(f"  {j['run_name']}")
        print(f"    src={Path(j['base_model']).name}")
        print(f"    freeze={j['freeze_type']}  →  {j['adapt_b']}  seed={j['seed']}")
        print(f"    json={Path(j['json_path']).name}")
    if len(JOBS) > 16:
        print(f"  ... (+{len(JOBS)-16} more)")
    sys.exit(0)


# =========================
# 5) Scheduler
# =========================
gpu_slots = {g: 0 for g in GPUS}
procs: dict[str, dict] = {}
queue = JOBS.copy()

def _pick_gpu():
    for g in GPUS:
        if gpu_slots[g] < SLOTS_PER_GPU:
            return g
    return None

while queue or procs:
    # Launch ready jobs
    while queue and len(procs) < GLOBAL_MAX:
        g = _pick_gpu()
        if g is None: break

        job = queue.pop(0)
        job["gpu"] = g

        cli = list(job["cli_template"])
        try:
            k = cli.index("--gpu"); cli[k + 1] = str(g)
        except Exception:
            pass

        cmd = _build_cmd(g, cli)
        proc = _run_bash(cmd, Path(job["stdout_log"]))
        gpu_slots[g] += 1
        procs[job["run_name"]] = {"p": proc, "gpu": g, "job": job}

        # append to index
        with index_csv.open("a") as f:
            f.write(",".join([
                job["run_name"], job["adapt_b"], job["freeze_type"],
                job["src_behavior"], str(job["seed"]), str(g),
                _shquote(job["base_model"]), _shquote(job["json_path"]),
                job["ckpt_label"], _shquote(job["run_dir"]),
                _shquote(job["stdout_log"]), str(int(time.time())),
                str(job["attempt"]),
            ]) + "\n")

        print(f"[launch] {job['run_name']}  GPU={g}  "
              f"freeze={job['freeze_type']}→{job['adapt_b']}  seed={job['seed']}  "
              f"attempt={job['attempt']}")

    # Poll for finished
    done_names = []
    for name, rec in list(procs.items()):
        ret = rec["p"].poll()
        if ret is None: continue

        job = rec["job"]
        g   = rec["gpu"]
        print(f"[done] {name}  GPU={g}  exit={ret}")

        if ret != 0:
            _print_tail(Path(job["stdout_log"]))

        _postprocess(job["run_dir"], job["ckpt_label"], int(job["seed"]))

        if ret != 0 and job["attempt"] < RETRY_MAX:
            job2 = dict(job)
            job2["attempt"] += 1
            print(f"[retry] {name}  attempt={job2['attempt']}")
            queue.append(job2)

        gpu_slots[g] = max(0, gpu_slots[g] - 1)
        done_names.append(name)

        # Collect rollout CSVs for this completed job
        rollouts_root = ckpt_root / "_rollouts"
        rollouts_root.mkdir(parents=True, exist_ok=True)
        rollout_src = Path(job["run_dir"]) / "rollout_log.csv"
        if rollout_src.exists() and ret == 0:
            _copy_file(rollout_src, rollouts_root / f"{name}__rollout_log.csv")

    for name in done_names:
        procs.pop(name, None)

    time.sleep(2.0)

print(f"\n[freeze] All runs finished.")
print(f"[freeze] Outputs : {ckpt_root}")
print(f"[freeze] Index   : {index_csv}")
print(f"[freeze] Rollouts: {ckpt_root / '_rollouts'}")