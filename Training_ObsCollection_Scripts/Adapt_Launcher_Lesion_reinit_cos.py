#!/usr/bin/env python3
from __future__ import annotations

"""
Adaptation Launcher — Baseline0 (NO training-script edits)
--------------------------------------------------------------------
Goal:
  - Launch adaptation runs from many *_for_play.pth base models
  - Run ONLY 1 behavior per run (walk-only / spin-only / jump-only)
  - Tries per model = 1 (i.e., 1 run per (base_model, behavior))
  - Retries = 1 (if a run exits nonzero, retry once)
  - DO NOT rely on training script writing into our desired folder.
    Instead, after each run finishes, we copy artifacts into:
      CKPT_ROOT/<RUN_NAME>/

  - We copy rollout_log.csv into:
      CKPT_ROOT/_rollouts/<RUN_NAME>__rollout_log.csv
    BUT ONLY if at least one of the 3 adaptation runs for that base_model succeeds.
    (If all three adaptation runs fail, we do NOT copy any rollouts to _rollouts.)

Critical behavior:
  - We intentionally make ckpt_label UNIQUE per job so the training script's
    pointer-run directory is unique and won't collide / instantly "[DONE]".
  - After job finishes, we read the pointer file created by the training script
    to find where it actually wrote outputs, then copy:
      - rollout_log.csv (most important)
      - phase_state.json (if present)
      - latest model checkpoints (optional but useful)
      - graphs/videos folders (optional)

How to run (from anywhere):
  python3 /home/adi/projects/CreativeMachinesAnt/Isaac/scripts/Adapt_Launcher_Lesion_reinit_cos.py --gpus 0,1,2,3,4,5,6 --slots_per_gpu 2

Note: We cd into projects/IsaacLab inside each launched process (per your request).
"""

import os, sys, time, json, argparse, subprocess, shlex, re, hashlib, shutil, glob
from pathlib import Path
from datetime import datetime

# =========================
# 0) PASTE MODELS HERE
# =========================
MODELS: list[str] = [
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run00_c003_b01_walk_original_plateau_2026-01-31_08-57-51_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run00_c003_b01_walk_selflesion_plateau_2026-01-31_08-57-51_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run00_c003_b01_walk_tasklesion_plateau_2026-01-31_08-57-51_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run00_c003_b02_spin_original_plateau_2026-01-31_12-44-10_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run00_c003_b02_spin_selflesion_plateau_2026-01-31_12-44-10_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run00_c003_b02_spin_tasklesion_plateau_2026-01-31_12-44-10_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run00_c003_b03_jump_original_plateau_2026-01-31_13-49-10_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run00_c003_b03_jump_selflesion_plateau_2026-01-31_13-49-10_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run00_c003_b03_jump_tasklesion_plateau_2026-01-31_13-49-10_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run01_c018_b01_walk_original_plateau_2026-02-03_21-26-35_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run01_c018_b01_walk_selflesion_plateau_2026-02-03_21-26-35_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run01_c018_b01_walk_tasklesion_plateau_2026-02-03_21-26-35_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run01_c018_b02_spin_original_plateau_2026-02-03_23-20-43_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run01_c018_b02_spin_selflesion_plateau_2026-02-03_23-20-43_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run01_c018_b02_spin_tasklesion_plateau_2026-02-03_23-20-43_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run01_c018_b03_jump_original_plateau_2026-02-04_00-45-09_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run01_c018_b03_jump_selflesion_plateau_2026-02-04_00-45-09_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run01_c018_b03_jump_tasklesion_plateau_2026-02-04_00-45-09_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run02_c022_b01_walk_original_plateau_2026-02-04_11-56-55_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run02_c022_b01_walk_selflesion_plateau_2026-02-04_11-56-55_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run02_c022_b01_walk_tasklesion_plateau_2026-02-04_11-56-55_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run02_c022_b02_spin_original_plateau_2026-02-04_14-11-09_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run02_c022_b02_spin_selflesion_plateau_2026-02-04_14-11-09_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run02_c022_b02_spin_tasklesion_plateau_2026-02-04_14-11-09_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run02_c022_b03_jump_original_plateau_2026-02-04_15-21-06_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run02_c022_b03_jump_selflesion_plateau_2026-02-04_15-21-06_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run02_c022_b03_jump_tasklesion_plateau_2026-02-04_15-21-06_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run03_c039_b01_walk_original_plateau_2026-02-09_18-18-50_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run03_c039_b01_walk_selflesion_plateau_2026-02-09_18-18-50_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run03_c039_b01_walk_tasklesion_plateau_2026-02-09_18-18-50_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run03_c039_b02_spin_original_plateau_2026-02-09_19-39-55_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run03_c039_b02_spin_selflesion_plateau_2026-02-09_19-39-55_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run03_c039_b02_spin_tasklesion_plateau_2026-02-09_19-39-55_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run03_c039_b03_jump_original_plateau_2026-02-09_20-45-50_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run03_c039_b03_jump_selflesion_plateau_2026-02-09_20-45-50_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run03_c039_b03_jump_tasklesion_plateau_2026-02-09_20-45-50_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run04_c027_b01_walk_original_plateau_2026-02-05_16-21-20_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run04_c027_b01_walk_selflesion_plateau_2026-02-05_16-21-20_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run04_c027_b01_walk_tasklesion_plateau_2026-02-05_16-21-20_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run04_c027_b02_spin_original_plateau_2026-02-05_18-02-28_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run04_c027_b02_spin_selflesion_plateau_2026-02-05_18-02-28_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run04_c027_b02_spin_tasklesion_plateau_2026-02-05_18-02-28_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run04_c027_b03_jump_original_plateau_2026-02-05_19-05-35_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run04_c027_b03_jump_selflesion_plateau_2026-02-05_19-05-35_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run04_c027_b03_jump_tasklesion_plateau_2026-02-05_19-05-35_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run06_c012_b01_walk_original_plateau_2026-02-02_06-01-34_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run06_c012_b01_walk_selflesion_plateau_2026-02-02_06-01-34_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run06_c012_b01_walk_tasklesion_plateau_2026-02-02_06-01-34_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run06_c012_b02_spin_original_plateau_2026-02-02_09-26-08_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run06_c012_b02_spin_selflesion_plateau_2026-02-02_09-26-08_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run06_c012_b02_spin_tasklesion_plateau_2026-02-02_09-26-08_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run06_c012_b03_jump_original_plateau_2026-02-02_11-06-53_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run06_c012_b03_jump_selflesion_plateau_2026-02-02_11-06-53_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run06_c012_b03_jump_tasklesion_plateau_2026-02-02_11-06-53_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run07_c005_b01_walk_original_plateau_2026-01-31_18-09-31_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run07_c005_b01_walk_selflesion_plateau_2026-01-31_18-09-31_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run07_c005_b01_walk_tasklesion_plateau_2026-01-31_18-09-31_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run07_c005_b02_spin_original_plateau_2026-01-31_20-33-37_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run07_c005_b02_spin_selflesion_plateau_2026-01-31_20-33-37_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run07_c005_b02_spin_tasklesion_plateau_2026-01-31_20-33-37_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run07_c005_b03_jump_original_plateau_2026-01-31_21-41-07_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run07_c005_b03_jump_selflesion_plateau_2026-01-31_21-41-07_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run07_c005_b03_jump_tasklesion_plateau_2026-01-31_21-41-07_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run07_c036_b01_walk_original_plateau_2026-02-09_09-47-50_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run07_c036_b01_walk_selflesion_plateau_2026-02-09_09-47-50_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run07_c036_b01_walk_tasklesion_plateau_2026-02-09_09-47-50_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run07_c036_b02_spin_original_plateau_2026-02-09_11-36-53_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run07_c036_b02_spin_selflesion_plateau_2026-02-09_11-36-53_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run07_c036_b02_spin_tasklesion_plateau_2026-02-09_11-36-53_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run07_c036_b03_jump_original_plateau_2026-02-09_12-40-27_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run07_c036_b03_jump_selflesion_plateau_2026-02-09_12-40-27_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run07_c036_b03_jump_tasklesion_plateau_2026-02-09_12-40-27_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run09_c031_b01_walk_original_plateau_2026-02-08_05-12-56_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run09_c031_b01_walk_selflesion_plateau_2026-02-08_05-12-56_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run09_c031_b01_walk_tasklesion_plateau_2026-02-08_05-12-56_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run09_c031_b02_spin_original_plateau_2026-02-08_07-08-48_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run09_c031_b02_spin_selflesion_plateau_2026-02-08_07-08-48_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run09_c031_b02_spin_tasklesion_plateau_2026-02-08_07-08-48_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run09_c031_b03_jump_original_plateau_2026-02-08_07-43-53_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run09_c031_b03_jump_selflesion_plateau_2026-02-08_07-43-53_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run09_c031_b03_jump_tasklesion_plateau_2026-02-08_07-43-53_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run09_c049_b01_walk_original_plateau_2026-02-12_15-16-13_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run09_c049_b01_walk_selflesion_plateau_2026-02-12_15-16-13_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run09_c049_b01_walk_tasklesion_plateau_2026-02-12_15-16-13_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run09_c049_b02_spin_original_plateau_2026-02-12_18-06-36_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run09_c049_b02_spin_selflesion_plateau_2026-02-12_18-06-36_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run09_c049_b02_spin_tasklesion_plateau_2026-02-12_18-06-36_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run09_c049_b03_jump_original_plateau_2026-02-12_21-10-51_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run09_c049_b03_jump_selflesion_plateau_2026-02-12_21-10-51_for_play.pth",
"/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_cos/models_all/run09_c049_b03_jump_tasklesion_plateau_2026-02-12_21-10-51_for_play.pth"
]


# =========================
# 1) DEFAULTS (match robustness unless overridden)
# =========================
ISAACLAB_DIR = str(Path.home() / "projects" / "IsaacLab")
ISAACLAB_SH  = str(Path.home() / "projects" / "IsaacLab" / "isaaclab.sh")

TRAIN_SCRIPT = "/home/adi/projects/CreativeMachinesAnt/Isaac/scripts/Isaac_WSJ_att69_cleanup.py"
CFG_YAML     = "/home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_walk_new_150_relu.yaml"
PLAYER_YAML  = "/home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_play_sac_ant_150_relu.yaml"

TASK         = "Ant-Walk-v0"
GYM_ENV_ID   = "Isaac-Ant-Direct-v0"

NUM_ENVS              = 8192
UPDATES_PER_STEP      = 32
OVERRIDE_WARMUP_STEPS = 10000
HEADLESS              = True
LAMBDA_BACK           = 1

# adaptation budget (your request)
PLATEAU_MIN_STEPS = 700_000_000
MAX_STEPS_PHASE   = 750_000_000

# per-run behavior schedule
BEHAVIORS = ["walk", "spin", "jump"]
N_CYCLES  = 1

# tries per model = 1 (one run per (base_model, behavior))
REPEATS_PER_BEHAVIOR = 1

# retries per job (if exit nonzero)
RETRY_MAX = 1

# logging/video (match robustness except log_interval_s=10)
LOG_INTERVAL_S = 10
RECORD_EVERY   = 0
VIDEO_GPU      = 6
VIDEO_WAIT_PCT = 50
VIDEO_WAIT_S   = 30

# outputs root (your desired organized folder)
CKPT_ROOT = "/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/AdaptionTesting/Baseline_Lesions_Reinit_cos"

# Where the training script writes pointer files
CKPT_PARENT = "/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints"


# =========================
# 2) CLI
# =========================
p = argparse.ArgumentParser("Adaptation launcher — Baseline0 (copy outputs into organized folders)")
p.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6",
               help="Comma-separated physical GPU ids to use, e.g. '0,1,2,3,4,5,6'.")
p.add_argument("--slots_per_gpu", type=int, default=2,
               help="Max concurrent runs per GPU (enforced by scheduler).")
p.add_argument("--concurrency", type=int, default=None,
               help="Optional global max concurrent runs. Default = slots_per_gpu * num_gpus.")
p.add_argument("--dryrun", action="store_true", help="Print planned runs and exit without launching.")
p.add_argument("--name_prefix", type=str, default="Adapt_Baseline0",
               help="Prefix for run names.")
args = p.parse_args()

GPUS = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
if len(GPUS) == 0:
    raise SystemExit("No GPUs specified. Use --gpus e.g. --gpus 0,1,2,3,4,5,6")

SLOTS_PER_GPU = int(args.slots_per_gpu)
if SLOTS_PER_GPU < 1:
    raise SystemExit("--slots_per_gpu must be >= 1")

GLOBAL_MAX = args.concurrency if args.concurrency is not None else (SLOTS_PER_GPU * len(GPUS))


# =========================
# 3) Helpers
# =========================
def _shquote(s: str) -> str:
    return shlex.quote(str(s))

def _safe_slug(s: str) -> str:
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", s)
    return s

def _hash6(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:6]

# =========================
# EDIT 1/2: update _parse_base_info() to extract run_id + lesion tag from filename
# =========================
def _parse_base_info(base_model_path: str) -> dict:
    """
    base_model_path example:
      .../models_all/run01_c018_b01_walk_selflesion_plateau_2026-02-03_21-26-35_for_play.pth

    We want:
      - run_id = run01
      - lesion = original/selflesion/tasklesion
      - model_tag = run01_c018_b01_walk_selflesion_plateau_2026-02-03_21-26-35
      - src_cycle = c018
      - src_behavior = walk/spin/jump (from b01_walk / b02_spin / b03_jump)
      - robustness_run = (keep old behavior if under checkpoints/, else fall back to run_id)
    """
    p = Path(base_model_path)
    parts = p.parts

    fn = p.name
    fn = fn.replace("_for_play.pth", "").replace(".pth", "")
    model_tag = fn

    m_run = re.match(r"(run\d+)_", fn)
    run_id = m_run.group(1) if m_run else "runXX"

    m_les = re.search(r"_(original|selflesion|tasklesion)_", fn)
    lesion = m_les.group(1) if m_les else "unklesion"

    robustness_run = run_id  # fallback if not under checkpoints/
    try:
        i = parts.index("checkpoints")
        robustness_run = parts[i + 1]
    except Exception:
        pass

    m = re.search(r"(c\d+)_b\d+_(walk|spin|jump)_", fn)
    src_cycle = m.group(1) if m else "cXXX"
    src_behavior = m.group(2) if m else "unk"

    return {
        "robustness_run": _safe_slug(robustness_run),
        "run_id": _safe_slug(run_id),
        "lesion": _safe_slug(lesion),
        "model_tag": _safe_slug(model_tag),
        "src_cycle": _safe_slug(src_cycle),
        "src_behavior": _safe_slug(src_behavior),
    }

def _job_seed(base_model: str, adapt_behavior: str) -> int:
    # Unique/stable seed per (model, behavior) to avoid pointer collisions.
    # Still "tries per model = 1" because we run only one seed for each job.
    h = int(hashlib.sha1((base_model + "|" + adapt_behavior).encode("utf-8")).hexdigest()[:8], 16)
    return 1000 + (h % 900_000_000)

def _unique_ckpt_label(prefix: str, robust_run: str, src_cycle: str, src_b: str, adapt_b: str, seed: int) -> str:
    # MUST be unique per job so the training script's pointer-run dir is unique.
    # Keep it reasonably short to avoid filesystem/pointer filename issues.
    short = _hash6(f"{robust_run}|{src_cycle}|{src_b}|{adapt_b}|{seed}")
    return f"{prefix}__{adapt_b}__{short}"

def _pointer_file_path(ckpt_label: str, seed: int) -> Path:
    # training script uses: .active_run__{ckpt_label}__seed{seed}.txt
    return Path(CKPT_PARENT) / f".active_run__{ckpt_label}__seed{seed}.txt"

def _read_pointer_run_root(ptr: Path) -> Path | None:
    try:
        if not ptr.exists():
            return None
        s = ptr.read_text().strip()
        if s and os.path.isdir(s):
            return Path(s).resolve()
    except Exception:
        return None
    return None

def _build_cmd_str(gpu: int, cli_args: list[str]) -> str:
    exports = f"CUDA_VISIBLE_DEVICES={gpu}"
    isaac = _shquote(str(Path(ISAACLAB_SH).resolve()))
    inner = " ".join(map(_shquote, cli_args))
    return (
        f"conda deactivate >/dev/null 2>&1 || true; "
        f"cd {_shquote(str(Path(ISAACLAB_DIR).resolve()))} && "
        f"{exports} {isaac} -p {inner}"
    )

def _run_bash(cmd_str: str, stdout_path: Path) -> subprocess.Popen:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w") as out:
        p = subprocess.Popen(["bash", "-lc", cmd_str], stdout=out, stderr=subprocess.STDOUT)
    return p

def _print_fail_tail(stdout_path: Path, lines: int = 160):
    try:
        with stdout_path.open("r") as f:
            content = f.readlines()
        tail = "".join(content[-lines:])
        print("\n----- FAIL LOG TAIL -----")
        print(tail)
        print("----- END FAIL LOG TAIL -----\n")
    except Exception as e:
        print(f"[warn] could not read fail log {stdout_path}: {e}")

def _copy_file(src: Path, dst: Path):
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
    except Exception as e:
        print(f"[copy] WARN failed to copy {src} -> {dst}: {e}")

def _copy_tree(src_dir: Path, dst_dir: Path, patterns: list[str] | None = None):
    """
    Copy selected files from src_dir into dst_dir.
    If patterns is None: copy everything (shallow via copytree not used; we do file-by-file).
    """
    try:
        if not src_dir.exists():
            return
        dst_dir.mkdir(parents=True, exist_ok=True)

        if patterns is None:
            files = [Path(p) for p in glob.glob(str(src_dir / "**" / "*"), recursive=True)]
            files = [p for p in files if p.is_file()]
        else:
            files = []
            for pat in patterns:
                files.extend([Path(p) for p in glob.glob(str(src_dir / pat), recursive=True)])
            files = [p for p in files if p.is_file()]

        for f in files:
            rel = f.relative_to(src_dir)
            _copy_file(f, dst_dir / rel)
    except Exception as e:
        print(f"[copy] WARN tree copy failed {src_dir} -> {dst_dir}: {e}")

def _postprocess_copy_outputs(run_dir: Path, run_name: str, ckpt_label: str, seed: int):
    """
    After a run finishes, find actual training outputs via pointer file and copy:
      - rollout_log.csv into run_dir
      - phase_state.json into run_dir
      - models latest checkpoints into run_dir/models_copied/
      - graphs/videos (optional) into run_dir/graphs_copied, run_dir/videos_copied

    NOTE: Global rollout collection into CKPT_ROOT/_rollouts/ is handled by the scheduler
          AFTER we know whether this base_model had at least one successful adaptation run.
    """
    ptr = _pointer_file_path(ckpt_label, seed)
    actual_root = _read_pointer_run_root(ptr)

    if actual_root is None:
        print(f"[post] WARN no pointer run root found for ckpt_label={ckpt_label} seed={seed} (ptr={ptr})")
        return

    # 1) rollout_log.csv (MOST IMPORTANT)
    rollout_src = actual_root / "rollout_log.csv"
    if rollout_src.exists():
        rollout_dst = run_dir / "rollout_log.csv"
        _copy_file(rollout_src, rollout_dst)
    else:
        print(f"[post] WARN rollout_log.csv not found in actual_root={actual_root}")

    # 2) phase_state.json (helps confirm completion)
    ps = actual_root / "phase_state.json"
    if ps.exists():
        _copy_file(ps, run_dir / "phase_state.json")

    # 3) copy some outputs (optional but helpful)
    #    Keep them in *_copied folders to make it explicit these are copies.
    models_src = actual_root / "models"
    graphs_src = actual_root / "graphs"
    videos_src = actual_root / "videos"

    # copy latest few checkpoints (by mtime) with renaming into run_dir/models_copied/
    try:
        if models_src.exists():
            dst_models = run_dir / "models_copied"
            dst_models.mkdir(parents=True, exist_ok=True)
            cands = sorted(models_src.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
            # copy up to 6 most recent .pth / for_play
            for f in cands[:6]:
                new_name = f"{run_name}__{f.name}"
                _copy_file(f, dst_models / new_name)
    except Exception as e:
        print(f"[post] WARN copying models failed: {e}")

    # shallow copy graphs/videos entirely (typically small)
    _copy_tree(graphs_src, run_dir / "graphs_copied", patterns=None)
    _copy_tree(videos_src, run_dir / "videos_copied", patterns=None)


# =========================
# 4) Build JOBS
# =========================
if len(MODELS) == 0:
    raise SystemExit("MODELS list is empty. Paste your *_for_play.pth paths into MODELS near the top.")

ckpt_root = Path(CKPT_ROOT).resolve()
ckpt_root.mkdir(parents=True, exist_ok=True)

launcher_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
index_csv = ckpt_root / f"index_{launcher_stamp}.csv"
stdout_dir = ckpt_root / f"_stdout_{launcher_stamp}"
manifests_dir = ckpt_root / f"_manifests_{launcher_stamp}"
stdout_dir.mkdir(parents=True, exist_ok=True)
manifests_dir.mkdir(parents=True, exist_ok=True)

with index_csv.open("w") as f:
    f.write(
        "run_name,adapt_behavior,seed,gpu,robustness_run,src_cycle,src_behavior,base_model,ckpt_label,stdout_log,manifest,run_dir,start_ts,attempt\n"
    )

JOBS = []
for base_model in MODELS:
    base_model = str(Path(base_model).resolve())
    if not os.path.isfile(base_model):
        print(f"[warn] base model missing on disk: {base_model}")
        continue

    info = _parse_base_info(base_model)
    robust_run = info["robustness_run"]
    src_cycle  = info["src_cycle"]
    src_b      = info["src_behavior"]
    model_tag  = info["model_tag"]

    for adapt_b in BEHAVIORS:
        # tries per model = 1 => exactly one seed per (model, adapt_behavior)
        seed = _job_seed(base_model, adapt_b)

        run_id  = info["run_id"]
        lesion  = info["lesion"]
        taskstr = f"{src_b}_to_{adapt_b}"  # e.g., walk_to_walk, spin_to_walk, etc.

        run_name = f"{args.name_prefix}__{run_id}__{src_cycle}_{src_b}__{lesion}__{taskstr}__seed{seed}"
        run_name = _safe_slug(run_name)[:220]

        run_dir = ckpt_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        stdout_log = stdout_dir / f"{run_name}.log"
        manifest = manifests_dir / f"{run_name}.json"

        # Unique ckpt_label so training script writes into unique pointer-run folder
        ckpt_label = _unique_ckpt_label(
            prefix="AdaptionTesting_Baseline0",
            robust_run=robust_run,
            src_cycle=src_cycle,
            src_b=src_b,
            adapt_b=adapt_b,
            seed=seed,
        )

        # training script CLI args (match robustness runs, adaptation changes only)
        cli = [
            TRAIN_SCRIPT,
            "--task", TASK,
            "--gym_env_id", GYM_ENV_ID,
            "--cfg_yaml", CFG_YAML,
            "--player_yaml", PLAYER_YAML,
            "--num_envs", str(NUM_ENVS),
            "--n_cycles", str(N_CYCLES),
            "--phase_order", adapt_b,
            "--updates_per_step", str(UPDATES_PER_STEP),
            "--plateau_min_steps", str(PLATEAU_MIN_STEPS),
            "--max_steps_phase", str(MAX_STEPS_PHASE),
            "--override_warmup_steps", str(OVERRIDE_WARMUP_STEPS),
            "--log_interval_s", str(LOG_INTERVAL_S),
            "--lambda_back", str(LAMBDA_BACK),
            "--gpu", str(-1),  # filled at launch
            "--record_every", str(RECORD_EVERY),
            "--video_gpu", str(VIDEO_GPU),
            "--video_wait_pct", str(VIDEO_WAIT_PCT),
            "--video_wait_s", str(VIDEO_WAIT_S),
            "--run_tag", f"{args.name_prefix}__to_{adapt_b}",
            "--ckpt_label", ckpt_label,
            "--seed", str(seed),
            "--resume_from", base_model,
        ]
        if HEADLESS:
            cli.append("--headless")

        JOBS.append({
            "run_name": run_name,
            "adapt_behavior": adapt_b,
            "seed": seed,
            "gpu": None,
            "robustness_run": robust_run,
            "src_cycle": src_cycle,
            "src_behavior": src_b,
            "base_model": base_model,
            "ckpt_label": ckpt_label,
            "stdout_log": str(stdout_log),
            "manifest": str(manifest),
            "run_dir": str(run_dir),
            "cli_template": cli,
            "attempt": 0,
        })

if len(JOBS) == 0:
    raise SystemExit("No jobs built (MODELS missing on disk?).")

print(f"[adapt] planned jobs: {len(JOBS)}  (tries/model=1, retries={RETRY_MAX})")
print(f"[adapt] outputs root: {ckpt_root}")
print(f"[adapt] index csv: {index_csv}")
print(f"[adapt] stdout dir: {stdout_dir}")
print(f"[adapt] manifests dir: {manifests_dir}")
print(f"[adapt] GPUs={GPUS} slots_per_gpu={SLOTS_PER_GPU} global_max={GLOBAL_MAX}")

if args.dryrun:
    for j in JOBS[:12]:
        print(f"  - {j['run_name']}  base={Path(j['base_model']).name}  to={j['adapt_behavior']}  seed={j['seed']}")
    if len(JOBS) > 12:
        print(f"  ... (+{len(JOBS)-12} more)")
    sys.exit(0)


# =========================
# EDIT 2/2: group-level rollout collection control
#   - Only copy to CKPT_ROOT/_rollouts if base_model had >=1 SUCCESS across its 3 adaptation runs.
# =========================
group_state: dict[str, dict] = {}
for j in JOBS:
    bm = j["base_model"]
    if bm not in group_state:
        group_state[bm] = {
            "expected": set(),
            "finished": set(),
            "success": set(),
            "run_dirs": {},
            "collected": False,
        }
    group_state[bm]["expected"].add(j["run_name"])
    group_state[bm]["run_dirs"][j["run_name"]] = j["run_dir"]


# =========================
# 5) Scheduler (slots_per_gpu enforced)
# =========================
gpu_slots_in_use = {g: 0 for g in GPUS}
procs: dict[str, dict] = {}  # run_name -> {"p": Popen, "gpu": int, "job": dict}

queue = JOBS.copy()

def _pick_gpu() -> int | None:
    for g in GPUS:
        if gpu_slots_in_use[g] < SLOTS_PER_GPU:
            return g
    return None

def _append_index(job: dict, gpu: int):
    with index_csv.open("a") as f:
        f.write(",".join([
            job["run_name"],
            job["adapt_behavior"],
            str(job["seed"]),
            str(gpu),
            job["robustness_run"],
            job["src_cycle"],
            job["src_behavior"],
            _shquote(job["base_model"]),
            job["ckpt_label"],
            _shquote(job["stdout_log"]),
            _shquote(job["manifest"]),
            _shquote(job["run_dir"]),
            str(int(time.time())),
            str(job["attempt"]),
        ]) + "\n")

while queue or procs:
    # launch while capacity exists
    while queue and len(procs) < GLOBAL_MAX:
        g = _pick_gpu()
        if g is None:
            break

        job = queue.pop(0)
        job["gpu"] = g

        # fill gpu in cli (both --gpu and CUDA_VISIBLE_DEVICES)
        cli = list(job["cli_template"])
        try:
            k = cli.index("--gpu")
            cli[k + 1] = str(g)
        except Exception:
            pass

        cmd_str = _build_cmd_str(gpu=g, cli_args=cli)

        # write manifest
        manifest_path = Path(job["manifest"])
        with manifest_path.open("w") as f:
            json.dump({
                "run_name": job["run_name"],
                "adapt_behavior": job["adapt_behavior"],
                "seed": job["seed"],
                "gpu": g,
                "robustness_run": job["robustness_run"],
                "src_cycle": job["src_cycle"],
                "src_behavior": job["src_behavior"],
                "base_model": job["base_model"],
                "ckpt_label": job["ckpt_label"],
                "cmd_str": cmd_str,
                "cli_args": cli,
                "launcher_ts": launcher_stamp,
                "attempt": job["attempt"],
                "run_dir": job["run_dir"],
                "stdout_log": job["stdout_log"],
            }, f, indent=2)

        _append_index(job, g)

        # launch
        print(f"[launch] {job['run_name']} on GPU {g} (slot {gpu_slots_in_use[g]+1}/{SLOTS_PER_GPU}) attempt={job['attempt']}")
        p = _run_bash(cmd_str, Path(job["stdout_log"]))

        gpu_slots_in_use[g] += 1
        procs[job["run_name"]] = {"p": p, "gpu": g, "job": job}

    # poll for finishes
    finished = []
    for name, rec in list(procs.items()):
        p: subprocess.Popen = rec["p"]
        ret = p.poll()
        if ret is not None:
            job = rec["job"]
            g = rec["gpu"]
            status = "OK" if ret == 0 else f"EXIT_{ret}"
            print(f"[done] {name} on GPU {g} → {status}")

            if ret != 0:
                _print_fail_tail(Path(job["stdout_log"]), lines=160)

            # postprocess copies (rollout logs into run_dir are the priority)
            run_dir = Path(job["run_dir"]).resolve()
            _postprocess_copy_outputs(
                run_dir=run_dir,
                run_name=job["run_name"],
                ckpt_label=job["ckpt_label"],
                seed=int(job["seed"]),
            )

            # retry once if failed
            will_retry = (ret != 0 and int(job["attempt"]) < RETRY_MAX)
            if will_retry:
                job2 = dict(job)
                job2["attempt"] = int(job["attempt"]) + 1
                print(f"[retry] re-queue {job2['run_name']} attempt={job2['attempt']}")
                queue.append(job2)

            # group accounting: only mark finished when this job is truly done (success OR final failure)
            is_final = (ret == 0) or (ret != 0 and int(job["attempt"]) >= RETRY_MAX)
            if is_final:
                bm = job["base_model"]
                gs = group_state.get(bm)
                if gs is not None:
                    gs["finished"].add(job["run_name"])
                    if ret == 0:
                        gs["success"].add(job["run_name"])

                    # if whole base_model group is done, decide whether to collect to CKPT_ROOT/_rollouts
                    if (not gs["collected"]) and (gs["finished"] == gs["expected"]):
                        if len(gs["success"]) == 0:
                            print(f"[collect] all adaptation runs failed for base_model={Path(bm).name}; skipping global rollout collection")
                        else:
                            rollouts_root = Path(CKPT_ROOT).resolve() / "_rollouts"
                            rollouts_root.mkdir(parents=True, exist_ok=True)
                            for rn in sorted(gs["success"]):
                                rd = Path(gs["run_dirs"][rn]).resolve()
                                src = rd / "rollout_log.csv"
                                if src.exists():
                                    dst = rollouts_root / f"{rn}__rollout_log.csv"
                                    _copy_file(src, dst)
                                else:
                                    print(f"[collect] WARN missing run_dir rollout_log.csv for {rn} ({src})")
                        gs["collected"] = True

            gpu_slots_in_use[g] = max(0, gpu_slots_in_use[g] - 1)
            finished.append(name)

    for name in finished:
        procs.pop(name, None)

    time.sleep(2.0)

print(f"[adapt] All runs finished. Outputs: {ckpt_root}")
print(f"[adapt] Index: {index_csv}")
print(f"[adapt] Rollouts (copies): {ckpt_root / '_rollouts'}")
