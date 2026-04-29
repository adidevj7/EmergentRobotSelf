# self_extractor_triplets.py
#
# Processes walk/jump/spin triplets of checkpoints.
# For each cycle triplet:
#   - Runs coactivation + block-diag pipeline (identical to lesion/plot script)
#   - Computes self-score per neuron (activation + connectivity stability across behaviors)
#   - Identifies self module (largest CC) and task module (all other CCs) per layer
#   - N_frozen = min(n_self, n_task) per layer — both pools trimmed to same size
#   - Saves per-model: *_self_freeze_idx.json and *_task_freeze_idx.json
#   - Copies original model .pth files into output folder
#   - Generates 2 PNGs per model: SelfFrozen and TaskFrozen
#     (bottom self-score scatter has frozen neurons marked in blue)
#
# Output layout:
#   OUT_ROOT/
#     models_all/         <- original .pth copies + JSONs
#     graphs/             <- PNG plots

import os
import re
import gc
import json
import shutil
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import seaborn as sns
from matplotlib.colors import TwoSlopeNorm, BoundaryNorm
from matplotlib.patches import Rectangle

from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee, connected_components


# ============================================================
# ========================= CONFIG ===========================
# ============================================================

MODELS: list[str] = [
    # Paste your model paths here — in triplets (walk, jump, spin per cycle)
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_0_2026-01-30_23-20-32/models/c015_b01_walk_plateau_2026-02-02_18-34-01_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_0_2026-01-30_23-20-32/models/c015_b03_jump_plateau_2026-02-02_21-44-22_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_0_2026-01-30_23-20-32/models/c015_b02_spin_plateau_2026-02-02_20-40-08_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_0_2026-01-30_23-20-32/models/c018_b01_walk_plateau_2026-02-03_09-22-46_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_0_2026-01-30_23-20-32/models/c018_b03_jump_plateau_2026-02-03_13-01-16_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_0_2026-01-30_23-20-32/models/c018_b02_spin_plateau_2026-02-03_12-06-39_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_0_2026-01-30_23-20-32/models/c040_b01_walk_plateau_2026-02-10_00-19-30_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_0_2026-01-30_23-20-32/models/c040_b03_jump_plateau_2026-02-10_03-10-42_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_0_2026-01-30_23-20-32/models/c040_b02_spin_plateau_2026-02-10_02-14-58_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_1_2026-01-30_23-24-17/models/c023_b01_walk_plateau_2026-02-04_21-25-32_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_1_2026-01-30_23-24-17/models/c023_b03_jump_plateau_2026-02-05_01-16-43_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_1_2026-01-30_23-24-17/models/c023_b02_spin_plateau_2026-02-04_23-32-28_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_1_2026-01-30_23-24-17/models/c024_b01_walk_plateau_2026-02-05_02-46-55_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_1_2026-01-30_23-24-17/models/c024_b03_jump_plateau_2026-02-05_06-09-26_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_1_2026-01-30_23-24-17/models/c024_b02_spin_plateau_2026-02-05_05-05-17_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_1_2026-01-30_23-24-17/models/c041_b01_walk_plateau_2026-02-10_12-45-58_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_1_2026-01-30_23-24-17/models/c041_b03_jump_plateau_2026-02-10_16-31-20_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_1_2026-01-30_23-24-17/models/c041_b02_spin_plateau_2026-02-10_15-04-22_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_2_2026-01-30_23-24-40/models/c018_b01_walk_plateau_2026-02-03_15-51-53_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_2_2026-01-30_23-24-40/models/c018_b03_jump_plateau_2026-02-03_18-48-06_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_2_2026-01-30_23-24-40/models/c018_b02_spin_plateau_2026-02-03_17-26-55_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_2_2026-01-30_23-24-40/models/c019_b01_walk_plateau_2026-02-03_20-19-07_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_2_2026-01-30_23-24-40/models/c019_b03_jump_plateau_2026-02-04_00-31-36_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_2_2026-01-30_23-24-40/models/c019_b02_spin_plateau_2026-02-03_22-47-20_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_2_2026-01-30_23-24-40/models/c022_b01_walk_plateau_2026-02-04_11-56-55_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_2_2026-01-30_23-24-40/models/c022_b03_jump_plateau_2026-02-04_15-21-06_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_2_2026-01-30_23-24-40/models/c022_b02_spin_plateau_2026-02-04_14-11-09_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_3_2026-01-30_23-24-53/models/c038_b01_walk_plateau_2026-02-09_14-27-25_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_3_2026-01-30_23-24-53/models/c038_b03_jump_plateau_2026-02-09_17-26-48_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_3_2026-01-30_23-24-53/models/c038_b02_spin_plateau_2026-02-09_15-57-43_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_3_2026-01-30_23-24-53/models/c041_b01_walk_plateau_2026-02-10_04-09-06_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_3_2026-01-30_23-24-53/models/c041_b03_jump_plateau_2026-02-10_06-51-09_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_3_2026-01-30_23-24-53/models/c041_b02_spin_plateau_2026-02-10_05-50-59_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_3_2026-01-30_23-24-53/models/c050_b01_walk_plateau_2026-02-12_07-42-16_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_3_2026-01-30_23-24-53/models/c050_b03_jump_plateau_2026-02-12_13-07-58_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_3_2026-01-30_23-24-53/models/c050_b02_spin_plateau_2026-02-12_09-30-34_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_4_2026-01-30_23-28-22/models/c019_b01_walk_plateau_2026-02-03_19-20-44_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_4_2026-01-30_23-28-22/models/c019_b03_jump_plateau_2026-02-03_22-30-30_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_4_2026-01-30_23-28-22/models/c019_b02_spin_plateau_2026-02-03_21-37-08_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_4_2026-01-30_23-28-22/models/c036_b01_walk_plateau_2026-02-09_09-53-22_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_4_2026-01-30_23-28-22/models/c036_b03_jump_plateau_2026-02-09_12-58-55_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_4_2026-01-30_23-28-22/models/c036_b02_spin_plateau_2026-02-09_11-30-31_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_4_2026-01-30_23-28-22/models/c037_b01_walk_plateau_2026-02-09_14-31-56_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_4_2026-01-30_23-28-22/models/c037_b03_jump_plateau_2026-02-09_18-42-32_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_4_2026-01-30_23-28-22/models/c037_b02_spin_plateau_2026-02-09_17-05-10_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_5_2026-01-30_23-28-40/models/c015_b01_walk_plateau_2026-02-03_03-48-55_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_5_2026-01-30_23-28-40/models/c015_b03_jump_plateau_2026-02-03_07-05-05_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_5_2026-01-30_23-28-40/models/c015_b02_spin_plateau_2026-02-03_06-01-08_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_5_2026-01-30_23-28-40/models/c016_b01_walk_plateau_2026-02-03_08-17-01_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_5_2026-01-30_23-28-40/models/c016_b03_jump_plateau_2026-02-03_11-35-04_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_5_2026-01-30_23-28-40/models/c016_b02_spin_plateau_2026-02-03_10-19-12_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_5_2026-01-30_23-28-40/models/c032_b01_walk_plateau_2026-02-07_01-57-17_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_5_2026-01-30_23-28-40/models/c032_b03_jump_plateau_2026-02-07_05-30-49_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_5_2026-01-30_23-28-40/models/c032_b02_spin_plateau_2026-02-07_04-08-10_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_6_2026-01-30_23-29-08/models/c020_b01_walk_plateau_2026-02-04_12-26-45_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_6_2026-01-30_23-29-08/models/c020_b03_jump_plateau_2026-02-04_17-44-18_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_6_2026-01-30_23-29-08/models/c020_b02_spin_plateau_2026-02-04_15-06-17_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_6_2026-01-30_23-29-08/models/c025_b01_walk_plateau_2026-02-05_20-17-34_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_6_2026-01-30_23-29-08/models/c025_b03_jump_plateau_2026-02-05_23-35-27_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_6_2026-01-30_23-29-08/models/c025_b02_spin_plateau_2026-02-05_22-25-58_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_6_2026-01-30_23-29-08/models/c026_b01_walk_plateau_2026-02-06_00-41-20_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_6_2026-01-30_23-29-08/models/c026_b03_jump_plateau_2026-02-06_03-57-41_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_6_2026-01-30_23-29-08/models/c026_b02_spin_plateau_2026-02-06_02-38-18_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_7_2026-01-30_23-29-24/models/c017_b01_walk_plateau_2026-02-03_17-12-17_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_7_2026-01-30_23-29-24/models/c017_b03_jump_plateau_2026-02-03_21-42-23_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_7_2026-01-30_23-29-24/models/c017_b02_spin_plateau_2026-02-03_19-24-47_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_7_2026-01-30_23-29-24/models/c040_b01_walk_plateau_2026-02-10_03-52-02_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_7_2026-01-30_23-29-24/models/c040_b03_jump_plateau_2026-02-10_07-15-13_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_7_2026-01-30_23-29-24/models/c040_b02_spin_plateau_2026-02-10_05-25-57_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_7_2026-01-30_23-29-24/models/c043_b01_walk_plateau_2026-02-10_18-18-16_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_7_2026-01-30_23-29-24/models/c043_b03_jump_plateau_2026-02-10_21-54-55_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_7_2026-01-30_23-29-24/models/c043_b02_spin_plateau_2026-02-10_20-40-28_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_8_2026-02-01_03-47-27/models/c021_b01_walk_plateau_2026-02-05_13-39-47_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_8_2026-02-01_03-47-27/models/c021_b03_jump_plateau_2026-02-05_18-22-08_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_8_2026-02-01_03-47-27/models/c021_b02_spin_plateau_2026-02-05_17-06-44_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_8_2026-02-01_03-47-27/models/c033_b01_walk_plateau_2026-02-08_03-41-38_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_8_2026-02-01_03-47-27/models/c033_b03_jump_plateau_2026-02-08_05-53-35_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_8_2026-02-01_03-47-27/models/c033_b02_spin_plateau_2026-02-08_05-02-23_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_8_2026-02-01_03-47-27/models/c039_b01_walk_plateau_2026-02-09_23-22-53_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_8_2026-02-01_03-47-27/models/c039_b03_jump_plateau_2026-02-10_02-18-44_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_8_2026-02-01_03-47-27/models/c039_b02_spin_plateau_2026-02-10_01-25-17_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_9_2026-02-01_03-48-07/models/c039_b01_walk_plateau_2026-02-10_07-45-00_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_9_2026-02-01_03-48-07/models/c039_b03_jump_plateau_2026-02-10_11-33-35_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_9_2026-02-01_03-48-07/models/c039_b02_spin_plateau_2026-02-10_10-22-23_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_9_2026-02-01_03-48-07/models/c044_b01_walk_plateau_2026-02-11_10-21-02_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_9_2026-02-01_03-48-07/models/c044_b03_jump_plateau_2026-02-11_13-35-32_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_9_2026-02-01_03-48-07/models/c044_b02_spin_plateau_2026-02-11_12-31-02_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_9_2026-02-01_03-48-07/models/c046_b01_walk_plateau_2026-02-11_21-20-51_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_9_2026-02-01_03-48-07/models/c046_b03_jump_plateau_2026-02-12_00-10-33_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_9_2026-02-01_03-48-07/models/c046_b02_spin_plateau_2026-02-11_23-24-06_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_10_2026-02-01_03-48-27/models/c020_b01_walk_plateau_2026-02-05_02-39-46_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_10_2026-02-01_03-48-27/models/c020_b03_jump_plateau_2026-02-05_05-03-56_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_10_2026-02-01_03-48-27/models/c020_b02_spin_plateau_2026-02-05_03-53-20_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_10_2026-02-01_03-48-27/models/c025_b01_walk_plateau_2026-02-05_22-41-24_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_10_2026-02-01_03-48-27/models/c025_b03_jump_plateau_2026-02-06_02-44-06_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_10_2026-02-01_03-48-27/models/c025_b02_spin_plateau_2026-02-06_01-50-49_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_10_2026-02-01_03-48-27/models/c032_b01_walk_plateau_2026-02-07_05-12-08_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_10_2026-02-01_03-48-27/models/c032_b03_jump_plateau_2026-02-07_08-17-02_for_play.pth',
  '/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_relu_10_2026-02-01_03-48-27/models/c032_b02_spin_plateau_2026-02-07_06-58-01_for_play.pth'
  ]

ALL_STATES_PATH = "/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/Obs/States/ALL_states_concat.npy"
N_ALL_STATES    = 500_000

LAYER_INDICES = [0, 1]   # H1=0, H2=1

SEED        = 42
FREEZE_SEED = 42         # seed for random sampling within self/task pools

EPS         = 1e-8
MIN_STD     = 1e-5
ACTIVATION  = "relu"     # {"relu","elu","tanh"}

TAU              = 0.70  # coactivation threshold for block-diag edges
BD_MIN_BLOCK_SIZE = 1

DEVICE = "cpu"

CMAP               = "RdBu_r"
VMIN, VCENTER, VMAX = -1.0, 0.0, 1.0

OUT_ROOT       = "/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/SelfExtractorTriplets"
OUT_MODELS_DIR = os.path.join(OUT_ROOT, "models_all")
OUT_GRAPHS_DIR = os.path.join(OUT_ROOT, "graphs")

sns.set_theme(style="white", font="Arial")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]


# ============================================================
# ========================= HELPERS ==========================
# ============================================================

def _seed_everything(seed: int):
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _activation_fn(x: np.ndarray, name: str) -> np.ndarray:
    if name == "relu":  return np.maximum(0.0, x)
    if name == "elu":
        y = x.copy(); y[x <= 0] = np.expm1(x[x <= 0]); return y
    if name == "tanh":  return np.tanh(x)
    raise ValueError(f"Unknown activation: {name}")

def _zscore_cols(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if X.size == 0:
        return X.astype(np.float32, copy=False)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return ((X - mu) / (sd + eps)).astype(np.float32, copy=False)

def _dead_alive_indices(acts: np.ndarray, min_std: float):
    if acts.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    sd = acts.std(axis=0)
    dead  = np.where(sd <= min_std)[0].astype(int)
    alive = np.where(sd >  min_std)[0].astype(int)
    return dead, alive

def corr_matrix(Xz: np.ndarray) -> np.ndarray:
    Xz = np.asarray(Xz, dtype=np.float32)
    n = Xz.shape[1]
    if n == 0: return np.zeros((0, 0), np.float32)
    if n == 1: return np.ones((1, 1), np.float32)
    norms = np.linalg.norm(Xz, axis=0, keepdims=True)
    norms = np.where(norms < EPS, 1.0, norms)
    sim = (Xz.T @ Xz) / (norms.T @ norms)
    sim = np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    np.fill_diagonal(sim, 1.0)
    return sim

def _safe_corr(a, b) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size != b.size or a.size == 0: return 0.0
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na < EPS or nb < EPS: return 0.0
    c = float(np.dot(a, b) / (na * nb))
    return 0.0 if np.isnan(c) else float(abs(c))

def _torch_load_compat(path: str):
    try:    return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError: return torch.load(path, map_location="cpu")
    except Exception:
        try:    return torch.load(path, map_location="cpu", weights_only=False)
        except: return torch.load(path, map_location="cpu")

def _load_rlg_forplay_state_dict(path: str) -> dict:
    payload = _torch_load_compat(path)
    if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        return {k: (v.detach().cpu().clone() if torch.is_tensor(v) else v)
                for k, v in payload["model"].items()}
    raise ValueError(f"[ckpt] {path} missing 'model' dict.")

def _discover_actor_mlp_layers(state_dict: dict) -> list:
    cands = []
    for k in state_dict:
        if k.endswith(".weight") and ("actor_mlp" in k or ".actor." in k
                                       or "actor_net" in k or "actor.trunk" in k):
            parts = k.split(".")
            try:   idx = int(parts[-2])
            except: idx = 10**6
            bkey = k[:-6] + "bias"
            if bkey in state_dict:
                cands.append((idx, k, bkey))
    cands.sort(key=lambda t: (t[0], t[1]))
    return cands

def _actor_hidden_forward(X: np.ndarray, actor_layers: list, sd: dict) -> list:
    h = X.astype(np.float32, copy=False)
    outs = []
    for _, wk, bk in actor_layers:
        W = sd[wk]; b = sd[bk]
        if torch.is_tensor(W): W = W.numpy()
        if torch.is_tensor(b): b = b.numpy()
        h = h @ W.T + b[None, :]
        h = _activation_fn(h, ACTIVATION)
        outs.append(h.astype(np.float32, copy=False))
    return outs

def _get_layer_acts(sd: dict, X_ref: np.ndarray, layer_idx: int, actor_layers: list) -> np.ndarray:
    outs = _actor_hidden_forward(X_ref, actor_layers, sd)
    if layer_idx < 0 or layer_idx >= len(outs):
        raise IndexError(f"layer_idx={layer_idx} out of range (0..{len(outs)-1})")
    return outs[layer_idx]

def _sanitize_filename(s: str, max_len: int = 220) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    s = re.sub(r"[^\w\-\.\(\)\[\] ]+", "", s).replace(" ", "_")
    return s[:max_len]

def _extract_plateau_ts(path: str) -> str:
    fname = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"plateau_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", fname)
    if m: return m.group(1)
    m2 = re.search(r"(\d{4}-\d{2}-\d{2}[_-]\d{2}-\d{2}-\d{2})", fname)
    return m2.group(1) if m2 else "unknown"

def _parse_title_fields(path: str):
    p = str(path).replace("\\", "/")
    run_dir = os.path.basename(os.path.dirname(os.path.dirname(p)))
    fname   = os.path.splitext(os.path.basename(p))[0]
    m_run = re.search(r"(?:^|_)relu_(\d+)(?:_|$)", run_dir)
    run_num = int(m_run.group(1)) if m_run else None
    m = re.search(r"c(\d+)_b(\d+)_([A-Za-z]+)", fname)
    cyc      = int(m.group(1))          if m else None
    beh_num  = int(m.group(2))          if m else None
    beh_name = m.group(3).capitalize()  if m else None
    return run_num, cyc, beh_num, beh_name

def _model_stem(path: str) -> str:
    """Standardised filename stem for output files."""
    run_num, cyc, beh_num, beh_name = _parse_title_fields(path)
    ts = _extract_plateau_ts(path)
    run_str  = f"{int(run_num):02d}" if run_num  is not None else "??"
    cyc_str  = f"{int(cyc):03d}"     if cyc      is not None else "???"
    beh_str  = f"{int(beh_num):02d}" if beh_num  is not None else "??"
    beh_low  = beh_name.lower()       if beh_name is not None else "unknown"
    return f"run{run_str}_c{cyc_str}_b{beh_str}_{beh_low}_plateau_{ts}"


# ============================================================
# ============= BLOCK-DIAGONALISATION ========================
# ============================================================

def _blockdiag_rcm(R_abs: np.ndarray, tau: float, min_block_size: int = 1):
    """
    Threshold |cosine| >= tau → connected components → sort by size (largest first)
    → RCM ordering within each block.
    Returns order (0-based into alive_idx), labels (1-based, 1 = largest = 'self').
    """
    n = R_abs.shape[0]
    if n == 0: return np.array([], dtype=int), np.array([], dtype=int)
    if n < 3:  return np.arange(n, dtype=int), np.ones(n, dtype=int)

    R = np.clip(0.5 * (R_abs + R_abs.T), 0.0, 1.0).astype(np.float32)
    np.fill_diagonal(R, 0.0)
    A = (R >= float(tau)).astype(np.int8)
    np.fill_diagonal(A, 0)
    G = csr_matrix(A)

    n_comp, comp = connected_components(G, directed=False, connection="weak")

    # Merge tiny blocks into sentinel
    if min_block_size > 1:
        sizes = np.bincount(comp, minlength=n_comp)
        small = np.where(sizes < min_block_size)[0]
        if small.size:
            comp[np.isin(comp, small)] = n_comp
            keep  = sorted(set(comp.tolist()))
            remap = {c: i for i, c in enumerate(keep)}
            comp  = np.array([remap[c] for c in comp], dtype=int)
            n_comp = len(keep)

    # Sort blocks largest-first; RCM inside each
    block_sizes = [(c, int(np.sum(comp == c))) for c in range(int(comp.max()) + 1)]
    block_sizes.sort(key=lambda t: -t[1])

    parts = []
    for c, _ in block_sizes:
        idxs = np.where(comp == c)[0]
        if idxs.size <= 2:
            parts.append(idxs)
        else:
            sub_ord = reverse_cuthill_mckee(G[idxs][:, idxs], symmetric_mode=True)
            parts.append(idxs[np.asarray(sub_ord, dtype=int)])

    order = np.concatenate(parts).astype(int)

    # Labels: rank by block size (1 = largest = self)
    labels = np.empty(n, dtype=int)
    for rank, (c, _) in enumerate(block_sizes, start=1):
        labels[comp == c] = rank

    return order, labels

def _relabel_by_size(labels: np.ndarray) -> np.ndarray:
    """Re-number labels so 1 = largest cluster."""
    if labels.size == 0: return labels.astype(int)
    uids, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)
    remap = {uid: rank + 1 for rank, uid in enumerate(uids[order])}
    return np.array([remap[c] for c in labels], dtype=int)

def _order_by_cluster_size(labels: np.ndarray, base_order: np.ndarray = None) -> np.ndarray:
    labels = np.asarray(labels)
    n = labels.size
    if n == 0: return np.arange(0, dtype=int)
    if base_order is None: base_order = np.arange(n, dtype=int)
    base_order = np.asarray(base_order)
    uids, counts = np.unique(labels, return_counts=True)
    sorted_uids = [uid for uid, _ in sorted(zip(uids, counts), key=lambda t: -t[1])]
    labels_in_base = labels[base_order]
    parts = [base_order[labels_in_base == uid] for uid in sorted_uids]
    return np.concatenate(parts) if parts else np.arange(n, dtype=int)


# ============================================================
# =================== GROUPING ===============================
# ============================================================

def _behavior_from_path(path: str) -> str:
    low = os.path.basename(path).lower()
    for b in ("walk", "jump", "spin"):
        if f"_{b}_" in low: return b
    m = re.search(r"_([a-zA-Z]+)_plateau", low)
    if m and m.group(1).lower() in ("walk", "jump", "spin"):
        return m.group(1).lower()
    raise RuntimeError(f"Cannot infer behavior from: {path}")

def _cycle_key_from_path(path: str):
    p = str(path).replace("\\", "/")
    run_dir = os.path.basename(os.path.dirname(os.path.dirname(p)))
    fname   = os.path.splitext(os.path.basename(p))[0]
    m = re.search(r"c(\d+)_b(\d+)_", fname)
    if not m: raise RuntimeError(f"Cannot parse cycle from: {path}")
    return (run_dir, int(m.group(1)))

def group_into_cycles(models: list) -> list:
    groups = {}
    for p in models:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing model: {p}")
        k   = _cycle_key_from_path(p)
        beh = _behavior_from_path(p)
        groups.setdefault(k, {})
        if beh in groups[k]:
            raise RuntimeError(f"Duplicate behavior '{beh}' for cycle {k}: {p}")
        groups[k][beh] = p
    out = []
    for k, mp in sorted(groups.items(), key=lambda t: (t[0][0], t[0][1])):
        missing = [b for b in ("walk", "jump", "spin") if b not in mp]
        if missing:
            raise RuntimeError(f"Cycle {k} missing behaviors: {missing}")
        out.append((k, {"walk": mp["walk"], "jump": mp["jump"], "spin": mp["spin"]}))
    return out


# ============================================================
# =================== SELF-SCORE =============================
# ============================================================

def hungarian_match_cosine(A: np.ndarray, B: np.ndarray, eps: float = 1e-8):
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    if A.size == 0 or B.size == 0:
        return (np.zeros((0, 0), np.float32),
                np.array([], dtype=int), np.array([], dtype=int),
                np.array([], dtype=float))
    nA = np.linalg.norm(A, axis=0, keepdims=True); nA[nA < eps] = 1.0
    nB = np.linalg.norm(B, axis=0, keepdims=True); nB[nB < eps] = 1.0
    sim = np.nan_to_num((A.T @ B) / (nA.T @ nB))
    i_idx, j_idx = linear_sum_assignment(-sim)
    vals = sim[i_idx, j_idx] if i_idx.size else np.array([], dtype=float)
    return sim, i_idx, j_idx, vals

def _compute_self_scores(A_z_by_model: dict, model_order: list):
    """
    Cross-behavior Hungarian matching → per-neuron activation + connectivity stability.
    Returns (fam_idx_by_model, activation_stab, connectivity_stab, self_score).
    All arrays indexed over the family (matched) neurons of the reference model (walk).
    """
    empty = {m: np.array([], dtype=int) for m in model_order}
    A_ref = A_z_by_model["walk"]
    n_ref = int(A_ref.shape[1])
    if n_ref <= 0: return empty, np.array([]), np.array([]), np.array([])
    if any(A_z_by_model[m].shape[1] <= 0 for m in ("jump", "spin")):
        return empty, np.array([]), np.array([]), np.array([])

    maps = {}
    for other in ("jump", "spin"):
        _, i_idx, j_idx, _ = hungarian_match_cosine(A_ref, A_z_by_model[other])
        idx_map = np.full(n_ref, -1, dtype=int)
        if i_idx.size: idx_map[i_idx] = j_idx
        maps[other] = idx_map

    good = np.where((maps["jump"] >= 0) & (maps["spin"] >= 0))[0].astype(int)
    if good.size == 0: return empty, np.array([]), np.array([]), np.array([])

    fam = {"walk": good,
           "jump": maps["jump"][good].astype(int),
           "spin": maps["spin"][good].astype(int)}
    n_fam = int(good.size)

    W  = {m: A_z_by_model[m][:, fam[m]]   for m in model_order}
    Rf = {m: corr_matrix(W[m])              for m in model_order}

    pairs = [("walk","jump"), ("walk","spin"), ("jump","spin")]
    act_stab  = np.zeros(n_fam, dtype=np.float32)
    conn_stab = np.zeros(n_fam, dtype=np.float32)
    for k in range(n_fam):
        act_stab[k]  = float(np.mean([_safe_corr(W[a][:, k], W[b][:, k]) for a, b in pairs]))
        conn_stab[k] = float(np.mean([_safe_corr(Rf[a][k, :], Rf[b][k, :]) for a, b in pairs]))

    self_score = 0.5 * (act_stab + conn_stab)
    return fam, act_stab, conn_stab, self_score.astype(np.float32)


# ============================================================
# =================== PLOT HELPERS ===========================
# ============================================================

def _cluster_bounds(labels: np.ndarray, order: np.ndarray) -> list:
    lr = labels[order]
    return [i - 0.5 for i in range(1, len(lr)) if lr[i] != lr[i - 1]]

def _cluster_spans(labels_ordered: np.ndarray) -> dict:
    spans = {}
    if labels_ordered.size == 0: return spans
    start = 0; cur = int(labels_ordered[0])
    for i in range(1, labels_ordered.size):
        cid = int(labels_ordered[i])
        if cid != cur:
            spans.setdefault(cur, []).append((start, i - 1))
            cur = cid; start = i
    spans.setdefault(cur, []).append((start, labels_ordered.size - 1))
    return spans

def _overlay_module_boxes(ax, labels_ord, cmap, cnorm, alpha_fill=0.52, edge_alpha=0.95):
    for cid, segs in _cluster_spans(labels_ord).items():
        color = cmap(cnorm(int(cid)))
        for (a, b) in segs:
            w = b - a + 1
            ax.add_patch(Rectangle(
                (a - 0.5, a - 0.5), w, w,
                facecolor=color,
                edgecolor=(color[0], color[1], color[2], edge_alpha),
                linewidth=2.1, alpha=alpha_fill, zorder=3))

def _force_square(ax):
    pos = ax.get_position()
    cx = pos.x0 + pos.width * 0.5; cy = pos.y0 + pos.height * 0.5
    side = pos.width
    ax.set_position([cx - side * 0.5, cy - side * 0.5, side, side])

def _frozen_plot_positions(frozen_orig_idx: np.ndarray,
                            alive_idx:        np.ndarray,
                            order:            np.ndarray) -> np.ndarray:
    """
    Map original neuron indices → positions in the ordered scatter plot.
    Returns array of x-positions (may be empty).
    """
    if frozen_orig_idx.size == 0 or alive_idx.size == 0:
        return np.array([], dtype=int)
    alive_map = {int(orig): pos for pos, orig in enumerate(alive_idx)}
    inv_order = np.empty(len(order), dtype=int)
    inv_order[order] = np.arange(len(order), dtype=int)
    positions = []
    for f in frozen_orig_idx:
        ap = alive_map.get(int(f), None)
        if ap is not None and ap < len(inv_order):
            positions.append(int(inv_order[ap]))
    return np.array(positions, dtype=int)


# ============================================================
# =================== MAIN PLOT FUNCTION =====================
# ============================================================

def _plot_triplet_model(
    model_name:      str,
    model_path:      str,
    freeze_type:     str,          # "self" or "task"
    layer_payloads:  dict,         # layer_idx -> (A_z_by_model, R_by_model, alive, dead, total)
    orig_fixed_cache: dict,
    frozen_orig:     dict,         # layer_idx -> np.ndarray of original neuron indices
    out_png_path:    str,
):
    """
    Generate one PNG for a model, with frozen neurons marked blue in the self-score scatter.
    freeze_type is used only for the title label.
    """
    model_order = ["walk", "jump", "spin"]

    (A0, R0_dict, alive0, dead0, total0) = layer_payloads[0]
    (A1, R1_dict, alive1, dead1, total1) = layer_payloads[1]

    fam0, act0, conn0, self_sc0 = _compute_self_scores(A0, model_order)
    fam1, act1, conn1, self_sc1 = _compute_self_scores(A1, model_order)

    labels0 = np.asarray(orig_fixed_cache["labels"][model_name][0], dtype=int)
    order0  = np.asarray(orig_fixed_cache["order"][model_name][0],  dtype=int)
    labels1 = np.asarray(orig_fixed_cache["labels"][model_name][1], dtype=int)
    order1  = np.asarray(orig_fixed_cache["order"][model_name][1],  dtype=int)

    R0 = np.asarray(R0_dict[model_name])
    R1 = np.asarray(R1_dict[model_name])
    R0_plot = R0[np.ix_(order0, order0)] if R0.size and order0.size else R0
    R1_plot = R1[np.ix_(order1, order1)] if R1.size and order1.size else R1

    labels0_ord = labels0[order0] if labels0.size and order0.size else labels0
    labels1_ord = labels1[order1] if labels1.size and order1.size else labels1

    max_cid = int(max(
        labels0.max() if labels0.size else 0,
        labels1.max() if labels1.size else 0,
    ))
    max_cid = max(1, max_cid)
    cmap_c  = plt.get_cmap("tab20", max_cid)
    cnorm_c = BoundaryNorm(np.arange(0.5, max_cid + 1.5, 1.0), max_cid)

    def _ordered_scores(n_units, fam_idx, score, order):
        out = np.full(n_units, np.nan, dtype=np.float32)
        fam_idx = np.asarray(fam_idx, dtype=int)
        if fam_idx.size and score.size == fam_idx.size:
            out[fam_idx] = score.astype(np.float32, copy=False)
        return out[order] if order.size else out

    fam0_idx = fam0.get(model_name, np.array([], dtype=int))
    fam1_idx = fam1.get(model_name, np.array([], dtype=int))
    self0_ord = _ordered_scores(R0.shape[0], fam0_idx, self_sc0, order0)
    self1_ord = _ordered_scores(R1.shape[0], fam1_idx, self_sc1, order1)

    # Map frozen original indices → scatter x-positions
    alive_idx_0 = np.asarray(orig_fixed_cache["alive"][model_name][0], dtype=int)
    alive_idx_1 = np.asarray(orig_fixed_cache["alive"][model_name][1], dtype=int)
    frozen_plot0 = _frozen_plot_positions(frozen_orig.get(0, np.array([], dtype=int)), alive_idx_0, order0)
    frozen_plot1 = _frozen_plot_positions(frozen_orig.get(1, np.array([], dtype=int)), alive_idx_1, order1)

    # Title
    run_num, cyc, beh_num, beh_name = _parse_title_fields(model_path)
    type_label = "Self-Frozen" if freeze_type == "self" else "Task-Frozen"
    suptitle = (f"WSJ - Run {run_num} - Cycle {cyc:03d} - "
                f"B{beh_num:02d} {beh_name} - {type_label}")

    fig = plt.figure(figsize=(15.5, 10.6))
    gs = fig.add_gridspec(
        nrows=3, ncols=2,
        width_ratios=[1.0, 1.0],
        height_ratios=[0.06, 1.0, 0.34],
        wspace=0.10, hspace=0.08,
    )
    ax_top0 = fig.add_subplot(gs[0, 0]);  ax_top1 = fig.add_subplot(gs[0, 1])
    ax_hm0  = fig.add_subplot(gs[1, 0]);  ax_hm1  = fig.add_subplot(gs[1, 1])
    ax_sc0  = fig.add_subplot(gs[2, 0]);  ax_sc1  = fig.add_subplot(gs[2, 1])

    # ── cluster-colour top bars ──────────────────────────────────────────────
    for ax_top, labs_ord in [(ax_top0, labels0_ord), (ax_top1, labels1_ord)]:
        if labs_ord.size:
            sns.heatmap(labs_ord[None, :], ax=ax_top,
                        cmap=cmap_c, norm=cnorm_c,
                        cbar=False, xticklabels=False, yticklabels=False)
        else:
            ax_top.axis("off")

    # ── correlation heatmaps ─────────────────────────────────────────────────
    for ax_hm, R_plot, labs_ord, labs, ord_ in [
        (ax_hm0, R0_plot, labels0_ord, labels0, order0),
        (ax_hm1, R1_plot, labels1_ord, labels1, order1),
    ]:
        if R_plot.size:
            sns.heatmap(R_plot, ax=ax_hm,
                        cmap=CMAP, vmin=VMIN, vmax=VMAX, center=VCENTER,
                        square=False, cbar=False,
                        xticklabels=False, yticklabels=False)
            if labs_ord.size:
                _overlay_module_boxes(ax_hm, labs_ord, cmap_c, cnorm_c)
                for b in _cluster_bounds(labs, ord_):
                    ax_hm.axhline(b, color="k", lw=0.9, zorder=5)
                    ax_hm.axvline(b, color="k", lw=0.9, zorder=5)
        else:
            ax_hm.text(0.5, 0.5, "Empty", ha="center", va="center")
            ax_hm.set_xticks([]); ax_hm.set_yticks([])
        ax_hm.set_title("")

    # ── self-score scatter ───────────────────────────────────────────────────
    n_alive0 = int(alive0.get(model_name, R0.shape[0]))
    n_tot0   = int(total0.get(model_name, n_alive0))
    n_alive1 = int(alive1.get(model_name, R1.shape[0]))
    n_tot1   = int(total1.get(model_name, n_alive1))

    n_frozen_0 = len(frozen_orig.get(0, []))
    n_frozen_1 = len(frozen_orig.get(1, []))

    def _scatter(ax, y_ord, labs_ord, frozen_pos, xlabel, show_ylabel, hide_yticks=False):
        ax.set_ylim(0.0, 1.05)
        ax.axhline(1.0, ls="--", lw=1.2, alpha=0.5)
        ax.set_xticks([])
        ax.set_xlabel(xlabel, fontsize=18, fontweight="bold", labelpad=6)
        if show_ylabel:
            ax.set_ylabel("Self-Score", fontsize=16)
        if hide_yticks:
            ax.set_yticks([]); ax.set_yticklabels([])
        else:
            ax.set_yticks([0.0, 0.5, 1.0]); ax.set_yticklabels(["0", "0.5", "1"])

        if y_ord.size:
            x = np.arange(y_ord.size)
            mask = np.isfinite(y_ord)

            # All neurons — grey
            if np.any(mask):
                sns.scatterplot(x=x[mask], y=y_ord[mask],
                                ax=ax, s=18, alpha=0.30,
                                color="grey", edgecolor=None)

            # Frozen neurons — blue (on top)
            if frozen_pos.size:
                fmask = np.zeros(y_ord.size, dtype=bool)
                fmask[frozen_pos[frozen_pos < y_ord.size]] = True
                fmask &= mask
                if np.any(fmask):
                    sns.scatterplot(x=x[fmask], y=y_ord[fmask],
                                    ax=ax, s=38, alpha=0.85,
                                    color="royalblue", edgecolor=None,
                                    zorder=5, label=f"frozen ({freeze_type}, n={frozen_pos.size})")
                    ax.legend(fontsize=8, loc="upper right", framealpha=0.6)

            # Per-cluster mean lines
            if labs_ord.size:
                for cid in np.unique(labs_ord):
                    idxs = np.where(labs_ord == cid)[0]
                    yv   = y_ord[idxs]; yv = yv[np.isfinite(yv)]
                    if yv.size:
                        ax.hlines(float(yv.mean()), idxs.min() - 0.5, idxs.max() + 0.5, lw=3.0)

                # Self / task module mean lines
                self_mask_ = (labs_ord == 1) & mask
                task_mask_ = (labs_ord != 1) & mask
                if np.any(self_mask_):
                    sm = float(y_ord[self_mask_].mean())
                    ax.axhline(sm, color="k", lw=2.4, alpha=0.35, zorder=6)
                    ax.text(0.01, sm + 0.01, "self mean",
                            transform=ax.get_yaxis_transform(),
                            ha="left", va="bottom", fontsize=9, color="k", alpha=0.6)
                if np.any(task_mask_):
                    tm = float(y_ord[task_mask_].mean())
                    ax.axhline(tm, color="r", lw=2.4, alpha=0.28, zorder=6)
                    ax.text(0.01, tm + 0.01, "task mean",
                            transform=ax.get_yaxis_transform(),
                            ha="left", va="bottom", fontsize=9, color="r", alpha=0.55)

            ax.set_xlim(-0.5, len(y_ord) - 0.5)
        else:
            ax.text(0.5, 0.5, "Empty", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlim(-0.5, 1.0)

    _scatter(ax_sc0, self0_ord, labels0_ord, frozen_plot0,
             xlabel=f"Layer 1  (frozen={n_frozen_0})",
             show_ylabel=True, hide_yticks=False)
    _scatter(ax_sc1, self1_ord, labels1_ord, frozen_plot1,
             xlabel=f"Layer 2  (frozen={n_frozen_1})",
             show_ylabel=False, hide_yticks=True)

    # Layout polish
    fig.canvas.draw()
    _force_square(ax_hm0); _force_square(ax_hm1)
    fig.canvas.draw()

    pos0 = ax_hm0.get_position(); pos1 = ax_hm1.get_position()
    def _match_x(ax, ref):
        p = ax.get_position()
        ax.set_position([ref.x0, p.y0, ref.width, p.height])
    _match_x(ax_top0, pos0); _match_x(ax_sc0, pos0)
    _match_x(ax_top1, pos1); _match_x(ax_sc1, pos1)

    ax_sc0.text(0.5, -0.155, f"(alive {n_alive0}/{n_tot0})",
                transform=ax_sc0.transAxes, ha="center", va="top", fontsize=10)
    ax_sc1.text(0.5, -0.155, f"(alive {n_alive1}/{n_tot1})",
                transform=ax_sc1.transAxes, ha="center", va="top", fontsize=10)

    # Colorbar
    if R0_plot.size:
        import matplotlib as mpl
        sm_ = mpl.cm.ScalarMappable(
            cmap=plt.get_cmap(CMAP),
            norm=TwoSlopeNorm(vmin=VMIN, vcenter=VCENTER, vmax=VMAX))
        sm_.set_array([])
        cb_w = 0.012 * 1.2
        cb_h = pos0.height * 1.6
        cb_y = pos0.y0 + (pos0.height - cb_h) * 0.5 + 0.02
        cb_x = pos0.x0 - 5.9 * cb_w
        cax  = fig.add_axes([cb_x, cb_y, cb_w, cb_h])
        cb   = fig.colorbar(sm_, cax=cax)
        cb.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        cb.ax.tick_params(labelsize=10)

    fig.suptitle(suptitle, fontsize=16, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.95, bottom=0.06)

    _ensure_dir(os.path.dirname(out_png_path))
    fig.savefig(out_png_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved → {os.path.basename(out_png_path)}")


# ============================================================
# ========================= MAIN =============================
# ============================================================

def main():
    _seed_everything(SEED)
    _ensure_dir(OUT_MODELS_DIR)
    _ensure_dir(OUT_GRAPHS_DIR)

    cycles = group_into_cycles(MODELS)
    print(f"[INFO] {len(cycles)} cycle triplets ({len(MODELS)} models total)")

    # ── Load reference states ────────────────────────────────────────────────
    if not os.path.exists(ALL_STATES_PATH):
        raise FileNotFoundError(ALL_STATES_PATH)
    all_states = np.load(ALL_STATES_PATH)
    if all_states.ndim != 2:
        raise ValueError(f"Expected 2D states array, got {all_states.shape}")

    rng_ref = np.random.RandomState(SEED)
    n_avail = all_states.shape[0]
    idx_ref = rng_ref.choice(n_avail, size=min(N_ALL_STATES, n_avail),
                              replace=(n_avail < N_ALL_STATES))
    X_ref = all_states[idx_ref].astype(np.float32, copy=False)
    print(f"[INFO] Reference states: {X_ref.shape}")

    model_order = ["walk", "jump", "spin"]

    total_models = 0
    total_jsons  = 0
    total_plots  = 0

    for (run_dir, cyc), m_paths in cycles:
        print(f"\n[cycle] {run_dir} / c{cyc:03d}")

        # Load state-dicts + actor layer descriptors
        state_dicts = {}
        actor_layers = {}
        for beh in model_order:
            sd  = _load_rlg_forplay_state_dict(m_paths[beh])
            lyr = _discover_actor_mlp_layers(sd)
            if not lyr: raise RuntimeError(f"No actor MLP layers: {m_paths[beh]}")
            state_dicts[beh]  = sd
            actor_layers[beh] = lyr

        # ── Per-model, per-layer: block-diag → self/task pools ───────────────
        #    We also build the fixed-cache needed for the plotting functions.
        orig_fixed = {
            "alive":  {m: {} for m in model_order},
            "order":  {m: {} for m in model_order},
            "labels": {m: {} for m in model_order},
        }

        # self_pool[model][layer_idx] = original neuron indices in self module
        # task_pool[model][layer_idx] = original neuron indices in task module
        self_pool = {m: {} for m in model_order}
        task_pool = {m: {} for m in model_order}

        for layer_idx in LAYER_INDICES:
            for beh in model_order:
                acts_full = _get_layer_acts(state_dicts[beh], X_ref, layer_idx, actor_layers[beh])
                _, alive_idx = _dead_alive_indices(acts_full, MIN_STD)
                alive_idx = np.asarray(alive_idx, dtype=int)

                acts_alive = (acts_full[:, alive_idx] if alive_idx.size
                              else acts_full[:, :0])
                A_z   = _zscore_cols(acts_alive) if acts_alive.size else acts_alive
                R_abs = np.abs(corr_matrix(A_z))

                order_raw, labels_raw = _blockdiag_rcm(R_abs, tau=TAU,
                                                        min_block_size=BD_MIN_BLOCK_SIZE)
                labels = _relabel_by_size(labels_raw)
                order  = _order_by_cluster_size(labels, base_order=order_raw)

                orig_fixed["alive"][beh][layer_idx]  = alive_idx
                orig_fixed["order"][beh][layer_idx]  = order.astype(int)
                orig_fixed["labels"][beh][layer_idx] = labels.astype(int)

                # self = label 1 (largest CC), task = everything else
                self_alive_pos = np.where(labels == 1)[0].astype(int)
                task_alive_pos = np.where(labels != 1)[0].astype(int)
                self_pool[beh][layer_idx] = (alive_idx[self_alive_pos]
                                              if self_alive_pos.size else np.array([], dtype=int))
                task_pool[beh][layer_idx] = (alive_idx[task_alive_pos]
                                              if task_alive_pos.size else np.array([], dtype=int))

        # ── For each model: determine N_frozen per layer, sample, save JSONs ─
        for mi, beh in enumerate(model_order):
            orig_path = m_paths[beh]
            stem      = _model_stem(orig_path)

            # Copy original model into output folder
            dst_pth = os.path.join(OUT_MODELS_DIR, stem + "_for_play.pth")
            if not os.path.exists(dst_pth):
                shutil.copy2(orig_path, dst_pth)
                total_models += 1
            else:
                total_models += 1

            # Determine N_frozen per layer = min(|self|, |task|) in that layer
            self_frozen_orig = {}  # layer_idx -> np.ndarray
            task_frozen_orig = {}

            for layer_idx in LAYER_INDICES:
                s_pool = self_pool[beh][layer_idx]
                t_pool = task_pool[beh][layer_idx]
                N      = min(len(s_pool), len(t_pool))

                rng_s = np.random.RandomState(FREEZE_SEED + 1000 * layer_idx + 31 * (mi + 1) + 1)
                rng_t = np.random.RandomState(FREEZE_SEED + 1000 * layer_idx + 31 * (mi + 1) + 2)

                pick_s = (rng_s.choice(s_pool, size=N, replace=False)
                          if N > 0 else np.array([], dtype=int))
                pick_t = (rng_t.choice(t_pool, size=N, replace=False)
                          if N > 0 else np.array([], dtype=int))

                self_frozen_orig[layer_idx] = np.asarray(pick_s, dtype=int)
                task_frozen_orig[layer_idx] = np.asarray(pick_t, dtype=int)

            # ── Write self freeze JSON ────────────────────────────────────────
            self_json = {
                "checkpoint": orig_path,
                "tau": TAU,
                "freeze_type": "self",
                "description": ("Neurons belonging to the largest coactivation block "
                                 "(self module) per layer. N = min(n_self, n_task) per layer."),
            }
            for layer_idx in LAYER_INDICES:
                s_pool  = self_pool[beh][layer_idx]
                t_pool  = task_pool[beh][layer_idx]
                frozen  = self_frozen_orig[layer_idx]
                self_json[f"layer_{layer_idx}"]             = frozen.tolist()
                self_json[f"layer_{layer_idx}_n_frozen"]    = int(len(frozen))
                self_json[f"layer_{layer_idx}_n_self_pool"] = int(len(s_pool))
                self_json[f"layer_{layer_idx}_n_task_pool"] = int(len(t_pool))
                self_json[f"layer_{layer_idx}_n_alive"]     = int(
                    len(orig_fixed["alive"][beh][layer_idx]))

            self_json_path = os.path.join(OUT_MODELS_DIR, stem + "_self_freeze_idx.json")
            with open(self_json_path, "w") as f:
                json.dump(self_json, f, indent=2)
            total_jsons += 1

            # ── Write task freeze JSON ────────────────────────────────────────
            task_json = {
                "checkpoint": orig_path,
                "tau": TAU,
                "freeze_type": "task",
                "description": ("Neurons belonging to all non-self coactivation blocks "
                                 "(task modules) per layer. N = min(n_self, n_task) per layer."),
            }
            for layer_idx in LAYER_INDICES:
                s_pool  = self_pool[beh][layer_idx]
                t_pool  = task_pool[beh][layer_idx]
                frozen  = task_frozen_orig[layer_idx]
                task_json[f"layer_{layer_idx}"]             = frozen.tolist()
                task_json[f"layer_{layer_idx}_n_frozen"]    = int(len(frozen))
                task_json[f"layer_{layer_idx}_n_self_pool"] = int(len(s_pool))
                task_json[f"layer_{layer_idx}_n_task_pool"] = int(len(t_pool))
                task_json[f"layer_{layer_idx}_n_alive"]     = int(
                    len(orig_fixed["alive"][beh][layer_idx]))

            task_json_path = os.path.join(OUT_MODELS_DIR, stem + "_task_freeze_idx.json")
            with open(task_json_path, "w") as f:
                json.dump(task_json, f, indent=2)
            total_jsons += 1

            print(f"  [{beh}] L0: self_pool={len(self_pool[beh][0])} "
                  f"task_pool={len(task_pool[beh][0])} "
                  f"N_frozen={len(self_frozen_orig[0])} | "
                  f"L1: self_pool={len(self_pool[beh][1])} "
                  f"task_pool={len(task_pool[beh][1])} "
                  f"N_frozen={len(self_frozen_orig[1])}")

        # ── Build layer_payloads for plotting (uses fixed alive mask) ─────────
        layer_payloads = {}
        for layer_idx in LAYER_INDICES:
            A_z_by  = {}; R_by   = {}
            alive_c = {}; dead_c = {}; total_c = {}
            for beh in model_order:
                acts_full = _get_layer_acts(state_dicts[beh], X_ref, layer_idx, actor_layers[beh])
                total_c[beh] = int(acts_full.shape[1])
                alive_idx_fixed = orig_fixed["alive"][beh][layer_idx]
                acts_use = (acts_full[:, alive_idx_fixed] if alive_idx_fixed.size
                            else acts_full[:, :0])
                alive_c[beh] = int(alive_idx_fixed.size)
                dead_c[beh]  = int(total_c[beh] - alive_idx_fixed.size)
                A_z = _zscore_cols(acts_use) if acts_use.size else acts_use
                A_z_by[beh] = A_z
                R_by[beh]   = corr_matrix(A_z)
            layer_payloads[layer_idx] = (A_z_by, R_by, alive_c, dead_c, total_c)

        # ── Generate 2 plots per model (self-frozen / task-frozen) ────────────
        for mi, beh in enumerate(model_order):
            orig_path = m_paths[beh]
            stem      = _model_stem(orig_path)

            # Recompute per-model frozen sets for plotting
            self_frozen_orig = {}
            task_frozen_orig = {}
            for layer_idx in LAYER_INDICES:
                s_pool = self_pool[beh][layer_idx]
                t_pool = task_pool[beh][layer_idx]
                N      = min(len(s_pool), len(t_pool))
                rng_s  = np.random.RandomState(FREEZE_SEED + 1000 * layer_idx + 31 * (mi + 1) + 1)
                rng_t  = np.random.RandomState(FREEZE_SEED + 1000 * layer_idx + 31 * (mi + 1) + 2)
                self_frozen_orig[layer_idx] = (
                    np.asarray(rng_s.choice(s_pool, size=N, replace=False), dtype=int)
                    if N > 0 else np.array([], dtype=int))
                task_frozen_orig[layer_idx] = (
                    np.asarray(rng_t.choice(t_pool, size=N, replace=False), dtype=int)
                    if N > 0 else np.array([], dtype=int))

            # Self-frozen plot
            png_self = os.path.join(OUT_GRAPHS_DIR, _sanitize_filename(stem + "_SelfFrozen") + ".png")
            _plot_triplet_model(
                model_name=beh,
                model_path=orig_path,
                freeze_type="self",
                layer_payloads=layer_payloads,
                orig_fixed_cache=orig_fixed,
                frozen_orig=self_frozen_orig,
                out_png_path=png_self,
            )
            total_plots += 1

            # Task-frozen plot
            png_task = os.path.join(OUT_GRAPHS_DIR, _sanitize_filename(stem + "_TaskFrozen") + ".png")
            _plot_triplet_model(
                model_name=beh,
                model_path=orig_path,
                freeze_type="task",
                layer_payloads=layer_payloads,
                orig_fixed_cache=orig_fixed,
                frozen_orig=task_frozen_orig,
                out_png_path=png_task,
            )
            total_plots += 1

        gc.collect()

    print(f"\n[DONE] models copied : {total_models}  → {OUT_MODELS_DIR}")
    print(f"[DONE] JSONs written  : {total_jsons}   (2 per model)")
    print(f"[DONE] PNGs written   : {total_plots}   (2 per model)")
    print(f"[DONE] Output root    : {OUT_ROOT}")


if __name__ == "__main__":
    main()