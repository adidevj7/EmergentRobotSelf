# ant_module_explorer_lesion_batch.py
# Runs Ant Actor "Self vs Task" module explorer + self/task lesion + PNG export
# for a batch of models arranged as (walk, jump, spin) triplets per cycle.
#
# Usage (server / CLI):
#   python Analysis_forIsaac/Lesion_and_Plotter_reint_BD_new.py
#/Users/adi/Desktop/FreshAntProject/Analysis_forIsaac/Lesion_and_Plotter_reint_BD.py
# Notes:
# - This script is written to be headless (matplotlib Agg) and saves PNGs to disk.
# - It expects MODELS contains 3 checkpoints per cycle: walk/jump/spin.

import os
import re
import gc
import math
import shutil
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import seaborn as sns
from matplotlib.colors import TwoSlopeNorm, BoundaryNorm
from matplotlib.patches import Rectangle

from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, fcluster, leaves_list
from scipy.spatial.distance import squareform
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee, connected_components


# ============================================================
# ========================= CONFIG ===========================
# ============================================================

MODELS: list[str] = [
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

# Layers: H1=0, H2=1
LAYER_INDICES = [0, 1]

SEED       = 42
LESION_SEED = 42

EPS        = 1e-8
MIN_STD    = 1e-5
ACTIVATION = "relu"  # {"elu","relu","tanh"}

ALIVE_ONLY_CLUSTERING = True
USE_FIXED_ORIG_ALIVE_AND_ORDER = True

# Correlation heatmap colors
CMAP               = "RdBu_r"
VMIN, VCENTER, VMAX = -1.0, 0.0, 1.0

# K selection (legacy; not used with block-diagonalisation)
K_MIN, K_MAX  = 2, 10
ALPHA_SMALL_K = 0.05

# =========================
# Block-diagonalisation config (NEW)
# =========================
# Keep edges where |cos| >= TAU, blocks are connected components, order = blocks by size + RCM inside.
TAU = 0.70
BD_MIN_BLOCK_SIZE = 1

DEVICE = "cpu"

# Reinit / lesion config
NUM_LESION = 40
LESION_NOISE_STD_FRAC = 0.15

DO_SELF_LESION = True
DO_TASK_LESION = True

# Output folders (explicit directory in-file)
OUT_ROOT = "/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/ModuleExplorerLesionReinitBatch_BD_newRobust"
OUT_MODELS_DIR = os.path.join(OUT_ROOT, "models_all")     # <- all checkpoints (original + lesions) go here
OUT_GRAPHS_DIR = os.path.join(OUT_ROOT, "graphs")

# Style
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

def _activation(x: np.ndarray, name: str) -> np.ndarray:
    if name == "elu":
        y = x.copy()
        neg = x <= 0
        y[neg] = np.expm1(x[neg])
        return y
    if name == "relu":
        return np.maximum(0.0, x)
    if name == "tanh":
        return np.tanh(x)
    raise ValueError(f"Unsupported ACTIVATION '{name}'")

def _zscore_cols(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if X.size == 0:
        return X.astype(np.float32, copy=False)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return ((X - mu) / (sd + eps)).astype(np.float32, copy=False)

def _dead_alive_indices(acts_post: np.ndarray, min_std: float):
    if acts_post.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    sd = acts_post.std(axis=0)
    dead_mask = sd <= min_std
    dead_idx = np.where(dead_mask)[0].astype(int)
    alive_idx = np.where(~dead_mask)[0].astype(int)
    return dead_idx, alive_idx

def corr_matrix(Xz):
    Xz = np.asarray(Xz, dtype=np.float32)
    if Xz.ndim != 2:
        raise ValueError("corr_matrix expects 2D array")
    n = Xz.shape[1]
    if n == 0:
        return np.zeros((0, 0), np.float32)
    if n == 1:
        return np.ones((1, 1), np.float32)

    # Cosine similarity between columns
    norms = np.linalg.norm(Xz, axis=0, keepdims=True).astype(np.float32, copy=False)
    norms = np.where(norms < EPS, 1.0, norms)
    sim = (Xz.T @ Xz) / (norms.T @ norms)
    sim = np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    np.fill_diagonal(sim, 1.0)
    return sim

def _within_between_stats(M, labels):
    if M.size == 0 or labels.size == 0:
        return dict(within=np.nan, between=np.nan, delta=np.nan)
    same = (labels[:, None] == labels[None, :])
    np.fill_diagonal(same, False)
    within = M[same]
    between = M[~same]
    w = float(within.mean()) if within.size else np.nan
    b = float(between.mean()) if between.size else np.nan
    return dict(within=w, between=b, delta=(w - b) if np.isfinite(w) and np.isfinite(b) else np.nan)

def _modularity_Q_pos(R_abs, labels):
    A = np.clip(R_abs, 0.0, None).astype(np.float64)
    np.fill_diagonal(A, 0.0)
    m = A.sum() / 2.0
    if m <= 0:
        return np.nan
    k = A.sum(axis=1)
    Q = 0.0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if labels[i] == labels[j]:
                Q += A[i, j] - (k[i]*k[j] / (2.0*m))
    return float(Q / (2.0*m))

def _norm01(a):
    a = np.array(a, dtype=np.float64)
    if np.all(np.isnan(a)):
        return np.zeros_like(a)
    lo, hi = np.nanmin(a), np.nanmax(a)
    return np.zeros_like(a) if hi - lo < 1e-12 else (a - lo) / (hi - lo)

def _cluster_order_and_labels_abs(R_abs, K):
    n = int(R_abs.shape[0])
    if n == 0:
        return np.array([], int), np.array([], int)
    if n < 3:
        order = np.arange(n, dtype=int)
        labels = np.ones(n, dtype=int)
        return order, labels
    D = 1.0 - R_abs
    np.fill_diagonal(D, 0.0)
    y = squareform(D, checks=False)
    Z = linkage(y, method="average")
    Z = optimal_leaf_ordering(Z, y)
    order = leaves_list(Z)
    labels = fcluster(Z, t=K, criterion="maxclust")
    return order, labels

def choose_K_and_order_abs(R_abs, k_min=2, k_max=10, alpha_small_k=0.05):
    n = int(R_abs.shape[0])
    if n == 0:
        return 2, np.array([], int), np.array([], int)
    if n < 3:
        order = np.arange(n, dtype=int)
        labels = np.ones(n, dtype=int)
        return 1, order, labels

    Ks = list(range(max(2, k_min), max(k_min, k_max) + 1))
    deltas, Qs, orders, labels_store = [], [], {}, {}
    for K in Ks:
        order, labels = _cluster_order_and_labels_abs(R_abs, K)
        orders[K] = order
        labels_store[K] = labels
        st = _within_between_stats(R_abs, labels)
        deltas.append(st["delta"])
        Qs.append(_modularity_Q_pos(R_abs, labels))
    dN = _norm01(deltas)
    qN = _norm01(Qs)
    kN = (np.array(Ks) - min(Ks)) / (max(Ks) - min(Ks) + 1e-12)
    score = 0.6*dN + 0.4*qN - alpha_small_k*kN
    K_best = Ks[int(np.nanargmax(score))]
    return K_best, orders[K_best], labels_store[K_best]

def _order_by_cluster_size(labels, base_order=None):
    labels = np.asarray(labels)
    n = labels.size
    if n == 0:
        return np.arange(0, dtype=int)
    if base_order is None:
        base_order = np.arange(n, dtype=int)
    base_order = np.asarray(base_order)
    unique_cids = np.unique(labels)
    sizes = [int(np.sum(labels == cid)) for cid in unique_cids]
    sorted_cids = [cid for cid, _ in sorted(zip(unique_cids, sizes), key=lambda t: -t[1])]
    new_order_list = []
    labels_in_base = labels[base_order]
    for cid in sorted_cids:
        mask = (labels_in_base == cid)
        new_order_list.append(base_order[mask])
    new_order = np.concatenate(new_order_list) if new_order_list else np.arange(n, dtype=int)
    return new_order

def _relabel_clusters_by_size(labels):
    labels = np.asarray(labels)
    if labels.size == 0:
        return labels.astype(int)
    unique_cids, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)
    sorted_cids = unique_cids[order]
    cid_to_rank = {cid: rank + 1 for rank, cid in enumerate(sorted_cids)}
    new_labels = np.array([cid_to_rank[c] for c in labels], dtype=int)
    return new_labels

def _cluster_bounds(labels, order):
    lr = labels[order]
    return [i - 0.5 for i in range(1, len(lr)) if lr[i] != lr[i-1]]

def _cluster_spans_from_ordered_labels(labels_ordered: np.ndarray):
    labels_ordered = np.asarray(labels_ordered)
    spans = {}
    if labels_ordered.size == 0:
        return spans
    start = 0
    cur = int(labels_ordered[0])
    for i in range(1, labels_ordered.size):
        cid = int(labels_ordered[i])
        if cid != cur:
            spans.setdefault(cur, []).append((start, i - 1))
            cur = cid
            start = i
    spans.setdefault(cur, []).append((start, labels_ordered.size - 1))
    return spans

def hungarian_match_cosine(A, B, eps=1e-8):
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)

    if A.size == 0 or B.size == 0:
        sim = np.zeros((A.shape[1] if A.ndim == 2 else 0, B.shape[1] if B.ndim == 2 else 0), dtype=np.float32)
        return sim, np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)

    nA = np.linalg.norm(A, axis=0, keepdims=True)
    nB = np.linalg.norm(B, axis=0, keepdims=True)
    nA[nA < eps] = 1.0
    nB[nB < eps] = 1.0

    sim = (A.T @ B) / (nA.T @ nB)
    sim = np.nan_to_num(sim)

    i_idx, j_idx, = linear_sum_assignment(-sim)
    matched_vals = sim[i_idx, j_idx] if i_idx.size else np.array([], dtype=float)
    return sim, i_idx, j_idx, matched_vals

def _safe_corr(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size != b.size or a.size == 0:
        return 0.0

    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < EPS or nb < EPS:
        return 0.0

    c = float(np.dot(a, b) / (na * nb))
    if np.isnan(c):
        return 0.0
    return float(abs(c))

def _torch_load_compat(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

def _load_rlg_forplay_state_dict(path: str) -> dict:
    payload = _torch_load_compat(path)
    if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        sd = {}
        for k, v in payload["model"].items():
            sd[k] = v.detach().cpu().clone() if torch.is_tensor(v) else v
        return sd
    raise ValueError(f"[ckpt] {path} missing 'model' dict (for-play).")

def _discover_actor_mlp_layers(state_dict: dict):
    candidates = []
    for k in state_dict.keys():
        if k.endswith(".weight") and ("actor_mlp" in k or ".actor." in k or "actor_net" in k or "actor.trunk" in k):
            parts = k.split(".")
            try:
                idx = int(parts[-2])
                bias_key = k[:-6] + "bias"
                if bias_key in state_dict:
                    candidates.append((idx, k, bias_key))
            except Exception:
                candidates.append((10**6, k, k[:-6] + "bias"))
    candidates.sort(key=lambda t: (t[0], t[1]))
    return candidates

def _first_linear_in(actor_layers, sd):
    if not actor_layers:
        raise RuntimeError("No actor MLP layers found.")
    _, wkey, _ = actor_layers[0]
    W = sd[wkey]
    if isinstance(W, torch.Tensor):
        W = W.detach().cpu().numpy()
    if W.ndim != 2:
        raise RuntimeError(f"Unexpected weight shape for {wkey}: {tuple(W.shape)}")
    return W.shape[1], W.shape[0]  # in_dim, out_dim

def _actor_hidden_forward_numpy(X, actor_layers, sd, activation_name):
    outs = []
    h = X
    for _, wkey, bkey in actor_layers:
        W = sd[wkey]
        if isinstance(W, torch.Tensor):
            W = W.detach().cpu().numpy()
        b = sd[bkey]
        if isinstance(b, torch.Tensor):
            b = b.detach().cpu().numpy()
        h = h @ W.T + b[None, :]
        h = _activation(h, activation_name)
        outs.append(h.astype(np.float32, copy=False))
    return outs

def get_layer_output_batch_from_ckpt(sd, inputs: torch.Tensor, layer_idx: int, actor_layers, activation_name):
    X = inputs.detach().cpu().numpy().astype(np.float32, copy=False)
    outs = _actor_hidden_forward_numpy(X, actor_layers, sd, activation_name)
    if layer_idx < 0 or layer_idx >= len(outs):
        raise IndexError(f"layer_idx={layer_idx} out of range (0..{len(outs)-1})")
    return outs[layer_idx]

def _as_numpy_2d(W):
    if isinstance(W, torch.Tensor):
        W = W.detach().cpu().numpy()
    W = np.asarray(W)
    if W.ndim != 2:
        raise RuntimeError(f"Expected 2D weight matrix, got shape {W.shape}")
    return W

def _overlay_module_boxes(ax, labels_ord, cluster_cmap, cluster_norm, alpha_fill=0.52, edge_alpha=0.95):
    spans = _cluster_spans_from_ordered_labels(labels_ord)
    for cid, segs in spans.items():
        color = cluster_cmap(cluster_norm(int(cid)))
        for (a, b) in segs:
            w = (b - a + 1)
            ax.add_patch(Rectangle(
                (a - 0.5, a - 0.5), w, w,
                facecolor=color,
                edgecolor=(color[0], color[1], color[2], edge_alpha),
                linewidth=2.1,
                alpha=alpha_fill,
                zorder=3
            ))

def _force_axes_square_by_width(ax):
    pos = ax.get_position()
    cx = pos.x0 + pos.width * 0.5
    cy = pos.y0 + pos.height * 0.5
    side = pos.width
    new_x0 = cx - side * 0.5
    new_y0 = cy - side * 0.5
    ax.set_position([new_x0, new_y0, side, side])

def _parse_title_fields_from_path(path: str):
    p = str(path).replace("\\", "/")
    run_dir = os.path.basename(os.path.dirname(os.path.dirname(p)))
    fname = os.path.splitext(os.path.basename(p))[0]

    m_run = re.search(r"(?:^|_)relu_(\d+)(?:_|$)", run_dir)
    run_num = int(m_run.group(1)) if m_run else None
    if run_num is None:
        m_run2 = re.search(r"(?:^|_)run(\d+)(?:_|$)", fname.lower())
        run_num = int(m_run2.group(1)) if m_run2 else None

    m = re.search(r"c(\d+)_b(\d+)_([A-Za-z]+)", fname)
    cyc = int(m.group(1)) if m else None
    beh = int(m.group(2)) if m else None
    beh_name = (m.group(3).capitalize() if m else None)

    return run_num, cyc, beh, beh_name

def _title_text_from_path(path: str, group_title: str) -> str:
    run_num, cyc, beh, beh_name = _parse_title_fields_from_path(path)
    run_str = f"{run_num}" if run_num is not None else "?"
    cyc_str = f"{cyc:03d}" if cyc is not None else "???"
    beh_str = f"{beh:02d}" if beh is not None else "??"
    beh_name = beh_name if beh_name is not None else "?"

    base = f"WSJ - Run {run_str} - Cycle {cyc_str} - Behavior - {beh_str} - {beh_name}"

    up = group_title.upper()
    if "ORIGINAL" in up:
        return base + " - Original"
    if "SELF-LESION" in up:
        return base + f" - Self Lesion ({NUM_LESION}n)"
    if "TASK-LESION" in up:
        return base + f" - Task Lesion ({NUM_LESION}n)"
    return base

def _sanitize_filename(s: str, max_len: int = 220) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\-\.\(\)\[\] ]+", "", s)
    s = s.replace(" ", "_")
    if len(s) > max_len:
        s = s[:max_len]
    return s

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _extract_plateau_timestamp_from_path(path: str) -> str:
    fname = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"plateau_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", fname)
    if m:
        return m.group(1)
    # fallback: try any datetime-like blob
    m2 = re.search(r"(\d{4}-\d{2}-\d{2}[_-]\d{2}-\d{2}-\d{2})", fname)
    return m2.group(1).replace("-", "-").replace("_", "_") if m2 else "unknown"

def _build_out_model_path(orig_path: str, tag: str, dst_dir: str) -> str:
    run_num, cyc, beh, beh_name = _parse_title_fields_from_path(orig_path)
    beh_lower = (beh_name.lower() if beh_name is not None else None)
    if beh_lower is None:
        low = os.path.basename(orig_path).lower()
        if "_walk_" in low: beh_lower = "walk"
        elif "_jump_" in low: beh_lower = "jump"
        elif "_spin_" in low: beh_lower = "spin"
        else: beh_lower = "unknown"

    ts = _extract_plateau_timestamp_from_path(orig_path)

    run_str = f"{int(run_num):02d}" if run_num is not None else "??"
    cyc_str = f"{int(cyc):03d}" if cyc is not None else "???"
    beh_str = f"{int(beh):02d}" if beh is not None else "??"

    # Keep cXXX_bYY_behavior substring for existing regexes, and keep plateau timestamp.
    base = f"run{run_str}_c{cyc_str}_b{beh_str}_{beh_lower}_{tag}_plateau_{ts}_for_play.pth"
    _ensure_dir(dst_dir)
    return os.path.join(dst_dir, base)

def _copy_model_to_out(src_path: str, dst_dir: str, tag: str):
    dst = _build_out_model_path(src_path, tag=tag, dst_dir=dst_dir)
    if not os.path.exists(dst):
        shutil.copy2(src_path, dst)
    return dst

# ============================================================
# ===================== BLOCK DIAGONALISATION =================
# ============================================================

def blockdiag_rcm_blocks_from_abs(R_abs: np.ndarray, tau: float, min_block_size: int):
    """
    Keep edges where |cos|>=tau, blocks={connected components}, order=blocks by size + RCM inside.
    Returns:
      order (0-based), labels (1-based)
    """
    R_abs = np.asarray(R_abs, dtype=np.float32)
    n = int(R_abs.shape[0])
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    if n < 3:
        return np.arange(n, dtype=int), np.ones(n, dtype=int)

    R_abs = 0.5 * (R_abs + R_abs.T)
    R_abs = np.clip(R_abs, 0.0, 1.0)
    np.fill_diagonal(R_abs, 0.0)

    tau = float(max(0.0, min(1.0, tau)))
    A = (R_abs >= tau).astype(np.int8)
    np.fill_diagonal(A, 0)
    G = csr_matrix(A)

    n_comp, comp0 = connected_components(G, directed=False, connection="weak")  # 0-based
    comp = comp0.copy()

    if min_block_size is not None and int(min_block_size) > 1:
        sizes = np.bincount(comp, minlength=int(n_comp))
        small = np.where(sizes < int(min_block_size))[0]
        if small.size:
            comp[np.isin(comp, small)] = -1
            keep = [c for c in np.unique(comp) if c != -1]
            remap = {c: i for i, c in enumerate(keep)}
            comp2 = np.empty(n, dtype=int)
            for i in range(n):
                c = comp[i]
                comp2[i] = remap[c] if c != -1 else len(keep)
            comp = comp2
            n_comp = int(comp.max()) + 1

    sizes = [(c, int(np.sum(comp == c))) for c in range(int(n_comp))]
    sizes.sort(key=lambda t: -t[1])

    parts = []
    for c, _ in sizes:
        idxs = np.where(comp == c)[0]
        if idxs.size <= 2:
            parts.append(idxs)
        else:
            subG = G[idxs][:, idxs]
            sub_ord = reverse_cuthill_mckee(subG, symmetric_mode=True)
            parts.append(idxs[np.asarray(sub_ord, dtype=int)])

    order = np.concatenate(parts).astype(int)
    labels = (comp + 1).astype(int)
    return order, labels


# ============================================================
# ========================= GROUPING =========================
# ============================================================

def _behavior_key_from_path(path: str) -> str:
    low = os.path.basename(path).lower()
    if "_walk_" in low:
        return "walk"
    if "_jump_" in low:
        return "jump"
    if "_spin_" in low:
        return "spin"
    # fallback: try parse cXXX_bYY_name
    m = re.search(r"_([a-zA-Z]+)_plateau", low)
    if m:
        name = m.group(1).lower()
        if name in ("walk", "jump", "spin"):
            return name
    raise RuntimeError(f"Could not infer behavior (walk/jump/spin) from filename: {path}")

def _cycle_key_from_path(path: str):
    # group by (run_dir, cycle_id)
    p = str(path).replace("\\", "/")
    run_dir = os.path.basename(os.path.dirname(os.path.dirname(p)))
    fname = os.path.splitext(os.path.basename(p))[0]
    m = re.search(r"c(\d+)_b(\d+)_", fname)
    if not m:
        raise RuntimeError(f"Could not parse cycle/behavior from: {path}")
    cyc = int(m.group(1))
    return (run_dir, cyc)

def group_into_cycles(models: list[str]):
    groups = {}
    for p in models:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing model: {p}")
        k = _cycle_key_from_path(p)
        beh = _behavior_key_from_path(p)
        groups.setdefault(k, {})
        if beh in groups[k]:
            raise RuntimeError(f"Duplicate behavior '{beh}' for cycle group {k}: {p}")
        groups[k][beh] = p

    # enforce complete triplets
    out = []
    for k, mp in sorted(groups.items(), key=lambda t: (t[0][0], t[0][1])):
        missing = [b for b in ("walk", "jump", "spin") if b not in mp]
        if missing:
            raise RuntimeError(f"Cycle group {k} missing behaviors: {missing}. Has keys={list(mp.keys())}")
        out.append((k, {"walk": mp["walk"], "jump": mp["jump"], "spin": mp["spin"]}))
    return out


# ============================================================
# ====================== ANALYSIS CORE =======================
# ============================================================

def _compute_families_and_scores(model_order, A_z_by_model: dict):
    fam_idx_by_model = {m: np.array([], dtype=int) for m in model_order}
    activation_stab = np.array([], dtype=np.float32)
    connectivity_stab = np.array([], dtype=np.float32)
    self_score = np.array([], dtype=np.float32)

    A_ref = A_z_by_model["walk"]
    n_ref_alive = int(A_ref.shape[1])
    if n_ref_alive <= 0:
        return fam_idx_by_model, activation_stab, connectivity_stab, self_score
    if A_z_by_model["jump"].shape[1] <= 0 or A_z_by_model["spin"].shape[1] <= 0:
        return fam_idx_by_model, activation_stab, connectivity_stab, self_score

    maps = {}
    for other in ("jump", "spin"):
        _, i_idx, j_idx, _ = hungarian_match_cosine(A_ref, A_z_by_model[other])
        idx_map = np.full(n_ref_alive, -1, dtype=int)
        if i_idx.size:
            idx_map[i_idx] = j_idx
        maps[other] = idx_map

    good_ref = np.where((maps["jump"] >= 0) & (maps["spin"] >= 0))[0].astype(int)
    if good_ref.size <= 0:
        return fam_idx_by_model, activation_stab, connectivity_stab, self_score

    fam_idx_by_model = {
        "walk": good_ref,
        "jump": maps["jump"][good_ref].astype(int),
        "spin": maps["spin"][good_ref].astype(int),
    }
    n_fam = int(good_ref.size)

    W_by_model = {m: A_z_by_model[m][:, fam_idx_by_model[m]] for m in model_order}
    R_fam_by_model = {m: corr_matrix(W_by_model[m]) for m in model_order}

    pairings = [("walk", "jump"), ("walk", "spin"), ("jump", "spin")]
    activation_stab = np.zeros(n_fam, dtype=np.float32)
    connectivity_stab = np.zeros(n_fam, dtype=np.float32)

    for k in range(n_fam):
        act_corrs = [_safe_corr(W_by_model[a][:, k], W_by_model[b][:, k]) for (a, b) in pairings]
        activation_stab[k] = float(np.mean(act_corrs))
        conn_corrs = [_safe_corr(R_fam_by_model[a][k, :], R_fam_by_model[b][k, :]) for (a, b) in pairings]
        connectivity_stab[k] = float(np.mean(conn_corrs))

    self_score = 0.5 * (activation_stab + connectivity_stab)
    return fam_idx_by_model, activation_stab, connectivity_stab, self_score.astype(np.float32, copy=False)

def _plot_one_model_two_layers_to_file(group_title: str,
                                       model_order,
                                       model_name: str,
                                       model_path: str,
                                       layer_payloads: dict,
                                       orig_fixed_cache: dict,
                                       out_png_path: str):
    (A0_by_model, R0_by_model, alive0, dead0, total0) = layer_payloads[0]
    (A1_by_model, R1_by_model, alive1, dead1, total1) = layer_payloads[1]

    fam0_by_model, act0, conn0, self0 = _compute_families_and_scores(model_order, A0_by_model)
    fam1_by_model, act1, conn1, self1 = _compute_families_and_scores(model_order, A1_by_model)

    labels0 = np.asarray(orig_fixed_cache["ORIG_LABELS_ALIVE"][model_name][0], dtype=int)
    order0  = np.asarray(orig_fixed_cache["ORIG_ORDER_ALIVE"][model_name][0], dtype=int)
    labels1 = np.asarray(orig_fixed_cache["ORIG_LABELS_ALIVE"][model_name][1], dtype=int)
    order1  = np.asarray(orig_fixed_cache["ORIG_ORDER_ALIVE"][model_name][1], dtype=int)

    R0 = np.asarray(R0_by_model[model_name])
    R1 = np.asarray(R1_by_model[model_name])
    R0_plot = R0[np.ix_(order0, order0)] if R0.size and order0.size else R0
    R1_plot = R1[np.ix_(order1, order1)] if R1.size and order1.size else R1

    labels0_ord = labels0[order0] if labels0.size and order0.size else labels0
    labels1_ord = labels1[order1] if labels1.size and order1.size else labels1

    max_cluster_id = int(max(labels0.max() if labels0.size else 0, labels1.max() if labels1.size else 0))
    max_cluster_id = max(1, max_cluster_id)
    cluster_cmap = plt.get_cmap("tab20", max_cluster_id)
    cluster_norm = BoundaryNorm(np.arange(0.5, max_cluster_id + 1.5, 1.0), max_cluster_id)

    def _ordered_scores(n_units, fam_idx, score, order):
        out = np.full(n_units, np.nan, dtype=np.float32)
        fam_idx = np.asarray(fam_idx, dtype=int)
        if fam_idx.size and score.size == fam_idx.size:
            out[fam_idx] = score.astype(np.float32, copy=False)
        return out[order] if order.size else out

    self0_ord = _ordered_scores(int(R0.shape[0]), fam0_by_model.get(model_name, np.array([], int)), self0, order0)
    self1_ord = _ordered_scores(int(R1.shape[0]), fam1_by_model.get(model_name, np.array([], int)), self1, order1)

    suptitle = _title_text_from_path(model_path, group_title)

    fig = plt.figure(figsize=(15.5, 10.6))
    gs = fig.add_gridspec(
        nrows=3, ncols=2,
        width_ratios=[1.0, 1.0],
        height_ratios=[0.06, 1.0, 0.34],
        wspace=0.10, hspace=0.08
    )

    ax_top0 = fig.add_subplot(gs[0, 0])
    ax_top1 = fig.add_subplot(gs[0, 1])
    ax_hm0  = fig.add_subplot(gs[1, 0])
    ax_hm1  = fig.add_subplot(gs[1, 1])
    ax_sc0  = fig.add_subplot(gs[2, 0])
    ax_sc1  = fig.add_subplot(gs[2, 1])

    if labels0_ord.size:
        sns.heatmap(labels0_ord[None, :], ax=ax_top0, cmap=cluster_cmap, norm=cluster_norm,
                    cbar=False, xticklabels=False, yticklabels=False)
    else:
        ax_top0.axis("off")

    if labels1_ord.size:
        sns.heatmap(labels1_ord[None, :], ax=ax_top1, cmap=cluster_cmap, norm=cluster_norm,
                    cbar=False, xticklabels=False, yticklabels=False)
    else:
        ax_top1.axis("off")

    if R0_plot.size:
        sns.heatmap(
            R0_plot, ax=ax_hm0,
            cmap=CMAP, vmin=VMIN, vmax=VMAX, center=VCENTER,
            square=False, cbar=False,
            xticklabels=False, yticklabels=False
        )
    else:
        ax_hm0.text(0.5, 0.5, "Empty", ha="center", va="center")
        ax_hm0.set_xticks([]); ax_hm0.set_yticks([])

    if R1_plot.size:
        sns.heatmap(
            R1_plot, ax=ax_hm1,
            cmap=CMAP, vmin=VMIN, vmax=VMAX, center=VCENTER,
            square=False, cbar=False,
            xticklabels=False, yticklabels=False
        )
    else:
        ax_hm1.text(0.5, 0.5, "Empty", ha="center", va="center")
        ax_hm1.set_xticks([]); ax_hm1.set_yticks([])

    if labels0_ord.size:
        _overlay_module_boxes(ax_hm0, labels0_ord, cluster_cmap, cluster_norm, alpha_fill=0.52, edge_alpha=0.95)
        for b in _cluster_bounds(labels0, order0) if labels0.size and order0.size else []:
            ax_hm0.axhline(b, color="k", lw=0.9, zorder=5)
            ax_hm0.axvline(b, color="k", lw=0.9, zorder=5)

    if labels1_ord.size:
        _overlay_module_boxes(ax_hm1, labels1_ord, cluster_cmap, cluster_norm, alpha_fill=0.52, edge_alpha=0.95)
        for b in _cluster_bounds(labels1, order1) if labels1.size and order1.size else []:
            ax_hm1.axhline(b, color="k", lw=0.9, zorder=5)
            ax_hm1.axvline(b, color="k", lw=0.9, zorder=5)

    ax_hm0.set_title("")
    ax_hm1.set_title("")

    n_alive0 = int(alive0.get(model_name, R0.shape[0]))
    n_tot0   = int(total0.get(model_name, n_alive0 + int(dead0.get(model_name, 0))))
    n_alive1 = int(alive1.get(model_name, R1.shape[0]))
    n_tot1   = int(total1.get(model_name, n_alive1 + int(dead1.get(model_name, 0))))

    def _plot_self(ax, y_ord, labels_ord, xlabel, show_ylabel, hide_yticks=False):
        ax.set_ylim(0.0, 1.05)
        ax.axhline(1.0, ls="--", lw=1.2, alpha=0.5)
        ax.set_xticks([])
        ax.set_xlabel(xlabel, fontsize=18, fontweight="bold", labelpad=6)

        if show_ylabel:
            ax.set_ylabel("Self - Score", fontsize=16)
        else:
            ax.set_ylabel("")

        if hide_yticks:
            ax.set_yticks([])
            ax.set_yticklabels([])
        else:
            ax.set_yticks([0.0, 0.5, 1.0])
            ax.set_yticklabels(["0", "0.5", "1"])

        if y_ord.size:
            x = np.arange(y_ord.size)
            mask = np.isfinite(y_ord)
            if np.any(mask):
                sns.scatterplot(x=x[mask], y=y_ord[mask], ax=ax, s=18, alpha=0.35, edgecolor=None)
            else:
                ax.text(0.5, 0.5, "No families", ha="center", va="center", transform=ax.transAxes)

            if labels_ord.size:
                for cid in np.unique(labels_ord):
                    idxs = np.where(labels_ord == cid)[0]
                    if idxs.size:
                        yvals = y_ord[idxs]
                        yvals = yvals[np.isfinite(yvals)]
                        if yvals.size:
                            mean_c = float(np.mean(yvals))
                            ax.hlines(mean_c, idxs.min() - 0.5, idxs.max() + 0.5, linewidth=3.0)

                # NEW: self vs task module mean lines (self = cluster 1; task = all others)
                self_mask = (labels_ord == 1) & np.isfinite(y_ord)
                task_mask = (labels_ord != 1) & np.isfinite(y_ord)

                if np.any(self_mask):
                    self_mean = float(np.mean(y_ord[self_mask]))
                    ax.axhline(self_mean, color="k", lw=2.4, alpha=0.35, zorder=6)
                    ax.text(0.01, self_mean + 0.01, "self mean",
                            transform=ax.get_yaxis_transform(), ha="left", va="bottom",
                            fontsize=9, color="k", alpha=0.6)

                if np.any(task_mask):
                    task_mean = float(np.mean(y_ord[task_mask]))
                    ax.axhline(task_mean, color="r", lw=2.4, alpha=0.28, zorder=6)
                    ax.text(0.01, task_mean + 0.01, "task mean",
                            transform=ax.get_yaxis_transform(), ha="left", va="bottom",
                            fontsize=9, color="r", alpha=0.55)

            ax.set_xlim(-0.5, len(y_ord) - 0.5)
        else:
            ax.text(0.5, 0.5, "Empty", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlim(-0.5, 1.0)

    _plot_self(ax_sc0, self0_ord, labels0_ord, xlabel="Layer 1", show_ylabel=True, hide_yticks=False)
    _plot_self(ax_sc1, self1_ord, labels1_ord, xlabel="Layer 2", show_ylabel=False, hide_yticks=True)

    fig.canvas.draw()
    _force_axes_square_by_width(ax_hm0)
    _force_axes_square_by_width(ax_hm1)

    fig.canvas.draw()
    pos_hm0 = ax_hm0.get_position()
    pos_hm1 = ax_hm1.get_position()

    def _match_x(ax, pos_ref):
        pos = ax.get_position()
        ax.set_position([pos_ref.x0, pos.y0, pos_ref.width, pos.height])

    _match_x(ax_top0, pos_hm0)
    _match_x(ax_sc0,  pos_hm0)
    _match_x(ax_top1, pos_hm1)
    _match_x(ax_sc1,  pos_hm1)

    ax_sc0.text(0.5, -0.155, f"(alive {n_alive0}/{n_tot0})", transform=ax_sc0.transAxes,
                ha="center", va="top", fontsize=10)
    ax_sc1.text(0.5, -0.155, f"(alive {n_alive1}/{n_tot1})", transform=ax_sc1.transAxes,
                ha="center", va="top", fontsize=10)

    if R0_plot.size:
        import matplotlib as mpl
        norm = TwoSlopeNorm(vmin=VMIN, vcenter=VCENTER, vmax=VMAX)
        sm = mpl.cm.ScalarMappable(cmap=plt.get_cmap(CMAP), norm=norm)
        sm.set_array([])

        cb_w_base = 0.012
        cb_w = cb_w_base * 1.2
        cb_h = pos_hm0.height * (1.6)
        cb_y = pos_hm0.y0 + (pos_hm0.height - cb_h) * 0.50 + 0.02
        cb_x = pos_hm0.x0 - 5.9 * cb_w
        cax = fig.add_axes([cb_x, cb_y, cb_w, cb_h])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        cb.ax.tick_params(labelsize=10)

    fig.suptitle(suptitle, fontsize=16, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.95, bottom=0.06)

    _ensure_dir(os.path.dirname(out_png_path))
    fig.savefig(out_png_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

def _save_lesioned_copy(orig_path: str, new_path: str, actor_layers, pool_h1: np.ndarray, pool_h2: np.ndarray,
                        lesion_h1: np.ndarray, lesion_h2: np.ndarray, rng: np.random.RandomState):
    payload = _torch_load_compat(orig_path)
    if not (isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict)):
        raise ValueError(f"[ckpt] {orig_path} missing 'model' dict (for-play).")
    model_sd = payload["model"]

    out_idx = max(LAYER_INDICES) + 1
    if out_idx >= len(actor_layers):
        raise RuntimeError(f"actor_layers too short: need out_idx={out_idx}, len={len(actor_layers)}. actor_layers={actor_layers}")

    wkey_in_h1  = actor_layers[0][1]
    bkey_in_h1  = actor_layers[0][2]
    wkey_h1_h2  = actor_layers[1][1]
    bkey_h1_h2  = actor_layers[1][2]
    wkey_h2_out = actor_layers[out_idx][1]

    W0 = _as_numpy_2d(model_sd[wkey_in_h1])    # (H1, obs)
    W1 = _as_numpy_2d(model_sd[wkey_h1_h2])    # (H2, H1)
    W2 = _as_numpy_2d(model_sd[wkey_h2_out])   # (out, H2)

    b0 = model_sd[bkey_in_h1]
    if isinstance(b0, torch.Tensor):
        b0 = b0.detach().cpu().numpy()
    b0 = np.asarray(b0).reshape(-1)

    b1 = model_sd[bkey_h1_h2]
    if isinstance(b1, torch.Tensor):
        b1 = b1.detach().cpu().numpy()
    b1 = np.asarray(b1).reshape(-1)

    lesion_h1 = np.asarray(lesion_h1, dtype=int)
    lesion_h2 = np.asarray(lesion_h2, dtype=int)

    lesion_h1 = lesion_h1[(lesion_h1 >= 0) & (lesion_h1 < W0.shape[0])]
    lesion_h2 = lesion_h2[(lesion_h2 >= 0) & (lesion_h2 < W1.shape[0])]

    lesion_h1_cols = lesion_h1[(lesion_h1 >= 0) & (lesion_h1 < W1.shape[1])]
    lesion_h2_cols = lesion_h2[(lesion_h2 >= 0) & (lesion_h2 < W2.shape[1])]

    W0_new = W0.copy()
    W1_new = W1.copy()
    W2_new = W2.copy()
    b0_new = b0.copy()
    b1_new = b1.copy()

    mu_W0, sd_W0 = float(W0.mean()), float(W0.std())
    mu_W1, sd_W1 = float(W1.mean()), float(W1.std())
    mu_W2, sd_W2 = float(W2.mean()), float(W2.std())
    mu_b0, sd_b0 = float(b0.mean()), float(b0.std())
    mu_b1, sd_b1 = float(b1.mean()), float(b1.std())

    if sd_W0 < 1e-12: sd_W0 = 1.0
    if sd_W1 < 1e-12: sd_W1 = 1.0
    if sd_W2 < 1e-12: sd_W2 = 1.0
    if sd_b0 < 1e-12: sd_b0 = 1.0
    if sd_b1 < 1e-12: sd_b1 = 1.0

    # Reinitialize H1 neurons: incoming row in W0, bias b0, outgoing column(s) in W1
    if lesion_h1.size:
        W0_new[lesion_h1, :] = rng.normal(mu_W0, sd_W0, size=(lesion_h1.size, W0_new.shape[1])).astype(W0_new.dtype, copy=False)
        if b0_new.size:
            b0_new[lesion_h1] = rng.normal(mu_b0, sd_b0, size=(lesion_h1.size,)).astype(b0_new.dtype, copy=False)
        if lesion_h1_cols.size:
            W1_new[:, lesion_h1_cols] = rng.normal(mu_W1, sd_W1, size=(W1_new.shape[0], lesion_h1_cols.size)).astype(W1_new.dtype, copy=False)

    # Reinitialize H2 neurons: incoming row(s) in W1, bias b1, outgoing column(s) in W2
    if lesion_h2.size:
        W1_new[lesion_h2, :] = rng.normal(mu_W1, sd_W1, size=(lesion_h2.size, W1_new.shape[1])).astype(W1_new.dtype, copy=False)
        if b1_new.size:
            b1_new[lesion_h2] = rng.normal(mu_b1, sd_b1, size=(lesion_h2.size,)).astype(b1_new.dtype, copy=False)
        if lesion_h2_cols.size:
            W2_new[:, lesion_h2_cols] = rng.normal(mu_W2, sd_W2, size=(W2_new.shape[0], lesion_h2_cols.size)).astype(W2_new.dtype, copy=False)

    model_sd[wkey_in_h1]  = torch.as_tensor(W0_new, dtype=model_sd[wkey_in_h1].dtype)
    model_sd[bkey_in_h1]  = torch.as_tensor(b0_new, dtype=model_sd[bkey_in_h1].dtype)
    model_sd[wkey_h1_h2]  = torch.as_tensor(W1_new, dtype=model_sd[wkey_h1_h2].dtype)
    model_sd[bkey_h1_h2]  = torch.as_tensor(b1_new, dtype=model_sd[bkey_h1_h2].dtype)
    model_sd[wkey_h2_out] = torch.as_tensor(W2_new, dtype=model_sd[wkey_h2_out].dtype)

    payload["model"] = model_sd
    _ensure_dir(os.path.dirname(new_path))
    torch.save(payload, new_path)


# ============================================================
# =========================== MAIN ===========================
# ============================================================

def main():
    _seed_everything(SEED)

    if not os.path.exists(ALL_STATES_PATH):
        raise FileNotFoundError(f"Missing ALL_STATES_PATH: {ALL_STATES_PATH}")

    _ensure_dir(OUT_MODELS_DIR)
    _ensure_dir(OUT_GRAPHS_DIR)

    cycles = group_into_cycles(MODELS)
    print(f"[INFO] Found {len(cycles)} cycle triplets ({len(MODELS)} models total).")
    print(f"[INFO] Saving all checkpoints into: {OUT_MODELS_DIR}")

    all_states = np.load(ALL_STATES_PATH)
    if all_states.ndim != 2:
        raise ValueError(f"{ALL_STATES_PATH} has shape {all_states.shape}, expected 2D (*,obs_dim)")

    first_cycle_paths = cycles[0][1]
    sd0 = _load_rlg_forplay_state_dict(first_cycle_paths["walk"])
    layers0 = _discover_actor_mlp_layers(sd0)
    if not layers0:
        raise RuntimeError("No actor MLP layers found in first walk model.")
    obs_dim, _ = _first_linear_in(layers0, sd0)
    if all_states.shape[1] != obs_dim:
        raise ValueError(f"{ALL_STATES_PATH} has obs_dim={all_states.shape[1]}, expected {obs_dim}")

    n_avail = all_states.shape[0]
    if n_avail >= N_ALL_STATES:
        idx = np.random.choice(n_avail, size=N_ALL_STATES, replace=False)
    else:
        idx = np.random.choice(n_avail, size=N_ALL_STATES, replace=True)
    X = all_states[idx].astype(np.float32, copy=False)
    ref_inputs = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)

    model_order = ["walk", "jump", "spin"]

    total_png = 0
    total_models_written = 0

    for (cycle_key, m_paths_orig) in cycles:
        run_dir, cyc = cycle_key
        print(f"\n[cycle] {run_dir} / c{cyc:03d}")

        # Copy originals into OUT_MODELS_DIR with standardized name
        m_paths = {}
        for b in model_order:
            m_paths[b] = _copy_model_to_out(m_paths_orig[b], OUT_MODELS_DIR, tag="original")
            total_models_written += 1

        # Load state dicts + actor layer descriptors for this cycle (from ORIGINAL sources)
        state_dicts = {}
        actor_layers_by_model = {}
        for m in model_order:
            sd = _load_rlg_forplay_state_dict(m_paths_orig[m])
            layers = _discover_actor_mlp_layers(sd)
            if not layers:
                raise RuntimeError(f"No actor MLP layers found for {m_paths_orig[m]}")
            state_dicts[m] = sd
            actor_layers_by_model[m] = layers

        # -------------------------
        # ORIGINAL FIXED CACHE (alive/order/labels + self/task pools + pick lesion units)
        # -------------------------
        ORIG_ALIVE_TOTAL_IDX = {m: {} for m in model_order}
        ORIG_ORDER_ALIVE     = {m: {} for m in model_order}
        ORIG_LABELS_ALIVE    = {m: {} for m in model_order}
        ORIG_K_BY_MODEL      = {m: {} for m in model_order}

        SELF_TOTAL_IDX = {m: {} for m in model_order}
        TASK_TOTAL_IDX = {m: {} for m in model_order}

        LESION_SELF_NEURONS_TOTAL = {m: {} for m in model_order}
        LESION_TASK_NEURONS_TOTAL = {m: {} for m in model_order}

        for layer_idx in LAYER_INDICES:
            for mi, m_name in enumerate(model_order):
                sd = state_dicts[m_name]
                layers = actor_layers_by_model[m_name]

                acts_full = get_layer_output_batch_from_ckpt(sd, ref_inputs, layer_idx, layers, ACTIVATION)
                dead_idx, alive_idx = _dead_alive_indices(acts_full, MIN_STD)
                alive_idx = np.asarray(alive_idx, dtype=int)

                if ALIVE_ONLY_CLUSTERING:
                    acts_use = acts_full[:, alive_idx] if alive_idx.size else acts_full[:, :0]
                else:
                    alive_idx = np.arange(acts_full.shape[1], dtype=int)
                    acts_use = acts_full

                A_z = _zscore_cols(acts_use, eps=EPS) if acts_use.size else acts_use.astype(np.float32, copy=False)
                R_signed = corr_matrix(A_z)
                R_abs = np.abs(R_signed)

                # =========================
                # NEW: abs-threshold (TAU) + CC + RCM-inside-blocks
                # =========================
                order0, labels0 = blockdiag_rcm_blocks_from_abs(
                    R_abs,
                    tau=TAU,
                    min_block_size=BD_MIN_BLOCK_SIZE,
                )
                labels = _relabel_clusters_by_size(labels0)
                order  = _order_by_cluster_size(labels, base_order=order0)
                K_best = int(np.max(labels) if labels.size else 0)

                ORIG_ALIVE_TOTAL_IDX[m_name][layer_idx] = alive_idx
                ORIG_ORDER_ALIVE[m_name][layer_idx]    = order.astype(int)
                ORIG_LABELS_ALIVE[m_name][layer_idx]   = labels.astype(int)
                ORIG_K_BY_MODEL[m_name][layer_idx]     = int(K_best)

                # Define self module = largest cluster (cluster 1 after relabel-by-size)
                self_alive_idx = np.where(labels == 1)[0].astype(int)
                task_alive_idx = np.where(labels != 1)[0].astype(int)

                self_total = alive_idx[self_alive_idx] if self_alive_idx.size else np.array([], dtype=int)
                task_total = alive_idx[task_alive_idx] if task_alive_idx.size else np.array([], dtype=int)

                SELF_TOTAL_IDX[m_name][layer_idx] = self_total
                TASK_TOTAL_IDX[m_name][layer_idx] = task_total

                rng_self = np.random.RandomState(LESION_SEED + 1000 * layer_idx + 31 * (mi + 1) + 1)
                rng_task = np.random.RandomState(LESION_SEED + 1000 * layer_idx + 31 * (mi + 1) + 2)

                n_pick = int(min(NUM_LESION, self_total.size, task_total.size))

                pick_self = rng_self.choice(self_total, size=n_pick, replace=False) if n_pick > 0 else np.array([], dtype=int)
                pick_task = rng_task.choice(task_total, size=n_pick, replace=False) if n_pick > 0 else np.array([], dtype=int)

                LESION_SELF_NEURONS_TOTAL[m_name][layer_idx] = np.asarray(pick_self, dtype=int)
                LESION_TASK_NEURONS_TOTAL[m_name][layer_idx] = np.asarray(pick_task, dtype=int)

        orig_fixed_cache = dict(
            ORIG_ALIVE_TOTAL_IDX=ORIG_ALIVE_TOTAL_IDX,
            ORIG_ORDER_ALIVE=ORIG_ORDER_ALIVE,
            ORIG_LABELS_ALIVE=ORIG_LABELS_ALIVE,
            ORIG_K_BY_MODEL=ORIG_K_BY_MODEL,
        )

        # -------------------------
        # Save reinitialized checkpoints (per model) into OUT_MODELS_DIR (renamed)
        # -------------------------
        self_paths = {}
        task_paths = {}

        for mi, m_name in enumerate(model_order):
            orig_path = m_paths_orig[m_name]   # use original path for timestamp/run parsing
            layers = actor_layers_by_model[m_name]

            self_out = _build_out_model_path(orig_path, tag="selflesion", dst_dir=OUT_MODELS_DIR)
            task_out = _build_out_model_path(orig_path, tag="tasklesion", dst_dir=OUT_MODELS_DIR)

            self_paths[m_name] = self_out
            task_paths[m_name] = task_out

            self_h1_pool = SELF_TOTAL_IDX[m_name][0]
            self_h2_pool = SELF_TOTAL_IDX[m_name][1]
            self_h1_les  = LESION_SELF_NEURONS_TOTAL[m_name][0]
            self_h2_les  = LESION_SELF_NEURONS_TOTAL[m_name][1]

            task_h1_pool = TASK_TOTAL_IDX[m_name][0]
            task_h2_pool = TASK_TOTAL_IDX[m_name][1]
            task_h1_les  = LESION_TASK_NEURONS_TOTAL[m_name][0]
            task_h2_les  = LESION_TASK_NEURONS_TOTAL[m_name][1]

            if DO_SELF_LESION:
                rng = np.random.RandomState(LESION_SEED + 999 * (mi + 1) + 111)
                _save_lesioned_copy(
                    orig_path=orig_path,
                    new_path=self_out,
                    actor_layers=layers,
                    pool_h1=self_h1_pool, pool_h2=self_h2_pool,
                    lesion_h1=self_h1_les, lesion_h2=self_h2_les,
                    rng=rng
                )
                total_models_written += 1

            if DO_TASK_LESION:
                rng = np.random.RandomState(LESION_SEED + 999 * (mi + 1) + 222)
                _save_lesioned_copy(
                    orig_path=orig_path,
                    new_path=task_out,
                    actor_layers=layers,
                    pool_h1=task_h1_pool, pool_h2=task_h2_pool,
                    lesion_h1=task_h1_les, lesion_h2=task_h2_les,
                    rng=rng
                )
                total_models_written += 1

        # -------------------------
        # Plotting (3 groups: ORIGINAL / SELF-LESION / TASK-LESION)
        # -------------------------
        def _make_group_plots(group_title: str, paths_dict: dict):
            layer_payloads = {}
            for layer_idx in LAYER_INDICES:
                A_z_by_model = {}
                R_signed_by_model = {}
                alive_counts_by_model = {}
                dead_counts_by_model = {}
                total_counts_by_model = {}

                for m_name in model_order:
                    sd = _load_rlg_forplay_state_dict(paths_dict[m_name])
                    layers = actor_layers_by_model[m_name]

                    acts_full = get_layer_output_batch_from_ckpt(sd, ref_inputs, layer_idx, layers, ACTIVATION)
                    total_counts_by_model[m_name] = int(acts_full.shape[1])

                    if USE_FIXED_ORIG_ALIVE_AND_ORDER:
                        alive_total = ORIG_ALIVE_TOTAL_IDX[m_name][layer_idx]
                        acts_use = acts_full[:, alive_total] if alive_total.size else acts_full[:, :0]
                        alive_counts_by_model[m_name] = int(alive_total.size)
                        dead_counts_by_model[m_name] = int(total_counts_by_model[m_name] - alive_total.size)
                    else:
                        dead_idx, alive_idx = _dead_alive_indices(acts_full, MIN_STD)
                        alive_total = alive_idx
                        acts_use = acts_full[:, alive_total] if alive_total.size else acts_full[:, :0]
                        alive_counts_by_model[m_name] = int(alive_total.size)
                        dead_counts_by_model[m_name] = int(dead_idx.size)

                    A_z = _zscore_cols(acts_use, eps=EPS) if acts_use.size else acts_use.astype(np.float32, copy=False)
                    R_signed = corr_matrix(A_z)

                    A_z_by_model[m_name] = A_z
                    R_signed_by_model[m_name] = R_signed

                layer_payloads[layer_idx] = (A_z_by_model, R_signed_by_model, alive_counts_by_model, dead_counts_by_model, total_counts_by_model)

            for m_name in model_order:
                title = _title_text_from_path(paths_dict[m_name], group_title)
                fname = _sanitize_filename(title) + ".png"
                out_png = os.path.join(OUT_GRAPHS_DIR, fname)
                _plot_one_model_two_layers_to_file(
                    group_title=group_title,
                    model_order=model_order,
                    model_name=m_name,
                    model_path=paths_dict[m_name],
                    layer_payloads=layer_payloads,
                    orig_fixed_cache=orig_fixed_cache,
                    out_png_path=out_png
                )

        _make_group_plots("ORIGINAL", m_paths)
        total_png += 3

        if DO_SELF_LESION:
            _make_group_plots("SELF-LESION", self_paths)
            total_png += 3

        if DO_TASK_LESION:
            _make_group_plots("TASK-LESION", task_paths)
            total_png += 3

        gc.collect()

    print(f"\n[DONE] Wrote models: {total_models_written} into {OUT_MODELS_DIR}")
    print(f"[DONE] Wrote PNGs:   {total_png} into {OUT_GRAPHS_DIR}")
    print(f"[DONE] Output root:  {OUT_ROOT}")


if __name__ == "__main__":
    main()