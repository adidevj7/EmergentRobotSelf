# Training and Setup

This document describes the training, rollout recording, and state collection side of the pipeline used in **“Evidence of an Emergent ‘Self’ in Continual Robot Learning.”**

The main README is aimed at paper readers and figure reproduction. This file is for users who want to understand or rerun the upstream pipeline that produces the checkpoints, recorded states, and cached analysis artifacts.

See the main repository page here:

- [`README.md`](README.md)

---

## Pipeline overview

The full pipeline is:

1. **Train policies** across one or more behavior phases
2. **Record states / observations / videos** from trained checkpoints
3. **Run MAPS analysis** on those saved states and checkpoints
4. **Generate figures** using the notebooks in `AnalysisScripts/`

In practice, most users will not need to rerun the entire training process, because the repository already includes cached artifacts for figure reproduction.

---

## Important folders

### Training and rollout utilities
- `Training_ObsCollection_Scripts/`

### Analysis notebooks and scripts
- `AnalysisScripts/`

### Cached outputs used by the analysis notebooks
- `Checkpoints_States_selectedGraphs/`

---

## Main training entry point

The most important training file is:

- `Training_ObsCollection_Scripts/Isaac_TrainingLauncher.py`

This is the main launcher used for the training scheme behind the paper experiments.

Other files in `Training_ObsCollection_Scripts/` are primarily helpers for:
- configuration
- rollout collection
- checkpoint conversion
- video recording
- plotting / inspection
- sim-to-real playback

---

## Core training command

A representative training command is:

```bash
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p /home/adi/projects/CreativeMachinesAnt/Isaac/Training_ObsCollection_Scripts/Isaac_TrainingLauncher.py \
  --task Ant-Walk-v0 --gym_env_id Isaac-Ant-Direct-v0 \
  --cfg_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/Training_ObsCollection_Scripts/cfg/rlg_walk_new_150_relu.yaml \
  --player_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/Training_ObsCollection_Scripts/cfg/rlg_play_sac_ant_150_relu.yaml \
  --num_envs 8192 --n_cycles 2 --phase_order walk --updates_per_step 32 \
  --plateau_min_steps 250000 --max_steps_phase 500000 \
  --override_warmup_steps 10000 --log_interval_s 30 --headless --lambda_back 1 \
  --gpu 0 --record_every 0 --video_gpu 6 --video_wait_pct 50 --video_wait_s 30 \
  --run_tag WSJ_att69_WalkOnly_relu_0 --ckpt_label WSJ_att69_WalkOnly_relu_0 --seed 0
