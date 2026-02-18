#!/usr/bin/env python3
from __future__ import annotations
"""
Unified Ant (WALK → JUMP → SPIN) trainer — phases + plateau-switching scaffold
(Option 5b applied: each phase gets its own fresh Isaac app/env and we shut it
down at the end of the phase so cycle 2+ won't hang)

Key points baked in:
  • Separate replay buffers per phase (size via --replay_buffer_size, default 1e6)
  • Same policy weights carried across phases on switch
  • Plateau detection controls switching; N cycles via --n_cycles
  • Rewards: WALK uses env reward, JUMP rewards vertical velocity, SPIN rewards spin
  • log_reward_breakdown is a global toggle

  • 5b: Isaac AppLauncher + env_cfg are created INSIDE each phase, closed at
        the end of that phase (so we can do multiple phases/cycles in one process)

test

seed 0
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p /home/adi/projects/CreativeMachinesAnt/Isaac/scripts/Isaac_WSJ_att69_cleanup.py \
    --task Ant-Walk-v0   --gym_env_id Isaac-Ant-Direct-v0   --cfg_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_walk_new_150_relu.yaml \
    --player_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_play_sac_ant_150_relu.yaml   --num_envs 8192   --n_cycles 50 \
    --phase_order walk,spin,jump   --updates_per_step 32   --plateau_min_steps 10_000_000   --max_steps_phase 15_000_000 \
    --override_warmup_steps 10000   --log_interval_s 5   --headless  --lambda_back 1  --gpu 0 \
    --record_every 0   --video_gpu 6   --video_wait_pct 50   --video_wait_s 30   --run_tag WSJ_att69_RB_norm_relu_0_nonorm --ckpt_label WSJ_att69_RB_norm_relu_0_nonorm --seed 0

NO norm
seed 0
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p /home/adi/projects/CreativeMachinesAnt/Isaac/scripts/Isaac_WSJ_att69_cleanup.py \
    --task Ant-Walk-v0   --gym_env_id Isaac-Ant-Direct-v0   --cfg_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_walk_new_150_relu.yaml \
    --player_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_play_sac_ant_150_relu.yaml   --num_envs 8192   --n_cycles 50 \
    --phase_order walk,spin,jump   --updates_per_step 32   --plateau_min_steps 250_000_000   --max_steps_phase 1_500_000_000 \
    --override_warmup_steps 10000   --log_interval_s 30   --headless  --lambda_back 1  --gpu 0 \
    --record_every 0   --video_gpu 6   --video_wait_pct 50   --video_wait_s 30   --run_tag WSJ_att69_relu_0 --ckpt_label WSJ_att69_relu_0 --seed 0

    
seed 1
CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p /home/adi/projects/CreativeMachinesAnt/Isaac/scripts/Isaac_WSJ_att69_cleanup.py \
    --task Ant-Walk-v0   --gym_env_id Isaac-Ant-Direct-v0   --cfg_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_walk_new_150_relu.yaml \
    --player_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_play_sac_ant_150_relu.yaml   --num_envs 8192   --n_cycles 50 \
    --phase_order walk,spin,jump   --updates_per_step 32   --plateau_min_steps 250_000_000   --max_steps_phase 1_500_000_000 \
    --override_warmup_steps 10000   --log_interval_s 30   --headless  --lambda_back 1  --gpu 1 \
    --record_every 0   --video_gpu 6   --video_wait_pct 50   --video_wait_s 30   --run_tag WSJ_att69_relu_1 --ckpt_label WSJ_att69_relu_1 --seed 1

seed 2
CUDA_VISIBLE_DEVICES=2 ./isaaclab.sh -p /home/adi/projects/CreativeMachinesAnt/Isaac/scripts/Isaac_WSJ_att69_cleanup.py \
    --task Ant-Walk-v0   --gym_env_id Isaac-Ant-Direct-v0   --cfg_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_walk_new_150_relu.yaml \
    --player_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_play_sac_ant_150_relu.yaml   --num_envs 8192   --n_cycles 50 \
    --phase_order walk,spin,jump   --updates_per_step 32   --plateau_min_steps 250_000_000   --max_steps_phase 1_500_000_000 \
    --override_warmup_steps 10000   --log_interval_s 30   --headless  --lambda_back 1  --gpu 2 \
    --record_every 0   --video_gpu 6   --video_wait_pct 50   --video_wait_s 30   --run_tag WSJ_att69_relu_2 --ckpt_label WSJ_att69_relu_2 --seed 2
    
seed 3
CUDA_VISIBLE_DEVICES=3 ./isaaclab.sh -p /home/adi/projects/CreativeMachinesAnt/Isaac/scripts/Isaac_WSJ_att69_cleanup.py \
    --task Ant-Walk-v0   --gym_env_id Isaac-Ant-Direct-v0   --cfg_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_walk_new_150_relu.yaml \
    --player_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_play_sac_ant_150_relu.yaml   --num_envs 8192   --n_cycles 50 \
    --phase_order walk,spin,jump   --updates_per_step 32   --plateau_min_steps 250_000_000   --max_steps_phase 1_500_000_000 \
    --override_warmup_steps 10000   --log_interval_s 30   --headless  --lambda_back 1  --gpu 3 \
    --record_every 0   --video_gpu 6   --video_wait_pct 50   --video_wait_s 30   --run_tag WSJ_att69_relu_3 --ckpt_label WSJ_att69_relu_3 --seed 3

seed 4
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p /home/adi/projects/CreativeMachinesAnt/Isaac/scripts/Isaac_WSJ_att69_cleanup.py \
    --task Ant-Walk-v0   --gym_env_id Isaac-Ant-Direct-v0   --cfg_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_walk_new_150_relu.yaml \
    --player_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_play_sac_ant_150_relu.yaml   --num_envs 8192   --n_cycles 50 \
    --phase_order walk,spin,jump   --updates_per_step 32   --plateau_min_steps 250_000_000   --max_steps_phase 1_500_000_000 \
    --override_warmup_steps 10000   --log_interval_s 30   --headless  --lambda_back 1  --gpu 0 \
    --record_every 0   --video_gpu 6   --video_wait_pct 50   --video_wait_s 30   --run_tag WSJ_att69_relu_4 --ckpt_label WSJ_att69_relu_4 --seed 4

seed 5
CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p /home/adi/projects/CreativeMachinesAnt/Isaac/scripts/Isaac_WSJ_att69_cleanup.py \
    --task Ant-Walk-v0   --gym_env_id Isaac-Ant-Direct-v0   --cfg_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_walk_new_150_relu.yaml \
    --player_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_play_sac_ant_150_relu.yaml   --num_envs 8192   --n_cycles 50 \
    --phase_order walk,spin,jump   --updates_per_step 32   --plateau_min_steps 250_000_000   --max_steps_phase 1_500_000_000 \
    --override_warmup_steps 10000   --log_interval_s 30   --headless  --lambda_back 1  --gpu 1 \
    --record_every 0   --video_gpu 6   --video_wait_pct 50   --video_wait_s 30   --run_tag WSJ_att69_relu_5 --ckpt_label WSJ_att69_relu_5 --seed 5

seed 6
CUDA_VISIBLE_DEVICES=2 ./isaaclab.sh -p /home/adi/projects/CreativeMachinesAnt/Isaac/scripts/Isaac_WSJ_att69_cleanup.py \
    --task Ant-Walk-v0   --gym_env_id Isaac-Ant-Direct-v0   --cfg_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_walk_new_150_relu.yaml \
    --player_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_play_sac_ant_150_relu.yaml   --num_envs 8192   --n_cycles 50 \
    --phase_order walk,spin,jump   --updates_per_step 32   --plateau_min_steps 250_000_000   --max_steps_phase 1_500_000_000 \
    --override_warmup_steps 10000   --log_interval_s 30   --headless  --lambda_back 1  --gpu 2 \
    --record_every 0   --video_gpu 6   --video_wait_pct 50   --video_wait_s 30   --run_tag WSJ_att69_relu_6 --ckpt_label WSJ_att69_relu_6 --seed 6

seed 7
CUDA_VISIBLE_DEVICES=3 ./isaaclab.sh -p /home/adi/projects/CreativeMachinesAnt/Isaac/scripts/Isaac_WSJ_att69_cleanup.py \
    --task Ant-Walk-v0   --gym_env_id Isaac-Ant-Direct-v0   --cfg_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_walk_new_150_relu.yaml \
    --player_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_play_sac_ant_150_relu.yaml   --num_envs 8192   --n_cycles 50 \
    --phase_order walk,spin,jump   --updates_per_step 32   --plateau_min_steps 250_000_000   --max_steps_phase 1_500_000_000 \
    --override_warmup_steps 10000   --log_interval_s 30   --headless  --lambda_back 1  --gpu 3 \
    --record_every 0   --video_gpu 6   --video_wait_pct 50   --video_wait_s 30   --run_tag WSJ_att69_relu_7 --ckpt_label WSJ_att69_relu_7 --seed 7

To be set

seed 8
CUDA_VISIBLE_DEVICES=4 ./isaaclab.sh -p /home/adi/projects/CreativeMachinesAnt/Isaac/scripts/Isaac_WSJ_att69_cleanup.py \
    --task Ant-Walk-v0   --gym_env_id Isaac-Ant-Direct-v0   --cfg_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_walk_new_150_relu.yaml \
    --player_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_play_sac_ant_150_relu.yaml   --num_envs 8192   --n_cycles 50 \
    --phase_order walk,spin,jump   --updates_per_step 32   --plateau_min_steps 250_000_000   --max_steps_phase 1_500_000_000 \
    --override_warmup_steps 10000   --log_interval_s 30   --headless  --lambda_back 1  --gpu 4 \
    --record_every 0   --video_gpu 6   --video_wait_pct 50   --video_wait_s 30   --run_tag WSJ_att69_relu_8 --ckpt_label WSJ_att69_relu_8 --seed 8

seed 9
CUDA_VISIBLE_DEVICES=4 ./isaaclab.sh -p /home/adi/projects/CreativeMachinesAnt/Isaac/scripts/Isaac_WSJ_att69_cleanup.py \
    --task Ant-Walk-v0   --gym_env_id Isaac-Ant-Direct-v0   --cfg_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_walk_new_150_relu.yaml \
    --player_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_play_sac_ant_150_relu.yaml   --num_envs 8192   --n_cycles 50 \
    --phase_order walk,spin,jump   --updates_per_step 32   --plateau_min_steps 250_000_000   --max_steps_phase 1_500_000_000 \
    --override_warmup_steps 10000   --log_interval_s 30   --headless  --lambda_back 1  --gpu 4 \
    --record_every 0   --video_gpu 6   --video_wait_pct 50   --video_wait_s 30   --run_tag WSJ_att69_relu_9 --ckpt_label WSJ_att69_relu_9 --seed 9


seed 10
CUDA_VISIBLE_DEVICES=5 ./isaaclab.sh -p /home/adi/projects/CreativeMachinesAnt/Isaac/scripts/Isaac_WSJ_att69_cleanup.py \
    --task Ant-Walk-v0   --gym_env_id Isaac-Ant-Direct-v0   --cfg_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_walk_new_150_relu.yaml \
    --player_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_play_sac_ant_150_relu.yaml   --num_envs 8192   --n_cycles 50 \
    --phase_order walk,spin,jump   --updates_per_step 32   --plateau_min_steps 250_000_000   --max_steps_phase 1_500_000_000 \
    --override_warmup_steps 10000   --log_interval_s 30   --headless  --lambda_back 1  --gpu 5 \
    --record_every 0   --video_gpu 6   --video_wait_pct 50   --video_wait_s 30   --run_tag WSJ_att69_relu_10 --ckpt_label WSJ_att69_relu_10 --seed 10

walk only
CUDA_VISIBLE_DEVICES=5 ./isaaclab.sh -p /home/adi/projects/CreativeMachinesAnt/Isaac/scripts/Isaac_WSJ_att69_cleanup.py \
    --task Ant-Walk-v0   --gym_env_id Isaac-Ant-Direct-v0   --cfg_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_walk_new_150_relu.yaml \
    --player_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_play_sac_ant_150_relu.yaml   --num_envs 8192   --n_cycles 150 \
    --phase_order walk   --updates_per_step 32   --plateau_min_steps 250_000_000   --max_steps_phase 1_500_000_000 \
    --override_warmup_steps 10000   --log_interval_s 30   --headless  --lambda_back 1  --gpu 5 \
    --record_every 0   --video_gpu 6   --video_wait_pct 50   --video_wait_s 30   --run_tag WSJ_att69_WalkOnly_relu_42 --ckpt_label WSJ_att69_WalkOnly_relu_42 --seed 42

spin only
CUDA_VISIBLE_DEVICES=6 ./isaaclab.sh -p /home/adi/projects/CreativeMachinesAnt/Isaac/scripts/Isaac_WSJ_att69_cleanup.py \
    --task Ant-Walk-v0   --gym_env_id Isaac-Ant-Direct-v0   --cfg_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_walk_new_150_relu.yaml \
    --player_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_play_sac_ant_150_relu.yaml   --num_envs 8192   --n_cycles 150 \
    --phase_order spin   --updates_per_step 32   --plateau_min_steps 250_000_000   --max_steps_phase 1_500_000_000 \
    --override_warmup_steps 10000   --log_interval_s 30   --headless  --lambda_back 1  --gpu 6 \
    --record_every 0   --video_gpu 6   --video_wait_pct 50   --video_wait_s 30   --run_tag WSJ_att69_SpinOnly_relu_42 --ckpt_label WSJ_att69_SpinOnly_relu_42 --seed 42

jump only
CUDA_VISIBLE_DEVICES=6 ./isaaclab.sh -p /home/adi/projects/CreativeMachinesAnt/Isaac/scripts/Isaac_WSJ_att69_cleanup.py \
    --task Ant-Walk-v0   --gym_env_id Isaac-Ant-Direct-v0   --cfg_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_walk_new_150_relu.yaml \
    --player_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_play_sac_ant_150_relu.yaml   --num_envs 8192   --n_cycles 150 \
    --phase_order jump   --updates_per_step 32   --plateau_min_steps 250_000_000   --max_steps_phase 1_500_000_000 \
    --override_warmup_steps 10000   --log_interval_s 30   --headless  --lambda_back 1  --gpu 6 \
    --record_every 0   --video_gpu 6   --video_wait_pct 50   --video_wait_s 30   --run_tag WSJ_att69_JumpOnly_relu_42 --ckpt_label WSJ_att69_JumpOnly_relu_42 --seed 42


VAST
Walk Spin
seed 0
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p /home/adi/projects/CreativeMachinesAnt/Isaac/scripts/Isaac_WSJ_att69_cleanup.py \
    --task Ant-Walk-v0   --gym_env_id Isaac-Ant-Direct-v0   --cfg_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_walk_new_150_relu.yaml \
    --player_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_play_sac_ant_150_relu.yaml   --num_envs 8192   --n_cycles 50 \
    --phase_order walk,spin   --updates_per_step 32   --plateau_min_steps 250_000_000   --max_steps_phase 1_500_000_000 \
    --override_warmup_steps 10000   --log_interval_s 30   --headless  --lambda_back 1  --gpu 0 \
    --record_every 0   --video_gpu 4   --video_wait_pct 50   --video_wait_s 30   --run_tag WSJ_att69_walkspin_relu_0 --ckpt_label WSJ_att69_walkspin_relu_0 --seed 0 \
    --resume_from /home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_walkspin_relu_0_2026-02-03_15-31-36/models/c023_b02_spin_plateau_2026-02-04_23-32-09_for_play.pth


    
seed 1
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p /home/adi/projects/CreativeMachinesAnt/Isaac/scripts/Isaac_WSJ_att69_cleanup.py \
    --task Ant-Walk-v0   --gym_env_id Isaac-Ant-Direct-v0   --cfg_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_walk_new_150_relu.yaml \
    --player_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_play_sac_ant_150_relu.yaml   --num_envs 8192   --n_cycles 35 \
    --phase_order walk,spin   --updates_per_step 32   --plateau_min_steps 250_000_000   --max_steps_phase 1_500_000_000 \
    --override_warmup_steps 10000   --log_interval_s 30   --headless  --lambda_back 1  --gpu 0 \
    --record_every 0   --video_gpu 4   --video_wait_pct 50   --video_wait_s 30   --run_tag WSJ_att69_walkspin_relu_1 --ckpt_label WSJ_att69_walkspin_relu_1 --seed 1 \
    --resume_from /home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_walkspin_relu_1_2026-02-03_15-31-09/models/c015_b02_spin_plateau_2026-02-04_22-51-43_for_play.pth

CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p /home/adi/projects/CreativeMachinesAnt/Isaac/scripts/Isaac_WSJ_att69_cleanup.py \
    --task Ant-Walk-v0   --gym_env_id Isaac-Ant-Direct-v0   --cfg_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_walk_new_150_relu.yaml \
    --player_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_play_sac_ant_150_relu.yaml   --num_envs 8192   --n_cycles 50 \
    --phase_order walk,spin   --updates_per_step 32   --plateau_min_steps 250_000_000   --max_steps_phase 1_500_000_000 \
    --override_warmup_steps 10000   --log_interval_s 30   --headless  --lambda_back 1  --gpu 0 \
    --record_every 0   --video_gpu 4   --video_wait_pct 50   --video_wait_s 30   --run_tag WSJ_att69_walkspin_relu_1 --ckpt_label WSJ_att69_walkspin_relu_1 --seed 1 \
    --resume_from /home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_walkspin_relu_1_2026-02-03_15-31-09/models/c035_b02_spin_plateau_2026-02-06_13-37-59_for_play.pth


    
    
Walk Jump
cd /workspace/projects/IsaacLab/IsaacLab
seed 0
CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p /home/adi/projects/CreativeMachinesAnt/Isaac/scripts/Isaac_WSJ_att69_cleanup.py \
    --task Ant-Walk-v0   --gym_env_id Isaac-Ant-Direct-v0   --cfg_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_walk_new_150_relu.yaml \
    --player_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_play_sac_ant_150_relu.yaml   --num_envs 8192   --n_cycles 30 \
    --phase_order walk,jump   --updates_per_step 32   --plateau_min_steps 250_000_000   --max_steps_phase 1_500_000_000 \
    --override_warmup_steps 10000   --log_interval_s 30   --headless  --lambda_back 1  --gpu 1 \
    --record_every 0   --video_gpu 4   --video_wait_pct 50   --video_wait_s 30   --run_tag WSJ_att69_walkjump_relu_0 --ckpt_label WSJ_att69_walkjump_relu_0 --seed 0 \
    --resume_from /home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_walkjump_relu_0_2026-02-03_15-30-06/models/c020_b02_jump_plateau_2026-02-04_22-37-23_for_play.pth

CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p /home/adi/projects/CreativeMachinesAnt/Isaac/scripts/Isaac_WSJ_att69_cleanup.py \
    --task Ant-Walk-v0   --gym_env_id Isaac-Ant-Direct-v0   --cfg_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_walk_new_150_relu.yaml \
    --player_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_play_sac_ant_150_relu.yaml   --num_envs 8192   --n_cycles 50 \
    --phase_order walk,jump   --updates_per_step 32   --plateau_min_steps 250_000_000   --max_steps_phase 1_500_000_000 \
    --override_warmup_steps 10000   --log_interval_s 30   --headless  --lambda_back 1  --gpu 1 \
    --record_every 0   --video_gpu 4   --video_wait_pct 50   --video_wait_s 30   --run_tag WSJ_att69_walkjump_relu_0 --ckpt_label WSJ_att69_walkjump_relu_0 --seed 0 \
    --resume_from /home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_walkjump_relu_0_2026-02-03_15-30-06/models/c030_b02_jump_plateau_2026-02-05_16-49-40_for_play.pth

    
seed 1
CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p /home/adi/projects/CreativeMachinesAnt/Isaac/scripts/Isaac_WSJ_att69_cleanup.py \
    --task Ant-Walk-v0   --gym_env_id Isaac-Ant-Direct-v0   --cfg_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_walk_new_150_relu.yaml \
    --player_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_play_sac_ant_150_relu.yaml   --num_envs 8192   --n_cycles 32 \
    --phase_order walk,jump   --updates_per_step 32   --plateau_min_steps 250_000_000   --max_steps_phase 1_500_000_000 \
    --override_warmup_steps 10000   --log_interval_s 30   --headless  --lambda_back 1  --gpu 1 \
    --record_every 0   --video_gpu 4   --video_wait_pct 50   --video_wait_s 30   --run_tag WSJ_att69_walkjump_relu_1 --ckpt_label WSJ_att69_walkjump_relu_1 --seed 1 \
    --resume_from /home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_walkjump_relu_1_2026-02-03_15-30-19/models/c018_b02_jump_plateau_2026-02-04_23-29-40_for_play.pth

Jump Spin
seed 0
CUDA_VISIBLE_DEVICES=2 ./isaaclab.sh -p /home/adi/projects/CreativeMachinesAnt/Isaac/scripts/Isaac_WSJ_att69_cleanup.py \
    --task Ant-Walk-v0   --gym_env_id Isaac-Ant-Direct-v0   --cfg_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_walk_new_150_relu.yaml \
    --player_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_play_sac_ant_150_relu.yaml   --num_envs 8192   --n_cycles 50 \
    --phase_order spin,jump   --updates_per_step 32   --plateau_min_steps 250_000_000   --max_steps_phase 1_500_000_000 \
    --override_warmup_steps 10000   --log_interval_s 30   --headless  --lambda_back 1  --gpu 2 \
    --record_every 0   --video_gpu 4   --video_wait_pct 50   --video_wait_s 30   --run_tag WSJ_att69_jumpspin_relu_0 --ckpt_label WSJ_att69_jumpspin_relu_0 --seed 0 \
    --resume_from /home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_jumpspin_relu_0_2026-02-03_15-29-41/models/c026_b02_jump_plateau_2026-02-04_23-22-45_for_play.pth

    
seed 1
CUDA_VISIBLE_DEVICES=2 ./isaaclab.sh -p /home/adi/projects/CreativeMachinesAnt/Isaac/scripts/Isaac_WSJ_att69_cleanup.py \
    --task Ant-Walk-v0   --gym_env_id Isaac-Ant-Direct-v0   --cfg_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_walk_new_150_relu.yaml \
    --player_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_play_sac_ant_150_relu.yaml   --num_envs 8192   --n_cycles 32 \
    --phase_order spin,jump   --updates_per_step 32   --plateau_min_steps 250_000_000   --max_steps_phase 1_500_000_000 \
    --override_warmup_steps 10000   --log_interval_s 30   --headless  --lambda_back 1  --gpu 2 \
    --record_every 0   --video_gpu 4   --video_wait_pct 50   --video_wait_s 30   --run_tag WSJ_att69_jumpspin_relu_1 --ckpt_label WSJ_att69_jumpspin_relu_1 --seed 1 \
    --resume_from /home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_jumpspin_relu_1_2026-02-03_15-29-49/models/c018_b02_jump_plateau_2026-02-04_22-31-44_for_play.pth

control 1
CUDA_VISIBLE_DEVICES=3 ./isaaclab.sh -p /home/adi/projects/CreativeMachinesAnt/Isaac/scripts/Isaac_WSJ_att69_cleanup.py \
    --task Ant-Walk-v0   --gym_env_id Isaac-Ant-Direct-v0   --cfg_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_walk_new_150_relu.yaml \
    --player_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_play_sac_ant_150_relu.yaml   --num_envs 8192   --n_cycles 38 \
    --phase_order walk,spin,jump   --updates_per_step 32   --plateau_min_steps 250_000_000   --max_steps_phase 1_500_000_000 \
    --override_warmup_steps 10000   --log_interval_s 5   --headless  --lambda_back 1  --gpu 3 \
    --record_every 0   --video_gpu 6   --video_wait_pct 50   --video_wait_s 30   --run_tag WSJ_att69_Control_11 --ckpt_label WSJ_att69_Control_11 --seed 11 \
    --resume_from /home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_Control_11_2026-02-03_15-29-13/models/c012_b03_jump_plateau_2026-02-04_22-36-11_for_play.pth
controlk 2

CUDA_VISIBLE_DEVICES=3 ./isaaclab.sh -p /home/adi/projects/CreativeMachinesAnt/Isaac/scripts/Isaac_WSJ_att69_cleanup.py \
    --task Ant-Walk-v0   --gym_env_id Isaac-Ant-Direct-v0   --cfg_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_walk_new_150_relu.yaml \
    --player_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_play_sac_ant_150_relu.yaml   --num_envs 8192   --n_cycles 36 \
    --phase_order walk,spin,jump   --updates_per_step 32   --plateau_min_steps 250_000_000   --max_steps_phase 1_500_000_000 \
    --override_warmup_steps 10000   --log_interval_s 5   --headless  --lambda_back 1  --gpu 3 \
    --record_every 0   --video_gpu 6   --video_wait_pct 50   --video_wait_s 30   --run_tag WSJ_att69_Control_12 --ckpt_label WSJ_att69_Control_12 --seed 12 \
    --resume_from /home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/WSJ_att69_Control_12_2026-02-03_15-29-22/models/c014_b03_jump_plateau_2026-02-04_23-04-35_for_play.pth

    """

# ──────────────────────────────────────────────────────────────────────────────
# 00. Imports & Globals
# ──────────────────────────────────────────────────────────────────────────────
import os, sys, argparse, json, time, shutil, signal, random
from pathlib import Path
from collections import deque
from copy import deepcopy
from datetime import datetime

# Thread caps BEFORE numpy/torch
os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("MKL_NUM_THREADS", "12")

import numpy as np
import torch
import gymnasium as gym
import yaml
import subprocess

try:
    import pynvml  # type: ignore
except Exception:
    pynvml = None

# Hard-disable Dynamo/JIT overhead (diagnostic parity with your scripts)
try:
    import torch._dynamo as _d
    _d.disable()
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
except Exception:
    pass
os.environ.setdefault("PYTORCH_JIT", "0")

# (RB helper will be used in VecEnv later — import now for parity)
from reward_breakdown_helper import RewardBreakdown  # noqa: F401

_GLOBAL_ENV = None
_GLOBAL_VECENV = None
_GLOBAL_ALGO = None

_RUN_STAMP = time.strftime("%Y-%m-%d_%H-%M-%S")
_SHUTDOWN_REQUESTED = False
_SEED = None

# vx source once (used by logging later)
_VX_SOURCE_PRINTED = False
_VX_SOURCE_NAME = None

# CSV safety bookkeeping (sync writer)
_CSV_EXPECTED_HEADER = "time_s,episodes_done,total_steps,avg_return_window,avg_len_window,fps,mean_vx_b,timeout_pct,non_timeout_pct\n"
_CSV_WRITE_COUNT = 0
_CSV_FSYNC_EVERY = 50  # fsync every N writes

# Paths set later (per-phase)
_ROLLOUT_LOG_PATH = None
_CKPT_DIR = None
_ROLLOUT_MIRROR_CSV = None

# NEW: per-run subfolders
_MODELS_DIR = None
_VIDEOS_DIR = None
_GRAPHS_DIR = None

# globals near the top
_GLOBAL_APP = None

def _usd_detach_stage():
    try:
        import omni.usd
        ctx = omni.usd.get_context()
        if ctx is not None:
            ctx.detach_stage()
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# 05. Variables (single source of truth for defaults & hard-coded knobs)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULTS = {
    # Core run / env boot
    "core": {
        "task": "Ant-Walk-v0",
        "gym_env_id": "Isaac-Ant-Direct-v0",
        "num_envs": 8192,
        "seed": 42,
        "gpu": 0,
        "headless": False,
        "cfg_yaml_required": True,  # still required by CLI
    },

    # Logging / diagnostics
    "logging": {
        "logdir": str(Path.home()/ "projects/CreativeMachinesAnt/Isaac/logs/rl_games/ant_unified"),
        "run_name": None,
        "run_tag": None,
        "log_window": 100,
        "log_interval_s": 30.0,
        "return_torch": 1,  # keep int to match choices=[0,1]
        "gpu_util": False,
        "perf_diag": False,
        "nvtx": False,
        "log_reward_breakdown": False,
        # NEW: print full reward breakdown every N rollout logs (your request: N=1)
        "rb_print_every_n": 1,
    },

    # Checkpoint / training horizon ergonomics
    "checkpoint": {
        "max_episodes": None,               # legacy guard (not used for plateau)
        "manual_ckpt_every_eps": None,      # if set, per-phase cadence
    },

    # Normalization toggles (we still default to DISABLED after parse)
    "normalization": {
        "disable_obs_norm_flag_default": False,
        "disable_rew_norm_flag_default": False,
    },

    # RL training meta
    "training": {
        "batch_scale": 2.0,                 # multiplies YAML batch sizes
        "replay_buffer_size": 1_000_000,    # per-phase buffer capacity
        "override_warmup_steps": None,      # if set, forces warmup on agent
        "force_no_warmup": False,           # if True, sets warmup to 0
    },

    # Phase ordering / cycles
    "phases": {
        "n_cycles": 1,
        "phase_order": "walk,jump,spin",
    },

    # Plateau GUARDS (timestep-based; used as guards only)
    "plateau_steps": {
        "plateau_window_steps": 200_000,    # legacy (not used for decision)
        "plateau_abs_std": 100.0,           # legacy (not used for decision)
        "plateau_rel_std": 0.05,            # legacy (not used for decision)
        "plateau_min_steps": 50_000_000,    # aggregated env-steps before enabling decision
        "max_steps_phase": 2_500_000_000,   # hard cap on aggregated env-steps per phase
    },

    # Plateau DECISION (MuJoCo-style, episode-based)
    "plateau_episodes": {
        "plateau_episode_window": 50_000,      # episodes per window
        "plateau_rel_change": 0.05,         # |μr-μp|/|μp|
        "plateau_std_coeff": 1.0,           # |μr-μp| ≤ coeff · σ(two windows)
        "plateau_min_return": 500.0,          # require μr ≥ this
    },

    # Reward knobs (phase-specific)
    "rewards": {
        "spin": {
            "alpha_abs_wz": 2.0,
            "lambda_back": 1,
            "lambda_vxy": 0.5,
            "lambda_jerk": 1.0,
            "spin_direction": 1,           # {-1,+1}
            # NEW: streak bonus params
            "streak_k": 0.10,              # reward per second of clean spin
            "streak_cap_s": 200.0,           # cap streak at this many seconds
            "streak_eps": 0.1,             # tolerance around zero for sign checks
        },
        "jump": {
            "beta_up": 8.0,
            "lambda_drift": 2,
            "lambda_jerk_jump": 1,
        },
    },

    # Isaac env tweaks we applied in-code
    "env": {
        "decimation": 2,                    # control/action decimation
        "episode_length_steps": 1000,       # timeout-only horizon
        "timeout_only": True,               # disable other terminations
    },

    # rl_games grid / agent config (the "minimal algo grid")
    "grid": {
        "batch_size": 32768,
        "updates_per_step": 32,
        "train_every_n_steps": 1,
        "actor_update_interval": 1,
        "warmup_steps": 10_000,
        "replay_buffer_size": 1_000_000,
        "target_entropy": -8.0,
        "alpha_lr": 0.003,
        "critic_tau": 0.003,
        "gamma": 0.99,
    },
}

# Convenience alias for internal use (read-only)
GRID_DEFAULTS = DEFAULTS["grid"]

_LAST_FOR_PLAY = None
_LAST_STOP_REASON = None
_LAST_PLATEAU_MU = None
_LAST_FINAL_WINDOW_MEAN = None


# ──────────────────────────────────────────────────────────────────────────────
# 10. Helpers
# ──────────────────────────────────────────────────────────────────────────────
_PHASE_STATE_PATH = None

def _state_path() -> Path:
    global _PHASE_STATE_PATH
    if _PHASE_STATE_PATH is None:
        _PHASE_STATE_PATH = _CKPT_DIR / "phase_state.json"
    return _PHASE_STATE_PATH

def _load_state() -> dict:
    p = _state_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}

def _save_state_atomic(state: dict) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(p)

def _hard_restart_self():
    # Relaunch via isaaclab.sh so the environment is identical.
    isaaclab_sh = str(Path.home() / "projects" / "IsaacLab" / "isaaclab.sh")
    script_path = str(Path(__file__).resolve())
    argv = ["/bin/bash", isaaclab_sh, "-p", script_path] + sys.argv[1:]
    print(f"[hard-restart] execv: {' '.join(argv)}")
    os.execv("/bin/bash", argv)


def _safe_write_line(csv_path: Path, line: str):
    """Append a line and flush; fsync every N writes to reduce data loss."""
    global _CSV_WRITE_COUNT
    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("a") as f:
            f.write(line)
            f.flush()
            _CSV_WRITE_COUNT += 1
            if (_CSV_WRITE_COUNT % _CSV_FSYNC_EVERY) == 0:
                os.fsync(f.fileno())
    except Exception:
        pass

def _script_run_tag() -> str:
    return f"{Path(__file__).stem}_{_RUN_STAMP}"

def _init_ckpt_dir() -> Path:
    # If user asked to reuse a specific checkpoints dir, do that.
    override = getattr(args, "ckpt_dir_override", None)
    if override and str(override).strip():
        run_dir = Path(override).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    base = Path.home() / "projects" / "CreativeMachinesAnt" / "Isaac" / "checkpoints"
    label = getattr(args, "ckpt_label", None)
    if label and isinstance(label, str) and label.strip():
        run_dir = base / f"{label.strip()}_{_RUN_STAMP}"
    else:
        run_dir = base / _script_run_tag()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir



def _native_save(algo, out_path: str) -> tuple[bool, str]:
    """Use rl_games native saver if available."""
    if algo is None:
        return False, "algo is None"
    saver = getattr(algo, 'saver', None)
    if saver is not None and hasattr(saver, 'save') and callable(saver.save):
        try:
            saver.save(out_path); return True, "saver.save"
        except Exception as e:
            return False, f"saver.save exception: {e}"
    if hasattr(algo, 'save') and callable(algo.save):
        try:
            algo.save(out_path); return True, "algo.save"
        except Exception as e:
            return False, f"algo.save exception: {e}"
    return False, "no rl_games native saver present"

def _export_for_play_model(algo, out_path: str) -> tuple[bool, str]:
    """
    Build a CPU state_dict clone and save it (don't touch live GPU weights).
    """
    if algo is None: return False, "algo is None"
    model = getattr(algo, 'model', None)
    if model is None: return False, "algo.model missing"
    try:
        with torch.no_grad():
            sd = model.state_dict()
            sd_cpu = {k: v.detach().to('cpu').clone() for k, v in sd.items()}
            payload = {'model': sd_cpu}
            for rms_key in ('running_mean_std', 'obs_rms', 'rms'):
                rms = getattr(model, rms_key, None)
                if rms is not None:
                    try:
                        rms_sd = rms.state_dict() if hasattr(rms, 'state_dict') else rms
                        if isinstance(rms_sd, dict):
                            for rk, rv in list(rms_sd.items()):
                                if torch.is_tensor(rv):
                                    rms_sd[rk] = rv.detach().to('cpu').clone()
                        payload[rms_key] = rms_sd
                        break
                    except Exception:
                        pass
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, out_path)
        return True, "cpu_state_dict"
    except Exception as e:
        return False, f"export exception: {e}"

def _ensure_csv_header(path: Path):
    """Rotate if wrong header and write the expected header."""
    try:
        if path.exists():
            with path.open("r") as f:
                first = f.readline()
            if first != _CSV_EXPECTED_HEADER:
                backup = path.with_suffix(path.suffix + ".old")
                shutil.move(str(path), str(backup))
                print(f"[csv] Rotated stale CSV header to: {backup.name}")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w") as f:
                f.write(_CSV_EXPECTED_HEADER)
    except Exception as e:
        print(f"[csv] WARNING could not validate header: {e}")

def _device_banner(selected_gpu_env: str, torch_device: torch.device):
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print("[device] CUDA_VISIBLE_DEVICES=", vis if vis != "" else "(unset)")
    if torch.cuda.is_available() and torch_device.type == "cuda":
        try: name = torch.cuda.get_device_name(0)
        except Exception: name = "(unknown)"
        print(f"[device] torch device: {torch_device} name: {name}")
        print(f"[device] mem_alloc={torch.cuda.memory_allocated(0)/1e9:.3f} GB, mem_reserved={torch.cuda.memory_reserved(0)/1e9:.3f} GB at t0")
    else:
        print(f"[device] torch device: {torch_device} (CPU)")

def _seed_banner(seed: int):
    mp = getattr(torch.backends, "cudnn", None)
    print("[seed] seed=", seed)
    print("[seed] torch.manual_seed set; numpy/random seeded; PYTHONHASHSEED set")
    try:
        print(f"[seed] cudnn.benchmark={bool(getattr(mp,'benchmark',None))} cudnn.deterministic={bool(getattr(mp,'deterministic',None))}")
    except Exception:
        pass
    try:
        print(f"[seed] matmul_precision={torch.get_float32_matmul_precision()}")
    except Exception:
        pass
    try:
        import rl_games  # type: ignore
        print(f"[versions] torch={torch.__version__} numpy={np.__version__} rl_games={getattr(rl_games,'__version__','(n/a)')}")
        try:
            import isaaclab  # type: ignore
            print(f"[versions] isaaclab={getattr(isaaclab,'__version__','(n/a)')}")
        except Exception:
            print("[versions] isaaclab=(unknown)")
    except Exception:
        print(f"[versions] torch={torch.__version__} numpy={np.__version__} rl_games=(unknown)")

def _install_signal_handlers():
    def _handler(signum, frame):
        global _SHUTDOWN_REQUESTED, _MODELS_DIR
        if _SHUTDOWN_REQUESTED:
            print(f"\n[signal] Received {signum} again → forcing exit.")
            os._exit(1)

        _SHUTDOWN_REQUESTED = True
        stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        base_dir = _MODELS_DIR if _MODELS_DIR is not None else _CKPT_DIR
        base = base_dir / f"interrupt_{stamp}"
        native_out = base
        sidecar_out = Path(str(base) + "_for_play.pth")
        print(f"\n[signal] Received {signum}. Saving interrupt checkpoint → {native_out}")

        ok, detail = _native_save(_GLOBAL_ALGO, str(native_out))
        if ok:
            _safe_write_line(_ROLLOUT_MIRROR_CSV, f"{time.time():.3f},CKPT,{native_out}\n")
            print(f"[signal] Saved native: {native_out} ({detail})")
        else:
            print(f"[signal] Save failed: {detail}")

        ok2, detail2 = _export_for_play_model(_GLOBAL_ALGO, str(sidecar_out))
        if ok2:
            _safe_write_line(_ROLLOUT_MIRROR_CSV, f"{time.time():.3f},CKPT,{sidecar_out}\n")
            print(f"[signal] Saved for-play sidecar: {sidecar_out} ({detail2})")
        else:
            print(f"[signal] For-play export failed: {detail2}")

        print("[signal] Will stop after the current step…")
        print("[signal] Next SIGINT will hard-exit.")

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def _load_for_play_state_dict(path: str) -> dict:
    try:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            if "model" in payload and isinstance(payload["model"], dict):
                return payload["model"]
            if "state_dict" in payload and isinstance(payload["state_dict"], dict):
                return payload["state_dict"]
        if isinstance(payload, dict) and all(torch.is_tensor(v) for v in payload.values()):
            return payload
    except Exception as e:
        print(f"[resume] Failed to load for_play checkpoint: {e}")
    return {}

def _get_or_create_run_root(checkpoints_parent: str, ckpt_label: str, seed: int) -> str:
    """
    Ensures ALL restarts reuse the SAME run directory (even across processes).
    Stores the chosen run root in a small pointer file:
      checkpoints_parent/.active_run__{ckpt_label}__seed{seed}.txt
    """
    os.makedirs(checkpoints_parent, exist_ok=True)
    key = f"{ckpt_label}__seed{seed}"
    ptr = os.path.join(checkpoints_parent, f".active_run__{key}.txt")

    # Reuse existing pointer if valid
    if os.path.exists(ptr):
        run_root = open(ptr, "r").read().strip()
        if run_root and os.path.isdir(run_root):
            return run_root

    # Otherwise create a new run root ONCE and persist it
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_root = os.path.join(checkpoints_parent, f"{ckpt_label}_{stamp}")
    os.makedirs(run_root, exist_ok=True)
    with open(ptr, "w") as f:
        f.write(run_root)

    return run_root

# ──────────────────────────────────────────────────────────────────────────────
# 15. NVTX helper (no-op if not requested)
# ──────────────────────────────────────────────────────────────────────────────
def _nvtx_range(label):
    if "--nvtx" in sys.argv or getattr(args, "nvtx", False):
        try:
            import torch
            return torch.cuda.nvtx.range(label)
        except Exception:
            class _Noop:
                def __enter__(self): return None
                def __exit__(self, *a): return False
            return _Noop()
    class _Noop:
        def __enter__(self): return None
        def __exit__(self, *a): return False
    return _Noop()


# ──────────────────────────────────────────────────────────────────────────────
# 20. CLI
# ──────────────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser("Unified Ant (walk → jump → spin) trainer — Isaac + rl_games")

# core run
p.add_argument("--task", type=str, default=DEFAULTS["core"]["task"])
p.add_argument("--gym_env_id", type=str, default=DEFAULTS["core"]["gym_env_id"])
p.add_argument("--cfg_yaml", type=str, required=True)
p.add_argument("--num_envs", type=int, default=DEFAULTS["core"]["num_envs"])
p.add_argument("--seed", type=int, default=DEFAULTS["core"]["seed"])
p.add_argument("--gpu", type=int, default=DEFAULTS["core"]["gpu"])
p.add_argument("--headless", action="store_true", default=DEFAULTS["core"]["headless"])

p.add_argument("--logdir", type=str, default=DEFAULTS["logging"]["logdir"])
p.add_argument("--run_name", type=str, default=DEFAULTS["logging"]["run_name"])
p.add_argument("--run_tag", type=str, default=DEFAULTS["logging"]["run_tag"])
p.add_argument("--log_window", type=int, default=DEFAULTS["logging"]["log_window"])
p.add_argument("--log_interval_s", type=float, default=DEFAULTS["logging"]["log_interval_s"])
p.add_argument("--return_torch", type=int, default=DEFAULTS["logging"]["return_torch"], choices=[0,1])
p.add_argument("--updates_per_step", type=int, default=DEFAULTS["grid"]["updates_per_step"],
               help="SAC updates_per_step for rl_games (writes to algo and algo.config).")

# NEW: allow overriding print cadence from CLI if needed
p.add_argument("--rb_print_every_n", type=int, default=DEFAULTS["logging"]["rb_print_every_n"],
               help="Print full reward breakdown every N rollout logs (1 = every log).")

# training horizon & bookkeeping
p.add_argument("--max_episodes", type=int, default=DEFAULTS["checkpoint"]["max_episodes"], help="Optional global cap (legacy; not used for plateau).")
p.add_argument("--manual_ckpt_every_eps", type=int, default=DEFAULTS["checkpoint"]["manual_ckpt_every_eps"], help="Manual checkpoint cadence; resets on phase entry.")
p.add_argument("--batch_scale", type=float, default=DEFAULTS["training"]["batch_scale"], help="Multiply YAML batch sizes (minibatch_size/mini_batch_size).")

# normalization (ALWAYS DISABLED by default; can opt-in)
p.add_argument("--disable_obs_norm", action="store_true", default=not DEFAULTS["normalization"]["disable_obs_norm_flag_default"], help="Disable observation normalization")
p.add_argument("--disable_rew_norm", action="store_true", default=not DEFAULTS["normalization"]["disable_rew_norm_flag_default"], help="Disable reward normalization")

# perf toggles
p.add_argument("--perf_diag", action="store_true", default=DEFAULTS["logging"]["perf_diag"])
p.add_argument("--nvtx", action="store_true", default=DEFAULTS["logging"]["nvtx"])
p.add_argument("--gpu_util", action="store_true", default=DEFAULTS["logging"]["gpu_util"])

# warmup control
p.add_argument("--override_warmup_steps", type=int, default=DEFAULTS["training"]["override_warmup_steps"], help="Force num_warmup_steps to a specific value")
p.add_argument("--force_no_warmup", action="store_true", default=DEFAULTS["training"]["force_no_warmup"], help="Start updating immediately (sets warmup to 0)")

# phases & scheduling
p.add_argument("--n_cycles", type=int, default=DEFAULTS["phases"]["n_cycles"], help="Number of full (walk→jump→spin) cycles")
p.add_argument("--phase_order", type=str, default=DEFAULTS["phases"]["phase_order"], help="Comma list of phases to run in order")

# ── TIMESTEP-BASED caps (remain as-is; used only for guards) ────────────────
p.add_argument("--plateau_window_steps", type=int, default=DEFAULTS["plateau_steps"]["plateau_window_steps"],
               help="Timestep window (legacy; kept for compatibility, not used by plateau decision).")
p.add_argument("--plateau_abs_std", type=float, default=DEFAULTS["plateau_steps"]["plateau_abs_std"],
               help="Legacy abs std over step-window (not used by plateau decision).")
p.add_argument("--plateau_rel_std", type=float, default=DEFAULTS["plateau_steps"]["plateau_rel_std"],
               help="Legacy rel std over step-window (not used by plateau decision).")
p.add_argument("--plateau_min_steps", type=int, default=DEFAULTS["plateau_steps"]["plateau_min_steps"],
               help="Minimum AGGREGATED env-steps in-phase before enabling plateau checks.")
p.add_argument("--max_steps_phase", type=int, default=DEFAULTS["plateau_steps"]["max_steps_phase"],
               help="Hard cap on AGGREGATED env-steps per phase (stop even if not plateau).")

# (legacy episode-based knobs kept for backward compatibility; not used)
p.add_argument("--plateau_window", type=int, default=200, help=argparse.SUPPRESS)
p.add_argument("--plateau_min_episodes", type=int, default=50_000, help=argparse.SUPPRESS)

# ── NEW: MuJoCo-style EPISODE-BASED plateau controls ─────────────────────────
p.add_argument("--plateau_episode_window", type=int, default=DEFAULTS["plateau_episodes"]["plateau_episode_window"],
               help="Episodes per window for plateau detection (uses two back-to-back windows).")
p.add_argument("--plateau_rel_change", type=float, default=DEFAULTS["plateau_episodes"]["plateau_rel_change"],
               help="Relative mean change threshold: |μ_recent-μ_prev| / (|μ_prev|+eps).")
p.add_argument("--plateau_std_coeff", type=float, default=DEFAULTS["plateau_episodes"]["plateau_std_coeff"],
               help="Std safeguard: |μ_recent-μ_prev| ≤ coeff * σ(combined two windows).")
p.add_argument("--plateau_min_return", type=float, default=DEFAULTS["plateau_episodes"]["plateau_min_return"],
               help="Minimum μ_recent to consider plateau (episode return scale).")

# replay buffers
p.add_argument("--replay_buffer_size", type=int, default=DEFAULTS["training"]["replay_buffer_size"])
p.add_argument("--reset_buffer_on_phase_entry", action="store_true", help="If set, clear the buffer when entering a phase")

# reward-breakdown (global)
p.add_argument("--log_reward_breakdown", action="store_true", default=DEFAULTS["logging"]["log_reward_breakdown"], help="Append reward component breakdown columns to rollout CSV")

# SPIN reward knobs (keep your exact defaults)
p.add_argument("--alpha_abs_wz", type=float, default=DEFAULTS["rewards"]["spin"]["alpha_abs_wz"], help="α · |ω_b,z|")
p.add_argument("--lambda_back", type=float, default=DEFAULTS["rewards"]["spin"]["lambda_back"], help="λ_back · max(0, −s · ω_b,z)")
p.add_argument("--lambda_vxy", type=float, default=DEFAULTS["rewards"]["spin"]["lambda_vxy"], help="λ_vxy · ||v_b,xy||")
p.add_argument("--lambda_jerk", type=float, default=DEFAULTS["rewards"]["spin"]["lambda_jerk"], help="λ_jerk · ||Δa||^2")
p.add_argument("--spin_direction", type=int, default=DEFAULTS["rewards"]["spin"]["spin_direction"], choices=[-1, +1], help="Chosen global spin direction s∈{+1,−1}")
# NEW: spin streak bonus knobs
p.add_argument(
    "--spin_streak_k",
    type=float,
    default=DEFAULTS["rewards"]["spin"]["streak_k"],
    help="Linear coefficient for progressive spin streak bonus (reward per second).",
)
p.add_argument(
    "--spin_streak_cap_s",
    type=float,
    default=DEFAULTS["rewards"]["spin"]["streak_cap_s"],
    help="Cap, in seconds, for the spin streak used in the bonus.",
)
p.add_argument(
    "--spin_streak_eps",
    type=float,
    default=DEFAULTS["rewards"]["spin"]["streak_eps"],
    help="Angular-velocity tolerance around zero for deciding good/bad spin direction.",
)



# JUMP reward knobs (keep your exact defaults)
p.add_argument("--beta_up", type=float, default=DEFAULTS["rewards"]["jump"]["beta_up"], help="β for ReLU(vz)")
p.add_argument("--lambda_drift", type=float, default=DEFAULTS["rewards"]["jump"]["lambda_drift"], help="λ for vxy_b (linear)")
p.add_argument("--lambda_jerk_jump", type=float, default=DEFAULTS["rewards"]["jump"]["lambda_jerk_jump"], help="λ for jerk penalty in JUMP phase")

# ergonomics
p.add_argument("--dry_run", action="store_true")

# ── [ADDED] video recording knobs ─────────────────────────────────────────────
p.add_argument("--record_every", type=int, default=0,
               help="0 disables. N=record a video after each behavior on every Nth cycle.")
p.add_argument("--video_gpu", type=int, default=None,
               help="GPU to use for video; defaults to --gpu if unset.")

p.add_argument("--ckpt_label", type=str, default=None,
               help="If set, checkpoints folder will be checkpoints/<ckpt_label>_[timestamp]/")
p.add_argument("--player_yaml", type=str,
               default="/home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_play_sac_ant.yaml",
               help="Player YAML used by the async video recorder (per-run override).")
p.add_argument("--video_wait_pct", type=float, default=10.0,
               help="Recorder waits until GPU utilization is below this percent.")
p.add_argument("--video_wait_s", type=int, default=30,
               help="How many consecutive seconds GPU util must stay below the threshold before recording.")

# ---- add near other CLI args ----
p.add_argument("--resume_from", type=str, default=None,
               help="Path to a *_for_play.pth whose 'model' state_dict will seed training.")
p.add_argument("--ckpt_dir_override", type=str, default=None,
               help="Absolute path to an existing checkpoints folder to reuse (no new timestamp).")

# [PHASE RETRY PATCH] --- CLI knobs
p.add_argument("--min_phase_mean_reward_on_switch", type=float, default=1000.0,
               help="If phase plateaus and μ_recent < this threshold, retry the phase by reloading the prior model.")
p.add_argument("--phase_retry_max", type=int, default=2,
               help="Max retries for a phase that plateaus below the minimum mean reward.")

p.add_argument(
    "--restart_min_frac",
    type=float,
    default=0.25,   # e.g., start checking after 10% of max steps
    help="Fraction of max_steps_phase before restart-floor logic activates.")


# add near the existing normalization args
p.add_argument("--enable_obs_norm", action="store_true", help="Enable observation normalization")
p.add_argument("--enable_rew_norm", action="store_true", help="Enable reward normalization")

# keep your existing args as-is ...

args = p.parse_args()

# add right after parse_args()
if getattr(args, "enable_obs_norm", False):
    args.disable_obs_norm = False
if getattr(args, "enable_rew_norm", False):
    args.disable_rew_norm = False

# # ALWAYS DISABLE obs/rew norm by default (you can re-enable by removing defaults below)
# if "--disable_obs_norm" not in sys.argv and "--enable_obs_norm" not in sys.argv:
#     setattr(args, "disable_obs_norm", DEFAULTS["normalization"]["disable_obs_norm_flag_default"])
# if "--disable_rew_norm" not in sys.argv and "--enable_rew_norm" not in sys.argv:
#     setattr(args, "disable_rew_norm", DEFAULTS["normalization"]["disable_rew_norm_flag_default"])


LOG_WINDOW = int(args.log_window)
LOG_INTERVAL_S = float(args.log_interval_s)
RETURN_TORCH = bool(args.return_torch)

# ──────────────────────────────────────────────────────────────────────────────
# 30. Seed hygiene & device
# ──────────────────────────────────────────────────────────────────────────────
def _set_global_seed(seed: int):
    """Set all RNGs + global _SEED to a given value."""
    global _SEED
    _SEED = int(seed)
    os.environ["PYTHONHASHSEED"] = str(_SEED)
    random.seed(_SEED)
    np.random.seed(_SEED)
    try:
        torch.manual_seed(_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

_SEED = int(args.seed)
_set_global_seed(_SEED)

# Respect external CUDA_VISIBLE_DEVICES if already set; otherwise honor --gpu
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

os.environ.setdefault("TORCH_ALLOW_TF32", "1")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────────────────────
# 40. Isaac helpers (5b: RESTART Isaac per phase; no Kit reuse)
# ──────────────────────────────────────────────────────────────────────────────
def _import_applauncher():
    try:
        from omni.isaac.lab.app import AppLauncher
        return AppLauncher
    except Exception:
        from isaaclab.app import AppLauncher
        return AppLauncher

_GLOBAL_APP = None  # current phase's SimulationApp/Kit (closed at end of phase)

def _drain_kit_frames(n: int = 3):
    """Advance Kit a few frames to flush pending USD/physics work."""
    try:
        import omni.kit.app
        app = omni.kit.app.get_app()
        for _ in range(max(0, int(n))):
            app.update()
    except Exception:
        pass

def _ensure_usd_stage():
    """
    Make sure there is an open USD stage. Create a fresh empty one if missing.
    Returns True if a stage is available after this call.
    """
    try:
        import omni.usd
        ctx = omni.usd.get_context()
        stage = None
        try:
            stage = ctx.get_stage()
        except Exception:
            stage = None
        if stage is None:
            if hasattr(ctx, "new_stage"):
                ctx.new_stage()
                _drain_kit_frames(2)
                print("[usd] Ensured new_stage() exists before env construction.")
            else:
                print("[usd] WARN: no get_stage/new_stage on context; proceeding.")
        return True
    except Exception as e:
        print(f"[usd] ensure stage warning: {e}")
        return False

def _shutdown_isaac_app(app):
    if app is None:
        return
    try:
        # optional: drain a couple frames
        for _ in range(3):
            try:
                app.update()
            except Exception:
                break

        try:
            app.close()
            print("[isaac] app.close() returned normally.")
        except SystemExit as se:
            # IMPORTANT: SystemExit is not an Exception; it will kill the script if uncaught
            print(f"[isaac] app.close() raised SystemExit(code={getattr(se, 'code', None)}). Swallowing to continue.")
        except BaseException as be:
            # catch-all for other non-Exception terminations
            print(f"[isaac] app.close() raised BaseException: {be}. Swallowing to continue.")
    except Exception as e:
        print(f"[isaac] shutdown warning: {e}")


def _boot_isaac_fresh():
    """
    5b implementation:
      - ALWAYS create a brand-new SimulationApp for the current phase
      - If an old one exists, close it first
    """
    global _GLOBAL_APP
    if _GLOBAL_APP is not None:
        _shutdown_isaac_app(_GLOBAL_APP)
        _GLOBAL_APP = None

    AppLauncher = _import_applauncher()
    AppLauncher.add_app_launcher_args = getattr(AppLauncher, "add_app_launcher_args", lambda *_: None)
    _GLOBAL_APP = AppLauncher(headless=args.headless).app

    # Force GPU PhysX via Carb settings (parity with your walk script)
    try:
        import carb
        cs = carb.settings.get_settings()
        cs.set("/physics/useGpuPipeline", True)
        cs.set("/omni/physics/useGpuPipeline", True)
        cs.set("/app/omniverseKit/forcePhysxGpuPipeline", True)
        print("[trainer] Forced PhysX GPU pipeline via Carb settings.")
    except Exception as e:
        print("[trainer] Could not set Carb GPU-PhysX settings:", e)

    # Ensure our repo scripts path (+ register tasks)
    repo_scripts = Path.home() / "projects" / "CreativeMachinesAnt" / "Isaac" / "scripts"
    if str(repo_scripts) not in sys.path:
        sys.path.insert(0, str(repo_scripts))
    from ant_cfg_registry import register_ant_cfg_tasks
    register_ant_cfg_tasks()

    return _GLOBAL_APP

def _tune_env_cfg(env_cfg):
    """
    Apply decimation, horizon, and termination tweaks (timeout-only) to env_cfg.
    Returns ep_len for reference.
    """
    # Decimation/frame-skip adjustments → DEFAULTS["env"]["decimation"]
    _set = False
    for path in ("control.decimation", "actions.decimation", "decimation"):
        obj = env_cfg
        parts = path.split(".")
        ok = True
        for name in parts[:-1]:
            if not hasattr(obj, name):
                ok = False
                break
            obj = getattr(obj, name)
        if ok and hasattr(obj, parts[-1]):
            setattr(obj, parts[-1], int(DEFAULTS["env"]["decimation"]))
            print(f"[cfg] set {path}={int(DEFAULTS['env']['decimation'])}")
            _set = True
            break
    if not _set and hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "substeps"):
        env_cfg.sim.substeps = max(
            2,
            int(getattr(env_cfg, "substeps", 1)) * int(DEFAULTS["env"]["decimation"])
        )
        print(f"[cfg] bumped sim.substeps to {env_cfg.sim.substeps}")

    # Fixed horizon
    if hasattr(env_cfg, "episode_length_steps"):
        env_cfg.episode_length_steps = int(DEFAULTS["env"]["episode_length_steps"])
    ep_len = int(getattr(env_cfg, "episode_length_steps", int(DEFAULTS["env"]["episode_length_steps"])))

    # Terminations: timeout only (if requested)
    if bool(DEFAULTS["env"]["timeout_only"]):
        terms = getattr(env_cfg, "terminations", None)
        if terms is not None:
            if hasattr(terms, "only_timeout"):
                terms.only_timeout = True
            for name in [
                "unhealthy_state","fall","unhealthy_tilt","unhealthy_height","out_of_bounds",
                "termination_height","termination_tilt"
            ]:
                t = getattr(terms, name, None)
                if t is not None:
                    for flag in ("enable","enabled","use"):
                        if hasattr(t, flag):
                            setattr(t, flag, False)

    return ep_len

def _phase_setup_isaac():
    """
    5b implementation:
      - Start a fresh Isaac SimulationApp *per phase*
      - Build env_cfg fresh per phase
      - Return (make_env_func, app) so caller can close app at phase end
    """
    import gc
    global _GLOBAL_ENV, _SEED, EP_LEN

    app = _boot_isaac_fresh()   # <-- brand new Kit for this phase
    _GLOBAL_ENV = None          # force brand-new gym env instance for this phase

    _ensure_usd_stage()
    _drain_kit_frames(1)

    from isaaclab_tasks.utils import parse_env_cfg
    env_cfg = parse_env_cfg(
        args.task,
        device=str(DEVICE),
        num_envs=int(args.num_envs),
        use_fabric=True,
    )

    # Set seed on cfg to avoid the “Seed not set” warning
    if hasattr(env_cfg, "seed"):
        try:
            env_cfg.seed = int(_SEED)
        except Exception:
            pass

    _tune_env_cfg(env_cfg)
    EP_LEN = int(getattr(env_cfg, "episode_length_steps", int(DEFAULTS["env"]["episode_length_steps"])))

    def make_env_func():
        global _GLOBAL_ENV, _SEED
        if _GLOBAL_ENV is None:
            _ensure_usd_stage()
            _drain_kit_frames(1)

            e = gym.make(
                args.gym_env_id,
                cfg=env_cfg,
                disable_env_checker=True,
                render_mode=None,
            )
            if len(e.observation_space.shape) == 2:
                obs_dim = e.observation_space.shape[1]
                e.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            if len(e.action_space.shape) == 2:
                act_dim = e.action_space.shape[1]
                low = np.asarray(e.action_space.low[0], dtype=np.float32)
                high = np.asarray(e.action_space.high[0], dtype=np.float32)
                e.action_space = gym.spaces.Box(low=low, high=high, shape=(act_dim,), dtype=np.float32)
            try:
                e.reset(seed=_SEED)
            except Exception:
                pass
            _GLOBAL_ENV = e
        return _GLOBAL_ENV

    test_env = make_env_func()
    assert hasattr(test_env.unwrapped, "num_envs")
    assert test_env.unwrapped.num_envs == args.num_envs, (
        f"env.num_envs={test_env.unwrapped.num_envs} but --num_envs={args.num_envs}"
    )
    print(
        f"[INFO] num_envs={test_env.unwrapped.num_envs}, "
        f"obs_dim={test_env.observation_space.shape[0]}, "
        f"act_dim={test_env.action_space.shape[0]}, device={DEVICE}"
    )

    gc.collect()
    _drain_kit_frames(1)
    return make_env_func, app




# ──────────────────────────────────────────────────────────────────────────────
# 60. Checkpoints dir & signal handlers
# ──────────────────────────────────────────────────────────────────────────────
from pathlib import Path
from datetime import datetime
import os

def _init_ckpt_dir() -> Path:
    """
    IMPORTANT: This MUST return the SAME directory across process restarts.
    We do that by storing the chosen run dir in a tiny pointer file:
      checkpoints/.active_run__{ckpt_label}__seed{seed}.txt
    """
    parent = Path("/home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints")
    parent.mkdir(parents=True, exist_ok=True)

    ckpt_label = str(getattr(args, "ckpt_label", "run"))
    seed = int(getattr(args, "seed", 0))

    ptr = parent / f".active_run__{ckpt_label}__seed{seed}.txt"

    # Reuse prior run dir if pointer exists + dir still exists
    if ptr.exists():
        run_root = ptr.read_text().strip()
        if run_root:
            p = Path(run_root)
            if p.is_dir():
                return p

    # Otherwise create a new run root ONCE, write the pointer, and reuse forever
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_root = parent / f"{ckpt_label}_{stamp}"
    run_root.mkdir(parents=True, exist_ok=True)
    ptr.write_text(str(run_root))
    return run_root


_CKPT_DIR = _init_ckpt_dir()

# NEW: create subfolders and set paths
_MODELS_DIR = _CKPT_DIR / "models"
_VIDEOS_DIR = _CKPT_DIR / "videos"
_GRAPHS_DIR = _CKPT_DIR / "graphs"
_FAILED_DIR = _CKPT_DIR / "failed"
for _d in (_MODELS_DIR, _VIDEOS_DIR, _GRAPHS_DIR, _FAILED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# These should now ALWAYS point into the pinned _CKPT_DIR
_PHASE_STATE_PATH = _CKPT_DIR / "phase_state.json"
_ROLLOUT_MIRROR_CSV = _CKPT_DIR / "rollout_log.csv"

def _safe_write_header():
    _ensure_csv_header(_ROLLOUT_MIRROR_CSV)

_safe_write_header()
_install_signal_handlers()


# ──────────────────────────────────────────────────────────────────────────────
# 70. OBS index constants (shared across phases)
# ──────────────────────────────────────────────────────────────────────────────
# (these were previously inline near rl_games integration)
_OBS_IDX_VX_B = 1       # body-frame vx
_OBS_IDX_VY_B = 2       # body-frame vy
_OBS_IDX_WZ_B = 6       # body-frame yaw rate
_OBS_IDX_POS_START = 7  # world position slice: x,y,z


# ──────────────────────────────────────────────────────────────────────────────
# 80. rl_games integration
# ──────────────────────────────────────────────────────────────────────────────
from rl_games.torch_runner import Runner
from rl_games.common import env_configurations, vecenv

# We will capture the algo instance when rl_games finishes constructing it,
# and (if provided) immediately load pending weights into it.
_PENDING_LOAD_SD = None  # will be set by _run_phase() BEFORE runner.run()

def _fingerprint_state_dict(sd) -> str:
    """
    Lightweight fingerprint: total param count + SHA1 of the first tensor bytes.
    Avoids extra imports at module top; imports hashlib locally.
    """
    import hashlib
    import torch as _torch

    if not isinstance(sd, dict):
        return "(no-sd)"
    # Total tensor params
    total = 0
    first_bytes = None
    for k, v in sd.items():
        if _torch.is_tensor(v):
            total += int(v.numel())
            if first_bytes is None:
                # Use up to ~100k elements to keep hashing cheap
                t = v.detach().to("cpu").contiguous().view(-1)
                if t.numel() > 100_000:
                    t = t[:100_000]
                first_bytes = t.numpy().tobytes()
        elif isinstance(v, dict):
            # some modules nest (e.g., running stats); include their tensors too
            for kk, vv in v.items():
                if _torch.is_tensor(vv):
                    total += int(vv.numel())
                    if first_bytes is None:
                        t = vv.detach().to("cpu").contiguous().view(-1)
                        if t.numel() > 100_000:
                            t = t[:100_000]
                        first_bytes = t.numpy().tobytes()
    if first_bytes is None:
        return f"(params={total}, sha1=none)"
    h = hashlib.sha1(first_bytes).hexdigest()[:12]
    return f"(params={total}, sha1={h})"

try:
    from rl_games.common.algo_observer import DefaultAlgoObserver
    _old_after_init = getattr(DefaultAlgoObserver, "after_init", None)

    def _hook_after_init(self, algo):
        """
        Runs once per Runner when the algorithm object is ready.
        We capture the algo for logging/ckpts and opportunistically load any
        pending weights queued by the phase orchestrator.
        """
        global _GLOBAL_ALGO, _GLOBAL_VECENV, _PENDING_LOAD_SD
        _GLOBAL_ALGO = algo
        try:
            if _GLOBAL_VECENV is not None:
                _GLOBAL_VECENV.set_algo(algo)
        except Exception:
            pass

        # If a prior phase queued weights to be loaded, do it now.
        if _PENDING_LOAD_SD:
            try:
                if hasattr(algo, "model") and hasattr(algo.model, "load_state_dict"):
                    algo.model.load_state_dict(_PENDING_LOAD_SD, strict=False)
                    fp = _fingerprint_state_dict(getattr(algo.model, "state_dict")())
                    print(f"[weights] Queued incoming weights were loaded into algo.model (strict=False). fp={fp}")
                else:
                    print("[weights] WARNING: algo.model not present; could not load queued weights.")
            except Exception as e:
                print(f"[weights] WARNING: loading queued weights failed: {e}")
            _PENDING_LOAD_SD = None  # clear after attempt

        if callable(_old_after_init):
            try:
                _old_after_init(self, algo)
            except TypeError:
                _old_after_init(self)

    DefaultAlgoObserver.after_init = _hook_after_init
    print("[observer-hook] Hooked DefaultAlgoObserver.after_init to capture algo and load queued weights.")
except Exception as _e:
    print(f"[observer-hook] Could not hook DefaultAlgoObserver: {_e}")

# Global phase runtime config populated by the phase orchestrator
_PHASE_RUNTIME = {
    "name": "walk",               # 'walk' | 'jump' | 'spin'
    "rollout_log_path": None,     # Path
    "manual_ckpt_every_eps": None,# int or None
    "nn_dir_path": None,          # str path for checkpoints
    "make_env_func": None,        # closure to build/get env for THIS phase
    "app": None,                  # informational handle
    "plateau": {                  # plateau config (TIMESTEP GUARDS + EPISODE DECISION)
        "window_steps": int(args.plateau_window_steps),
        "abs_std": float(args.plateau_abs_std),
        "rel_std": float(args.plateau_rel_std),
        "min_steps": int(args.plateau_min_steps),
        "max_steps": int(args.max_steps_phase) if args.max_steps_phase is not None else None,
        # MuJoCo-style decision knobs:
        "ep_window": int(args.plateau_episode_window),
        "rel_change": float(args.plateau_rel_change),
        "std_coeff": float(args.plateau_std_coeff),
        "min_return": float(args.plateau_min_return),
    },
    "reward": {                   # per-phase reward knobs
        # SPIN
        "alpha_abs_wz": float(args.alpha_abs_wz),
        "lambda_back": float(args.lambda_back),
        "lambda_vxy": float(args.lambda_vxy),
        "lambda_jerk": float(args.lambda_jerk),
        "spin_direction": int(args.spin_direction),
        "spin_streak_k": float(getattr(args, "spin_streak_k", DEFAULTS["rewards"]["spin"]["streak_k"])),
        "spin_streak_cap_s": float(getattr(args, "spin_streak_cap_s", DEFAULTS["rewards"]["spin"]["streak_cap_s"])),
        "spin_streak_eps": float(getattr(args, "spin_streak_eps", DEFAULTS["rewards"]["spin"]["streak_eps"])),

        # JUMP
        "beta_up": float(args.beta_up),
        "lambda_drift": float(args.lambda_drift),
        "lambda_jerk_jump": float(args.lambda_jerk_jump),
    },
    "log_reward_breakdown": bool(args.log_reward_breakdown),
    "cycle_idx": 0,  # <-- ADD
}

def _isaaclab_env_creator(**kwargs):
    """
    rl_games calls this to peek env spaces. We delegate to the active phase's
    make_env_func().
    """
    make_env_func = _PHASE_RUNTIME.get("make_env_func", None)
    if make_env_func is None:
        raise RuntimeError("make_env_func not set in _PHASE_RUNTIME before Runner.load()")
    return make_env_func()

def _local_vecenv(_config_name, _num_actors, **kwargs):
    """
    rl_games LOCAL vecenv factory for this phase. We inject the one
    phase-scoped env instance + plateau logic + reward shaping.
    """
    global _GLOBAL_VECENV
    make_env_func = _PHASE_RUNTIME.get("make_env_func", None)
    if make_env_func is None:
        raise RuntimeError("make_env_func not set in _PHASE_RUNTIME before creating vecenv")
    ve = _UnifiedRLgLocalVecEnv(
        env=make_env_func(),
        phase_name=str(_PHASE_RUNTIME["name"]),
        reward_cfg=_PHASE_RUNTIME["reward"],
        plateau_cfg=_PHASE_RUNTIME["plateau"],
        rollout_log_path=_PHASE_RUNTIME["rollout_log_path"],
        manual_ckpt_every_eps=_PHASE_RUNTIME["manual_ckpt_every_eps"],
        nn_dir_path=_PHASE_RUNTIME["nn_dir_path"],
        rb_enabled=bool(_PHASE_RUNTIME["log_reward_breakdown"]),
        use_obs_norm=not args.disable_obs_norm,
        use_rew_norm=not args.disable_rew_norm,
        # add this:
        cycle_idx=int(_PHASE_RUNTIME.get("cycle_idx", 0)),
    )

    _GLOBAL_VECENV = ve
    try:
        if _GLOBAL_ALGO is not None:
            _GLOBAL_VECENV.set_algo(_GLOBAL_ALGO)
    except Exception as e:
        print(f"[vecenv] Warning: could not set algo on vecenv: {e}")
    return _GLOBAL_VECENV

# Register the env family so rl_games can find it.
env_configurations.register('isaaclab', {'env_creator': _isaaclab_env_creator, 'vecenv_type': 'LOCAL'})

# Register LOCAL vecenv
from rl_games.common import vecenv as _rlg_vecenv
try:
    _rlg_vecenv.vecenv_config['LOCAL'] = _local_vecenv
except Exception:
    try:
        _rlg_vecenv.register('LOCAL', _local_vecenv)
    except Exception:
        _rlg_vecenv.create_local = _local_vecenv


# ──────────────────────────────────────────────────────────────────────────────
# 85. Local VecEnv (phase-aware, separate reward shaping + plateau detection)
# ──────────────────────────────────────────────────────────────────────────────
class _RunningMeanStd(torch.nn.Module):
    def __init__(self, shape, device, eps: float = 1e-8):
        super().__init__()
        self.register_buffer("_mean", torch.zeros(shape, dtype=torch.float32, device=device))
        self.register_buffer("_var", torch.ones(shape, dtype=torch.float32, device=device))
        self.register_buffer("_count", torch.tensor(eps, dtype=torch.float32, device=device))
        self._eps = eps
    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.detach()
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = torch.tensor(x.shape[0], dtype=torch.float32, device=x.device)
        delta = batch_mean - self._mean
        tot = self._count + batch_count
        new_mean = self._mean + delta * (batch_count / tot)
        m_a = self._var * self._count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta**2) * (self._count * batch_count / tot)
        new_var = M2 / tot
        self._mean.copy_(new_mean)
        self._var.copy_(torch.clamp(new_var, min=self._eps))
        self._count.copy_(tot)
    @property
    def mean(self): return self._mean
    @property
    def var(self): return self._var
    @property
    def std(self): return torch.sqrt(self._var + self._eps)
    def normalize(self, x: torch.Tensor, clip: float | None = None):
        z = (x - self._mean) / self.std
        if clip is not None:
            z = torch.clamp(z, -clip, clip)
        return z


class _UnifiedRLgLocalVecEnv:
    """
    One-phase wrapper that:
      • applies phase-specific reward shaping
      • logs consistent rollout CSV (+optional reward breakdown)
      • detects plateau via MuJoCo-style EPISODE windows (with timestep guards)
      • optionally normalizes obs/reward for the learner path (off by default)
    """
    def __init__(self, env, phase_name: str, reward_cfg: dict, plateau_cfg: dict,
                 rollout_log_path: Path | None, manual_ckpt_every_eps: int | None,
                 nn_dir_path: str | None, rb_enabled: bool,
                 use_obs_norm: bool, use_rew_norm: bool,
                 cycle_idx: int = 0):  # <-- ADD
        self.env = env
        self.num_agents = 1
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._cycle_idx = int(cycle_idx)  # <-- ADD

        self._phase = str(phase_name).lower().strip()
        self._reward_cfg = reward_cfg
        self._plateau = plateau_cfg
        self._rb_enabled = bool(rb_enabled)

        self._n = int(getattr(self.env.unwrapped, "num_envs", 1))
        dev = getattr(self.env.unwrapped, 'device', 'cpu')
        self._dev = dev if isinstance(dev, torch.device) else torch.device(dev)
        # spin streak (per-env consecutive steps spinning the right way)
        self._spin_streak_steps = torch.zeros(self._n, dtype=torch.float32, device=self._dev)


        # per-env episode accounting (UNNORMALIZED reward)
        self._ep_ret = torch.zeros(self._n, dtype=torch.float32, device=self._dev)
        self._ep_len = torch.zeros(self._n, dtype=torch.int32, device=self._dev)

        # logging windows (episode-level)
        self._hist_ret = deque(maxlen=int(args.log_window))
        self._hist_len = deque(maxlen=int(args.log_window))

        # TIMESTEP-BASED (legacy) step window — retained only for logging/hygiene
        self._plateau_steps_deque = deque(maxlen=int(self._plateau.get("window_steps", 100_000)))

        # EPISODE-BASED window for plateau decision
        self._ep_window = int(self._plateau.get("ep_window", 200))
        self._ep_returns = deque(maxlen=2 * self._ep_window)

        self._last_log_ts = time.time()

        self._episodes_done = 0
        self._total_steps = 0
        self._t0 = time.time()

        # manual checkpoint cadence (resets per phase)
        self._manual_interval = int(manual_ckpt_every_eps) if manual_ckpt_every_eps else None
        self._next_manual_ep = self._manual_interval if self._manual_interval else None

        self._rollout_log_path = Path(rollout_log_path) if rollout_log_path else None
        self._nn_dir = Path(nn_dir_path) if nn_dir_path else None
        self._algo = None

        # mean vx logging
        self._vx_sum = torch.tensor(0.0, device=self._dev)
        self._vx_count = torch.tensor(0, dtype=torch.int64, device=self._dev)
        self._vx_reader = None
        self._vz_reader = None
        self._last_info = None

        # [PHASE RETRY PATCH]
        self._last_plateau_mu_recent = None
        self._last_window_mean_reward = None  # NEW: last avg_return_window we computed


        self._done_term = 0
        self._done_trunc = 0
        self._first3_counter = 0

        # normalization toggles & rms
        self._use_obs_norm = bool(use_obs_norm)
        self._use_rew_norm = bool(use_rew_norm)
        self._obs_clip = 10.0
        self._rew_clip = 10.0
        self._gamma = 0.99
        obs_shape = (self.observation_space.shape[0],)
        self._obs_rms = _RunningMeanStd(obs_shape, self._dev)
        self._rew_rms = _RunningMeanStd((1,), self._dev)
        self._running_ret = torch.zeros(self._n, dtype=torch.float32, device=self._dev)

        # prev action & FD buffers (jump/spin jerk and vz fallback)
        self._prev_action = torch.zeros((self._n, self.action_space.shape[0]), dtype=torch.float32, device=self._dev)
        self._prev_pos_w = None
        self._dt = self._detect_dt()

        # reward breakdown helper (global toggle)
        self._rb = RewardBreakdown(self._n, self._dev, window_size=int(args.log_window)) if self._rb_enabled else None
        self._csv_header_frozen = False  # for dynamic header extension

        # ---- PRIME RB SCHEMA WITH ALL PHASE TERMS + KIN KEYS (zeros now) ----
        if self._rb_enabled:
            self._prime_rb_schema()

        # write base header for rollout CSV (mirror handled at module level)
        if self._rollout_log_path:
            _ensure_csv_header(self._rollout_log_path)

        # RB print cadence
        self._rb_print_every = max(1, int(getattr(args, "rb_print_every_n", 1)))
        self._rb_print_tick = 0

        # Restart floor logic
        self._restart_floor_steps = 20_000_000    # need this many "bad" steps in a row
        self._restart_floor_return = 500.0        # avg_return_window must stay below this
        # restart-floor activation threshold = fraction of max_steps_phase
        max_steps_phase = int(args.max_steps_phase) if args.max_steps_phase is not None else 0
        frac = float(getattr(args, "restart_min_frac", 0.10))
        self._restart_min_steps = int(max_steps_phase * frac)

        self._restart_bad_steps = 0               # accumulated bad steps since last good window
        self._last_restart_check_steps = 0        # total_steps at last restart check


    # ---- standard glue ----
    def set_algo(self, algo): self._algo = algo

    def get_env_info(self):
        return {'observation_space': self.observation_space, 'action_space': self.action_space, 'agents': self.num_agents}

    def close(self):
        try: self.env.close()
        except Exception: pass

    # ---- tiny helpers ----
    def _detect_dt(self) -> float:
        u = getattr(self.env, "unwrapped", None)
        for name in ("step_dt","dt","physics_dt","sim_dt","control_dt"):
            if hasattr(u, name):
                try:
                    val = float(getattr(u, name))
                    if val > 0: return val
                except Exception:
                    pass
        return 1.0/60.0

    def _extract_obs(self, o):
        if isinstance(o, dict):
            for k in ('obs', 'policy', 'observation'):
                if k in o:
                    o = o[k]
                    break
            else:
                vals = list(o.values())
                if any(torch.is_tensor(v) for v in vals):
                    vals = [v if torch.is_tensor(v) else torch.as_tensor(v, device=self._dev) for v in vals]
                    o = torch.cat(vals, dim=-1)
                else:
                    o = torch.as_tensor(np.concatenate(vals, axis=-1), device=self._dev)
        return o

    # ---- RB schema priming (register all columns upfront with zeros) ----
    def _prime_rb_schema(self):
        zeros = torch.zeros(self._n, dtype=torch.float32, device=self._dev)
        comps = {
            # WALK (env reward passthrough)
            "walk_env_rew": zeros,
            # JUMP
            "jump_up_term": zeros,
            "jump_drift_xy_pen": zeros,
            "jump_jerk_pen": zeros,
            # SPIN
            "spin_spin_speed": zeros,
            "spin_reverse_pen": zeros,
            "spin_trans_pen": zeros,
            "spin_jerk_pen": zeros,
            "spin_streak_bonus": zeros,   # NEW
        }

        kin = {
            "vx_b": zeros,
            "vy_b": zeros,
            "vxy_b": zeros,
            "wz_b": zeros,
            "vz_fd": zeros,
            # NEW: upward-only vertical velocity (ReLU(vz))
            "vz_up": zeros,
        }
        self._rb.accumulate_step(comps, kin)  # registers keys without changing totals

    # ---- normalization ----
    @torch.no_grad()
    def _normalize_obs(self, obs_t: torch.Tensor) -> torch.Tensor:
        if not self._use_obs_norm: return obs_t
        if obs_t.ndim == 1:
            x = obs_t.unsqueeze(0)
            self._obs_rms.update(x)
            out = self._obs_rms.normalize(obs_t, self._obs_clip)
        else:
            self._obs_rms.update(obs_t)
            out = self._obs_rms.normalize(obs_t, self._obs_clip)
        return out

    @torch.no_grad()
    def _normalize_rew(self, rew_t: torch.Tensor, term_t: torch.Tensor, trunc_t: torch.Tensor) -> torch.Tensor:
        if not self._use_rew_norm: return rew_t
        self._running_ret.mul_(self._gamma).add_(rew_t)
        self._rew_rms.update(self._running_ret.view(-1, 1))
        finished = (term_t | trunc_t)
        if torch.any(finished):
            self._running_ret[finished] = 0.0
        std = torch.clamp(self._rew_rms.std.squeeze(0), min=1e-8)
        norm_rew = rew_t / std
        return torch.clamp(norm_rew, -self._rew_clip, self._rew_clip)

    # ---- vx/vz readers ----
    def _choose_vx_reader_once(self, info) -> None:
        global _VX_SOURCE_PRINTED, _VX_SOURCE_NAME
        if self._vx_reader is not None: return
        u = getattr(self.env, "unwrapped", None)
        def _getattr_path(root, dotted):
            cur = root
            for part in dotted.split("."):
                if not hasattr(cur, part): return None
                cur = getattr(cur, part)
            return cur
        for dotted in (
            "robot.data.root_lin_vel_b",
            "robot.data.base_lin_vel_b",
            "robot.data.root_com_lin_vel_b",
            "robot.data.root_link_lin_vel_b",
            "vel_loc",
        ):
            v = _getattr_path(u, dotted)
            if v is not None:
                def _reader_envbuf(src_path=dotted):
                    arr = _getattr_path(self.env.unwrapped, src_path)
                    if torch.is_tensor(arr): return arr[:, 0].sum(), torch.tensor(arr.shape[0], device=arr.device)
                    if isinstance(arr, np.ndarray): return torch.tensor(arr[:, 0].sum(), device=self._dev), torch.tensor(arr.shape[0], device=self._dev)
                    return torch.tensor(0.0, device=self._dev), torch.tensor(0, device=self._dev)
                self._vx_reader = _reader_envbuf
                if not _VX_SOURCE_PRINTED:
                    frame = "body-frame" if "lin_vel_b" in dotted else "world-frame"
                    _VX_SOURCE_NAME = f"{dotted}[:,0] ({frame})"
                    print(f"[vx_b] using env buffer: {_VX_SOURCE_NAME}")
                    _VX_SOURCE_PRINTED = True
                return
        # fallback via info dicts
        candidates = [
            ("extras", "lin_vel_b"), ("extras", "base_lin_vel_b"), ("extras", "root_lin_vel_b"),
            ("metrics", "lin_vel_b"), ("lin_vel_b",), ("base_lin_vel_b",), ("root_lin_vel_b",),
            ("base_lin_vel",), ("root_lin_vel",), ("lin_vel",),
        ]
        for path in candidates:
            cur = info; ok = True
            for k in path:
                if isinstance(cur, dict) and k in cur: cur = cur[k]
                else: ok = False; break
            if ok:
                def _reader_info(src_path=path):
                    cur2 = self._last_info
                    if not isinstance(cur2, dict): return torch.tensor(0.0, device=self._dev), torch.tensor(0, device=self._dev)
                    try:
                        t = cur2
                        for k in src_path: t = t[k]
                        if torch.is_tensor(t):
                            if t.ndim == 2: return t[:, 0].sum(), torch.tensor(t.shape[0], device=t.device)
                            if t.ndim == 1: return (t[0] * self._n), torch.tensor(self._n, device=self._dev)
                        elif isinstance(t, np.ndarray):
                            if t.ndim == 2: return torch.tensor(t[:, 0].sum(), device=self._dev), torch.tensor(t.shape[0], device=self._dev)
                            if t.ndim == 1: return torch.tensor(float(t[0]) * self._n, device=self._dev), torch.tensor(self._n, device=self._dev)
                    except Exception:
                        pass
                    return torch.tensor(0.0, device=self._dev), torch.tensor(0, device=self._dev)
                self._vx_reader = _reader_info
                if not _VX_SOURCE_PRINTED:
                    frame = "body-frame" if "lin_vel_b" in ".".join(path) else "world-frame"
                    _VX_SOURCE_NAME = ".".join(path) + f"[:,0] ({frame})"
                    print(f"[vx_b] using info path: {_VX_SOURCE_NAME}")
                    _VX_SOURCE_PRINTED = True
                return
        if not _VX_SOURCE_PRINTED:
            print("[vx_b] WARNING: could not determine velocity source; mean_vx_b will be NaN")
            _VX_SOURCE_PRINTED = True
        self._vx_reader = lambda: (torch.tensor(0.0, device=self._dev), torch.tensor(0, device=self._dev))

    def _choose_vz_reader_once(self):
        if self._vz_reader is not None: return
        u = getattr(self.env, "unwrapped", None)
        self._dt_cached = self._detect_dt()
        def _getattr_path(root, dotted):
            cur = root
            for part in dotted.split("."):
                if not hasattr(cur, part): return None
                cur = getattr(cur, part)
            return cur
        for dotted in (
            "robot.data.root_lin_vel_w",
            "robot.data.root_link_lin_vel_w",
            "robot.data.base_lin_vel_w",
            "robot.data.root_com_lin_vel_w",
            "vel_w",
        ):
            v = _getattr_path(u, dotted)
            if v is not None:
                def _reader_envbuf(_pos_w_unused, _dpos_unused, src_path=dotted):
                    arr = _getattr_path(self.env.unwrapped, src_path)
                    if torch.is_tensor(arr):   return arr[:, 2]
                    if isinstance(arr, np.ndarray): return torch.as_tensor(arr[:, 2], device=self._dev)
                    return torch.zeros(self._n, device=self._dev)
                self._vz_reader = _reader_envbuf
                self._vz_src_name = f"{dotted}[:,2] (world-frame)"
                print(f"[vz] using env buffer: {self._vz_src_name}")
                return
        def _reader_fd(pos_w, dpos):
            dt = self._dt_cached if (self._dt_cached is not None) else 1.0/60.0
            return dpos[:, 2] / max(1e-8, dt)
        self._vz_reader = _reader_fd
        self._vz_src_name = "FD(pos_w.z)/dt (world-frame)"
        print(f"[vz] using fallback: {self._vz_src_name}")

    # ---- rollout CSV util ----
    def _write_both_csvs(self, line: str):
        if _ROLLOUT_LOG_PATH is not None:
            _safe_write_line(_ROLLOUT_LOG_PATH, line)
        _safe_write_line(_ROLLOUT_MIRROR_CSV, line)

    # ---- plateau detection ----
    def _check_plateau(self) -> bool:
        # Guard: require enough AGGREGATED steps seen in-phase
        min_steps = int(self._plateau.get("min_steps", 0))
        if self._total_steps < max(1, min_steps):
            return False

        # Decision: require 2 * ep_window episode returns
        ep_w = int(self._ep_window)
        if len(self._ep_returns) < 2 * ep_w:
            return False

        arr = np.asarray(self._ep_returns, dtype=np.float64)
        prev = arr[-2 * ep_w : -ep_w]
        recent = arr[-ep_w :]
        mu_p = float(prev.mean())
        mu_r = float(recent.mean())
        sigma = float(arr[-2 * ep_w :].std(ddof=0))
        rel_change = abs(mu_r - mu_p) / max(1e-8, abs(mu_p))

        self._last_plateau_mu_recent = mu_r

        min_ret = float(self._plateau.get("min_return", 0.0))
        rel_thr = float(self._plateau.get("rel_change", 0.05))
        std_coeff = float(self._plateau.get("std_coeff", 1.0))



        cond_min = (mu_r >= min_ret)
        cond_rel = (rel_change <= rel_thr)
        cond_std = (abs(mu_r - mu_p) <= std_coeff * sigma)

        return cond_min and cond_rel and cond_std

    # [PHASE RETRY PATCH]
    def get_last_plateau_mu(self):
        return self._last_plateau_mu_recent
    
    def get_last_window_mean_reward(self):
        # Prefer the most recent logged window mean; otherwise compute from history if possible.
        if self._last_window_mean_reward is not None:
            return float(self._last_window_mean_reward)
        if len(self._hist_ret) > 0:
            try:
                return float(np.mean(self._hist_ret))
            except Exception:
                return None
        return None

    # ---- reset ----
    def reset(self):
        if _SHUTDOWN_REQUESTED: raise StopIteration("INTERRUPTED")
        obs, _ = self.env.reset(seed=_SEED)
        self._ep_ret.zero_(); self._ep_len.zero_()
        self._running_ret.zero_()
        self._prev_action.zero_()
        self._vx_sum.zero_(); self._vx_count.zero_()
        self._done_term = 0; self._done_trunc = 0
        self._spin_streak_steps.zero_()
        if hasattr(self, "_plateau_steps_deque"):
            self._plateau_steps_deque.clear()
        self._hist_ret.clear(); self._hist_len.clear()
        self._first3_counter = 0
        # NOTE: _ep_returns persists across the phase; do not clear here.

        obs = self._extract_obs(obs)
        if not torch.is_tensor(obs): obs = torch.as_tensor(obs, device=self._dev)

        # FD buffers & readers
        if obs.shape[1] >= (_OBS_IDX_POS_START + 3):
            self._prev_pos_w = obs[:, _OBS_IDX_POS_START:_OBS_IDX_POS_START+3].clone()
        else:
            self._prev_pos_w = torch.zeros((self._n, 3), dtype=torch.float32, device=self._dev)
        self._choose_vz_reader_once()

        out = self._normalize_obs(obs)
        return out if RETURN_TORCH else out.detach().cpu().numpy()

    # ---- manual checkpoint ----
    def _save_native(self, path: str) -> tuple[bool, str]:
        return _native_save(self._algo, path)

    def _maybe_manual_checkpoint(self, mean_window_reward: float | None):
        if self._manual_interval is None or self._algo is None or self._nn_dir is None:
            return
        if self._episodes_done >= self._next_manual_ep:
            stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            fname = f"{self._phase}_c{self._cycle_idx+1}_manual_ep_{self._next_manual_ep:06d}"

            if mean_window_reward is not None:
                fname += f"_rew_{mean_window_reward:.5f}"
            out_path = self._nn_dir / f"{fname}"
            self._nn_dir.mkdir(parents=True, exist_ok=True)
            ok, detail = self._save_native(str(out_path))
            if ok:
                print(f"[manual-ckpt] phase={self._phase} episodes={self._episodes_done} → saved: {out_path} ({detail})")
                _safe_write_line(_ROLLOUT_MIRROR_CSV, f"{time.time():.3f},CKPT,{out_path}\n")
            else:
                print(f"[manual-ckpt] FAILED at episodes={self._episodes_done}: {detail}")
            for_play = self._nn_dir / f"{fname}_for_play.pth"
            ok2, detail2 = _export_for_play_model(self._algo, str(for_play))
            if ok2:
                print(f"[manual-ckpt] wrote for-play sidecar: {for_play} ({detail2})")
                _safe_write_line(_ROLLOUT_MIRROR_CSV, f"{time.time():.3f},CKPT,{for_play}\n")
            else:
                print(f"[manual-ckpt] for-play export FAILED: {detail2}")
            self._next_manual_ep += self._manual_interval

    # ---- step ----
    def step(self, actions):
        if _SHUTDOWN_REQUESTED: raise StopIteration("INTERRUPTED")

        # to tensor
        if isinstance(actions, np.ndarray):
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self._dev)
        elif not torch.is_tensor(actions):
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self._dev)

        with _nvtx_range("env.step"):
            obs, env_rew, term, trunc, info = self.env.step(actions)

        # vx log
        self._last_info = info
        if self._vx_reader is None: self._choose_vx_reader_once(info)
        try:
            sx, cnt = self._vx_reader()
            self._vx_sum += sx; self._vx_count += cnt
        except Exception:
            pass

        # tensors
        obs = self._extract_obs(obs)
        if not torch.is_tensor(obs):   obs = torch.as_tensor(obs, device=self._dev)
        if not torch.is_tensor(term):  term = torch.as_tensor(term, dtype=torch.bool, device=self._dev)
        if not torch.is_tensor(trunc): trunc = torch.as_tensor(trunc, dtype=torch.bool, device=self._dev)
        if not torch.is_tensor(env_rew): env_rew = torch.as_tensor(env_rew, dtype=torch.float32, device=self._dev)

        # common kinematics
        vx_b = obs[:, _OBS_IDX_VX_B] if obs.shape[1] > _OBS_IDX_VX_B else torch.zeros(self._n, device=self._dev)
        vy_b = obs[:, _OBS_IDX_VY_B] if obs.shape[1] > _OBS_IDX_VY_B else torch.zeros(self._n, device=self._dev)
        vxy_b = torch.sqrt(torch.clamp(vx_b * vx_b + vy_b * vy_b, min=0.0))
        wz_b = obs[:, _OBS_IDX_WZ_B] if obs.shape[1] > _OBS_IDX_WZ_B else torch.zeros(self._n, device=self._dev)

        # pos & FD for jump vertical velocity
        if obs.shape[1] >= (_OBS_IDX_POS_START + 3):
            pos_w = obs[:, _OBS_IDX_POS_START:_OBS_IDX_POS_START+3]
        else:
            pos_w = torch.zeros((self._n, 3), dtype=torch.float32, device=self._dev)
        if self._prev_pos_w is None: self._prev_pos_w = pos_w.clone()
        dpos = pos_w - self._prev_pos_w
        self._prev_pos_w.copy_(pos_w)
        if self._vz_reader is None: self._choose_vz_reader_once()
        vz_src = self._vz_reader(pos_w, dpos)
        vz_up = torch.clamp(vz_src, min=0.0)  # NEW: upward-only speed (ReLU(vz))

        # ===== Phase rewards =====
        comps = {}
        if self._phase == "walk":
            # Pass-through env reward; expose as component for RB with phase prefix
            rew = env_rew
            comps["walk_env_rew"] = env_rew
        elif self._phase == "jump":
            up_term      = float(self._reward_cfg["beta_up"])      * vz_up
            drift_xy_pen = float(self._reward_cfg["lambda_drift"]) * vxy_b
            jerk_pen     = float(self._reward_cfg["lambda_jerk_jump"]) * torch.sum((actions - self._prev_action) ** 2, dim=1)
            rew = up_term - drift_xy_pen - jerk_pen
            # Phase-prefixed keys
            comps.update({
                "jump_up_term": up_term,
                "jump_drift_xy_pen": drift_xy_pen,
                "jump_jerk_pen": jerk_pen
            })
        elif self._phase == "spin":
            s = float(self._reward_cfg["spin_direction"])  # +1 or -1
            dir_wz = s * wz_b  # >0 when spinning in the desired direction

            # --- progressive spin streak bonus ---
            eps   = float(self._reward_cfg.get("spin_streak_eps", 0.1))
            k     = float(self._reward_cfg.get("spin_streak_k", 0.05))
            cap_s = float(self._reward_cfg.get("spin_streak_cap_s", 5.0))

            good = dir_wz > eps       # spinning clearly in the desired direction
            bad  = dir_wz < -eps      # spinning clearly in the opposite direction

            # update streak (in steps)
            self._spin_streak_steps[good] += 1
            self._spin_streak_steps[bad] = 0
            # |dir_wz| <= eps leaves the streak unchanged

            # convert to seconds and compute bounded bonus
            streak_sec = self._spin_streak_steps * self._dt
            streak_bonus = k * torch.clamp(streak_sec, max=cap_s)
            # --------------------------------------

            # Reward only for spinning in the desired direction
            spin_speed  = float(self._reward_cfg["alpha_abs_wz"]) * torch.clamp(dir_wz, min=0.0)

            # Extra penalty for spinning the wrong way
            reverse_pen = float(self._reward_cfg["lambda_back"]) * torch.clamp(-dir_wz, min=0.0)

            trans_pen   = float(self._reward_cfg["lambda_vxy"])   * vxy_b
            jerk_pen    = float(self._reward_cfg["lambda_jerk"])  * torch.sum((actions - self._prev_action) ** 2, dim=1)

            # add streak bonus
            rew = spin_speed - reverse_pen - trans_pen - jerk_pen + streak_bonus

            comps.update({
                "spin_spin_speed": spin_speed,
                "spin_reverse_pen": reverse_pen,
                "spin_trans_pen": trans_pen,
                "spin_jerk_pen": jerk_pen,
                "spin_streak_bonus": streak_bonus,
            })

        # update prev action
        self._prev_action.copy_(actions)

        # RB accumulation (auto-discover still works; we’ve already primed schema)
        if self._rb_enabled:
            kin = {
                "vx_b": vx_b, "vy_b": vy_b, "vxy_b": vxy_b, "wz_b": wz_b,
                "vz_fd": vz_src, "vz_up": vz_up  # include upward-only mean metric
            }
            self._rb.accumulate_step(comps, kin)

        # normalize for learning path if enabled (logging uses raw `rew`)
        obs_out = self._normalize_obs(obs)
        rew_out = self._normalize_rew(rew, term, trunc)

        # bookkeeping/logging on raw reward + plateau check
        self._account_and_maybe_log_and_stop(rew, term, trunc)

        if RETURN_TORCH:
            return obs_out, rew_out, (term | trunc), {}
        else:
            obs_np = obs_out.detach().cpu().numpy()
            rew_np = rew_out.detach().cpu().numpy().astype(np.float32, copy=False)
            return obs_np, rew_np, (term.cpu().numpy() | trunc.cpu().numpy()), {}

    # ---- accounting + logging ----
    def _account_and_maybe_log_and_stop(self, rew_env_t: torch.Tensor, term_t: torch.Tensor, trunc_t: torch.Tensor):
        self._ep_ret += rew_env_t
        self._ep_len += 1
        self._total_steps += self._n

        # (legacy) per-step mean reward across all envs — retained for hygiene
        try:
            self._plateau_steps_deque.append(float(rew_env_t.mean().item()))
        except Exception:
            pass

        # Optional hard cap on steps per phase (AGGREGATED across envs)
        max_steps = self._plateau.get("max_steps", None)
        if max_steps is not None and self._total_steps >= int(max_steps):
            raise StopIteration("STOP_AFTER_STEPS")

        finished_mask = (term_t | trunc_t)
        if torch.any(finished_mask):
            if self._rb_enabled:
                self._rb.finish_episodes(finished_mask)
            # reset spin streak for finished envs
            self._spin_streak_steps[finished_mask] = 0
            idx = torch.where(finished_mask)[0]
            self._episodes_done += int(idx.numel())
            self._done_term += int(term_t[idx].sum().item())
            self._done_trunc += int(trunc_t[idx].sum().item())
            for i in idx.tolist():
                ret_i = float(self._ep_ret[i].item())
                len_i = int(self._ep_len[i].item())
                self._hist_ret.append(ret_i)
                self._hist_len.append(len_i)
                # EPISODE-BASED plateau window feed
                self._ep_returns.append(ret_i)

                if self._first3_counter < 3:
                    print(f"[first-episodes/{self._phase}] return={ret_i:.3f} len={len_i}")
                    self._first3_counter += 1
                self._ep_ret[i] = 0.0
                self._ep_len[i] = 0

        mean_window_reward = None
        now = time.time()
        if (now - self._last_log_ts) >= LOG_INTERVAL_S and len(self._hist_ret) > 0:
            avg_r = float(np.mean(self._hist_ret))
            avg_l = float(np.mean(self._hist_len))
            self._last_window_mean_reward = avg_r  # NEW


            # --- Restart floor: require sustained bad reward over a window of steps ---
            # Compute how many env-steps have occurred since the last restart check.
            steps_since_last = self._total_steps - self._last_restart_check_steps
            if steps_since_last < 0:
                steps_since_last = 0
            self._last_restart_check_steps = self._total_steps

            if avg_r < self._restart_floor_return:
                self._restart_bad_steps += steps_since_last
            else:
                # Good window → reset bad streak
                self._restart_bad_steps = 0

            if (
                self._total_steps >= self._restart_min_steps and
                self._restart_bad_steps >= self._restart_floor_steps
            ):
                print(
                    f"[RESTART_FLOOR/{self._phase}] total_steps={self._total_steps} "
                    f"bad_steps={self._restart_bad_steps} avg_return_window={avg_r:.2f} "
                    f"< {self._restart_floor_return} for ≈{self._restart_floor_steps} steps "
                    f"→ requesting phase restart."
                )
                raise StopIteration("PHASE_RESTART_FLOOR")
            mean_window_reward = avg_r
            vx_cnt = max(1, int(self._vx_count.item()))
            mean_vx = float((self._vx_sum / vx_cnt).item()) if vx_cnt > 0 else float('nan')
            total_d = self._done_term + self._done_trunc
            timeout_pct = (100.0 * self._done_trunc / total_d) if total_d > 0 else 0.0
            elapsed = now - self._t0
            fps = (self._total_steps / max(1.0, elapsed))
            src = _VX_SOURCE_NAME if _VX_SOURCE_NAME is not None else "unknown"

            # RB suffix (dynamic header on first print)
            extra_csv = ""
            extra_print = ""
            if self._rb_enabled:
                comp_fields, kin_fields, comp_sum = self._rb.window_means()
                c_delta = comp_sum - avg_r
                suffix_cols = self._rb.csv_suffix_columns()
                vals = []
                for col in suffix_cols:
                    if col == "c_components_sum":
                        vals.append(comp_sum)
                    elif col == "c_components_delta":
                        vals.append(c_delta)
                    elif col.startswith("mean_"):
                        vals.append(kin_fields.get(col, 0.0))
                    else:
                        vals.append(comp_fields.get(col, 0.0))
                extra_csv = "," + ",".join(f"{v:.6f}" for v in vals)
                extra_print = f" | comps_sum={comp_sum:.2f} Δ={c_delta:.2f}"

                if not self._csv_header_frozen:
                    # Upgrade rollout & mirror headers dynamically
                    def _ensure_dynamic_header(path: Path, cols: list[str]):
                        try:
                            if path.exists():
                                with path.open("r") as f:
                                    first = f.readline()
                                new_header = _CSV_EXPECTED_HEADER[:-1] + ("," + ",".join(cols) if cols else "") + "\n"
                                if first != new_header:
                                    backup = path.with_suffix(path.suffix + ".old")
                                    shutil.move(str(path), str(backup))
                                    print(f"[csv] Rotated stale CSV header to: {backup.name}")
                            if not path.exists():
                                path.parent.mkdir(parents=True, exist_ok=True)
                                with path.open("w") as f:
                                    f.write(_CSV_EXPECTED_HEADER[:-1] + ("," + ",".join(cols) if cols else "") + "\n")
                        except Exception as e:
                            print(f"[csv] WARNING could not validate header: {e}")
                    _ensure_dynamic_header(self._rollout_log_path, suffix_cols)
                    _ensure_dynamic_header(_ROLLOUT_MIRROR_CSV, suffix_cols)
                    self._csv_header_frozen = True
                    self._rb.freeze_header()

                # ALSO print every N logs: full component & kinematic means
                self._rb_print_tick += 1
                if (self._rb_print_tick % self._rb_print_every) == 0:
                    # Sorted for stable readability
                    kin_str = ", ".join([f"{k}={kin_fields.get(k,0.0):.3f}" for k in sorted(kin_fields.keys())])
                    comp_str = ", ".join([f"{k}={comp_fields.get(k,0.0):.3f}" for k in sorted(comp_fields.keys())])
                    print(f"[RB/{self._phase}] KIN({kin_str})")
                    print(f"[RB/{self._phase}] COMP({comp_str})")

            print(
                f"[rollout/{self._phase}] {len(self._hist_ret)}ep avg_return={avg_r:.2f} avg_len={avg_l:.1f} | "
                f"episodes={self._episodes_done} steps={self._total_steps} elapsed={elapsed:.1f}s | "
                f"mean_vx_b={mean_vx:.3f} m/s (src={src}) | timeout={timeout_pct:.1f}% non_timeout={100.0-timeout_pct:.1f}%"
                f"{extra_print}"
            )

            line = (
                f"{elapsed:.3f},{self._episodes_done},{self._total_steps},"
                f"{avg_r:.6f},{avg_l:.6f},{fps:.2f},{mean_vx:.6f},{timeout_pct:.2f},{(100.0-timeout_pct):.2f}"
                f"{extra_csv}\n"
            )
            self._write_both_csvs(line)

            self._last_log_ts = now
            self._vx_sum.zero_(); self._vx_count.zero_()
            self._done_term = 0; self._done_trunc = 0

            # manual ckpt
            self._maybe_manual_checkpoint(mean_window_reward)

        # plateau stop (EPISODE decision, after logging and deque append)
        if self._check_plateau():
            print(f"[STOP/{self._phase}] Plateau detected → stopping phase.")
            # final line flush (optional snapshot)
            elapsed = time.time() - self._t0
            if len(self._hist_ret) > 0:
                vx_cnt = max(1, int(self._vx_count.item()))
                mean_vx = float((self._vx_sum / vx_cnt).item()) if vx_cnt > 0 else float('nan')
                total_d = self._done_term + self._done_trunc
                timeout_pct = (100.0 * self._done_trunc / total_d) if total_d > 0 else 0.0
                fps = (self._total_steps / max(1.0, elapsed))
                line = (
                    f"{elapsed:.3f},{self._episodes_done},{self._total_steps},"
                    f"{float(np.mean(self._hist_ret)):.6f},{float(np.mean(self._hist_len)):.6f},{fps:.2f},"
                    f"{mean_vx:.6f},{timeout_pct:.2f},{(100.0-timeout_pct):.2f}\n"
                )
                self._write_both_csvs(line)
            raise StopIteration("PHASE_PLATEAU")


# ──────────────────────────────────────────────────────────────────────────────
# 90. Load YAML → patch run-time knobs (per-phase)
# ──────────────────────────────────────────────────────────────────────────────
def _build_cfg_for_phase(
    phase_name: str,
    cycle_idx: int,
    attempt_suffix: str = "",
    phase_seed: int | None = None,
) -> tuple[dict, Path]:
    with open(args.cfg_yaml, "r") as f:
        params = yaml.safe_load(f)
    run_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    base_name = args.run_name or run_stamp
    if args.run_tag:
        base_name = f"{base_name}_{args.run_tag}"
    run_name = f"{base_name}_{phase_name}_c{cycle_idx+1}{attempt_suffix}"  # now defined

    seed_val = int(phase_seed) if phase_seed is not None else int(args.seed)

    cfg = params.copy()
    cfg.setdefault("seed", seed_val)

    cfg.setdefault("params", {})
    cfg["params"].setdefault("config", {})
    pc = cfg["params"]["config"]

    pc["env_name"] = "isaaclab"
    pc["vecenv_type"] = "LOCAL"
    # Pin rl_games run_dir under the SAME _CKPT_DIR across restarts
    pc["train_dir"] = str(Path(args.logdir).expanduser().resolve())
    pc["name"] = run_name

    run_dir = Path(pc["train_dir"]) / pc["name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    pc["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    pc["record_video"] = False
    pc["record_best_video"] = False
    pc["record_stats"] = True
    pc["save_best"] = False
    pc["save_best_after"] = 10**15
    pc["save_frequency"] = 10**15
    pc["num_actors"] = int(args.num_envs)
    pc["max_env_steps"] = 10**15
    pc["max_epochs"]    = 10**15

    # Batch scaling
    if args.batch_scale and args.batch_scale != 1.0:
        algo = cfg["params"].setdefault("algo", {})
        for k in ("batch_size", "minibatch_size", "mini_batch_size"):
            if k in algo:
                old = int(algo[k]); new = max(1, int(round(old * args.batch_scale)))
                algo[k] = new
                print(f"[CONFIG/{phase_name}] scaled {k}: {old} -> {new}")

    cfg["seed"] = seed_val

    # Minimal algo grid (parity with your scripts)
    cfg["params"].setdefault("algo", {})
    cfg["params"].setdefault("model", {})
    cfg["params"].setdefault("network", {})
    algo_cfg = cfg["params"]["algo"]
    algo_cfg.setdefault("config", {})
    cfg_params = algo_cfg["config"]

    # Write into algo.config…
    cfg_params["batch_size"]            = GRID_DEFAULTS["batch_size"]
    cfg_params["updates_per_step"]      = int(getattr(args, "updates_per_step", GRID_DEFAULTS["updates_per_step"]))
    cfg_params["train_every_n_steps"]   = GRID_DEFAULTS["train_every_n_steps"]
    cfg_params["actor_update_interval"] = GRID_DEFAULTS["actor_update_interval"]
    cfg_params["replay_buffer_size"]    = GRID_DEFAULTS["replay_buffer_size"] if args.replay_buffer_size is None else int(args.replay_buffer_size)
    cfg_params["target_entropy"]        = GRID_DEFAULTS["target_entropy"]
    cfg_params["alpha_lr"]              = GRID_DEFAULTS["alpha_lr"]
    cfg_params["critic_tau"]            = GRID_DEFAULTS["critic_tau"]
    cfg_params["gamma"]                 = GRID_DEFAULTS["gamma"]
    cfg_params["num_warmup_steps"]      = GRID_DEFAULTS["warmup_steps"]
    cfg_params["warmup_steps"]          = GRID_DEFAULTS["warmup_steps"]

    # …and mirror critical keys flat into algo to defeat later merges.
    algo_cfg["updates_per_step"] = int(getattr(args, "updates_per_step", GRID_DEFAULTS["updates_per_step"]))

    if args.force_no_warmup:
        cfg_params["num_warmup_steps"] = 0
        cfg_params["warmup_steps"] = 0
        algo_cfg["num_warmup_steps"] = 0
        algo_cfg["warmup_steps"] = 0
        print(f"[CONFIG/{phase_name}] force_no_warmup -> 0")

    if getattr(args, "override_warmup_steps", None) is not None:
        wu = int(args.override_warmup_steps)
        cfg_params["num_warmup_steps"] = wu
        cfg_params["warmup_steps"] = wu
        algo_cfg["num_warmup_steps"] = wu
        algo_cfg["warmup_steps"] = wu
        print(f"[CONFIG/{phase_name}] override num_warmup_steps -> {wu}")

    # Prepare run dir
    cfg["rollout_log"] = str(_ROLLOUT_MIRROR_CSV)

    run_dir.mkdir(parents=True, exist_ok=True)
    return cfg, run_dir

# ──────────────────────────────────────────────────────────────────────────────
# 95. Minimal video recorder launcher (ASYNC; runs in background on --video_gpu)
#     & Run rollout_plotter.py after each phase, overwriting graphs folder
# ──────────────────────────────────────────────────────────────────────────────
def _write_wait_then_record_script(script_path: Path, vid_gpu: int, threshold_pct: float, wait_s: int, cmd_list: list[str]):
    """
    Writes a standalone Python script that:
      • polls GPU `vid_gpu` utilization,
      • requires it to stay < threshold_pct for wait_s consecutive seconds,
      • then executes `cmd_list` (the isaaclab player command).
    Uses pynvml if available; falls back to nvidia-smi; if both missing, proceeds immediately.
    """
    script = f"""#!/usr/bin/env python3
import os, sys, time, subprocess

VID = int({vid_gpu})
THRESH = float({threshold_pct})
REQUIRED = int({wait_s})
CMD = {cmd_list!r}

def _nvml_util():
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(VID)
        util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
        return float(util)
    except Exception:
        return None

def _smi_util():
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
            "-i", str(VID)
        ], stderr=subprocess.STDOUT, text=True, timeout=3.0)
        return float(out.strip().split("\\n")[0])
    except Exception:
        return None

def gpu_util():
    u = _nvml_util()
    if u is not None: return u
    return _smi_util()

def wait_quiet():
    ok_for = 0
    last_print = 0
    while ok_for < REQUIRED:
        u = gpu_util()
        if u is None:
            print("[video-wait] No NVML/SMI → skipping wait.", flush=True)
            return
        if u < THRESH:
            ok_for += 1
        else:
            ok_for = 0
        now = time.time()
        if now - last_print >= 5:
            print(f"[video-wait] util={{u:.1f}}%  quiet={{ok_for}}/{{REQUIRED}} sec (threshold={{THRESH}}%)", flush=True)
            last_print = now
        time.sleep(1.0)
    print("[video-wait] quiet window satisfied → starting recorder.", flush=True)

def main():
    wait_quiet()
    # Run the actual recorder command
    try:
        p = subprocess.Popen(CMD)
        p.wait()
        sys.exit(p.returncode)
    except Exception as e:
        print(f"[video-wait] recorder failed: {{e}}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
"""
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script)
    try:
        os.chmod(script_path, 0o755)
    except Exception:
        pass

def _spawn_background(cmd: list[str], env: dict, log_path: Path) -> int:
    """
    Launch a child process detached from this trainer:
      • does NOT block the training loop
      • inherits only the provided env (incl. CUDA_VISIBLE_DEVICES for video GPU)
      • stdout/stderr are appended to log_path (so we don't hold the TTY)
    Returns the PID of the spawned process.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # open in append-binary; close on return to avoid FD leaks
    with open(log_path, "ab", buffering=0) as lf, open(os.devnull, "rb") as devnull:
        p = subprocess.Popen(
            cmd,
            stdin=devnull,
            stdout=lf,
            stderr=lf,
            env=env,
            start_new_session=True,  # fully detach (no signals delivered to group)
            close_fds=True,
        )
    # write a tiny .pid file next to the log for bookkeeping
    try:
        log_path.with_suffix(".pid").write_text(str(p.pid))
    except Exception:
        pass
    return p.pid

def _maybe_record_video(phase_name: str, cycle_idx: int, for_play_path: Path):
    """
    Spawn the player **asynchronously** so training keeps running.
    Video recording is pinned to --video_gpu (or --gpu if unset).
    Now includes a background wrapper that waits for GPU {util<thresh} for N seconds.
    """
    N = int(getattr(args, "record_every", 0) or 0)
    if N <= 0 or ((cycle_idx + 1) % N) != 0:
        return

    vid_gpu = args.video_gpu if args.video_gpu is not None else args.gpu
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(vid_gpu)

    # Exact recorder command (unchanged), but will be executed by a small wait-wrapper.
    recorder_cmd = [
        str(Path.home() / "projects" / "IsaacLab" / "isaaclab.sh"), "-p",
        "/home/adi/projects/CreativeMachinesAnt/Isaac/scripts/play_ant_force_load_splithead_v2.py",
        "--task", str(args.gym_env_id),
        "--cfg_yaml", str(getattr(args, "player_yaml", "/home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_play_sac_ant.yaml")),
        "--checkpoint", str(for_play_path),
        "--steps", "1000",
        "--video_dir", str(_VIDEOS_DIR),
        "--headless", "--enable_cameras", "--rendering_mode", "quality",
    ]

    # Write per-invocation wait-wrapper script into videos dir
    wait_pct = float(getattr(args, "video_wait_pct", 10.0))
    wait_s   = int(getattr(args, "video_wait_s", 30))
    script_path = _VIDEOS_DIR / f"wait_then_record_c{cycle_idx+1:03d}_{phase_name}.py"
    try:
        _write_wait_then_record_script(script_path, int(vid_gpu), wait_pct, wait_s, recorder_cmd)
    except Exception as e:
        print(f"[video] Could not create wait-wrapper script: {e}")
        # Fallback to immediate recorder if wrapper creation fails
        script_path = None

    log_path = _VIDEOS_DIR / f"record_c{cycle_idx+1:03d}_{phase_name}.log"
    try:
        if script_path is not None and script_path.exists():
            # Spawn the wrapper (which waits, then runs the recorder)
            pid = _spawn_background([sys.executable, str(script_path)], env, log_path)
            print(f"[video] (async+wait) cycle={cycle_idx+1} phase={phase_name} gpu={vid_gpu} "
                  f"wait<{wait_pct}% for {wait_s}s → PID={pid} | logs={log_path.name}")
        else:
            # Fallback: spawn recorder immediately
            pid = _spawn_background(recorder_cmd, env, log_path)
            print(f"[video] (async) cycle={cycle_idx+1} phase={phase_name} gpu={vid_gpu} "
                  f"→ PID={pid} | logs={log_path.name}")
    except Exception as e:
        print(f"[video] Launch failed (async): {e}")


def _generate_phase_graphs():
    try:
        # clean & recreate (so there's only one most-recent set)
        if _GRAPHS_DIR.exists():
            shutil.rmtree(_GRAPHS_DIR, ignore_errors=True)
        _GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "/home/adi/projects/CreativeMachinesAnt/Analysis_forIsaac/rollout_plotter.py",
            "--csv", str(_ROLLOUT_MIRROR_CSV),
            "--outdir", str(_GRAPHS_DIR),
        ]
        print(f"[graphs] Generating plots → outdir={_GRAPHS_DIR}")
        subprocess.run(cmd, check=False)
    except Exception as e:
        print(f"[graphs] Plot generation failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# 100. Prepare per-phase run dirs & CSVs (done in orchestrator)
# ───────────────────s───────────────────────────────────────────────────────────
# (handled in _run_phase)
# ──────────────────────────────────────────────────────────────────────────────
# 110. Phase orchestrator (ONE attempt per process; HARD restart happens in __main__)
# ──────────────────────────────────────────────────────────────────────────────
def _run_phase(
    phase_name: str,
    cycle_idx: int,
    incoming_model_sd: dict | None,
    behavior_idx: int,
    _retry: int = 0,
) -> dict:
    """
    Run a single phase attempt (no in-process retries).
    This function:
      • boots a fresh Isaac app for THIS phase
      • builds a fresh rl_games Runner/Algo (fresh replay buffer)
      • loads ONLY the incoming model weights (if provided)
      • trains until a stop reason (plateau/step-cap/restart-floor/etc.)
      • saves checkpoint + for_play + plots
      • DOES NOT call app.close(); we will hard-restart the entire process after returning
    """
    import gc
    global _GLOBAL_ALGO, _GLOBAL_VECENV, _ROLLOUT_LOG_PATH, _GLOBAL_ENV, _PHASE_RUNTIME, _PENDING_LOAD_SD, _LAST_FOR_PLAY, _LAST_STOP_REASON, _LAST_PLATEAU_MU


    print(f"[phase] ==== SETUP {phase_name.upper()} (cycle {cycle_idx+1}) retry={_retry} ====")

    # Per-phase, per-retry seed so retries explore different randomness.
    base_seed = int(args.seed)
    phase_index = {"walk": 0, "jump": 1, "spin": 2}.get(phase_name, 0)
    phase_seed = base_seed + 1000 * cycle_idx + 100 * phase_index + _retry
    print(f"[phase] Using seed {phase_seed} for phase={phase_name} cycle={cycle_idx+1} retry={_retry}")
    _set_global_seed(phase_seed)

    # Launch Isaac for THIS phase + get env factory (5b)
    make_env_func, phase_app = _phase_setup_isaac()

    # Build rl_games cfg for this phase — add retry suffix so logs are distinct
    attempt_suffix = f"_retry{_retry}" if _retry > 0 else ""
    try:
        cfg, run_dir = _build_cfg_for_phase(
            phase_name,
            cycle_idx,
            attempt_suffix=attempt_suffix,
            phase_seed=phase_seed,
        )
    except TypeError:
        cfg, run_dir = _build_cfg_for_phase(phase_name, cycle_idx, attempt_suffix=attempt_suffix)

    # Rollout CSV in this phase's run_dir
    _ROLLOUT_LOG_PATH = run_dir / "rollout_log.csv"
    _ensure_csv_header(_ROLLOUT_LOG_PATH)

    # Configure phase runtime for vecenv factory BEFORE Runner.load()
    _PHASE_RUNTIME["cycle_idx"] = int(cycle_idx)
    _PHASE_RUNTIME["name"] = phase_name
    _PHASE_RUNTIME["rollout_log_path"] = _ROLLOUT_LOG_PATH
    _PHASE_RUNTIME["manual_ckpt_every_eps"] = args.manual_ckpt_every_eps
    _PHASE_RUNTIME["nn_dir_path"] = str(_MODELS_DIR)
    _PHASE_RUNTIME["plateau"] = {
        "window_steps": int(args.plateau_window_steps),
        "abs_std": float(args.plateau_abs_std),
        "rel_std": float(args.plateau_rel_std),
        "min_steps": int(args.plateau_min_steps),
        "max_steps": int(args.max_steps_phase) if args.max_steps_phase is not None else None,
        "ep_window": int(args.plateau_episode_window),
        "rel_change": float(args.plateau_rel_change),
        "std_coeff": float(args.plateau_std_coeff),
        "min_return": float(args.plateau_min_return),
    }
    _PHASE_RUNTIME["log_reward_breakdown"] = bool(args.log_reward_breakdown)
    _PHASE_RUNTIME["make_env_func"] = make_env_func
    _PHASE_RUNTIME["app"] = phase_app  # informational only (DO NOT close in-process)

    print(f"[rollout] run_dir={run_dir}")
    print(f"[rollout] nn_dir={_MODELS_DIR}   # (models/checkpoints destination)")
    print(f"[rollout] videos_dir={_VIDEOS_DIR}")
    print(f"[rollout] graphs_dir={_GRAPHS_DIR}")
    print(f"[rollout] log={_ROLLOUT_LOG_PATH}")
    print(f"[rollout] mirror_csv={_ROLLOUT_MIRROR_CSV}")

    print(f"[CONFIG/{phase_name}] rl_games params (minimal):")
    pc = cfg["params"]["config"]
    print(json.dumps({
        "seed": cfg["seed"],
        "num_actors": pc["num_actors"],
        "train_dir": pc["train_dir"],
        "name": pc["name"],
        "vecenv_type": pc["vecenv_type"],
        "device": pc["device"],
        "return_torch": RETURN_TORCH,
        "rollout_log": str(_ROLLOUT_LOG_PATH),
        "save_frequency": pc.get("save_frequency"),
        "save_best_after": pc.get("save_best_after"),
        "save_best": pc.get("save_best", False),
        "manual_ckpt_every_eps": args.manual_ckpt_every_eps,
        "ckpt_dir": str(_MODELS_DIR),
        "gym_env_id": args.gym_env_id,
        "phase": phase_name,
        "batch_scale": args.batch_scale,
        "obs_norm": not args.disable_obs_norm,
        "rew_norm": not args.disable_rew_norm,
        "nvtx": getattr(args, "nvtx", False),
    }, indent=2))

    # Reset last-attempt globals
    _LAST_FOR_PLAY = None
    _LAST_STOP_REASON = None
    _LAST_PLATEAU_MU = None

    if args.dry_run:
        print(f"[dry_run/{phase_name}] Exiting before training.")
        return incoming_model_sd or {}

    # Queue incoming weights to be loaded at algo construction time.
    # The DefaultAlgoObserver.after_init hook will consume this.
    _PENDING_LOAD_SD = incoming_model_sd if incoming_model_sd else None
    if _PENDING_LOAD_SD is not None:
        print("[weights] Queued incoming weights for loading at algo init.")

    # Fresh runner → fresh replay buffer (owned by algo/agent)
    runner = Runner()
    runner.load(cfg)

    # Warmup override on agent (best-effort tweak; env-side warmup is in cfg)
    try:
        if getattr(args, "override_warmup_steps", None) is not None and hasattr(runner, "algo") and runner.algo is not None:
            wu = int(args.override_warmup_steps)
            if hasattr(runner.algo, "config") and isinstance(runner.algo.config, dict):
                for k in ("num_warmup_steps", "warmup_steps"):
                    runner.algo.config[k] = wu
            for k in ("num_warmup_steps", "warmup_steps"):
                if hasattr(runner.algo, k):
                    setattr(runner.algo, k, wu)
            print(f"[CONFIG/{phase_name}] SACAgent warmup forced -> {wu}")
    except Exception as e:
        print(f"[CONFIG/{phase_name}] (info) warmup tweak will proceed with cfg only: {e}")

    # Train until stop (VecEnv raises StopIteration)
    model_sd_out: dict = {}
    _wall_t0 = time.time()
    reason = ""

    try:
        runner.run({'train': True})
    except StopIteration as e:
        reason = str(e)
        _LAST_STOP_REASON = reason
        if reason in ("PHASE_PLATEAU", "STOP_AFTER_EPISODES", "STOP_AFTER_STEPS", "INTERRUPTED", "PHASE_RESTART_FLOOR"):
            print(f"[STOP/{phase_name}] Training loop exited: {reason}.")
        else:
            raise

    finally:
        # ---- Plateau μ_recent (if available) ----
        plateau_mu = None
        try:
            if _GLOBAL_VECENV is not None and hasattr(_GLOBAL_VECENV, "get_last_plateau_mu"):
                plateau_mu = _GLOBAL_VECENV.get_last_plateau_mu()
        except Exception:
            plateau_mu = None

        final_mean = None  # NEW: end-of-phase mean return (fallback if plateau_mu is None)
        try:
            if _GLOBAL_VECENV is not None and hasattr(_GLOBAL_VECENV, "get_last_window_mean_reward"):
                final_mean = _GLOBAL_VECENV.get_last_window_mean_reward()
        except Exception:
            final_mean = None


        if reason == "PHASE_RESTART_FLOOR":
            plateau_mu = float("-inf")

        _LAST_PLATEAU_MU = plateau_mu
        global _LAST_FINAL_WINDOW_MEAN
        _LAST_FINAL_WINDOW_MEAN = final_mean  # NEW

        thr = float(getattr(args, "min_phase_mean_reward_on_switch", 1000.0))

        # NEW: Determine pass/fail using plateau_mu if it exists, otherwise fall back to final_mean.
        # This is the “end-of-phase” failure criterion you want.
        score_for_passfail = plateau_mu if plateau_mu is not None else final_mean

        phase_passed = True
        if score_for_passfail is not None:
            phase_passed = (score_for_passfail >= thr)


        # Attempt-scoped artifact dirs (failed → checkpoints/.../failed/...)
        if not phase_passed:
            failed_root = _CKPT_DIR / "failed" / f"{phase_name}_c{cycle_idx+1}_retry{_retry}"
            models_dir = failed_root / "models"
            videos_dir = failed_root / "videos"
            graphs_dir = failed_root / "graphs"
        else:
            models_dir = _MODELS_DIR
            videos_dir = _VIDEOS_DIR
            graphs_dir = _GRAPHS_DIR

        for _d in (models_dir, videos_dir, graphs_dir):
            try:
                _d.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

        # ---- Save checkpoint + for_play from GLOBAL_ALGO ----
        try:
            algo_ref = _GLOBAL_ALGO  # set by observer hook
            if algo_ref is not None:
                stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                fname = f"c{cycle_idx+1:03d}_b{behavior_idx+1:02d}_{phase_name}_plateau_{stamp}"

                out_path = models_dir / fname
                ok, detail = _native_save(algo_ref, str(out_path))
                if ok:
                    print(f"[CHECKPOINT/{phase_name}] Saved native: {out_path} ({detail})")
                    _safe_write_line(_ROLLOUT_MIRROR_CSV, f"{time.time():.3f},CKPT,{out_path}\n")
                else:
                    print(f"[CHECKPOINT/{phase_name}] Native save failed: {detail}")

                for_play = models_dir / f"{fname}_for_play.pth"
                ok2, detail2 = _export_for_play_model(algo_ref, str(for_play))
                if ok2:
                    print(f"[CHECKPOINT/{phase_name}] Wrote for-play sidecar: {for_play} ({detail2})")
                    _safe_write_line(_ROLLOUT_MIRROR_CSV, f"{time.time():.3f},CKPT,{for_play}\n")
                    _LAST_FOR_PLAY = str(for_play)
                    if phase_passed:
                        _maybe_record_video(phase_name, cycle_idx, for_play)
                else:
                    print(f"[CHECKPOINT/{phase_name}] For-play export FAILED: {detail2}")

                # Optional: return a CPU clone of weights (not used by hard-restart flow, but kept for compatibility)
                try:
                    with torch.no_grad():
                        model = getattr(algo_ref, "model", None)
                        if model is not None:
                            sd = model.state_dict()
                            model_sd_out = {k: v.detach().to("cpu").clone() for k, v in sd.items()}
                except Exception:
                    model_sd_out = {}
            else:
                print(f"[CHECKPOINT/{phase_name}] No GLOBAL_ALGO captured; skipping save.")
        except Exception as e:
            print(f"[CHECKPOINT/{phase_name}] WARNING: post-phase save/export failed: {e}")

        # ---- Generate plots into attempt graphs_dir ----
        try:
            if graphs_dir.exists():
                shutil.rmtree(graphs_dir, ignore_errors=True)
            graphs_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                "/home/adi/projects/CreativeMachinesAnt/Analysis_forIsaac/rollout_plotter.py",
                "--csv", str(_ROLLOUT_MIRROR_CSV),
                "--outdir", str(graphs_dir),
            ]
            print(f"[graphs] Generating plots → outdir={graphs_dir}")
            subprocess.run(cmd, check=False)
        except Exception as e:
            print(f"[graphs] Plot generation failed: {e}")

        # ---- Minimal cleanup (NO app.close(); process will hard-restart) ----
        try:
            if _GLOBAL_ENV is not None:
                _GLOBAL_ENV.close()
        except Exception:
            pass

        try:
            if _GLOBAL_VECENV is not None and hasattr(_GLOBAL_VECENV, "set_algo"):
                _GLOBAL_VECENV.set_algo(None)
        except Exception:
            pass

        _GLOBAL_ENV = None
        _GLOBAL_VECENV = None
        _PHASE_RUNTIME["make_env_func"] = None

        # IMPORTANT: DO NOT close phase_app here. We will execv (hard restart) in __main__.
        # Calling app.close() can terminate the process before we can restart cleanly.

        _GLOBAL_ALGO = None
        _PENDING_LOAD_SD = None
        runner = None

        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        print(f"[wall/{phase_name}] total wall time: {time.time() - _wall_t0:.1f}s")

    return model_sd_out


# ──────────────────────────────────────────────────────────────────────────────
# 120. Master loop (STATE MACHINE): run ONE phase attempt, save state, HARD restart
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    phases = [p.strip().lower() for p in str(args.phase_order).split(",") if p.strip()]
    assert all(p in ("walk", "jump", "spin") for p in phases), f"Invalid phase in {phases}"

    # Load or initialize persistent phase state
    st = _load_state()
    if not st:
        st = {
            "cycle": 0,
            "phase_idx": 0,
            "retry": 0,
            "incoming_for_play": getattr(args, "resume_from", None),
            "done": False,
        }
        _save_state_atomic(st)

    if st.get("done", False):
        print("[DONE] phase_state.json says done.")
        sys.exit(0)

    MAX_RESTARTS_PER_PHASE = 3  # after retry=0 and retry=1 fail, advance on next restart

    # Deterministic advance: if we've already restarted enough times on this phase,
    # move to the next phase BEFORE we pick phase_idx/retry for this process run.
    retry = int(st.get("retry", 0))
    if retry >= MAX_RESTARTS_PER_PHASE:
        st["retry"] = 0
        st["phase_idx"] = int(st.get("phase_idx", 0)) + 1

        if st["phase_idx"] >= len(phases):
            st["phase_idx"] = 0
            st["cycle"] = int(st.get("cycle", 0)) + 1

        _save_state_atomic(st)

    cycle = int(st["cycle"])
    phase_idx = int(st["phase_idx"])
    retry = int(st["retry"])
    incoming_for_play = st.get("incoming_for_play", None)

    # Completion check
    if cycle >= int(args.n_cycles):
        st["done"] = True
        _save_state_atomic(st)
        print("[DONE] All cycles complete.")
        sys.exit(0)

    ph = phases[phase_idx]
    print(f"\n[STATE] cycle={cycle+1}/{args.n_cycles} phase={ph} idx={phase_idx} retry={retry}")

    # Load ONLY the incoming model (the only thing allowed to carry across phases)
    incoming_sd = None
    if incoming_for_play:
        print(f"[STATE] Loading incoming model: {incoming_for_play}")
        incoming_sd = _load_for_play_state_dict(incoming_for_play)
        if not incoming_sd:
            print("[STATE] WARNING: incoming_for_play provided but failed to load; starting fresh for this attempt.")
            incoming_sd = None

    # Run exactly one attempt (phase + retry index)
    _run_phase(ph, cycle, incoming_sd, behavior_idx=phase_idx, _retry=retry)

    # If user interrupted, do NOT auto-restart into the next attempt.
    if _LAST_STOP_REASON == "INTERRUPTED":
        print("[STATE] Interrupted. State saved; exiting without restart.")
        _save_state_atomic(st)
        sys.exit(0)

    # Decide retry vs advance using globals written by _run_phase
    thr = float(getattr(args, "min_phase_mean_reward_on_switch", 1000.0))
    retry_max = int(getattr(args, "phase_retry_max", 2))

    reason = _LAST_STOP_REASON
    mu = _LAST_PLATEAU_MU
    final_mean = _LAST_FINAL_WINDOW_MEAN  # NEW
    out_for_play = _LAST_FOR_PLAY

    score_for_retry = mu if mu is not None else final_mean
    need_retry = (reason == "PHASE_RESTART_FLOOR") or (score_for_retry is not None and score_for_retry < thr)


    if need_retry and retry < retry_max:
        # Retry: keep the SAME incoming model; only increment retry counter.
        st["retry"] = retry + 1
        print(f"[STATE] retrying {ph}: reason={reason} mu={mu} retry={st['retry']}/{retry_max}")
    else:
        # Accept this attempt’s model (if produced), then advance phase/cycle.
        if out_for_play:
            st["incoming_for_play"] = out_for_play
            print(f"[STATE] accepted model: {out_for_play}")
        else:
            print("[STATE] WARNING: no for_play produced; keeping previous incoming model")

        st["retry"] = 0
        st["phase_idx"] = phase_idx + 1
        if st["phase_idx"] >= len(phases):
            st["phase_idx"] = 0
            st["cycle"] = cycle + 1

    _save_state_atomic(st)

    # HARD restart the entire process (reliably restarts Isaac)
    _hard_restart_self()
