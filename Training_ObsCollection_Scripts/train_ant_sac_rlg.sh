#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train_ant_sac_rlg.sh [GPU_ID] [NUM_ENVS]
# Example:
#   bash scripts/train_ant_sac_rlg.sh 2 128

GPU_ID="${1:-0}"
NUM_ENVS="${2:-128}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
./isaac.sh -p "$HOME/projects/IsaacLab/scripts/reinforcement_learning/rl_games/train.py" \
  --task=Isaac-Ant-Direct-v0 \
  --num_envs="${NUM_ENVS}" \
  --headless \
  agent.params.algo.name=sac \
  agent.params.model.name=soft_actor_critic \
  ++agent.params.model.separate=false \
  ++agent.params.network.name=soft_actor_critic \
  ++agent.params.network.separate=true \
  ++agent.params.network.space.continuous.mlp.units=[256,256] \
  ++agent.params.network.log_std_bounds=[-7,2] \
  ++agent.params.config.gamma=0.99 \
  ++agent.params.config.critic_tau=0.005 \
  ++agent.params.config.batch_size=1024 \
  ++agent.params.config.replay_buffer_size=1000000 \
  ++agent.params.config.num_warmup_steps=10000 \
  ++agent.params.config.actor_lr=0.0003 \
  ++agent.params.config.critic_lr=0.0003 \
  ++agent.params.config.alpha_lr=0.003 \
  ++agent.params.config.learnable_temperature=true \
  ++agent.params.config.init_alpha=0.2 \
  ++agent.params.config.target_entropy=auto
