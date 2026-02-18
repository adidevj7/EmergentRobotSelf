#!/usr/bin/env bash
set -euo pipefail

# Deploy SAC config into IsaacLab's Ant agent config folder.
# Usage: bash scripts/deploy_cfg.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO_ROOT/cfg/ant_sac.yaml"
DST="$HOME/projects/IsaacLab/source/isaaclab_tasks/direct/ant/agents/rl_games_sac_cfg.yaml"

install -D "$SRC" "$DST"
echo "[OK] Deployed:"
echo "  $SRC"
echo "→ $DST"
