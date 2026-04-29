#!/usr/bin/env bash
# =============================================================================
# setup_vast.sh  —  One-shot setup for EmergentRobotSelf on a Vast.ai instance
#
# Tested against:
#   • Vast.ai PyTorch base image (Ubuntu 22.04/24.04, root access)
#   • Isaac Sim 5.0.0-rc.45 (pip install route)
#   • Python 3.10 (required by IsaacLab for Isaac Sim 5.x pip route)
#
# What this script does:
#   1. Installs system dependencies (conda, system libs)
#   2. Clones your repo + reconstructs IsaacLab from the bundled snapshot
#   3. Creates a conda env (ant_rl, Python 3.10) matching your private server
#   4. Installs Isaac Sim 5.0 via pip into that env
#   5. Installs IsaacLab extensions (editable)
#   6. Installs all training runtime dependencies
#   7. Creates path symlinks so your absolute paths work unchanged
#   8. Runs a full smoke test
#
# Usage (as root on Vast):
#   bash setup_vast.sh
#
# After setup, launch training with:
#   cd /workspace/IsaacLab
#   conda run -n ant_rl --no-capture-output \
#     ./isaaclab.sh -p /home/adi/projects/CreativeMachinesAnt/Isaac/scripts/Isaac_WSJ_att69_cleanup.py \
#     --task Ant-Walk-v0 ...etc...
# =============================================================================

set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
section() { echo -e "\n${GREEN}════════════════════════════════════════${NC}"; \
            echo -e "${GREEN}  $*${NC}"; \
            echo -e "${GREEN}════════════════════════════════════════${NC}\n"; }

# ── Constants — edit these if your repo/branch changes ───────────────────────
REPO_URL="https://github.com/adidevj7/EmergentRobotSelf.git"
BUNDLE_REL="vendor/isaaclab/IsaacLab_working-20250929-1237.bundle"
BUNDLE_BRANCH="working-20250929-1237"
CONDA_ENV="ant_rl"
PYTHON_VERSION="3.10"
ISAACSIM_VERSION="5.0.0"          # pip package version
ISAACSIM_PIP_TAG="5.0.0"          # as it appears on pypi.nvidia.com

# =============================================================================
# SECTION 0 — Must run as root
# =============================================================================
section "0 · Preflight"
if [ "$EUID" -ne 0 ]; then
    warn "Not running as root. Some apt-get steps may fail."
    warn "Re-run with: sudo bash setup_vast.sh"
fi
info "Running as: $(whoami)  host: $(hostname)"
info "Working dir: $(pwd)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null \
    && info "GPU detected" || warn "nvidia-smi failed — continuing anyway"

# =============================================================================
# SECTION 1 — System dependencies
# =============================================================================
section "1 · System dependencies"
apt-get update -qq
apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    curl \
    wget \
    cmake \
    build-essential \
    libxt6 \
    libglu1-mesa \
    libusb-1.0-0 \
    libudev1 \
    libhidapi-hidraw0 \
    libhidapi-libusb0 \
    libegl1 \
    libgl1 \
    libgles2 \
    libglvnd0 \
    2>/dev/null || warn "Some apt packages may have failed — continuing"

info "System deps installed."

# =============================================================================
# SECTION 2 — Install Miniconda (if not present)
# =============================================================================
section "2 · Miniconda"
if command -v conda &>/dev/null; then
    info "conda already available: $(conda --version)"
else
    info "Installing Miniconda3..."
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -o /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /opt/miniconda3
    rm /tmp/miniconda.sh
    export PATH="/opt/miniconda3/bin:$PATH"
    conda init bash
    info "Miniconda installed at /opt/miniconda3"
fi

# Make conda available in this shell session regardless of init state
CONDA_BASE="$(conda info --base 2>/dev/null || echo /opt/miniconda3)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
info "conda base: ${CONDA_BASE}"

# =============================================================================
# SECTION 3 — Clone repo + reconstruct IsaacLab from bundle
# =============================================================================
section "3 · Clone repo & reconstruct IsaacLab"
cd /workspace

# Clean slate
rm -rf EmergentRobotSelf IsaacLab
info "Cleaned old workspace"

# Clone repo
info "Cloning EmergentRobotSelf..."
git clone "${REPO_URL}" EmergentRobotSelf
cd EmergentRobotSelf
git lfs install
git lfs pull
info "LFS pull complete"

# Sanity: bundle must exist and be non-trivial
BUNDLE_PATH="/workspace/EmergentRobotSelf/${BUNDLE_REL}"
if [ ! -f "${BUNDLE_PATH}" ]; then
    echo -e "${RED}[ERROR]${NC} Bundle not found at: ${BUNDLE_PATH}"
    echo "       Check that git-lfs pulled correctly: ls -lh $(dirname ${BUNDLE_PATH})"
    exit 1
fi
BUNDLE_SIZE=$(du -m "${BUNDLE_PATH}" | cut -f1)
info "Bundle size: ${BUNDLE_SIZE} MB"
if [ "${BUNDLE_SIZE}" -lt 50 ]; then
    echo -e "${RED}[ERROR]${NC} Bundle looks too small (${BUNDLE_SIZE} MB) — git-lfs pull likely failed."
    echo "       Run: cd /workspace/EmergentRobotSelf && git lfs pull"
    exit 1
fi

# Reconstruct IsaacLab from bundle
info "Reconstructing IsaacLab from bundle..."
cd /workspace
git clone "${BUNDLE_PATH}" IsaacLab
cd IsaacLab
git checkout "${BUNDLE_BRANCH}"
info "IsaacLab HEAD: $(git rev-parse HEAD)"
info "IsaacLab tag:  $(git describe --tags --always 2>/dev/null || echo 'no tag')"

# =============================================================================
# SECTION 4 — Create conda environment (Python 3.10)
# =============================================================================
section "4 · Conda environment (${CONDA_ENV}, Python ${PYTHON_VERSION})"

# Remove old env if it exists
if conda env list | grep -q "^${CONDA_ENV} "; then
    warn "Removing existing conda env: ${CONDA_ENV}"
    conda env remove -n "${CONDA_ENV}" -y
fi

info "Creating conda env: ${CONDA_ENV} with Python ${PYTHON_VERSION}"
conda create -n "${CONDA_ENV}" -y \
    python="${PYTHON_VERSION}" \
    pip \
    setuptools \
    wheel

# Grab the python/pip from this env directly (no activate needed)
PY="/opt/miniconda3/envs/${CONDA_ENV}/bin/python"
PIP="/opt/miniconda3/envs/${CONDA_ENV}/bin/pip"

# Fallback: conda might be at /root/miniconda3 or user home
if [ ! -f "${PY}" ]; then
    PY="${CONDA_BASE}/envs/${CONDA_ENV}/bin/python"
    PIP="${CONDA_BASE}/envs/${CONDA_ENV}/bin/pip"
fi

if [ ! -f "${PY}" ]; then
    echo -e "${RED}[ERROR]${NC} Cannot find python at ${PY}"
    echo "  Try: conda info --envs  to find the env path"
    exit 1
fi

info "Using python: ${PY}"
info "Python version: $(${PY} --version)"

# Conservative pip
${PIP} install --no-cache-dir --upgrade "pip<25"

# =============================================================================
# SECTION 5 — Install Isaac Sim via pip
# =============================================================================
section "5 · Isaac Sim ${ISAACSIM_VERSION} (pip, NVIDIA index)"

# IMPORTANT: Accept EULA
export OMNI_KIT_ACCEPT_EULA=YES

info "Installing isaacsim[all,extscache]==${ISAACSIM_PIP_TAG} ..."
info "(This will download ~10-15 GB — expect 10-30 min depending on bandwidth)"

${PIP} install --no-cache-dir \
    "isaacsim[all,extscache]==${ISAACSIM_PIP_TAG}" \
    --extra-index-url https://pypi.nvidia.com

info "Isaac Sim pip install complete."

# Pin core deps that Isaac Sim is sensitive to
# (matches what works on your private server's ant_rl env)
info "Re-pinning core deps..."
${PIP} install --no-deps --force-reinstall \
    "packaging==25.0" \
    "numpy==2.2.6" \
    "scipy==1.15.3" \
    "imageio==2.37.0" \
    "matplotlib==3.10.3" \
    "setuptools==78.1.1"

# =============================================================================
# SECTION 6 — Install IsaacLab extensions (editable, no extra deps)
# =============================================================================
section "6 · IsaacLab extensions (editable)"

cd /workspace/IsaacLab

info "Installing source/isaaclab (core)..."
${PIP} install --no-deps -e source/isaaclab

info "Installing source/isaaclab_assets..."
${PIP} install --no-deps -e source/isaaclab_assets 2>/dev/null || \
    warn "isaaclab_assets not present or failed — skipping"

info "Installing source/isaaclab_tasks..."
${PIP} install --no-deps -e source/isaaclab_tasks 2>/dev/null || \
    warn "isaaclab_tasks not present or failed — skipping"

info "Installing source/isaaclab_rl..."
${PIP} install --no-deps -e source/isaaclab_rl 2>/dev/null || \
    warn "isaaclab_rl not present or failed — skipping"

info "Installing source/isaaclab_mimic..."
${PIP} install --no-deps -e source/isaaclab_mimic 2>/dev/null || \
    warn "isaaclab_mimic not present or failed — skipping"

# =============================================================================
# SECTION 7 — Runtime training dependencies
# =============================================================================
section "7 · Training runtime dependencies"

# Install everything your training script and ant_rl env needs,
# with --upgrade-strategy only-if-needed to avoid breaking Isaac Sim pins.
${PIP} install --upgrade-strategy only-if-needed \
    "torch==2.7.1" \
    "torchvision==0.22.1" \
    --index-url https://download.pytorch.org/whl/cu126

${PIP} install --upgrade-strategy only-if-needed \
    "rl-games==1.6.1" \
    "gymnasium==1.1.1" \
    "pandas==2.3.1" \
    "seaborn==0.13.2" \
    "tensorboard==2.19.0" \
    "tensorboardX==2.6.4" \
    "tqdm==4.67.1" \
    "scikit-learn" \
    "scikit-image" \
    "opencv-python==4.12.0.88" \
    "imageio-ffmpeg" \
    "moviepy" \
    "pynvml" \
    "nvidia-ml-py" \
    "flatdict" \
    "prettytable" \
    "h5py" \
    "hid" \
    "wandb==0.12.21" \
    "psutil" \
    "PyYAML==6.0.2"

# Re-assert pins one final time (pip dependency resolution can drift these)
${PIP} install --no-deps --force-reinstall \
    "numpy==2.2.6" \
    "packaging==25.0" \
    "scipy==1.15.3"

info "Training deps installed."

# =============================================================================
# SECTION 8 — Path symlinks (make your absolute paths work unchanged)
# =============================================================================
section "8 · Path symlinks"

# /home/adi mirrors /root on Vast (Vast gives root access, home is /root)
mkdir -p /home/adi/projects/CreativeMachinesAnt
mkdir -p /root/projects/CreativeMachinesAnt

# Primary symlinks
ln -sfn /workspace/EmergentRobotSelf  /home/adi/projects/CreativeMachinesAnt/Isaac
ln -sfn /workspace/IsaacLab           /home/adi/projects/IsaacLab

# Mirror under /root (some paths in logs resolve here)
ln -sfn /workspace/EmergentRobotSelf  /root/projects/CreativeMachinesAnt/Isaac
ln -sfn /workspace/IsaacLab           /root/projects/IsaacLab

# Analysis scripts
if [ -d /workspace/EmergentRobotSelf/Analysis_forIsaac ]; then
    ln -sfn /workspace/EmergentRobotSelf/Analysis_forIsaac \
        /home/adi/projects/CreativeMachinesAnt/Analysis_forIsaac
    ln -sfn /workspace/EmergentRobotSelf/Analysis_forIsaac \
        /root/projects/CreativeMachinesAnt/Analysis_forIsaac
    info "Analysis_forIsaac symlink created"
else
    warn "Analysis_forIsaac not found in repo — skipping that symlink"
fi

# Checkpoints dir (pre-create so training doesn't need to)
mkdir -p /home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints
mkdir -p /root/projects/CreativeMachinesAnt/Isaac/checkpoints

info "Symlinks:"
ls -la /home/adi/projects/CreativeMachinesAnt/
ls -la /home/adi/projects/

# =============================================================================
# SECTION 9 — Smoke tests
# =============================================================================
section "9 · Smoke tests"

SMOKE_FAIL=0

run_test() {
    local label="$1"; local code="$2"
    if ${PY} -c "${code}" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC}  ${label}"
    else
        echo -e "  ${RED}✗${NC}  ${label}"
        SMOKE_FAIL=$((SMOKE_FAIL + 1))
    fi
}

# Core imports
run_test "isaacsim importable"   "import isaacsim; print(isaacsim.__file__)"
run_test "isaaclab importable"   "import isaaclab; print(isaaclab.__file__)"
run_test "isaaclab_tasks importable" "import isaaclab_tasks"

# Training deps
run_test "torch (CUDA)"          "import torch; assert torch.cuda.is_available(), 'no CUDA'"
run_test "numpy 2.2.x"           "import numpy as np; assert np.__version__.startswith('2.2')"
run_test "scipy"                 "import scipy; print(scipy.__version__)"
run_test "rl_games"              "import rl_games"
run_test "gymnasium"             "import gymnasium; print(gymnasium.__version__)"
run_test "pandas"                "import pandas; print(pandas.__version__)"
run_test "pynvml"                "import pynvml"
run_test "cv2"                   "import cv2; print(cv2.__version__)"
run_test "flatdict"              "import flatdict"
run_test "prettytable"           "import prettytable"
run_test "h5py"                  "import h5py"
run_test "hid"                   "import hid"
run_test "tensorboard"           "import tensorboard"
run_test "matplotlib"            "import matplotlib"
run_test "imageio"               "import imageio"
run_test "pkg_resources"         "import pkg_resources"
run_test "yaml"                  "import yaml"

# Path sanity
run_test "/home/adi/projects/IsaacLab symlink" \
    "import os; assert os.path.isdir('/home/adi/projects/IsaacLab')"
run_test "/home/adi/projects/CreativeMachinesAnt/Isaac symlink" \
    "import os; assert os.path.isdir('/home/adi/projects/CreativeMachinesAnt/Isaac')"
run_test "isaaclab.sh exists" \
    "import os; assert os.path.isfile('/home/adi/projects/IsaacLab/isaaclab.sh')"

echo ""
if [ "${SMOKE_FAIL}" -eq 0 ]; then
    echo -e "${GREEN}✅  All smoke tests passed.${NC}"
else
    echo -e "${YELLOW}⚠️   ${SMOKE_FAIL} smoke test(s) failed — see above.${NC}"
    echo "    This may be fine if hid/flatdict/etc are only used at Kit boot time."
    echo "    Proceed to the launch test below and check for runtime errors."
fi

# =============================================================================
# SECTION 10 — Print launch instructions
# =============================================================================
section "10 · How to launch training"

CONDA_RUN="conda run -n ${CONDA_ENV} --no-capture-output"
ISAACLAB="cd /workspace/IsaacLab && ${CONDA_RUN} ./isaaclab.sh"

cat << EOF

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Setup complete. Launch training like this:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  cd /workspace/IsaacLab

  conda run -n ${CONDA_ENV} --no-capture-output \\
    ./isaaclab.sh -p \\
    /home/adi/projects/CreativeMachinesAnt/Isaac/scripts/Isaac_WSJ_att69_cleanup.py \\
    --task Ant-Walk-v0 \\
    --gym_env_id Isaac-Ant-Direct-v0 \\
    --cfg_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_walk_new_150_relu.yaml \\
    --player_yaml /home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_play_sac_ant_150_relu.yaml \\
    --num_envs 8192 --n_cycles 50 \\
    --phase_order walk,spin,jump \\
    --headless --gpu 0 \\
    [... remaining args ...]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Key paths:
    IsaacLab:      /workspace/IsaacLab   → /home/adi/projects/IsaacLab
    Your scripts:  /workspace/EmergentRobotSelf → /home/adi/projects/CreativeMachinesAnt/Isaac
    Checkpoints:   /home/adi/projects/CreativeMachinesAnt/Isaac/checkpoints/
    Conda env:     ${CONDA_BASE}/envs/${CONDA_ENV}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  NOTE: Do NOT conda activate before launching. Use 'conda run' as shown above.
  This matches how your private server works (isaaclab.sh detects CONDA_PREFIX
  and uses that env's python automatically when the env is active, but 
  'conda run' is cleaner for scripting and avoids sourcing issues).

EOF

info "Done. Total time: ${SECONDS}s"