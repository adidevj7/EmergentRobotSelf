#!/usr/bin/env python3
"""
Batch launcher: record 1 video per checkpoint in /home/adi/projects/CreativeMachinesAnt/Isaac/tapes/models
Outputs to:     /home/adi/projects/CreativeMachinesAnt/Isaac/tapes/videos

Behavior:
- Uses ONLY GPU 6 (CUDA_VISIBLE_DEVICES=6).
- Walks checkpoints in alphabetical order.
- For each checkpoint: if any matching video already exists, skip it.
- Otherwise: launch the IsaacLab player to record the video.
- Between launches: wait until physical GPU 6 has been FREE for 10 continuous seconds.

Run:
  chmod +x /home/adi/projects/CreativeMachinesAnt/Isaac/scripts/batch_record_tapes_videos.py
  /home/adi/projects/CreativeMachinesAnt/Isaac/scripts/batch_record_tapes_videos.py
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import List


# =========================
# CONFIG (edit if needed)
# =========================
GPU_INDEX_PHYSICAL = 6
CUDA_VISIBLE_DEVICES_VALUE = "6"

MODELS_DIR = Path("/home/adi/projects/CreativeMachinesAnt/Isaac/tapes/models")
VIDEOS_DIR = Path("/home/adi/projects/CreativeMachinesAnt/Isaac/tapes/videos")

ISAACLAB_ROOT = Path("/home/adi/projects/IsaacLab")
ISAACLAB_SH = ISAACLAB_ROOT / "isaaclab.sh"

PLAYER_SCRIPT = Path("/home/adi/projects/CreativeMachinesAnt/Isaac/scripts/play_ant_force_load_splithead_v2.py")

TASK = "Isaac-Ant-Direct-v0"
CFG_YAML = Path("/home/adi/projects/CreativeMachinesAnt/Isaac/cfg/rlg_play_sac_ant_150_relu.yaml")
STEPS = 1000

KIT_ARGS = [
    "--headless",
    "--enable_cameras",
    "--rendering_mode", "quality",
]

GPU_FREE_CONTINUOUS_SECONDS = 10
GPU_FREE_POLL_SECONDS = 1


# =========================
# GPU utilities
# =========================
def _run_nvidia_smi(args: List[str]) -> str:
    cmd = ["nvidia-smi"] + args
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"nvidia-smi failed: {e.output}") from e
    except FileNotFoundError as e:
        raise RuntimeError("nvidia-smi not found on PATH.") from e


def gpu_has_compute_procs(gpu_index: int) -> bool:
    """
    True if GPU has any compute apps listed by nvidia-smi.
    """
    out = _run_nvidia_smi([
        f"-i={gpu_index}",
        "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ])
    # If no processes, nvidia-smi often returns empty string.
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return len(lines) > 0


def wait_for_gpu_free_continuous(gpu_index: int, seconds_free: int, poll_s: int) -> None:
    """
    Wait until GPU shows no compute processes for `seconds_free` consecutive seconds.
    """
    free_streak = 0
    while free_streak < seconds_free:
        busy = gpu_has_compute_procs(gpu_index)
        if busy:
            free_streak = 0
        else:
            free_streak += poll_s
        time.sleep(poll_s)


# =========================
# Video existence logic
# =========================
def checkpoint_has_video(ckpt: Path, videos_dir: Path) -> bool:
    """
    Your player writes name_prefix = f"ant_{ckpt.stem}_{int(time.time())}"
    Gym RecordVideo typically outputs: "<name_prefix>-episode-0.mp4" (and possibly more).
    So we consider it "done" if ANY mp4 starting with "ant_{stem}_" exists.
    """
    stem = ckpt.stem
    pattern = f"ant_{stem}_*.mp4"
    return any(videos_dir.glob(pattern))


def list_checkpoints(models_dir: Path) -> List[Path]:
    """
    Recursively find .pth checkpoints and sort alphabetically by path string.
    """
    ckpts = sorted(models_dir.rglob("*.pth"), key=lambda p: str(p))
    return ckpts


# =========================
# Launch
# =========================
def launch_one(ckpt: Path) -> int:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES_VALUE

    cmd = [
        str(ISAACLAB_SH),
        "-p",
        str(PLAYER_SCRIPT),
        "--task", TASK,
        "--cfg_yaml", str(CFG_YAML),
        "--checkpoint", str(ckpt),
        "--steps", str(STEPS),
        "--video_dir", str(VIDEOS_DIR),
        *KIT_ARGS,
    ]

    print("\n" + "=" * 90)
    print(f"[launch] checkpoint: {ckpt}")
    print(f"[launch] cmd: CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES_VALUE} " + " ".join(cmd))
    print("=" * 90 + "\n", flush=True)

    # Run from IsaacLab root (matches how you run ./isaaclab.sh manually)
    p = subprocess.run(cmd, cwd=str(ISAACLAB_ROOT), env=env)
    return int(p.returncode)


def main() -> int:
    # makedir
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not ISAACLAB_SH.exists():
        print(f"[error] IsaacLab launcher not found: {ISAACLAB_SH}", file=sys.stderr)
        return 2
    if not PLAYER_SCRIPT.exists():
        print(f"[error] Player script not found: {PLAYER_SCRIPT}", file=sys.stderr)
        return 2
    if not CFG_YAML.exists():
        print(f"[error] CFG YAML not found: {CFG_YAML}", file=sys.stderr)
        return 2

    ckpts = list_checkpoints(MODELS_DIR)
    if not ckpts:
        print(f"[done] No .pth checkpoints found under: {MODELS_DIR}")
        return 0

    print(f"[info] models_dir: {MODELS_DIR}")
    print(f"[info] videos_dir: {VIDEOS_DIR}")
    print(f"[info] found {len(ckpts)} checkpoints (.pth)")
    print(f"[info] using physical GPU index {GPU_INDEX_PHYSICAL} free-check; launching with CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES_VALUE}")

    # Before starting, require GPU 6 free for 10 continuous seconds
    print(f"[wait] waiting for GPU {GPU_INDEX_PHYSICAL} to be free for {GPU_FREE_CONTINUOUS_SECONDS}s ...", flush=True)
    wait_for_gpu_free_continuous(GPU_INDEX_PHYSICAL, GPU_FREE_CONTINUOUS_SECONDS, GPU_FREE_POLL_SECONDS)

    num_skipped = 0
    num_recorded = 0

    for ckpt in ckpts:
        if checkpoint_has_video(ckpt, VIDEOS_DIR):
            print(f"[skip] video exists for: {ckpt.name}")
            num_skipped += 1
            continue

        # Only start when GPU is free for 10 continuous seconds
        print(f"[wait] GPU {GPU_INDEX_PHYSICAL} free for {GPU_FREE_CONTINUOUS_SECONDS}s before recording: {ckpt.name}", flush=True)
        wait_for_gpu_free_continuous(GPU_INDEX_PHYSICAL, GPU_FREE_CONTINUOUS_SECONDS, GPU_FREE_POLL_SECONDS)

        rc = launch_one(ckpt)
        if rc != 0:
            print(f"[error] recording failed (return code {rc}) for: {ckpt}", file=sys.stderr)
            return rc

        num_recorded += 1

        # After completion, wait until GPU is actually free for 10 continuous seconds
        print(f"[wait] waiting for GPU {GPU_INDEX_PHYSICAL} to be free for {GPU_FREE_CONTINUOUS_SECONDS}s after run ...", flush=True)
        wait_for_gpu_free_continuous(GPU_INDEX_PHYSICAL, GPU_FREE_CONTINUOUS_SECONDS, GPU_FREE_POLL_SECONDS)

    print("\n" + "-" * 90)
    print(f"[done] recorded: {num_recorded} | skipped (already had videos): {num_skipped} | total: {len(ckpts)}")
    print(f"[done] videos in: {VIDEOS_DIR}")
    print("-" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
