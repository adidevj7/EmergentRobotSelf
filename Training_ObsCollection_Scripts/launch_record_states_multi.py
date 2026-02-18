#!/usr/bin/env python3
"""
launch_record_states_multi.py

Multi-GPU launcher for RecordStates_forAnalysis3.py over many checkpoints.

Robustness fix:
- --models_glob can be:
  (1) a quoted glob pattern string: "/path/to/models/*.pth"
  (2) a directory: "/path/to/models"
  (3) an expanded list of .pth files (if you forgot quotes and the shell expands)

Safe parallelism:
- Uses atomic lock files in states_dir to avoid double-processing across multiple launchers.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
import glob as _glob


def _atomic_create_lock(lock_path: Path, payload: str) -> bool:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    except Exception:
        return False

    try:
        os.write(fd, payload.encode("utf-8"))
    finally:
        os.close(fd)
    return True


def _remove_silent(path: Path):
    try:
        path.unlink()
    except Exception:
        pass


def _resolve_ckpts(models_glob_args) -> list[Path]:
    """
    models_glob_args is a list of strings (because argparse uses nargs='+').

    Behaviors:
    - If it's a list of existing .pth files (shell-expanded), return them sorted.
    - If it's a single arg that is:
        * a directory -> glob dir/*.pth
        * a glob pattern -> glob it
        * a file -> that single file
    """
    # Shell-expanded case: many .pth paths
    if len(models_glob_args) > 1:
        pths = [Path(x).expanduser().resolve() for x in models_glob_args]
        if all(p.exists() and p.is_file() and p.suffix == ".pth" for p in pths):
            return sorted(pths)
        # otherwise fall through: maybe user mixed args

    # Single token case
    token = models_glob_args[0]
    p = Path(token).expanduser()

    # Directory
    if p.exists() and p.is_dir():
        return sorted([Path(x).resolve() for x in _glob.glob(str(p / "*.pth"))])

    # Single file
    if p.exists() and p.is_file() and p.suffix == ".pth":
        return [p.resolve()]

    # Glob pattern
    return sorted([Path(x).resolve() for x in _glob.glob(token)])


def main():
    p = argparse.ArgumentParser("Multi-GPU launcher for RecordStates_forAnalysis3.py over many .pth checkpoints")
    p.add_argument("--gpu", type=int, required=True, help="GPU index to use (sets CUDA_VISIBLE_DEVICES for the child)")
    p.add_argument("--task", required=True, help="Isaac task, e.g., Isaac-Ant-Direct-v0")
    p.add_argument("--cfg_yaml", required=True, help="rl_games play YAML")
    p.add_argument("--n_states", type=int, required=True, help="Number of observations to collect per checkpoint")

    p.add_argument(
        "--record_script",
        default="/home/adi/projects/CreativeMachinesAnt/Isaac/scripts/RecordStates_forAnalysis3.py",
        help="Path to RecordStates_forAnalysis3.py",
    )
    p.add_argument(
        "--isaaclab_sh",
        default="./isaaclab.sh",
        help="Path to isaaclab.sh (usually ./isaaclab.sh if you run from IsaacLab repo root)",
    )

    # IMPORTANT: nargs='+' makes this robust if the shell expands *.pth into many tokens.
    p.add_argument(
        "--models_glob",
        nargs="+",
        default=["/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/Obs/models/*.pth"],
        help='Glob pattern OR directory OR an expanded list of .pth files. Recommended: quote globs like "/path/*.pth".',
    )
    p.add_argument(
        "--states_dir",
        default="/home/adi/projects/CreativeMachinesAnt/Isaac/analysis/Obs/States",
        help="Directory where *_states.npy are written",
    )

    # pass-through flags for RecordStates_forAnalysis3.py
    p.add_argument("--headless", action="store_true")
    p.add_argument("--disable_fabric", action="store_true")

    p.add_argument("--poll_s", type=float, default=1.0, help="Sleep time between scans when running in loop mode")
    p.add_argument("--once", action="store_true", help="Process at most one checkpoint then exit")
    p.add_argument(
        "--keep_lock_on_fail",
        action="store_true",
        help="If set, keep the .lock file when a job fails (default removes lock to allow retries)",
    )

    args = p.parse_args()

    record_script = Path(args.record_script).resolve()
    isaaclab_sh = Path(args.isaaclab_sh).resolve()
    states_dir = Path(args.states_dir).resolve()
    states_dir.mkdir(parents=True, exist_ok=True)

    if not record_script.exists():
        print(f"[fatal] record_script not found: {record_script}", file=sys.stderr)
        sys.exit(2)
    if not isaaclab_sh.exists():
        print(f"[fatal] isaaclab_sh not found: {isaaclab_sh}", file=sys.stderr)
        print(
            "        Tip: cd into your IsaacLab repo so ./isaaclab.sh exists, or pass --isaaclab_sh /full/path/isaaclab.sh",
            file=sys.stderr,
        )
        sys.exit(2)

    def states_path_for(ckpt: Path) -> Path:
        return states_dir / f"{ckpt.stem}_states.npy"

    def lock_path_for(ckpt: Path) -> Path:
        return states_dir / f"{ckpt.stem}.lock"

    def fail_log_for(ckpt: Path) -> Path:
        return states_dir / f"{ckpt.stem}.fail.log"

    print(f"[launcher] gpu={args.gpu}")
    print(f"[launcher] models_glob={' '.join(args.models_glob)}")
    print(f"[launcher] states_dir={states_dir}")
    print(f"[launcher] record_script={record_script}")

    while True:
        ckpts = _resolve_ckpts(args.models_glob)
        if not ckpts:
            print("[launcher] no checkpoints found; exiting.")
            return

        claimed = None
        for ckpt in ckpts:
            out_npy = states_path_for(ckpt)
            if out_npy.exists():
                continue

            lock_path = lock_path_for(ckpt)
            payload = f"pid={os.getpid()}\ntime={time.time()}\ngpu={args.gpu}\nckpt={ckpt}\n"
            if not _atomic_create_lock(lock_path, payload):
                continue  # someone else is doing it

            claimed = ckpt
            break

        if claimed is None:
            print("[launcher] no remaining work (all states exist or are locked).")
            return

        ckpt = claimed
        out_npy = states_path_for(ckpt)
        lock_path = lock_path_for(ckpt)
        fail_log = fail_log_for(ckpt)

        cmd = [
            str(isaaclab_sh),
            "-p",
            str(record_script),
            "--task",
            args.task,
            "--cfg_yaml",
            args.cfg_yaml,
            "--checkpoints",
            str(ckpt),
            "--n_states",
            str(args.n_states),
            "--out_dir",
            str(states_dir),
        ]
        if args.headless:
            cmd.append("--headless")
        if args.disable_fabric:
            cmd.append("--disable_fabric")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

        print(f"\n[run] ckpt={ckpt.name}")
        print(f"[run] out={out_npy.name}")
        print(f"[run] cmd={' '.join(cmd)}")

        t0 = time.time()
        rc = 999
        try:
            rc = subprocess.call(cmd, env=env)
        except KeyboardInterrupt:
            print("[launcher] interrupted; leaving lock in place.")
            raise
        except Exception as e:
            print(f"[run] exception: {e}", file=sys.stderr)
            rc = 111

        dt = time.time() - t0

        if rc == 0 and out_npy.exists():
            print(f"[done] ok rc=0  time={dt:.1f}s  wrote={out_npy}")
            _remove_silent(fail_log)
            _remove_silent(lock_path)
        else:
            msg = f"[fail] rc={rc}  time={dt:.1f}s  wrote={out_npy.exists()}  ckpt={ckpt}\n"
            print(msg, file=sys.stderr)
            try:
                fail_log.write_text(msg)
            except Exception:
                pass
            if not args.keep_lock_on_fail:
                _remove_silent(lock_path)

        if args.once:
            return

        time.sleep(max(0.0, float(args.poll_s)))


if __name__ == "__main__":
    main()
