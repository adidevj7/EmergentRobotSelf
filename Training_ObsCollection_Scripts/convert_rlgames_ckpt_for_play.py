#!/usr/bin/env python3
"""
Convert an rl_games checkpoint to the format IsaacLab's play.py expects:
ensure top-level key 'model' exists (mapping from common variants).
"""
import argparse
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_ckpt", required=True)
    p.add_argument("--out_ckpt", required=True)
    args = p.parse_args()

    import torch
    ckpt = torch.load(args.in_ckpt, map_location="cpu")
    keys = sorted(ckpt.keys())
    print("[INFO] input keys:", keys)

    # preferred already-correct form
    if "model" in ckpt:
        out = ckpt
        print("[INFO] 'model' key already present; copying as-is.")
    # common rl_games save
    elif "state_dict" in ckpt:
        out = {"model": ckpt["state_dict"]}
        print("[INFO] mapping 'state_dict' -> 'model'")
    # SAC-style split (actor/critic/etc) — use actor for play.py
    elif "actor" in ckpt and isinstance(ckpt["actor"], dict):
        out = {"model": ckpt["actor"]}
        print("[INFO] mapping 'actor' -> 'model'")
    # nested agent.model
    elif "agent" in ckpt and isinstance(ckpt["agent"], dict) and "model" in ckpt["agent"]:
        out = {"model": ckpt["agent"]["model"]}
        print("[INFO] mapping 'agent.model' -> 'model'")
    else:
        raise KeyError(
            "Cannot find weights to map to 'model'. "
            f"Top-level keys: {keys}"
        )

    Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out_ckpt)
    print(f"[OK] wrote converted checkpoint -> {args.out_ckpt}")

if __name__ == "__main__":
    main()
