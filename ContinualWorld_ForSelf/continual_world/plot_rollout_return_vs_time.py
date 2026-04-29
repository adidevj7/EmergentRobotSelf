#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot avg_return_window vs time_s from a rollout CSV and save the PNG next to the CSV."
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to rollout.csv",
    )
    parser.add_argument(
        "--x_col",
        type=str,
        default="time_s",
        help="Column to use for x-axis (default: time_s)",
    )
    parser.add_argument(
        "--y_col",
        type=str,
        default="avg_return_window",
        help="Column to use for y-axis (default: avg_return_window)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if args.x_col not in df.columns:
        raise ValueError(f"Missing x column '{args.x_col}'. Available columns: {list(df.columns)}")
    if args.y_col not in df.columns:
        raise ValueError(f"Missing y column '{args.y_col}'. Available columns: {list(df.columns)}")

    x = pd.to_numeric(df[args.x_col], errors="coerce")
    y = pd.to_numeric(df[args.y_col], errors="coerce")
    valid = x.notna() & y.notna()

    if valid.sum() == 0:
        raise ValueError(
            f"No valid numeric rows found for x='{args.x_col}' and y='{args.y_col}'."
        )

    x = x[valid]
    y = y[valid]

    png_path = csv_path.parent / f"{csv_path.stem}_{args.y_col}_vs_{args.x_col}.png"

    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.xlabel(args.x_col)
    plt.ylabel(args.y_col)
    plt.title(args.title if args.title is not None else f"{args.y_col} vs {args.x_col}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    print(f"Saved plot to: {png_path}")


if __name__ == "__main__":
    main()