#!/usr/bin/env python3
# JUPYTER-RUNNABLE ROLLOUT PLOTS (no seaborn; separate figures)
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Union, Tuple, Callable
import re
import io
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Ensure headless-friendly backend when run as a script
plt.switch_backend("Agg")

# ──────────────────────────────────────────────────────────────────────────────
# Parse helpers
# ──────────────────────────────────────────────────────────────────────────────
EVENT_PAT = re.compile(r"(ckpt|phase switch)", re.IGNORECASE)

def _stitch_monotone(series: pd.Series) -> pd.Series:
    """
    Make a numeric series strictly non-decreasing by adding an offset after each reset.
    A {reset} is any point where diff < 0 (next value smaller than previous).
    """
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if x.size == 0:
        return pd.Series(x, index=series.index)
    offsets = np.zeros_like(x)
    running_offset = 0.0
    for i in range(1, x.size):
        if not np.isfinite(x[i]) or not np.isfinite(x[i-1]):
            offsets[i] = running_offset
            continue
        if x[i] < x[i-1]:
            running_offset += max(0.0, x[i-1] - x[i])
        offsets[i] = running_offset
    return pd.Series(x + offsets, index=series.index)

def _detect_reset_step_positions(steps: pd.Series) -> List[int]:
    """
    Return indices (row numbers) where a new segment starts due to a reset in total_steps.
    Concretely, indices i such that steps[i] < steps[i-1] → segment starts at i.
    """
    s = pd.to_numeric(steps, errors="coerce")
    d = s.diff()
    return [int(i) for i, v in enumerate(d) if i > 0 and np.isfinite(v) and v < 0]

def load_rollout_df(source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    """
    Accepts:
      - Path/str path to a CSV file
      - Raw CSV string (heuristic: contains '\\n' and ',' and not an existing file)
      - Already-constructed DataFrame
    Returns a DataFrame with normalized columns.
    Also stitches 'time_s', 'episodes_done', and 'total_steps' across phase resets so
    they remain monotonic within the returned DataFrame.
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
        raw_text = None
    elif isinstance(source, (str, Path)):
        p = Path(source) if not isinstance(source, Path) else source
        if isinstance(source, str) and ("\n" in source and "," in source) and (not Path(source).exists()):
            raw_text = source
            df = pd.read_csv(io.StringIO(source))
        else:
            raw_text = Path(p).read_text(encoding="utf-8", errors="ignore")
            df = pd.read_csv(p)
    else:
        raise TypeError("Unsupported source type. Use path, raw CSV string, or DataFrame.")

    # Normalize headers (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    # Core columns required for the first two plots
    required = ["time_s", "episodes_done", "total_steps", "avg_return_window", "avg_len_window"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Coerce numeric where it matters
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows missing axes
    df = df.dropna(subset=["time_s", "total_steps"]).reset_index(drop=True)

    # ── stitch across phase resets so axes are monotonic ──────────────────────
    df["time_s"]        = _stitch_monotone(df["time_s"])
    df["episodes_done"] = _stitch_monotone(df["episodes_done"])
    df["total_steps"]   = _stitch_monotone(df["total_steps"])

    # Optional: flag rows that follow a reset
    resets = _detect_reset_step_positions(df["total_steps"])
    flag = np.zeros(len(df), dtype=bool)
    for i in resets:
        flag[i] = True
    df["__segment_start__"] = flag

    return df

def parse_phase_starts(source: Union[str, Path, pd.DataFrame]) -> List[int]:
    """
    Detect 'CKPT' or 'phase switch' event markers from the raw CSV text file (or string).
    Returns the list of TOTAL_STEPS at which they occurred (raw values BEFORE stitching).
    Note: if your log resets counters after a switch, these raw values won't be global.
    The plotting utilities below will auto-replace these with reset-based positions
    if needed so your lines still land in the right place.
    """
    if isinstance(source, pd.DataFrame):
        return []
    text: Optional[str] = None
    if isinstance(source, (str, Path)):
        p = Path(source) if not isinstance(source, Path) else source
        if isinstance(source, str) and ("\n" in source and "," in source) and (not Path(source).exists()):
            text = source
        else:
            text = Path(p).read_text(encoding="utf-8", errors="ignore")
    else:
        return []

    switches: List[int] = []
    last_steps: Optional[int] = None

    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        parts = [x.strip() for x in s.split(",")]

        if parts and parts[0].lower() != "time_s" and len(parts) >= 3:
            try:
                last_steps = int(float(parts[2]))  # total_steps
            except Exception:
                pass

        if EVENT_PAT.search(s):
            if last_steps is not None:
                switches.append(last_steps)

    return switches

# ──────────────────────────────────────────────────────────────────────────────
# Component discovery & aggregation
# ──────────────────────────────────────────────────────────────────────────────
def discover_components(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns (reward_cols, penalty_cols, ignored_cols)
      - penalty: c_* with 'pen' in the name (case-insensitive)
      - reward : c_* not in {c_components_sum, c_components_delta} and not penalty
      - ignored: non 'c_*' columns or excluded sums/deltas
    """
    c_cols = [c for c in df.columns if c.startswith("c_")]
    penalty_cols: List[str] = []
    reward_cols: List[str] = []
    ignored_cols: List[str] = []

    for c in c_cols:
        lc = c.lower()
        if lc in ("c_components_sum", "c_components_delta"):
            ignored_cols.append(c)
            continue
        if "pen" in lc:
            penalty_cols.append(c)
        else:
            reward_cols.append(c)

    return reward_cols, penalty_cols, ignored_cols

def total_reward_series(df: pd.DataFrame, reward_cols: List[str]) -> pd.Series:
    """
    Sum of all reward components (as-is), then clip at ≥0 for the green fill.
    Always returns a pandas Series aligned to df.index.
    """
    if not reward_cols:
        print("[WARN] No reward components (c_* without 'pen') were found. Using zeros.")
        return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)

    parts = [pd.to_numeric(df[c], errors="coerce").fillna(0.0) for c in reward_cols]
    total = pd.concat(parts, axis=1).sum(axis=1)
    return total.clip(lower=0.0)

def penalty_magnitudes(df: pd.DataFrame, penalty_cols: List[str]) -> Tuple[pd.Series, List[pd.Series]]:
    """
    Returns (total_pen_mag, list_of_each_pen_mag). Each magnitude is abs(value).
    Always returns pandas Series aligned to df.index.
    """
    if not penalty_cols:
        print("[WARN] No penalty components (c_* with 'pen') were found. Using zeros.")
        zero = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
        return zero, []

    mags = [pd.to_numeric(df[c], errors="coerce").fillna(0.0).abs() for c in penalty_cols]
    total = pd.concat(mags, axis=1).sum(axis=1)
    return total, mags

def print_component_summary(reward_cols: List[str], penalty_cols: List[str]):
    r = ", ".join(reward_cols) if reward_cols else "—"
    p = ", ".join(penalty_cols) if penalty_cols else "—"
    print(f"[components] rewards ({r}) — penalties ({p})")

# ──────────────────────────────────────────────────────────────────────────────
# Tick formatters
# ──────────────────────────────────────────────────────────────────────────────
def _fmt_big_numbers(x: float, _pos=None) -> str:
    x = float(x)
    absx = abs(x)
    if absx >= 1_000_000_000:
        return f"{x/1_000_000_000:.1f}B"
    if absx >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if absx >= 1_000:
        return f"{x/1_000:.1f}k"
    return f"{int(x)}"

def _fmt_seconds(x: float, _pos=None) -> str:
    x = max(0.0, float(x))
    hours = int(x // 3600)
    minutes = int((x % 3600) // 60)
    seconds = int(x % 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"

BIG_NUM_FORMATTER = FuncFormatter(_fmt_big_numbers)
SECONDS_FORMATTER = FuncFormatter(_fmt_seconds)

# ──────────────────────────────────────────────────────────────────────────────
# >>> NEW (vz robustness): column finder and vz candidates
# ──────────────────────────────────────────────────────────────────────────────
def _normalize_name(name: str) -> str:
    """lowercase, strip, replace spaces and dots with underscores for tolerant matching."""
    return name.strip().lower().replace(" ", "_").replace(".", "_")

def _find_first_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Return the first existing column name in df that matches any candidate
    under a tolerant normalization (case-insensitive; dots/underscores treated the same).
    """
    norm_map = {_normalize_name(c): c for c in df.columns}
    for cand in candidates:
        key = _normalize_name(cand)
        if key in norm_map:
            return norm_map[key]
    # Also allow prefix/contains matching as a last resort (e.g., 'mean_vz' in 'mean_vz_fd_win')
    for cand in candidates:
        key = _normalize_name(cand)
        for k, real in norm_map.items():
            if key in k:
                return real
    return None

# Canonical list of reasonable vz header variants (ordered by preference)
_VZ_CANDIDATES = [
    "mean_vz_fd_win",  # your current CSV header
    "mean_vz_b_win",
    "mean_vz_win",
    "v_z_win",
    "vz_win",
    "mean_vz_fd",
    "mean_vz_b",
    "mean_vz",
    "v_z",
    "vz",
    "lin_vel_b.z",
    "lin_vel_b_z",
]

# ──────────────────────────────────────────────────────────────────────────────
# Legacy two plots (kept for compatibility)
# ──────────────────────────────────────────────────────────────────────────────
def plot_return_len_vs_steps(df: pd.DataFrame, switches: Optional[List[int]] = None, save_path: Optional[Union[str, Path]] = None):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ln1, = ax1.plot(df["total_steps"], df["avg_return_window"], label="Episode Return", alpha=0.85, color="C0")
    ax1.set_xlabel("Total Timesteps")
    ax1.set_ylabel("Episode Return")
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()
    ln2, = ax2.plot(df["total_steps"], df["avg_len_window"], label="Episode Length", alpha=0.85, color="C1")
    ax2.set_ylabel("Episode Length")
    ax2.tick_params(axis="y")

    if switches:
        reset_idx = _detect_reset_step_positions(df["total_steps"])
        if reset_idx:
            stitched_switch_steps = [float(df["total_steps"].iloc[i]) for i in reset_idx]
        else:
            stitched_switch_steps = switches
        for s in stitched_switch_steps:
            ax1.axvline(s, linestyle="--", alpha=0.6)

    ax1.legend([ln1, ln2], ["Episode Return", "Episode Length"], loc="upper left")
    ax1.set_title("Episode Return & Length vs Total Timesteps")
    fig.tight_layout()
    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

def plot_return_len_vs_episodes(df: pd.DataFrame, switches: Optional[List[int]] = None, save_path: Optional[Union[str, Path]] = None):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ln1, = ax1.plot(df["episodes_done"], df["avg_return_window"], label="Episode Return", alpha=0.85, color="C0")
    ax1.set_xlabel("Episodes Done")
    ax1.set_ylabel("Episode Return")
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()
    ln2, = ax2.plot(df["episodes_done"], df["avg_len_window"], label="Episode Length", alpha=0.85, color="C1")
    ax2.set_ylabel("Episode Length")
    ax2.tick_params(axis="y")

    if switches:
        reset_idx = _detect_reset_step_positions(df["total_steps"])
        if reset_idx:
            eps_marks = [float(df["episodes_done"].iloc[i]) for i in reset_idx]
            for e in eps_marks:
                ax1.axvline(e, linestyle="--", alpha=0.6, color="k")

    ax1.legend([ln1, ln2], ["Episode Return", "Episode Length"], loc="upper left")
    ax1.set_title("Episode Return & Length vs Episodes")
    fig.tight_layout()
    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# Vectorized piecewise-linear mappings for secondary axes
# ──────────────────────────────────────────────────────────────────────────────
def _build_monotone_mappings(
    x_primary: np.ndarray,
    x_other: np.ndarray
) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """
    Returns (forward, inverse) functions mapping:
      forward:  x_primary → x_other
      inverse:  x_other   → x_primary
    Both accept numpy arrays (or scalars) and return arrays.
    Robust to NaNs and duplicate x_primary; keeps last occurrence.
    """
    xp = np.asarray(x_primary, dtype=float)
    xo = np.asarray(x_other,   dtype=float)

    m = np.isfinite(xp) & np.isfinite(xo)
    xp, xo = xp[m], xo[m]

    if xp.size == 0 or xo.size == 0:
        def _id(u): return np.asarray(u, dtype=float)
        return _id, _id

    order = np.argsort(xp, kind="mergesort")
    xp = xp[order]; xo = xo[order]

    keep = np.ones_like(xp, dtype=bool)
    keep[:-1] = xp[1:] != xp[:-1]
    xp = xp[keep]; xo = xo[keep]

    if xp.size < 2:
        def _const(u):
            u = np.asarray(u, dtype=float)
            return np.full_like(u, fill_value=xo[-1])
        return _const, _const

    def fwd(u):
        u = np.asarray(u, dtype=float)
        return np.interp(u, xp, xo)

    def inv(v):
        v = np.asarray(v, dtype=float)
        return np.interp(v, xo, xp)

    return fwd, inv

# ──────────────────────────────────────────────────────────────────────────────
# One plot with 3 X-axes (episodes bottom; steps + wallclock as twin tops)
# ──────────────────────────────────────────────────────────────────────────────
def plot_return_len_with_multi_x(df: pd.DataFrame, switches: Optional[List[int]] = None, save_path: Optional[Union[str, Path]] = None):
    x_eps  = pd.to_numeric(df["episodes_done"], errors="coerce").to_numpy()
    y_ret  = pd.to_numeric(df["avg_return_window"], errors="coerce").to_numpy()
    y_len  = pd.to_numeric(df["avg_len_window"],  errors="coerce").to_numpy()
    x_step = pd.to_numeric(df["total_steps"],     errors="coerce").to_numpy()
    x_time = pd.to_numeric(df["time_s"],          errors="coerce").to_numpy()

    fig, ax = plt.subplots(figsize=(12, 6))

    ln1, = ax.plot(x_eps, y_ret, label="Episode Return", alpha=0.90, color="C0")
    ax.set_xlabel("Episodes Done")
    ax.set_ylabel("Episode Return")
    ax.xaxis.set_major_formatter(BIG_NUM_FORMATTER)

    ax2 = ax.twinx()
    ln2, = ax2.plot(x_eps, y_len, label="Episode Length", alpha=0.85, color="C1")
    ax2.set_ylabel("Episode Length")

    if switches:
        reset_idx = _detect_reset_step_positions(df["total_steps"])
        if reset_idx:
            eps_marks = [float(df["episodes_done"].iloc[i]) for i in reset_idx]
            for e in eps_marks:
                ax.axvline(e, linestyle="--", alpha=0.6, color="k")

    fwd_steps, inv_steps = _build_monotone_mappings(x_eps, x_step)
    sec_top_steps = ax.secondary_xaxis('top', functions=(fwd_steps, inv_steps))
    sec_top_steps.set_xlabel("Total Timesteps")
    sec_top_steps.xaxis.set_major_formatter(BIG_NUM_FORMATTER)

    fwd_time, inv_time = _build_monotone_mappings(x_eps, x_time)
    sec_top_time = ax.secondary_xaxis('top', functions=(fwd_time, inv_time))
    sec_top_time.set_xlabel("Wall Clock Time")
    sec_top_time.xaxis.set_major_formatter(SECONDS_FORMATTER)
    sec_top_time.spines["top"].set_position(("outward", 36))
    sec_top_time.set_frame_on(True)

    ax.legend([ln1, ln2], ["Episode Return", "Episode Length"], loc="upper left")
    ax.set_title("Episode Return & Length — Episodes (primary) with Steps & Time (secondary x-axes)")
    fig.tight_layout()
    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# Bracket plots — EPISODES on x-axis
# ──────────────────────────────────────────────────────────────────────────────
def plot_reward_penalties_bracket(
    df: pd.DataFrame,
    switches: Optional[List[int]] = None,
    offset_frac: float = 0.02,
    save_path: Optional[Union[str, Path]] = None,
):
    reward_cols, penalty_cols, _ = discover_components(df)
    print_component_summary(reward_cols, penalty_cols)

    x_eps  = pd.to_numeric(df["episodes_done"], errors="coerce")
    total_reward = total_reward_series(df, reward_cols)
    total_pen_mag, _ = penalty_magnitudes(df, penalty_cols)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(x_eps, 0.0, total_reward, alpha=0.6, color="green", label="Rewards (Σ c_*)")

    pbase = total_reward.copy()
    ax.fill_between(x_eps, pbase, pbase + total_pen_mag, alpha=0.30, color="red", label="Penalties (Σ|c_*pen|)")

    reward_minus_pen = total_reward - total_pen_mag
    if "avg_return_window" in df.columns:
        y_ref = np.vstack([
            reward_minus_pen.to_numpy(),
            pd.to_numeric(df["avg_return_window"], errors="coerce").to_numpy()
        ]).astype(float)
    else:
        y_ref = reward_minus_pen.to_numpy()[None, :]

    y_min = float(np.nanmin(y_ref)); y_max = float(np.nanmax(y_ref))
    y_span = max(1e-6, y_max - y_min)
    eps = offset_frac * y_span
    ax.plot(x_eps, reward_minus_pen + eps, linestyle="--", linewidth=2.4, color="black",
            label=f"ΣRewards − Σ|Penalties| (+{offset_frac:.0%} offset)", zorder=6, dash_capstyle="round")

    if "avg_return_window" in df.columns:
        ax.plot(x_eps, df["avg_return_window"], linewidth=2.0, color="orange", alpha=0.95,
                label="Return (avg_return_window)", zorder=5)

    if switches:
        reset_idx = _detect_reset_step_positions(df["total_steps"])
        if reset_idx:
            eps_marks = [float(df["episodes_done"].iloc[i]) for i in reset_idx]
            for e in eps_marks:
                ax.axvline(e, linestyle="--", alpha=0.6, color="k")

    ax.set_xlabel("Episodes Done")
    ax.set_ylabel("Magnitude")
    ax.set_title("Rewards vs Penalties (Bracket View — x=Episodes)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

def plot_reward_penalties_split_bracket(
    df: pd.DataFrame,
    switches: Optional[List[int]] = None,
    offset_frac: float = 0.02,
    save_path: Optional[Union[str, Path]] = None,
):
    """
    Per your spec:
      • Solid green line = Σ rewards (c_* without 'pen')
      • Orange line      = Return (avg_return_window)
      • Red stacked area = BETWEEN orange (bottom) and green (top) — penalties
      • Area under the RETURN curve (down to 0) is green
      • Also plots ΣRewards−Σ|Penalties| + offset (black dotted)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # ── Early guard: empty dataframe → placeholder and return ────────────────
    if df is None or len(df) == 0:
        print("[components] DataFrame is empty; emitting placeholder plot.")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No data in CSV", ha="center", va="center")
        ax.set_axis_off()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()
        return

    # Discover columns (use your script's helpers)
    reward_cols, penalty_cols, _ = discover_components(df)
    print_component_summary(reward_cols, penalty_cols)

    # X axis (episodes) — robust to missing/NaN
    if "episodes_done" in df.columns:
        x_eps = pd.to_numeric(df["episodes_done"], errors="coerce").to_numpy()
    else:
        x_eps = np.arange(len(df), dtype=float)

    # Core series (robust to missing/NaN)
    y_ret = pd.to_numeric(df.get("avg_return_window", 0.0), errors="coerce").fillna(0.0).to_numpy()

    # Σ rewards (c_* without 'pen'); if none, use zeros
    if reward_cols:
        y_rew = total_reward_series(df, reward_cols).fillna(0.0).to_numpy()
    else:
        print("[components] No reward component columns found; using zeros for Σ rewards.")
        y_rew = np.zeros_like(y_ret, dtype=float)

    # Penalty magnitudes per component (positive values); if none, keep empty
    pen_mags = []
    if penalty_cols:
        _, pen_mags_raw = penalty_magnitudes(df, penalty_cols)
        pen_mags = [m.fillna(0.0).to_numpy() for m in pen_mags_raw if m is not None]
    else:
        print("[components] No penalty component columns found; will skip red stacked area.")

    # Ensure arrays have common length (clip to min length)
    lens = [len(x_eps), len(y_ret), len(y_rew)] + [len(m) for m in pen_mags]
    L = min(lens) if lens else 0
    if L == 0:
        # Degenerate case: nothing to draw; emit placeholder
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No valid numeric data to plot", ha="center", va="center")
        ax.set_axis_off()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()
        return

    x_eps = x_eps[:L]
    y_ret = y_ret[:L]
    y_rew = y_rew[:L]
    pen_mags = [m[:L] for m in pen_mags]

    # Build scaled stacked penalties so that sum of layers spans from return up to Σ rewards
    # diff = how much of Σ rewards sits above return (can't be negative)
    diff = np.maximum(y_rew - y_ret, 0.0)

    # Sum of raw penalty magnitudes
    pen_sum = np.zeros(L, dtype=float)
    if pen_mags:
        pen_sum = np.sum(np.vstack(pen_mags), axis=0)

    # If there are no penalties (or all zeros), don't try to stack red areas
    pen_mags_scaled = []
    if pen_mags and np.any(pen_sum > 0.0):
        denom = np.clip(pen_sum, 1e-12, None)  # avoid division by zero
        scale = diff / denom
        # Avoid pathological explosions when denom ~ 0
        scale = np.clip(scale, 0.0, 10.0)
        pen_mags_scaled = [mag * scale for mag in pen_mags]

    # Build the reference "ΣRewards − Σ|Penalties|" curve
    if penalty_cols and pen_mags:
        pens_total, _ = penalty_magnitudes(df.iloc[:L], penalty_cols)
        pens = pens_total.fillna(0.0).to_numpy()
    else:
        pens = np.zeros(L, dtype=float)
    reward_minus_pen = y_rew - pens

    # Robust y-limits
    y_ref_parts = []
    for arr in (reward_minus_pen, y_ret, y_rew):
        if arr.size:
            finite = arr[np.isfinite(arr)]
            if finite.size:
                y_ref_parts.append(finite)
    if y_ref_parts:
        y_ref = np.concatenate(y_ref_parts)
        y_min = float(np.nanmin(y_ref))
        y_max = float(np.nanmax(y_ref))
        if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
            y_min, y_max = -1.0, 1.0
    else:
        y_min, y_max = -1.0, 1.0
    y_span = max(1e-6, y_max - y_min)
    eps = offset_frac * y_span

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    # Area under the return curve (down to 0)
    ax.fill_between(x_eps, 0.0, y_ret, alpha=0.30, color="green", label="Area under Return")

    # Lines: Σ rewards (green) and Return (orange)
    ax.plot(x_eps, y_rew, color="green", linewidth=2.2, label="Σ rewards (c_*)")
    ax.plot(x_eps, y_ret, color="orange", linewidth=2.0, label="Return (avg_return_window)")

    # Red stacked area between return (base) and Σ rewards (top), scaled by component magnitudes
    if pen_mags_scaled:
        n = max(1, len(pen_mags_scaled))
        reds = [plt.cm.Reds(0.35 + 0.55 * (i / max(1, n - 1))) for i in range(n)]
        base = y_ret.copy()
        for cname, mag_s, color in zip(penalty_cols, pen_mags_scaled, reds):
            top = np.minimum(base + mag_s, y_rew)  # never exceed Σ rewards
            ax.fill_between(x_eps, base, top, alpha=0.35, color=color, label=f"|{cname}|")
            base = top

    # Black dotted: ΣRewards − Σ|Penalties| (with slight upward offset for readability)
    ax.plot(
        x_eps,
        reward_minus_pen + eps,
        linestyle="--",
        linewidth=2.4,
        color="black",
        label=f"ΣRewards − Σ|Penalties| (+{offset_frac:.0%} offset)",
        zorder=6,
        dash_capstyle="round",
    )

    # Phase boundaries (vertical dashed lines at episode indices where total_steps reset)
    if switches:
        try:
            reset_idx = _detect_reset_step_positions(df["total_steps"].iloc[:L])
        except Exception:
            reset_idx = []
        if reset_idx:
            eps_marks = []
            for i in reset_idx:
                try:
                    eps_marks.append(float(df["episodes_done"].iloc[int(i)]))
                except Exception:
                    pass
            for e in eps_marks:
                ax.axvline(e, linestyle="--", alpha=0.6, color="k")

    ax.set_xlabel("Episodes Done")
    ax.set_ylabel("Magnitude")
    ax.set_title("ΣRewards (green) vs Return (orange) — Red Area = Penalties (x = Episodes)")
    ax.set_ylim(y_min - 0.05 * y_span, y_max + 0.05 * y_span)
    ax.legend(loc="upper left")
    fig.tight_layout()

    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Kinematics — EPISODES on x-axis
# ──────────────────────────────────────────────────────────────────────────────
def plot_kinematics_over_time(df: pd.DataFrame, save_path: Optional[Union[str, Path]] = None):
    # >>> NEW: include vz candidates and be tolerant to header variants
    base_candidates = ["mean_wz_b_win", "mean_vxy_b_win", "mean_vx_b_win", "mean_vx_b"]
    present = [c for c in base_candidates if c in df.columns]

    # try to find a vz column robustly
    vz_col = _find_first_column(df, _VZ_CANDIDATES)
    if vz_col is not None and vz_col not in present:
        present.append(vz_col)

    if not present:
        print("[INFO] No kinematic columns found among:", ", ".join(base_candidates + _VZ_CANDIDATES))
        return

    x_eps = pd.to_numeric(df["episodes_done"], errors="coerce")
    fig, ax = plt.subplots(figsize=(12, 6))
    for c in present:
        ax.plot(x_eps, pd.to_numeric(df[c], errors="coerce"), label=c, alpha=0.95)

    ax.set_xlabel("Episodes Done")
    ax.set_ylabel("Kinematic Value")
    ax.set_title("Kinematics vs Episodes")
    ax.legend(loc="best")
    fig.tight_layout()
    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# >>> NEW: Dedicated vertical velocity plot — WALL CLOCK on x-axis
# ──────────────────────────────────────────────────────────────────────────────
def plot_vz_vs_wallclock(df: pd.DataFrame, save_path: Optional[Union[str, Path]] = None):
    """
    Robust vertical velocity plot (vz) using wall-clock time on x.
    Picks the first matching vz column among a list of tolerant candidates.
    """
    vz_col = _find_first_column(df, _VZ_CANDIDATES)
    if vz_col is None:
        print("[INFO] No vertical-velocity column found. Tried:", ", ".join(_VZ_CANDIDATES))
        return

    x_time = pd.to_numeric(df["time_s"], errors="coerce")
    y_vz   = pd.to_numeric(df[vz_col],   errors="coerce")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_time, y_vz, linewidth=2.0)
    ax.set_xlabel("Wall Clock Time (s)")
    ax.set_ylabel(f"{vz_col} (vertical velocity)")
    ax.set_title(f"Vertical Velocity vs Wall Clock — using '{vz_col}'")
    ax.xaxis.set_major_formatter(SECONDS_FORMATTER)
    fig.tight_layout()
    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# Return components (ONLY penalties as lines + total rewards line), x=episodes
# ──────────────────────────────────────────────────────────────────────────────
def plot_return_components_lines_vs_episodes(df: pd.DataFrame, switches: Optional[List[int]] = None, save_path: Optional[Union[str, Path]] = None):
    """
    Per your spec:
      • Plot each PENALTY component (c_*pen*) as its own line.
      • Do NOT plot individual reward components.
      • Plot only the TOTAL Σ rewards line (green).
      • Overlay the return (orange).
      • Excludes c_components_sum / c_components_delta by design.
    """
    reward_cols, penalty_cols, _ = discover_components(df)

    x_eps = pd.to_numeric(df["episodes_done"], errors="coerce")
    fig, ax = plt.subplots(figsize=(12, 6))

    for cname in penalty_cols:
        y = pd.to_numeric(df[cname], errors="coerce")
        ax.plot(x_eps, y, label=cname, alpha=0.9, linestyle="--")

    y_rew_sum = total_reward_series(df, reward_cols)
    ax.plot(x_eps, y_rew_sum, color="green", linewidth=2.2, label="Σ rewards (sum)")
    if "avg_return_window" in df.columns:
        ax.plot(x_eps, pd.to_numeric(df["avg_return_window"], errors="coerce"),
                color="orange", linewidth=2.0, label="Return (avg_return_window)")

    if switches:
        reset_idx = _detect_reset_step_positions(df["total_steps"])
        if reset_idx:
            eps_marks = [float(df["episodes_done"].iloc[i]) for i in reset_idx]
            for e in eps_marks:
                ax.axvline(e, linestyle="--", alpha=0.6, color="k")

    ax.set_xlabel("Episodes Done")
    ax.set_ylabel("Component Value")
    ax.set_title("Return Components vs Episodes (penalties lines + Σ rewards)")
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

def plot_components_stacked_vs_steps(
    df: pd.DataFrame,
    switches: Optional[List[int]] = None,
    include_avg_return_overlay: bool = True,
    save_path: Optional[Union[str, Path]] = None,
):
    plot_reward_penalties_bracket(df, switches=switches, offset_frac=0.02, save_path=save_path)

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate rollout plots and save to a directory.")
    parser.add_argument("--csv", required=True, help="Path to rollout_log.csv")
    parser.add_argument("--outdir", required=True, help="Directory to save plots")
    parser.add_argument("--prefix", default="", help="Optional filename prefix for saved plots")
    args = parser.parse_args()

    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_rollout_df(args.csv)
    switches = parse_phase_starts(args.csv)

    def out(name: str) -> Path:
        base = f"{args.prefix + ('_' if args.prefix and not args.prefix.endswith('_') else '')}{name}"
        return outdir / f"{base}.png"

    # Generate and save plots
    plot_return_len_with_multi_x(df, switches, save_path=out("return_len_multi_x"))
    plot_components_stacked_vs_steps(df, switches, save_path=out("rewards_bracket_episodes"))
    plot_reward_penalties_split_bracket(df, switches, save_path=out("rewards_split_bracket_episodes"))
    plot_kinematics_over_time(df, save_path=out("kinematics_vs_episodes"))
    plot_return_components_lines_vs_episodes(df, switches, save_path=out("components_penalties_lines_vs_episodes"))
    plot_vz_vs_wallclock(df, save_path=out("vz_vs_wallclock"))

    # Optional legacy plots
    plot_return_len_vs_steps(df, switches, save_path=out("return_len_vs_steps"))
    plot_return_len_vs_episodes(df, switches, save_path=out("return_len_vs_episodes"))

    print(f"[done] Saved plots to: {outdir.resolve()}")

if __name__ == "__main__":
    main()
