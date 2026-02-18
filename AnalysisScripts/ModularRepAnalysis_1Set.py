#!/usr/bin/env python3
# === Ant Actor (3 models) — Cosine module explorer + Model2-reference stability + OLD-style network sketch (dead neurons removed) ===
#
# Script conversion of your single Jupyter cell.
#
# Examples:
#   python ant_actor_cosine_module_explorer.py
#
#   python ant_actor_cosine_module_explorer.py \
#     --model-walk "/path/to/walk_for_play.pth" \
#     --model-spin "/path/to/spin_for_play.pth" \
#     --model-jump "/path/to/jump_for_play.pth" \
#     --all-states "/path/to/ALL_states_concat.npy" \
#     --n-all-states 500000
#
# Headless behavior:
#   - If no DISPLAY is available, the script will save figures to --out-dir
#     (default: ./_module_explorer_figs) instead of calling plt.show().

from __future__ import annotations

import os
import re
import math
import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

# ------------------------------------------------------------
# Matplotlib backend selection (must happen BEFORE pyplot import)
# ------------------------------------------------------------
import matplotlib


def _is_headless() -> bool:
    # Common headless check: no DISPLAY on *nix. On macOS local, DISPLAY may be empty
    # but GUI can still work; we treat macOS as non-headless by default.
    if sys.platform == "darwin":
        return False
    return os.environ.get("DISPLAY", "") == ""


HEADLESS = _is_headless()
if HEADLESS:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import TwoSlopeNorm, BoundaryNorm
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, fcluster, leaves_list
from scipy.spatial.distance import squareform
from scipy.optimize import linear_sum_assignment


# ============================================================
# ========================= CONFIG ===========================
# ============================================================

# --- REQUIRED: set your 3 model paths here (defaults can be overridden via CLI) ---
DEFAULT_MODEL1_PATH = "/Users/adi/Desktop/testing/c024_b01_walk_plateau_2026-02-05_10-36-38_for_play.pth"
DEFAULT_MODEL2_PATH = "/Users/adi/Desktop/testing/c024_b02_spin_plateau_2026-02-05_12-00-53_for_play.pth"  # reference
DEFAULT_MODEL3_PATH = "/Users/adi/Desktop/testing/c024_b03_jump_plateau_2026-02-05_13-09-45_for_play.pth"

# --- States pool default (constructed from REPO_ROOT; can override via CLI) ---
# NOTE: your original code had a space in folder name; we keep it exactly.
DEFAULT_ALL_STATES_REL = "Checkpoints_ States_selectedGraphs/ALL_states_concat.npy"
DEFAULT_N_ALL_STATES = 500_000

# --- Layers to analyze (hidden layers) ---
DEFAULT_LAYER_INDICES = [0, 1]  # h1=0, h2=1

# --- Similarity / normalization config ---
DEFAULT_SEED = 42
EPS = 1e-8
MIN_STD = 1e-5
DEFAULT_ACTIVATION = "relu"  # {"elu","relu","tanh"}

# --- Clustering config (on |cosine|) ---
K_MIN, K_MAX = 2, 10
ALPHA_SMALL_K = 0.05

# --- Heatmap visuals ---
CMAP = "RdBu_r"
VMIN, VCENTER, VMAX = -1.0, 0.0, 1.0

# --- OLD-style network sketch config (vertical, black edges) ---
SKETCH_THRESHOLD_PERCENT_BY_SEG = (97, 97, 85)  # (input→h1, h1→h2, h2→out) percentile on |weights|
SKETCH_MAX_EDGES_PER_SEG = 12000
SKETCH_EDGE_ALPHA_BY_SEG = (1, 1, 1)
SKETCH_LW_SCALE_BY_SEG = (1.25, 1.25, 1.25)
SKETCH_MAX_LW = 12.0
SKETCH_CIRCLE_SIZE = 100

OUTPUT_MOTORS = 8
OUTPUT_OFFSET = 0


# ============================================================
# ========================= SETUP ============================
# ============================================================

def _seed_everything(seed: int):
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _torch_load_compat(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")


def _load_rlg_forplay_state_dict(path: str) -> dict:
    payload = _torch_load_compat(path)
    if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        sd = {}
        for k, v in payload["model"].items():
            sd[k] = v.detach().cpu().clone() if torch.is_tensor(v) else v
        return sd
    raise ValueError(f"[ckpt] {path} missing 'model' dict (for-play).")


def _discover_actor_mlp_layers(state_dict: dict):
    candidates = []
    for k in state_dict.keys():
        if k.endswith(".weight") and ("actor_mlp" in k or ".actor." in k or "actor_net" in k or "actor.trunk" in k):
            parts = k.split(".")
            try:
                idx = int(parts[-2])
                bias_key = k[:-6] + "bias"
                if bias_key in state_dict:
                    candidates.append((idx, k, bias_key))
            except Exception:
                candidates.append((10**6, k, k[:-6] + "bias"))
    candidates.sort(key=lambda t: (t[0], t[1]))
    return candidates


def _first_linear_in(actor_layers, sd):
    if not actor_layers:
        raise RuntimeError("No actor MLP layers found.")
    _, wkey, _ = actor_layers[0]
    W = sd[wkey]
    if isinstance(W, torch.Tensor):
        W = W.detach().cpu().numpy()
    if W.ndim != 2:
        raise RuntimeError(f"Unexpected weight shape for {wkey}: {tuple(W.shape)}")
    return W.shape[1], W.shape[0]  # in_dim, out_dim


def _activation(x: np.ndarray, name: str) -> np.ndarray:
    if name == "elu":
        y = x.copy()
        neg = x <= 0
        y[neg] = np.expm1(x[neg])
        return y
    if name == "relu":
        return np.maximum(0.0, x)
    if name == "tanh":
        return np.tanh(x)
    raise ValueError(f"Unsupported ACTIVATION '{name}'")


def _actor_hidden_forward_numpy(X, actor_layers, sd, activation_name, max_layers: int = 2):
    outs = []
    h = X
    L = min(max_layers, len(actor_layers))
    for li in range(L):
        _, wkey, bkey = actor_layers[li]
        W = sd[wkey]
        if isinstance(W, torch.Tensor):
            W = W.detach().cpu().numpy()
        b = sd[bkey]
        if isinstance(b, torch.Tensor):
            b = b.detach().cpu().numpy()
        h = h @ W.T + b[None, :]
        h = _activation(h, activation_name)
        outs.append(h.astype(np.float32, copy=False))
    return outs


def get_layer_output_batch_from_ckpt(sd, inputs: torch.Tensor, layer_idx: int, actor_layers, activation_name, max_layer_idx: int):
    X = inputs.detach().cpu().numpy().astype(np.float32, copy=False)
    outs = _actor_hidden_forward_numpy(X, actor_layers, sd, activation_name, max_layers=max_layer_idx + 1)
    if layer_idx < 0 or layer_idx >= len(outs):
        raise IndexError(f"layer_idx={layer_idx} out of range (0..{len(outs)-1})")
    return outs[layer_idx]


def _zscore_cols(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if X.size == 0:
        return X.astype(np.float32, copy=False)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return ((X - mu) / (sd + eps)).astype(np.float32, copy=False)


def _dead_alive_indices(acts_post: np.ndarray, min_std: float):
    if acts_post.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    sd = acts_post.std(axis=0)
    dead_mask = sd <= min_std
    dead_idx = np.where(dead_mask)[0].astype(int)
    alive_idx = np.where(~dead_mask)[0].astype(int)
    return dead_idx, alive_idx


def cosine_sim_matrix_cols(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Cosine similarity between columns of X (T,N). Returns (N,N) in [-1,1], diag=1.
    """
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[1]
    if n == 0:
        return np.zeros((0, 0), np.float32)
    if n == 1:
        return np.ones((1, 1), np.float32)

    norms = np.linalg.norm(X, axis=0, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    Xn = X / norms
    R = (Xn.T @ Xn).astype(np.float32, copy=False)
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    R = np.clip(R, -1.0, 1.0)
    np.fill_diagonal(R, 1.0)
    return R


def _norm01(a):
    a = np.array(a, dtype=np.float64)
    if np.all(np.isnan(a)):
        return np.zeros_like(a)
    lo, hi = np.nanmin(a), np.nanmax(a)
    return np.zeros_like(a) if hi - lo < 1e-12 else (a - lo) / (hi - lo)


def _within_between_stats(R_abs, labels):
    if R_abs.size == 0 or labels.size == 0:
        return dict(within=np.nan, between=np.nan, delta=np.nan)
    same = (labels[:, None] == labels[None, :])
    np.fill_diagonal(same, False)
    within = R_abs[same]
    between = R_abs[~same]
    w = float(within.mean()) if within.size else np.nan
    b = float(between.mean()) if between.size else np.nan
    return dict(within=w, between=b, delta=(w - b) if np.isfinite(w) and np.isfinite(b) else np.nan)


def _modularity_Q_pos(R_abs, labels):
    A = np.clip(R_abs, 0.0, None).astype(np.float64)
    np.fill_diagonal(A, 0.0)
    m = A.sum() / 2.0
    if m <= 0:
        return np.nan
    k = A.sum(axis=1)
    Q = 0.0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if labels[i] == labels[j]:
                Q += A[i, j] - (k[i] * k[j] / (2.0 * m))
    return float(Q / (2.0 * m))


def _cluster_order_and_labels_abs(R_abs, K):
    n = int(R_abs.shape[0])
    if n == 0:
        return np.array([], int), np.array([], int)
    if n < 3:
        order = np.arange(n, dtype=int)
        labels = np.ones(n, dtype=int)
        return order, labels

    R_abs = 0.5 * (R_abs + R_abs.T)
    R_abs = np.clip(R_abs, 0.0, 1.0)

    D = 1.0 - R_abs
    D = np.clip(D, 0.0, 1.0)
    np.fill_diagonal(D, 0.0)

    y = squareform(D, checks=False)
    Z = linkage(y, method="average")
    Z = optimal_leaf_ordering(Z, y)
    order = leaves_list(Z)
    labels = fcluster(Z, t=K, criterion="maxclust")
    return order, labels


def choose_K_and_order_abs(R_abs, k_min=2, k_max=10, alpha_small_k=0.05):
    n = int(R_abs.shape[0])
    if n == 0:
        return 2, np.array([], int), np.array([], int)
    if n < 3:
        order = np.arange(n, dtype=int)
        labels = np.ones(n, dtype=int)
        return 1, order, labels

    Ks = list(range(max(2, k_min), max(k_min, k_max) + 1))
    deltas, Qs, orders, labels_store = [], [], {}, {}
    for K in Ks:
        order, labels = _cluster_order_and_labels_abs(R_abs, K)
        orders[K] = order
        labels_store[K] = labels
        st = _within_between_stats(R_abs, labels)
        deltas.append(st["delta"])
        Qs.append(_modularity_Q_pos(R_abs, labels))

    dN = _norm01(deltas)
    qN = _norm01(Qs)
    kN = (np.array(Ks) - min(Ks)) / (max(Ks) - min(Ks) + 1e-12)
    score = 0.6 * dN + 0.4 * qN - alpha_small_k * kN
    K_best = Ks[int(np.nanargmax(score))]
    return K_best, orders[K_best], labels_store[K_best]


def _relabel_clusters_by_size(labels):
    labels = np.asarray(labels)
    if labels.size == 0:
        return labels.astype(int)
    unique_cids, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)
    sorted_cids = unique_cids[order]
    cid_to_rank = {cid: rank + 1 for rank, cid in enumerate(sorted_cids)}
    return np.array([cid_to_rank[c] for c in labels], dtype=int)


def _order_by_cluster_size(labels, base_order=None):
    labels = np.asarray(labels)
    n = labels.size
    if n == 0:
        return np.arange(0, dtype=int)
    if base_order is None:
        base_order = np.arange(n, dtype=int)
    base_order = np.asarray(base_order)

    unique_cids = np.unique(labels)
    sizes = [int(np.sum(labels == cid)) for cid in unique_cids]
    sorted_cids = [cid for cid, _ in sorted(zip(unique_cids, sizes), key=lambda t: -t[1])]

    labels_in_base = labels[base_order]
    new_order_list = []
    for cid in sorted_cids:
        new_order_list.append(base_order[labels_in_base == cid])
    return np.concatenate(new_order_list) if new_order_list else np.arange(n, dtype=int)


def _cluster_bounds(labels, order):
    lr = labels[order]
    return [i - 0.5 for i in range(1, len(lr)) if lr[i] != lr[i - 1]]


def hungarian_match_cosine(A, B, eps=1e-8):
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    if A.size == 0 or B.size == 0:
        sim = np.zeros((A.shape[1] if A.ndim == 2 else 0, B.shape[1] if B.ndim == 2 else 0), dtype=np.float32)
        return sim, np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)

    nA = np.linalg.norm(A, axis=0, keepdims=True)
    nB = np.linalg.norm(B, axis=0, keepdims=True)
    nA[nA < eps] = 1.0
    nB[nB < eps] = 1.0

    sim = (A.T @ B) / (nA.T @ nB)
    sim = np.nan_to_num(sim).astype(np.float32, copy=False)
    sim = np.clip(sim, -1.0, 1.0)

    i_idx, j_idx = linear_sum_assignment(-sim)
    matched_vals = sim[i_idx, j_idx] if i_idx.size else np.array([], dtype=float)
    return sim, i_idx, j_idx, matched_vals


def _safe_cos_abs(a, b, eps: float = 1e-8) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.size != b.size or a.size == 0:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return 0.0
    c = float(np.dot(a, b) / (na * nb))
    if not np.isfinite(c):
        return 0.0
    c = max(-1.0, min(1.0, c))
    return float(abs(c))


def _overlay_module_boxes(ax, labels_ord, cluster_cmap, cluster_norm, alpha_fill=0.45, edge_alpha=0.90):
    labels_ord = np.asarray(labels_ord, dtype=int)
    if labels_ord.size == 0:
        return
    start = 0
    cur = int(labels_ord[0])
    spans = []
    for i in range(1, labels_ord.size):
        cid = int(labels_ord[i])
        if cid != cur:
            spans.append((cur, start, i - 1))
            cur = cid
            start = i
    spans.append((cur, start, labels_ord.size - 1))

    for cid, a, b in spans:
        color = cluster_cmap(cluster_norm(int(cid)))
        w = (b - a + 1)
        ax.add_patch(
            Rectangle(
                (a - 0.5, a - 0.5),
                w,
                w,
                facecolor=color,
                edgecolor=(color[0], color[1], color[2], edge_alpha),
                linewidth=2.0,
                alpha=alpha_fill,
                zorder=3,
            )
        )


# ============================================================
# ===================== PLOTTING HELPERS =====================
# ============================================================

def _maybe_save_or_show(fig, out_path: Path | None, show: bool):
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"[saved] {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def _plot_stability_6graphs(layer_indices, layer_stability, cluster_vis, out_path: Path | None, show: bool):
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.0))
    fig.subplots_adjust(wspace=0.25, hspace=0.30)

    STABILITY_DOT_SIZE = 60  # bigger dots (as in your cell)

    for r, layer_idx in enumerate(layer_indices):
        st = layer_stability.get(layer_idx, None)
        if st is None:
            for c in range(3):
                axes[r, c].axis("off")
            continue

        o1, o2 = st["others"]
        cmap = cluster_vis[layer_idx]["cmap"]
        norm = cluster_vis[layer_idx]["norm"]
        cids = st["cluster_ids_ref"]

        plots = [
            ("Model2(ref) vs Model1", st["act_21"], st["conn_21"]),
            ("Model2(ref) vs Model3", st["act_23"], st["conn_23"]),
            ("Avg (Model2 vs both)", st["act_avg"], st["conn_avg"]),
        ]

        last_sc = None
        for c, (title, x, y) in enumerate(plots):
            ax = axes[r, c]
            last_sc = ax.scatter(
                x,
                y,
                c=cids,
                cmap=cmap,
                norm=norm,
                s=STABILITY_DOT_SIZE,
                alpha=0.75,
                linewidths=0.0,
            )
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, alpha=0.15)
            ax.set_title(f"Layer {layer_idx+1}: {title}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Activation stability (abs cosine)")
            ax.set_ylabel("Connectivity stability (abs cosine)")

        cb = fig.colorbar(last_sc, ax=axes[r, :].tolist(), fraction=0.018, pad=0.01)
        cb.set_label("Cluster (size-rank, from Model2)", fontsize=10)

    fig.suptitle(
        "Activation vs Connectivity stability (reference = Model2 / spin)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    _maybe_save_or_show(fig, out_path, show)


def _plot_one_model_heatmaps_inline(
    m: str,
    model_paths,
    layer_indices,
    layer_data,
    layer_stability,
    cluster_vis,
    out_path: Path | None,
    show: bool,
):
    fig = plt.figure(figsize=(15.5, 10.6))
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        width_ratios=[1.0, 1.0],
        height_ratios=[0.06, 1.0, 0.34],
        wspace=0.10,
        hspace=0.08,
    )

    ax_top0 = fig.add_subplot(gs[0, 0])
    ax_top1 = fig.add_subplot(gs[0, 1])
    ax_hm0 = fig.add_subplot(gs[1, 0])
    ax_hm1 = fig.add_subplot(gs[1, 1])
    ax_sc0 = fig.add_subplot(gs[2, 0])
    ax_sc1 = fig.add_subplot(gs[2, 1])

    for col, layer_idx in enumerate(layer_indices):
        d = layer_data[layer_idx][m]
        labels = d["labels"]
        order = d["order"]
        R = d["R_signed"]
        R_plot = R[np.ix_(order, order)] if R.size else R
        labels_ord = labels[order] if labels.size else labels

        cmapC = cluster_vis[layer_idx]["cmap"]
        normC = cluster_vis[layer_idx]["norm"]

        ax_top = ax_top0 if col == 0 else ax_top1
        ax_hm = ax_hm0 if col == 0 else ax_hm1
        ax_sc = ax_sc0 if col == 0 else ax_sc1

        if labels_ord.size:
            sns.heatmap(
                labels_ord[None, :],
                ax=ax_top,
                cmap=cmapC,
                norm=normC,
                cbar=False,
                xticklabels=False,
                yticklabels=False,
            )
        else:
            ax_top.axis("off")

        if R_plot.size:
            sns.heatmap(
                R_plot,
                ax=ax_hm,
                cmap=CMAP,
                vmin=VMIN,
                vmax=VMAX,
                center=VCENTER,
                square=False,
                cbar=False,
                xticklabels=False,
                yticklabels=False,
            )
        else:
            ax_hm.text(0.5, 0.5, "Empty", ha="center", va="center")
            ax_hm.set_xticks([])
            ax_hm.set_yticks([])

        if labels_ord.size and R_plot.size:
            _overlay_module_boxes(ax_hm, labels_ord, cmapC, normC, alpha_fill=0.35, edge_alpha=0.85)
            for b in _cluster_bounds(labels, order):
                ax_hm.axhline(b, color="k", lw=0.85, zorder=5)
                ax_hm.axvline(b, color="k", lw=0.85, zorder=5)

        st = layer_stability.get(layer_idx, None)
        ax_sc.set_ylim(0.0, 1.05)
        ax_sc.axhline(1.0, ls="--", lw=1.0, alpha=0.35)
        ax_sc.set_xticks([])
        ax_sc.set_xlabel(f"Layer {layer_idx+1}", fontsize=18, fontweight="bold", labelpad=6)
        if col == 0:
            ax_sc.set_ylabel("Persistence score", fontsize=15)
        else:
            ax_sc.set_ylabel("")
            ax_sc.set_yticks([])
            ax_sc.set_yticklabels([])

        if st is None:
            ax_sc.text(0.5, 0.5, "No families", ha="center", va="center", transform=ax_sc.transAxes)
        else:
            scores = st["score_by_model"][m]
            fam_idx = st["fam_idx_by_model"][m]

            n_alive = int(layer_data[layer_idx][m]["A_z"].shape[1])
            score_full = np.full(n_alive, np.nan, dtype=np.float32)
            score_full[fam_idx] = scores

            score_ord = score_full[order]
            x = np.arange(score_ord.size)
            mask = np.isfinite(score_ord)
            if np.any(mask):
                ax_sc.scatter(x[mask], score_ord[mask], s=18, alpha=0.35)
            else:
                ax_sc.text(0.5, 0.5, "No matched units", ha="center", va="center", transform=ax_sc.transAxes)

            if labels_ord.size:
                for cid in np.unique(labels_ord):
                    idxs = np.where(labels_ord == cid)[0]
                    if idxs.size:
                        yvals = score_ord[idxs]
                        yvals = yvals[np.isfinite(yvals)]
                        if yvals.size:
                            ax_sc.hlines(float(yvals.mean()), idxs.min() - 0.5, idxs.max() + 0.5, linewidth=3.0)

            ax_sc.set_xlim(-0.5, len(score_ord) - 0.5)

        alive = int(layer_data[layer_idx][m]["alive_total_idx"].size)
        tot = int(layer_data[layer_idx][m]["acts_full"].shape[1])
        ax_sc.text(0.5, -0.155, f"(alive {alive}/{tot})", transform=ax_sc.transAxes, ha="center", va="top", fontsize=10)

    fig.suptitle(f"{m.upper()} — {os.path.basename(model_paths[m])}", fontsize=15, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.94, bottom=0.06)
    _maybe_save_or_show(fig, out_path, show)


# ============================================================
# ===== OLD-STYLE NETWORK SKETCH (ONLY SKETCHING CODE; DEAD REMOVED) ====
# ============================================================

def _x_positions_from_order(order: np.ndarray, n: int) -> np.ndarray:
    order = np.asarray(order, dtype=int)
    if order.size != n:
        raise ValueError(f"order size {order.size} != n {n}")
    inv = np.empty(n, dtype=int)
    inv[order] = np.arange(n, dtype=int)
    grid = np.linspace(0.0, 1.0, n)
    return grid[inv]


def _extract_weight_numpy(sd, wkey: str) -> np.ndarray:
    W = sd[wkey]
    if isinstance(W, torch.Tensor):
        W = W.detach().cpu().numpy()
    return np.asarray(W, dtype=np.float32)


def _draw_segment_black_weight(
    ax,
    mat,
    src_x,
    dst_x,
    y_src,
    y_dst,
    pct: float,
    alpha: float,
    lw_scale: float,
    lw_min: float = 0.25,
):
    if mat.size == 0:
        return

    mags = np.abs(mat).ravel()
    thr = float(np.percentile(mags, pct)) if mags.size else 0.0
    src_idx, dst_idx = np.where(np.abs(mat) >= thr)

    if src_idx.size == 0:
        flat = np.abs(mat).ravel()
        k = min(SKETCH_MAX_EDGES_PER_SEG, flat.size)
        if k <= 0:
            return
        top = np.argpartition(-flat, k - 1)[:k]
        src_idx, dst_idx = np.unravel_index(top, mat.shape)
    else:
        if src_idx.size > SKETCH_MAX_EDGES_PER_SEG:
            vals = np.abs(mat[src_idx, dst_idx])
            k = SKETCH_MAX_EDGES_PER_SEG
            top = np.argpartition(-vals, k - 1)[:k]
            src_idx = src_idx[top]
            dst_idx = dst_idx[top]

    vals = np.abs(mat[src_idx, dst_idx]).astype(np.float32)
    ord_draw = np.argsort(vals)
    src_idx = src_idx[ord_draw]
    dst_idx = dst_idx[ord_draw]
    vals = vals[ord_draw]

    vmin = float(np.min(vals)) if vals.size else 0.0
    vmax = float(np.max(vals)) if vals.size else 1.0
    denom = (vmax - vmin) + 1e-12

    for si, di, mag in zip(src_idx, dst_idx, vals):
        mag_n = float((mag - vmin) / denom)
        lw = min(lw_min + (mag_n**1.5) * lw_scale, SKETCH_MAX_LW)
        ax.plot(
            [src_x[si], dst_x[di]],
            [y_src, y_dst],
            color=(0.0, 0.0, 0.0, 1.0),
            lw=lw,
            alpha=float(alpha),
        )


def plot_network_sketch_dead_removed(
    model_order,
    actor_layers_by_model,
    state_dicts,
    layer_data,
    cluster_vis,
    out_path: Path | None,
    show: bool,
):
    for m in model_order:
        if len(actor_layers_by_model[m]) < 3:
            raise RuntimeError(f"[{m}] Need >=3 actor Linear layers for sketch. Found {len(actor_layers_by_model[m])}.")

    num_models = len(model_order)

    fig, axes = plt.subplots(1, num_models, figsize=(5.0 * num_models, 6.5), sharex=False, sharey=False)
    axes = np.atleast_1d(axes)

    NODE_S = max(12, int(SKETCH_CIRCLE_SIZE * 0.55))

    percs = SKETCH_THRESHOLD_PERCENT_BY_SEG
    alphas = SKETCH_EDGE_ALPHA_BY_SEG
    lwsc = SKETCH_LW_SCALE_BY_SEG
    y_levels = {"in": 3.0, "h1": 2.0, "h2": 1.0, "out": 0.0}

    for j, m in enumerate(model_order):
        ax = axes[j]

        layers = actor_layers_by_model[m]
        sd = state_dicts[m]

        _, w0, _ = layers[0]
        _, w1, _ = layers[1]
        _, w2, _ = layers[2]

        W_in_full = _extract_weight_numpy(sd, w0)
        W_mid_full = _extract_weight_numpy(sd, w1)
        W_out_full = _extract_weight_numpy(sd, w2)

        alive_h1_total = layer_data[0][m]["alive_total_idx"]
        alive_h2_total = layer_data[1][m]["alive_total_idx"]

        W_in = W_in_full[alive_h1_total, :]
        W_mid = W_mid_full[alive_h2_total, :][:, alive_h1_total]

        if W_out_full.shape[0] < OUTPUT_OFFSET + OUTPUT_MOTORS:
            raise RuntimeError(
                f"[{m}] Output layer has {W_out_full.shape[0]} rows, cannot slice motors "
                f"[{OUTPUT_OFFSET}:{OUTPUT_OFFSET + OUTPUT_MOTORS}]."
            )
        W_out = W_out_full[OUTPUT_OFFSET : OUTPUT_OFFSET + OUTPUT_MOTORS, :][:, alive_h2_total]

        n_h1, n_in = W_in.shape
        n_h2, n_h1b = W_mid.shape
        n_out, n_h2b = W_out.shape
        if n_h1b != n_h1 or n_h2b != n_h2:
            raise RuntimeError(f"[{m}] weight shape mismatch after dead-removal.")

        x_in = np.linspace(0.0, 1.0, n_in)
        x_out = np.linspace(0.0, 1.0, n_out)

        order_h1 = layer_data[0][m]["order"]
        order_h2 = layer_data[1][m]["order"]
        if order_h1.size != n_h1 or order_h2.size != n_h2:
            raise RuntimeError(f"[{m}] order size mismatch after dead-removal (did alive sets change?)")

        x_h1 = _x_positions_from_order(order_h1, n_h1)
        x_h2 = _x_positions_from_order(order_h2, n_h2)

        cmap0, norm0 = cluster_vis[0]["cmap"], cluster_vis[0]["norm"]
        cmap1, norm1 = cluster_vis[1]["cmap"], cluster_vis[1]["norm"]
        labels_h1 = layer_data[0][m]["labels"]
        labels_h2 = layer_data[1][m]["labels"]

        cols0 = cmap0(norm0(labels_h1)) if labels_h1.size == n_h1 else np.tile(np.array([[0.8, 0.8, 0.8, 1.0]]), (n_h1, 1))
        cols1 = cmap1(norm1(labels_h2)) if labels_h2.size == n_h2 else np.tile(np.array([[0.8, 0.8, 0.8, 1.0]]), (n_h2, 1))

        # transpose to match your original draw call shapes
        M_in = W_in.T
        M_mid = W_mid.T
        M_out = W_out.T

        _draw_segment_black_weight(ax, M_in, x_in, x_h1, y_levels["in"], y_levels["h1"], percs[0], alphas[0], lwsc[0], lw_min=0.25)
        _draw_segment_black_weight(ax, M_mid, x_h1, x_h2, y_levels["h1"], y_levels["h2"], percs[1], alphas[1], lwsc[1], lw_min=0.25)
        _draw_segment_black_weight(ax, M_out, x_h2, x_out, y_levels["h2"], y_levels["out"], percs[2], alphas[2], lwsc[2], lw_min=0.25)

        ax.scatter(x_in, np.full(n_in, y_levels["in"]), s=NODE_S, edgecolor="black", facecolor="white", zorder=3)
        ax.scatter(x_h1, np.full(n_h1, y_levels["h1"]), s=NODE_S, color=cols0, edgecolor="black", zorder=3)
        ax.scatter(x_h2, np.full(n_h2, y_levels["h2"]), s=NODE_S, color=cols1, edgecolor="black", zorder=3)
        ax.scatter(x_out, np.full(n_out, y_levels["out"]), s=NODE_S, edgecolor="black", facecolor="white", zorder=3)

        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.4, 3.4)
        ax.set_title(f"{m}\n(dead removed)\n(output motors: {OUTPUT_MOTORS})", fontsize=11)
        ax.axis("off")

    fig.suptitle(
        "Network sketch (OLD formatting) — edges BLACK; thickness=|w|; h1/h2 ordered by block-diag; dead neurons removed",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    _maybe_save_or_show(fig, out_path, show)


# ============================================================
# ============================== MAIN ========================
# ============================================================

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-walk", type=str, default=DEFAULT_MODEL1_PATH)
    p.add_argument("--model-spin", type=str, default=DEFAULT_MODEL2_PATH)
    p.add_argument("--model-jump", type=str, default=DEFAULT_MODEL3_PATH)

    p.add_argument("--all-states", type=str, default=None, help="Path to ALL_states_concat.npy (overrides repo-root default).")
    p.add_argument("--n-all-states", type=int, default=DEFAULT_N_ALL_STATES)
    p.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYER_INDICES)

    p.add_argument("--activation", type=str, default=DEFAULT_ACTIVATION, choices=["elu", "relu", "tanh"])
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)

    p.add_argument("--out-dir", type=str, default=None, help="If set (or headless), save figures here.")
    p.add_argument("--no-show", action="store_true", help="Do not plt.show() even if not headless.")
    return p.parse_args()


def main():
    args = _parse_args()

    # Resolve REPO_ROOT as in your notebook cell
    REPO_ROOT = Path.cwd().resolve()
    if REPO_ROOT.name == "AnalysisScripts":
        REPO_ROOT = REPO_ROOT.parent

    all_states_path = args.all_states
    if all_states_path is None:
        all_states_path = str(REPO_ROOT / DEFAULT_ALL_STATES_REL)

    MODEL_PATHS = {
        "walk": args.model_walk,
        "spin": args.model_spin,  # reference
        "jump": args.model_jump,
    }
    MODEL_ORDER = ["walk", "spin", "jump"]
    REF_MODEL = "spin"
    LAYER_INDICES = list(args.layers)

    # Styling (same spirit as notebook; Arial may not exist everywhere)
    sns.set_theme(style="white", font="Arial")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]

    _seed_everything(args.seed)

    # Out dir / show logic
    show = (not args.no_show) and (not HEADLESS)
    out_dir = None
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    elif HEADLESS:
        out_dir = Path("./_module_explorer_figs")
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Validate paths
    for pth in list(MODEL_PATHS.values()) + [all_states_path]:
        if not os.path.exists(pth):
            raise FileNotFoundError(f"Missing path: {pth}")

    # ============================================================
    # ===================== LOAD STATES + MODELS =================
    # ============================================================
    all_states = np.load(all_states_path)
    if all_states.ndim != 2:
        raise ValueError(f"{all_states_path} has shape {all_states.shape}, expected 2D (*, obs_dim)")

    state_dicts = {}
    actor_layers_by_model = {}

    for name in MODEL_ORDER:
        sd = _load_rlg_forplay_state_dict(MODEL_PATHS[name])
        layers = _discover_actor_mlp_layers(sd)
        if not layers:
            raise RuntimeError(f"[{name}] no actor layers discovered.")
        state_dicts[name] = sd
        actor_layers_by_model[name] = layers
        print(f"[model] {name}: {MODEL_PATHS[name]} — {len(layers)} actor layer(s)")

    # Infer obs_dim from reference
    obs_dim, _ = _first_linear_in(actor_layers_by_model[REF_MODEL], state_dicts[REF_MODEL])
    if all_states.shape[1] != obs_dim:
        raise ValueError(f"ALL_STATES obs_dim={all_states.shape[1]}, expected {obs_dim}")

    # Sample states
    n_avail = all_states.shape[0]
    n_pick = int(args.n_all_states)
    idx = np.random.choice(n_avail, size=n_pick, replace=(n_avail < n_pick))
    X = all_states[idx].astype(np.float32, copy=False)
    ref_inputs = torch.as_tensor(X, dtype=torch.float32, device="cpu")

    # ============================================================
    # ============ PER-LAYER: ACTIVATIONS, CLUSTERS, FAMILIES =====
    # ============================================================
    layer_data = {layer_idx: {} for layer_idx in LAYER_INDICES}
    max_layer_idx = max(LAYER_INDICES) if len(LAYER_INDICES) else 0

    for layer_idx in LAYER_INDICES:
        for m in MODEL_ORDER:
            acts_full = get_layer_output_batch_from_ckpt(
                state_dicts[m],
                ref_inputs,
                layer_idx,
                actor_layers_by_model[m],
                args.activation,
                max_layer_idx=max_layer_idx,
            )
            dead_idx, alive_idx = _dead_alive_indices(acts_full, MIN_STD)

            acts_alive = acts_full[:, alive_idx] if alive_idx.size else acts_full[:, :0]
            A_z = _zscore_cols(acts_alive, eps=EPS) if acts_alive.size else acts_alive.astype(np.float32, copy=False)

            R_signed = cosine_sim_matrix_cols(A_z, eps=EPS)
            R_abs = np.abs(R_signed)

            K_best, order, labels = choose_K_and_order_abs(R_abs, k_min=K_MIN, k_max=K_MAX, alpha_small_k=ALPHA_SMALL_K)
            labels = _relabel_clusters_by_size(labels)
            order = _order_by_cluster_size(labels, base_order=order)

            layer_data[layer_idx][m] = dict(
                acts_full=acts_full,
                alive_total_idx=alive_idx.astype(int),
                dead_total_idx=dead_idx.astype(int),
                A_z=A_z,
                R_signed=R_signed,
                labels=labels.astype(int),
                order=order.astype(int),
                K_best=int(K_best),
            )

            print(f"[layer {layer_idx+1}] {m}: alive={alive_idx.size}/{acts_full.shape[1]} dead={dead_idx.size} K={K_best}")

    # Build shared cluster colormaps per layer (consistent across 3 models)
    cluster_vis = {}
    for layer_idx in LAYER_INDICES:
        all_labels = []
        for m in MODEL_ORDER:
            lab = layer_data[layer_idx][m]["labels"]
            if lab.size:
                all_labels.append(lab)
        if all_labels:
            max_cluster = int(np.max(np.concatenate(all_labels)))
        else:
            max_cluster = 1
        cmap = plt.get_cmap("tab20", max_cluster)
        norm = BoundaryNorm(np.arange(0.5, max_cluster + 1.5, 1.0), max_cluster)
        cluster_vis[layer_idx] = dict(cmap=cmap, norm=norm, max_cluster=max_cluster)

    # Compute families + stability with REF_MODEL as reference (per layer)
    layer_stability = {}
    for layer_idx in LAYER_INDICES:
        A_ref = layer_data[layer_idx][REF_MODEL]["A_z"]
        n_ref = int(A_ref.shape[1])
        if n_ref == 0:
            layer_stability[layer_idx] = None
            continue

        others = [m for m in MODEL_ORDER if m != REF_MODEL]
        o1, o2 = others[0], others[1]

        _, i1, j1, _ = hungarian_match_cosine(A_ref, layer_data[layer_idx][o1]["A_z"], eps=EPS)
        _, i2, j2, _ = hungarian_match_cosine(A_ref, layer_data[layer_idx][o2]["A_z"], eps=EPS)

        map1 = np.full(n_ref, -1, dtype=int)
        map2 = np.full(n_ref, -1, dtype=int)
        if i1.size:
            map1[i1] = j1
        if i2.size:
            map2[i2] = j2

        good_ref = np.where((map1 >= 0) & (map2 >= 0))[0].astype(int)
        if good_ref.size == 0:
            layer_stability[layer_idx] = None
            continue

        fam_idx_by_model = {
            REF_MODEL: good_ref,
            o1: map1[good_ref].astype(int),
            o2: map2[good_ref].astype(int),
        }

        W_ref = A_ref[:, fam_idx_by_model[REF_MODEL]]
        W_1 = layer_data[layer_idx][o1]["A_z"][:, fam_idx_by_model[o1]]
        W_2 = layer_data[layer_idx][o2]["A_z"][:, fam_idx_by_model[o2]]

        R_ref = cosine_sim_matrix_cols(W_ref, eps=EPS)
        np.fill_diagonal(R_ref, 0.0)
        R_1 = cosine_sim_matrix_cols(W_1, eps=EPS)
        np.fill_diagonal(R_1, 0.0)
        R_2 = cosine_sim_matrix_cols(W_2, eps=EPS)
        np.fill_diagonal(R_2, 0.0)

        n_fam = int(good_ref.size)
        act_21 = np.zeros(n_fam, dtype=np.float32)
        conn_21 = np.zeros(n_fam, dtype=np.float32)
        act_23 = np.zeros(n_fam, dtype=np.float32)
        conn_23 = np.zeros(n_fam, dtype=np.float32)

        for k in range(n_fam):
            act_21[k] = _safe_cos_abs(W_ref[:, k], W_1[:, k], eps=EPS)
            act_23[k] = _safe_cos_abs(W_ref[:, k], W_2[:, k], eps=EPS)
            conn_21[k] = _safe_cos_abs(R_ref[k, :], R_1[k, :], eps=EPS)
            conn_23[k] = _safe_cos_abs(R_ref[k, :], R_2[k, :], eps=EPS)

        act_avg = 0.5 * (act_21 + act_23)
        conn_avg = 0.5 * (conn_21 + conn_23)

        score_ref = 0.5 * (act_avg + conn_avg)
        score_o1 = 0.5 * (act_21 + conn_21)
        score_o2 = 0.5 * (act_23 + conn_23)

        labels_ref = layer_data[layer_idx][REF_MODEL]["labels"]
        cluster_ids = labels_ref[fam_idx_by_model[REF_MODEL]]

        layer_stability[layer_idx] = dict(
            fam_idx_by_model=fam_idx_by_model,
            act_21=act_21,
            conn_21=conn_21,
            act_23=act_23,
            conn_23=conn_23,
            act_avg=act_avg,
            conn_avg=conn_avg,
            score_by_model={REF_MODEL: score_ref, o1: score_o1, o2: score_o2},
            cluster_ids_ref=cluster_ids.astype(int),
            others=(o1, o2),
        )

    # ============================================================
    # ===================== PLOTS ===============================
    # ============================================================
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_stability = (out_dir / f"stability_6graphs_{ts}.png") if out_dir is not None else None
    _plot_stability_6graphs(LAYER_INDICES, layer_stability, cluster_vis, out_stability, show=show)

    for m in MODEL_ORDER:
        out_hm = (out_dir / f"{m}_heatmaps_{ts}.png") if out_dir is not None else None
        _plot_one_model_heatmaps_inline(
            m=m,
            model_paths=MODEL_PATHS,
            layer_indices=LAYER_INDICES,
            layer_data=layer_data,
            layer_stability=layer_stability,
            cluster_vis=cluster_vis,
            out_path=out_hm,
            show=show,
        )

    out_sketch = (out_dir / f"network_sketch_dead_removed_{ts}.png") if out_dir is not None else None
    plot_network_sketch_dead_removed(
        model_order=MODEL_ORDER,
        actor_layers_by_model=actor_layers_by_model,
        state_dicts=state_dicts,
        layer_data=layer_data,
        cluster_vis=cluster_vis,
        out_path=out_sketch,
        show=show,
    )

    if out_dir is not None:
        print(f"[done] figures in: {out_dir.resolve()}")
    else:
        print("[done]")


if __name__ == "__main__":
    main()
