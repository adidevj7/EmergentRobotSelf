#!/usr/bin/env python3
# === Ant Actor (triplet) — Cosine module explorer + center-reference stability + uncolored-style plots ===
# CLI:
#   python maps_triplets.py --run_dir /path/to/RUNDIR
# Expects:
#   RUNDIR/models contains *_for_play.pth
# Writes:
#   RUNDIR/MAPS_plots/*.png
#
# Center-only saving:
#   For each center checkpoint i (excluding edges), compute with (i-1, i, i+1) but SAVE ONLY the center plot.

from __future__ import annotations

import os, re, argparse, gc
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import Normalize
from scipy.optimize import linear_sum_assignment

# NEW (required for tau-based block diagonalisation)
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee, connected_components


# ============================================================
# ========================= CONFIG ===========================
# ============================================================

# --- Layers to analyze (hidden layers) ---
LAYER_INDICES = [0, 1]   # h1=0, h2=1

# --- Similarity / normalization config ---
SEED        = 42
EPS         = 1e-8
MIN_STD     = 1e-5
ACTIVATION  = "relu"     # {"elu","relu","tanh"}

# --- NEW: tau-based block diagonalisation config ---
TAU = 0.70
BD_MIN_BLOCK_SIZE = 1

# --- Heatmap visuals ---
CMAP = "RdBu_r"   # -1 blue, +1 red

# --- UNCOLORED: SINGLE shared colorbar for correlation heatmaps (editable) ---
# All are in FIGURE FRACTION coordinates (0..1).
UNCOLORED_CBAR_X_RIGHT   = 0.035
UNCOLORED_CBAR_W         = 0.015
UNCOLORED_CBAR_Y_BOTTOM  = 0.325
UNCOLORED_CBAR_H         = 0.60
UNCOLORED_CBAR_TICK_FS   = 11
UNCOLORED_CBAR_LABEL     = ""
UNCOLORED_CBAR_LABEL_FS  = 12
UNCOLORED_CBAR_LABELPAD  = 10


# ============================================================
# ========================= SETUP ============================
# ============================================================

sns.set_theme(style="white", font="Arial")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]


def _seed_everything(seed: int):
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _infer_behavior_from_path(p: str) -> str | None:
    s = os.path.basename(p).lower()
    if re.search(r"(^|[^a-z])walk([^a-z]|$)", s) or "_walk_" in s:
        return "walk"
    if re.search(r"(^|[^a-z])(spin|wiggle)([^a-z]|$)", s) or "_spin_" in s or "_wiggle_" in s:
        return "spin"
    if re.search(r"(^|[^a-z])(jump|bob)([^a-z]|$)", s) or "_jump_" in s or "_bob_" in s:
        return "jump"
    return None


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
    return W.shape[1], W.shape[0]


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


def get_layer_output_batch_from_ckpt(sd, inputs: torch.Tensor, layer_idx: int, actor_layers, activation_name):
    X = inputs.detach().cpu().numpy().astype(np.float32, copy=False)
    outs = _actor_hidden_forward_numpy(X, actor_layers, sd, activation_name, max_layers=max(LAYER_INDICES) + 1)
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


def blockdiag_rcm_blocks_from_abs(R_abs: np.ndarray, tau: float, min_block_size: int):
    R_abs = np.asarray(R_abs, dtype=np.float32)
    n = int(R_abs.shape[0])
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    if n < 3:
        return np.arange(n, dtype=int), np.ones(n, dtype=int)

    R_abs = 0.5 * (R_abs + R_abs.T)
    R_abs = np.clip(R_abs, 0.0, 1.0)
    np.fill_diagonal(R_abs, 0.0)

    tau = float(max(0.0, min(1.0, tau)))
    A = (R_abs >= tau).astype(np.int8)
    np.fill_diagonal(A, 0)
    G = csr_matrix(A)

    n_comp, comp0 = connected_components(G, directed=False, connection="weak")
    comp = comp0.copy()

    if min_block_size is not None and int(min_block_size) > 1:
        sizes = np.bincount(comp, minlength=int(n_comp))
        small = np.where(sizes < int(min_block_size))[0]
        if small.size:
            comp[np.isin(comp, small)] = -1
            keep = [c for c in np.unique(comp) if c != -1]
            remap = {c: i for i, c in enumerate(keep)}
            comp2 = np.empty(n, dtype=int)
            for i in range(n):
                c = comp[i]
                comp2[i] = remap[c] if c != -1 else len(keep)
            comp = comp2
            n_comp = int(comp.max()) + 1

    sizes = [(c, int(np.sum(comp == c))) for c in range(int(n_comp))]
    sizes.sort(key=lambda t: -t[1])

    parts = []
    for c, _ in sizes:
        idxs = np.where(comp == c)[0]
        if idxs.size <= 2:
            parts.append(idxs)
        else:
            subG = G[idxs][:, idxs]
            sub_ord = reverse_cuthill_mckee(subG, symmetric_mode=True)
            parts.append(idxs[np.asarray(sub_ord, dtype=int)])

    order = np.concatenate(parts).astype(int)
    labels = (comp + 1).astype(int)
    return order, labels


def _add_shared_cbar_left_of_heatmaps(fig, ax_hm_anchor):
    bbox = ax_hm_anchor.get_position()

    if UNCOLORED_CBAR_Y_BOTTOM is None or UNCOLORED_CBAR_H is None:
        y0 = float(bbox.y0)
        h  = float(bbox.y1 - bbox.y0)
    else:
        y0 = float(UNCOLORED_CBAR_Y_BOTTOM)
        h  = float(UNCOLORED_CBAR_H)

    x_right = float(UNCOLORED_CBAR_X_RIGHT)
    w = float(UNCOLORED_CBAR_W)
    x0 = x_right - w

    cax = fig.add_axes([x0, y0, w, h])

    norm = Normalize(vmin=-1.0, vmax=1.0)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(CMAP))
    sm.set_array([])

    cb = fig.colorbar(sm, cax=cax, orientation="vertical", ticks=[-1.0, 0.0, 1.0])
    cb.ax.tick_params(labelsize=int(UNCOLORED_CBAR_TICK_FS))
    cb.set_label(UNCOLORED_CBAR_LABEL, fontsize=int(UNCOLORED_CBAR_LABEL_FS), labelpad=float(UNCOLORED_CBAR_LABELPAD))
    return cb


def _plot_one_model_heatmaps(fig_title: str, out_path: Path, layer_data, layer_stability, model_key: str):
    fig = plt.figure(figsize=(15.5, 10.2))
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.0, 0.36],
        wspace=0.10, hspace=0.10
    )

    ax_hm0  = fig.add_subplot(gs[0, 0])
    ax_hm1  = fig.add_subplot(gs[0, 1])
    ax_sc0  = fig.add_subplot(gs[1, 0])
    ax_sc1  = fig.add_subplot(gs[1, 1])

    SEP_LW = 2.6
    SEP_ALPHA = 0.95

    for col, layer_idx in enumerate(LAYER_INDICES):
        d = layer_data[layer_idx][model_key]
        labels = d["labels"]
        order  = d["order"]
        R      = d["R_signed"]
        R_plot = R[np.ix_(order, order)] if R.size else R
        labels_ord = labels[order] if labels.size else labels

        ax_hm  = ax_hm0  if col == 0 else ax_hm1
        ax_sc  = ax_sc0  if col == 0 else ax_sc1

        if R_plot.size:
            sns.heatmap(
                R_plot, ax=ax_hm,
                cmap=CMAP, vmin=-1.0, vmax=1.0, center=0.0,
                square=False, cbar=False,
                xticklabels=False, yticklabels=False
            )
        else:
            ax_hm.text(0.5, 0.5, "Empty", ha="center", va="center")
            ax_hm.set_xticks([]); ax_hm.set_yticks([])

        ax_hm.set_title("")

        st = layer_stability[layer_idx]
        ax_sc.set_ylim(0.0, 1.05)
        ax_sc.axhline(1.0, ls="--", lw=1.0, alpha=0.25)
        ax_sc.set_xticks([])
        ax_sc.set_xlabel(f"Layer {layer_idx+1}", fontsize=18, fontweight="bold", labelpad=6)
        if col == 0:
            ax_sc.set_ylabel("Persistence score", fontsize=15, fontweight="bold")
        else:
            ax_sc.set_ylabel("")
            ax_sc.set_yticks([])
            ax_sc.set_yticklabels([])

        if st is None:
            ax_sc.text(0.5, 0.5, "No families", ha="center", va="center", transform=ax_sc.transAxes)
            continue

        scores = st["score_by_model"][model_key]
        fam_idx = st["fam_idx_by_model"][model_key]

        n_alive = int(layer_data[layer_idx][model_key]["A_z"].shape[1])
        score_full = np.full(n_alive, np.nan, dtype=np.float32)
        score_full[fam_idx] = scores

        score_ord = score_full[order]
        x = np.arange(score_ord.size)
        mask = np.isfinite(score_ord)

        if np.any(mask):
            ax_sc.scatter(x[mask], score_ord[mask], s=18, alpha=0.35, color=(0.35, 0.35, 0.35, 1.0))
        else:
            ax_sc.text(0.5, 0.5, "No matched units", ha="center", va="center", transform=ax_sc.transAxes)

        if labels_ord.size:
            idx_self = np.where(labels_ord == 1)[0]
            idx_task = np.where(labels_ord != 1)[0]

            # ONLY the self-vs-rest separator
            if idx_self.size:
                boundary = float(idx_self.max() + 0.5)
                if R_plot.size:
                    ax_hm.axvline(boundary, color="k", lw=SEP_LW, ls="--", alpha=SEP_ALPHA, zorder=6)
                ax_sc.axvline(boundary, color="k", lw=SEP_LW, ls="--", alpha=SEP_ALPHA, zorder=6)

            # bold mean lines
            if idx_self.size:
                y_self = score_ord[idx_self]
                y_self = y_self[np.isfinite(y_self)]
                if y_self.size:
                    yv = float(y_self.mean())
                    x0, x1 = int(idx_self.min()), int(idx_self.max())
                    ax_sc.hlines(yv, x0 - 0.5, x1 + 0.5, linewidth=4.0, color="k", alpha=0.95, zorder=5)

            if idx_task.size:
                y_task = score_ord[idx_task]
                y_task = y_task[np.isfinite(y_task)]
                if y_task.size:
                    yv = float(y_task.mean())
                    x0, x1 = int(idx_task.min()), int(idx_task.max())
                    ax_sc.hlines(yv, x0 - 0.5, x1 + 0.5, linewidth=4.0, color="red", alpha=0.95, zorder=5)

        ax_sc.set_xlim(-0.5, len(score_ord) - 0.5)

    _add_shared_cbar_left_of_heatmaps(fig, ax_hm0)
    fig.suptitle(fig_title, fontsize=20, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.06)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# ============ DIRECTORY PARSING + NEIGHBOR PICKING ==========
# ============================================================

_CKPT_RE = re.compile(r"(?:^|/)(c(\d+)_b(\d+)_.*?_for_play\.pth)$")


def _parse_cycle_behavior(p: str):
    m = _CKPT_RE.search(p)
    if not m:
        return None
    fname = m.group(1)
    c = int(m.group(2))
    b = int(m.group(3))
    return (c, b, fname)


def _discover_for_play_ckpts(models_dir: Path):
    paths = []
    for p in sorted(models_dir.glob("*_for_play.pth")):
        key = _parse_cycle_behavior(str(p))
        if key is None:
            continue
        c, b, fname = key
        paths.append((c, b, fname, p))
    paths.sort(key=lambda t: (t[0], t[1], t[2]))
    return paths


# ============================================================
# ============================ MAIN ==========================
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="RUNDIR containing a 'models' folder.")
    ap.add_argument("--states_path", type=str, default="", help="Path to ALL_states_concat.npy (optional).")
    ap.add_argument("--n_states", type=int, default=400_000, help="How many states to sample.")
    ap.add_argument("--tau", type=float, default=TAU, help="Tau for block diagonalization.")
    ap.add_argument("--min_block_size", type=int, default=BD_MIN_BLOCK_SIZE, help="Min block size.")
    ap.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    args = ap.parse_args()

    _seed_everything(args.seed)

    run_dir = Path(args.run_dir).expanduser().resolve()
    models_dir = run_dir / "models"
    if not models_dir.exists():
        raise FileNotFoundError(f"Missing models dir: {models_dir}")

    plots_dir = run_dir / "MAPS_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Default states path mirrors your notebook convention (relative to repo root)
    if args.states_path.strip():
        states_path = Path(args.states_path).expanduser().resolve()
    else:
        repo_root = Path.cwd().resolve()
        if repo_root.name == "AnalysisScripts":
            repo_root = repo_root.parent
        states_path = repo_root / "Checkpoints_States_selectedGraphs" / "StatesConcat" / "ALL_states_concat.npy"

    if not states_path.exists():
        raise FileNotFoundError(f"Missing states file: {states_path}")

    ckpts = _discover_for_play_ckpts(models_dir)
    if len(ckpts) < 3:
        raise RuntimeError(f"Need at least 3 *_for_play.pth in {models_dir} to form centers. Found {len(ckpts)}.")

    # Memmap the states (RAM-safe)
    all_states = np.load(str(states_path), mmap_mode="r")
    if all_states.ndim != 2:
        raise ValueError(f"{states_path} has shape {all_states.shape}, expected 2D (*, obs_dim)")

    n_avail = int(all_states.shape[0])
    n_pick = int(args.n_states)

    # Fixed indices across entire run (reproducible + avoids repeated huge allocations)
    idx = np.random.choice(n_avail, size=n_pick, replace=(n_avail < n_pick))
    X = np.asarray(all_states[idx], dtype=np.float32)  # materialize sampled batch once
    ref_inputs = torch.as_tensor(X, dtype=torch.float32, device="cpu")

    print(f"[discover] {len(ckpts)} for-play ckpts in {models_dir}")
    print("[mode] Skipping edges (first/last) and saving CENTER ONLY for each valid center")

    # Centers are i=1..len-2 (skip edges)
    for i in range(1, len(ckpts) - 1):
        prev_p   = ckpts[i - 1][3]
        center_p = ckpts[i][3]
        next_p   = ckpts[i + 1][3]

        # Define ordering: prev, center (reference), next
        MODEL_PATHS = {
            "prev": str(prev_p),
            "center": str(center_p),
            "next": str(next_p),
        }
        MODEL_ORDER = ["prev", "center", "next"]
        REF_MODEL = "center"

        # Load 3 models (then discard after saving center plot)
        state_dicts = {}
        actor_layers_by_model = {}

        for name in MODEL_ORDER:
            sd = _load_rlg_forplay_state_dict(MODEL_PATHS[name])
            layers = _discover_actor_mlp_layers(sd)
            if not layers:
                raise RuntimeError(f"[{name}] no actor layers discovered.")
            state_dicts[name] = sd
            actor_layers_by_model[name] = layers

        obs_dim, _ = _first_linear_in(actor_layers_by_model[REF_MODEL], state_dicts[REF_MODEL])
        if int(all_states.shape[1]) != int(obs_dim):
            raise ValueError(f"ALL_STATES obs_dim={all_states.shape[1]}, expected {obs_dim}")

        # PER-LAYER compute
        layer_data = {layer_idx: {} for layer_idx in LAYER_INDICES}

        for layer_idx in LAYER_INDICES:
            for m in MODEL_ORDER:
                acts_full = get_layer_output_batch_from_ckpt(
                    state_dicts[m], ref_inputs, layer_idx, actor_layers_by_model[m], ACTIVATION
                )
                dead_idx, alive_idx = _dead_alive_indices(acts_full, MIN_STD)

                acts_alive = acts_full[:, alive_idx] if alive_idx.size else acts_full[:, :0]
                A_z = _zscore_cols(acts_alive, eps=EPS) if acts_alive.size else acts_alive.astype(np.float32, copy=False)

                R_signed = cosine_sim_matrix_cols(A_z, eps=EPS)
                R_abs = np.abs(R_signed)

                order0, labels0 = blockdiag_rcm_blocks_from_abs(
                    R_abs, tau=float(args.tau), min_block_size=int(args.min_block_size)
                )
                labels = _relabel_clusters_by_size(labels0)
                order  = _order_by_cluster_size(labels, base_order=order0)

                layer_data[layer_idx][m] = dict(
                    A_z=A_z,
                    R_signed=R_signed,
                    labels=labels.astype(int),
                    order=order.astype(int),
                    alive_total_idx=alive_idx.astype(int),
                    dead_total_idx=dead_idx.astype(int),
                )

        # Families + stability vs REF_MODEL (center)
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
            W_1   = layer_data[layer_idx][o1]["A_z"][:, fam_idx_by_model[o1]]
            W_2   = layer_data[layer_idx][o2]["A_z"][:, fam_idx_by_model[o2]]

            R_ref = cosine_sim_matrix_cols(W_ref, eps=EPS); np.fill_diagonal(R_ref, 0.0)
            R_1   = cosine_sim_matrix_cols(W_1,   eps=EPS); np.fill_diagonal(R_1,   0.0)
            R_2   = cosine_sim_matrix_cols(W_2,   eps=EPS); np.fill_diagonal(R_2,   0.0)

            n_fam = int(good_ref.size)
            act_21  = np.zeros(n_fam, dtype=np.float32)
            conn_21 = np.zeros(n_fam, dtype=np.float32)
            act_23  = np.zeros(n_fam, dtype=np.float32)
            conn_23 = np.zeros(n_fam, dtype=np.float32)

            for k in range(n_fam):
                act_21[k]  = _safe_cos_abs(W_ref[:, k], W_1[:, k], eps=EPS)
                act_23[k]  = _safe_cos_abs(W_ref[:, k], W_2[:, k], eps=EPS)
                conn_21[k] = _safe_cos_abs(R_ref[k, :], R_1[k, :], eps=EPS)
                conn_23[k] = _safe_cos_abs(R_ref[k, :], R_2[k, :], eps=EPS)

            act_avg  = 0.5 * (act_21  + act_23)
            conn_avg = 0.5 * (conn_21 + conn_23)

            score_ref = 0.5 * (act_avg + conn_avg)
            score_o1  = 0.5 * (act_21  + conn_21)
            score_o2  = 0.5 * (act_23  + conn_23)

            layer_stability[layer_idx] = dict(
                fam_idx_by_model=fam_idx_by_model,
                score_by_model={REF_MODEL: score_ref, o1: score_o1, o2: score_o2},
                others=(o1, o2),
            )

        # SAVE ONLY the center plot
        center_name = Path(center_p).name.replace(".pth", "")
        beh = _infer_behavior_from_path(str(center_p)) or "center"
        out_file = plots_dir / f"{center_name}__center_{beh}.png"
        fig_title = f"center: {Path(center_p).name}"
        _plot_one_model_heatmaps(fig_title, out_file, layer_data, layer_stability, "center")

        print(f"[save] center idx={i}/{len(ckpts)-1}: {out_file.name}")

        # RAM cleanup: drop big per-center tensors/arrays
        del state_dicts, actor_layers_by_model, layer_data, layer_stability
        gc.collect()

    print(f"[done] Saved center-only plots to: {plots_dir}")


if __name__ == "__main__":
    main()