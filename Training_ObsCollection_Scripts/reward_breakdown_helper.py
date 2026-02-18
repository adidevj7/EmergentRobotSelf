# reward_breakdown_helper.py
from __future__ import annotations
from collections import deque
from typing import Dict, List, Tuple
import torch

class RewardBreakdown:
    """
    Behavior-agnostic reward diagnostics for rl_games LOCAL vecenvs.

    API:
      - accumulate_step(components, kin):
          components: dict[str, Tensor(N,)]  (e.g., {"up_term": ..., "jerk_pen": ...})
          kin       : dict[str, Tensor(N,)]  (e.g., {"vx_b": ..., "vz_fd": ..., "vxy_fd": ...})
      - finish_episodes(done_mask): push per-episode totals/means into rolling windows
      - window_means(): returns (comp_fields, kin_fields, comp_sum)
          comp_fields: {"c_<name>": window_mean}
          kin_fields : {"mean_<key>_win": window_mean}
          comp_sum   : sum of component means (sanity vs avg_return_window)

    Notes:
      - Component names are dynamic; their CSV keys are prefixed with "c_".
      - Kinematics are logged as "mean_<key>_win".
    """
    def __init__(self, num_envs: int, device: torch.device, window_size: int):
        self.N = int(num_envs)
        self.dev = device
        self.window_size = int(window_size)

        # Per-episode accumulators (size N)
        self._comp_acc: Dict[str, torch.Tensor] = {}
        self._kin_sum: Dict[str, torch.Tensor] = {}
        self._kin_count = torch.zeros(self.N, dtype=torch.int32, device=self.dev)

        # Episode windows (host-side deques of floats)
        self._comp_hist: Dict[str, deque] = {}
        self._kin_hist: Dict[str, deque] = {}

        # Stable display order
        self._component_keys: List[str] = []
        self._kin_keys: List[str] = []

        self.header_frozen = False  # your caller can freeze once header is written

    def _ensure_comp_key(self, key: str):
        if key not in self._comp_acc:
            self._comp_acc[key] = torch.zeros(self.N, dtype=torch.float32, device=self.dev)
            self._comp_hist.setdefault(key, deque(maxlen=self.window_size))
            if key not in self._component_keys:
                self._component_keys.append(key)

    def _ensure_kin_key(self, key: str):
        if key not in self._kin_sum:
            self._kin_sum[key] = torch.zeros(self.N, dtype=torch.float32, device=self.dev)
            self._kin_hist.setdefault(key, deque(maxlen=self.window_size))
            if key not in self._kin_keys:
                self._kin_keys.append(key)

    @torch.no_grad()
    def accumulate_step(self, components: Dict[str, torch.Tensor], kin: Dict[str, torch.Tensor]):
        for k, v in components.items():
            if v is None: continue
            self._ensure_comp_key(k)
            self._comp_acc[k] += v.detach()

        have_kin = False
        for k, v in kin.items():
            if v is None: continue
            self._ensure_kin_key(k)
            self._kin_sum[k] += v.detach()
            have_kin = True
        if have_kin:
            self._kin_count += 1

    @torch.no_grad()
    def finish_episodes(self, done_mask: torch.Tensor):
        if not torch.any(done_mask):
            return
        idx = torch.where(done_mask)[0]

        # components → totals
        for k in self._component_keys:
            vals = self._comp_acc[k][idx]
            for f in vals.tolist():
                self._comp_hist[k].append(float(f))
            self._comp_acc[k][idx] = 0.0

        # kinematics → per-episode means
        cnt = torch.clamp(self._kin_count[idx].to(torch.float32), min=1.0)
        for k in self._kin_keys:
            means = self._kin_sum[k][idx] / cnt
            for f in means.tolist():
                self._kin_hist[k].append(float(f))
            self._kin_sum[k][idx] = 0.0
        self._kin_count[idx] = 0

    def window_means(self) -> Tuple[Dict[str, float], Dict[str, float], float]:
        comp_fields: Dict[str, float] = {}
        kin_fields: Dict[str, float] = {}
        comp_sum = 0.0

        for k in self._component_keys:
            dq = self._comp_hist.get(k)
            m = float(sum(dq) / len(dq)) if dq and len(dq) > 0 else 0.0
            comp_fields[f"c_{k}"] = m
            comp_sum += m

        for k in self._kin_keys:
            dq = self._kin_hist.get(k)
            m = float(sum(dq) / len(dq)) if dq and len(dq) > 0 else 0.0
            kin_fields[f"mean_{k}_win"] = m

        return comp_fields, kin_fields, comp_sum

    def csv_suffix_columns(self) -> List[str]:
        cols: List[str] = []
        # kinematics first (in discovered order)
        for k in self._kin_keys:
            cols.append(f"mean_{k}_win")
        # components next
        for k in self._component_keys:
            cols.append(f"c_{k}")
        # sums last
        cols += ["c_components_sum", "c_components_delta"]
        return cols

    def freeze_header(self):
        self.header_frozen = True
