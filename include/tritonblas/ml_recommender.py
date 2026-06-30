"""Optional ML re-ranking of Origami's candidate tile configs.

This is *inference only*. A model bundle (one small two-tower scorer per
categorical cell of GEMM-shape space) is trained offline; at runtime, when
enabled, this module re-ranks the LDS-legal tile candidates Origami already
enumerated and returns the top-1. If anything is missing or fails, callers
fall back silently to Origami's analytical pick.

Enable with:
    TRITONBLAS_ML_RECOMMENDER=1
    TRITONBLAS_ML_BUNDLE=/path/to/models.pt

The training pipeline that produces the bundle lives out-of-tree (it is
benchmark/analysis tooling, not library code). The bundle's ``meta`` records
the feature dimensions; this module asserts they match before scoring, so a
stale bundle degrades to fallback rather than producing garbage.
"""
from __future__ import annotations

import os
from math import ceil, log2
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn

    _TORCH_OK = True
except ImportError:  # pragma: no cover
    _TORCH_OK = False

DTYPE_BYTES = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "tf32": 4.0,
               "f8": 1.0, "i8": 1.0, "f4": 0.5}

_MI_DIMS = {
    ("gfx950", 32): (16, 16, 4), ("gfx950", 16): (16, 16, 32),
    ("gfx950", 8): (16, 16, 128), ("gfx950", 4): (16, 16, 128),
    ("gfx942", 32): (16, 16, 4), ("gfx942", 16): (16, 16, 16),
    ("gfx942", 8): (16, 16, 32),
}


def _cell_key(M: int, N: int, K: int, B: int = 1) -> str:
    def mn(v):
        return "Tiny" if v <= 32 else "Small" if v <= 128 else "Mid" if v <= 512 else "Large"

    kt = "TinyK" if K <= 32 else "MidK" if K <= 512 else "LargeK"
    bt = "Bnone" if B == 1 else "Bany"
    return f"{mn(M)}|{mn(N)}|{kt}|{bt}"


def _mi_k(arch: str, a_dtype: str, b_dtype: str) -> int:
    largest = max(int(DTYPE_BYTES[a_dtype] * 8), int(DTYPE_BYTES[b_dtype] * 8))
    if largest < 8:
        largest = 4
    return _MI_DIMS.get((arch, largest), (16, 16, 16))[2]


def _lds_bytes(bm, bn, bk, ba, bb, ns):
    a, b = bm * bk * ba, bk * bn * bb
    return max(a, b) if ns <= 1 else (ns - 1) * (a + b)


def _safe_log2(x):
    return log2(max(float(x), 1.0))


def _problem_features(M, N, K, B, a_dtype, out_dtype):
    bi, bo = DTYPE_BYTES.get(a_dtype, 2.0), DTYPE_BYTES.get(out_dtype, 2.0)
    flops = 2.0 * M * N * K * B
    mem = ((M * K + K * N) * bi + M * N * bo) * B
    return [_safe_log2(M), _safe_log2(N), _safe_log2(K), _safe_log2(B),
            _safe_log2(M * N), _safe_log2(M * N * K),
            log2(M / N) if N else 0.0, log2(M / K) if K else 0.0,
            _safe_log2(flops / max(mem, 1.0)), 1.0 if min(M, N) <= 128 else 0.0]


def _kernel_features(bm, bn, bk, ns, ba, bb, mi_k):
    lds = _lds_bytes(bm, bn, bk, ba, bb, ns)
    return [log2(bm), log2(bn), log2(bk), log2(bm * bn), log2(bm * bk),
            log2(bn * bk), float(ns), _safe_log2(lds), float(mi_k), float(bk) / float(mi_k)]


def _interaction_features(M, N, K, bm, bn, bk, num_cus):
    tiles_m, tiles_n = ceil(M / bm), ceil(N / bn)
    total = tiles_m * tiles_n
    k_iters = ceil(K / bk)
    waves = ceil(total / max(num_cus, 1))
    last_wave = total - (waves - 1) * num_cus
    return [_safe_log2(tiles_m), _safe_log2(tiles_n), _safe_log2(total),
            _safe_log2(k_iters), total / max(num_cus, 1), last_wave / max(num_cus, 1),
            (tiles_m * bm - M) / max(M, 1), (tiles_n * bn - N) / max(N, 1),
            (k_iters * bk - K) / max(K, 1), 1.0 if bm > M else 0.0, 1.0 if bn > N else 0.0]


if _TORCH_OK:

    class _TwoTower(nn.Module):
        def __init__(self, q_dim, i_dim, x_dim, embed_dim=32, hidden_dim=64, inter_hidden=16):
            super().__init__()

            def mlp(d_in, hid, d_out, depth):
                layers, d = [], d_in
                for _ in range(depth - 1):
                    layers += [nn.Linear(d, hid), nn.ReLU()]
                    d = hid
                layers.append(nn.Linear(d, d_out))
                return nn.Sequential(*layers)

            self.q_proj = mlp(q_dim, hidden_dim, embed_dim, 3)
            self.i_proj = mlp(i_dim, hidden_dim, embed_dim, 2)
            self.inter_mlp = nn.Sequential(nn.Linear(x_dim, inter_hidden), nn.ReLU(),
                                           nn.Linear(inter_hidden, 1))
            self.temperature = nn.Parameter(torch.tensor(1.0))

        def score_pairs(self, q, i, x):
            eg, et = self.q_proj(q), self.i_proj(i)
            T = self.temperature.abs().clamp(min=0.1)
            return (eg * et).sum(dim=-1) / T + self.inter_mlp(x).squeeze(-1)


def _apply_whiten(a, mean, std):
    std = np.where(std < 1e-6, 1.0, std)
    return ((a - mean) / std).astype(np.float32)


class MLRecommender:
    """Loads a bundle and re-ranks candidate tiles for a problem."""

    def __init__(self, bundle_path: str):
        self._bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        self.meta = self._bundle["meta"]
        self._cache = {}

    def _model_for(self, cell: str):
        if cell in self._cache:
            return self._cache[cell]
        entry = self._bundle["cells"].get(cell)
        if entry is None:
            self._cache[cell] = None
            return None
        m = _TwoTower(self.meta["q_dim"], self.meta["i_dim"], self.meta["x_dim"])
        m.load_state_dict(entry["state_dict"])
        m.eval()
        self._cache[cell] = (m, entry["norms"])
        return self._cache[cell]

    def best_tile(self, M: int, N: int, K: int,
                  candidates: Sequence[Tuple[int, int, int, int]],
                  B: int = 1) -> Optional[Tuple[int, int, int]]:
        """candidates: list of (bm, bn, bk, num_stages). Returns the top-1
        (bm, bn, bk) per the cell's model, or None to fall back to Origami."""
        got = self._model_for(_cell_key(M, N, K, B))
        if got is None or not candidates:
            return None
        model, norms = got
        meta = self.meta
        ba, bb = DTYPE_BYTES[meta["a_dtype"]], DTYPE_BYTES[meta["b_dtype"]]
        mi_k = _mi_k(meta["arch"], meta["a_dtype"], meta["b_dtype"])
        qf = _problem_features(M, N, K, B, meta["a_dtype"], meta["out_dtype"])
        q = np.asarray([qf] * len(candidates), np.float32)
        i = np.asarray([_kernel_features(bm, bn, bk, ns, ba, bb, mi_k)
                        for (bm, bn, bk, ns) in candidates], np.float32)
        x = np.asarray([_interaction_features(M, N, K, bm, bn, bk, meta["num_cus"])
                        for (bm, bn, bk, _) in candidates], np.float32)
        if (q.shape[1], i.shape[1], x.shape[1]) != (meta["q_dim"], meta["i_dim"], meta["x_dim"]):
            return None  # stale bundle / feature drift -> fall back
        qw = _apply_whiten(q, np.asarray(norms["q_mean"], np.float32), np.asarray(norms["q_std"], np.float32))
        iw = _apply_whiten(i, np.asarray(norms["i_mean"], np.float32), np.asarray(norms["i_std"], np.float32))
        xw = _apply_whiten(x, np.asarray(norms["x_mean"], np.float32), np.asarray(norms["x_std"], np.float32))
        with torch.no_grad():
            scores = model.score_pairs(torch.tensor(qw), torch.tensor(iw), torch.tensor(xw)).numpy()
        bm, bn, bk, _ = candidates[int(np.argmax(scores))]
        return (bm, bn, bk)


_RECOMMENDER: Optional[MLRecommender] = None
_TRIED = False


def get_recommender() -> Optional[MLRecommender]:
    """Lazily build the singleton recommender from env. Returns None when
    disabled or unavailable (caller falls back to Origami)."""
    global _RECOMMENDER, _TRIED
    if _TRIED:
        return _RECOMMENDER
    _TRIED = True
    if not _TORCH_OK:
        return None
    if os.environ.get("TRITONBLAS_ML_RECOMMENDER", "").strip().lower() not in ("1", "true", "on", "yes"):
        return None
    bundle = os.environ.get("TRITONBLAS_ML_BUNDLE", "").strip()
    if not bundle or not os.path.isfile(bundle):
        return None
    try:
        _RECOMMENDER = MLRecommender(bundle)
    except Exception:
        _RECOMMENDER = None
    return _RECOMMENDER
