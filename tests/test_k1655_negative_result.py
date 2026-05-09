"""K-1655 regression test for documented negative result.

K-1641 §5 proposed three remediations for the M=N=4096 K-COMPLEMENT cohort
(K∈{2048, 16384, 32768} × {bf16, fp16}) on MI300X (gfx942):

  1. NS bump (num_stages 2 → 3)
  2. BK bump (BLOCK_K 64 → 128) — already infeasible at the (256, 256) tile
     because 256x256x128 bf16 NS=2 needs 128 KB > 64 KB hardware LDS limit.
  3. Narrower BN (e.g. 256x192)

K-1655 measured the LDS-feasible proxies and recorded:

  - NS=3 at (256, 256, 64) bf16/fp16 needs 128 KB > 64 KB → must raise
    `OutOfResources` at compile time on every gfx942 cell.
  - BK=32 (NS-direction LDS-feasible proxy) regresses ~12.4% geomean
    over the 6-cell cohort.
  - num_warps=4 (LDS-port-contention proxy) regresses ~33.7% geomean.
  - Narrower BN (256x192x64): tested at K=16384 bf16 only — within paired
    measurement noise of baseline (~0.999×). NOT validated cohort-wide.

This test pins the LDS-static facts so the comment in
`include/tritonblas/matmul.py::persistent_matmul_lt` cannot silently rot
across ROCm/Triton bumps:

  * Static (no-GPU): LDS budget at the cohort's (BM, BN, BK) shape exceeds
    gfx942 capacity for NS≥3 — falsifies K-1641 §5.1 by hardware constraint.
  * GPU-gated: re-run the 3 K-1641 §5 proxies on the 6 K-1641 cells and
    assert the documented regression direction (NS=3 raises OutOfResources;
    BK=32 and NW=4 ratios are within tolerance of the recorded values).

If a future Triton/ROCm release lifts the LDS limit or changes the LDS
buffer count for NS=3, this test will fail loudly and force the K-1655
comment block to be revisited.
"""
from __future__ import annotations

import math
import os

import pytest


# Recorded K-1655 6-cell paired n=30 hot-cache HIP-graph results
# (MI300X, useocpm2m-097-137 GPU 4, rocm/pytorch:rocm7.2_pytorch_2.10).
# See output/paired_6cell.json in K-1655 task workspace.
RECORDED_CELLS = [
    # (M, N, K, dt, base_us, BK32_us, NW4_us, hbl_us)
    (4096, 4096,  2048, "bf16",  130.428,  145.007,  201.048,  113.980),
    (4096, 4096,  2048, "fp16",  135.328,  150.231,  201.680,  119.382),
    (4096, 4096, 16384, "bf16",  971.660, 1117.343, 1488.888,  812.508),
    (4096, 4096, 16384, "fp16", 1005.239, 1156.738, 1491.644,  870.525),
    (4096, 4096, 32768, "bf16", 1960.760, 2273.436, 2977.262, 1687.290),
    (4096, 4096, 32768, "fp16", 2003.399, 2341.795, 2974.168, 1798.194),
]

# Documented geomean regressions (negative-result acceptance numbers).
RECORDED_BK32_GEOMEAN = 0.876   # 12.4% regression
RECORDED_NW4_GEOMEAN  = 0.663   # 33.7% regression

# Tolerance: paired n=30 hot-cache median has ~3% MoE on MI300X; allow 6%
# absolute drift on the ratio so single-node noise does not flake CI, but a
# direction reversal (>10% drift, or ratio crossing 1.0) trips the test.
RATIO_TOL = 0.06


def _geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


# --------------------------------------------------------------------------
# Static (no-GPU) tests: LDS budget falsifies K-1641 §5.1 by construction.
# --------------------------------------------------------------------------

class TestStaticLdsConstraint:
    """LDS arithmetic — runs without a GPU. Verifies K-1641 §5.1 (NS=3) is
    structurally infeasible at the cohort's tile shape (BM=BN=256, BK=64)."""

    def test_baseline_tile_fills_lds_exactly(self):
        """K-1655 documents (256, 256, 64) bf16 NS=2 = 64 KB on gfx942."""
        from tritonblas.origami import estimate_triton_lds_bytes
        # NS=2 → (BM*BK + BN*BK) * 2 bytes = (256*64 + 64*256) * 2 = 65536
        assert estimate_triton_lds_bytes(256, 256, 64, 2, 2, 2) == 65536

    def test_ns3_exceeds_gfx942_lds_limit(self):
        """K-1641 §5.1 (NS bump 2→3) — 128 KB > 64 KB gfx942 limit."""
        from tritonblas.origami import (
            check_triton_lds_capacity,
            estimate_triton_lds_bytes,
        )
        assert estimate_triton_lds_bytes(256, 256, 64, 2, 2, 3) == 131072
        assert check_triton_lds_capacity(256, 256, 64, 2, 2, 65536, 3) is False

    def test_bk128_exceeds_gfx942_lds_limit(self):
        """K-1641 §5.2 (BK bump 64→128) — 128 KB > 64 KB at NS=2."""
        from tritonblas.origami import check_triton_lds_capacity
        # (256*128 + 128*256) * 2 = 131072
        assert check_triton_lds_capacity(256, 256, 128, 2, 2, 65536, 2) is False

    def test_documented_kernel_defaults_unchanged(self):
        """The K-1655 comment claims persistent_matmul_lt still uses
        (NS=2, NW=8, kpack=1). Pin this so a regression-by-knob-tweak is
        flagged immediately."""
        import inspect
        from tritonblas.matmul import persistent_matmul_lt
        src = inspect.getsource(persistent_matmul_lt)
        assert "num_warps = 8" in src, "K-1655: NW=8 default removed unexpectedly"
        assert "kpack = 1" in src, "K-1655: kpack=1 default removed unexpectedly"
        # NS sourced from origami selector with default 2:
        assert 'num_stages = getattr(selector, "num_stages", 2)' in src, (
            "K-1655: NS default-of-2 source no longer matches comment"
        )


# --------------------------------------------------------------------------
# GPU-gated tests: reproduce the recorded ratios on the 6-cell cohort.
# --------------------------------------------------------------------------

def _have_mi300x():
    try:
        import torch
    except Exception:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        name = torch.cuda.get_device_name(0).lower()
    except Exception:
        return False
    return "mi300" in name or "gfx942" in name


# Default-SKIP unless explicitly requested via env. The benchmark takes ~5
# minutes on MI300X and burns a paired n=30 capture; CI should not run it
# unconditionally. Set K1655_RUN_BENCH=1 in the K-1655 reproducer harness.
@pytest.mark.skipif(
    not _have_mi300x() or os.environ.get("K1655_RUN_BENCH") != "1",
    reason="K-1655 GPU bench: needs MI300X and K1655_RUN_BENCH=1",
)
class TestK1655PairedBench:
    """Paired hot-cache reproduction of the K-1655 negative-result ratios."""

    @staticmethod
    def _bench_one(label, M, N, K, dt, num_stages=None, blk_k=None,
                   num_warps=None, n_replays=30, inner=20):
        """Single-cell HIP-graph hot-cache median in microseconds."""
        import torch
        import tritonblas

        torch.manual_seed(0)
        dt_t = torch.bfloat16 if dt == "bf16" else torch.float16
        a = torch.randn(M, K, dtype=dt_t, device="cuda")
        b = torch.randn(K, N, dtype=dt_t, device="cuda")
        c = torch.empty(M, N, dtype=dt_t, device="cuda")

        if label == "hbl":
            def fwd():
                torch.matmul(a, b, out=c)
        else:
            # Optional knob overrides — patched persistent kernel respects
            # K1655_* env vars per scripts/bench_one_cell.py.
            if num_stages is not None:
                os.environ["K1655_NUM_STAGES"] = str(num_stages)
            if blk_k is not None:
                os.environ["K1655_BLK_K_OVERRIDE"] = str(blk_k)
            if num_warps is not None:
                os.environ["K1655_NUM_WARPS"] = str(num_warps)

            def fwd():
                tritonblas.matmul(a, b, out=c)

        # Warmup
        for _ in range(5):
            fwd()
        torch.cuda.synchronize()

        # Capture HIP graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(inner):
                fwd()
        # Discard 3 post-capture replays
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()

        # n=30 timed replays
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_replays)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_replays)]
        for i in range(n_replays):
            starts[i].record()
            g.replay()
            ends[i].record()
        torch.cuda.synchronize()
        per_launch_us = sorted(
            (s.elapsed_time(e) * 1000.0) / inner for s, e in zip(starts, ends)
        )
        return per_launch_us[len(per_launch_us) // 2]

    def test_ns3_oom_lds_at_cohort_tile(self):
        """K-1641 §5.1 falsifier: NS=3 must raise OutOfResources on all 6 cells."""
        from triton.runtime.errors import OutOfResources
        for M, N, K, dt, *_ in RECORDED_CELLS:
            with pytest.raises(OutOfResources):
                self._bench_one("tb_alt_NS3", M, N, K, dt, num_stages=3)

    def test_bk32_regresses_within_tolerance(self):
        """BK=32 geomean across 6 cells must reproduce ~0.876× ± 6%."""
        ratios = []
        for M, N, K, dt, base_recorded, bk32_recorded, _, _ in RECORDED_CELLS:
            base = self._bench_one("tb_baseline", M, N, K, dt)
            bk32 = self._bench_one("tb_alt_BK32", M, N, K, dt, blk_k=32)
            ratios.append(base / bk32)
        gm = _geomean(ratios)
        assert gm < 1.0, f"K-1655: BK=32 must regress (got {gm:.3f}×)"
        assert abs(gm - RECORDED_BK32_GEOMEAN) < RATIO_TOL, (
            f"K-1655: BK=32 geomean {gm:.3f} drifted >6% from recorded "
            f"{RECORDED_BK32_GEOMEAN:.3f}"
        )

    def test_nw4_regresses_within_tolerance(self):
        """num_warps=4 geomean across 6 cells must reproduce ~0.663× ± 6%."""
        ratios = []
        for M, N, K, dt, base_recorded, _, nw4_recorded, _ in RECORDED_CELLS:
            base = self._bench_one("tb_baseline", M, N, K, dt)
            nw4 = self._bench_one("tb_alt_NW4", M, N, K, dt, num_warps=4)
            ratios.append(base / nw4)
        gm = _geomean(ratios)
        assert gm < 1.0, f"K-1655: NW=4 must regress (got {gm:.3f}×)"
        assert abs(gm - RECORDED_NW4_GEOMEAN) < RATIO_TOL, (
            f"K-1655: NW=4 geomean {gm:.3f} drifted >6% from recorded "
            f"{RECORDED_NW4_GEOMEAN:.3f}"
        )


# --------------------------------------------------------------------------
# Recorded-data structural assertions (run without GPU): pins the regression
# direction and magnitude documented in the source comment so the comment
# cannot silently outlive the data behind it.
# --------------------------------------------------------------------------

class TestRecordedNegativeResult:
    """Assert the recorded K-1655 numbers match the magnitudes/directions
    documented in `persistent_matmul_lt`'s K-1655 comment block."""

    def test_recorded_bk32_geomean_matches_comment(self):
        ratios = [base / bk32 for _, _, _, _, base, bk32, _, _ in RECORDED_CELLS]
        gm = _geomean(ratios)
        assert gm < 1.0, "BK=32 must regress"
        # Comment says 0.876× — verify recorded data still matches.
        assert abs(gm - RECORDED_BK32_GEOMEAN) < 0.005, (
            f"recorded BK=32 geomean drifted from K-1655 comment: {gm:.3f}"
        )

    def test_recorded_nw4_geomean_matches_comment(self):
        ratios = [base / nw4 for _, _, _, _, base, _, nw4, _ in RECORDED_CELLS]
        gm = _geomean(ratios)
        assert gm < 1.0, "NW=4 must regress"
        assert abs(gm - RECORDED_NW4_GEOMEAN) < 0.005, (
            f"recorded NW=4 geomean drifted from K-1655 comment: {gm:.3f}"
        )

    def test_baseline_loses_to_hbl_on_cohort(self):
        """Documents the residual cohort-wide deficit vs hipBLASLt."""
        ratios = [base / hbl for _, _, _, _, base, _, _, hbl in RECORDED_CELLS]
        gm = _geomean(ratios)
        # K-1655 comment claims ~15% deficit; recorded is ~1.15×.
        assert gm > 1.05, f"baseline must trail hipBLASLt on cohort: {gm:.3f}×"
        assert gm < 1.25, f"baseline deficit out of K-1655 envelope: {gm:.3f}×"
