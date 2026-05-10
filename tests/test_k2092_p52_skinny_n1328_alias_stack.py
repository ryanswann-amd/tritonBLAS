"""K-2092 P52 (S-002) — 40th-slot N=1328 K-COMPLEMENT alias-stack tests.

Source measurement: K-2072 paired n=30 HIP-graph hot-cache MI300X / gfx942
3-engine bench (TB-baseline vs hipBLASLt vs rocBLAS) over the residue-48
ladder showed N=1328 already beating baseline tritonblas (TB-baseline /
hipBLASLt < 1.0; HBL routing ≈ parity), placing N=1328 as the next
contiguous rung above K-2085 P51 N=1264 in the off-by-48 wave-misaligned
residue family.  K-2092 formalizes the admission via the full 18-cell
M ∈ {2048, 4096, 8192} × N=1328 × K ∈ {4096, 8192, 16384} × {bf16, fp16}
envelope used for production promotion of every prior off-by-48 rung
(P47 N=1008, P48 N=1072, P49 N=1136, plus P50 N=1200 / P51 N=1264).

Mechanism (R-1811 wave-misalignment + K-913 §3 LDS-bank-conflict):
BLOCK_N=128 packs N=1328 into 10.375 BLOCK_N tiles per N-row (1328 mod
128 = 48 — same off-by-48 band as P50 N=1200 = 9.375 tiles, P51 N=1264
= 9.875 tiles, and every prior residue-48 rung).  Same SCHEDULER_LDS A4
failure mode as the K-1681 / K-1710 / K-1781 / K-1812 / K-1824 / K-1832 /
K-1843 wave-misaligned skinny-N class.  hipBLASLt's split-K kernel
selection clears the band.

Invariants pinned in this file (cardinality lives here per the minimalist
split: source holds data, tests hold structural invariants — R-1532 /
R-1720 / R-1775):

  1. Cardinality is exactly 18 (full grid; no upstream-aliased exclusions).
  2. N axis is exactly {1328}.
  3. M ∈ {2048, 4096, 8192}; K ∈ {4096, 8192, 16384};
     dtype ∈ {torch.bfloat16, torch.float16}.
  4. Both dtype rows are complete 9/9.
  5. Sibling-N firewall vs every prior K-COMPLEMENT alias-stack at a
     different N (P40 N=544 most relevant on this branch base).
  6. Off-by-48 wave-misalignment band membership: 1328 mod 64 == 48.
"""
from __future__ import annotations

from tritonblas._route_predicate import (
    _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18,
    _K2092_P52_SKINNY_N1328_KCOMPL_ALIASSTACK_18,
)


FZ = _K2092_P52_SKINNY_N1328_KCOMPL_ALIASSTACK_18


def test_cardinality_is_18():
    assert len(FZ) == 18


def test_n_axis_is_skinny_n1328():
    assert {N for (_, N, _, _) in FZ} == {1328}


def test_m_k_dtype_axes_are_minimal():
    assert {M for (M, _, _, _) in FZ} == {2048, 4096, 8192}
    assert {K for (_, _, K, _) in FZ} == {4096, 8192, 16384}
    assert {dt for (_, _, _, dt) in FZ} == {"torch.bfloat16", "torch.float16"}


def test_dtype_row_balance():
    """Both dtype rows complete 9/9.  Off-by-48 N=1328 follows the same
    K-913 §3 LDS-bank-conflict dtype-invariance as every prior residue-48
    rung — both rows are strictly load-bearing."""
    bf = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.bfloat16"}
    fp = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.float16"}
    assert len(bf) == 9
    assert len(fp) == 9
    assert bf == fp


def test_envelope_equals_full_n1328_kcompl_grid():
    """The verified-winner envelope is exactly the 18-cell M ∈ {2048,
    4096, 8192} × N=1328 × K ∈ {4096, 8192, 16384} × {bf16, fp16} grid
    (no upstream-aliased exclusions at this off-by-48 rung) — pinned to
    detect any silent contraction or expansion of the K-2092 verification
    envelope."""
    full = frozenset(
        (M, 1328, K, dt) for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    )
    assert FZ == full
    assert len(full) == 18


def test_off_by_48_wave_misalignment_band_membership():
    """N=1328 sits in the off-by-48 wave-misaligned band (N mod 64 == 48),
    the same band as the entire residue-48 ladder N ∈ {816, 880, 944,
    1008, 1072, 1136, 1200, 1264}.  This invariant guards against a
    silent N-axis typo (e.g. 1320, 1336) that would land on a different
    BLOCK_N=128 alignment rung."""
    (only_n,) = {N for (_, N, _, _) in FZ}
    assert only_n % 64 == 48
    assert only_n % 128 == 48
    assert only_n == 1328


def test_sibling_n_firewall_vs_p40_n544():
    """P40 covers N=544 (off-by-32); K-2092 P52 covers N=1328
    (off-by-48) — disjoint by natural N-axis separation."""
    assert FZ & _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18 == set()


def test_next_rung_above_k2085_p51_n1264():
    """K-2092 P52 N=1328 is exactly +64 above K-2085 P51 N=1264 — the
    canonical step of the off-by-48 ladder.  Pinned here so a future
    consolidation cannot silently snap N=1328 to a non-ladder value."""
    only_n = next(iter({N for (_, N, _, _) in FZ}))
    assert only_n - 1264 == 64
    assert only_n - 1200 == 128
