"""K-2097 P52 (S-002) — 9th-rung off-by-48 N=1328 K-COMPLEMENT alias-stack tests.

Source measurement: K-2072 paired residue-48 sweep (N ∈ {1328..1776}, M=4096,
K=8192, MI300X / gfx942) pre-validated this rung at TB-BEFORE / hipBLASLt
1.748× (bf16) / 1.745× (fp16) — both strongly admit-eligible at the ≥1.10×
gate.  K-2097 extends to the full 18-cell M ∈ {2048, 4096, 8192} × N=1328 ×
K ∈ {4096, 8192, 16384} × {bf16, fp16} grid for production promotion.

Off-by-48 ladder context (R-1811 wave-misalignment + K-913 §3 LDS-bank-conflict
mechanism, dtype-invariant): N=1328 is the 9th contiguous rung of the
off-by-48 wave-misaligned residue family above N ∈ {816, 880, 944, 1008,
1072, 1136, 1200, 1264} (K-1978 / K-1994 / K-2010 / K-2031 / K-2047 / K-2055 /
K-2085 P50 / K-2085 P51).  All nine rungs satisfy (N mod 64) == 48.

Sub-family placement: N=1328 mod 128 == 48 — canonical-tail sub-family
shared with N ∈ {816, 944, 1072, 1200} (vs the off-tail sub-family at
N ∈ {880, 1008, 1136, 1264} where N mod 128 == 112).  Step +64 from the
K-2085 P51 rung at N=1264.

Envelope choice: 18-cell K ∈ {4096, 8192, 16384} (matches the K-2055 P48
N=1136 18-cell envelope convention) rather than the K-2085 12-cell
envelope; K=4096 is retained because the K-2097 confirmation sweep
explicitly enumerates K ∈ {4096, 8192, 16384} for the 9th-rung promotion.

Invariants pinned in this file (cardinality lives here per the minimalist
split: source holds data, tests hold structural invariants — R-1532 /
R-1720 / R-1775):

  1. Cardinality is exactly 18 (full grid; no upstream-aliased exclusions).
  2. N axis is exactly {1328}.
  3. M ∈ {2048, 4096, 8192}; K ∈ {4096, 8192, 16384};
     dtype ∈ {torch.bfloat16, torch.float16}.
  4. Both dtype rows are complete 9/9 (off-band N=1328 has no R-K979 P5
     Clause-3 bf16 alias — strictly load-bearing on both rows).
  5. Sibling-N firewall vs the K-1922 P40 N=544 alias-stack (the only
     other directly-imported single-N alias-stack at this dispatcher
     position).
  6. Off-by-48 wave-misalignment band membership: 1328 mod 64 == 48 (same
     ladder as the 8 prior confirmed rungs).
  7. Canonical-tail sub-family membership: 1328 mod 128 == 48 (same
     sub-family as N ∈ {816, 944, 1072, 1200}).
"""
from __future__ import annotations

from tritonblas._route_predicate import (
    _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18,
    _K2097_P52_SKINNY_N1328_KCOMPL_ALIASSTACK_18,
)


FZ = _K2097_P52_SKINNY_N1328_KCOMPL_ALIASSTACK_18


def test_cardinality_is_18():
    assert len(FZ) == 18


def test_n_axis_is_skinny_n1328():
    assert {N for (_, N, _, _) in FZ} == {1328}


def test_m_k_dtype_axes_are_minimal():
    assert {M for (M, _, _, _) in FZ} == {2048, 4096, 8192}
    assert {K for (_, _, K, _) in FZ} == {4096, 8192, 16384}
    assert {dt for (_, _, _, dt) in FZ} == {"torch.bfloat16", "torch.float16"}


def test_dtype_row_balance():
    """Both dtype rows complete 9/9.  Off-band N=1328 has no R-K979 P5
    Clause-3 bf16 alias coverage at this rung; both rows are strictly
    load-bearing.  K-913 §3 LDS-bank-conflict is dtype-invariant on the
    column-narrow N=1328 tile (same fingerprint as the 8 prior off-by-48
    rungs)."""
    bf = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.bfloat16"}
    fp = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.float16"}
    assert len(bf) == 9
    assert len(fp) == 9
    assert bf == fp


def test_envelope_equals_full_n1328_kcompl_grid():
    """The verified-winner envelope is exactly the 18-cell M ∈ {2048,
    4096, 8192} × N=1328 × K ∈ {4096, 8192, 16384} × {bf16, fp16} grid
    (no upstream-aliased exclusions at this off-by-48 rung) — pinned to
    detect any silent contraction (a false-NEGATIVE that would leak
    winner cells back to TB) or expansion (a false-POSITIVE that would
    leak non-winner cells out to HBL) of the K-2097 promotion envelope."""
    full = frozenset(
        (M, 1328, K, dt) for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    )
    assert FZ == full
    assert len(full) == 18


def test_off_by_48_wave_misalignment_ladder_membership():
    """N=1328 sits on the off-by-48 wave-misaligned ladder (N mod 64 == 48)
    above the 8 prior confirmed rungs N ∈ {816, 880, 944, 1008, 1072, 1136,
    1200, 1264}.  Step +64 from the K-2085 P51 N=1264 rung.  This invariant
    guards against a silent N-axis typo (e.g. 1326, 1330, 1392) that would
    land on a different BLOCK_N=64 alignment rung and reflect a different
    mechanism."""
    (only_n,) = {N for (_, N, _, _) in FZ}
    assert only_n % 64 == 48
    assert only_n == 1264 + 64
    assert only_n == 1328


def test_canonical_tail_sub_family_membership():
    """N=1328 mod 128 == 48 — canonical-tail sub-family shared with
    N ∈ {816, 944, 1072, 1200} (R-1811 wave-misalignment sub-classification).
    Distinct from the off-tail sub-family at N mod 128 == 112 (N ∈ {880,
    1008, 1136, 1264})."""
    (only_n,) = {N for (_, N, _, _) in FZ}
    assert only_n % 128 == 48


def test_sibling_n_firewall_vs_p40_n544():
    """The only other single-N alias-stack imported at this dispatcher
    position is K-1922 P40 N=544.  Disjoint by natural N-axis separation
    (1328 ≠ 544); guards against a silent N-axis collision."""
    assert FZ & _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18 == set()
