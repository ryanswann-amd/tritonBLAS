"""P32 (S-002) — 23rd-slot N=160 K-COMPLEMENT verified-winner subset tests.

Source measurement: K-1794 paired n=30 HIP-graph hot-cache capture/replay
on MI300X / gfx942 across the 18-cell N=160 sub-cohort = M ∈ {2048, 4096,
8192} × N=160 × K ∈ {4096, 8192, 16384} × {bf16, fp16}.  3-engine sweep
(tb_oracle / tb_streamk / hbl) recorded:

  * bf16 N=160 (9/9): cohort geomean r_oracle = 0.997×.  All 9 cells were
    already routed via the upstream R-K979 P5 Clause-3 alias (broader K-
    COMPLEMENT envelope) so the post-route ratio is at parity with HBL.
    Including these cells in P32 is alias-overlap by design — the upstream
    P5 admit fires before P32 in the dispatch chain so the cells are
    documentation, not load-bearing in production.
  * fp16 N=160 (9/9): cohort UNROUTED in K-1794's measurement (0/9 routed
    via any upstream alias; r_oracle < 1.0 systemically per K-1794 R-
    K1794.FP16-MIRROR-SYSTEMIC-AT-CROSS-BAND-LEVEL).  These 9 cells are
    the load-bearing portion of P32 — they close the dtype-mirror gap at
    the wave-misaligned N=160 column-narrow tile.

Mechanism (K-913 §3 / R-K1673 dtype-invariance): BLOCK_N=128 packs N=160
into a single wave-misaligned K-block column; SQ_LDS_BANK_CONFLICT/inst
stays elevated and persistent_matmul cannot trade tile reshape for
atomic-reduction.  hipBLASLt's split-K kernel selection clears the band.

Invariants pinned in this file (cardinality lives here per the minimalist
split: source holds data, tests hold structural invariants — R-1532 /
R-1720 / R-1775):

  1. Cardinality is exactly 18.
  2. N axis is exactly {160}.
  3. M ∈ {2048, 4096, 8192}; K ∈ {4096, 8192, 16384};
     dtype ∈ {torch.bfloat16, torch.float16}.
  4. Per-shape dtype-mirror (bf16 ↔ fp16 admit sets coincide).
  5. Sibling-N firewall vs every prior K-COMPLEMENT alias-stack at a
     different N (P28 N=128, P29 N=64, P30 N ∈ {384, 768, 1536}, P31
     N=256, and the K-1367/K-1397 P13 N ∈ {128, 256} envelopes).
  6. Envelope is exactly the full 18-cell M ∈ {2048,4096,8192} × N=160
     × K ∈ {4096,8192,16384} × {bf16,fp16} grid.
"""
from __future__ import annotations

from tritonblas._route_predicate import (
    _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18,
    _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12,
    _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30,
    _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29,
    _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34,
    _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28,
    _P32_SKINNY_N160_KCOMPL_VERIFIED_WIN_18,
)


FZ = _P32_SKINNY_N160_KCOMPL_VERIFIED_WIN_18


def test_cardinality_is_18():
    assert len(FZ) == 18


def test_n_axis_is_skinny_n160():
    assert {N for (_, N, _, _) in FZ} == {160}


def test_m_k_dtype_axes_are_minimal():
    assert {M for (M, _, _, _) in FZ} == {2048, 4096, 8192}
    assert {K for (_, _, K, _) in FZ} == {4096, 8192, 16384}
    assert {dt for (_, _, _, dt) in FZ} == {"torch.bfloat16", "torch.float16"}


def test_dtype_mirror_is_complete():
    """Each shape (M, N, K) admitted under bf16 is also admitted under fp16
    (and vice versa).  K-913 §3 LDS-bank-conflict is dtype-invariant on the
    column-narrow N=160 tile (same mechanism as K-1673 P28 at N=128)."""
    bf = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.bfloat16"}
    fp = {(M, N, K) for (M, N, K, dt) in FZ if dt == "torch.float16"}
    assert bf == fp
    assert len(bf) == 9


def test_sibling_n_firewall_vs_p28_n128():
    assert FZ & _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30 == set()
    assert FZ & _K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18 == set()


def test_sibling_n_firewall_vs_p29_n64():
    assert FZ & _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29 == set()


def test_sibling_n_firewall_vs_p30_nmid():
    assert FZ & _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34 == set()


def test_sibling_n_firewall_vs_p31_n256():
    assert FZ & _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28 == set()
    assert FZ & _K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12 == set()


def test_envelope_equals_full_n160_kcompl_grid():
    """The verified-winner envelope is exactly the full 18-cell M ∈ {2048,
    4096,8192} × N=160 × K ∈ {4096,8192,16384} × {bf16,fp16} grid — pinned
    to detect any silent contraction or expansion of the K-1794 sub-cohort
    envelope.  Per K-1794, all 9 bf16 cells were already routed via the
    upstream R-K979 P5 alias (overlap-by-design); the 9 fp16 cells close
    the dtype-mirror gap and are the load-bearing portion of P32."""
    full = frozenset(
        (M, 160, K, dt) for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    )
    assert FZ == full
    assert len(full) == 18
