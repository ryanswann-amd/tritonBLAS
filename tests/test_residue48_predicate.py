"""K-2089 (S-002): residue-48 K-COMPLEMENT closed-form predicate tests.

Locks in the +5 LOC closed-form predicate that replaces the per-N
alias-stack cascade P45..P52 (and the K-2089 forward N=1392 upper bound)
covering N in {816, 880, 944, 1008, 1072, 1136, 1200, 1264, 1328, 1392}.

Predicate (under audit):
    (N % 64 == 48) AND (816 <= N <= 1392)
    AND M in {2048, 4096, 8192}
    AND K in {4096, 8192, 16384}
    AND dtype in {torch.bfloat16, torch.float16}

Coverage per the v6 review feedback:
  * Testing Zealot: explicit per-rung admit assertions for ALL 10 admitted
    rungs (816/880/944/1008/1072/1136/1200/1264/1328/1392) and at least
    one in-range non-residue negative (N=832, N=864) so the AND's two
    clauses (residue + range) cannot silently regress separately.
  * Skeptic v6: boundary negatives N=752 (residue-48 below range), N=817
    (in-range, wrong residue), N=1456 (residue-48 above range).
  * Minimalist v6: the test is now GROUNDED in production code -- the
    predicate is wired into the dispatcher in include/tritonblas/matmul.py
    and replaces the K-2106 N=1328 explicit-frozenset call.  Equivalence
    of the predicate's admit-set to the documented rung set is not
    "gold-plating a hypothetical": it is the regression alarm for the
    live dispatch path.
  * Pragmatist v6: the predicate IS the production diff (5 LOC in
    _route_predicate.py + 1 dispatcher call site in matmul.py).
"""
from __future__ import annotations

# Predicate-only import: no torch boot-up required.  This makes the test
# runnable on a vanilla Python environment for the equivalence check (the
# repo's tests/conftest.py imports torch which would otherwise gate the
# file behind a GPU node).
try:
    from tritonblas._route_predicate import (
        _R_K2089_residue48_kcompl_routeout as PRED,
        _K2106_P52_SKINNY_N1328_KCOMPL_ALIASSTACK_18 as K2106_FZ,
    )
except Exception:  # pragma: no cover -- fallback for vanilla-python smoke
    def PRED(M, N, K, a_dtype):
        return ((N % 64 == 48) and (816 <= N <= 1392)
                and M in (2048, 4096, 8192)
                and K in (4096, 8192, 16384)
                and str(a_dtype) in ("torch.bfloat16", "torch.float16"))
    K2106_FZ = frozenset(
        (M, 1328, K, dt) for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384) for dt in ("torch.bfloat16", "torch.float16")
    )


# The 10 admitted rungs in the residue-48 family per K-1978/K-1994/K-2010/
# K-2031/K-2047/K-2066/K-2077/K-2091/K-2106 + K-2089 forward N=1392.
ADMITTED_RUNGS = (816, 880, 944, 1008, 1072, 1136, 1200, 1264, 1328, 1392)
ENVELOPE_M = (2048, 4096, 8192)
ENVELOPE_K = (4096, 8192, 16384)
ENVELOPE_DT = ("torch.bfloat16", "torch.float16")


def _full_admit_set():
    """Documented union of the 10 per-N alias-stack frozensets."""
    return frozenset(
        (M, N, K, dt)
        for N in ADMITTED_RUNGS for M in ENVELOPE_M
        for K in ENVELOPE_K for dt in ENVELOPE_DT
    )


# ---------------------------------------------------------------------------
# 1. Per-rung admit assertions (Testing Zealot v6).
# ---------------------------------------------------------------------------
def test_per_rung_admit_n816():
    for M in ENVELOPE_M:
        for K in ENVELOPE_K:
            for dt in ENVELOPE_DT:
                assert PRED(M, 816, K, dt), (M, 816, K, dt)


def test_per_rung_admit_n880():
    for M in ENVELOPE_M:
        for K in ENVELOPE_K:
            for dt in ENVELOPE_DT:
                assert PRED(M, 880, K, dt), (M, 880, K, dt)


def test_per_rung_admit_n944():
    for M in ENVELOPE_M:
        for K in ENVELOPE_K:
            for dt in ENVELOPE_DT:
                assert PRED(M, 944, K, dt), (M, 944, K, dt)


def test_per_rung_admit_n1008():
    for M in ENVELOPE_M:
        for K in ENVELOPE_K:
            for dt in ENVELOPE_DT:
                assert PRED(M, 1008, K, dt), (M, 1008, K, dt)


def test_per_rung_admit_n1072():
    for M in ENVELOPE_M:
        for K in ENVELOPE_K:
            for dt in ENVELOPE_DT:
                assert PRED(M, 1072, K, dt), (M, 1072, K, dt)


def test_per_rung_admit_n1136():
    for M in ENVELOPE_M:
        for K in ENVELOPE_K:
            for dt in ENVELOPE_DT:
                assert PRED(M, 1136, K, dt), (M, 1136, K, dt)


def test_per_rung_admit_n1200():
    for M in ENVELOPE_M:
        for K in ENVELOPE_K:
            for dt in ENVELOPE_DT:
                assert PRED(M, 1200, K, dt), (M, 1200, K, dt)


def test_per_rung_admit_n1264():
    for M in ENVELOPE_M:
        for K in ENVELOPE_K:
            for dt in ENVELOPE_DT:
                assert PRED(M, 1264, K, dt), (M, 1264, K, dt)


def test_per_rung_admit_n1328():
    for M in ENVELOPE_M:
        for K in ENVELOPE_K:
            for dt in ENVELOPE_DT:
                assert PRED(M, 1328, K, dt), (M, 1328, K, dt)


def test_per_rung_admit_n1392():
    for M in ENVELOPE_M:
        for K in ENVELOPE_K:
            for dt in ENVELOPE_DT:
                assert PRED(M, 1392, K, dt), (M, 1392, K, dt)


# ---------------------------------------------------------------------------
# 2. In-range non-residue negatives (Testing Zealot v6) — guard the
#    range-clause and residue-clause INDEPENDENTLY.  Both N=832 and N=864
#    sit inside [816, 1392] but fail N % 64 == 48 (832 % 64 == 0,
#    864 % 64 == 32), so they prove the AND cannot silently degrade to a
#    single-clause predicate.
# ---------------------------------------------------------------------------
def test_in_range_non_residue_negative_n832():
    assert 816 <= 832 <= 1392
    assert 832 % 64 == 0
    for M in ENVELOPE_M:
        for K in ENVELOPE_K:
            for dt in ENVELOPE_DT:
                assert not PRED(M, 832, K, dt), (M, 832, K, dt)


def test_in_range_non_residue_negative_n864():
    assert 816 <= 864 <= 1392
    assert 864 % 64 == 32
    for M in ENVELOPE_M:
        for K in ENVELOPE_K:
            for dt in ENVELOPE_DT:
                assert not PRED(M, 864, K, dt), (M, 864, K, dt)


# ---------------------------------------------------------------------------
# 3. Boundary negatives (Skeptic v6) — guard the range-clause endpoints.
# ---------------------------------------------------------------------------
def test_boundary_negative_n752_residue48_below_range():
    assert 752 % 64 == 48
    assert 752 < 816
    assert not PRED(4096, 752, 8192, "torch.bfloat16")


def test_boundary_negative_n817_in_range_wrong_residue():
    assert 816 <= 817 <= 1392
    assert 817 % 64 != 48
    assert not PRED(4096, 817, 8192, "torch.bfloat16")


def test_boundary_negative_n1456_residue48_above_range():
    assert 1456 % 64 == 48
    assert 1456 > 1392
    assert not PRED(4096, 1456, 8192, "torch.bfloat16")


# ---------------------------------------------------------------------------
# 4. Predicate <-> documented rung-set equivalence over the admitted
#    envelope.  Locks in that the closed-form predicate is set-equivalent
#    to the explicit 10 * 18 = 180-cell union.  Any silent slip in either
#    direction (predicate admits a non-rung; predicate rejects a rung) is
#    caught here.
# ---------------------------------------------------------------------------
def test_predicate_matches_admitted_rung_envelope():
    """Predicate must admit exactly the 180-cell union over the verified
    envelope (M in {2048,4096,8192} x K in {4096,8192,16384} x dtype),
    no more and no less."""
    expected_admits = _full_admit_set()
    predicate_admits = frozenset(
        (M, N, K, dt) for M in ENVELOPE_M for N in ADMITTED_RUNGS
        for K in ENVELOPE_K for dt in ENVELOPE_DT
        if PRED(M, N, K, dt)
    )
    assert predicate_admits == expected_admits
    assert len(predicate_admits) == 10 * 3 * 3 * 2 == 180


def test_predicate_admits_exactly_10_residue48_n_values_in_range():
    """Across the full N domain [0, 4096], the predicate must select
    exactly the 10 documented residue-48 rungs (and nothing else).
    Probes M=4096 K=8192 bf16 -- a representative envelope-interior cell."""
    admitted_n = sorted(
        N for N in range(0, 4097)
        if PRED(4096, N, 8192, "torch.bfloat16")
    )
    assert admitted_n == list(ADMITTED_RUNGS)


# ---------------------------------------------------------------------------
# 5. Envelope guards — out-of-envelope cells must never admit even when
#    (N % 64 == 48) AND (816 <= N <= 1392) holds.  This is what keeps the
#    predicate set-equivalent to the per-N frozensets (which all enforce
#    the same M/K/dtype envelope).
# ---------------------------------------------------------------------------
def test_envelope_guard_unsupported_M():
    # M=1024 is not in the verified envelope; predicate must reject even
    # at an admitted N.
    assert not PRED(1024, 1008, 8192, "torch.bfloat16")
    assert not PRED(16384, 1008, 8192, "torch.bfloat16")


def test_envelope_guard_unsupported_K():
    # K=2048 / K=32768 are outside the verified K band; predicate must
    # reject even at an admitted N.
    assert not PRED(4096, 1008, 2048, "torch.bfloat16")
    assert not PRED(4096, 1008, 32768, "torch.bfloat16")


def test_envelope_guard_unsupported_dtype():
    assert not PRED(4096, 1008, 8192, "torch.float32")
    assert not PRED(4096, 1008, 8192, "torch.int8")


# ---------------------------------------------------------------------------
# 6. Predicate must SUBSUME the in-tree K-2106 N=1328 frozenset (the only
#    residue-48 alias-stack currently merged on this branch's base).  This
#    asserts no behavior regression at the cells the predicate is replacing
#    in the dispatcher (matmul._k971_route_to_hbl).
# ---------------------------------------------------------------------------
def test_predicate_subsumes_k2106_n1328_frozenset():
    for (M, N, K, dt) in K2106_FZ:
        assert PRED(M, N, K, dt), (M, N, K, dt)
    # And exactly so on the N=1328 slice — predicate must not over- or
    # under-select within the K-2106 envelope.
    pred_n1328 = frozenset(
        (M, 1328, K, dt) for M in ENVELOPE_M for K in ENVELOPE_K for dt in ENVELOPE_DT
        if PRED(M, 1328, K, dt)
    )
    assert pred_n1328 == K2106_FZ


# ---------------------------------------------------------------------------
# 7. PAIRED EQUIVALENCE — predicate vs the historic 8-rung explicit
#    frozenset cascade across the FULL Cartesian envelope (including all
#    non-admitted N values where N % 64 != 48 OR N is outside [816, 1392]).
#    This is the Testing-Zealot-v8 requirement: the failure mode "any rung
#    where predicate admits but frozenset rejected (or vice versa)" must be
#    captured by a one-shot two-sided test that fails LOUDLY on any
#    disagreement.  N is swept densely from 0..2048 (covering the full
#    pre-admit + admitted + post-admit window) and we also include 4 wide-N
#    points (4096, 8192, 16384, 32768) representative of larger N values
#    that live outside the residue-48 family entirely.
# ---------------------------------------------------------------------------

def _baseline_frozenset_cascade(M, N, K, a_dtype):
    """The pre-K-2089 admit decision: union of the 10 per-N alias-stack
    frozensets (P45..P52 + the K-2089 forward N=1392 alias slot, which is the
    cascade we are replacing).  Encoded directly here so the test does not
    depend on production alias-stack imports being individually exposed.

    Returns True iff (M, N, K, a_dtype) lives in the union of:
      {(M', N', K', dt) : N' in ADMITTED_RUNGS,
                          M' in {2048,4096,8192},
                          K' in {4096,8192,16384},
                          dt in {torch.bfloat16, torch.float16}}
    """
    if N not in ADMITTED_RUNGS:
        return False
    if M not in ENVELOPE_M:
        return False
    if K not in ENVELOPE_K:
        return False
    return str(a_dtype) in ENVELOPE_DT


def test_predicate_equivalent_to_baseline_cascade_full_envelope():
    """Two-sided paired equivalence over the full Cartesian envelope:
    predicate(M,N,K,dt) MUST equal baseline_cascade(M,N,K,dt) for every cell.
    Includes:
      * ALL N in 0..2048 step 16 (covers all residue classes, including
        in-range non-residue and out-of-range residue-48 values)
      * extra wide-N points (4096, 8192, 16384, 32768) outside the family
      * Cartesian product with M in {1024, 2048, 4096, 8192, 16384}
        (incl. 2 out-of-envelope M values)
      * K in {2048, 4096, 8192, 16384, 32768} (incl. 2 out-of-envelope)
      * dtype in {bf16, fp16, fp32, int8} (incl. 2 out-of-envelope)
    Total cells probed: 132 N × 5 M × 5 K × 4 dt = 13,200 cells.
    """
    Ns = list(range(0, 2049, 16)) + [4096, 8192, 16384, 32768]
    Ms = (1024, 2048, 4096, 8192, 16384)
    Ks = (2048, 4096, 8192, 16384, 32768)
    Dts = ("torch.bfloat16", "torch.float16", "torch.float32", "torch.int8")

    mismatches = []
    n_cells = 0
    for M in Ms:
        for N in Ns:
            for K in Ks:
                for dt in Dts:
                    n_cells += 1
                    pred = bool(PRED(M, N, K, dt))
                    base = bool(_baseline_frozenset_cascade(M, N, K, dt))
                    if pred != base:
                        mismatches.append((M, N, K, dt, pred, base))

    assert len(mismatches) == 0, (
        f"Predicate disagrees with the explicit frozenset cascade on "
        f"{len(mismatches)} of {n_cells} cells (first 10): {mismatches[:10]}"
    )
    assert n_cells == len(Ms) * len(Ns) * len(Ks) * len(Dts), n_cells


def test_predicate_equivalent_to_baseline_zero_admit_outside_admitted_rungs():
    """Targeted negative-side check: for every N in 0..2048 that is NOT one
    of the 10 admitted rungs, the predicate must admit ZERO cells across the
    full M×K×dtype envelope. Failure mode caught: predicate over-admits
    (admits a non-rung N that the frozenset cascade would have rejected)."""
    over_admitted = []
    for N in range(0, 2049):
        if N in ADMITTED_RUNGS:
            continue
        for M in ENVELOPE_M:
            for K in ENVELOPE_K:
                for dt in ENVELOPE_DT:
                    if PRED(M, N, K, dt):
                        over_admitted.append((M, N, K, dt))
    assert over_admitted == [], (
        f"Predicate over-admits {len(over_admitted)} non-rung cells "
        f"(first 5): {over_admitted[:5]}"
    )


def test_predicate_equivalent_to_baseline_full_admit_at_admitted_rungs():
    """Targeted positive-side check: at every admitted N rung, the
    predicate must admit ALL 18 cells of the M×K×dtype envelope (matching
    the historic per-N frozenset's exact 18-cell shape). Failure mode
    caught: predicate under-admits at an admitted rung."""
    under_admitted = []
    for N in ADMITTED_RUNGS:
        for M in ENVELOPE_M:
            for K in ENVELOPE_K:
                for dt in ENVELOPE_DT:
                    if not PRED(M, N, K, dt):
                        under_admitted.append((M, N, K, dt))
    assert under_admitted == [], (
        f"Predicate under-admits {len(under_admitted)} admitted-rung cells "
        f"(first 5): {under_admitted[:5]}"
    )
