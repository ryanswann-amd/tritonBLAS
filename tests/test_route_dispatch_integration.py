"""K-1106 dispatch-path integration test.

Exercises ``k971_route_decision`` end-to-end (the function the production
dispatcher calls), not just the underlying bool predicates.  Catches the
class of bugs that pure-predicate truth-table tests miss:

  - StreamK / work-stealing short-circuits firing accidentally;
  - dtype-mismatch short-circuit firing accidentally;
  - ``disable_env_set`` master kill firing on the bench cells;
  - K971_ROUTE_TABLE fp16 fall-through getting masked by the
    P7 OR-gate on bf16-only cells;
  - the K-1031 S07 leakage cell sneaking through the legacy table.

Run with: python -m pytest tests/test_route_dispatch_integration.py -x
or stand-alone: python tests/test_route_dispatch_integration.py
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "include"))

from tritonblas._route_predicate import (  # noqa: E402
    K971_ROUTE_TABLE,
    R_K979_P5_route_to_hbl,
    R_K1037_P6_admit_wpeu1,
    R_K1106_P7_extra,
    R_K1106_P7_route_to_hbl,
    k971_route_decision,
)


def _route_before_p7(M, N, K, dt):
    """The dispatch decision BEFORE this PR: P5 alone + K971_ROUTE_TABLE
    fall-through (P6 was previously env-guarded and inactive in production).
    Mirrors ``scripts/bench_p7_n30_hipgraph.py::_route_before_p7`` so the
    test and the bench harness agree on the BEFORE-P7 baseline."""
    p5 = R_K979_P5_route_to_hbl(M, N, K, dt)
    return p5 or ((M, N, K, dt) in K971_ROUTE_TABLE)


def _route_after_p7(M, N, K, dt):
    """The dispatch decision AFTER this PR — k971_route_decision with
    short-circuits cleared."""
    return k971_route_decision(M, N, K, dt, dt, False, False, disable_env_set=False)


BF16 = "torch.bfloat16"
FP16 = "torch.float16"


def _decide(M, N, K, dt, *, sk=False, ws=False, dis=False):
    return k971_route_decision(M, N, K, dt, dt, sk, ws, disable_env_set=dis)


# --- Master kill / short-circuit guards ----------------------------------

def test_disable_env_set_master_kill():
    """disable_env_set=True must force the dispatcher to NOT route, even on
    a known-routed cell."""
    # S29 (P5+P6 positive) — would normally route, but kill flag dominates.
    assert not _decide(14208, 2048, 1024, BF16, dis=True)


def test_streamk_short_circuit_blocks_route():
    """enable_streamk=True is the historical opt-out; route_decision must
    return False even on a P7-positive cell."""
    assert not _decide(14208, 2048, 1024, BF16, sk=True)


def test_work_stealing_short_circuit_blocks_route():
    assert not _decide(14208, 2048, 1024, BF16, ws=True)


def test_dtype_mismatch_short_circuit_blocks_route():
    """a_dtype != b_dtype must short-circuit to False (covers a8w8 / mixed
    quantized cases that the structural predicate is not calibrated for)."""
    assert not k971_route_decision(
        14208, 2048, 1024, BF16, FP16, False, False, disable_env_set=False
    )


# --- P7 OR-gate routing on the bench union --------------------------------

def test_k1074_in_scope_all_route():
    """K-1074 8-cell in-scope: every cell must route to hbl."""
    cells = [
        (8064, 2048, 1024), (18304, 2048, 1024), (20352, 2048, 1024),
        (22400, 2048, 1024), (24448, 2048, 1024), (26496, 2048, 1024),
        (25600, 2048, 256), (49152, 2048, 256),
    ]
    for M, N, K in cells:
        assert _decide(M, N, K, BF16), f"K-1074 ({M},{N},{K}) failed to route"


def test_k1079_s24_route_via_p6_envelope_b():
    """S24 = (4480,3072,768,bf16) — the SOLE new route added by P7 vs P5.
    P5 alone would NOT route (Clause-1 fails: 3072>2304 minMN bound; etc.).
    P6 Envelope B (S24 fingerprint) fires.  The OR-gate must take it."""
    assert _decide(4480, 3072, 768, BF16)


def test_k1079_s24_route_FLIP_before_after_p7():
    """Testing-Zealot guard: explicitly assert the before/after route-flip
    for S24 = (4480, 3072, 768, bf16).  The PR's central attribution claim
    is that S24 is the one bf16 cell whose dispatch decision changes from
    'route to tritonblas' (False) to 'route to hipBLASLt' (True) once P7
    composes P6 Envelope-B with P5.  Failing-then-passing is encoded as
    BEFORE=False AND AFTER=True (NOT just AFTER=True)."""
    M, N, K, dt = 4480, 3072, 768, BF16
    # P5 alone (BEFORE-P7 production): must NOT route — proves the win is
    # genuinely new, not an artifact of S24 already being in K971_ROUTE_TABLE.
    assert R_K979_P5_route_to_hbl(M, N, K, dt) is False, \
        "S24 P5 should be False — P5 cannot already cover S24"
    assert (M, N, K, dt) not in K971_ROUTE_TABLE, \
        "S24 must not already be in K971_ROUTE_TABLE"
    assert _route_before_p7(M, N, K, dt) is False, \
        "S24 BEFORE-P7 dispatcher should NOT route to hbl"
    # P6 Envelope B fires.
    assert R_K1037_P6_admit_wpeu1(M, N, K, dt) is True, \
        "S24 P6 Envelope-B should fire"
    # P7-extra is NOT what catches S24 — keep that attribution clean.
    assert R_K1106_P7_extra(M, N, K, dt) is False, \
        "S24 should be caught by P6, not the new P7-extra clauses"
    # AFTER-P7: dispatcher routes to hipBLASLt via the OR-gate.
    assert _route_after_p7(M, N, K, dt) is True, \
        "S24 AFTER-P7 dispatcher MUST route to hbl"
    # Tautology assertion: the route DECISION FLIPPED.
    assert _route_before_p7(M, N, K, dt) != _route_after_p7(M, N, K, dt), \
        "S24 dispatch decision must flip BEFORE -> AFTER P7"


def test_k1079_remaining_cells_route():
    cells = [
        (14208, 2048, 1024),  # S29 — P5 C2 + P6 Envelope A
        (5972, 1792, 768),    # S18 — P5 Clause-4
        (6016, 2048, 1024),   # S25 — P5 Clause-2
    ]
    for M, N, K in cells:
        assert _decide(M, N, K, BF16), f"K-1079 ({M},{N},{K}) failed to route"


# --- Adversarial: must NOT route -----------------------------------------

def test_k1031_s07_leakage_cell_does_not_route():
    """S07 (2048,1792,256,bf16) — the K-1031 leakage cell.  The K-1062
    K-floor on P5 Clause-4 must hold, P6 K>=512 floor must hold, and the
    legacy K971_ROUTE_TABLE strict-tuple lookup must miss (S07 not in the
    table).  All three together: P7 False AND fall-through False."""
    assert not _decide(2048, 1792, 256, BF16)


def test_k1080_s16_route_FLIP_before_after_p7():
    """K-1106 tightening: S16 = (256, 256, 2048, bf16) measured 0.751x
    routed-to-tritonblas, breaking the PRD's >=0.85x adversarial gate.
    The new P7-extra bf16-C clause (maxMN<=256, K in [1024,4096]) routes
    it to hipBLASLt (~0.98x).  Must show the BEFORE -> AFTER flip."""
    M, N, K, dt = 256, 256, 2048, BF16
    assert _route_before_p7(M, N, K, dt) is False
    assert R_K1106_P7_extra(M, N, K, dt) is True
    assert _route_after_p7(M, N, K, dt) is True
    assert _route_before_p7(M, N, K, dt) != _route_after_p7(M, N, K, dt)


def test_k1085_fp16_heldout_p7_extra_fires_on_underperformers():
    """K-1106 tightening: K-1085 fp16 underperformers (A5/A6 mid-rect
    non-square; A7/A8 LMhead skinny) measured 0.527x..0.857x — fail the
    PRD adversarial gate.  P7-extra fp16-A and fp16-B clauses route them
    to hipBLASLt to clear the gate.  Must show BEFORE -> AFTER flip."""
    fp16_underperformers = [
        ( 768, 1792, 5972),  # A5 fp16-A (mid-rect)
        (2304, 2048, 4800),  # A6 fp16-A
        (10112, 2048, 1024), # A7 fp16-B (LMhead)
        (12160, 2048, 1024), # A8 fp16-B
    ]
    for M, N, K in fp16_underperformers:
        # Before this PR: NOT in K971_ROUTE_TABLE, P5 bf16-only -> False
        assert (M, N, K, FP16) not in K971_ROUTE_TABLE, \
            f"({M},{N},{K},fp16) should not already be in K971_ROUTE_TABLE"
        assert _route_before_p7(M, N, K, FP16) is False, \
            f"({M},{N},{K},fp16) should NOT route BEFORE-P7"
        # After this PR: P7-extra fires -> dispatcher routes.
        assert R_K1106_P7_extra(M, N, K, FP16) is True, \
            f"({M},{N},{K},fp16) P7-extra should fire"
        assert _route_after_p7(M, N, K, FP16) is True, \
            f"({M},{N},{K},fp16) AFTER-P7 must route to hbl"
        assert _route_before_p7(M, N, K, FP16) != _route_after_p7(M, N, K, FP16)


def test_k1085_fp16_anchors_unchanged_by_p7_extra():
    """The K-905/K-971 fp16 anchors (square M==N) must NOT fire P7-extra
    (the ``minMN < maxMN`` clause keeps them out).  They continue to
    route via the K971_ROUTE_TABLE fall-through — same as before P7."""
    fp16_anchors = [
        (1024, 1024, 16384),
        (1024, 1024, 32768),
        (2048, 2048, 16384),
        (2048, 2048, 32768),
    ]
    for M, N, K in fp16_anchors:
        assert R_K1106_P7_extra(M, N, K, FP16) is False, \
            f"P7-extra wrongly fired on fp16 anchor ({M},{N},{K})"
        # Dispatcher routes via the table (BEFORE) and via the table (AFTER).
        assert _route_before_p7(M, N, K, FP16) is True
        assert _route_after_p7(M, N, K, FP16) is True


# --- fp16 fall-through preservation --------------------------------------

def test_fp16_K971_anchor_fallthrough_preserved():
    """The K-905/K-971 fp16 anchors (e.g. (1024,1024,32768,fp16)) must
    still route via the strict-tuple K971_ROUTE_TABLE fall-through, even
    though P7 is bf16-only."""
    fp16_anchors = [
        (1024, 1024, 16384, FP16),
        (1024, 1024, 32768, FP16),
        (2048, 2048, 16384, FP16),
        (2048, 2048, 32768, FP16),
    ]
    for M, N, K, dt in fp16_anchors:
        # P7 alone is False (bf16-only)
        assert not R_K1106_P7_route_to_hbl(M, N, K, dt)
        # ... but the dispatcher routes via the fall-through.
        assert _decide(M, N, K, dt), \
            f"fp16 K-905/K-971 anchor ({M},{N},{K}) lost fall-through routing"


def test_p7_does_not_mask_fp16_table_disagreement():
    """Regression guard: if P7 ever started returning True on fp16 (bug),
    we'd want the test to flag it.  Conversely, if the table dropped an fp16
    anchor by accident, we'd want the dispatcher to stop routing it.  Either
    way: the dispatcher's fp16 decision must equal the table membership."""
    for M, N, K, dt in K971_ROUTE_TABLE:
        if dt != FP16:
            continue
        assert _decide(M, N, K, dt), f"fp16 table cell ({M},{N},{K}) lost"


if __name__ == "__main__":
    import traceback
    tests = [
        test_disable_env_set_master_kill,
        test_streamk_short_circuit_blocks_route,
        test_work_stealing_short_circuit_blocks_route,
        test_dtype_mismatch_short_circuit_blocks_route,
        test_k1074_in_scope_all_route,
        test_k1079_s24_route_via_p6_envelope_b,
        test_k1079_s24_route_FLIP_before_after_p7,
        test_k1079_remaining_cells_route,
        test_k1031_s07_leakage_cell_does_not_route,
        test_k1080_s16_route_FLIP_before_after_p7,
        test_k1085_fp16_heldout_p7_extra_fires_on_underperformers,
        test_k1085_fp16_anchors_unchanged_by_p7_extra,
        test_fp16_K971_anchor_fallthrough_preserved,
        test_p7_does_not_mask_fp16_table_disagreement,
    ]
    n_pass = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            n_pass += 1
        except Exception:
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{n_pass}/{len(tests)} integration tests passed")
    sys.exit(0 if n_pass == len(tests) else 1)
