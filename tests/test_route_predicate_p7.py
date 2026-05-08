"""K-1106 P7 OR-gate unit tests.

Verifies R_K1106_P7_route_to_hbl semantics on:
  - K-1074 8-cell in-scope (S26, S31-S35, S37, S39)
  - K-1079 5-cell in-scope (S24, S29, S18, S25 + S07 K-1031 leakage)
  - K-1080 4-cell Occupancy adversarial (S16, S26, S30, S37) + S07 ref
  - K-1085 8-cell held-out adversarial (all fp16 -> bf16-only gate kills)

Run with: python -m pytest tests/test_route_predicate_p7.py -x
or stand-alone: python tests/test_route_predicate_p7.py
"""
from __future__ import annotations
import sys
import os

# Allow running from repo root without install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "include"))

from tritonblas._route_predicate import (  # noqa: E402
    R_K979_P5_route_to_hbl,
    R_K1037_P6_admit_wpeu1,
    R_K1106_P7_extra,
    R_K1106_P7_route_to_hbl,
)


BF16 = "torch.bfloat16"
FP16 = "torch.float16"


# (label, M, N, K, dtype, expect_p5, expect_p6, expect_p7)
K1074_CELLS = [
    ("S26", 8064,  2048, 1024, BF16, True,  False, True),
    ("S31", 18304, 2048, 1024, BF16, True,  False, True),
    ("S32", 20352, 2048, 1024, BF16, True,  False, True),
    ("S33", 22400, 2048, 1024, BF16, True,  False, True),
    ("S34", 24448, 2048, 1024, BF16, True,  False, True),
    ("S35", 26496, 2048, 1024, BF16, True,  False, True),
    ("S37", 25600, 2048,  256, BF16, True,  False, True),
    ("S39", 49152, 2048,  256, BF16, True,  False, True),
]

K1079_CELLS = [
    ("S24",  4480, 3072,  768, BF16, False, True,  True),   # P6 Envelope B
    ("S29", 14208, 2048, 1024, BF16, True,  True,  True),   # P5 C2 + P6 Env A
    ("S18",  5972, 1792,  768, BF16, True,  False, True),   # P5 C4
    ("S25",  6016, 2048, 1024, BF16, True,  False, True),   # P5 C2
    ("S07",  2048, 1792,  256, BF16, False, False, False),  # K-1031 leakage ctrl
]

# K-1080 Occupancy neighbours adversarial.  K-1106 tightening: S16
# (256,256,2048,bf16) measured 0.751x routed-to-tritonblas — the small-square
# bf16 mid-K bf16-C clause now routes it to hipBLASLt (~0.98x) to clear the
# PRD's absolute >=0.85x adversarial gate.  S30 already fires P5 Clause-2.
K1080_ADVERSARIAL = [
    ("S16",   256,  256, 2048, BF16, False, False, True),   # K-1106 P7-extra bf16-C
    ("S30", 16256, 2048, 1024, BF16, True,  False, True),   # already P5+
    ("S07",  2048, 1792,  256, BF16, False, False, False),  # K-1031 leakage
]

# K-1085 fp16 held-out.  K-1106 tightening: A5/A6 (mid-rect non-square)
# fire P7-extra fp16-A; A7/A8 (LMhead skinny) fire P7-extra fp16-B.
# A1-A4 are square fp16 K-905/K-971 anchors — Clause-1's ``minMN<maxMN``
# bound keeps them False so they route via K971_ROUTE_TABLE strict-tuple
# fall-through (preserves K-905/K-971 anchor handling end-to-end).
# Tuple: (M, N, K, dtype, expect_p7_extra)
K1085_HELDOUT_FP16 = [
    ( 1024, 1024, 16384, FP16, False),  # K-905 anchor (square -> fall-through)
    ( 1024, 1024, 32768, FP16, False),  # K-971 anchor
    ( 2048, 2048, 16384, FP16, False),  # K-971 anchor
    ( 2048, 2048, 32768, FP16, False),  # K-971 anchor
    (  768, 1792,  5972, FP16, True),   # A5 mid-rect non-square
    ( 2304, 2048,  4800, FP16, True),   # A6 mid-rect non-square
    (10112, 2048,  1024, FP16, True),   # A7 LMhead skinny
    (12160, 2048,  1024, FP16, True),   # A8 LMhead skinny
]


def _check(cells, label):
    failures = []
    for cell in cells:
        name, M, N, K, dtype, ep5, ep6, ep7 = cell
        gp5 = R_K979_P5_route_to_hbl(M, N, K, dtype)
        gp6 = R_K1037_P6_admit_wpeu1(M, N, K, dtype)
        gx  = R_K1106_P7_extra(M, N, K, dtype)
        gp7 = R_K1106_P7_route_to_hbl(M, N, K, dtype)
        # P7 OR-gate invariant — now includes P7-extra
        assert gp7 == (gp5 or gp6 or gx), \
            f"{label}/{name}: P7 != P5 OR P6 OR P7-extra"
        if (gp5, gp6, gp7) != (ep5, ep6, ep7):
            failures.append((label, name, (gp5, gp6, gp7), (ep5, ep6, ep7)))
    return failures


def test_k1074_in_scope():
    fails = _check(K1074_CELLS, "K1074")
    assert not fails, f"K-1074 cell mismatches: {fails}"


def test_k1079_in_scope():
    fails = _check(K1079_CELLS, "K1079")
    assert not fails, f"K-1079 cell mismatches: {fails}"


def test_k1080_occupancy_neighbours_no_regression():
    fails = _check(K1080_ADVERSARIAL, "K1080")
    assert not fails, f"K-1080 adversarial mismatches: {fails}"


def test_k1085_fp16_heldout_p7_extra_fires_on_underperformers():
    """K-1106 tightening: held-out fp16 underperformers (A5-A8) must fire
    P7-extra to clear the PRD's absolute >=0.85x adversarial gate; the
    square K-905/K-971 fp16 anchors (A1-A4) must NOT fire P7 (they are
    handled by the K971_ROUTE_TABLE strict-tuple fall-through)."""
    for M, N, K, dt, expect_extra in K1085_HELDOUT_FP16:
        gx = R_K1106_P7_extra(M, N, K, dt)
        gp7 = R_K1106_P7_route_to_hbl(M, N, K, dt)
        assert gx == expect_extra, \
            f"K-1085 fp16 ({M},{N},{K}) P7-extra={gx} expected {expect_extra}"
        # P5 and P6 are bf16-only -> P7 == P7-extra on every fp16 cell.
        assert gp7 == expect_extra, \
            f"K-1085 fp16 ({M},{N},{K}) P7={gp7} expected {expect_extra}"


def test_p7_or_gate_invariant_random_grid():
    """P7 must equal (P5 OR P6 OR P7-extra) on a structural grid."""
    for M in (256, 1024, 2048, 4480, 5972, 8064, 14208, 25600, 49152):
        for N in (256, 1792, 2048, 3072):
            for K in (200, 256, 512, 768, 1024, 2048, 4096):
                for dt in (BF16, FP16):
                    p5 = R_K979_P5_route_to_hbl(M, N, K, dt)
                    p6 = R_K1037_P6_admit_wpeu1(M, N, K, dt)
                    px = R_K1106_P7_extra(M, N, K, dt)
                    p7 = R_K1106_P7_route_to_hbl(M, N, K, dt)
                    assert p7 == (p5 or p6 or px), \
                        f"P7 OR-gate broken at ({M},{N},{K},{dt})"


def test_k1031_s07_leakage_immunity():
    """S07 (2048,1792,256,bf16) — K-1031 leakage cell. ALL FOUR
    predicates (P5, P6, P7-extra, P7) must return False so the
    K-1031 leakage cell continues to route via the tritonblas path."""
    assert not R_K979_P5_route_to_hbl(2048, 1792, 256, BF16)
    assert not R_K1037_P6_admit_wpeu1(2048, 1792, 256, BF16)
    assert not R_K1106_P7_extra(2048, 1792, 256, BF16)
    assert not R_K1106_P7_route_to_hbl(2048, 1792, 256, BF16)


if __name__ == "__main__":
    # Stand-alone runner
    import traceback
    tests = [
        test_k1074_in_scope,
        test_k1079_in_scope,
        test_k1080_occupancy_neighbours_no_regression,
        test_k1085_fp16_heldout_p7_extra_fires_on_underperformers,
        test_p7_or_gate_invariant_random_grid,
        test_k1031_s07_leakage_immunity,
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
    print(f"\n{n_pass}/{len(tests)} tests passed")
    sys.exit(0 if n_pass == len(tests) else 1)
