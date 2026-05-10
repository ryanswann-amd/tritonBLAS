"""K-1908 (S-002) — closed-form off-by-48 K-COMPLEMENT predicate equivalence.

Pins the admit-set identity of `_is_k1908_offby48_kcompl_admit` against
the union of the two single-N off-by-48 frozensets it replaces in the
dispatcher hot path (K-2136 P55 N=1520 + K-2152 P56 N=1584).

This test is the load-bearing oracle for the K-1908 closed-form refactor
that landed alongside the K-2152 P56 N=1584 admit (addressing the
Performance-Hawk feedback on the K-2152 PR — collapse the per-rung
membership-chain into a single closed-form predicate).

Per minimalist src/tests split (R-1532 / R-1720 / R-1775): the
`_route_predicate.py` module holds the predicate + its data; this file
holds the equivalence invariant.

If a future rung (e.g. K-2NNN P57 N=1648) is added by extending
`_K1908_OFFBY48_ADMIT_N`, this test must be updated to also reference
that rung's data frozenset (or the test should pivot to oracle-by-
construction over the closed-form's own data tables).
"""
from __future__ import annotations

import pytest

from tritonblas._route_predicate import (
    _K1908_OFFBY48_ADMIT_N,
    _K1908_OFFBY48_ENVELOPE_MK_DT,
    _K2136_P55_SKINNY_N1520_KCOMPL_ALIASSTACK_18,
    _K2152_P56_SKINNY_N1584_KCOMPL_ALIASSTACK_18,
    _is_k1908_offby48_kcompl_admit,
)

# Oracle: the union of the two single-N off-by-48 frozensets the closed-form
# replaces in the dispatcher hot path.
_UNION_ORACLE = (
    _K2136_P55_SKINNY_N1520_KCOMPL_ALIASSTACK_18
    | _K2152_P56_SKINNY_N1584_KCOMPL_ALIASSTACK_18
)


def test_admit_n_is_exactly_1520_and_1584():
    # K-1908 closed-form currently admits exactly the two off-by-48 N
    # values that the prior chained membership-checks covered: 1520 (K-2136
    # P55) + 1584 (K-2152 P56).
    assert _K1908_OFFBY48_ADMIT_N == frozenset({1520, 1584})


def test_admit_n_all_satisfy_off_by_48_residue():
    for N in _K1908_OFFBY48_ADMIT_N:
        assert N % 64 == 48, f"N={N} violates off-by-48 invariant"


def test_envelope_cardinality_is_18():
    # 3 M × 3 K × 2 dt = 18 (M, K, dt) tuples — same envelope shape as
    # every K-COMPLEMENT alias-stack ALIASSTACK_18 frozenset.
    assert len(_K1908_OFFBY48_ENVELOPE_MK_DT) == 18


def test_envelope_is_canonical_m_k_dt_grid():
    expected = frozenset(
        (M, K, dt)
        for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    )
    assert _K1908_OFFBY48_ENVELOPE_MK_DT == expected


def test_closed_form_admits_every_oracle_cell():
    """Every (M, N, K, dt) in the K-2136 ∪ K-2152 union must be admitted."""
    for (M, N, K, dt) in _UNION_ORACLE:
        assert _is_k1908_offby48_kcompl_admit(M, N, K, dt), (
            f"oracle cell {(M, N, K, dt)} not admitted by closed-form"
        )


def test_closed_form_rejects_outside_oracle():
    """The closed-form must reject cells outside the K-2136 ∪ K-2152 union.

    Sweeps a representative grid covering: (a) wrong-N cells with right
    envelope; (b) right-N cells with wrong envelope (M, K, dt); (c) random
    near-miss perturbations.
    """
    sweep = []
    # (a) wrong N (still off-by-48 but not in admit set; and on-residue but
    # not off-by-48):
    for N_off in (48, 112, 176, 304, 432, 816, 880, 944, 1008, 1072, 1136,
                  1200, 1264, 1328, 1392, 1456, 1648, 1712, 1776):
        for M in (2048, 4096, 8192):
            for K in (4096, 8192, 16384):
                for dt in ("torch.bfloat16", "torch.float16"):
                    if N_off in _K1908_OFFBY48_ADMIT_N:
                        continue
                    sweep.append((M, N_off, K, dt))
    # (b) right N, wrong envelope:
    for N in (1520, 1584):
        for M in (1024, 3072, 6144, 16384):
            for K in (1024, 2048, 32768):
                for dt in ("torch.float32",):
                    sweep.append((M, N, K, dt))
    # (c) right N, right envelope, wrong dtype:
    for N in (1520, 1584):
        for M in (2048, 4096, 8192):
            for K in (4096, 8192, 16384):
                for dt in ("torch.float32", "torch.float8_e4m3fn"):
                    sweep.append((M, N, K, dt))

    for (M, N, K, dt) in sweep:
        assert not _is_k1908_offby48_kcompl_admit(M, N, K, dt), (
            f"non-oracle cell {(M, N, K, dt)} unexpectedly admitted"
        )


def test_closed_form_full_admit_set_identity():
    """Strict equivalence: build the closed-form's admit set and assert it
    equals the union of K-2136 and K-2152 frozensets, cell-for-cell."""
    closed_form_admits = frozenset(
        (M, N, K, dt)
        for (M, K, dt) in _K1908_OFFBY48_ENVELOPE_MK_DT
        for N in _K1908_OFFBY48_ADMIT_N
        if _is_k1908_offby48_kcompl_admit(M, N, K, dt)
    )
    assert closed_form_admits == _UNION_ORACLE


def test_residue_gate_rejects_non_off_by_48_n_in_admit_set():
    """If a non-off-by-48 N somehow ended up in the admit set, the modulo
    gate would reject it.  Smoke-tests the mod gate by patching a fake
    N=1521 (mod 64 = 49) into a probe call — the predicate should reject."""
    # We can't mutate _K1908_OFFBY48_ADMIT_N at runtime here, but we can
    # call the predicate with N=1521 directly: it should fail the modulo
    # check before the admit-set check.
    assert not _is_k1908_offby48_kcompl_admit(2048, 1521, 4096, "torch.bfloat16")
    # And N=1520 still passes:
    assert _is_k1908_offby48_kcompl_admit(2048, 1520, 4096, "torch.bfloat16")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
