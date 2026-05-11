"""K-1700 P29 — N=64 K-COMPLEMENT alias-stack 29-cell route-OUT.

Minimal pin tests: cardinality, structure, sibling-N firewall (N != 64
disjoint), the explicitly dropped bubble cell, P5 bf16 alias coverage,
and adjacent-N regression (N=96 NOT in P29; N=128 still routes via P28).
"""
import pytest

from tritonblas._route_predicate import (
    _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29 as _P29,
    _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30 as _P28,
)


def test_cardinality_29():
    assert len(_P29) == 29


def test_envelope_structure_minus_bubble():
    # Full M×K×dtype grid at N=64 is 30 cells; the bubble cell
    # (2048, 64, 4096, fp16) failed the strict 1.05 gate (CI95-lo=1.045)
    # and is the only intentional omission.
    full = {
        (M, 64, K, dt)
        for M in (2048, 4096, 8192)
        for K in (2048, 4096, 8192, 16384, 32768)
        for dt in ("torch.bfloat16", "torch.float16")
    }
    assert _P29 == full - {(2048, 64, 4096, "torch.float16")}


def test_bubble_cell_explicitly_excluded():
    assert (2048, 64, 4096, "torch.float16") not in _P29


def test_all_cells_have_n64():
    assert {N for (_, N, _, _) in _P29} == {64}


def test_sibling_n_firewall_disjoint():
    # No N != 64 cell may live in P29; this is the structural firewall.
    assert all(N == 64 for (_, N, _, _) in _P29)
    # And P29 must be fully disjoint from the sibling N=128 P28 set.
    assert _P29.isdisjoint(_P28)


def test_p5_bf16_alias_coverage():
    # All 15 bf16 cells in P29 are also caught by the upstream R-K979 P5
    # Clause-3 (minMN ≤ 192 ∧ K ≥ 2048, bf16-only).  P5 fires earlier in
    # the dispatch chain, so the bf16 subset is documentation; the load-
    # bearing portion is the 14 fp16 cells (15 fp16 minus the bubble).
    bf16_cells = {c for c in _P29 if c[3] == "torch.bfloat16"}
    fp16_cells = {c for c in _P29 if c[3] == "torch.float16"}
    assert len(bf16_cells) == 15
    assert len(fp16_cells) == 14  # 15 minus bubble


@pytest.mark.parametrize("M", [2048, 4096, 8192])
@pytest.mark.parametrize("K", [2048, 4096, 8192, 16384, 32768])
def test_adjacent_n96_not_in_p29(M, K):
    # Sibling-N firewall: adjacent N=96 must NOT be admitted.
    assert (M, 96, K, "torch.bfloat16") not in _P29
    assert (M, 96, K, "torch.float16") not in _P29


@pytest.mark.parametrize("M", [2048, 4096, 8192])
@pytest.mark.parametrize("K", [2048, 4096, 8192, 16384, 32768])
def test_adjacent_n128_unaffected_by_p29(M, K):
    # P28 (N=128 alias-stack) still owns its 30 cells; P29 must not have
    # silently shadowed any of them.
    for dt in ("torch.bfloat16", "torch.float16"):
        assert (M, 128, K, dt) in _P28
        assert (M, 128, K, dt) not in _P29
