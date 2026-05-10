"""K-2067 P49 (S-002) — 39th-slot N=1200 K-COMPLEMENT verified-winner alias-stack tests.

Source measurement: K-2067 paired n=30 HIP-graph hot-cache MI300X / gfx942
(vs the live post-K-1922 oracle, HEAD = 0024a71) over the 18-cell sub-cohort
M ∈ {2048, 4096, 8192} × N=1200 × K ∈ {4096, 8192, 16384} × {bf16, fp16}.

K-2067 is the 7th confirmed wave-misaligned rung in the off-by-48 (mod 64)
residue family above K-1978 P45 N=816, K-1990/K-1994 P45 N=880, K-2010/K-2028
P46 N=944, K-2031/K-2036 P47 N=1008, K-2055 P48 N=1136 (and the implicit N=1072
5th rung).  1200 = 1136 + 64 — next 64-stride rung in the off-by-48 class.

Mechanism (K-913 §3 / R-K1673 dtype-invariance + R-1811 wave-misalignment):
BLOCK_N=128 packs N=1200 into 9.375 BLOCK_N tiles per N-row (1200 mod 128
= 48 = 9/10-tile tail).  1200 mod 64 = 48 places it in the off-by-48
wave-misaligned class — the same SIMD-lane mismatch as N=816, N=880, N=944,
N=1008, N=1072, N=1136.  N=1200 has the same `mod 128 = 48` residue as
N=816, N=944, and N=1072 (a mod-128 sub-family of the off-by-48 class),
distinct from N=880, N=1008, N=1136 (which are mod-128=112).

R-1532 / R-1720 / R-1775 minimalist split: src holds the data literal,
tests hold all invariants.
"""
import pytest
from include.tritonblas._route_predicate import (
    _K2067_P49_SKINNY_N1200_KCOMPL_ALIASSTACK_18,
    _K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18,
)


_S = _K2067_P49_SKINNY_N1200_KCOMPL_ALIASSTACK_18


# ---- cardinality / shape ----

def test_cardinality_18():
    assert len(_S) == 18


def test_n_is_1200_only():
    assert {t[1] for t in _S} == {1200}


def test_m_coverage():
    assert {t[0] for t in _S} == {2048, 4096, 8192}


def test_k_coverage():
    assert {t[2] for t in _S} == {4096, 8192, 16384}


def test_dtype_coverage():
    assert {t[3] for t in _S} == {"torch.bfloat16", "torch.float16"}


def test_full_cartesian_product():
    expected = {
        (M, 1200, K, dt)
        for M in (2048, 4096, 8192)
        for K in (4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    }
    assert _S == expected


# ---- modular invariants ----

def test_off_by_48_mod_64():
    """Every cell's N must be in the (mod 64 == 48) wave-misaligned class."""
    for M, N, K, dt in _S:
        assert N % 64 == 48, f"N={N} not in mod-64==48 residue class"


def test_mod_128_48_subfamily():
    """N=1200 is in the mod-128=48 sub-family (along with N=816, 944, 1072)."""
    for M, N, K, dt in _S:
        assert N % 128 == 48, f"N={N} not in mod-128==48 sub-family"


def test_block_n_128_tail_geometry():
    """N=1200 packs into 9 full BLOCK_N=128 tiles + 1 BN=48 tail (9/10-tile tail)."""
    for M, N, K, dt in _S:
        full = N // 128
        tail = N % 128
        assert full == 9
        assert tail == 48


# ---- sibling-N firewall ----

def test_disjoint_from_p40():
    """No (M,N,K,dt) tuple shared with K-1922 P40 N=544 alias-stack."""
    assert _S.isdisjoint(_K1922_P40_SKINNY_N544_KCOMPL_ALIASSTACK_18)


def test_disjoint_from_other_n_via_n_axis():
    """Sibling-N firewall: no shared N with any prior K-COMPLEMENT slot."""
    prior_ns = {544, 816, 880, 944, 1008, 1072, 1136}  # all prior slots
    n_axis = {t[1] for t in _S}
    assert n_axis.isdisjoint(prior_ns)
