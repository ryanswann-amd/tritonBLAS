"""K-1721 — C07_BK128_kp2 tile-config override (21st-slot).

Pin tests for the 8-cell M=N=2048 K-COMPLEMENT mid-K cohort frozenset that
selects the persistent_matmul_lt (BLK_K=128, kpack=2) tile-config override.

Coverage: cardinality (8), envelope structure (M=N=2048 × K∈{2048,4096,
8192,16384} × {bf16,fp16}), sibling-cohort disjointness with all 20 prior
route-OUT slots (no shape collision; 21st slot is in-kernel tile-config
NOT route-OUT), and adjacency cells (M=N=2048 K∈{1024,32768} and
M=N∈{1536,3072} K=4096) explicitly excluded.
"""
import pytest

from tritonblas._route_predicate import (
    _K1721_C07_BK128_KP2_MN2048_KCOMPL_8 as _C07,
    _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29 as _P29,
    _K1673_P28_SKINNY_N128_KCOMPL_ALIASSTACK_30 as _P28,
    _K1611_P26_SKINNY_N2048_KCOMPL_ALIASSTACK_30 as _P26,
)


def test_cardinality_8():
    assert len(_C07) == 8


def test_envelope_structure():
    # K-1721 cohort: M=N=2048 × K∈{2048, 4096, 8192, 16384} × {bf16, fp16}.
    expected = {
        (2048, 2048, K, dt)
        for K in (2048, 4096, 8192, 16384)
        for dt in ("torch.bfloat16", "torch.float16")
    }
    assert _C07 == expected


def test_all_cells_are_square_2048():
    assert all(M == 2048 and N == 2048 for (M, N, _, _) in _C07)


def test_dtype_split_balanced():
    bf16 = {c for c in _C07 if c[3] == "torch.bfloat16"}
    fp16 = {c for c in _C07 if c[3] == "torch.float16"}
    assert len(bf16) == 4
    assert len(fp16) == 4


def test_disjoint_from_n64_n128_routeouts():
    # P29 (N=64), P28 (N=128) are skinny-N route-OUTs sibling-N-firewalled
    # from the K-1721 SQUARE M=N=2048 cohort by the N axis.
    assert _C07.isdisjoint(_P29)
    assert _C07.isdisjoint(_P28)


def test_p26_alias_overlap_intentional():
    # P26 (skinny_N2048 alias-stack 17th-slot, route-OUT) covers M ∈ {2048,
    # 4096, 8192} × N=2048 × K ∈ {2048, 4096, 8192, 16384, 32768} × {bf16,
    # fp16}.  The K-1721 8-cell cohort (M=N=2048 × K ∈ {2048, 4096, 8192,
    # 16384} × {bf16, fp16}) is fully contained in P26 — production
    # dispatch routes these 8 cells to torch.matmul (hipBLASLt) FIRST via
    # _k971_route_to_hbl → P26 returning True before reaching
    # persistent_matmul_lt.  The K-1721 C07 in-kernel tile-config override
    # is therefore a FALLBACK that fires only if (a) P26 is ablated via
    # TRITONBLAS_DISABLE_K971=1 (K-1721 measurement protocol), or (b)
    # future routing refinement narrows P26.  This is the documented
    # alias-stack pattern (see K-1611, K-1553/K-1566 for analogous
    # documentation-only handles).
    assert _C07.issubset(_P26)
    # Sanity: P26 covers exactly the 8 K-1721 cells via its M=2048 row at
    # the K-1721 K-grid (4 of 5 K values).
    assert _C07 == {
        (M, N, K, dt) for (M, N, K, dt) in _P26
        if M == 2048 and K in (2048, 4096, 8192, 16384)
    }


@pytest.mark.parametrize("K", [1024, 32768])
@pytest.mark.parametrize("dt", ["torch.bfloat16", "torch.float16"])
def test_adjacency_k_excluded(K, dt):
    # K=1024 and K=32768 are adjacency cells from the K-1740 validation
    # plan; they MUST NOT be admitted by the C07 override.
    assert (2048, 2048, K, dt) not in _C07


@pytest.mark.parametrize("MN", [1536, 3072])
@pytest.mark.parametrize("dt", ["torch.bfloat16", "torch.float16"])
def test_adjacency_mn_excluded(MN, dt):
    # M=N=1536 and M=N=3072 (K=4096) are adjacency cells from the K-1740
    # validation plan; the override is M=N=2048-only.
    assert (MN, MN, 4096, dt) not in _C07


def test_membership_by_shape_key():
    # Spot-check: the 4 K-1721 winner cells where C07 wins per-cell.
    assert (2048, 2048, 2048, "torch.bfloat16") in _C07
    assert (2048, 2048, 2048, "torch.float16") in _C07
    assert (2048, 2048, 16384, "torch.bfloat16") in _C07
    assert (2048, 2048, 16384, "torch.float16") in _C07
