"""
Broad correctness tests for grouped GEMM across a wide variety of shapes.

Covers:
- Homogeneous groups (all same M) across many (G, M, K, N) combos
- Heterogeneous group patterns (decreasing, dominant+small, random, etc.)
- Edge cases (M=1, K=1, unbalanced, single group, etc.)
- Non-power-of-2 / non-aligned dimensions
- Multiple dtypes (fp16, bf16)
- Large group counts (G=16, 32)
- Tiny groups (M=1, 2, 4)

Target: ~200-500 parametrized test cases, each fast enough to run in <1 second.

Run with:
    cd /data0/ryaswann/projects/tritonBLAS
    PYTHONPATH=$(pwd)/include/:$PYTHONPATH \
    HIP_VISIBLE_DEVICES=0 python3 -m pytest tests/test_grouped_gemm_broad.py -x -v --tb=short
"""

import itertools
import random

import pytest
import torch
import sys

sys.path.insert(0, "/data0/ryaswann/projects/tritonBLAS/include/")
import tritonblas


# ---------------------------------------------------------------------------
# Tolerances (match existing test conventions)
# ---------------------------------------------------------------------------

FP16_ATOL = 0.5
FP16_RTOL = 0.01
BF16_ATOL = 1.0
BF16_RTOL = 0.02


def _tols(dtype):
    if dtype == torch.bfloat16:
        return BF16_ATOL, BF16_RTOL
    return FP16_ATOL, FP16_RTOL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_and_check(shapes, dtype=torch.float16):
    """Build inputs, run tritonblas.grouped_gemm, compare with torch.matmul.

    Parameters
    ----------
    shapes : list of (M, N, K) tuples
    dtype  : torch.dtype

    Raises AssertionError on mismatch.
    """
    atol, rtol = _tols(dtype)

    group_a, group_b, refs = [], [], []
    for m, n, k in shapes:
        a = torch.randn(m, k, device="cuda", dtype=dtype)
        b = torch.randn(k, n, device="cuda", dtype=dtype)
        group_a.append(a)
        group_b.append(b)
        refs.append(torch.matmul(a, b))

    results = tritonblas.grouped_gemm(group_a, group_b)

    for i, (res, ref) in enumerate(zip(results, refs)):
        m, n, k = shapes[i]
        torch.testing.assert_close(
            res, ref, atol=atol, rtol=rtol,
            msg=f"Group {i} mismatch (M={m}, N={n}, K={k}, dtype={dtype})",
        )


# ---------------------------------------------------------------------------
# 1. Homogeneous groups (all groups share the same M)
# ---------------------------------------------------------------------------

# Curated (G, M, K, N) combos -- not a full cross product.
_HOMO_PARAMS = []

# Small group counts x representative shapes
for g in [1, 2, 4, 8]:
    for m, k, n in [
        (64, 128, 128),
        (128, 256, 256),
        (256, 512, 512),
        (512, 1024, 1024),
        (1024, 512, 2048),
    ]:
        _HOMO_PARAMS.append((g, m, k, n))

# Larger group counts with smaller shapes to keep runtime low
for g in [12, 16, 32]:
    for m, k, n in [
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
    ]:
        _HOMO_PARAMS.append((g, m, k, n))

# Powers-of-2 M sweep with fixed G, K, N
for m in [16, 32, 64, 128, 256, 512, 1024, 2048]:
    _HOMO_PARAMS.append((4, m, 256, 256))

# Large M with small group count
for m in [2048, 4096]:
    _HOMO_PARAMS.append((2, m, 256, 256))


class TestHomogeneousSizes:
    """All groups in a call share the same (M, N, K)."""

    @pytest.mark.parametrize("g, m, k, n", _HOMO_PARAMS,
                             ids=[f"G{g}_M{m}_K{k}_N{n}" for g, m, k, n in _HOMO_PARAMS])
    def test_homogeneous(self, g, m, k, n):
        shapes = [(m, n, k)] * g
        _run_and_check(shapes)


# ---------------------------------------------------------------------------
# 2. Heterogeneous group patterns
# ---------------------------------------------------------------------------

def _make_decreasing(g, m_start, k, n):
    """Geometrically decreasing M: m_start, m_start//2, ..."""
    ms = []
    m = m_start
    for _ in range(g):
        ms.append(max(m, 1))
        m = max(m // 2, 1)
    return [(mi, n, k) for mi in ms]


def _make_dominant_small(g, m_big, m_small, k, n):
    """One dominant group + (g-1) small groups."""
    shapes = [(m_big, n, k)]
    for _ in range(g - 1):
        shapes.append((m_small, n, k))
    return shapes


def _make_random(g, k, n, seed=42):
    """Random M per group drawn from [8, 1024]."""
    rng = random.Random(seed)
    return [(rng.randint(8, 1024), n, k) for _ in range(g)]


def _make_all_tiny(g, m, k, n):
    """All groups have a tiny M."""
    return [(m, n, k)] * g


_HETERO_PATTERNS = []

# Decreasing
for g, m_start in [(4, 1024), (8, 2048), (5, 512)]:
    shapes = _make_decreasing(g, m_start, k=256, n=256)
    _HETERO_PATTERNS.append(pytest.param(shapes, id=f"decreasing_G{g}_start{m_start}"))

# Dominant + small
for g, m_big, m_small in [(4, 2048, 32), (8, 4096, 16), (3, 1024, 64)]:
    shapes = _make_dominant_small(g, m_big, m_small, k=256, n=256)
    _HETERO_PATTERNS.append(pytest.param(shapes, id=f"dominant_G{g}_{m_big}vs{m_small}"))

# Random
for g, seed in [(4, 1), (8, 2), (16, 3)]:
    shapes = _make_random(g, k=256, n=256, seed=seed)
    ms = [s[0] for s in shapes]
    _HETERO_PATTERNS.append(pytest.param(shapes, id=f"random_G{g}_seed{seed}_ms{'_'.join(map(str, ms[:4]))}"))

# Increasing
for g, m_start in [(4, 32), (8, 16)]:
    shapes = [(m_start * (2 ** i), 256, 256) for i in range(g)]
    _HETERO_PATTERNS.append(pytest.param(shapes, id=f"increasing_G{g}_start{m_start}"))

# Alternating large/small
for g in [4, 8]:
    shapes = [(2048 if i % 2 == 0 else 32, 256, 256) for i in range(g)]
    _HETERO_PATTERNS.append(pytest.param(shapes, id=f"alternating_G{g}"))

# All tiny
for m_tiny in [1, 2, 4]:
    shapes = _make_all_tiny(8, m_tiny, k=128, n=128)
    _HETERO_PATTERNS.append(pytest.param(shapes, id=f"all_tiny_G8_M{m_tiny}"))


class TestHeterogeneousPatterns:
    """Groups with different M dimensions in a single call."""

    @pytest.mark.parametrize("shapes", _HETERO_PATTERNS)
    def test_heterogeneous(self, shapes):
        _run_and_check(shapes)


# ---------------------------------------------------------------------------
# 3. Edge cases
# ---------------------------------------------------------------------------

_EDGE_CASES = [
    # Single row per group
    pytest.param([(1, 128, 64)] * 4, id="M1_G4"),
    pytest.param([(1, 256, 256)], id="M1_G1"),
    pytest.param([(1, 64, 64)] * 16, id="M1_G16"),

    # Single group
    pytest.param([(256, 256, 128)], id="single_group_256"),
    pytest.param([(1024, 1024, 512)], id="single_group_1024"),
    pytest.param([(1, 64, 64)], id="single_group_M1"),

    # Very unbalanced (one big, rest tiny)
    pytest.param(
        [(4096, 256, 256)] + [(16, 256, 256)] * 7,
        id="unbalanced_4096_vs_16_G8",
    ),
    pytest.param(
        [(2048, 512, 256)] + [(1, 512, 256)] * 3,
        id="unbalanced_2048_vs_1_G4",
    ),

    # Large group count, tiny sizes
    pytest.param([(8, 64, 64)] * 32, id="G32_M8"),
    pytest.param([(16, 64, 64)] * 32, id="G32_M16"),
    pytest.param([(4, 32, 32)] * 16, id="G16_M4_small_KN"),

    # All large
    pytest.param([(4096, 256, 256)] * 2, id="all_large_M4096_G2"),
    pytest.param([(2048, 512, 512)] * 4, id="all_large_M2048_G4"),

    # M < typical tile sizes (BLK_M is usually 64 or 128)
    pytest.param([(8, 256, 128)] * 4, id="M_below_tile_8"),
    pytest.param([(16, 256, 128)] * 4, id="M_below_tile_16"),
    pytest.param([(32, 256, 128)] * 4, id="M_below_tile_32"),

    # Mix of M=1 and normal
    pytest.param(
        [(1, 256, 128), (256, 256, 128), (1, 256, 128), (512, 256, 128)],
        id="mixed_M1_and_normal",
    ),

    # G=3, 5 (odd group counts)
    pytest.param([(128, 128, 64)] * 3, id="G3"),
    pytest.param([(128, 128, 64)] * 5, id="G5"),

    # Extreme K (very deep reduction)
    pytest.param([(64, 64, 2048)] * 4, id="deep_K_2048"),

    # K=64 (minimal K)
    pytest.param([(256, 256, 64)] * 4, id="small_K_64"),
]


class TestEdgeCases:
    """Tricky and boundary conditions."""

    @pytest.mark.parametrize("shapes", _EDGE_CASES)
    def test_edge(self, shapes):
        _run_and_check(shapes)


# ---------------------------------------------------------------------------
# 4. Non-aligned / non-power-of-2 dimensions
# ---------------------------------------------------------------------------

_NON_ALIGNED_M = [17, 33, 65, 100, 200, 300, 500, 768, 1000, 1500]

_NON_ALIGNED_PARAMS = []

# Non-aligned M with aligned K, N
for m in _NON_ALIGNED_M:
    _NON_ALIGNED_PARAMS.append(pytest.param(
        [(m, 256, 256)] * 4,
        id=f"M{m}_K256_N256_G4",
    ))

# Non-aligned M with non-aligned K and N
_NON_ALIGNED_KN = [
    (100, 200, 300),
    (33, 65, 100),
    (768, 500, 1000),
    (17, 127, 65),
    (200, 300, 500),
]
for m, k, n in _NON_ALIGNED_KN:
    _NON_ALIGNED_PARAMS.append(pytest.param(
        [(m, n, k)] * 2,
        id=f"M{m}_K{k}_N{n}_G2",
    ))

# Mixed aligned and non-aligned in same call
_NON_ALIGNED_PARAMS.append(pytest.param(
    [(128, 256, 256), (100, 200, 300), (64, 128, 64), (33, 65, 100)],
    id="mixed_aligned_nonaligned",
))
_NON_ALIGNED_PARAMS.append(pytest.param(
    [(256, 256, 256), (17, 256, 256), (768, 256, 256), (1500, 256, 256)],
    id="mixed_M_aligned_nonaligned",
))


class TestNonAligned:
    """Non-power-of-2 and non-tile-aligned dimensions."""

    @pytest.mark.parametrize("shapes", _NON_ALIGNED_PARAMS)
    def test_non_aligned(self, shapes):
        _run_and_check(shapes)


# ---------------------------------------------------------------------------
# 5. Dtype coverage
# ---------------------------------------------------------------------------

_DTYPE_SHAPES = [
    [(128, 128, 64)] * 4,
    [(256, 256, 128)] * 2,
    [(64, 256, 128), (128, 128, 64), (256, 64, 128)],
    [(1, 128, 64)] * 4,
    [(512, 512, 256)],
    [(100, 200, 300)] * 2,
]


class TestDtypes:
    """Verify correctness under fp16 and bf16 for representative shapes."""

    @pytest.mark.parametrize(
        "shapes",
        _DTYPE_SHAPES,
        ids=[f"shapes{i}" for i in range(len(_DTYPE_SHAPES))],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16],
                             ids=["fp16", "bf16"])
    def test_dtype(self, shapes, dtype):
        _run_and_check(shapes, dtype=dtype)


# ---------------------------------------------------------------------------
# 6. Large group counts
# ---------------------------------------------------------------------------

_LARGE_GROUP_PARAMS = []

# G=16
for m, k, n in [(32, 64, 64), (64, 128, 128), (128, 256, 256), (256, 128, 128)]:
    _LARGE_GROUP_PARAMS.append(pytest.param(
        [(m, n, k)] * 16,
        id=f"G16_M{m}_K{k}_N{n}",
    ))

# G=32
for m, k, n in [(16, 64, 64), (32, 64, 64), (64, 128, 128)]:
    _LARGE_GROUP_PARAMS.append(pytest.param(
        [(m, n, k)] * 32,
        id=f"G32_M{m}_K{k}_N{n}",
    ))

# G=16 heterogeneous
_LARGE_GROUP_PARAMS.append(pytest.param(
    _make_random(16, k=128, n=128, seed=100),
    id="G16_random",
))
_LARGE_GROUP_PARAMS.append(pytest.param(
    _make_decreasing(16, m_start=2048, k=128, n=128),
    id="G16_decreasing",
))
_LARGE_GROUP_PARAMS.append(pytest.param(
    _make_dominant_small(16, m_big=2048, m_small=16, k=128, n=128),
    id="G16_dominant",
))

# G=32 heterogeneous
_LARGE_GROUP_PARAMS.append(pytest.param(
    _make_random(32, k=64, n=64, seed=200),
    id="G32_random",
))


class TestLargeGroupCounts:
    """G=16 and G=32 with various size patterns."""

    @pytest.mark.parametrize("shapes", _LARGE_GROUP_PARAMS)
    def test_large_groups(self, shapes):
        _run_and_check(shapes)


# ---------------------------------------------------------------------------
# 7. Tiny groups (M=1, 2, 4)
# ---------------------------------------------------------------------------

_TINY_PARAMS = []

for m_tiny in [1, 2, 4]:
    for g in [1, 2, 4, 8, 16]:
        for k, n in [(64, 64), (128, 128), (256, 256)]:
            _TINY_PARAMS.append(pytest.param(
                [(m_tiny, n, k)] * g,
                id=f"M{m_tiny}_G{g}_K{k}_N{n}",
            ))


class TestTinyGroups:
    """Very small M per group (M=1, 2, 4) across group counts and K/N."""

    @pytest.mark.parametrize("shapes", _TINY_PARAMS)
    def test_tiny(self, shapes):
        _run_and_check(shapes)


# ---------------------------------------------------------------------------
# 8. K dimension sweep
# ---------------------------------------------------------------------------

_K_SWEEP_PARAMS = []

for k in [64, 128, 256, 512, 1024, 2048]:
    _K_SWEEP_PARAMS.append(pytest.param(
        [(128, 128, k)] * 4,
        id=f"K{k}_M128_N128_G4",
    ))

# K sweep with non-aligned M
for k in [64, 256, 1024]:
    _K_SWEEP_PARAMS.append(pytest.param(
        [(100, 200, k)] * 2,
        id=f"K{k}_M100_N200_G2",
    ))


class TestKDimensionSweep:
    """Sweep K from small to large with fixed M and N."""

    @pytest.mark.parametrize("shapes", _K_SWEEP_PARAMS)
    def test_k_sweep(self, shapes):
        _run_and_check(shapes)


# ---------------------------------------------------------------------------
# 9. N dimension sweep
# ---------------------------------------------------------------------------

_N_SWEEP_PARAMS = []

for n in [64, 128, 256, 512, 1024, 2048, 4096]:
    _N_SWEEP_PARAMS.append(pytest.param(
        [(128, n, 256)] * 4,
        id=f"N{n}_M128_K256_G4",
    ))


class TestNDimensionSweep:
    """Sweep N from small to large with fixed M and K."""

    @pytest.mark.parametrize("shapes", _N_SWEEP_PARAMS)
    def test_n_sweep(self, shapes):
        _run_and_check(shapes)


# ---------------------------------------------------------------------------
# 10. Group count sweep (G=1..32) with fixed shape
# ---------------------------------------------------------------------------

_G_SWEEP_PARAMS = [1, 2, 3, 4, 5, 8, 12, 16, 32]


class TestGroupCountSweep:
    """Sweep group count from 1 to 32 with fixed per-group shape."""

    @pytest.mark.parametrize("g", _G_SWEEP_PARAMS, ids=[f"G{g}" for g in _G_SWEEP_PARAMS])
    def test_group_count(self, g):
        shapes = [(128, 128, 128)] * g
        _run_and_check(shapes)


# ---------------------------------------------------------------------------
# 11. Stress combos: (M, K, N) full coverage matrix (curated subset)
# ---------------------------------------------------------------------------

_STRESS_MKN = []

# Pick a representative subset from the full M x K x N space to avoid
# combinatorial explosion while still covering the boundaries.
_STRESS_M = [1, 16, 64, 128, 256, 512, 1024]
_STRESS_K = [64, 256, 1024]
_STRESS_N = [64, 256, 1024]

for m in _STRESS_M:
    for k in _STRESS_K:
        for n in _STRESS_N:
            _STRESS_MKN.append(pytest.param(
                [(m, n, k)] * 2,
                id=f"M{m}_K{k}_N{n}_G2",
            ))


class TestStressMKN:
    """Curated cross-product of M, K, N dimensions with G=2."""

    @pytest.mark.parametrize("shapes", _STRESS_MKN)
    def test_stress(self, shapes):
        _run_and_check(shapes)
