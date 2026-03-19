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


# ---------------------------------------------------------------------------
# 12. Large group counts (stress dispatch) — NEW
# ---------------------------------------------------------------------------

_LARGE_GROUP_COUNTS2_PARAMS = [
    # G=32, 64 with M=64 each (stress the group dispatch)
    pytest.param([(64, 128, 128)] * 32, id="G32_M64"),
    pytest.param([(64, 128, 128)] * 64, id="G64_M64"),
    # G=16 with M=1 each (extreme tiny per-group)
    pytest.param([(1, 128, 128)] * 16, id="G16_M1_extreme_tiny"),
    # G=24 with mixed M (non-power-of-2 group count)
    pytest.param(
        [(32 + i * 16, 128, 128) for i in range(24)],
        id="G24_mixed_M_nonpow2",
    ),
]


class TestLargeGroupCounts2:
    """Stress test with many groups: G=32, 64, non-power-of-2 group counts."""

    @pytest.mark.parametrize("shapes", _LARGE_GROUP_COUNTS2_PARAMS)
    def test_large_group_counts(self, shapes):
        _run_and_check(shapes)


# ---------------------------------------------------------------------------
# 13. Large problem sizes — NEW
# ---------------------------------------------------------------------------

_LARGE_PROBLEM_PARAMS = [
    # G=2, M=4096, K=4096, N=4096
    pytest.param([(4096, 4096, 4096)] * 2, id="G2_4096x4096x4096"),
    # G=4, M=2048, K=8192, N=2048
    pytest.param([(2048, 2048, 8192)] * 4, id="G4_2048x8192x2048"),
    # G=1, M=8192, K=8192, N=8192 (single huge GEMM through grouped path)
    pytest.param([(8192, 8192, 8192)], id="G1_8192x8192x8192"),
]


class TestLargeProblemSizes:
    """Big GEMMs to stress memory and compute."""

    @pytest.mark.parametrize("shapes", _LARGE_PROBLEM_PARAMS)
    def test_large_problem(self, shapes):
        _run_and_check(shapes)


# ---------------------------------------------------------------------------
# 14. Prime number dimensions — NEW
# ---------------------------------------------------------------------------

_PRIME_PARAMS = [
    # M=127, K=127, N=127 (all prime)
    pytest.param([(127, 127, 127)] * 2, id="G2_127x127x127"),
    # M=257, K=509, N=131 (all prime)
    pytest.param([(257, 131, 509)] * 2, id="G2_257x509x131"),
    # G=3 with M=97, K=211, N=163
    pytest.param([(97, 163, 211)] * 3, id="G3_97x211x163"),
]


class TestPrimeNumbers:
    """Non-aligned dimensions using prime numbers."""

    @pytest.mark.parametrize("shapes", _PRIME_PARAMS)
    def test_prime(self, shapes):
        _run_and_check(shapes)


# ---------------------------------------------------------------------------
# 15. BF16 dtype coverage — NEW
# ---------------------------------------------------------------------------

_BF16_PARAMS = [
    # Same shapes as fp16 tests but with bf16
    pytest.param([(128, 128, 64)] * 4, id="G4_128x64x128"),
    pytest.param([(256, 256, 128)] * 2, id="G2_256x128x256"),
    pytest.param([(64, 256, 128), (128, 128, 64), (256, 64, 128)], id="hetero_3group"),
    # G=4, M=512, K=1024, N=512
    pytest.param([(512, 512, 1024)] * 4, id="G4_512x1024x512"),
    # G=8, M=256, K=256, N=256
    pytest.param([(256, 256, 256)] * 8, id="G8_256x256x256"),
    # Heterogeneous G=4 with [1024, 512, 256, 128]
    pytest.param(
        [(1024, 256, 256), (512, 256, 256), (256, 256, 256), (128, 256, 256)],
        id="G4_hetero_1024_512_256_128",
    ),
]


class TestBF16:
    """BF16 dtype coverage with various shapes."""

    @pytest.mark.parametrize("shapes", _BF16_PARAMS)
    def test_bf16(self, shapes):
        _run_and_check(shapes, dtype=torch.bfloat16)


# ---------------------------------------------------------------------------
# 16. Extreme skew — NEW
# ---------------------------------------------------------------------------

_EXTREME_SKEW_PARAMS = [
    # [8192, 1, 1, 1] — one huge, rest tiny
    pytest.param(
        [(8192, 256, 256), (1, 256, 256), (1, 256, 256), (1, 256, 256)],
        id="skew_8192_1_1_1",
    ),
    # [4096, 2, 2, 2, 2, 2, 2, 2] — one large, 7 tiny
    pytest.param(
        [(4096, 256, 256)] + [(2, 256, 256)] * 7,
        id="skew_4096_seven_2s",
    ),
    # [1]*32 — 32 groups of M=1
    pytest.param(
        [(1, 128, 128)] * 32,
        id="skew_32_groups_M1",
    ),
]


class TestExtremeSkew:
    """Wildly unbalanced group sizes."""

    @pytest.mark.parametrize("shapes", _EXTREME_SKEW_PARAMS)
    def test_extreme_skew(self, shapes):
        _run_and_check(shapes)


# ---------------------------------------------------------------------------
# 17. Output correctness with known patterns — NEW
# ---------------------------------------------------------------------------

def _run_and_check_with_blk(shapes, dtype=torch.float16, BLK_M=None, BLK_N=None, BLK_K=None):
    """Like _run_and_check but passes explicit block size overrides."""
    atol, rtol = _tols(dtype)

    group_a, group_b, refs = [], [], []
    for m, n, k in shapes:
        a = torch.randn(m, k, device="cuda", dtype=dtype)
        b = torch.randn(k, n, device="cuda", dtype=dtype)
        group_a.append(a)
        group_b.append(b)
        refs.append(torch.matmul(a, b))

    results = tritonblas.grouped_gemm(group_a, group_b, BLK_M=BLK_M, BLK_N=BLK_N, BLK_K=BLK_K)

    for i, (res, ref) in enumerate(zip(results, refs)):
        m, n, k = shapes[i]
        torch.testing.assert_close(
            res, ref, atol=atol, rtol=rtol,
            msg=f"Group {i} mismatch (M={m}, N={n}, K={k}, dtype={dtype})",
        )


class TestOutputCorrectness:
    """Verify output values with known input patterns and explicit block sizes."""

    def test_identity_matrix(self):
        """Multiply by identity matrix — output should equal input."""
        for n in [64, 128, 256]:
            a = torch.randn(n, n, device="cuda", dtype=torch.float16)
            eye = torch.eye(n, device="cuda", dtype=torch.float16)
            results = tritonblas.grouped_gemm([a], [eye])
            torch.testing.assert_close(
                results[0], a, atol=FP16_ATOL, rtol=FP16_RTOL,
                msg=f"Identity test failed for N={n}",
            )

    def test_ones_matrix(self):
        """A * ones should produce row sums."""
        m, k, n = 128, 64, 32
        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = torch.ones(k, n, device="cuda", dtype=torch.float16)
        ref = torch.matmul(a, b)
        results = tritonblas.grouped_gemm([a], [b])
        torch.testing.assert_close(
            results[0], ref, atol=FP16_ATOL, rtol=FP16_RTOL,
            msg="Ones matrix test failed",
        )

    def test_multi_group_identity(self):
        """Multiple groups, each multiplied by identity."""
        shapes = [(64, 64), (128, 128), (256, 256)]
        group_a, group_b, refs = [], [], []
        for m, n in shapes:
            a = torch.randn(m, n, device="cuda", dtype=torch.float16)
            eye = torch.eye(n, device="cuda", dtype=torch.float16)
            group_a.append(a)
            group_b.append(eye)
            refs.append(a)
        results = tritonblas.grouped_gemm(group_a, group_b)
        for i, (res, ref) in enumerate(zip(results, refs)):
            torch.testing.assert_close(
                res, ref, atol=FP16_ATOL, rtol=FP16_RTOL,
                msg=f"Multi-group identity test failed for group {i}",
            )

    @pytest.mark.parametrize("blk_m, blk_n, blk_k", [
        (64, 64, 64),
        (256, 256, 64),
    ], ids=["blk_64x64x64", "blk_256x256x64"])
    def test_explicit_block_sizes(self, blk_m, blk_n, blk_k):
        """Test with explicit BLK_M/BLK_N/BLK_K overrides."""
        shapes = [(256, 256, 256)] * 4
        _run_and_check_with_blk(shapes, BLK_M=blk_m, BLK_N=blk_n, BLK_K=blk_k)


# ---------------------------------------------------------------------------
# 18. Repeated calls (stability) — NEW
# ---------------------------------------------------------------------------

class TestRepeatedCalls:
    """Verify stability over many calls and JIT cache correctness."""

    def test_repeated_same_inputs(self):
        """Call grouped_gemm 100 times with same inputs, verify consistent output."""
        torch.manual_seed(42)
        shapes = [(128, 128, 128)] * 4
        group_a = [torch.randn(m, k, device="cuda", dtype=torch.float16) for m, _, k in shapes]
        group_b = [torch.randn(k, n, device="cuda", dtype=torch.float16) for _, n, k in shapes]

        # Get reference output from first call
        ref_results = tritonblas.grouped_gemm(group_a, group_b)

        # Run 99 more times
        for iteration in range(99):
            results = tritonblas.grouped_gemm(group_a, group_b)
            for i, (res, ref) in enumerate(zip(results, ref_results)):
                assert torch.equal(res, ref), (
                    f"Iteration {iteration + 1}, group {i}: output differs from first call"
                )

    def test_alternating_shapes(self):
        """Alternate between different shapes to test JIT cache correctness."""
        torch.manual_seed(123)
        shape_sets = [
            [(64, 64, 64)] * 2,
            [(128, 256, 128)] * 4,
            [(256, 128, 256)] * 3,
        ]

        # Pre-build inputs for each shape set
        inputs = []
        for shapes in shape_sets:
            ga = [torch.randn(m, k, device="cuda", dtype=torch.float16) for m, _, k in shapes]
            gb = [torch.randn(k, n, device="cuda", dtype=torch.float16) for _, n, k in shapes]
            refs = [torch.matmul(a, b) for a, b in zip(ga, gb)]
            inputs.append((ga, gb, refs, shapes))

        # Alternate between shape sets 10 times each
        for _ in range(10):
            for ga, gb, refs, shapes in inputs:
                results = tritonblas.grouped_gemm(ga, gb)
                for i, (res, ref) in enumerate(zip(results, refs)):
                    m, n, k = shapes[i]
                    torch.testing.assert_close(
                        res, ref, atol=FP16_ATOL, rtol=FP16_RTOL,
                        msg=f"Alternating shapes mismatch: M={m}, N={n}, K={k}",
                    )


# ---------------------------------------------------------------------------
# 19. Weird edge cases — NEW
# ---------------------------------------------------------------------------

_WEIRD_EDGE_PARAMS = [
    # Absolute minimum: 1x1x1
    pytest.param([(1, 1, 1)] * 4, id="M1_K1_N1_G4"),
    # Extreme K with tiny M/N
    pytest.param([(1, 1, 16384)] * 2, id="M1_K16384_N1_G2"),
    # Very tall, K=1
    pytest.param([(65536, 1, 1)], id="M65536_K1_N1_G1"),
    # Single element GEMM
    pytest.param([(1, 1, 1)], id="single_element_G1"),
    # 128 groups of single-row GEMM
    pytest.param([(1, 64, 64)] * 128, id="G128_M1_K64_N64"),
    # All small primes
    pytest.param([(13, 11, 7)] * 5, id="M13_K7_N11_G5"),
    # All primes, single group
    pytest.param([(3, 7, 5)], id="M3_K5_N7_G1"),
]


class TestWeirdEdgeCases:
    """Really unusual dimension patterns that stress corner cases."""

    @pytest.mark.parametrize("shapes", _WEIRD_EDGE_PARAMS)
    def test_weird_edge(self, shapes):
        _run_and_check(shapes)


# ---------------------------------------------------------------------------
# 20. Mismatched memory layout — NEW
# ---------------------------------------------------------------------------

class TestMismatchedLayouts:
    """Inputs with non-standard memory layouts and extreme dimension ratios."""

    def test_column_major_a(self):
        """A is column-major (transposed view made contiguous in column order)."""
        m, k, n = 128, 256, 128
        # Create column-major A by transposing a (K, M) tensor
        a_col = torch.randn(k, m, device="cuda", dtype=torch.float16).t().contiguous()
        b = torch.randn(k, n, device="cuda", dtype=torch.float16)
        ref = torch.matmul(a_col, b)
        results = tritonblas.grouped_gemm([a_col], [b])
        torch.testing.assert_close(
            results[0], ref, atol=FP16_ATOL, rtol=FP16_RTOL,
            msg="Column-major A test failed",
        )

    def test_large_k_small_mn(self):
        """Very large K with small M and N (K=8192, M=4, N=4)."""
        shapes = [(4, 4, 8192)] * 4
        _run_and_check(shapes)


# ---------------------------------------------------------------------------
# 21. Power-of-2 boundary dimensions — NEW
# ---------------------------------------------------------------------------

_POW2_BOUNDARY_PARAMS = [
    # Just below/above 128
    pytest.param([(127, 128, 128)] * 4, id="M127_below_128"),
    pytest.param([(129, 128, 128)] * 4, id="M129_above_128"),
    # K boundary around 64
    pytest.param([(128, 128, 63)] * 4, id="K63_below_64"),
    pytest.param([(128, 128, 65)] * 4, id="K65_above_64"),
    # N boundary around 256
    pytest.param([(128, 255, 128)] * 4, id="N255_below_256"),
    pytest.param([(128, 257, 128)] * 4, id="N257_above_256"),
    # M boundary around 512
    pytest.param([(511, 128, 128)] * 2, id="M511_below_512"),
    pytest.param([(513, 128, 128)] * 2, id="M513_above_512"),
]


class TestPowerOf2Boundaries:
    """Dimensions at +/-1 around power-of-2 tile boundaries."""

    @pytest.mark.parametrize("shapes", _POW2_BOUNDARY_PARAMS)
    def test_pow2_boundary(self, shapes):
        _run_and_check(shapes)


# ---------------------------------------------------------------------------
# 22. Group count boundaries — NEW
# ---------------------------------------------------------------------------

_GROUP_COUNT_BOUNDARY_PARAMS = [
    pytest.param([(128, 128, 128)] * 1, id="G1_min"),
    pytest.param([(128, 128, 128)] * 2, id="G2"),
    pytest.param([(128, 128, 128)] * 3, id="G3_prime"),
    pytest.param([(128, 128, 128)] * 5, id="G5_prime"),
    pytest.param([(128, 128, 128)] * 7, id="G7_prime"),
    # Many tiny groups
    pytest.param([(8, 64, 64)] * 100, id="G100_M8_tiny"),
    # Single huge group
    pytest.param([(16384, 64, 64)], id="G1_M16384_huge"),
]


class TestGroupCountBoundaries:
    """Boundary group counts: minimum, primes, and extremes."""

    @pytest.mark.parametrize("shapes", _GROUP_COUNT_BOUNDARY_PARAMS)
    def test_group_count_boundary(self, shapes):
        _run_and_check(shapes)


# ---------------------------------------------------------------------------
# 23. Numerical stability — NEW
# ---------------------------------------------------------------------------

class TestNumericalStability:
    """Inputs that stress floating-point precision."""

    def test_all_ones(self):
        """A=ones, B=ones => each element should be exactly K."""
        for m, k, n in [(64, 128, 64), (128, 256, 128)]:
            a = torch.ones(m, k, device="cuda", dtype=torch.float16)
            b = torch.ones(k, n, device="cuda", dtype=torch.float16)
            ref = torch.matmul(a, b)
            results = tritonblas.grouped_gemm([a], [b])
            torch.testing.assert_close(
                results[0], ref, atol=FP16_ATOL, rtol=FP16_RTOL,
                msg=f"All-ones test failed (M={m}, K={k}, N={n})",
            )

    def test_alternating_signs(self):
        """Alternating +1/-1 to test cancellation."""
        m, k, n = 128, 256, 128
        a = torch.ones(m, k, device="cuda", dtype=torch.float16)
        # B alternates +1/-1 along K dimension
        b = torch.ones(k, n, device="cuda", dtype=torch.float16)
        b[1::2, :] = -1.0
        ref = torch.matmul(a, b)
        results = tritonblas.grouped_gemm([a], [b])
        torch.testing.assert_close(
            results[0], ref, atol=FP16_ATOL, rtol=FP16_RTOL,
            msg="Alternating signs cancellation test failed",
        )

    def test_large_values(self):
        """Large-magnitude inputs (1000 * randn) — check overflow handling."""
        m, k, n = 64, 64, 64
        a = 1000.0 * torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = torch.randn(k, n, device="cuda", dtype=torch.float16)
        ref = torch.matmul(a, b)
        results = tritonblas.grouped_gemm([a], [b])
        # Use larger tolerance since magnitudes are bigger
        torch.testing.assert_close(
            results[0], ref, atol=FP16_ATOL * 1000, rtol=FP16_RTOL,
            msg="Large values test failed",
        )

    def test_small_values(self):
        """Small-magnitude inputs (1e-3 * randn) — check underflow."""
        m, k, n = 64, 64, 64
        a = 1e-3 * torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = 1e-3 * torch.randn(k, n, device="cuda", dtype=torch.float16)
        ref = torch.matmul(a, b)
        results = tritonblas.grouped_gemm([a], [b])
        # Products are ~1e-6, use proportional tolerance
        torch.testing.assert_close(
            results[0], ref, atol=1e-6, rtol=FP16_RTOL,
            msg="Small values underflow test failed",
        )


# ---------------------------------------------------------------------------
# 24. Variable K and N per group — true heterogeneous grouped GEMM
# ---------------------------------------------------------------------------

_VARIABLE_KN_PARAMS = [
    pytest.param(
        [(256, 512, 128), (512, 256, 64), (128, 128, 256), (64, 1024, 512)],
        id="4groups_varied",
    ),
    pytest.param(
        [(1024, 2048, 512), (512, 1024, 1024)],
        id="2groups_large",
    ),
    pytest.param(
        [(100, 200, 300), (200, 300, 100), (300, 100, 200)],
        id="3groups_non_pow2",
    ),
    pytest.param(
        [(1, 1, 1), (2, 3, 4), (5, 6, 7)],
        id="3groups_tiny",
    ),
    pytest.param(
        [(4096, 128, 256), (128, 4096, 128)],
        id="2groups_skewed_K",
    ),
    pytest.param(
        [(64, 64, 64), (128, 128, 128), (256, 256, 256), (512, 512, 512)],
        id="4groups_doubling",
    ),
    pytest.param(
        [(1024, 1, 1024), (1024, 1024, 1)],
        id="2groups_extreme_KN_ratio",
    ),
]


class TestVariableKN:
    """True grouped GEMM: M, K, and N all vary per group."""

    @pytest.mark.parametrize("shapes", _VARIABLE_KN_PARAMS)
    def test_variable_kn(self, shapes):
        _run_and_check(shapes)
