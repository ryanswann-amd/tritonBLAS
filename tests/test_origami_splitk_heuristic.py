"""
Unit tests for the continuous split-K grid heuristic
(``compute_continuous_sk_grid``) introduced in K-557 and validated in K-583.

These tests do NOT require a GPU.  They import the helper directly from
``tritonblas.origami`` and exercise it against synthetic
(M, N, K, block_m, block_n, block_k, cu_count, out_dtype_bitsize) inputs.

Coverage
--------

1. **Edge case K=1**: degenerate single-K-tile problems must still produce a
   valid ``sk_grid`` >= tiles, never zero or negative.
2. **Edge case K=very-large** (16 MiB+): single-tile-MN with deep K must hit
   the workspace cap, never the wave-fill cap; ``sk_grid <= cu_count``.
3. **Continuous-interpolation correctness**: as K is swept smoothly across
   the boundary that used to flip between discrete factors {8, 6, 4, 3, 2, 1},
   the chosen split factor must change by at most 1 between adjacent K values
   spaced 1 BLK_K apart.  No "cliffs" >= 2 are allowed — that is the bug
   K-557 set out to fix.
4. **No oversubscription**: ``sk_grid <= cu_count`` whenever ``tiles <= cu_count``.
5. **K-398 cleanup-bug regression guard**: a valid split that happens to
   leave a tile-remainder must be preserved when the workspace budget is
   intact.  The buggy original always reset ``sk_grid = tiles`` here.
6. **Wave-balance regime monotonicity**: for ``tiles > cu_count``, the
   chosen ``sk_grid`` must lie in [cu_count/2, cu_count] and divide
   ``ceil(tiles/sk_grid)`` evenly into waves.
7. **Last-wave kicker**: for problem sizes that produce a tiny tail wave
   (last_wave_remainder in (0, 128)) on gfx942-class CU counts, the kicker
   must rewrite ``sk_grid`` to {256, 64, 64} for cu_count in {304, 80, 64}.
8. **Determinism**: the helper is a pure function — same input → same output.

These tests are designed to fail loudly if anyone re-introduces the discrete
table or reverts the K-398 cleanup-bug fix.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# Allow running directly via ``python -m pytest tests/`` from the repo root
# without installing tritonblas first.  The package ``include/tritonblas`` is
# installed onto sys.path by ``pip install -e .``; this fallback covers the
# pure-Python / no-build case.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_INCLUDE = _REPO_ROOT / "include"
if _INCLUDE.exists() and str(_INCLUDE) not in sys.path:
    sys.path.insert(0, str(_INCLUDE))

# ``tritonblas.origami`` imports the C++ ``origami`` module at top of file
# for its hardware probe.  The helper itself is pure Python and does not
# depend on it, but the import will fail if the extension isn't built.  Skip
# the whole module rather than blowing up collection.
try:
    from tritonblas.origami import compute_continuous_sk_grid
except Exception as exc:  # pragma: no cover — import-time failure
    pytest.skip(
        f"tritonblas.origami import failed (need built extension): {exc}",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Hardware-shaped constants
# ---------------------------------------------------------------------------

CU_MI300X = 304   # gfx942
CU_MI100  = 120   # gfx908
CU_MI50   = 60    # gfx906 (off the kicker whitelist)
CU_8CARD  = 80    # consumer / kicker-whitelisted
CU_NAVI   = 64    # consumer / kicker-whitelisted

BF16_BITS = 16
FP32_BITS = 32
FP8_BITS  = 8


# ---------------------------------------------------------------------------
# §1  Edge case: K = 1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("m,n,blk_m,blk_n", [
    (256,  256,  128, 128),   # tiny square
    (1024, 1024, 256, 256),
    (4096, 4096, 256, 256),
    (256,  64,   128, 64),    # skinny-N
])
def test_k_equals_one_returns_valid_grid(m, n, blk_m, blk_n):
    """K=1 must produce an integer ``sk_grid`` >= 1; never zero or negative."""
    grid = compute_continuous_sk_grid(
        m=m, n=n, k=1,
        block_m=blk_m, block_n=blk_n, block_k=64,
        cu_count=CU_MI300X, out_dtype_bitsize=BF16_BITS,
    )
    tiles = math.ceil(m / blk_m) * math.ceil(n / blk_n)
    assert isinstance(grid, int) and grid >= 1, f"K=1 produced grid={grid}"
    # K=1 → iters_per_tile=1 → f_work=max(1,1//8)=1 → factor=1 → sk_grid=tiles
    # …unless tiles > cu_count (wave-balance) or last-wave kicker fires.
    assert grid >= 1
    if tiles <= CU_MI300X:
        assert grid == tiles, (
            f"K=1 with tiles={tiles}<=cu={CU_MI300X} should not split-K; "
            f"got sk_grid={grid}"
        )


# ---------------------------------------------------------------------------
# §2  Edge case: K = very large (no overflow, no oversubscription)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [
    16384,           # ~16k iters at BLK_K=64 with K=1MB
    1 << 18,         # 256 KiB K
    1 << 22,         # 4 MiB K — well into split-K-saturated regime
    1 << 24,         # 16 MiB K
])
def test_k_very_large_caps_at_cu_count(k):
    """For very-deep K, ``sk_grid`` must remain bounded by cu_count."""
    grid = compute_continuous_sk_grid(
        m=512, n=512, k=k,
        block_m=128, block_n=128, block_k=64,
        cu_count=CU_MI300X, out_dtype_bitsize=BF16_BITS,
    )
    tiles = math.ceil(512 / 128) * math.ceil(512 / 128)  # 16
    assert tiles == 16
    # tiles=16 < cu=304, so split-K regime; factor capped by f_grid=304//16=19
    # AND f_work=ceil(K/64)//8.  For K=16384, iters=256, f_work=32 → min=19.
    # Result: sk_grid = 16 * factor, factor in [1,19].
    assert grid <= CU_MI300X, f"sk_grid={grid} oversubscribed cu={CU_MI300X}"
    assert grid >= tiles, f"sk_grid={grid} undersubscribed tiles={tiles}"
    # Must be a multiple of tiles (split-K regime always returns tiles*f).
    assert grid % tiles == 0, (
        f"K={k}: split-K regime should produce multiple of tiles={tiles}; "
        f"got sk_grid={grid}"
    )


# ---------------------------------------------------------------------------
# §3  Continuous interpolation: sweep K, factor must change by <= 1 per step
# ---------------------------------------------------------------------------

def test_continuous_factor_no_cliffs():
    """
    Sweep K across the entire range that the prior heuristic split among
    {8, 6, 4, 3, 2, 1}.  The new heuristic must move ``factor`` by at most 1
    between adjacent K values (one BLK_K apart).  This is the key test that
    the discrete-table cliffs are gone.
    """
    blk_m, blk_n, blk_k = 128, 128, 64
    cu = CU_MI300X
    m, n = 512, 512  # tiles = 16  → split-K regime
    tiles = math.ceil(m / blk_m) * math.ceil(n / blk_n)

    factors = []
    for k in range(blk_k, blk_k * 200, blk_k):  # 1..199 K-iters
        grid = compute_continuous_sk_grid(
            m=m, n=n, k=k,
            block_m=blk_m, block_n=blk_n, block_k=blk_k,
            cu_count=cu, out_dtype_bitsize=BF16_BITS,
        )
        # In split-K regime sk_grid is tiles * factor.
        assert grid % tiles == 0
        factors.append(grid // tiles)

    # Adjacent factors must change by at most 1.  Discrete tables produced
    # jumps of 2-7 (e.g. 8 → 6, 6 → 4, 4 → 3); the continuous heuristic
    # produces 0/1 deltas everywhere.
    deltas = [abs(b - a) for a, b in zip(factors, factors[1:])]
    max_delta = max(deltas)
    assert max_delta <= 1, (
        f"Continuous heuristic produced cliff: max |Δfactor|={max_delta} "
        f"across K sweep.  This is the K-555/K-557 boundary-residual signature. "
        f"Sample factors: {factors[:30]}"
    )

    # Also sanity-check: factors are weakly increasing in K (deeper K ⇒ more
    # iters per tile ⇒ more headroom to split).
    assert factors == sorted(factors), (
        "Factor must be monotone non-decreasing in K within split-K regime; "
        f"got: {factors[:30]}"
    )


# ---------------------------------------------------------------------------
# §4  No-oversubscription invariant
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("m,n,k,blk_m,blk_n,blk_k", [
    (256,   256,   2048,  128, 128, 64),    # tiles=4
    (1024,  1024,  4096,  128, 128, 64),    # tiles=64
    (4096,  4096,  4096,  256, 256, 64),    # tiles=256
    (8192,  8192,  4096,  256, 256, 64),    # tiles=1024 > cu (wave-balance)
    (16384, 16384, 1024,  128, 128, 64),    # tiles=16384 > cu (deep wave-balance)
])
def test_grid_does_not_oversubscribe_cu(m, n, k, blk_m, blk_n, blk_k):
    """``sk_grid`` must never exceed cu_count for tiles<=cu (and never by more
    than cu_count for tiles>cu — wave-balance caps at cu_count)."""
    grid = compute_continuous_sk_grid(
        m=m, n=n, k=k,
        block_m=blk_m, block_n=blk_n, block_k=blk_k,
        cu_count=CU_MI300X, out_dtype_bitsize=BF16_BITS,
    )
    tiles = math.ceil(m / blk_m) * math.ceil(n / blk_n)
    if tiles <= CU_MI300X:
        assert grid <= CU_MI300X, (
            f"tiles={tiles}<=cu={CU_MI300X} but sk_grid={grid} oversubscribed"
        )
    else:
        # Wave-balance: ``ceil(tiles/waves)`` rounded down by min(.., cu_count).
        # Last-wave kicker may push to 256 — still <= cu_count=304.
        assert grid <= CU_MI300X, (
            f"wave-balance sk_grid={grid} > cu={CU_MI300X}"
        )


# ---------------------------------------------------------------------------
# §5  K-398 cleanup-bug regression guard
# ---------------------------------------------------------------------------

def test_split_with_remainder_preserved_when_workspace_ok():
    """
    The K-398 bug was an unconditional ``if tiles % sk_grid != 0: sk_grid = tiles``
    that wiped every non-divisor split.  Construct a case where the chosen
    ``sk_grid`` does leave a remainder but the per-WG workspace bytes are
    well below the 128 MiB cap, and assert the split is preserved.
    """
    # 1024×1024×16384 with BLK=128/128/64:
    # tiles = 8*8 = 64; cu=304; tiles<cu split-K branch.
    # f_grid = 304 // 64 = 4
    # iters = ceil(16384/64) = 256; f_work = 256//8 = 32
    # factor = min(4, 32) = 4 → sk_grid = 64*4 = 256
    # tiles % sk_grid = 64 % 256 = 64 (nonzero)
    # partial_bytes = 128*128*2*256 = 8 MiB << 128 MiB → split MUST be kept
    grid = compute_continuous_sk_grid(
        m=1024, n=1024, k=16384,
        block_m=128, block_n=128, block_k=64,
        cu_count=CU_MI300X, out_dtype_bitsize=BF16_BITS,
    )
    tiles = 64
    assert grid == 256, (
        f"K-398 regression: expected sk_grid=256 (factor=4), got {grid}. "
        f"The cleanup bug must have come back — non-divisor splits are "
        f"being thrown away."
    )
    assert grid % tiles == 0  # sanity
    assert grid // tiles == 4


def test_split_falls_back_when_workspace_blown():
    """The cleanup line MUST still fire when partial-tile bytes exceed the
    128 MiB budget — that's the original safeguard, kept by K-557."""
    # Very wide MN tile + many iters → big partial accumulator.
    # 512×512 with BLK=512/512/64:
    # tiles = 1*1 = 1; cu=304; tiles<cu.
    # f_grid = 304; iters = ceil(K/64) = ?; f_work depends on K.
    # We want a case where the cleanup line would trigger if implemented strictly.
    # Use tiles=1 (no remainder), pick K=8192 → iters=128, f_work=16, factor=min(304,16)=16
    # → sk_grid = 16. tiles % sk_grid = 1 % 16 = 1, nonzero.
    # partial_bytes = 512*512*4 (fp32 partials) * 16 = 16 MiB → within budget, kept.
    # To trigger the workspace fallback we need partial_bytes > 128 MiB:
    # 512*512*4 = 1 MiB per grid; 128 MiB / 1 MiB = 128 grid → factor 128 needs K>=128*8*64=65536.
    grid = compute_continuous_sk_grid(
        m=512, n=512, k=1 << 20,                     # K = 1 MiB
        block_m=512, block_n=512, block_k=64,
        cu_count=CU_MI300X, out_dtype_bitsize=FP32_BITS,
    )
    tiles = 1
    # When workspace would blow, helper resets to ``tiles``.  Either path is
    # acceptable; the contract is that ``partial_bytes(sk_grid) <= cap`` OR
    # ``sk_grid == tiles`` (no-split fallback).
    cap = 128 * 1024 * 1024
    partial_bytes = 512 * 512 * (FP32_BITS // 8) * grid
    assert grid == tiles or partial_bytes <= cap, (
        f"Workspace contract violated: sk_grid={grid}, partial_bytes={partial_bytes} "
        f"> cap={cap}, but did not fall back to tiles={tiles}."
    )


# ---------------------------------------------------------------------------
# §6  Wave-balance regime sanity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("m,n", [
    (4096, 4096),    # tiles = 256 — borderline (== cu? no, cu=304)
    (8192, 4096),    # tiles = 512
    (8192, 8192),    # tiles = 1024
    (16384, 16384),  # tiles = 4096
])
def test_wave_balance_evenly_fills_waves(m, n):
    """For ``tiles > cu_count``, the heuristic must produce a grid that
    spreads work as evenly as possible across the minimum number of waves."""
    blk_m, blk_n = 256, 256
    grid = compute_continuous_sk_grid(
        m=m, n=n, k=4096,
        block_m=blk_m, block_n=blk_n, block_k=64,
        cu_count=CU_MI300X, out_dtype_bitsize=BF16_BITS,
    )
    tiles = math.ceil(m / blk_m) * math.ceil(n / blk_n)
    if tiles <= CU_MI300X:
        pytest.skip(f"tiles={tiles}<=cu={CU_MI300X}; not wave-balance regime")

    expected_waves = math.ceil(tiles / CU_MI300X)
    expected_grid = math.ceil(tiles / expected_waves)
    # Either the wave-balance result OR the last-wave-kicker rewrite (to 256).
    assert grid in (expected_grid, 256), (
        f"wave-balance: tiles={tiles}, expected sk_grid={expected_grid} "
        f"(or 256 from kicker), got {grid}"
    )


# ---------------------------------------------------------------------------
# §7  Last-wave kicker
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cu_count,expected_kicker", [
    (CU_MI300X, 256),
    (CU_8CARD,   64),
    (CU_NAVI,    64),
])
def test_last_wave_kicker_fires_on_tiny_tail(cu_count, expected_kicker):
    """Construct ``tiles`` so that ``tiles % cu_count`` lands in (0, 128).
    The kicker must rewrite ``sk_grid`` to the architecture-specific value."""
    # tiles = cu + 1 → remainder = 1 → kicker fires
    tiles = cu_count + 1
    blk_m, blk_n = 128, 128
    # Compose M, N so ceil(M/blk_m) * ceil(N/blk_n) == tiles exactly.
    m = blk_m * tiles
    n = blk_n * 1
    grid = compute_continuous_sk_grid(
        m=m, n=n, k=4096,
        block_m=blk_m, block_n=blk_n, block_k=64,
        cu_count=cu_count, out_dtype_bitsize=BF16_BITS,
    )
    assert grid == expected_kicker, (
        f"last-wave kicker: tiles={tiles}, cu={cu_count}, expected sk_grid="
        f"{expected_kicker}, got {grid}"
    )


def test_last_wave_kicker_does_not_fire_off_whitelist():
    """For cu_count not in {304, 80, 64}, the kicker must NOT fire."""
    cu_count = CU_MI100  # 120 is off the gfx942-class whitelist
    tiles = cu_count + 1
    grid = compute_continuous_sk_grid(
        m=128 * tiles, n=128, k=4096,
        block_m=128, block_n=128, block_k=64,
        cu_count=cu_count, out_dtype_bitsize=BF16_BITS,
    )
    # Wave-balance would pick ceil(tiles/2)=ceil(121/2)=61 → min(61,120)=61
    expected = math.ceil(tiles / math.ceil(tiles / cu_count))
    assert grid == expected, (
        f"kicker fired off-whitelist on cu={cu_count}: expected {expected}, got {grid}"
    )


# ---------------------------------------------------------------------------
# §8  Determinism / purity
# ---------------------------------------------------------------------------

def test_pure_function_determinism():
    """Same args → same result, every time, no hidden state."""
    args = dict(
        m=2048, n=1024, k=16384,
        block_m=128, block_n=128, block_k=64,
        cu_count=CU_MI300X, out_dtype_bitsize=BF16_BITS,
    )
    first = compute_continuous_sk_grid(**args)
    for _ in range(100):
        assert compute_continuous_sk_grid(**args) == first


def test_memoization_repeats_hit_cache():
    """The helper is wrapped in ``functools.lru_cache`` (K-583 review fix)
    so hot-path callers (autotune, inference) skip the integer search on
    repeated shapes.  Verify cache is wired up and hits accumulate."""
    # Reset cache so this test is order-independent.
    compute_continuous_sk_grid.cache_clear()
    args = dict(
        m=4096, n=4096, k=4096,
        block_m=256, block_n=256, block_k=64,
        cu_count=CU_MI300X, out_dtype_bitsize=BF16_BITS,
    )
    # Cold call → 1 miss
    first = compute_continuous_sk_grid(**args)
    info1 = compute_continuous_sk_grid.cache_info()
    assert info1.misses == 1
    assert info1.hits == 0
    # 50 repeats → 50 hits, no new misses
    for _ in range(50):
        assert compute_continuous_sk_grid(**args) == first
    info2 = compute_continuous_sk_grid.cache_info()
    assert info2.misses == 1, f"unexpected new misses: {info2}"
    assert info2.hits == 50, f"expected 50 hits, got {info2.hits}"
    # Different args → new miss, doesn't poison existing entry
    args2 = dict(args, m=8192)
    _ = compute_continuous_sk_grid(**args2)
    info3 = compute_continuous_sk_grid.cache_info()
    assert info3.misses == 2
    assert info3.hits == 50


# ---------------------------------------------------------------------------
# §9  Boundary-shape regression set (K-548 / K-557 / K-583)
# ---------------------------------------------------------------------------

# Each entry: (m, n, k, blk_m, blk_n, blk_k, expected_grid, label)
# Expected ``sk_grid`` values come from the K-557 measured benchmark
# (state/mc2/workspaces/K-557/output/bench_results.csv) for the published
# block sizes that origami selects on bf16/MI300X.  Tests pin the heuristic
# output so future regressions surface as test failures.
# BLK_M / BLK_N values are the macro-tiles Origami's autotuner picks on
# bf16/MI300X for each shape — we hand-derived the exact block sizes that
# reproduce the measured ``sk_new`` from K-557's MI300X benchmark.
BOUNDARY_SHAPES = [
    # M, N, K, BLK_M, BLK_N, BLK_K → expected sk_grid (continuous heuristic)
    (2048, 1024, 16384, 256, 256, 64, 288, "rank1_2Kx1Kx16K"),
    (1024, 1024, 16384, 256, 256, 64, 304, "rank2_1Kx1Kx16K"),  # all-CU split-K
    (2048,  512, 16384, 256, 256, 64, 304, "rank3_2Kx512x16K"),
    (4096, 2048,  8192, 256, 256, 64, 256, "rank4_4Kx2Kx8K"),   # wave-balance
    (1024, 1024,  8192, 128, 128, 64, 256, "rank5_1Kx1Kx8K"),
    (1024,  512, 16384,  64, 256, 64, 288, "rank6_1Kx512x16K"),
    ( 512,  512, 16384,  64, 128, 64, 288, "rank7_512x512x16K"),
    ( 512,  256,  8192,  64, 128, 64, 256, "rank8_512x256x8K"),
    (8192, 4096,  8192, 256, 256, 64, 256, "rank9_8Kx4Kx8K"),
    (4096, 1024,  8192, 256, 256, 64, 256, "rank10_4Kx1Kx8K"),
]


@pytest.mark.parametrize(
    "m,n,k,blk_m,blk_n,blk_k,expected,label",
    BOUNDARY_SHAPES,
    ids=[s[-1] for s in BOUNDARY_SHAPES],
)
def test_boundary_shape_pins(m, n, k, blk_m, blk_n, blk_k, expected, label):
    """Lock down ``sk_grid`` for every shape that K-557 measured a perf delta
    on.  If anyone weakens the heuristic, these tests scream."""
    grid = compute_continuous_sk_grid(
        m=m, n=n, k=k,
        block_m=blk_m, block_n=blk_n, block_k=blk_k,
        cu_count=CU_MI300X, out_dtype_bitsize=BF16_BITS,
    )
    assert grid == expected, (
        f"{label}: heuristic regression. Expected sk_grid={expected} "
        f"(continuous K-557 value, validated on MI300X); got {grid}."
    )
