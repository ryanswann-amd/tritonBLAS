"""K-1693 — regression test: pin the negative result that the proposed
mid-K split-path (NS=3, BK=128) targeted at M=N=4096, K∈{4096,8192} is
LDS-infeasible at every reasonable square tile on gfx942 / MI300X, and
that the live ``OrigamiMatmulSelector`` therefore never routes those
cells to a (NS=3, BK=128) configuration.

This file is the only artifact landed for K-1693: the ticket falsified
the split-path hypothesis (see the K-1693 PR description and
``output/paired_split_path.csv``), so no production code was changed.
The tests below stop a future contributor from re-piloting the same
dead-end after the LDS estimator, the hardware budget, or the selector
filter changes.

Source-of-truth:
  - ``estimate_triton_lds_bytes`` / ``check_triton_lds_capacity`` in
    ``tritonblas/origami.py``
  - ``OrigamiMatmulSelector`` in ``tritonblas/origami.py``
gfx942 / MI300X per-WG LDS budget = 64 KB (65536 bytes).
"""
import pytest

from tritonblas.origami import (
    estimate_triton_lds_bytes,
    check_triton_lds_capacity,
)

GFX942_LDS_CAP = 64 * 1024  # MI300X per-WG LDS limit


# ---------------------------------------------------------------------------
# (a) LDS-budget invariant
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bm,bn,bk,ns,expected_lds_bytes",
    [
        # Brief literal target -- 4× the MI300X per-WG budget
        (256, 256, 128, 3, 262144),
        # Brief title alt: BK=128 → 256 -- 8× the budget
        (256, 256, 256, 3, 524288),
        # Smaller squares K-1655 already screened
        (128, 128, 128, 3, 131072),
        (256, 256,  64, 3, 131072),  # the K-1655 OOM result
    ],
)
def test_k1693_lds_budget_rules_out_ns3_bk128(bm, bn, bk, ns, expected_lds_bytes):
    """Predicted LDS for the K-1693 split-path candidates must exceed
    the MI300X per-WG budget.

    Failing this test means the estimator drifted *or* the hardware
    budget grew enough to admit the literal brief target -- in either
    case, re-run the K-1693 paired benchmark before reviving the split.
    """
    bytes_a = bytes_b = 2  # bf16/fp16
    lds = estimate_triton_lds_bytes(bm, bn, bk, bytes_a, bytes_b, ns)
    assert lds == expected_lds_bytes, (
        f"K-1693: LDS estimator drifted for {bm}x{bn}x{bk} ns={ns}: "
        f"got {lds}, expected {expected_lds_bytes} -- re-derive the "
        f"split-path infeasibility argument."
    )
    assert lds > GFX942_LDS_CAP, (
        f"K-1693: brief target ({bm}x{bn}x{bk} ns={ns}) was pinned as "
        f"INFEASIBLE on gfx942 (≤{GFX942_LDS_CAP} bytes), but the "
        f"estimator now reports {lds} bytes which fits. Re-run the "
        f"paired n=30 split-path benchmark before assuming it is still "
        f"a dead end (see output/paired_split_path.csv)."
    )


def test_k1693_estimator_explicit_invariant():
    """Direct, unparametrised assertion the Testing Zealot called out:
    ``estimate_triton_lds_bytes(256, 256, 128, 2, 2, 3) > 65536``.

    Kept separate from the parametrised case so a single ``-k`` filter
    can name it.
    """
    lds = estimate_triton_lds_bytes(256, 256, 128, 2, 2, 3)
    assert lds > 65536, (
        f"K-1693 LDS-budget guard: (256,256,128, ns=3) needs {lds} B "
        f"but must exceed the 65536 B MI300X per-WG limit."
    )


def test_k1693_no_square_tile_admits_ns3_bk128():
    """Across Origami's default square-tile range BM=BN ∈ {16..256},
    NS=3 + BK=128 only fits at BM=BN ≤ 64.  At BM=BN=64 the M=N=4096
    grid is 4096 WGs (13× the 304-CU MI300X count), which breaks the
    persistent_matmul one-tile-per-WG invariant.
    """
    feasible_squares = []
    for bm in (16, 32, 64, 128, 256):
        if check_triton_lds_capacity(bm, bm, 128, 2, 2, GFX942_LDS_CAP, num_stages=3):
            feasible_squares.append(bm)
    assert feasible_squares == [16, 32, 64], (
        f"K-1693: square-tile NS=3 BK=128 LDS-feasibility set drifted: "
        f"{feasible_squares}. Original assumption: only BM=BN ∈ {{16,32,64}} fit."
    )
    M = N = 4096
    bm = 64
    grid = (M // bm) * (N // bm)
    cu_count_mi300x = 304
    assert grid > 10 * cu_count_mi300x, (
        f"K-1693: BM=BN={bm} grid for {M}x{N} = {grid} WGs -- expected "
        f">10× the {cu_count_mi300x}-CU count to break persistent grid."
    )


# ---------------------------------------------------------------------------
# (b) Selector-routing invariant
#
# These tests need an MI300X device + the origami C-extension; they are
# skipped cleanly off-GPU so the rest of the suite stays runnable on
# laptops / CI without ROCm.
# ---------------------------------------------------------------------------
def _selector_or_skip(M, N, K, num_stages):
    """Build an ``OrigamiMatmulSelector`` for an MI300X cell, or skip."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("K-1693 selector test needs an MI300X device")
    pytest.importorskip("origami")
    from tritonblas.origami import OrigamiMatmulSelector

    dev = torch.device(f"cuda:{torch.cuda.current_device()}")
    arch = getattr(torch.cuda.get_device_properties(dev), "gcnArchName", "")
    if "gfx942" not in arch:
        pytest.skip(f"K-1693 selector test pinned to gfx942 (got {arch!r})")

    return OrigamiMatmulSelector(
        m=M,
        n=N,
        k=K,
        a_dtype=torch.bfloat16,
        b_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        device=dev,
        num_stages=num_stages,
    )


@pytest.mark.parametrize("K", [4096, 8192])
@pytest.mark.parametrize("num_stages", [2, 3])
def test_k1693_selector_does_not_route_midk_to_ns3_bk128(K, num_stages):
    """For the M=N=4096 mid-K cells K-1693 targeted, the live
    OrigamiMatmulSelector must NEVER pick a (BK=128, NS=3) tile -- the
    only way it could would be if the LDS-capacity filter regressed and
    started admitting (256, 256, 128, ns=3) (or any (bm, bn, 128, ns=3)
    square that does not fit in 64KB LDS).

    We assert against BOTH NS=2 (the production default) and NS=3 (the
    config K-1693 piloted) so the guard fires regardless of which
    num_stages a future caller asks for.
    """
    M = N = 4096
    sel = _selector_or_skip(M, N, K, num_stages=num_stages)

    chosen = (sel.block_m, sel.block_n, sel.block_k, sel.num_stages)

    # Hard fail: the literal brief target is forbidden everywhere.
    assert chosen != (256, 256, 128, 3), (
        f"K-1693: selector routed mid-K M=N={M} K={K} (ns_req={num_stages}) "
        f"to the K-1693 brief target {chosen}, which was pinned as "
        f"LDS-infeasible (256 KB > 64 KB MI300X budget). The LDS-capacity "
        f"filter in OrigamiMatmulSelver.__init__ has regressed -- re-run "
        f"the K-1693 paired benchmark before allowing this config."
    )

    # Stronger pin: any BK=128 + NS=3 square would also break the LDS
    # budget at the (256, 256, *) tile Origami otherwise prefers.
    if num_stages == 3 and chosen[2] == 128:
        bm, bn, bk, ns = chosen
        lds = estimate_triton_lds_bytes(bm, bn, bk, 2, 2, ns)
        assert lds <= GFX942_LDS_CAP, (
            f"K-1693: selector picked BK=128 NS=3 tile {chosen} for "
            f"M=N={M} K={K} but it needs {lds} B > {GFX942_LDS_CAP} B "
            f"MI300X budget -- LDS filter has regressed."
        )
