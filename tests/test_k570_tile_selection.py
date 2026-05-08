"""Regression test locking in Origami tile selection for the K-570 large-K
square fp16/bf16 cohort on MI300X.

Background
----------
K-570 reported a 0.2574x cold-cache geomean regression vs hipBLASLt on the
18-shape large-K square cohort (M=N in {1024,2048,4096} x K in {1024,2048,4096}
x {fp16,bf16}). K-618 hypothesized that the cause was an Origami tile-selector
error (claim: Origami picks tiles=256, NS=2 instead of optimal tiles=128, NS=4)
and proposed a narrow shape-keyed override (BM=BN=128, BK=64, NS=3, NW=4,
WPEU=2) mirroring hipBLASLt's MT128x128x64 PGR2_PLR1 pick.

K-660 investigation (paired graph-captured kernel-only A/B sweep on MI300X,
50-op replay x 30-iter median x 3 graphs, rocm/pytorch image):

  * Baseline geomean(hipBLASLt_us / tritonBLAS_baseline_us) = 0.9725 (>= 0.85x)
    --- already passes the K-570 success criterion, so the K-570 wrapper-bound
    figure is *not* tile-selector-bound; it lives in K-282/K-398/K-451 launch
    plumbing, out of scope for tile/NS work.
  * The K-618 proposed override regresses the entire envelope (geomean
    mod/baseline = 0.7194; max regression -58.9 %) because at 1024^2 the
    128x128 tile gives only 64 grid elements (~21 % of MI300X's compute
    units) vs the 64x64 baseline giving ~84 % CU utilisation.
  * Focused tile/NS search on the worst two cohort shapes (2048x2048x4096
    fp16/bf16) with NS in {2,3,4}, BK in {64,128,256}, BM/BN in
    {128,256} confirmed the Origami pick (BM=BN=128, BK=128, NS=2) is the
    *local optimum* within MI300X's 64 KB LDS envelope (NS=3 with
    BK=128 needs 131 KB, NS=4 needs 196 KB --- both LDS-overflow).

Therefore K-660 ships *no* tile/NS override. This test locks in the
investigated tile selection so that any future change reintroducing the K-618
hypothesis (or any other override that flips these picks) fails CI.

Per-shape evidence is preserved in the K-660 workspace at
output/sweep_results.csv (full 30-shape A/B sweep) and output/mini_sweep.json
(focused 11-candidate search on the worst two shapes).
"""
import pytest
import torch

from tritonblas.origami import OrigamiMatmulSelector


# Expected Origami picks on MI300X for the K-570 large-K square cohort.
# Format: (M, N, K, dtype) -> (BM, BN, BK, NS).
# These were measured from the unmodified Origami selector on c42 / MI300X.
# A future override that reintroduces the K-618 hypothesis (e.g.
# (BM=128, BN=128, BK=64, NS=3) for the 2048^2 K=4096 shapes) would
# flip these values and trip this assertion, with the failure message
# pointing back to the K-660 negative-result evidence.
K570_EXPECTED_PICKS = {
    (1024, 1024, 1024): (64, 64, 256, 2),
    (1024, 1024, 2048): (64, 64, 256, 2),
    (1024, 1024, 4096): (64, 64, 256, 2),
    (2048, 2048, 1024): (128, 128, 128, 2),
    (2048, 2048, 2048): (128, 128, 128, 2),
    (2048, 2048, 4096): (128, 128, 128, 2),
    (4096, 4096, 1024): (256, 256, 64, 2),
    (4096, 4096, 2048): (256, 256, 64, 2),
    (4096, 4096, 4096): (256, 256, 64, 2),
}

# K-618 proposed override (do NOT add for the cohort; the assertion below
# will flip if it is ever reintroduced for any of the 18 cohort shapes).
K618_PROPOSED_OVERRIDE = (128, 128, 64, 3)


def _is_mi300x() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        gcn = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
    except Exception:
        return False
    return gcn.split(":")[0] == "gfx942"


pytestmark = pytest.mark.skipif(
    not _is_mi300x(),
    reason="K-570 cohort tile picks are MI300X-specific (gfx942)",
)


@pytest.mark.parametrize("dtype_str", ["fp16", "bf16"])
@pytest.mark.parametrize("M,N,K", sorted(K570_EXPECTED_PICKS.keys()))
def test_k570_origami_pick_locked(M, N, K, dtype_str):
    """Origami's tile/NS pick for the K-570 cohort must match the K-660-
    investigated baseline. If a future change adds a shape-keyed override
    that flips a pick (e.g. the K-618 proposal for the 2048^2 K=4096
    cohort), this test fails and the change must justify the override
    against the K-660 negative-result evidence (workspace
    output/mini_sweep.json + output/sweep_results.csv).
    """
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_str]
    device = torch.device("cuda:0")
    selector = OrigamiMatmulSelector(
        M, N, K, dtype, dtype, dtype, device, num_stages=2,
    )
    actual = (selector.block_m, selector.block_n, selector.block_k,
              getattr(selector, "num_stages", 2))
    expected = K570_EXPECTED_PICKS[(M, N, K)]

    assert actual == expected, (
        f"Origami tile/NS pick changed for K-570 cohort shape "
        f"M={M} N={N} K={K} dtype={dtype_str}: "
        f"expected (BM,BN,BK,NS)={expected}, got {actual}. "
        f"This test guards the K-660 finding that no tile/NS override "
        f"beats the Origami baseline within MI300X's 64 KB LDS envelope "
        f"(see workspace output/mini_sweep.json). If the change is the "
        f"K-618 proposed override {K618_PROPOSED_OVERRIDE}, note that "
        f"K-660 measured it at -13.4 % to -58.9 % per shape across the "
        f"full envelope (geomean mod/baseline = 0.7194)."
    )


def test_k618_override_not_present_for_2048_k4096():
    """The two worst K-570 cohort shapes (2048x2048x4096 fp16/bf16) sit
    at hipBLASLt/tritonBLAS = 0.82 under the K-660 measurement methodology.
    K-660 ruled out the K-618 proposed override (BM=BN=128, BK=64, NS=3)
    for these shapes because every NS>=3 candidate that fits in MI300X
    LDS regressed kernel-only timing by >10 %. This test asserts the
    selector still returns the locally-optimal Origami pick for these
    shapes. If a future change introduces the K-618 override for these
    shapes (i.e. the predicate fires and BK or NS changes), this test
    will flag it.
    """
    dtype = torch.float16
    device = torch.device("cuda:0")
    for dt in (torch.float16, torch.bfloat16):
        sel = OrigamiMatmulSelector(2048, 2048, 4096, dt, dt, dt, device,
                                    num_stages=2)
        actual = (sel.block_m, sel.block_n, sel.block_k,
                  getattr(sel, "num_stages", 2))
        assert actual != K618_PROPOSED_OVERRIDE, (
            f"K-618 proposed override (BM=128, BN=128, BK=64, NS=3) "
            f"appears to have been reintroduced for the 2048x2048x4096 "
            f"{dt} shape. K-660 measured this override at -16.6 % "
            f"(fp16) / -16.8 % (bf16) on this exact shape; do not ship "
            f"it without paired graph-captured kernel-only evidence "
            f"that beats the (128,128,128,NS=2) Origami baseline."
        )
