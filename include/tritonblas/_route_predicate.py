"""K-1938 / K-1926 — Per-shape route predicate for the wave-misaligned
skinny-N K-COMPLEMENT cluster on MI300X gfx942.

This module hosts the verified-winner alias-stack used by the
``_route_to_hbl()`` dispatcher in ``matmul.py``.  Slots are added one
ticket at a time; the latest addition is the P41 N=608 slot landed by
K-1938 (paired n=30 HIP-graph hot-cache bench, gate hbl/tb <= 0.85 and
paired-t p < 0.05 on MI300X).

Wave-misalignment signature for the cluster (all share the same modular
class):

    N mod 64 == 32       (N in {160, 224, 288, 352, 416, 480, 544, 608})
    N mod 128 in {32, 96} (partial tail-wave on the BLOCK_N=128 grid)

For these shapes the tritonblas persistent-matmul kernel issues a tail
wavefront that is only half-populated; LDS bank-conflict pressure
dominates the steady-state instruction mix and hipBLASLt's MFMA-friendly
tile layout wins the wall-time race.  See the gist in
``output/gist_K1938.md`` for the full mechanism description and per-cell
ratios.
"""
import torch  # noqa: F401  (referenced in dtype literals)


# ---------------------------------------------------------------------------
# K-1938 / K-1926 — P41 N=608 K-COMPLEMENT verified-winner cells.
#
# Auto-generated from scripts/build_n608_frozenset.py against the paired
# n=30 HIP-graph hot-cache bench on MI300X (job id 24574, node
# useocpm2m-097-137).  15 of 18 cohort cells qualify (gate: hbl/tb<=0.85
# and paired-t p<0.05).
# ---------------------------------------------------------------------------
_P41_SKINNY_N608_KCOMPL_VERIFIED_WIN = frozenset({
    (2048, 608, 4096, "torch.bfloat16"),   # hbl/tb=0.761 p=1.0e-19
    (2048, 608, 8192, "torch.bfloat16"),   # hbl/tb=0.683 p=3.9e-39
    (2048, 608, 16384, "torch.bfloat16"),  # hbl/tb=0.507 p=6.2e-42
    (4096, 608, 16384, "torch.bfloat16"),  # hbl/tb=0.688 p=1.5e-36
    (8192, 608, 4096, "torch.bfloat16"),   # hbl/tb=0.665 p=3.0e-48
    (8192, 608, 8192, "torch.bfloat16"),   # hbl/tb=0.607 p=2.4e-47
    (8192, 608, 16384, "torch.bfloat16"),  # hbl/tb=0.490 p=2.2e-53
    (2048, 608, 4096, "torch.float16"),    # hbl/tb=0.778 p=3.2e-16
    (2048, 608, 8192, "torch.float16"),    # hbl/tb=0.661 p=1.3e-43
    (2048, 608, 16384, "torch.float16"),   # hbl/tb=0.508 p=2.1e-37
    (4096, 608, 8192, "torch.float16"),    # hbl/tb=0.827 p=2.1e-34
    (4096, 608, 16384, "torch.float16"),   # hbl/tb=0.678 p=9.0e-31
    (8192, 608, 4096, "torch.float16"),    # hbl/tb=0.681 p=1.3e-44
    (8192, 608, 8192, "torch.float16"),    # hbl/tb=0.622 p=3.4e-44
    (8192, 608, 16384, "torch.float16"),   # hbl/tb=0.506 p=7.4e-50
})

# Rejected cells (cohort cells that did NOT meet the gate; documented
# for completeness so future K-19xx work can revisit them under
# different K/M tilings):
#   (4096, 608,  4096, "torch.bfloat16")  reject=ratio  hbl/tb=0.922
#   (4096, 608,  8192, "torch.bfloat16")  reject=ratio  hbl/tb=0.859
#   (4096, 608,  4096, "torch.float16")   reject=ratio  hbl/tb=0.889


def _p41_skinny_n608_kcompl_route_to_hbl(M, N, K, a_dtype, b_dtype):
    """K-1938 — P41 N=608 K-COMPLEMENT verified-winner check.

    Returns True iff (M, 608, K, a_dtype) is in the verified-winner
    frozenset built from the paired n=30 HIP-graph hot-cache bench on
    MI300X with HBL ratio <= 0.85 and paired-t p < 0.05.

    R-1811.WAVE-MISALIGNMENT-IS-ROOT-MECHANISM:
      608 mod 64  = 32   (matches P32-P40 cluster signature)
      608 mod 128 = 96   (partial tail-wave on BLOCK_N=128, same modular
                          class as N=96 / N=224 / N=352 / N=480)
    """
    if N != 608:
        return False
    if a_dtype is not b_dtype:
        return False
    return (M, N, K, str(a_dtype)) in _P41_SKINNY_N608_KCOMPL_VERIFIED_WIN


def route_to_hbl(M, N, K, a_dtype, b_dtype):
    """Return True iff the (M,N,K,dtype) tuple has been benchmarked and
    verified to win on hipBLASLt over tritonblas by >= 17.6%.

    Currently the only active alias-stack slot is P41 (N=608, K-1938).
    Prior tickets in the K-185x..K-192x series characterised the
    N in {384, 416, 448, 480, 512, 544, 576} rungs but landed via
    the upstream config table rather than via this dispatcher; future
    K-19xx work will fold those into per-N slots here.
    """
    if _p41_skinny_n608_kcompl_route_to_hbl(M, N, K, a_dtype, b_dtype):
        return True
    return False
