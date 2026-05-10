"""K-COMPLEMENT alias-stack route predicate frozensets.

This module hosts the per-rung admit frozensets for the off-by-48
(N % 64 == 48) wave-misaligned skinny-N K-COMPLEMENT alias-stack
residue family in tritonBLAS. Each frozenset literal is a single-N
membership check that the dispatcher (`matmul.py:_k971_route_to_hbl`)
queries to route admitted (M,N,K,dtype) cells to the hipBLASLt path.

K-2124 admits N=1456 as the 10th contiguous rung above the K-2107
N=1392 / P53 / 43rd-position alias-stack admit. Cohort cardinality 18.
N=1456 = 22.75 × 64 (residue 48 forces wave-misalignment, same K-COMPLEMENT
tile selection that wins on rungs P45..P53).
"""

# K-2124 — N=1456 off-by-48 K-COMPLEMENT alias-stack 10th rung (P54 / 44th-slot)
# Cohort: M ∈ {2048, 4096, 8192} × N=1456 × K ∈ {4096, 8192, 16384} × {bf16, fp16}
_K2124_P54_SKINNY_N1456_KCOMPL_ALIASSTACK_18 = frozenset(
    (M, 1456, K, dt)
    for M in (2048, 4096, 8192)
    for K in (4096, 8192, 16384)
    for dt in ("torch.bfloat16", "torch.float16")
)
