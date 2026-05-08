# K-827 paired ON/OFF cohort + guard data

Hardware: c42 / MI300X / m20u13 · gfx942 · Slurm 10914
Container: rocm/pytorch (project ROCm image)
Methodology: HIP-graph kernel-only, n_capture=10, n_warm=5, n_rounds=50
Both arms run in a single process with shared CUDA allocator state.
OFF arm patches `tritonblas.matmul.maybe_dispatch_splitk2` to `lambda a,b,out: False`.
ON arm restores the original dispatch.

## Cohort (n=6, override fires)

| M | N | K | dtype | OFF (ms) | ON (ms) | hipBLASLt (ms) | on/off | on/hbl | off/hbl |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 2048 | 2048 |  4096 | float16  | 0.0935 | 0.2029 | 0.0777 | **2.171** | 2.612 | 1.245 |
| 2048 | 2048 |  4096 | bfloat16 | 0.0914 | 0.1947 | 0.0771 | **2.131** | 2.525 | 1.171 |
| 2048 | 2048 |  8192 | float16  | 0.1768 | 0.2980 | 0.1537 | **1.686** | 1.939 | 1.147 |
| 2048 | 2048 |  8192 | bfloat16 | 0.1714 | 0.2898 | 0.1376 | **1.691** | 2.107 | 1.248 |
| 2048 | 2048 | 16384 | float16  | 0.3485 | 0.4698 | 0.2537 | **1.348** | 1.852 | 1.391 |
| 2048 | 2048 | 16384 | bfloat16 | 0.3403 | 0.4516 | 0.2403 | **1.327** | 1.879 | 1.422 |

**cohort geomean on/off = 1.69**  → ON is 69 percentage points SLOWER than OFF
**cohort geomean on/hbl = 2.10**  → ON is 110 pp SLOWER than hipBLASLt
**cohort geomean off/hbl = 1.27** → OFF baseline is 27 pp slower than hipBLASLt
ON makes the gap to hipBLASLt **wider** (1.27→2.10) on every cohort cell.

## Guard (n=12, override does NOT fire)

| M | N | K | dtype | OFF (ms) | ON (ms) | on/off | in_cohort |
|---:|---:|---:|---|---:|---:|---:|---:|
| 4096 | 4096 |  4096 | float16  | 0.2538 | 0.2548 | 1.004 | 0 |
| 4096 | 4096 |  4096 | bfloat16 | 0.2435 | 0.2462 | 1.011 | 0 |
| 4096 | 4096 |  8192 | float16  | 0.4896 | 0.4932 | 1.007 | 0 |
| 4096 | 4096 |  8192 | bfloat16 | 0.4670 | 0.4732 | 1.013 | 0 |
| 4096 | 4096 | 16384 | float16  | 0.9956 | 1.0043 | 1.009 | 0 |
| 4096 | 4096 | 16384 | bfloat16 | 0.9581 | 0.9618 | 1.004 | 0 |
| 1024 | 1024 |  4096 | float16  | 0.0406 | 0.0408 | 1.005 | 0 |
| 1024 | 1024 |  4096 | bfloat16 | 0.0398 | 0.0392 | 0.985 | 0 |
| 1024 | 1024 |  8192 | float16  | 0.0784 | 0.0788 | 1.005 | 0 |
| 1024 | 1024 |  8192 | bfloat16 | 0.0774 | 0.0773 | 0.999 | 0 |
| 1024 | 1024 | 16384 | float16  | 0.1541 | 0.1589 | 1.031 | 0 |
| 1024 | 1024 | 16384 | bfloat16 | 0.1522 | 0.1515 | 0.995 | 0 |

**guard geomean on/off = 1.006** → within ±2 pp noise, gate correctly does not fire on any guard cell.

## Ablation summary (109-shape K-805 family)

The override's `lookup_splitk2_override` table is keyed on exact equality of
`(M, N, K, dtype)` with the additional precondition `a.dtype == b.dtype == out.dtype`.
The 6 cohort entries are exhaustively enumerated above.  An 81-cell exhaustive
sweep over `M ∈ {1024, 2048, 4096} × N ∈ {1024, 2048, 4096} × K ∈ {4096, 8192, 16384}
× dtype ∈ {fp16, bf16, fp32}` confirms the gate fires on exactly 6 of 81 keys
(the 6 cohort entries) and never on the other 75 keys; the 12-cell guard above
is the residual + nearest-neighbor subset of that ablation.

## Verdict

NO-LAND.  Fourth independent reproduction of the K-795 / K-809 / K-811 / K-817
SHIP-NULL on the split-K=2 atomic-reduction prototype: the override regresses
every cohort cell 33-117 pp on/off and widens the hipBLASLt gap on every cell.
K-axis monotonicity reproduces (smaller K regresses harder because the FP32
atomic-reduction fixed cost dominates when there is less work to amortise it
over).  Production code path is unchanged on this branch — the kernel module
is preserved unwired as a re-runnable diagnostic for the next round of
investigation once the Triton-AMD compiler backend exposes MIWT chaining
(K-760 H1) and LDSB1-equivalent LDS swizzle (K-760 H2).
