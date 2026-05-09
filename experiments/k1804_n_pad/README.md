# K-1804 — N-padding dispatcher gate prototype (negative result)

## TL;DR

**Outcome: REJECTED.** Padding wave-misaligned N to the next BLOCK_N tile size
({64, 128, 256}) **slows down** tritonblas on the K-1795 12-cell cohort by
~17% (gmean speedup 0.827×) on MI300X bf16 mid-K, while leaving aligned
controls within noise. The hypothesis from K-1795 — that wave-misaligned N
explains the residual gap vs hipBLASLt and that padding can recover it — is
not supported by this prototype.

This experiment lives entirely under `experiments/` and makes **zero changes
to `tritonblas.matmul`** or any other production code path. The padding
behavior is applied externally by the harness (caller-side), so there is no
env-gated dead code in the hot dispatch path.

## Files

| File | Purpose |
|---|---|
| `bench_n_pad.py` | Harness: paired n=30 hot-cache benchmark on the 12-cell cohort, padded vs unpadded, with hipBLASLt baseline. |
| `test_n_pad_bit_exact.py` | Pytest: asserts bit-identity of padded-then-sliced output for all 6 gated cells; asserts the production matmul is untouched by `TRITONBLAS_N_PAD` env var; asserts the pad function correctly extends to 256 for N=192. |
| `bench_n_pad_bf16.csv` | Per-cell measurements (12 rows). |
| `bench_n_pad_bf16.summary.json` | Cohort gmeans for aligned/misaligned/gated/not_gated. |

## Run

```bash
# bit-exactness + gate-coverage tests
pytest -xvs experiments/k1804_n_pad/test_n_pad_bit_exact.py

# benchmark
python3 experiments/k1804_n_pad/bench_n_pad.py \
    --out experiments/k1804_n_pad/bench_n_pad_bf16.csv --n 30 --warmup 5
```

## Pad rule

Smallest BLOCK_N tile size that fits N exactly, from {64, 128, 256}:

| N | n_pad | rationale |
|---|---|---|
| 48 | 64  | 1 wave  |
| 64 | 64  | already aligned (no-op) |
| 80 | 128 | 1 BLOCK_N=128 tile |
| 128 | 128 | already aligned (no-op) |
| 192 | 256 | 1 BLOCK_N=256 tile (this addresses the v1 gate's silent no-op on 192%64==0) |
| ≥256 | round up to next 64 | already large enough |

## Result

Cohort: M=4096, dtype=bf16, K∈{4096, 8192, 16384}, N∈{48, 64, 80, 128, 192} on MI300X (one MI300X GPU, paired n=30 HIP-graph hot-cache).

| cohort | n_cells | gmean speedup (pad / base) | bit-exact |
|---|---|---|---|
| **gated wave-misaligned** (N∈{48,80,192}) | 6 | **0.827×** (↓17.3%) | ✓ |
| aligned controls (N∈{64,128})             | 6 | 0.978× (↓2.2%, within noise) | ✓ |

All 6 gated cells produce bit-identical output (`max |Δ| = 0`).

The pad approach increases work proportionally — N=80→128 is +60% flops,
N=192→256 is +33% flops — and the throughput gain from a larger tile does
not offset the extra work and the B-tensor copy cost. The hipBLASLt residual
gap on this cohort must be addressed by a different mechanism (likely
BLOCK_N selection within tritonblas, not external N padding).

## Disposition

- **No production code change is needed or recommended.**
- Experiment artifacts retained in `experiments/k1804_n_pad/` for posterity
  and to anchor the negative-result claim with reproducible numbers.
- Follow-up: instead of N padding, investigate per-tile BLOCK_N override on
  this cohort (separate ticket).
