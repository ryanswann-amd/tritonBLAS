# `_k971_route_to_hipblaslt` — routing-predicate roadmap (placeholder)

> Status: **planning placeholder**. No implementation exists in `main` yet.
> Tracking ticket: K-1402 (productionize 12-cell skinny_N256 K-COMPLEMENT
> route-OUT envelope). Requires K-1367 / K-1387 / K-1389 prerequisites
> to land first.

## Why this file exists

Several downstream productionization tickets (K-1367, K-1387, K-1389,
K-1397, K-1402, …) are scoped against a *prospective* dispatch helper
called `_k971_route_to_hipblaslt` and a frozen-set predicate framework
that lives, by convention, in `include/tritonblas/_route_predicate.py`.

Neither symbol exists in `main` as of upstream HEAD `95e2c47`
(`Add compute-communication overlap benchmark suite (#84)`):

```text
$ git grep -l '_k971\|_route_predicate\|KCOMPL_ROUTEOUT' origin/main -- include/
(no output)
```

Productionization tickets that try to *stack on top of* this
non-existent framework will be unshippable until the framework lands.
This document records that fact so reviewers don't have to re-derive
the situation from scratch each time a stacked PR is opened against
the wrong base.

## Required upstream order

The dispatch chain has a mandatory landing order. Skipping a tier
means the predicate-position number in later tickets is undefined.

| Order | Ticket(s) | Brings in |
|-------|-----------|-----------|
| 1 | (design) | `_k971_route_to_hipblaslt` skeleton + `_route_predicate.py` module |
| 2 | K-1367   | `_K1367_P13_SKINNY_N128_KCOMPL_ROUTEOUT_18` frozen-set, predicate position 7 |
| 3 | K-1387   | P13 productionization wiring (55 → 73-cell envelope) |
| 4 | K-1389   | rename / consolidation of K-1367 frozen-set; predicate position 8 |
| 5 | **K-1402** | `_K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12` frozen-set, **predicate position 9** (73 → 85-cell envelope) |

## K-1402 deliverable, when unblocked

When tiers 1–4 above are merged to `ROCm/tritonBLAS:main`, the K-1402
diff is small and self-contained:

* +1 frozen-set definition `_K1397_P13_SKINNY_N256_KCOMPL_ROUTEOUT_12`
  in `include/tritonblas/_route_predicate.py`, adjacent to the
  existing K-1367 frozen-set.
* +1 OR-clause appended to `_k971_route_to_hipblaslt` at the 9th
  position of the dispatch chain.
* +1 unit test `tests/test_route_K1397.py`:
  - 12 cells in the admit set must route to hipBLASLt
  - 4 boundary-perturbed cells (K=4096, K=16384, N=128, N=512) must
    NOT match (strict-equality check)
  - envelope cardinality grows by exactly 12 (73 → 85)
* Validation on c42 / MI300X with paired n=30 HIP-graph hot-cache
  benchmark: geomean speedup ≥1.0× (95 % bootstrap CI lower bound
  ≥0.97×) on the admit set, ≤2 % regression on controls.
* Cross-arch backtest on MI325X and MI355X per the K-1373 / K-1356
  protocol; carve-OUT any MI355X cell with <0.95× CI95 lower bound.

A standalone, mock-based unit-test design demonstrating the four
assertions above is preserved in the K-1402 task workspace at
`scripts/test_K1402_route_intent.py` and runs green (4/4) against an
in-file router stub. It can be ported to `tests/test_route_K1397.py`
with a single import-line change once the real router exists.

## Mechanism / motivation

The 12-cell admit envelope (M ∈ {2048, 4096, 8192} × N = 256 ×
K ∈ {2048, 32 768} × {bf16, fp16}) is the last residual skinny_N256
bucket from K-1365's post-P12 4-bucket decomposition. K-1397 measured
1.04 – 1.12× per-cell speedups on this set; cohort-level lift is
projected at +0.8 – 1.2 % geomean. Mechanism: hipBLASLt's split-K
kernel selection wins over `triton.persistent_matmul` at extreme
aspect ratios where LDS bank conflicts dominate (consistent with
K-913 longK_smallSquare PMC findings).

This file will be deleted in the same commit that lands the real
`_k971_route_to_hipblaslt` skeleton.
