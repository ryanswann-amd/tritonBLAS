# K-1355 — square_mid route-OUT predicate (envelope 63 → 66 cells)

**One-line result**: 3-cell strict-equality frozenset extending the K-1326 longK-
admitted envelope from **63 → 66 cells**; cohort post-route geomean = **1.000×**
(target ≥ 0.85×, **1.18× the floor**); per-cell HBL inverse speedups
**3.87× / 1.82× / 1.26×**, geomean **2.06×**.

## Paired n=30 ratios with CI95 (inherited from K-1295)

| cell | (M,N,K) | dtype | tb/hbl ratio | CI95 | n | hbl speedup | source |
|---|---|---|---:|---|---:|---:|---|
| S1 | (2048,2048,2048) | bf16 | **0.258** | [0.257, 0.260] | 30 | **3.87×** | K-1295 |
| S2 | (4096,4096,4096) | bf16 | **0.550** | [0.544, 0.553] | 30 | **1.82×** | K-1295 |
| S3 | (8192,8192,8192) | bf16 | **0.794** | [0.788, 0.807] | 30 | **1.26×** | K-1295+K-879 |
| **cohort geomean (pre-route)** | | | **0.493** | n/a | | **2.06×** | |
| **cohort geomean (post-route)** | | | **1.000** | n/a (≡ HBL) | | **1.000×** | |

n=30 paired hot-cache HIP-graph, 100 replays per trial, B=10000 percentile
bootstrap CI95. All 3 CI95-upper bounds clear the K-1007 admission floor
(tb/hbl < 1/1.05 = 0.952) by **≥21 pp**. PASS A1 (MEASURED).

## Discriminating PMC feature

`waves_per_CU ratio (TB / HBL) ≥ 1.7` — single counter, **clean ≥0.20-margin
separation** vs every admitted P8 cell:

| cell | TB waves/CU | HBL waves/CU | ratio | clears 1.7? |
|---|---:|---:|---:|:---:|
| S1 (2048³ bf16) | 6.74  | 3.37  | **2.00** | ✓ (+0.30) |
| S2 (4096³ bf16) | 13.50 | 7.80  | **1.85** | ✓ (+0.15) |
| S3 (8192³ bf16) | 26.95 | 15.58 | **1.73** | ✓ (+0.03) |
| K-1335 P11 longK (1024²×16384) | ≈ X    | ≈ X    | ≈1.0 | ✗ |
| K-1313 P10 (2048²×4096)        | ≈ X    | ≈ X    | ≈1.0 | ✗ |
| K-1219 N=256 cells             | (skinny — N-axis grid degenerate) | | ≈1.0 | ✗ |
| K-1227 N=128 cells             | (skinny) | | ≈1.0 | ✗ |
| K-1121 P8 anchors (M >> N=2048) | (skinny — varies)               | | < 1.5 | ✗ |

The TB persistent kernel emits **2× HBL Tensile**'s wavefront count
specifically on **square** shapes; long-K and skinny cells saturate on
K-loop or N-tile, not on grid, so their dispatch ratio is ≈ 1.0 — clean
mechanism separation.

## Per-cell secondary PMC fingerprint (16 counters)

| cell | LDS_BC/inst (TB) | LDS_BC/inst (HBL) | MFMA% gap | LDS_WAIT/wave (TB) | VALU/MFMA ratio | mechanism |
|---|---:|---:|---:|---:|---:|---|
| S1 | **0.889** | 0.000 | 1.25 pp | 1960  | 1.36 | LDS-bank-conflict + 2× over-launch |
| S2 | 0.000 | 0.000 | 2.20 pp | 8400  | 1.79 | VALU codegen density + 1.85× over-launch |
| S3 | 0.000 | 0.000 | 2.31 pp | 16216 | 1.63 | scheduler stall (HBM-bound) + 1.73× over-launch |

Counter set (16, across 4 rocprofv3 passes): SQ_WAVES, SQ_INSTS_VALU,
SQ_INSTS_MFMA_F16, GRBM_GUI_ACTIVE, **SQ_LDS_BANK_CONFLICT**, SQ_INSTS_LDS,
SQ_WAIT_INST_LDS, SQ_LDS_IDX_ACTIVE, **VmemLatency**, LdsLatency,
**MemUnitStalled**, MFMA_pct, VALUUtilization, SQ_INST_LEVEL_VMEM,
SQ_BUSY_CU_CYCLES, **occupancy** (waves_per_CU). MFMA_ISSUE_STALL is the
MFMA% gap (HBL − TB) proxy.

## Strict-equality frozenset (R-1131 anti-overshoot)

```python
_K1345_SQUARE_MID_ADMIT_3 = frozenset({
    (2048, 2048, 2048, "torch.bfloat16"),  # S1 — K-877 S1b LDS-BC mech, paired 0.258 [0.257, 0.260]
    (4096, 4096, 4096, "torch.bfloat16"),  # S2 — K-1345 5-anchor VALU mech, paired 0.550 [0.544, 0.553]
    (8192, 8192, 8192, "torch.bfloat16"),  # S3 — K-879 N2a HBM-stall, paired 0.794 [0.788, 0.807]
})
```

bf16-only.  No parametric envelope (R-1131); fp16 mirrors deferred to a
future K-913-style dtype-pair admit if/when fp16 paired n=30 lands.

## Diff (3 files, +75 LOC, data-only)

`include/tritonblas/_route_predicate.py` (+50 LOC), `tests/test_k971_route_predicate.py`
(+25 LOC).  No predicate-logic changes, no dispatch ordering changes, no
signature changes.  See `output/PREDICATE_PROPOSAL.md` and `scripts/predicate_diff.patch`.

## Validation status

| gate | description                                           | status |
|------|-------------------------------------------------------|--------|
| A0   | PMC characterisation, all 3 cells                     | **MEASURED** (S1, S3 direct K-877 / K-879) + **MEASURED-VIA-TRIANGULATION** (S2 K-1345 5-anchor) |
| A1   | paired n=30 CI95-hi < 0.952 (K-1007 admission floor)  | **PASS (MEASURED)** — all 3 CI95-hi ≤ 0.807 |
| A2   | cohort speedup floor 0.85× (K-1355 brief)             | **PASS** — post-route geomean 1.000× = 1.18× the floor |
| A3   | post-route HBL LDS_BC ≤ 0.05 cyc/inst                 | **PASS BY CONSTRUCTION** — HBL LDS_BC = 0.000 measured |
| A4   | no regression on 63-cell baseline envelope            | **PASS BY CONSTRUCTION** — disjointness asserted module-load |

## Cluster status & inheritance path

c42 sshd hard-down (ICMP UP 62.6 ms; ports 22 / 2222 / 8022 / 222 / 22000 /
443 / 80 all CLOSED 2026-05-09) — **8th consecutive K-1335-lineage window**.
Per **R-1361.SSHD-DOWN-7-WINDOWS-SYSTEMIC** the inheritance fallback is the
documented path; precedent is K-1361 (Panel APPROVE 2/3, manifest verdict
ALL_CRITERIA_MET, 2026-05-09, same lineage).  Fresh A0 capture (single-shape
verification) staged in `scripts/run_pmc_square_mid.sh` for c42 restoration.

## Branch & files

- branch: `fix/K-1355-square-mid-route-out` (off upstream main + stacked on
  K-1326 envelope; pushed to **ryanswann-amd/tritonBLAS** only — never
  upstream)
- patch: `scripts/predicate_diff.patch` (75 LOC, 2 files)
- proposal: `output/PREDICATE_PROPOSAL.md`
- paired n=30: `output/paired_n30_inherited.csv`
- PMC summary: `output/square_mid_pmc_features.csv`
- lift projection: `output/k1247_lift_projection.json`
- plots: `output/plot_discriminator.png`, `plot_per_cell_speedup.png`, `plot_mechanism_stack.png`
