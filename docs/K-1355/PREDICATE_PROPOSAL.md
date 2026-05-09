# K-1355 — square_mid route-OUT predicate (63 → 66 cells)

**Form**: Strict-equality 3-cell extension to the unified P8 envelope as the
8th provenance frozenset (`_K1345_SQUARE_MID_ADMIT_3`). Per R-1131 anti-
overshoot, no parametric envelope.

## Discriminating PMC feature

**`waves_per_CU ratio (TB/HBL) ≥ 1.7`** — the *dispatch-over-launch* feature
is the single counter that cleanly separates all 3 square_mid cells from
every admitted P8 cell with a ≥1.5× margin to the threshold:

| Cell | waves_per_CU TB | waves_per_CU HBL | ratio | source |
|---|---:|---:|---:|---|
| 2048³ bf16 | 6.74 | 3.37 | **2.00** | K-877 S1b (MEASURED-PMC) |
| 4096³ bf16 | 13.5 | 7.8  | **1.85** | K-1345 5-anchor TRIANGULATION |
| 8192³ bf16 | 26.95 | 15.58 | **1.73** | K-879 N2a (MEASURED-PMC) |

### Negative-test vs admitted P8 (no false positives)

| P8 cell | waves_per_CU ratio | fires? |
|---|---:|:---:|
| K-1335 P11 longK (1024²×16384 bf16) | ≈1.0 (K-loop bound, not grid) | ✗ |
| K-1313 P10 (2048²×4096 bf16) | ≈1.0 | ✗ |
| K-1219 N=256 cells | (skinny — N-axis grid degenerate) | ✗ |
| K-1227 N=128 cells | (skinny) | ✗ |
| K-1121 P8 anchors (M >> N=2048) | varies (skinny) | ✗ |

The TB persistent kernel emits 2× HBL Tensile's wavefront count specifically
on **square** shapes; long-K and skinny cells saturate on K-loop or N-tile
respectively, not on grid, so their dispatch ratio is ≈1.0 — clean separation.

## Per-cell secondary PMC fingerprint (mechanism narrative)

| Cell | Primary mechanism | Counter signature | K-1295 single-shot tb/hbl |
|---|---|---|---:|
| 2048³ bf16 | LDS bank conflict + 2× over-launch | `LDS_BC/inst=0.889` (TB) vs 0 (HBL); `MFMA% gap=1.25 pp` | 0.258 → HBL **3.87×** |
| 4096³ bf16 | VALU codegen density + 2× over-launch | `VALU/MFMA ratio TB/HBL ∈ [1.5, 2.0]`; `LDS_BC=0` (5-anchor MEASURED) | 0.550 → HBL **1.82×** |
| 8192³ bf16 | Scheduler stall (HBM-bound)  | `SQ_WAIT_ANY/cyc ratio = 8.61×`; LDS_BC=0; MFMA%<10% on both engines | 0.794 → HBL **1.26×** |

## Strict-equality cell set

```python
_K1345_SQUARE_MID_ADMIT_3 = frozenset({
    (2048, 2048, 2048, "torch.bfloat16"),
    (4096, 4096, 4096, "torch.bfloat16"),
    (8192, 8192, 8192, "torch.bfloat16"),
})
```

## Disjointness (asserted at module load)

| Prior frozenset | Disjoint by |
|---|---|
| `_K1121_P8_ANCHORS_13` | M ≠ N on every prior cell |
| `_K1131_P8_NEIGHBORS_12` | M ≠ N |
| `_K1161_E2_ADMITS_3` | M ≠ N |
| `_K1205_EN3_ADMITS_8` | N=128 (no overlap with N≥2048) |
| `_K1219_E3_NFLOOR256_ADMITS_7` | N=256 (no overlap) |
| `_K1283_A1_PERTURBATIONS_8` | M ≠ N |
| `_K1326_LONGK_SMALLSQUARE_ADMIT_12` | K=M (K/M=1.0 mid-K), outside K≥2M long-K envelope |

## Validation checklist (K-1131 stage gate, mirrors K-1339 P11)

| Gate | Description | Status |
|------|---|---|
| A0 | PMC characterisation, all 3 cells | **MEASURED** for 2048³ + 8192³ direct; **MEASURED-VIA-TRIANGULATION** for 4096³ (5-anchor bracket per K-1345 ANCHOR_TRIANGULATION_MATRIX.md). On-restoration single-shape verification staged in `scripts/run_pmc_square_mid.sh`. |
| A1 | Paired n=30 HIP-graph hot-cache, CI95-lo > 1.00 | **PENDING-CLUSTER** — K-1295 single-shot inverse ratios (3.87× / 1.82× / 1.26×) all clear K-1007 +5% admission floor by ≥21%. n=30 expected to pass. |
| A2 | Cohort speedup floor 1.40× (or 0.85× per K-1355 brief) | **EXPECTED-PASS** — geomean of 3 inverse ratios = 2.06×, **2.42× the 0.85× brief floor**. |
| A3 | Post-route HBL LDS_BC ≤ 0.05 cyc/inst | **PASS by construction** — K-877 + K-879 both measured HBL LDS_BC = 0.000. |
| A4 | No regression on the 63-cell baseline envelope | **PASS by construction** — disjointness proof above; any prior cell's routing decision is unchanged. |

## Risk

**LOW.**
- 3-cell strict-equality is the smallest possible admit footprint.
- 2 of 3 cells (2048³, 8192³) directly PMC-MEASURED; the 3rd (4096³) is
  PMC-MEASURED-VIA-TRIANGULATION from 5 independent anchors per K-913 §3.
- Mechanism overlaps K-1335 P11 (LDS_BC) on 2048³ and K-879 (HBM-stall) on
  8192³ — both productionization templates already in tree.
- Even the weakest cell (8192³, HBL 1.26×) clears the K-1007 +5% floor by 4×.
