import functools
import os
import random
import time
from typing import Any, Dict, Optional, Tuple

import torch
from torch.library import triton_op, wrap_triton
from torch._subclasses.fake_tensor import is_fake
import triton

from .kernels import persistent_matmul, ws_persistent_matmul, streamk_matmul, ws_streamk_matmul
from .kernels.fp4_matmul import fp4_matmul
from .origami import OrigamiMatmulSelector
from .config import MatmulConfig, matmul_preamble, COUNTER_STRIDE
from . import guarded_override as _go
# NOTE: no override package is imported here.  GuardedOverride instances
# register themselves explicitly only after passing the K-883 §5 LAND
# verdict via run_falsification (K-901 design).  Keeping the dispatch site
# override-naive means an empty registry is the safe default.

# K-1003 R-K979 P5 Gate-0 admission predicate + K-905/K-971 anchors live in
# a torch-free helper module so unit/integration tests can import them
# without booting the GPU stack (triton/origami/torch.cuda init chain).
from ._route_predicate import (
    K971_ROUTE_TABLE as _K971_ROUTE_TABLE,
    R_K979_P5_route_to_hbl as _R_K979_P5_route_to_hbl,
    R_K1037_P6_admit_wpeu1 as _R_K1037_P6_admit_wpeu1,
    _p8_mfma_issue_stall_routeout as _R_K1144_P8_mfma_issue_stall_routeout,
    R_K1142_E1_route_to_hbl as _R_K1142_E1_route_to_hbl,
    # K-1361 (S-002): P12 square_mid PMC-driven 4-cell route-OUT (6th-position).
    _k1361_p12_square_mid_routeout as _R_K1361_P12_square_mid_routeout,
    # K-1367 (S-002): P13 skinny_N128 K-COMPLEMENT 18-cell route-OUT (7th-position).
    _k1367_p13_skinny_n128_routeout as _R_K1367_P13_skinny_n128_routeout,
    # K-1397 (S-002): P13 skinny_N256 K-COMPLEMENT 12-cell route-OUT (8th-position).
    _k1397_p13_skinny_n256_routeout as _R_K1397_P13_skinny_n256_routeout,
    # K-1417 (S-002): P15 skinny_N512 K-COMPLEMENT EXTENSION 12-cell route-OUT
    # (9th-position).
    _k1409_p15_skinny_n512_routeout as _R_K1409_P15_skinny_n512_routeout,
    # K-1429 (S-002): P16 skinny_N1024 K-COMPLEMENT 29-cell route-OUT
    # (10th-position).
    _k1429_p16_skinny_n1024_routeout as _R_K1429_P16_skinny_n1024_routeout,
    # P17 skinny_N512 K-COMPLEMENT BASE 17-cell route-OUT (11th-position) —
    # completes the P15 N=512 EXTREMES sibling at the BASE K band.
    _k1437_p17_skinny_n512_kcompl_base_routeout as _R_K1437_P17_skinny_n512_kcompl_base_routeout,
    # K-1478 (S-002): P19 skinny_N16384 K-COMPLEMENT 30-cell route-OUT
    # (12th-position) — extends the K-COMPLEMENT N-ladder one bucket up
    # to N=16384 (full K-grid).  30/30 admit at strict 1.05 gate; cohort
    # geomean tb/hbl = 1.174×.  Naturally disjoint with all P1-P17.
    _k1478_p19_skinny_n16384_routeout as _R_K1478_P19_skinny_n16384_routeout,
    # K-1503 (S-002): P21 skinny_N256 K-COMPLEMENT K-mid-band 17-cell route-OUT
    # (13th-position) — closes the K-1397 P13 N=256 K-mid-band gap (K in
    # {4096, 8192, 16384}).  17/30 admits at >=1.05x; cohort geomean 1.457x.
    _k1503_p21_skinny_n256_kmid_routeout as _R_K1503_P21_skinny_n256_kmid_routeout,
    # K-1513 (S-002): P22 skinny_N32768 K-COMPLEMENT 30-cell route-OUT
    # (14th-position) — closes the top rung of the K-COMPLEMENT N-ladder
    # at N=32768 on the full K-grid.  30/30 admit at strict 1.05 gate;
    # cohort geomean tb/hbl ≈ 1.118×.  Naturally disjoint with all P1-P21.
    _k1513_p22_skinny_n32768_routeout as _R_K1513_P22_skinny_n32768_routeout,
    # K-1552 (S-002): P23 skinny_N512 K-COMPLEMENT alias-stack 30-cell
    # route-OUT (15th-position).  ALIAS to P15 ⨄ P17 ⨄ P5 — the K-1534-
    # verified N=512 envelope (cohort geomean tb/hbl = 1.454×, range
    # 1.093×–2.111×, 0 regressions) is already routed 30/30 by upstream
    # layers; this slot freezes the admit set under a single symbol and is
    # load-bearing only if any upstream layer is ablated.
    _k1552_p23_skinny_n512_kcompl_aliasstack_routeout
        as _R_K1552_P23_skinny_n512_kcompl_aliasstack_routeout,
    # K-1566 (S-002): P24 skinny_N4096 K-COMPLEMENT 30-cell route-OUT
    # (16th-position).  Closes the previously-empty N=4096 rung of the
    # K-COMPLEMENT N-ladder (between P21 N=2048 and P19 N=16384) on the
    # full K-grid {2048, 4096, 8192, 16384, 32768}.  30/30 admit at the
    # strict 1.05 gate (K-1553 paired n=30 + B=10000 vectorised paired
    # bootstrap MI300X gfx942 vs the live post-K-1532 oracle); cohort
    # geomean tb/hbl = 1.234×, range 1.114×-1.501×.  Sibling-N firewall
    # disjoint with all P1-P23 except a single intentional 2-cell P12
    # alias overlap at (4096, 4096, 4096, {bf16, fp16}); 28 NEW cells +
    # 2 P12-alias cells.
    _k1566_p24_skinny_n4096_routeout as _R_K1566_P24_skinny_n4096_routeout,
    # K-1553 (S-002): the K-1553-named documentation-only handle is a
    # module-level alias of the K-1566 P24 frozenset
    # (`_K1553_P25_SKINNY_N4096_KCOMPL_ALIASSTACK_30 =
    # _K1566_P24_SKINNY_N4096_KCOMPL_ROUTEOUT_30`); no separate predicate
    # function is imported because the admit set is bit-identical and P24
    # fires first.  Per K-1489 reviewer-consensus precedent for
    # unreachable alias slots, the routing decision is fully carried by
    # P24 and the K-1553-named handle is documentation-only.
    # K-1611 (S-002): P26 skinny_N2048 K-COMPLEMENT alias-stack 30-cell
    # route-OUT (17th-position, load-bearing).  Closes the LAST untested
    # mid-N rung of the K-COMPLEMENT N-ladder at N=2048 on the full
    # K-grid; 30/30 admit at the strict 1.05 gate (K-1611 paired n=30 +
    # B=10000 vectorised paired bootstrap MI300X gfx942 vs the live post-
    # K-1581 oracle); cohort geomean tb/hbl = 1.451×, range
    # 1.108×-1.853×.  22 NEW cells + 8 alias cells (P12 + K971
    # union); alias overlaps fire BEFORE P26 in the dispatch chain.
    _k1611_p26_skinny_n2048_kcompl_aliasstack_routeout
        as _R_K1611_P26_skinny_n2048_kcompl_aliasstack_routeout,
    # K-1673 (S-002): P28 skinny_N128 K-COMPLEMENT alias-stack 30-cell
    # route-OUT (19th-position, load-bearing).  Closes the LAST untested
    # small-N rung (N=128) of the K-COMPLEMENT N-ladder at the dtype-
    # mirror gap (fp16 K ∈ {2048, 32768}); 30/30 admit at the strict 1.05
    # gate (K-1673 paired n=30 HIP-graph hot-cache MI300X gfx942 vs the
    # live post-K-1647 P27 oracle); cohort geomean tb/hbl = 1.694×, range
    # 1.162×-2.890×.  6 NEW cells (all fp16, K ∈ {2048, 32768}) + 24
    # alias cells (15 bf16 via R-K979 P5 Clause-3 + 9 fp16 K-mid via
    # K-1367 P13 N=128); alias overlaps fire BEFORE P28 in the dispatch
    # chain.
    _k1673_p28_skinny_n128_kcompl_aliasstack_routeout
        as _R_K1673_P28_skinny_n128_kcompl_aliasstack_routeout,
    # K-1700 P29 (20th-slot): N=64 K-COMPLEMENT alias-stack — 29 K-1709 admit cells.
    _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29,
    # K-1748 P30 (21st-slot): skinny_Nmid (N ∈ {384, 768, 1536}) K-COMPLEMENT
    # alias-stack — 34 K-1711 admit cells (closes 0/34 live-oracle gap).
    _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34,
    # P31 (22nd-slot): N=256 K-COMPLEMENT verified-winner subset — 28 cells
    # (M ∈ {2048,4096,8192} × N=256 × K ∈ {2048,4096,8192,16384,32768} ×
    # {bf16,fp16} minus 2 paired-n30 LOSER cells at (2048, 256, 2048, *)).
    _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28,
    # K-1881 (S-002): consolidated P32–P38 K-COMPLEMENT verified-winner
    # alias-stack — ONE 120-cell frozenset replacing what would otherwise
    # be 7 chained per-slot checks (P32 N=160 / P33 N=224 / P34 N=96 /
    # P35 N=288 / P36 N=320 / P37 N=352 / P38 N=384).  Extends the K-1864
    # 5-slot consolidation in-place to absorb the two newer K-COMPLEMENT
    # promotions before they could re-fragment the dispatcher into a
    # 7-probe chain.  Cells are inlined as literal tuples (single source
    # of truth) in `_route_predicate.py`; the legacy per-slot named
    # frozensets (`_P3{2..8}_..._WIN_{14,17,18}`) survive there ONLY as
    # N-axis projection views consumed by `tests/test_p3{2..8}_*_alias_
    # stack.py` — the dispatcher never references them.  Per-slot
    # mechanism / PMC RCA / provenance lives next to the canonical
    # inlined block in `_route_predicate.py` (P32 N=160 K-1794, P33 N=224
    # K-1794, P34 N=96 K-1818, P35 N=288 K-1832, P36 N=320 K-1843, P37
    # N=352 K-1843, P38 N=384 K-1857).
    _K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT_120,
    # K-1887 (S-002): P39 (30th-slot) N=448 K-COMPLEMENT verified-winner subset
    # — 18 cells (full M ∈ {2048,4096,8192} × N=448 × K ∈ {4096,8192,16384} ×
    # {bf16,fp16} grid; NO upstream alias overlap — N=448 is disjoint from
    # P30's N ∈ {384,768,1536} and from K-1881's N ∈ {96,160,224,288,320,
    # 352,384}).  K-1881-followup paired n=30 HIP-graph hot-cache + 3-pass
    # rocprofv2 PMC: 18/18 admit at strict ratio_TB/HBL ≥ 1.05 ∧ p<0.05 gate;
    # per-N geomean = 1.36×; same SCHEDULER_LDS A4 fingerprint as K-1857
    # P38 N=384 / K-1843 P36 N=320 (LDS_DOMINANT in 18/18, HBL SQ_LDS_BANK_
    # CONFLICT == 0 in all 18; N=448 = 1.75 × BN=256 → 0.75-wave tail tile).
    _P39_SKINNY_N448_KCOMPL_VERIFIED_WIN_18,
)



_tensor_cache = {}


def _k971_route_to_hbl(M, N, K, a_dtype, b_dtype, enable_streamk, work_stealing):
    # K-989: keep K-971 killswitch env var as the single L3 disable lever
    # (per K-883 R1 — one cohort, one disable). K-1003 layers the R-K979 P5
    # Gate-0 admission predicate AHEAD of the strict-equality table so the
    # broader K-984+K-989+K-931 cohort dispatches via closed-form rather
    # than per-shape entries; strict-equality table retained for K-905/K-971
    # mid-square long-K anchors that structurally collide with K-950 LAND.
    if os.environ.get("TRITONBLAS_DISABLE_K971") == "1": return False
    if enable_streamk or work_stealing or str(a_dtype) != str(b_dtype): return False
    # K-1144 (S-002): P8 MFMA-issue-stall direct hipBLASLt route-OUT for
    # the triply-validated 25-cell envelope (13 K-1121 anchors + 12 K-1131
    # neighbors).  Consulted BEFORE the K-1089 P6 admit so K-1121's paired
    # n=30 measurement evidence (hipBLASLt wins on S24, S29 at ~1.20-1.22x;
    # K-1131 N11 at >=1.15x) overrides the K-1089 envelope admit on the
    # small subset of cells where the two envelopes overlap.  K-1074's
    # paired n=30 LAND audit and K-1098's clause-by-clause backtest
    # falsified the K-1089 admit on those cells; P8 codifies the
    # correction.  No double-routing: for cells already routed OUT by P5
    # Clause-2 / K-1062 Clause-4, P8's strict-equality match returns the
    # same True verdict (frozenset O(1) lookup; harmless).
    if _R_K1144_P8_mfma_issue_stall_routeout(int(M), int(N), int(K), a_dtype): return True
    # K-1209-stacked / K-1216 (S-002): E1 axis-aligned envelope as defense-
    # in-depth AFTER P8 28-cell strict-equality. K-1209 ablation on K-931
    # always-uncovered top-40 (40 cells) confirmed E1 contributes 0 marginal
    # cells beyond P8+K-1175 (16/40 union vs 16/40 P8+K-1175); E1 is retained
    # for non-K-931 cohorts where the K-1142 -> K-1161 -> K-1175 audit chain
    # has not yet enumerated every K-1142-envelope-admittable cell. The
    # K-1142 carve-out (M >= 4480) ∧ (K >= 256) holds at 0 FPs on K-931.
    # E2 K-floor=128 EXCLUDED per K-1176 cross-arch failure (0/8 cells on
    # MI325X/MI355X) — the K-axis floor stays pinned at K=256.
    if _R_K1142_E1_route_to_hbl(int(M), int(N), int(K), a_dtype): return True
    # K-1089 (S-002): R-K1037 P6 structural surrogate admits MFMA-issue-stall
    # cells back to in-kernel dispatch with waves_per_eu=1 (set in the
    # persistent dispatch path below). When P6 admits, route-OUT (P5 + the
    # K-905/K-971 strict-equality table) is short-circuited for the cell.
    if _R_K1037_P6_admit_wpeu1(int(M), int(N), int(K), a_dtype): return False
    if _R_K979_P5_route_to_hbl(int(M), int(N), int(K), a_dtype): return True
    if (int(M), int(N), int(K), str(a_dtype)) in _K971_ROUTE_TABLE: return True
    # K-1361 (S-002): P12 square_mid PMC-driven 4-cell route-OUT (6th-position
    # envelope). Stacks AFTER the K971 LDS-BC table per K-1175 stacked-predicate
    # convention; productionises K-1345's Predicate-Q (square_mid 2048³) and
    # Predicate-R (square_mid 4096³) — see _K1295_P12_PMC_SQUARE_MID_ROUTEOUT_4
    # in _route_predicate.py for source measurement chain (K-877 / K-655 /
    # K-818 C2 / K-837 / K-879 anchors + K-913 §3 dtype invariance + K-1295
    # paired n=30 inheritance). Disjoint by construction with all P1–P11
    # sub-frozensets via cross-frozenset asserts at module load.
    if _R_K1361_P12_square_mid_routeout(int(M), int(N), int(K), a_dtype): return True
    # K-1367 (S-002): P13 skinny_N128 K-COMPLEMENT 18-cell route-OUT (7th-position
    # envelope). Stacks AFTER K-1361 P12 per K-1175 stacked-predicate convention;
    # closes the K-1308 skinny_N128 K-COMPLEMENT region (K >= 4096 above the
    # ~30 µs triton_op wrapper-overhead ceiling). Verified at paired n=30 +
    # B=10000 vectorised bootstrap CI95: 18/18 ROUTE-OUT, cohort geomean
    # tb/hbl=1.678×, min CI95-lo=1.124. PMC mechanism confirmed by 7-cell
    # rocprofv3 4-pass capture (TB SQ_LDS_BC/inst ≈ 1.45 vs HBL = 0; sharp
    # K-913 §3 LDS-bank-conflict discriminator on the persistent_matmul N=128
    # column-narrow LDS layout). Disjoint by construction with all P1–P12
    # sub-frozensets via cross-frozenset asserts at module load.
    if _R_K1367_P13_skinny_n128_routeout(int(M), int(N), int(K), a_dtype): return True
    # K-1397 (S-002): P13 skinny_N256 K-COMPLEMENT 12-cell route-OUT (8th-position
    # envelope). Stacks AFTER K-1367 P13 per K-1175 stacked-predicate convention;
    # closes the last K-1365 post-P12 4-bucket residual (skinny_N256 at K-axis
    # extremes K ∈ {2048, 32768}).  Mechanism: hipBLASLt's split-K kernel selection
    # wins over tritonblas persistent_matmul at extreme aspect ratios where LDS
    # bank conflicts dominate the persistent N=256 tile layout (consistent with
    # K-913 longK_smallSquare PMC findings).  K-1397 paired n=30 + B=10000
    # vectorised bootstrap CI95: 12/12 ROUTE-OUT, per-cell speedups 1.04×–1.12×;
    # envelope grows 73 → 85 cells.  Disjoint by construction with all P1–P13(N=128)
    # sub-frozensets via cross-frozenset asserts at module load.
    if _R_K1397_P13_skinny_n256_routeout(int(M), int(N), int(K), a_dtype): return True
    # K-1417 (S-002): P15 skinny_N512 K-COMPLEMENT EXTENSION 12-cell route-OUT
    # (9th-position envelope).  Stacks AFTER K-1397 P13 per K-1175 stacked-
    # predicate convention; closes the third-N successive sibling of the
    # K-COMPLEMENT EXTREMES axis (R-1417.SKINNY-N512-K-COMPLEMENT-EXTENSION-IS-EXTREMES)
    # at N=512 / K ∈ {2048, 32768} where (a) at K=2048 tritonblas
    # persistent_matmul tile parallelism is starved (TB ≈ 280 µs vs HBL ≈
    # 19-50 µs) and (b) at K=32768 hipBLASLt's split-K kernel selection wins
    # over tritonblas at the persistent N=512 tile layout where LDS bank
    # conflicts dominate (consistent with K-913 longK_smallSquare PMC
    # findings and K-1397 N=256 sibling).  K-1417 paired n=30 + B=10000
    # vectorised bootstrap CI95 on MI300X gfx942 (OCI MI300X fallback):
    # 12/12 ROUTE-OUT, cohort geomean tb/hbl = 3.43×, min CI95-lo = 1.359,
    # range 1.36×–14.06×; envelope grows 85 → 97 cells.  Disjoint by
    # construction with all P1–P14 sub-frozensets via cross-frozenset
    # asserts at module load.
    if _R_K1409_P15_skinny_n512_routeout(int(M), int(N), int(K), a_dtype): return True
    # K-1429 (S-002): P16 skinny_N1024 K-COMPLEMENT 29-cell route-OUT
    # (10th-position envelope).  Stacks AFTER K-1417 P15 per K-1175
    # stacked-predicate convention; closes the fourth-N successive sibling
    # of the K-COMPLEMENT axis at N=1024 across the FULL K-axis sweep
    # K ∈ {2048, 4096, 8192, 16384, 32768} (BASE + EXTREMES merged into a
    # single 29-cell frozenset).  Mechanism: at N=1024 the persistent_matmul
    # tile aspect misaligns against the M ∈ {2048, 4096, 8192} anchors,
    # accumulating LDS-bank conflicts beyond the K-913 §3 N=512 attenuation;
    # hipBLASLt's split-K kernel re-selects at N=1024 to a pattern that
    # better matches the M anchors.  The R-1409 monotone N-axis attenuation
    # (1.678 → 1.471 → 1.372 across N=128/256/512) REVERSES upward at
    # N=1024 to cohort geomean tb/hbl = 2.23× (BASE 2.08× / EXTREMES 2.48×),
    # exceeding every prior N bucket including N=128 — the K-1131 A2 1.40×
    # cohort floor (MISSED by 2.8 pp at N=512) is RECOVERED at N=1024
    # (R-1429.SKINNY-N1024-K-COMPLEMENT-DISCRIMINATOR-REVERSES-ATTENUATION-AT-N1024).
    # K-1429 paired n=30 + B=10000 vectorised bootstrap CI95 on MI300X
    # gfx942 (OCI amd-arad MI300X fallback): 29/30 ROUTE-OUT, range
    # 1.26×–8.85×, min CI95-lo = 1.267, single reject at
    # (2048, 1024, 4096, bf16) at r=1.015 (CI95-lo 1.008 > 1.0 but ratio
    # below strict 1.05 floor).  Envelope grows 97 → 126 cells.  Disjoint
    # by construction with all P1–P15 sub-frozensets via cross-frozenset
    # asserts at module load.
    if _R_K1429_P16_skinny_n1024_routeout(int(M), int(N), int(K), a_dtype): return True
    # P17 (11th-position): skinny_N512 K-COMPLEMENT BASE 17-cell route-OUT.
    # Closes the P15 N=512 EXTREMES sibling at the BASE K band (K ∈
    # {4096, 8192, 16384}) — N=512 K-COMPLEMENT goes from 12/30 (P15
    # EXTREMES only) to 29/30 (P15 ⨄ P17, with one cell P5-pre-routed at
    # chain pos 4).  Per-cell ratios 1.24×-1.64×; cohort geomean 1.40×.
    if _R_K1437_P17_skinny_n512_kcompl_base_routeout(int(M), int(N), int(K), a_dtype): return True
    # K-1478 P19 (12th-position): skinny_N16384 K-COMPLEMENT 30-cell route-OUT.
    # Stacks AFTER P17 per K-1175 stacked-predicate convention; closes the
    # N=16384 column along the K-COMPLEMENT axis (full K-grid 2048-32768).
    # 30/30 admit at strict ratio_median ≥ 1.05 ∧ p(<1.05) < 0.01 gate;
    # cohort geomean tb/hbl = 1.174×, range 1.056×-1.359×.  Natural
    # disjointness with all P1-P17 (sibling-N firewall + R-1465 #1
    # zero-P12-deferral invariant).
    if _R_K1478_P19_skinny_n16384_routeout(int(M), int(N), int(K), a_dtype): return True
    # K-1503 P21 (13th-position): skinny_N256 K-COMPLEMENT K-mid-band 17-cell
    # route-OUT.  Stacks AFTER P19 per K-1175 stacked-predicate convention;
    # closes the K-mid-band {4096, 8192, 16384} gap of K-1397 P13 N=256.
    # 17/30 admits at >=1.05x; cohort geomean 1.457x; closes ~89% of the
    # 30-cell N=256 gap.
    if _R_K1503_P21_skinny_n256_kmid_routeout(int(M), int(N), int(K), a_dtype): return True
    # K-1513 P22 (14th-position): skinny_N32768 K-COMPLEMENT 30-cell route-OUT.
    # Stacks AFTER P21 per the K-1175 stacked-predicate convention; closes
    # the top rung of the K-COMPLEMENT N-ladder at N=32768 on the full
    # K-grid {2048,4096,8192,16384,32768}.  30/30 admit at strict 1.05 gate;
    # cohort geomean tb/hbl ≈ 1.118× (range ≈ 1.045×-1.298×).  Naturally
    # disjoint with all P1-P21 (sibling-N firewall + R-1465 #1 invariant).
    if _R_K1513_P22_skinny_n32768_routeout(int(M), int(N), int(K), a_dtype): return True
    # K-1552 P23 (15th-position): skinny_N512 K-COMPLEMENT alias-stack 30-cell.
    # Stacks AFTER P22 per the K-1175 stacked-predicate convention.  ALIAS to
    # P15 ⨄ P17 ⨄ P5 — the K-1534 N=512 envelope (cohort geomean tb/hbl =
    # 1.454×, range 1.093×–2.111×, 0 regressions; paired n=30 HIP-graph
    # hot-cache MI300X gfx942 with TRITONBLAS_DISABLE_K971=1 against the live
    # post-K-1528 oracle) routes 30/30 via upstream layers, so this 15th-
    # position membership check is unreachable while P15+P17+P5 are enabled.
    # Slot is load-bearing only if an upstream layer is ablated; documents
    # the N=512 cohort under a single symbol per K-1493 / K-1538 alias-stack
    # convention.  K-axis trajectory: monotonic rise K=2048 (~1.10–1.24×)
    # → K=32768 (~1.43–2.11×), magnified at smaller M — K-913
    # longK_smallSquare LDS-bank-conflict signature on the persistent_matmul
    # N=512 tile, opposite to the R-1478 #1 N-axis attenuation trajectory.
    if _R_K1552_P23_skinny_n512_kcompl_aliasstack_routeout(int(M), int(N), int(K), a_dtype): return True
    # K-1566 P24 (16th-position): skinny_N4096 K-COMPLEMENT 30-cell route-OUT.
    # Stacks AFTER P23 per the K-1175 stacked-predicate convention; closes
    # the previously-empty N=4096 rung of the K-COMPLEMENT N-ladder
    # (between P21 N=2048 territory and P19 N=16384) on the full K-grid
    # {2048, 4096, 8192, 16384, 32768}.  30/30 admit at the strict 1.05
    # gate (K-1553 paired n=30 + B=10000 vectorised paired bootstrap on
    # MI300X gfx942 with TRITONBLAS_DISABLE_K971=1 vs the live post-K-1532
    # routing oracle); cohort geomean tb/hbl = 1.234×, range 1.114×-1.501×;
    # 0 regressions.  Per-row geomean: 1.402× (M=2048, K-913 LDS-BC band
    # fully live because min(M,N) ≤ 2048) / 1.146× (M=4096) / 1.180×
    # (M=8192).  Validates the R-1478 #1 N-axis attenuation chain anchor
    # at the previously-empty N=4096 rung (full chain 1.451 → 1.234 →
    # 1.220 → 1.174 → 1.118).  Sibling-N firewall disjoint with all P1-P23
    # except a single intentional 2-cell P12 alias overlap at
    # (4096, 4096, 4096, {bf16, fp16}) — P12 fires first so 28 cells are
    # NEW route-OUT and 2 cells are alias documentation.
    if _R_K1566_P24_skinny_n4096_routeout(int(M), int(N), int(K), a_dtype): return True
    # K-1611 P26 (17th-position): skinny_N2048 K-COMPLEMENT alias-stack
    # 30-cell route-OUT.  Stacks AFTER P24 per the K-1175 stacked-predicate
    # convention; closes the LAST untested mid-N rung of the K-COMPLEMENT
    # N-ladder at N=2048 on the full K-grid {2048, 4096, 8192, 16384,
    # 32768}.  Coverage now spans the full verified N range {128, 256, 512,
    # 1024, 2048, 4096, 8192, 16384, 32768}.  30/30 admit at the strict
    # 1.05 gate (K-1611 paired n=30 + B=10000 vectorised paired bootstrap
    # on MI300X gfx942 with TRITONBLAS_DISABLE_K971=1 vs the live post-
    # K-1581 routing oracle); cohort geomean tb/hbl = 1.451×, range
    # 1.108×-1.853×; 0 regressions.  Per-row geomean: 1.595× (M=2048,
    # K-913 LDS-BC band fully live because min(M,N)=2048) / 1.382×
    # (M=4096) / 1.391× (M=8192).  ALIAS-STACK structure: 22 NEW cells
    # + 8 alias cells (2 P12 at the (2048, 2048, 2048, {bf16, fp16})
    # diagonal + 6 K971_ROUTE_TABLE union: 4 K-905/K-971 LDS-BC anchors
    # at K∈{16384, 32768} ∪ 2 K-1335 longK_smallSquare bf16 cells at
    # K∈{4096, 8192}).  Alias overlaps fire BEFORE P26 in the dispatch
    # chain so the 8 alias cells are documentation; the 22 NEW cells are
    # the load-bearing portion (M ∈ {4096, 8192} × all K + the
    # (2048, 2048, K, fp16) pair for K ∈ {4096, 8192} that K-1335 (bf16-
    # only) does not cover).  Anchors the R-1478 #1 N-axis attenuation
    # chain head — full chain now 1.451 → 1.234 → 1.220 → 1.174 → 1.118
    # across N ∈ {2048, 4096, 8192, 16384, 32768}, strictly monotone-
    # decreasing.  Mechanism (K-1598 PMC RCA): TB SQ_INSTS_LDS counters
    # at N=2048 inflate ~3.1× over hbl on the M=2048 row, consistent with
    # the K-913 LDS_BC ≈ 1.6-1.8 cyc/inst signature; persistent_matmul
    # cannot relieve LDS-bank-conflict pressure via tile reshape.
    if _R_K1611_P26_skinny_n2048_kcompl_aliasstack_routeout(int(M), int(N), int(K), a_dtype): return True
    # K-1553 17th-slot handle: no executable code — the K-1553-named alias
    # `_K1553_P25_SKINNY_N4096_KCOMPL_ALIASSTACK_30` lives in
    # _route_predicate.py as a single module-level rebinding of P24's
    # frozenset, since the K-1553 admit set is bit-identical to P24 and P24
    # fires first.  Per K-1489 reviewer precedent we do not add a duplicate
    # predicate call for an unreachable alias.  K-1553 measurement provenance:
    # paired n=30 HIP-graph hot-cache MI300X gfx942 vs the live post-K-1532
    # oracle (combined with K-1559 60-cell N ∈ {4096, 8192} confirmation);
    # cohort geomean tb/hbl = 1.234×, range 1.114×-1.501×, 30/30 at strict
    # 1.05 gate — same evidence that backs the load-bearing P24 above.
    # K-1633 18th-slot handle: no executable code — the K-1633-named alias
    # `_K1633_P27_SKINNY_N512_KCOMPL_ALIASSTACK_30` lives in
    # _route_predicate.py as a single module-level rebinding of P23's
    # frozenset (K-1552), since the K-1633 admit set is bit-identical to
    # P23 and P23 fires first at the 15th slot.  Per K-1581 / K-1489
    # minimalist precedent we do not add a duplicate predicate call for
    # an unreachable alias.  K-1633 re-measurement provenance: paired
    # n=30 HIP-graph hot-cache MI300X gfx942 vs the live post-K-1611 P26
    # oracle; 30/30 admit at the strict 1.05 gate, cohort geomean tb/hbl
    # ≥ 1.45× — same shape that backs the load-bearing P23 above.
    # K-1673 (S-002): P28 skinny_N128 K-COMPLEMENT alias-stack 30-cell
    # route-OUT (19th-position, load-bearing).  Closes the LAST untested
    # small-N rung (N=128) of the K-COMPLEMENT N-ladder at the dtype-
    # mirror gap (fp16 K ∈ {2048, 32768}) on the M ∈ {2048, 4096, 8192}
    # rows; coverage now spans the FULL N-ladder {128, 256, 512, 1024,
    # 2048, 4096, 8192, 16384, 32768} for both bf16 and fp16.  K-1673
    # paired n=30 HIP-graph hot-cache + B=10000 vectorised paired
    # bootstrap on MI300X gfx942 (OCI useocpm2m-097-099 amd-rccl
    # partition / ROCm 7.2 / pytorch 2.10) against the LIVE post-K-1647
    # P27 routing oracle: 30/30 admit at the strict ratio_median ≥ 1.05
    # ∧ p(<1.05) < 0.01 gate; cohort geomean tb/hbl = 1.694×, range
    # 1.162×-2.890×, 0 regressions.  Per-row geomean: 1.889× (M=2048,
    # K-913 LDS-BC band fully live because min(M,N) = 128) / 1.458×
    # (M=4096) / 1.767× (M=8192).  ALIAS-STACK structure: 6 NEW cells +
    # 24 alias cells (15 bf16 via R-K979 P5 Clause-3 minMN ≤ 192 ∧ K ≥
    # 2048 + 9 fp16 K ∈ {4096, 8192, 16384} via K-1367 P13 N=128); alias
    # overlaps fire BEFORE P28 in the dispatch chain so the 24 alias
    # cells are documentation; the 6 NEW cells (all fp16 mirror at K ∈
    # {2048, 32768}) are the load-bearing portion that close the dtype-
    # mirror gap left by P5's bf16-only `_dtype_is_bf16` early-return
    # and P13's K-mid-only K-grid coverage.  Mechanism (K-1673 PMC RCA):
    # TB SQ_LDS_BANK_CONFLICT/inst = 1.45-2.13 cyc (vs HBL = 0.000
    # exactly); the K-913 §3 dtype-invariant LDS-BC signature on the
    # N=128 column-narrow tile.  Why not share P21 N=256 / P23 N=512:
    # at N=128 the BLOCK_N=128 tile has 1 K-block column and ALL
    # accumulator lanes serialise on one bank group (LDS_BC ≈ 1.45-2.13
    # cyc/inst); at N=256 the same tile has 2 K-block columns and the
    # LDS bank arbitration round-robins across 2 swizzle phases (LDS_BC
    # ≈ 0.6-0.9 cyc/inst) — sibling-N firewall preserves the per-N audit
    # handles per the K-1175 stacked-predicate convention.
    if _R_K1673_P28_skinny_n128_kcompl_aliasstack_routeout(int(M), int(N), int(K), a_dtype): return True
    # K-1700 P29 (20th-slot): N=64 K-COMPLEMENT alias-stack (29 cells; cohort geomean tb_forced/hbl=1.93×).
    if (int(M), int(N), int(K), str(a_dtype)) in _K1700_P29_SKINNY_N64_KCOMPL_ALIASSTACK_29: return True
    # K-1748 P30 (21st-slot): skinny_Nmid (N ∈ {384, 768, 1536}) K-COMPLEMENT alias-stack
    # (34 K-1711-verified cells; cohort geomean 1.456×, range 1.18×-1.83×).  Audit:
    # post-K-1709 oracle had 0/34 cells active; K-1720 (parallel branch off K-1685) never
    # merged into K-1709 lineage.  Ship at 21st slot per K-1709/K-1720 disjoint-N rationale.
    if (int(M), int(N), int(K), str(a_dtype)) in _K1711_P30_SKINNY_NMID_KCOMPL_ALIASSTACK_34: return True
    # P31 (22nd-slot): N=256 K-COMPLEMENT verified-winner subset — 28 cells (TB-native
    # vs HBL-native paired n=30 hot-cache HIP-graph on MI300X with route-OUT ablated:
    # geomean HBL/TB = 1.392×, range 1.07×–2.12×, 28/28 cells pass the strict
    # ≥1.05 ∧ p<0.05 gate; the 2 (M=2048, K=2048) LOSER cells are excluded).
    if (int(M), int(N), int(K), str(a_dtype)) in _P31_SKINNY_N256_KCOMPL_VERIFIED_WIN_28: return True
    # K-1881 (S-002): consolidated P32–P38 K-COMPLEMENT verified-winner
    # alias-stack lookup — ONE membership probe over a single 120-cell
    # frozenset (cells inlined as literal tuples in `_route_predicate.py`)
    # replacing what would otherwise be a chain of 7 back-to-back probes
    # (P32 N=160 / P33 N=224 / P34 N=96 / P35 N=288 / P36 N=320 /
    # P37 N=352 / P38 N=384).  Extends the K-1864 5-slot consolidation in
    # place to absorb the K-1868 P37 + K-1880 P38 promotions before they
    # could re-fragment the dispatcher.  Worst-case dispatch stays at
    # O(1) hash lookup + 1 tuple build per call (vs the 7-probe
    # alternative), halting the chain-growth pattern at this position
    # for future PMC promotions.
    #
    # Bit-identical routing equivalence vs the 7-chain hypothetical is
    # an algebraic identity (`x in (A1 ∪ … ∪ A7)` ≡ `any(x in Ai)` for
    # disjoint Python sets); empirical disjointness of the 7 N-axis
    # projections is enforced by the literal cell roster
    # (sorted({c[1] for c in canonical}) == [96, 160, 224, 288, 320, 352,
    # 384] — 7 distinct integers) and exercised cell-by-cell against the
    # legacy 7-chain reference oracle in
    # `tests/test_k1881_p32_p38_consolidation.py`.
    if (int(M), int(N), int(K), str(a_dtype)) in _K1881_P32_P38_SKINNY_KCOMPL_ROUTEOUT_120: return True
    # K-1887 (S-002): P39 (30th-slot) N=448 K-COMPLEMENT verified-winner
    # alias-stack — 18 cells (full M ∈ {2048,4096,8192} × N=448 × K ∈
    # {4096,8192,16384} × {bf16,fp16}; no upstream alias overlap).  K-1881-
    # followup paired n=30 HIP-graph + PMC: per-N geomean = 1.36× (range
    # 1.18×–1.74×, min CI95-lo = 1.146); N=448 = 1.75 × BN=256 → 0.75-wave
    # tail leaves 25% MFMA lanes idle; SCHEDULER_LDS A4 fingerprint
    # (HBL SQ_LDS_BANK_CONFLICT == 0 in 18/18; TB nonzero in 18/18).
    # Disjoint from P30 N ∈ {384,768,1536} and from K-1881 P32–P38
    # N ∈ {96,160,224,288,320,352,384} by sibling-N firewall.
    if (int(M), int(N), int(K), str(a_dtype)) in _P39_SKINNY_N448_KCOMPL_VERIFIED_WIN_18: return True
    return False


current_device_index = torch.cuda.current_device()
current_device = torch.cuda.get_device_properties(current_device_index)
MAX_SMS = current_device.multi_processor_count
MAX_BLOCK_SIZE = 65536

_global_locks = torch.empty(MAX_SMS, device="cuda", dtype=torch.uint8)
_global_P = torch.empty(MAX_SMS, MAX_BLOCK_SIZE, device="cuda", dtype=torch.float32)


def _maybe_wrap(fn, probe_tensor):
    # Use wrap_triton only under torch.compile tracing; otherwise direct call
    # in eager.  Can't use torch.compiler.is_compiling() here because the code
    # inside @triton_op but outside wrap_triton is part of the compile pass
    # itself and is_compiling() is never True.
    if is_fake(probe_tensor):
        return wrap_triton(fn)
    return fn


# Function will behave like an LRU-Cache of heuristic results
# Saves several microseconds for previously seen problems by not rerunning the heuristic unnecessarily
#@functools.lru_cache(maxsize=1024)
def _make_matmul_selector(
    M: int,
    N: int,
    K: int,
    a_dtype: torch.dtype,
    b_dtype: torch.dtype,
    c_dtype: torch.dtype,
    device: torch.device,
    mx_block_size=0,
    streamk=False,
    num_stages: int = 2,
):
    # Run Heuristic Results (Only if key has not been seen before)
    return OrigamiMatmulSelector(
        M,
        N,
        K,
        a_dtype,
        b_dtype,
        c_dtype,
        device,
        mx_block_size=mx_block_size,
        streamk=streamk,
        num_stages=num_stages,
    )


def persistent_matmul_lt(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    selector,
    config: Optional[MatmulConfig] = None,
    bias: Optional[torch.Tensor] = None,
    a_scale: Optional[torch.Tensor] = None,
    b_scale: Optional[torch.Tensor] = None,
    quantized: bool = False,
    work_stealing: bool = False,
):
    assert a.shape[1] == b.shape[0], "Incompatible Dimensions"
    M, K = a.shape
    _, N = b.shape

    BLK_M    = selector.block_m
    BLK_N    = selector.block_n
    BLK_K    = selector.block_k
    gsize_m  = selector.group_m
    num_xcds = selector.num_sms

    total_blocks_M = triton.cdiv(M, BLK_M)
    total_blocks_N = triton.cdiv(N, BLK_N)
    total_tiles = total_blocks_M * total_blocks_N
    total_programs = total_tiles
    even_k = K % BLK_K == 0

    num_stages = getattr(selector, "num_stages", 2)
    num_warps = 8
    waves_per_eu = 0
    mfmaInstrSize = 16
    kpack = 1
    CACHE_MODIFIER_A = None
    CACHE_MODIFIER_B = None

    # K-1089: structural surrogate of K-1037 P6 admits the MFMA-issue-stall
    # cohort to in-kernel dispatch with waves_per_eu=1 (K-1051 confirmed via
    # 4-iter PMC research that wpeu=1 wins are predictable on K-1032 cells).
    # The route-OUT short-circuit in `_k971_route_to_hbl` already vetoes
    # routing for these cells; we only need to set the in-kernel knob here.
    if _R_K1037_P6_admit_wpeu1(int(M), int(N), int(K), a.dtype):
        waves_per_eu = 1

    # Set chunk size to same area as L2 tiles.
    chunk_size = gsize_m * gsize_m
    if num_xcds > 0:
        chunk_size = min(chunk_size, max(1, total_programs // num_xcds))
    else:
        num_xcds = 1

    if work_stealing and config is not None:
        grids = selector._hardware.N_CU

        kk = _maybe_wrap(ws_persistent_matmul, probe_tensor=a)[(grids,)](
            a,
            b,
            c,
            a_scale if quantized else None,
            b_scale if quantized else None,
            bias if bias is not None else None,
            config.tile_counter,
            config.global_counter,
            M,
            N,
            K,
            a.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            bias.stride(0) if bias is not None else 0,
            stride_ak=a.stride(1),
            stride_bk=b.stride(0),
            BLOCK_SIZE_M=BLK_M,
            BLOCK_SIZE_N=BLK_N,
            BLOCK_SIZE_K=BLK_K,
            GROUP_SIZE_M=gsize_m,
            NUM_SMS=grids,
            NUM_XCDS=num_xcds,
            COUNTERS_PER_XCD=selector.COUNTERS_PER_XCD,
            COUNTER_STRIDE=COUNTER_STRIDE,
            BIAS=bias is not None,
            EVEN_K=even_k,
            CACHE_MODIFIER_A=CACHE_MODIFIER_A,
            CACHE_MODIFIER_B=CACHE_MODIFIER_B,
            QUANTIZED=quantized,
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
            GLOBAL_ATOMIC=config.global_atomic,
            HIERARCHICAL=False,
            LOCAL_TILES_PER_XCD=0,
            GLOBAL_TILES=0,
            USE_MASK=True,
            mask_ptr=config.mask,
            num_stages=num_stages,
            num_warps=num_warps,
            waves_per_eu=waves_per_eu,
            matrix_instr_nonkdim=mfmaInstrSize,
            kpack=kpack,
        )
    else:
        grids = total_tiles

        # K-883/K-901 guarded-override registry hook.  Single dispatch site,
        # routed first (before any future range-keyed hook, R1 — dispatch-
        # order-first routing).  The registry enforces L1/L2/L3/L5/L6; the
        # call-site honors L4 (LDS budget) below via _overrides_apply.
        _ovr_updates = _go.apply_override_in_dispatcher(
            M=M, N=N, K=K, dtype=a.dtype,
            block_m=BLK_M, block_n=BLK_N, block_k=BLK_K,
            num_stages=num_stages, num_warps=num_warps,
            waves_per_eu=waves_per_eu, kpack=kpack,
            mfma_instr_size=mfmaInstrSize,
            total_tiles=total_tiles, n_cu=selector._hardware.N_CU,
            work_stealing=work_stealing,
            bytes_per_elem=a.element_size(),
        )
        if _ovr_updates and not _ovr_updates.get("lds_blocked"):
            BLK_M = _ovr_updates.get("BLOCK_SIZE_M", BLK_M)
            BLK_N = _ovr_updates.get("BLOCK_SIZE_N", BLK_N)
            BLK_K = _ovr_updates.get("BLOCK_SIZE_K", BLK_K)
            num_stages = _ovr_updates.get("num_stages", num_stages)
            num_warps = _ovr_updates.get("num_warps", num_warps)
            waves_per_eu = _ovr_updates.get("waves_per_eu", waves_per_eu)
            kpack = _ovr_updates.get("kpack", kpack)
            mfmaInstrSize = _ovr_updates.get("matrix_instr_nonkdim", mfmaInstrSize)
            grids = _ovr_updates.get("grids", grids)
            total_programs = _ovr_updates.get("total_programs", total_programs)

        kk = _maybe_wrap(persistent_matmul, probe_tensor=a)[(grids,)](
            a,
            b,
            c,
            a_scale if quantized else None,
            b_scale if quantized else None,
            bias if bias is not None else None,
            M,
            N,
            K,
            a.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            bias.stride(0) if bias is not None else 0,
            stride_ak=a.stride(1),
            stride_bk=b.stride(0),
            BLOCK_SIZE_M=BLK_M,
            BLOCK_SIZE_N=BLK_N,
            BLOCK_SIZE_K=BLK_K,
            GROUP_SIZE_M=gsize_m,
            NUM_SMS=total_programs,
            NUM_XCDS=num_xcds,
            CHUNK_SIZE=chunk_size,
            BIAS=bias is not None,
            EVEN_K=even_k,
            CACHE_MODIFIER_A=CACHE_MODIFIER_A,
            CACHE_MODIFIER_B=CACHE_MODIFIER_B,
            QUANTIZED=quantized,
            num_stages=num_stages,
            num_warps=num_warps,
            waves_per_eu=waves_per_eu,
            matrix_instr_nonkdim=mfmaInstrSize,
            kpack=kpack,
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        )

    return c

def streamk_matmul_lt(
    a: torch.Tensor, 
    b: torch.Tensor, 
    c: torch.Tensor, 
    selector, 
    config: Optional[MatmulConfig] = None,
    bias: Optional[torch.Tensor] = None,
    sk_grid: Optional[int] = None,
    a_scale: Optional[torch.Tensor] = None,
    b_scale: Optional[torch.Tensor] = None,
    quantized: bool = False,
    work_stealing: bool = False,
):
    assert a.shape[1] == b.shape[0], "Incompatible Dimensions"
    M, K = a.shape
    _, N = b.shape

    BLK_M    = selector.block_m
    BLK_N    = selector.block_n
    BLK_K    = selector.block_k
    gsize_m  = selector.group_m
    num_xcds = selector.num_sms

    total_blocks_M = triton.cdiv(M, BLK_M)
    total_blocks_N = triton.cdiv(N, BLK_N)
    total_tiles = total_blocks_M * total_blocks_N
    even_k = K % BLK_K == 0

    ##
    # Grid Size
    ##
    if work_stealing:
        total_programs_streamk = selector._hardware.N_CU
    else:
        total_programs_streamk = selector.sk_grid

    if total_programs_streamk > 0:
        total_tiles_streamk = total_tiles % total_programs_streamk
    else:
        total_tiles_streamk = 0

    num_stages = getattr(selector, "num_stages", 2)
    num_warps = 8
    waves_per_eu = 0
    mfmaInstrSize = 16
    kpack = 1
    CACHE_MODIFIER_A = None
    CACHE_MODIFIER_B = None

    if sk_grid is not None:
        total_programs_streamk = sk_grid

    grids = total_programs_streamk
    block_size = BLK_M * BLK_N

    if config is not None:
        if grids <= config.locks.shape[0] and block_size <= config.P.shape[1]:
            locks = config.locks[:grids]
            P = config.P[:grids, :block_size]
        else:
            locks = torch.empty(grids, device=config.device, dtype=torch.uint8)
            P = torch.empty(grids, block_size, device=config.device, dtype=torch.float32)
    else:
        if grids <= MAX_SMS and block_size <= MAX_BLOCK_SIZE:
            locks = _global_locks[:grids]
            P = _global_P[:grids, :block_size]
        else:
            locks = torch.empty(grids, device=a.device, dtype=torch.uint8)
            P = torch.empty(grids, block_size, device=a.device, dtype=torch.float32)

    # Set chunk size to same area as L2 tiles.
    chunk_size = gsize_m * gsize_m
    if num_xcds > 0:
        chunk_size = min(chunk_size, grids // num_xcds)
    else:
        num_xcds = 1

    if work_stealing and config is not None:
        kk = _maybe_wrap(ws_streamk_matmul, probe_tensor=a)[(grids,)](
            a,
            b,
            c,
            a_scale if quantized else None,
            b_scale if quantized else None,
            bias if bias is not None else None,
            config.tile_counter,
            config.streamk_tile_counter,
            P,
            locks,
            M,
            N,
            K,
            a.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            bias.stride(0) if bias is not None else 0,
            stride_ak=a.stride(1),
            stride_bk=b.stride(0),
            BLOCK_SIZE_M=BLK_M,
            BLOCK_SIZE_N=BLK_N,
            BLOCK_SIZE_K=BLK_K,
            GROUP_SIZE_M=gsize_m,
            NUM_SMS=selector._ACTIVE_CU,
            NUM_XCDS=num_xcds,
            CHUNK_SIZE=chunk_size,
            STREAMK_TILES=total_tiles_streamk,
            COUNTERS_PER_XCD=selector.COUNTERS_PER_XCD,
            COUNTER_STRIDE=COUNTER_STRIDE,
            BIAS=bias is not None,
            EVEN_K=even_k,
            CACHE_MODIFIER_A=CACHE_MODIFIER_A,
            CACHE_MODIFIER_B=CACHE_MODIFIER_B,
            QUANTIZED=quantized,
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
            GLOBAL_ATOMIC=config.global_atomic,
            mask_ptr=config.mask,
            num_stages=num_stages,
            num_warps=num_warps,
            waves_per_eu=waves_per_eu,
            matrix_instr_nonkdim=mfmaInstrSize,
            kpack=kpack,
        )
    else:
        kk = _maybe_wrap(streamk_matmul, probe_tensor=a)[(grids,)](
            a,
            b,
            c,
            a_scale if quantized else None,
            b_scale if quantized else None,
            bias if bias is not None else None,
            P,
            locks,
            M,
            N,
            K,
            a.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            bias.stride(0) if bias is not None else None,
            stride_ak=a.stride(1),
            stride_bk=b.stride(0),
            BLOCK_SIZE_M=BLK_M,
            BLOCK_SIZE_N=BLK_N,
            BLOCK_SIZE_K=BLK_K,
            GROUP_SIZE_M=gsize_m,
            NUM_SMS=grids,
            NUM_XCDS=num_xcds,
            CHUNK_SIZE=chunk_size,
            STREAMK_TILES=total_tiles_streamk,
            BIAS=bias is not None,
            EVEN_K=even_k,
            CACHE_MODIFIER_A=CACHE_MODIFIER_A,
            CACHE_MODIFIER_B=CACHE_MODIFIER_B,
            QUANTIZED=quantized,
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
            num_stages=num_stages,
            num_warps=num_warps,
            waves_per_eu=waves_per_eu,
            matrix_instr_nonkdim=mfmaInstrSize,
            kpack=kpack,
        )

    return c

def matmul_lt(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
    selector, config: MatmulConfig,
    enable_streamk=False, work_stealing=False
):
    assert a.shape[1] == b.shape[0], "Incompatible Dimensions"

    if enable_streamk:
        return streamk_matmul_lt(a, b, c, selector, config, work_stealing=work_stealing)
    else:
        return persistent_matmul_lt(a, b, c, selector, config, work_stealing=work_stealing)

def matmul_a8w8_lt(
    a: torch.Tensor, b: torch.Tensor, a_scale: torch.Tensor, b_scale: torch.Tensor,
    c: torch.Tensor, selector, config: MatmulConfig,
    enable_streamk=False, work_stealing=False,
):
    assert a.shape[1] == b.shape[0], "Incompatible Dimensions"

    if enable_streamk:
        return streamk_matmul_lt(a, b, c, selector, config, a_scale=a_scale, b_scale=b_scale, quantized=True)
    else:
        return persistent_matmul_lt(a, b, c, selector, config, a_scale=a_scale, b_scale=b_scale, quantized=True, work_stealing=work_stealing)


@triton_op("tritonblas::_matmul", mutates_args={})
def _matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    enable_streamk: Optional[bool] = False,
    sk_grid: Optional[int] = None,
    work_stealing: Optional[bool] = False,
) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], "Incompatible A-B Dimensions"
    M, K = a.shape
    _, N = b.shape

    if _k971_route_to_hbl(M, N, K, a.dtype, b.dtype, enable_streamk, work_stealing):
        return torch.matmul(a, b)
    out = a.new_empty(M, N)

    selector = _make_matmul_selector(M, N, K, a.dtype, b.dtype, out.dtype, a.device, streamk=enable_streamk)
    config = matmul_preamble(selector) if work_stealing else None
    if enable_streamk:
        return streamk_matmul_lt(a, b, out, selector, config, sk_grid=sk_grid, work_stealing=work_stealing)
    else:
        return persistent_matmul_lt(a, b, out, selector, config, work_stealing=work_stealing)


def _setup_context_matmul_backwards(
    ctx: Any,
    inputs: tuple[Any, ...],
    output: Any
):
    a, b, enable_streamk, sk_grid, work_stealing = inputs
    ctx.save_for_backward(a, b)
    ctx.enable_streamk = enable_streamk
    ctx.sk_grid = sk_grid
    ctx.work_stealing = work_stealing


def _matmul_backwards(
    ctx: Any,
    grad_output: torch.Tensor
):
    a, b = ctx.saved_tensors
    enable_streamk = ctx.enable_streamk
    sk_grid = ctx.sk_grid
    work_stealing = ctx.work_stealing

    grad_output_cont = grad_output.contiguous()

    b_t = b.T.contiguous()
    grad_a = matmul(grad_output_cont, b_t, enable_streamk=enable_streamk, sk_grid=sk_grid, work_stealing=work_stealing)

    a_t = a.T.contiguous()
    grad_b = matmul(a_t, grad_output_cont, enable_streamk=enable_streamk, sk_grid=sk_grid, work_stealing=work_stealing)

    return grad_a, grad_b, None, None, None


_matmul.register_autograd(_matmul_backwards,
                          setup_context=_setup_context_matmul_backwards)


@triton_op("tritonblas::_matmul_out", mutates_args={'out'})
def _matmul_out(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    enable_streamk: Optional[bool] = False,
    sk_grid: Optional[int] = None,
    work_stealing: Optional[bool] = False,
) -> None:
    assert a.shape[1] == b.shape[0], "Incompatible A-B Dimensions"
    M, K = a.shape
    _, N = b.shape

    if _k971_route_to_hbl(M, N, K, a.dtype, b.dtype, enable_streamk, work_stealing):
        torch.matmul(a, b, out=out)
        return None
    selector = _make_matmul_selector(M, N, K, a.dtype, b.dtype, out.dtype, a.device, streamk=enable_streamk)
    config = matmul_preamble(selector) if work_stealing else None

    if enable_streamk:
        streamk_matmul_lt(a, b, out, selector, config, sk_grid=sk_grid, work_stealing=work_stealing)
    else:
        persistent_matmul_lt(a, b, out, selector, config, work_stealing=work_stealing)

    return None


def matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    enable_streamk: Optional[bool] = False,
    sk_grid: Optional[int] = None,
    work_stealing: Optional[bool] = False,
) -> Optional[torch.Tensor]:
    if out is None:
        return _matmul(a, b, enable_streamk, sk_grid, work_stealing)

    if torch.is_grad_enabled() and (
        a.requires_grad
        or b.requires_grad
        or out.requires_grad
    ):
        raise RuntimeError(
            "tritonblas.matmul(): functions with out=... arguments don't support "
            "automatic differentiation, but one of the arguments requires grad."
        )
    return _matmul_out(a, b, out, enable_streamk, sk_grid, work_stealing)


def matmul_a8w8(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    c: torch.Tensor,
    enable_streamk=False,
    work_stealing=False,
    sk_grid=None,
):
    assert a.shape[1] == b.shape[0], "Incompatible Dimensions"
    M, K = a.shape
    _, N = b.shape

    selector = _make_matmul_selector(M, N, K, a.dtype, b.dtype, c.dtype, a.device, streamk=enable_streamk)
    config = matmul_preamble(selector) if work_stealing else None
    if enable_streamk:
        return streamk_matmul_lt(a, b, c, selector, config, sk_grid=sk_grid, a_scale=a_scale, b_scale=b_scale, quantized=True, work_stealing=work_stealing)
    else:
        return persistent_matmul_lt(a, b, c, selector, config, a_scale=a_scale, b_scale=b_scale, quantized=True, work_stealing=work_stealing)

def matmul_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    block_m: int = None, #Overrides Origami value
    block_n: int = None, #Overrides Origami value
    block_k: int = None, #Overrides Origami value
    group_size_m: int = 8, #Overrides Origami value
    num_warps: int = 8,
    num_stages: int = 2,
):
    """
    FP4 matrix multiplication: C = A @ B
    
    Args:
        a: Input matrix A in FP4 format (M, K//2), packed 2 elements per uint8
        b: Input matrix B in FP4 format (N, K//2), packed 2 elements per uint8
        c: Output matrix C (M, N) in bfloat16 or float16
        a_scales: Scales for A in e8m0 format (M, K // 32)
        b_scales: Scales for B in e8m0 format (N, K // 32)
        block_m: Block size for M dimension
        block_n: Block size for N dimension
        block_k: Block size for K dimension (must be multiple of 64 for FP4)
        group_size_m: Group size for M dimension tiling
        num_warps: Number of warps per thread block (default: 8)
        num_stages: Number of pipeline stages (default: 2)
    
    Returns:
        Output matrix C
    """

    M, K = a.shape
    _, N = b.shape
    
    num_xcds = 8

    if(block_m == None):
        selector = _make_matmul_selector(M, N, K, "f4", "f4", c.dtype, a.device, mx_block_size=32)
        block_m      = selector.block_m
        block_n      = selector.block_n
        block_k      = selector.block_k
        group_size_m = selector.group_m
        num_xcds     = selector.num_sms
        if(block_m < M):
            block_m=128
        if(block_n < N):
            block_n=128
        if(block_k < K):
            block_k=128
        #print(f"Selected {block_m}x{block_n}x{block_k}")
    # Get actual dimensions (accounting for packing)
    M = a.shape[0]
    K = a.shape[1] * 2  # Unpacked K dimension
    N = b.shape[0]  # B has shape (N, K//2)
    
    # Verify dimensions are compatible
    assert b.shape[1] * 2 == K, f"Incompatible Dimensions: A has K={K}, B has K={b.shape[1] * 2}"
    
    # Transpose B to match kernel expectations (kernel expects B as K x N)
    b = b.T
    
    # Ensure block_k is appropriate for FP4 (must be multiple of 64)
    assert block_k % 64 == 0, "BLOCK_K must be multiple of 64 for FP4"
    
    total_blocks_M = triton.cdiv(M, block_m)
    total_blocks_N = triton.cdiv(N, block_n)
    total_tiles = total_blocks_M * total_blocks_N
    
    # Set chunk size to same area as L2 tiles
    chunk_size = group_size_m * group_size_m
    chunk_size = min(chunk_size, max(1, total_tiles // num_xcds))
    
    grid = (total_tiles,)
    
    fp4_matmul[grid](
        a,
        b,
        c,
        a_scales,
        b_scales,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        a_scales.stride(0),
        a_scales.stride(1),
        b_scales.stride(0),
        b_scales.stride(1),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        GROUP_SIZE_M=group_size_m,
        NUM_SMS=total_tiles,
        NUM_XCDS=num_xcds,
        CHUNK_SIZE=chunk_size,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    
    return c


@triton_op("tritonblas::_addmm", mutates_args={})
def _addmm(
    bias: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    enable_streamk: Optional[bool] = False,
    sk_grid: Optional[int] = None,
    work_stealing: Optional[bool] = False,
) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], "Incompatible A-B Dimensions"
    M, K = a.shape
    _, N = b.shape

    # Query Origami for solution
    selector = _make_matmul_selector(M, N, K, a.dtype, b.dtype, bias.dtype, a.device, streamk=enable_streamk)
    config = matmul_preamble(selector) if work_stealing else None

    # Allocate an output tensor
    out = a.new_empty(M, N)

    if enable_streamk:
        return streamk_matmul_lt(a, b, out, selector, config, bias=bias, sk_grid=sk_grid, work_stealing=work_stealing)
    else:
        return persistent_matmul_lt(a, b, out, selector, config, bias=bias, work_stealing=work_stealing)


def _setup_context_addmm_backwards(
    ctx: Any,
    inputs: tuple[Any, ...],
    output: Any
):
    bias, a, b, enable_streamk, sk_grid, work_stealing = inputs
    ctx.save_for_backward(a, b)
    ctx.enable_streamk = enable_streamk
    ctx.sk_grid = sk_grid
    ctx.work_stealing = work_stealing


def _addmm_backwards(
    ctx: Any,
    grad_output: torch.Tensor
):
    a, b = ctx.saved_tensors
    enable_streamk = ctx.enable_streamk
    sk_grid = ctx.sk_grid
    work_stealing = ctx.work_stealing

    # Make grad_output contiguous
    grad_output_cont = grad_output.contiguous()

    # grad_a = grad_output @ b^T
    b_t = b.T.contiguous()
    grad_a = matmul(grad_output_cont, b_t, enable_streamk=enable_streamk, sk_grid=sk_grid, work_stealing=work_stealing)

    # grad_b = a^T @ grad_output
    a_t = a.T.contiguous()
    grad_b = matmul(a_t, grad_output_cont, enable_streamk=enable_streamk, sk_grid=sk_grid, work_stealing=work_stealing)

    # grad_bias = sum(grad_output)
    grad_bias = grad_output.sum(dim=0)

    # tuple[bias, a, b, enable_streamk, sk_grid, work_stealing]
    #   First 3 must be in the order that matches addmm()'s forward args
    #   Last 3 are not part of the gradient and so are None
    return grad_bias, grad_a, grad_b, None, None, None


_addmm.register_autograd(_addmm_backwards,
                         setup_context=_setup_context_addmm_backwards)


@triton_op("tritonblas::_addmm_out", mutates_args={'out'})
def _addmm_out(
    bias: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    enable_streamk: Optional[bool] = False,
    sk_grid: Optional[int] = None,
    work_stealing: Optional[bool] = False,
) -> None:
    assert a.shape[1] == b.shape[0], "Incompatible A-B Dimensions"
    M, K = a.shape
    _, N = b.shape

    # Query Origami for solution
    selector = _make_matmul_selector(M, N, K, a.dtype, b.dtype, bias.dtype, a.device, streamk=enable_streamk)
    config = matmul_preamble(selector) if work_stealing else None

    if enable_streamk:
        streamk_matmul_lt(a, b, out, selector, config, bias=bias, sk_grid=sk_grid, work_stealing=work_stealing)
    else:
        persistent_matmul_lt(a, b, out, selector, config, bias=bias, work_stealing=work_stealing)

    # Custom torch ops cannot return a value which is an alias of an input.  So
    # even though torch returns a pointer to the out arg when used, we can't.
    return None


def addmm(
    bias: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    enable_streamk: Optional[bool] = False,
    sk_grid: Optional[int] = None,
    work_stealing: Optional[bool] = False,
) -> Optional[torch.Tensor]:
    # If no out tensor provided - we do the allocation - we support autograd
    if out is None:
        return _addmm(bias, a, b, enable_streamk, sk_grid, work_stealing)

    # If out tensor provided - in-place - we do NOT support autograd
    # Check for autograd conditions (global and per-tensor)
    if torch.is_grad_enabled() and (
        bias.requires_grad
        or a.requires_grad
        or b.requires_grad
        or out.requires_grad
    ):
        raise RuntimeError(
            "tritonblas.addmm(): functions with out=... arguments don't support "
            "automatic differentiation, but one of the arguments requires grad."
        )
    return _addmm_out(bias, a, b, out, enable_streamk, sk_grid, work_stealing)

