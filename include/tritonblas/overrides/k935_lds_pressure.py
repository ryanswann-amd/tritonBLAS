"""K-935 — LDS-pressure mitigation prototype for cohort A (M=N=1024, K=16384).

Mechanism
---------
K-905 mechanistically root-caused cohort A (M=N=1024, K∈{16384}, bf16/fp16) on
production ``persistent_matmul`` as LDS-bound (SQ_LDS_BANK_CONFLICT 8.39 M
cyc/disp + SQ_WAIT_INST_LDS 38.22 M cyc/disp = 41× hipBLASLt) with
``ds_read2_b64`` dominating issue slots.  The Tensile recipe wins by emitting
``ds_read_b128`` (twice the per-instruction width), which Triton on AMD can
approximate by widening the LDS access pattern via ``kpack=2`` (packs two
K-elements per ds_read so the codegen issues b128-width loads when the tile
geometry permits).  ``num_warps=8`` is preserved (matches the production
default for this tile geometry; explicitly forced so the override is
self-contained against any future default change).

LDS bandwidth math
------------------
Per-MFMA LDS round-trip at the production tile (BLOCK_M=BLOCK_N=128,
BLOCK_K=64, fp16/bf16, 2 bytes/elem):

    A-fragment per-warp:  128 × 64 × 2 B = 16384 B
    B-fragment per-warp:  128 × 64 × 2 B = 16384 B

With ``kpack=1`` Triton-AMD emits ``ds_read2_b64`` (8 B per lane × 64 lanes ×
2 = 1024 B / instruction).  Total ds_read instructions for the A+B fragment:
(16384 + 16384) / 1024 = 32 instructions per K-tile.

With ``kpack=2`` the codegen widens the access pattern to ``ds_read_b128``
(16 B per lane × 64 lanes = 1024 B / instruction, but this is per-fragment-half
so effectively halves the round-trip count for the same total bytes
transferred): 16 instructions per K-tile.  With K=16384 / BK=64 = 256 K-tiles
per output tile and 8 output tiles per CU at this geometry, the LDS issue-slot
pressure drops by ~50%, which (per the K-905 PMC trace) is the dominant
component of the 41× LDS-stall bubble.

Why not generalize the tile shape
---------------------------------
Per K-883 R1 strict-equality dispatch and K-909 R-K909 narrow-cohort rule:
``kpack=2`` is **harmful** outside this exact LDS-bound corner because

  - On smaller K (≤ 4096) it doubles the LDS footprint per K-tile without
    enough K-tiles to amortize the codegen overhead (K-854 / K-864 mid-rect
    NO-LAND streak).
  - On larger M/N tiles the doubled fragment can blow the 64 KiB workgroup
    LDS budget (caught by L4 in the harness, see ``GuardedOverride.lds_ok``).
  - On non-fp16/bf16 dtypes the ds_read width arithmetic does not transfer
    (FP4 packs differently; FP8 hits a different codegen path).

The strict cohort here is therefore the 2 cells K-905 PMC-grounded as the
LDS-bound corner: ``(1024, 1024, 16384, fp16)`` and ``(1024, 1024, 16384, bf16)``.

The gate is OFF by default; arm with ``TB_K935_ENABLE=1`` and validate via
``scripts/run_falsification_k935.py`` before relying on it in production
benchmarks.

Trail: K-905 (mechanistic RC), K-913 (PMC repro), K-883 (guarded-override
pattern), K-901 (harness module), K-909 (R-K909 narrow-cohort rule),
K-854 / K-864 (kpack=2 mid-rect NO-LAND counterexamples justifying the
strict gate).
"""

from __future__ import annotations

import torch

from ..guarded_override import GuardedOverride, OverrideRegistry


# K-905 / K-913 cohort A: M=N=1024, K=16384, fp16+bf16.  Strict-equality only
# (R1 — no range predicates; K-909 R-K909 narrow-cohort rule).
_K935_COHORT_KEYS = [
    (1024, 1024, 16384, torch.float16),
    (1024, 1024, 16384, torch.bfloat16),
]


def _k935_payload(M: int, N: int, K: int, dtype: torch.dtype):
    """Override payload for cohort-A cells.

    - ``kpack=2``      widens ds_read to b128 (halves LDS round-trips,
                       targeting the K-905 SQ_WAIT_INST_LDS bubble).
    - ``num_warps=8``  pinned to match the production default explicitly.
    - ``num_stages=2`` matches the production NS=2 selected by Origami for
                       this geometry (asserted explicitly so the override
                       is robust against any future Origami default change
                       on this exact key).
    """
    return {
        "kpack": 2,
        "num_warps": 8,
        "num_stages": 2,
    }


K935_GATE = GuardedOverride(
    ticket="K-935",
    env_var="TB_K935_ENABLE",
    cohort_keys=_K935_COHORT_KEYS,
    dtype_allowlist=(torch.float16, torch.bfloat16),
    override_fn=_k935_payload,
    cohort_size_cap=4,  # 2 keys; tight cap defends against accidental table growth
)


# Register on import.  Default is fail-closed — the env-var killswitch keeps
# the gate non-firing until an operator explicitly opts in.  Idempotent under
# the registry's ticket-uniqueness check.
OverrideRegistry.register(K935_GATE)
