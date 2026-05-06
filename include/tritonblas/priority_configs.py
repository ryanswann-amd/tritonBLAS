"""
Priority kernel configurations seeded from hipBLASLt-informed empirical search.

For shapes where the Origami selector under-performs vs the hipBLASLt
heuristic-picked solution by >=15 percentage points, we override the tile
choice and pipeline parameters with an empirically-validated configuration
discovered by sweeping a candidate pool informed by hipBLASLt's selection
patterns (asymmetric tiles + non-default group_m).

Source: K-549 — top-5 residual-shape sweep on MI300X. Each entry was the
best of ~280 (tile, ns, num_warps, matrix_instr) combinations evaluated by
direct kernel timing on a single MI300X (gfx942), beating the Origami +
post-hoc 256x256x64 heuristic by 5-22%.

Lookup key: (M, N, K, dtype_str) where dtype_str matches the string
returned by ``OrigamiMatmulSelector.dtype_to_str`` (one of
``"f16"``, ``"bf16"``, ``"f32"``, ``"f8"``, ``"f4"``, etc.).

Effect:
- ``OrigamiMatmulSelector`` consults this map AFTER its post-hoc 256x256x64
  override and, if the (M, N, K, dtype) is present, replaces the chosen
  tile, ``num_stages``, and ``group_m`` with the priority entry. The full
  config (incl. ``num_warps``, ``matrix_instr_nonkdim``, ``waves_per_eu``,
  ``kpack``) is exposed via ``selector.priority_config`` for callers (the
  persistent + StreamK matmul paths) to honor.
- The autotuner (``tools/triton_gemm_bench.py``) prepends the priority
  config to its tuning space so that the seeded candidates are evaluated
  first; this both reduces autotune wallclock for known-residual shapes
  and ensures the seeded config is always considered.

To extend this map:
1. Pick a residual shape whose tritonblas ratio is below 0.85x of the
   hipBLASLt baseline.
2. Run ``state/mc2/workspaces/K-549/scripts/sweep_residuals.py`` against
   that shape (extend the ``RESIDUALS`` list).
3. Take the row's ``best_tile``, ``best_ns``, ``best_nw``, ``best_mfma``,
   ``best_wpe``, ``best_gm`` and add a ``PriorityConfig`` entry.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict


@dataclass(frozen=True)
class PriorityConfig:
    """A complete kernel-launch configuration, used to override the
    Origami selector's pick for a known under-performing shape.

    All values must be in the tritonblas autotune search space
    (``tools/triton_gemm_bench.py:get_full_tuning_space``).

    Attributes:
        block_m: BLOCK_SIZE_M, in {16, 32, 64, 128, 256}.
        block_n: BLOCK_SIZE_N, in {16, 32, 64, 128, 256}.
        block_k: BLOCK_SIZE_K, in {16, 32, 64, 128, 256, 512}.
        group_m: GROUP_SIZE_M (workgroup mapping), in {1, 4, 6, 8, 16, 32}.
        num_stages: pipeline stage count, in {2, 3}.
        num_warps: warps per thread block, in {4, 8, 16}.
        matrix_instr_nonkdim: MFMA non-K dimension, in {16, 32}.
        waves_per_eu: waves per execution unit, in {0, 1, 2, 4}.
        kpack: K-axis packing factor; locked to 1 (per K-524 falsification
            of the K-383 kpack=2 prediction).
    """
    block_m: int
    block_n: int
    block_k: int
    group_m: int
    num_stages: int
    num_warps: int
    matrix_instr_nonkdim: int
    waves_per_eu: int = 0
    kpack: int = 1


# (M, N, K, dtype_str) -> PriorityConfig
# Empirically derived in K-549; see sweep_results.csv for source data.
PRIORITY_CONFIGS: Dict[Tuple[int, int, int, str], PriorityConfig] = {
    # K-549 sub-band B residuals (asymmetric-tile gap, post-hoc
    # 256x256x64 override harms): 128x256x64 (or transpose) wins.
    # 8192x1024x8192 — sub-band B asymmetric-tall;
    # tritonblas baseline 0.65x, after 0.78-0.80x.
    (8192, 1024, 8192, "f16"): PriorityConfig(
        block_m=256, block_n=128, block_k=64,
        group_m=8, num_stages=2,
        num_warps=8, matrix_instr_nonkdim=16),
    (8192, 1024, 8192, "bf16"): PriorityConfig(
        block_m=128, block_n=256, block_k=64,
        group_m=4, num_stages=2,
        num_warps=8, matrix_instr_nonkdim=16),
    # 2048x4096x4096 — sub-band B cube; baseline 0.63x, after 0.75-0.76x.
    (2048, 4096, 4096, "f16"): PriorityConfig(
        block_m=128, block_n=256, block_k=64,
        group_m=4, num_stages=2,
        num_warps=8, matrix_instr_nonkdim=16),
    (2048, 4096, 4096, "bf16"): PriorityConfig(
        block_m=128, block_n=256, block_k=64,
        group_m=4, num_stages=2,
        num_warps=8, matrix_instr_nonkdim=16),
    # 6144x4096x4096 — sub-band A overhang (codegen-bound per K-383);
    # tile change yields only +5% but is still positive.
    (6144, 4096, 4096, "f16"): PriorityConfig(
        block_m=256, block_n=128, block_k=64,
        group_m=4, num_stages=2,
        num_warps=8, matrix_instr_nonkdim=16),
}


def lookup(M: int, N: int, K: int, dtype_str: str) -> Optional[PriorityConfig]:
    """Return the priority configuration for ``(M, N, K, dtype_str)``, or
    ``None`` if the shape is not in the seeded set.

    ``dtype_str`` must use the abbreviated form from
    ``OrigamiMatmulSelector.dtype_to_str`` (e.g. ``"f16"``, ``"bf16"``).
    """
    return PRIORITY_CONFIGS.get((M, N, K, dtype_str))


def as_autotune_dict(pc: PriorityConfig, num_cus: int,
                     chunk_size: int = 32) -> Dict:
    """Convert a ``PriorityConfig`` to the dict shape consumed by the
    autotuner in ``tools/triton_gemm_bench.py``.
    """
    return {
        "BLOCK_SIZE_M": pc.block_m,
        "BLOCK_SIZE_N": pc.block_n,
        "BLOCK_SIZE_K": pc.block_k,
        "GROUP_SIZE_M": pc.group_m,
        "NUM_SMS": num_cus,
        "num_warps": pc.num_warps,
        "num_stages": pc.num_stages,
        "waves_per_eu": pc.waves_per_eu,
        "matrix_instr_nonkdim": pc.matrix_instr_nonkdim,
        "kpack": pc.kpack,
        "CHUNK_SIZE": chunk_size,
    }
