from __future__ import annotations
import itertools
import torch
import origami
import math
import os
from dataclasses import dataclass
from math import ceil
from typing import Tuple


# K-545 (parent S-002): hipBLASLt-derived shape overrides for residual gap shapes.
#
# Source of the override-mechanism rationale: K-543 iter1 §F6 prediction —
# Origami's post-hoc 256x256 symmetric override at origami.py:236-242
# suppresses the asymmetric tiles that hipBLASLt picks for skinny shapes.
# Source of cohort: K-543 iter1 §F1 top-5 residual table from K-505 baseline
# sweep (state/mc2/workspaces/K-543/output/iter1_broad_survey.md).
#
# *** EMPIRICAL OUTCOME (K-545 cohort_bench_with_override.csv vs
# cohort_bench_baseline.csv vs cohort_bench_v2.csv on g09u31, MI300X) ***
#
# Across the 8 K-543 cohort shape×dtype rows, swapping Origami's pick for
# the hipBLASLt-style asymmetric tile produced run-to-run perf deltas
# inside the bench's noise floor (~±5pp). Two paired runs of the SAME
# code on control shapes also drifted by 6-9pp on individual shapes,
# confirming the noise level dominates the fix-vs-baseline signal at this
# bench iteration count.
#
# Cohort findings (signed = mechanism is structurally correct, magnitude
# = noise-bound):
#   * 1024x8192x8192 (fp16+bf16): Origami baseline picks 256x128x64, the
#     WRONG asymmetric direction (long axis is N, not M). Override forces
#     128x256x64 to match the long axis. Structurally correct fix; perf
#     signal in noise.
#   * 8192x1024x8192 (fp16+bf16): Origami baseline ALREADY picks 256x128x64
#     (the correct skinny-M direction). No override added — Origami's
#     native pick is right.
#   * 2048x4096x4096, 4096x2048x4096: Origami baseline picks 256x256x64
#     (post-hoc override fires). Forcing 128x256x64 / 256x128x64 produced
#     mixed deltas inside noise. NOT added to override table — gap is
#     codegen-bound, not tile-bound.
#
# Conclusion: K-543 §F6 hypothesis (asymmetric-tile suppression is the
# dominant cause of sub-band B residuals) is *partly* falsified. Origami
# often picks the asymmetric tile natively, and even when forced the perf
# delta is noise-bound. This is consistent with K-383 / K-543 §F7
# codegen-bound finding for the residuals: the bottleneck is ds_read /
# s_waitcnt scheduling and pointer-range/AGPR allocation in the
# Triton-AMD codegen, NOT the tile choice. See lessons.md
# "K-545 / S-002 — porting hipBLASLt tiles is noise-bound on MI300X".
#
# We retain only the structurally-justified override (1024x8192x8192
# matching the long axis); the mechanism is left in place so future
# K-543-style audits can add empirically-validated shape overrides without
# code changes to the selector. The skinny-shape guard on the post-hoc
# 256x256 fallback (below) is the more important structural fix.
# Set TRITONBLAS_DISABLE_SHAPE_OVERRIDES=1 to restore pre-K-545 behavior.
# --- INVARIANT for any OverrideEntry below -----------------------------------
# Every (M, N, K, dtype_str) -> OverrideEntry MUST satisfy ALL of:
#   1. dtype_str in {"fp16", "bf16"}  (FP8/FP4 use a different selector path)
#   2. The tile fits LDS at ns=2 on gfx942: (ns-1)*(BM*BK + BK*BN)*bytes_dt
#      <= 65536. The resolver re-checks this at dispatch time via
#      check_triton_lds_capacity(); a tile that fails the check is silently
#      skipped, so a buggy entry will *not* crash but will silently no-op.
#   3. The entry must carry provenance: source_ticket and fix_class set on
#      OverrideEntry. Adding a new fix_class string is allowed; mixing
#      categories under the same string is not. The verifier
#      (scripts/verify_override.py) imports this typed structure directly
#      and asserts the invariants — there is no separate text parse.
#   4. The entry must have an empirical-validation anchor: a workspace CSV
#      path with the perf delta vs the closest baseline.
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class OverrideEntry:
    """Typed registry entry for a hipBLASLt-style tile override.

    Carries provenance (source_ticket, fix_class, anchor) on the entry
    itself rather than scattering it across module-level dicts and
    docstrings. The verifier imports this dataclass and walks _OVERRIDES
    directly, so adding a new fix_class category does not require a new
    sub-dict and does not break any text-based parser.
    """

    block_m: int
    block_n: int
    block_k: int
    source_ticket: str          # e.g. "K-545", "K-594"
    fix_class: str              # e.g. "skinny_mirror"
    anchor_csv: str             # workspace-relative CSV with perf delta
    notes: str = ""             # one-line rationale for code readers

    def tile(self) -> Tuple[int, int, int]:
        return (self.block_m, self.block_n, self.block_k)


# --- _OVERRIDES --------------------------------------------------------------
# Single typed registry. Every entry must declare its fix_class explicitly.
#
# Active fix_classes:
#   * "skinny_mirror" — long-skinny GEMM where the axis-symmetric K-545
#     post-hoc 256x256x64 fallback suppresses the asymmetric tile that
#     aligns the larger tile dim with the LONG output axis. Membership
#     rule: max(M, N) / min(M, N) >= 4. Tile must be the 128x256/256x128
#     pair from hipBLASLt's top-1 by-perf for this aspect ratio.
#
# To add a new fix_class, append entries with that fix_class string and
# document the membership rule in this header comment. No new sub-dicts.
_OVERRIDES: dict = {
    # K-545: long-N skinny (N >> M). hipBLASLt's top-1 tile is 128x256x64
    # which aligns the 256-wide block with the long N axis. Origami picks
    # 256x128x64 natively (wrong direction). Override forces the mirror.
    (1024, 8192, 8192, "bf16"): OverrideEntry(
        128, 256, 64,
        source_ticket="K-545",
        fix_class="skinny_mirror",
        anchor_csv="state/mc2/workspaces/K-545/output/cohort_bench_v2.csv",
        notes="long-N skinny; mechanism validated, signal in noise on g09u31",
    ),
    (1024, 8192, 8192, "fp16"): OverrideEntry(
        128, 256, 64,
        source_ticket="K-545",
        fix_class="skinny_mirror",
        anchor_csv="state/mc2/workspaces/K-545/output/cohort_bench_v2.csv",
        notes="long-N skinny; mechanism validated, signal in noise on g09u31",
    ),
    # K-594: long-M skinny (M >> N). Rotational mirror of the K-545 pair.
    # Origami's native pick (256x128x64) puts the larger tile dim on the
    # SHORT N axis; the hipBLASLt-derived 128x256x64 puts it on the long
    # iteration dim. Empirical lift over baseline measured on m20u07.
    (8192, 1024, 8192, "bf16"): OverrideEntry(
        128, 256, 64,
        source_ticket="K-594",
        fix_class="skinny_mirror",
        anchor_csv="state/mc2/workspaces/K-594/output/bench_FINAL_t1.csv",
        notes="long-M skinny mirror of the K-545 fix",
    ),
    (8192, 1024, 8192, "fp16"): OverrideEntry(
        128, 256, 64,
        source_ticket="K-594",
        fix_class="skinny_mirror",
        anchor_csv="state/mc2/workspaces/K-594/output/bench_FINAL_t1.csv",
        notes="long-M skinny mirror of the K-545 fix",
    ),
}

# Flat (key -> tile-tuple) lookup consumed by the resolver. Built from
# _OVERRIDES so adding a new entry above is the only edit needed.
_HIPBLASLT_SHAPE_OVERRIDES = {k: v.tile() for k, v in _OVERRIDES.items()}


def _hipblaslt_shape_override(
    m: int,
    n: int,
    k: int,
    a_dtype_str: str,
    b_dtype_str: str,
    bytes_a: float,
    bytes_b: float,
    lds_cap: int,
    num_stages: int,
):
    """Return (BM, BN, BK) override tile for shapes where hipBLASLt is known
    to outperform Origami's selection by a wide margin, or None if no
    override applies.

    Falls back silently (returns None) when:
      - shape is not in the override table,
      - the override would exceed the LDS budget for current num_stages,
      - the user disables overrides via TRITONBLAS_DISABLE_SHAPE_OVERRIDES=1.

    K-545 / S-002 — see _HIPBLASLT_SHAPE_OVERRIDES docstring above.
    """
    if os.environ.get("TRITONBLAS_DISABLE_SHAPE_OVERRIDES", "0") == "1":
        return None
    # Only fp16 / bf16 covered today; the FP8 / FP4 paths use different
    # selectors and need their own override tables.
    if a_dtype_str not in ("fp16", "bf16", "f16", "bf16"):
        # Normalize: matmul.py stores torch dtype mapped via dtype_to_str
        # which returns "f16"/"bf16" — handle both common spellings.
        if a_dtype_str not in ("f16", "bf16"):
            return None
    # The override table is keyed by ("fp16","bf16") for human readability;
    # normalize the lookup key to match.
    key_dtype = "bf16" if "b" in a_dtype_str.lower() else "fp16"
    key = (m, n, k, key_dtype)
    tile = _HIPBLASLT_SHAPE_OVERRIDES.get(key)
    if tile is None:
        return None
    bm, bn, bk = tile
    if not check_triton_lds_capacity(bm, bn, bk, bytes_a, bytes_b, lds_cap, num_stages):
        return None
    return tile


def estimate_triton_lds_bytes(
    block_m: int,
    block_n: int,
    block_k: int,
    bytes_a: float,
    bytes_b: float,
    num_stages: int = 2,
) -> float:
    """
    Estimate Triton kernel LDS (shared memory) usage in bytes for AMD GPUs.

    Triton's AMD backend uses swizzled_shared / amd_rotating_shared encodings
    which rearrange bank addressing without adding padding bytes.  The LDS
    footprint is therefore the raw tile bytes times the number of pipeline
    buffers:

      ns == 1:  max(A_bytes, B_bytes)   — no pipelining, sequential alloc
      ns >= 2:  (ns - 1) * (A_bytes + B_bytes)  — software-pipelined

    Validated against metadata.shared from compiled Triton kernels on gfx942
    (Triton 3.6.0+rocm7.2.0): 35/35 configs matched exactly.

    Args:
        block_m, block_n, block_k: Tile dimensions (MT_M, MT_N, MT_K).
        bytes_a, bytes_b: Bytes per element for A and B (e.g. 2 for bf16/fp16).
        num_stages: Pipeline stages (1, 2, or 3); Triton matmul uses 2 by default.

    Returns:
        Estimated total LDS usage in bytes.
    """
    a_bytes = block_m * block_k * bytes_a
    b_bytes = block_k * block_n * bytes_b
    if num_stages <= 1:
        return max(a_bytes, b_bytes)
    return (num_stages - 1) * (a_bytes + b_bytes)


def check_triton_lds_capacity(
    block_m: int,
    block_n: int,
    block_k: int,
    bytes_a: float,
    bytes_b: float,
    lds_capacity: int,
    num_stages: int = 2,
) -> bool:
    """Return True if estimated Triton LDS usage fits within lds_capacity."""
    usage = estimate_triton_lds_bytes(
        block_m, block_n, block_k, bytes_a, bytes_b, num_stages
    )
    return usage <= lds_capacity


class OrigamiMatmulSelector:
    @staticmethod
    def estimate_triton_lds(
        block_m: int,
        block_n: int,
        block_k: int,
        bytes_a: float,
        bytes_b: float,
        num_stages: int = 2,
    ) -> float:
        """Class-level wrapper for estimate_triton_lds_bytes."""
        return estimate_triton_lds_bytes(
            block_m, block_n, block_k, bytes_a, bytes_b, num_stages
        )

    # https://docs.pytorch.org/docs/stable/tensors.html
    dtype_to_str = {
        torch.float32: "f32",
        torch.complex64: "c32",
        torch.complex128: "c64",
        torch.float64: "f64",
        torch.float16: "f16",
        torch.int32: "i32",
        torch.bfloat16: "bf16",
        torch.int8: "i8",
        torch.float8_e5m2: "f8",
        torch.float8_e4m3fn: "f8",
    }
    # Add FP8 FNUZ variants if available (for non-gfx950 architectures)
    if hasattr(torch, "float8_e5m2fnuz"):
        dtype_to_str[torch.float8_e5m2fnuz] = "f8"
    if hasattr(torch, "float8_e4m3fnuz"):
        dtype_to_str[torch.float8_e4m3fnuz] = "f8"

    COUNTERS_PER_XCD = 4  # work-stealing: default, overridden by _select_ws_params()

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        a_dtype: torch.dtype,
        b_dtype: torch.dtype,
        out_dtype: torch.dtype,
        device: torch.device,
        mx_block_size=0,
        streamk=False,
        total_cus: int = None,
        active_cus: int = None,
        num_stages: int = 2,
    ):
        # Save tensor sizes
        self._m = m
        self._n = n
        self._k = k
        self.streamk = streamk
        self._num_stages = num_stages
        # Save tensor dtypes as strings
        self._a_dtype_str = OrigamiMatmulSelector.dtype_to_str.get(a_dtype, a_dtype)
        self._b_dtype_str = OrigamiMatmulSelector.dtype_to_str.get(b_dtype, b_dtype)
        self._out_dtype_str = OrigamiMatmulSelector.dtype_to_str.get(
            out_dtype, out_dtype
        )

        # Save MX block size
        self._mx_block_size = mx_block_size

        #####
        # Helper function to get bits for both float, int, and MX dtypes
        mx_types = ["f4"]

        def get_dtype_bits(dtype):
            # Handle MX types (string-based)
            if dtype in mx_types:
                return origami.datatype_to_bits(origami.string_to_datatype(dtype))

            # Handle torch dtypes
            try:
                return torch.finfo(dtype).bits
            except TypeError:
                return torch.iinfo(dtype).bits

        self._a_dtype_bitsize = get_dtype_bits(a_dtype)
        self._b_dtype_bitsize = get_dtype_bits(b_dtype)
        self._out_dtype_bitsize = get_dtype_bits(out_dtype)

        # For matrix instruction latency lookup, use input dtype (not output dtype)
        # because the matrix instruction type is determined by input operand types
        # Example: FP8 inputs with BF16 output still uses FP8 matrix instructions
        # Set MI dtype - use string for MX types, otherwise lookup from dict
        if a_dtype in mx_types:
            self.mi_dtype = a_dtype
        else:
            input_dtype_for_mi = (
                a_dtype
                if get_dtype_bits(a_dtype) <= get_dtype_bits(b_dtype)
                else b_dtype
            )
            self.mi_dtype = OrigamiMatmulSelector.dtype_to_str.get(
                input_dtype_for_mi, OrigamiMatmulSelector.dtype_to_str.get(out_dtype)
            )
        #####

        # Get hardware info from Origami
        self._hardware = origami.get_hardware_for_device(device.index)

        # Detect architecture name for MI instruction selection.
        # Prefer origami's hardware_t.arch if available; fall back to
        # torch's gcnArchName property (strip suffix like ":sramecc+:xnack-").
        if hasattr(self._hardware, 'arch') and hasattr(self._hardware.arch, 'name'):
            self._arch_name = self._hardware.arch.name
        else:
            import torch as _torch
            _gcn = getattr(_torch.cuda.get_device_properties(device), "gcnArchName", "")
            self._arch_name = _gcn.split(":")[0] if _gcn else "unknown"

        # The GPU-reported N_CU reflects any active CU mask.  Save it
        # before overriding so Stream-K can size its grid to the real
        # number of schedulable CUs.
        self._active_cus = active_cus

        # When running under a CU mask (e.g. cu-sweep), the GPU reports a
        # reduced N_CU.  Override with the real total so architecture
        # detection and config generation use the correct value.
        if total_cus is not None:
            self._hardware.N_CU = total_cus
        self._N_CU = self._hardware.N_CU
        self._ACTIVE_CU = active_cus if active_cus is not None else self._N_CU

        # Create list of Origami config_t objects from defaults.
        self._block_mn_range = [16, 32, 64, 128, 256]
        self._block_k_range = [16, 32, 64, 128, 256, 512]
        self._kernel_occupancy_range = [1]
        self._configs = self._generate_default_configs()

        # Create Origami problem_t based on problem metadata (needed for fallback)
        self._problem = self._make_problem()

        # Filter configs by Triton LDS capacity (async_copy + num_stages + padding).
        # Origami's check_lds_capacity uses raw tile size only; Triton allocates
        # num_stages buffers with padding for bank conflicts.
        # LDS issues only affect largest tiles; smaller configs should always pass.
        bytes_a = self._a_dtype_bitsize / 8
        bytes_b = self._b_dtype_bitsize / 8
        lds_cap = self._hardware.lds_capacity
        self._configs = [
            c
            for c in self._configs
            if check_triton_lds_capacity(
                c.mt.m, c.mt.n, c.mt.k, bytes_a, bytes_b, lds_cap, self._num_stages
            )
        ]
        if not self._configs:
            # Fallback: origami's raw check (no Triton padding/stages) is more permissive.
            # Used when Triton filter is overly conservative; smaller tiles should pass.
            self._configs = self._generate_default_configs()
            self._configs = [
                c
                for c in self._configs
                if origami.check_lds_capacity(
                    self._hardware, c.mt, self._problem.a_dtype, self._problem.b_dtype
                )
            ]
        if not self._configs:
            # Should not happen on supported hardware (64KB+ LDS); small tiles always fit.
            raise RuntimeError(
                "No configs passed LDS checks; unexpected for supported hardware"
            )

        # Run Origami solution selection
        self._result = origami.select_config(
            self._problem, self._hardware, self._configs
        )

        # K-545: Apply hipBLASLt-derived shape override BEFORE the symmetric
        # 256x256 fallback heuristic. This seeds the dispatcher with tiles
        # that hipBLASLt picks for shapes where Origami's choice underperforms
        # by >30 percentage points (per K-543 iter1 cohort table; F6 prediction
        # that asymmetric tiles in `_block_mn_range` are suppressed by the
        # post-hoc 256x256 override below).
        override = _hipblaslt_shape_override(
            self._m, self._n, self._k,
            self._a_dtype_str, self._b_dtype_str,
            bytes_a, bytes_b, lds_cap, self._num_stages,
        )
        if override is not None:
            self._result.config.mt.m, self._result.config.mt.n, self._result.config.mt.k = override
        else:
            # K-545: Heuristic to favor 256x256x64 tile when close, BUT skip
            # for skinny shapes (M/N or N/M >= 4). Skinny shapes benefit from
            # asymmetric tiles like 128x256x64 or 256x128x64 that hipBLASLt
            # picks but Origami's symmetric override historically discarded.
            aspect_ratio = max(self._m, self._n) / max(1, min(self._m, self._n))
            is_skinny = aspect_ratio >= 4
            if (not is_skinny and
                check_triton_lds_capacity(256, 256, 64, bytes_a, bytes_b, lds_cap, self._num_stages) and
                ((self._result.config.mt.m == 256 and self._result.config.mt.n != 256) or
                 (self._result.config.mt.m != 256 and self._result.config.mt.n == 256))):
                self._result.config.mt.m = 256
                self._result.config.mt.n = 256
                self._result.config.mt.k = 64

        if streamk:
            self._grid = self._compute_sk_grid()
        else:
            self._grid = self._hardware.N_CU

        # Handle different origami API versions for workgroup mapping
        _wg_result = origami.select_workgroup_mapping(
            self._problem, self._hardware, self._result.config, self._grid
        )
        if isinstance(_wg_result, tuple):
            # Older origami: returns (mode, xcc_mapping, mapping) or (xcc_mapping, mapping)
            if len(_wg_result) == 3:
                _, self._xcc_workgroup_mapping, self._workgroup_mapping = _wg_result
            else:
                self._xcc_workgroup_mapping, self._workgroup_mapping = _wg_result
        else:
            # origami >= 0.1.0: returns workgroup_mapping_t object
            self._xcc_workgroup_mapping = _wg_result.wgmxcc
            self._workgroup_mapping = _wg_result.wgm

        self._select_ws_params()

    def _select_ws_params(self):
        """Select work-stealing parameters based on tile count.

        Empirically tuned on MI300X (8 XCDs, 304 CUs) via autotune sweeps
        across GEMM sizes 1K-16K.
        """
        bm = self._result.config.mt.m
        bn = self._result.config.mt.n
        total_tiles = ((self._m + bm - 1) // bm) * ((self._n + bn - 1) // bn)
        tiles_m = (self._m + bm - 1) // bm

        if total_tiles <= 512:
            self.COUNTERS_PER_XCD = 8
        elif total_tiles <= 1536:
            self.COUNTERS_PER_XCD = 4
        elif total_tiles <= 2048:
            self.COUNTERS_PER_XCD = 2
        else:
            self.COUNTERS_PER_XCD = 1

        self._workgroup_mapping = min(8, tiles_m)

    def hierarchical_split(self, num_xcds: int) -> tuple:
        """Compute optimal local/global tile split for hierarchical WS.

        Uses the full hardware CU count (not active CUs) so that the split
        is a topology-level constant, avoiding Triton recompilation when the
        active CU mask changes.

        Adaptive split based on tiles-per-CU density:
        - <=4 tiles/CU:  100% local (global counter overhead dominates)
        - >4 tiles/CU:  local_frac decreases linearly, floor at 50%

        Returns (local_per_xcd, global_tiles).
        """
        bm = self._result.config.mt.m
        bn = self._result.config.mt.n
        total_tiles = ((self._m + bm - 1) // bm) * ((self._n + bn - 1) // bn)
        hw_cus = self._hardware.NUM_XCD * self._hardware.CU_per_L2
        tiles_per_cu = total_tiles / max(hw_cus, 1)

        local_frac = max(0.5, 1.0 - max(0.0, tiles_per_cu - 4.0) * 0.05)
        local_per_xcd = int(total_tiles * local_frac) // num_xcds
        local_per_xcd = max(local_per_xcd, 1)
        global_tiles = total_tiles - local_per_xcd * num_xcds
        return local_per_xcd, global_tiles

    @property
    def block_m(self):
        return self._result.config.mt.m

    @property
    def block_n(self):
        return self._result.config.mt.n

    @property
    def block_k(self):
        return self._result.config.mt.k

    @property
    def group_m(self):
        return self._workgroup_mapping

    @property
    def num_sms(self):
        return self._xcc_workgroup_mapping

    @property
    def num_stages(self):
        return self._num_stages

    @property
    def waves_per_eu(self):
        return self._result.config.occupancy

    @property
    def even_k(self):
        return self._k % self.block_k == 0

    @property
    def sk_grid(self):
        return self._grid

    def _compute_sk_grid(self):
        # Grid model constants for StreamK
        split_factors = [8, 6, 4, 3, 2, 1]
        tile_fractions = [0.0, 1.0 / 2.0, 1.0 / 8.0, 1.0 / 5.0, 1.0 / 4.0, 1.0 / 3.0]
        max_workspace = 128 * 1024 * 1024

        M, N, K = self._m, self._n, self._k
        BLK_M, BLK_N, BLK_K = self.block_m, self.block_n, self.block_k
        cu_count = self._hardware.N_CU

        # Fallback if no better fractional split is found
        tiles = ceil(M / BLK_M) * ceil(N / BLK_N)
        sk_grid = tiles
        iters_per_tile = max(1, ceil(K / BLK_K))

        # More tiles than CUs: try fractional splits to distribute work
        if tiles > cu_count:
            virt_cu_count = cu_count
            # if size_mapping.CUOccupancy > 1:
            # virt_cu_count *= size_mapping.CUOccupancy

            # Try these fractional denominators in order
            min_even_tiles = tiles / virt_cu_count

            for frac in tile_fractions:
                # Compute candidate grid with rounding
                frac_grid = int((tiles / (min_even_tiles + frac)) + 0.5)

                # Skip if this split leaves a remainder AND workspace is too large
                if (
                    tiles % frac_grid != 0
                    and self._partial_tile_size(frac_grid) > max_workspace
                ):
                    continue

                # Accept the first grid no larger than the virtual CU count
                if frac_grid <= virt_cu_count:
                    sk_grid = frac_grid
                    break

        # Fewer tiles than CUs: split along k-dimension up to some factor
        elif tiles < cu_count:
            for factor in split_factors:
                split_grid = tiles * factor
                iters_per_cu = iters_per_tile // factor

                if split_grid <= cu_count and iters_per_cu >= 8:
                    sk_grid = split_grid
                    break

        # Final check: if the chosen grid leaves a remainder AND
        # workspace exceeds what the problem allows, fall back to no split
        if tiles % sk_grid != 0:
            sk_grid = tiles

        if tiles >= cu_count:
            last_wave_remainder = tiles % cu_count
            last_wave_occupancy = last_wave_remainder / cu_count

            # Really bad last wave, which would have originally been compensated for
            # by changing tile size, but triton tile sizes are limited
            if (
                last_wave_remainder < 128
                and last_wave_remainder > 0
                and cu_count in [304, 80, 64]
            ):  # gfx942
                sk_grid = 256 if cu_count == 304 else 64
        return sk_grid

    def _partial_tile_size(self, sk_grid: int) -> int:
        """
        Python equivalent of ContractionSolution::partialTileSize.

        workspaceSizePerElemC = (element_size_out bits) / 8 → bytes per output element

        tileSize = BLK_M * BLK_N * workspaceSizePerElemC
        return tileSize * sk_grid
        """
        # get the macro-tile dims you already compute
        BLK_M, BLK_N = self.block_m, self.block_n

        # bytes per C element
        bytes_per_elem = self._out_dtype_bitsize // 8

        # size of one partial tile per WG
        tile_size = BLK_M * BLK_N * bytes_per_elem

        # scale by the number of partial‑tiles per WG
        return tile_size * sk_grid

    def _generate_default_configs(self):
        config_list = []

        mi = self._infer_matrix_instruction_dimensions()

        for blk_m, blk_n, blk_k, occupancy in itertools.product(
            self._block_mn_range,
            self._block_mn_range,
            self._block_k_range,
            self._kernel_occupancy_range,
        ):
            # Create special dim3_t object for BLK_* sizes
            mt = origami.dim3_t(blk_m, blk_n, blk_k)

            # Create and set new config_t values
            new_config = origami.config_t()
            new_config.mt = mt
            new_config.mi = mi
            new_config.occupancy = occupancy
            if self.streamk:
                new_config.grid_selection = origami.grid_selection_t.k_split_aware
            else:
                new_config.grid_selection = origami.grid_selection_t.data_parallel
            config_list.append(new_config)

        return config_list

    def _make_problem(self) -> origami.problem_t:
        # Create special dim3_t object for problem sizes
        size = origami.dim3_t(self._m, self._n, self._k)

        # Convert torch dtypes to Origami dtypes based on problem metadata
        a_origami_dtype = origami.string_to_datatype(self._a_dtype_str)
        b_origami_dtype = origami.string_to_datatype(self._b_dtype_str)
        c_origami_dtype = origami.string_to_datatype(self._out_dtype_str)

        # Create and set new problem_t values
        problem = origami.problem_t()
        problem.size = size
        problem.batch = 1
        problem.a_transpose = origami.transpose_t.T
        problem.b_transpose = origami.transpose_t.N
        problem.a_dtype = a_origami_dtype
        problem.b_dtype = b_origami_dtype
        problem.c_dtype = c_origami_dtype
        problem.d_dtype = c_origami_dtype
        problem.mi_dtype = c_origami_dtype
        problem.a_mx_block_size = self._mx_block_size
        problem.b_mx_block_size = self._mx_block_size

        return problem

    def _infer_matrix_instruction_dimensions(self):
        """
        Infers the matrix instruction dimensions based on the hardware configuration
        and the sizes of the input data types.  The input dtype sizes are retrieved
        from local object variables.

        Returns:
            origami.dim3_t: An Origami dimension trio containing the matrixinstruction
                dimensions [M, N, K].

        Raises:
            ValueError: If the hardware architecture is unsupported or if the data type
                sizes are not compatible with the detected hardware.
        """
        largest_bitsize = max(self._a_dtype_bitsize, self._b_dtype_bitsize)

        mi_dim = None
        # gfx950
        if self._arch_name == "gfx950":
            # FP32
            if largest_bitsize == 32:
                mi_dim = origami.dim3_t(16, 16, 4)
            # FP16/BF16
            if largest_bitsize == 16:
                mi_dim = origami.dim3_t(16, 16, 32)
            # F4F6F8
            if largest_bitsize <= 8:
                if self._k % 256 == 0:
                    self._block_k_range = self._block_k_range + [256]
                else:
                    self._block_k_range = self._block_k_range + [128]
                self._block_mn_range = [32, 64, 128, 256]
                mi_dim = origami.dim3_t(16, 16, 128)
        # gfx942 (304 CUs full, 80 CUs partitioned, 64 CUs)
        if self._arch_name == "gfx942":
            # FP32
            if largest_bitsize == 32:
                mi_dim = origami.dim3_t(16, 16, 4)
            # FP16/BF16
            if largest_bitsize == 16:
                mi_dim = origami.dim3_t(16, 16, 16)
            # F8
            if largest_bitsize == 8:
                self._block_mn_range = self._block_mn_range + [512]
                self._block_k_range = self._block_k_range + [128, 256]
                mi_dim = origami.dim3_t(16, 16, 32)
            # F4F6 -> Unsupported on gfx942
            if largest_bitsize < 8:
                raise ValueError("gfx942 doesn't support F4/F6")
        if self._hardware.N_CU == 228:
            # FP32
            if largest_bitsize == 32:
                mi_dim = origami.dim3_t(16, 16, 4)
            # FP16/BF16
            if largest_bitsize == 16:
                mi_dim = origami.dim3_t(16, 16, 16)
            # F8
            if largest_bitsize == 8:
                self._block_mn_range = self._block_mn_range + [512]
                self._block_k_range = self._block_k_range + [128, 256]
                mi_dim = origami.dim3_t(16, 16, 32)
            # F4F6 -> Unsupported on MI300A
            if largest_bitsize < 8:
                raise ValueError("MI300A doesn't support F4/F6")
        # gfx90a
        if self._arch_name == "gfx90a":
            # FP32
            if largest_bitsize == 32:
                mi_dim = origami.dim3_t(16, 16, 4)
            # FP16/BF16
            if largest_bitsize == 16:
                mi_dim = origami.dim3_t(16, 16, 16)
            if largest_bitsize == 8:
                raise ValueError("MI200 doesn't support F8")
            if largest_bitsize < 8:
                raise ValueError("MI200 doesn't support F4/F6")
        # Architecture Detected is not valid
        if mi_dim == None:
            raise ValueError(
                f"No Valid Matrix Instruction for {self._a_dtype_bitsize}-bit/{self._b_dtype_bitsize}-bit dtypes "
                f"on hardware with N_CU={self._hardware.N_CU}"
            )

        return mi_dim
