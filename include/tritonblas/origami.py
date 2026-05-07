from __future__ import annotations
import itertools
import os
import torch
import origami
import math
from math import ceil


# Shape-aware MFMA instruction size selection.
#
# Triton's AMD backend `matrix_instr_nonkdim` arg picks the M/N dim of the MFMA
# instruction family.  The two practical choices on gfx942/gfx950 for FP16/BF16/
# FP8 are 16 (16x16x16/32) and 32 (32x32x8/16).  Both have matched per-lane
# throughput, but trade off:
#   - 16x16:  more MFMA insts per tile -> better issue-overlap with mem ops;
#             smaller AGPR/VGPR footprint; better for memory-bound shapes.
#   - 32x32:  fewer MFMA insts per tile (each 2x latency); better at hiding
#             scheduling gaps for compute-bound, large-K, asymmetric shapes;
#             larger accumulator footprint.
#
# Historical default in tritonblas was a hardcoded 16 (see matmul.py).
# Compute-underutilized large-skinny shapes that survived the dispatch
# heuristic fixes can be underutilized at the 16x16 instruction shape;
# this module exposes 32 as an opt-in second instruction shape, gated by a
# tile/shape-alignment predicate so that switching is always legal.

_MFMA_OVERRIDE_ENV = "TRITONBLAS_MFMA_INSTR_SIZE"


def _parse_mfma_override(value):
    """Validate an mfma_instr_size override.  Accepts None / "" / 16 / 32."""
    if value is None or value == "":
        return None
    try:
        ival = int(value)
    except (TypeError, ValueError):
        return None
    if ival in (16, 32):
        return ival
    return None


def select_mfma_instr_size(
    m: int,
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    block_k: int,
    largest_input_bitsize: int,
    arch_name: str,
    override: int | None = None,
) -> int:
    """Choose `matrix_instr_nonkdim` (16 or 32) for a (problem, tile) pair.

    Default behavior (override is None) returns 16, matching the historical
    hardcoded value (matmul.py used `mfmaInstrSize = 16` for every kernel
    launch).  The 32-MFMA path is *opt-in only* via:
      * the explicit `mfma_instr_size=32` constructor arg on
        OrigamiMatmulSelector, or
      * the TRITONBLAS_MFMA_INSTR_SIZE=32 env var.

    Why opt-in only -- a benchmark sweep on MI300X / gfx942 showed 32x32
    MFMA regressing 5-10% on every large-skinny shape tested across both
    the persistent and stream-K kernel paths, with no shape achieving
    >=5% uplift.  Auto-promoting based on shape heuristics therefore
    introduces regressions, so the predicate is intentionally
    conservative: it never returns 32 unless the operator asked for it.

    The plumbing is still useful for autotuner integration and future
    architectures (gfx950 may behave differently); keeping the predicate
    function signature stable so Origami can extend the search space.

    Validation (the override / arch / tile / problem gates below) still
    applies when an override is supplied: an invalid request silently
    falls back to 16 to keep the kernel launches legal.
    """
    if override is None:
        return 16  # default: legacy behavior, no auto-promotion

    if override == 16:
        return 16
    if override != 32:
        return 16  # unknown override -> safe default

    # override == 32: validate that 32 is legal for this (problem, tile, dtype).
    # Architecture / dtype gate.
    if largest_input_bitsize > 16:
        # FP32 path uses 16x16x4 family; 32x32x* not supported on Triton's
        # AMD backend for FP32.
        return 16
    if arch_name not in ("gfx942", "gfx950"):
        # gfx90a / older archs lack the 32x32x16 family used here.
        return 16

    # Tile alignment gate: matrix_instr_nonkdim=32 requires the M and N
    # tile dims to be >= 32 and divisible by 32.  Otherwise Triton would
    # reject the config or silently fall back; force 16 instead.
    if block_m < 32 or block_n < 32:
        return 16
    if (block_m % 32) != 0 or (block_n % 32) != 0:
        return 16

    # Problem-dim sanity gate.
    if m < 32 or n < 32:
        return 16

    return 32


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
        mfma_instr_size: int | None = None,
    ):
        # Save tensor sizes
        self._m = m
        self._n = n
        self._k = k
        self.streamk = streamk
        self._num_stages = num_stages
        # MFMA shape: explicit override (autotune sweeps), else env var, else
        # let the shape-aware predicate decide once we know the picked tile.
        env_override = _parse_mfma_override(os.environ.get(_MFMA_OVERRIDE_ENV))
        self._mfma_instr_size_override = (
            mfma_instr_size if mfma_instr_size is not None else env_override
        )
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

        # Heuristic to favor 256x256x64 tile when close~
        if (check_triton_lds_capacity(256, 256, 64, bytes_a, bytes_b, lds_cap, self._num_stages) and
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
    def mfma_instr_size(self) -> int:
        """`matrix_instr_nonkdim` (16 or 32) for the picked tile.

        Returns the explicit override if one was supplied (constructor arg or
        TRITONBLAS_MFMA_INSTR_SIZE env var); otherwise applies the shape-aware
        predicate.  Cached on first access since selector state is immutable
        after __init__.
        """
        cached = getattr(self, "_mfma_instr_size_cached", None)
        if cached is not None:
            return cached
        largest_input_bitsize = max(self._a_dtype_bitsize, self._b_dtype_bitsize)
        chosen = select_mfma_instr_size(
            m=self._m,
            n=self._n,
            k=self._k,
            block_m=self.block_m,
            block_n=self.block_n,
            block_k=self.block_k,
            largest_input_bitsize=largest_input_bitsize,
            arch_name=self._arch_name,
            override=self._mfma_instr_size_override,
        )
        self._mfma_instr_size_cached = chosen
        return chosen

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
