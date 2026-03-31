"""
GcnSimSelector — ISA-based kernel configuration selector.

Uses amdgcn_analyzer's cycle-level simulator to rank candidate tile
configurations by predicted latency. Replaces Origami's C++ analytical
model with actual assembly analysis.

Enable via environment variable:
    TBLAS_SELECTOR=gcnsim python3 my_script.py

Or use directly:
    from tritonblas.gcn_selector import GcnSimSelector
    selector = GcnSimSelector(M, N, K, a_dtype, b_dtype, c_dtype, device)
    c = tritonblas.matmul_lt(a, b, out, selector)
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from .origami import estimate_triton_lds_bytes, check_triton_lds_capacity


# Assembly cache: (bm, bn, bk, stages, warps, dtype_str, arch) -> asm file path
_ASM_CACHE: Dict[tuple, str] = {}

# Disk cache directory
_CACHE_DIR = Path.home() / ".tritonblas_gcnsim_cache"


def _cache_key(bm, bn, bk, stages, warps, dtype_str, arch):
    return (bm, bn, bk, stages, warps, dtype_str, arch)


def _get_arch_from_device(device: torch.device) -> str:
    """Get architecture name from torch device."""
    props = torch.cuda.get_device_properties(device)
    gcn = getattr(props, "gcnArchName", "")
    return gcn.split(":")[0] if gcn else "unknown"


def _get_num_cus(device: torch.device) -> int:
    """Get number of compute units."""
    return torch.cuda.get_device_properties(device).multi_processor_count


def _get_num_xcds(num_cus: int, arch: str) -> int:
    """Infer number of XCDs from CU count and architecture."""
    if arch == "gfx942":
        if num_cus >= 256:
            return 8
        elif num_cus >= 128:
            return 4
        elif num_cus >= 64:
            return 2
        return 1
    elif arch == "gfx950":
        return 8
    elif arch == "gfx90a":
        return 1
    return 1


def _get_lds_capacity(arch: str) -> int:
    """Get LDS capacity in bytes per workgroup."""
    if arch == "gfx950":
        return 163840  # 160 KB
    return 65536  # 64 KB (gfx942, gfx90a)


def _hw_string(arch: str) -> str:
    """Map architecture to amdgcn_analyzer hardware string."""
    mapping = {
        "gfx942": "mi300x",
        "gfx950": "mi350x",
        "gfx90a": "mi200",
    }
    return mapping.get(arch, "mi300x")


def _dtype_bytes(dtype: torch.dtype) -> float:
    """Get bytes per element for a dtype."""
    try:
        return torch.finfo(dtype).bits / 8
    except TypeError:
        return torch.iinfo(dtype).bits / 8


def _dtype_str(dtype: torch.dtype) -> str:
    """Convert torch dtype to short string."""
    mapping = {
        torch.float32: "f32",
        torch.float16: "f16",
        torch.bfloat16: "bf16",
        torch.int8: "i8",
    }
    if hasattr(torch, "float8_e4m3fnuz"):
        mapping[torch.float8_e4m3fnuz] = "f8"
    if hasattr(torch, "float8_e5m2fnuz"):
        mapping[torch.float8_e5m2fnuz] = "f8"
    if hasattr(torch, "float8_e4m3fn"):
        mapping[torch.float8_e4m3fn] = "f8"
    if hasattr(torch, "float8_e5m2"):
        mapping[torch.float8_e5m2] = "f8"
    return mapping.get(dtype, "f16")


def _compile_config(bm, bn, bk, num_stages, num_warps, a_dtype, b_dtype,
                    arch, num_cus, num_xcds, cache_dir):
    """Compile the persistent_gemm kernel for a given tile config.

    Uses Triton's warmup mechanism — no GPU memory allocation, no kernel launch.
    Returns the path to the cached .amdgcn assembly file.
    """
    dtype_s = _dtype_str(a_dtype)
    key = _cache_key(bm, bn, bk, num_stages, num_warps, dtype_s, arch)

    # Check in-process cache
    if key in _ASM_CACHE:
        path = _ASM_CACHE[key]
        if Path(path).exists():
            return path

    # Check disk cache
    arch_dir = cache_dir / arch
    asm_file = arch_dir / f"persistent_gemm_{bm}x{bn}x{bk}_s{num_stages}_w{num_warps}_{dtype_s}.amdgcn"
    if asm_file.exists():
        _ASM_CACHE[key] = str(asm_file)
        return str(asm_file)

    # Compile via Triton warmup
    try:
        from .kernels.persistent_gemm import persistent_matmul
        import triton

        # Use representative dimensions for stride computation
        M, N, K = 4096, 4096, 4096
        group_m = 8
        total_tiles = math.ceil(M / bm) * math.ceil(N / bn)
        chunk_size = min(group_m * group_m, max(1, total_tiles // max(num_xcds, 1)))
        even_k = K % bk == 0

        # Determine torch dtype for MockTensor
        torch_dtype = a_dtype

        # Build args in kernel parameter order:
        # A, B, C, A_scale_ptr, B_scale_ptr, bias_ptr,
        # M, N, K, stride_am, stride_bn, stride_cm, stride_cn,
        # stride_bias, stride_ak, stride_bk
        args = [
            triton.MockTensor(torch_dtype),  # A
            triton.MockTensor(torch_dtype),  # B
            triton.MockTensor(torch_dtype),  # C
            None,                             # A_scale_ptr
            None,                             # B_scale_ptr
            None,                             # bias_ptr
            M,                                # M
            N,                                # N
            K,                                # K
            K,                                # stride_am
            1,                                # stride_bn
            N,                                # stride_cm
            1,                                # stride_cn
            0,                                # stride_bias
            1,                                # stride_ak
            N,                                # stride_bk
        ]

        grid = (min(total_tiles, num_cus),)

        compiled = persistent_matmul.warmup(
            *args,
            grid=grid,
            BLOCK_SIZE_M=bm,
            BLOCK_SIZE_N=bn,
            BLOCK_SIZE_K=bk,
            GROUP_SIZE_M=group_m,
            NUM_SMS=min(total_tiles, num_cus),
            NUM_XCDS=num_xcds,
            CHUNK_SIZE=chunk_size,
            CACHE_MODIFIER_A=None,
            CACHE_MODIFIER_B=None,
            BIAS=False,
            EVEN_K=even_k,
            QUANTIZED=False,
            ALLOW_TF32=True,
            num_stages=num_stages,
            num_warps=num_warps,
        )

        if compiled is None:
            return None

        # Handle FutureKernel (async compilation)
        if hasattr(compiled, 'result'):
            compiled = compiled.result()

        # Extract assembly
        asm_text = None
        if hasattr(compiled, 'asm'):
            for asm_key in ('amdgcn', 'gcn', 'amdgpu'):
                if asm_key in compiled.asm:
                    asm_text = compiled.asm[asm_key]
                    if isinstance(asm_text, bytes):
                        asm_text = asm_text.decode('utf-8', errors='replace')
                    break

        if not asm_text:
            return None

        # Write to disk cache
        arch_dir.mkdir(parents=True, exist_ok=True)
        asm_file.write_text(asm_text)
        _ASM_CACHE[key] = str(asm_file)
        return str(asm_file)

    except Exception as e:
        # Compilation failed for this config — skip it
        return None


def _compile_via_launch(bm, bn, bk, num_stages, a_dtype, arch, cache_dir):
    """Fallback: compile by launching a real matmul on tiny tensors, then scan cache."""
    dtype_s = _dtype_str(a_dtype)
    key = _cache_key(bm, bn, bk, num_stages, 8, dtype_s, arch)

    if key in _ASM_CACHE:
        path = _ASM_CACHE[key]
        if Path(path).exists():
            return path

    try:
        import tritonblas

        # Set a fresh triton cache dir so we can find the .amdgcn easily
        old_cache = os.environ.get('TRITON_CACHE_DIR', '')
        tmp_cache = tempfile.mkdtemp(prefix='gcnsim_')
        os.environ['TRITON_CACHE_DIR'] = tmp_cache

        # Create tiny tensors (just enough for the tile config)
        M = max(bm, 16)
        N = max(bn, 16)
        K = max(bk, 16)
        a = torch.randn(M, K, device='cuda', dtype=a_dtype)
        b = torch.randn(K, N, device='cuda', dtype=a_dtype)

        # Force specific config by creating a mock selector
        tritonblas.matmul(a, b)

        # Restore cache dir
        if old_cache:
            os.environ['TRITON_CACHE_DIR'] = old_cache
        else:
            os.environ.pop('TRITON_CACHE_DIR', None)

        # Scan for .amdgcn file
        best_path = None
        best_mtime = 0
        for root, dirs, files in os.walk(tmp_cache):
            for f in files:
                if f.endswith('.amdgcn'):
                    full = os.path.join(root, f)
                    mt = os.path.getmtime(full)
                    if mt > best_mtime:
                        best_mtime = mt
                        best_path = full

        if best_path:
            # Copy to our cache
            arch_dir = cache_dir / arch
            arch_dir.mkdir(parents=True, exist_ok=True)
            asm_file = arch_dir / f"persistent_gemm_{bm}x{bn}x{bk}_s{num_stages}_{dtype_s}.amdgcn"
            import shutil
            shutil.copy2(best_path, asm_file)
            _ASM_CACHE[key] = str(asm_file)
            return str(asm_file)

    except Exception:
        pass

    return None


class GcnSimSelector:
    """ISA-based tile config selector using amdgcn_analyzer cycle simulation.

    Drop-in replacement for OrigamiMatmulSelector. Same interface:
    - block_m, block_n, block_k: selected tile dimensions
    - group_m: workgroup swizzle group size
    - num_sms: XCD mapping parameter
    - num_stages: pipeline stages
    - sk_grid: Stream-K grid size
    """

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
        mode: str = 'simulate',
    ):
        self._m = m
        self._n = n
        self._k = k
        self._num_stages = num_stages

        # Hardware detection
        self._arch = _get_arch_from_device(device)
        self._num_cus = total_cus or _get_num_cus(device)
        self._num_xcds = _get_num_xcds(self._num_cus, self._arch)
        self._lds_cap = _get_lds_capacity(self._arch)
        self._hw_str = _hw_string(self._arch)
        self._mode = mode

        # Dtype info
        self._a_dtype = a_dtype
        self._b_dtype = b_dtype
        self._bytes_a = _dtype_bytes(a_dtype)
        self._bytes_b = _dtype_bytes(b_dtype)

        # Build hardware-like object for matmul.py compatibility
        self._hardware = _HardwareCompat(
            n_cu=self._num_cus,
            num_xcd=self._num_xcds,
            lds_capacity=self._lds_cap,
        )

        # Config space
        block_mn_range = [16, 32, 64, 128, 256]
        block_k_range = [16, 32, 64, 128, 256, 512]
        num_warps = 8

        # Generate and filter candidates
        candidates = []
        for bm in block_mn_range:
            for bn in block_mn_range:
                for bk in block_k_range:
                    if not check_triton_lds_capacity(
                        bm, bn, bk, self._bytes_a, self._bytes_b,
                        self._lds_cap, num_stages
                    ):
                        continue
                    # Skip configs that produce degenerate grids
                    tiles_m = math.ceil(m / bm)
                    tiles_n = math.ceil(n / bn)
                    if tiles_m == 0 or tiles_n == 0:
                        continue
                    candidates.append((bm, bn, bk))

        if not candidates:
            # Fallback: use 128x128x64
            candidates = [(128, 128, 64)]

        # Rank candidates by predicted latency
        self._block_m, self._block_n, self._block_k = self._select_best(
            candidates, m, n, k, num_warps
        )

        # Workgroup mapping: default heuristic
        tiles_m = math.ceil(m / self._block_m)
        self._group_m = min(8, tiles_m)

        # COUNTERS_PER_XCD for work-stealing
        total_tiles = tiles_m * math.ceil(n / self._block_n)
        if total_tiles <= 512:
            self.COUNTERS_PER_XCD = 8
        elif total_tiles <= 1536:
            self.COUNTERS_PER_XCD = 4
        elif total_tiles <= 2048:
            self.COUNTERS_PER_XCD = 2
        else:
            self.COUNTERS_PER_XCD = 1

    def _select_best(self, candidates, m, n, k, num_warps):
        """Rank candidates by amdgcn_analyzer predicted latency."""
        try:
            from amdgcn_analyzer import predict
        except ImportError:
            # amdgcn_analyzer not installed — use simple heuristic
            return self._heuristic_select(candidates, m, n, k)

        cache_dir = _CACHE_DIR
        best_config = None
        best_latency = float('inf')
        results = []

        for bm, bn, bk in candidates:
            # Get or compile assembly
            asm_path = _compile_config(
                bm, bn, bk, self._num_stages, num_warps,
                self._a_dtype, self._b_dtype, self._arch,
                self._num_cus, self._num_xcds, cache_dir,
            )
            if asm_path is None:
                continue

            # Compute num_workgroups for this config + shape
            tiles_m = math.ceil(m / bm)
            tiles_n = math.ceil(n / bn)
            num_wgs = tiles_m * tiles_n

            try:
                result = predict(
                    asm_path,
                    mode=self._mode,
                    hw=self._hw_str,
                    num_workgroups=num_wgs,
                    runtime_args={'K': k},
                )
                latency = result.latency_us
                results.append((bm, bn, bk, latency))

                if latency < best_latency:
                    best_latency = latency
                    best_config = (bm, bn, bk)

            except Exception:
                continue

        if best_config is None:
            return self._heuristic_select(candidates, m, n, k)

        if os.environ.get('TBLAS_GCN_DEBUG', ''):
            print(f"\n[GcnSimSelector] M={m} N={n} K={k} arch={self._arch}")
            print(f"  Evaluated {len(results)} configs:")
            for bm, bn, bk, lat in sorted(results, key=lambda x: x[3])[:10]:
                marker = " <-- BEST" if (bm, bn, bk) == best_config else ""
                print(f"    {bm:3d}x{bn:3d}x{bk:3d}  {lat:8.1f} us{marker}")

        return best_config

    def _heuristic_select(self, candidates, m, n, k):
        """Simple heuristic fallback when predict() unavailable."""
        # Prefer configs that:
        # 1. Have tile area close to 16384 (128x128)
        # 2. Don't have too many waves (total_tiles not >> num_cus)
        best = None
        best_score = float('inf')
        target_area = 128 * 128

        for bm, bn, bk in candidates:
            area = bm * bn
            area_diff = abs(area - target_area) / target_area
            tiles = math.ceil(m / bm) * math.ceil(n / bn)
            wave_eff = (tiles % self._num_cus) / max(self._num_cus, 1) if tiles > 0 else 1
            score = area_diff + wave_eff * 0.5
            if score < best_score:
                best_score = score
                best = (bm, bn, bk)

        return best or (128, 128, 64)

    def hierarchical_split(self, num_xcds: int) -> tuple:
        """Compute local/global tile split for hierarchical work-stealing."""
        bm, bn = self._block_m, self._block_n
        total_tiles = math.ceil(self._m / bm) * math.ceil(self._n / bn)
        tiles_per_cu = total_tiles / max(self._num_cus, 1)
        local_frac = max(0.5, 1.0 - max(0.0, tiles_per_cu - 4.0) * 0.05)
        local_per_xcd = int(total_tiles * local_frac) // num_xcds
        local_per_xcd = max(local_per_xcd, 1)
        global_tiles = total_tiles - local_per_xcd * num_xcds
        return local_per_xcd, global_tiles

    @property
    def block_m(self):
        return self._block_m

    @property
    def block_n(self):
        return self._block_n

    @property
    def block_k(self):
        return self._block_k

    @property
    def group_m(self):
        return self._group_m

    @property
    def num_sms(self):
        return self._num_xcds

    @property
    def num_stages(self):
        return self._num_stages

    @property
    def waves_per_eu(self):
        return 0

    @property
    def even_k(self):
        return self._k % self._block_k == 0

    @property
    def sk_grid(self):
        return self._num_cus


class _HardwareCompat:
    """Minimal hardware-like object for matmul.py compatibility."""

    def __init__(self, n_cu, num_xcd, lds_capacity):
        self.N_CU = n_cu
        self.NUM_XCD = num_xcd
        self.lds_capacity = lds_capacity
        self.CU_per_L2 = n_cu // max(num_xcd, 1)
