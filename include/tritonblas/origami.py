"""tritonBLAS origami integration — thin wrappers over origami's selection APIs.

All selection logic (LDS filtering, config generation, scoring, MI inference,
StreamK grid computation, work-stealing params) now lives in origami's
``TritonOrigamiMatmulSelector`` (``origami.selector`` module).

This module provides backward-compatible wrappers so that existing tritonBLAS
call sites (``matmul.py``, ``config.py``, benchmarks, tests) continue to work
without modification.

DEPRECATED: Direct use of ``OrigamiMatmulSelector`` from this module is
deprecated.  New code should use ``origami.selector.TritonOrigamiMatmulSelector``
directly.  This module will be removed in a future release.
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Import selection APIs from origami (the single source of truth)
# ---------------------------------------------------------------------------
from origami.selector import (
    estimate_triton_lds_bytes,
    check_triton_lds_capacity,
    TritonOrigamiMatmulSelector as _TritonSelector,
)


# ---------------------------------------------------------------------------
# Backward-compatible aliases — existing test and tool imports still work
# ---------------------------------------------------------------------------
# ``from tritonblas.origami import estimate_triton_lds_bytes`` — works.
# ``from tritonblas.origami import check_triton_lds_capacity`` — works.
# Both are re-exported directly from origami.selector above.


class OrigamiMatmulSelector(_TritonSelector):
    """Backward-compatible thin wrapper around origami's TritonOrigamiMatmulSelector.

    .. deprecated::
        Use ``origami.selector.TritonOrigamiMatmulSelector`` directly.
        ``OrigamiMatmulSelector`` now contains zero internal selection logic;
        all work is delegated to the origami C++ analytical model via
        ``TritonOrigamiMatmulSelector``.

    The constructor signature and all public properties (``block_m``,
    ``block_n``, ``block_k``, ``group_m``, ``num_sms``, ``num_stages``,
    ``waves_per_eu``, ``even_k``, ``sk_grid``, ``COUNTERS_PER_XCD``,
    ``hierarchical_split()``) are preserved identically.

    Internal attributes accessed by ``matmul.py`` and ``config.py``
    (``_hardware``, ``_N_CU``, ``_ACTIVE_CU``) are also preserved.
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
    ):
        # Delegate entirely to origami's TritonOrigamiMatmulSelector.
        # All LDS filtering, config generation, MI inference, StreamK grid,
        # work-stealing params, 256x256x64 heuristic, etc. are handled there.
        super().__init__(
            m=m,
            n=n,
            k=k,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            out_dtype=out_dtype,
            device=device,
            mx_block_size=mx_block_size,
            streamk=streamk,
            total_cus=total_cus,
            active_cus=active_cus,
            num_stages=num_stages,
        )

    # Class-level wrapper preserved for backward compatibility
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


# Legacy alias used by some tools (triton_gemm_bench.py, export_config.py, etc.)
MatmulHeuristicResult = OrigamiMatmulSelector
