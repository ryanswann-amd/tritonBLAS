"""Auto-dispatch routing for the eager tritonblas matmul path.

The persistent (data-parallel) GEMM kernel launches one workgroup per output
tile.  For "skinny" shapes whose tile count is much smaller than the device
CU count -- e.g. M=16, N=1024 with BM=16, BN=128 yields 8 tiles, leaving 296
of the 304 MI300X CUs idle for the entire kernel duration -- this
underutilises the device.

Stream-K splits the K dimension across the otherwise-idle CUs and atomically
reduces the per-CU partial sums into the output tile, which restores
occupancy.  This module owns the single decision: "does the persistent
dispatch leave too many CUs idle?".

Heuristic (matches the K-162 task spec verbatim):

    total_tiles = ceil(M/BLK_M) * ceil(N/BLK_N)
    if total_tiles * 2 < num_cus: route to Stream-K

Half the CU count is the cutoff because the persistent kernel dispatches
exactly ``total_tiles`` workgroups; below ``num_cus / 2`` more than half of
the CUs sit idle for the kernel's full duration, which is the regime where
Stream-K's K-split parallelism pays for the atomic-reduction overhead.

Set ``TBLAS_DISABLE_AUTO_STREAMK=1`` to bypass this routing and respect the
caller-supplied ``enable_streamk`` value verbatim (used by tests, sweeps,
and bisection).
"""
import math
import os
from typing import Dict, Tuple

import torch


# How many CUs each tile must "occupy" before the persistent dispatch is
# considered full enough.  At 2 the gate fires when more than half the
# device would be idle.  Single tunable -- raise (e.g. to 3) to be more
# aggressive about routing into Stream-K, lower (to 1) to be more
# conservative.  Held as a module constant rather than an env knob because
# it is baked into the spec and we have no plan to tune it at runtime.
_TILE_FILL_FACTOR = 2

# Cache CU count per (device_type, device_index); torch.cuda.get_device_properties
# is not free and dispatch is on the hot path.
_NUM_CU_CACHE: Dict[Tuple[str, int], int] = {}


def _num_cus(device: torch.device) -> int:
    if device.type != "cuda":
        return 1
    key = (device.type, device.index if device.index is not None else 0)
    cached = _NUM_CU_CACHE.get(key)
    if cached is not None:
        return cached
    props = torch.cuda.get_device_properties(device)
    cu = int(props.multi_processor_count)
    _NUM_CU_CACHE[key] = cu
    return cu


def _auto_streamk_disabled() -> bool:
    return os.environ.get("TBLAS_DISABLE_AUTO_STREAMK", "").lower() in (
        "1",
        "true",
        "yes",
    )


def should_use_streamk(
    M: int,
    N: int,
    BLK_M: int,
    BLK_N: int,
    device: torch.device,
) -> bool:
    """Return True when the persistent-tile dispatch would underutilise the GPU.

    Pure function of ``(M, N, BLK_M, BLK_N, device)``: no Origami selector
    rebuild, no extra kernel launch, no atomics.  The caller is expected to
    have already obtained ``BLK_M``/``BLK_N`` from its data-parallel
    selector (which it needs anyway for the persistent path), so the gate
    itself is one cdiv-product plus one compare on the cold path -- and the
    CU-count lookup is cached.
    """
    if _auto_streamk_disabled():
        return False
    if BLK_M <= 0 or BLK_N <= 0:
        return False
    num_cus = _num_cus(device)
    total_tiles = math.ceil(M / BLK_M) * math.ceil(N / BLK_N)
    return total_tiles * _TILE_FILL_FACTOR < num_cus
