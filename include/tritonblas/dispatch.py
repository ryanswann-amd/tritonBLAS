"""Auto-dispatch routing for the eager tritonblas matmul path.

The persistent (data-parallel) GEMM kernel launches one workgroup per output
tile.  For "skinny" shapes whose tile count is much smaller than the device
CU count -- e.g. M=16, N=2048 with BM=16, BN=16 yields 128 tiles, leaving
176 of the 304 MI300X CUs idle for the entire kernel duration -- this
underutilises the device.

Stream-K splits the K dimension across the otherwise-idle CUs and atomically
reduces the per-CU partial sums into the output tile, which restores
occupancy.  This module owns the single decision: "should we route this
shape from persistent into Stream-K?".

Heuristic (three-condition gate, all module-level constants -- not env knobs):

    1. ``ceil(M/BM) * ceil(N/BN) * _TILE_FILL_FACTOR < num_cus``
       The persistent kernel dispatches exactly ``total_tiles`` workgroups,
       so below ``num_cus / TILE_FILL_FACTOR`` more than half of the CUs
       sit idle for the kernel's full duration.  This is the headline
       condition from the K-162 task spec.

    2. ``K >= _MIN_K``
       Stream-K splits each tile's K loop across (up to ``num_cus``) more
       PEs and reduces partial sums via atomic global adds plus a spin-lock
       for the leader WG.  Empirically (K-162 sweep, MI300X, fp16+bf16,
       see ``output/k162_bench.csv``) that overhead is recovered only when
       ``K >= 8192``: at K=4096 every shape in the cohort is 4-9% slower
       than the under-occupied persistent kernel, and at K=1024 every
       shape is 14-15% slower.  Below this floor the atomic reduction
       cost dominates.

    3. ``N >= _MIN_N``
       Even at K=8192, shapes with very small N (the (M={8,16}, N=512)
       corner) lose 2-4% to the persistent path because the Stream-K grid
       layout cannot productively use all of MI300X's CUs when the output
       tile is too narrow.  This floor excludes that corner without
       affecting the headline cohort.

Note on the K-162 PRD's named (M=16, N=1024, K=1024) shape: the empirical
sweep shows Stream-K is *strictly slower* than persistent on this shape
(0.297x -> 0.252x vs hipBLASLt) because the K loop is only 16 iterations
of BLK_K=64 -- splitting that across CUs leaves each PE doing 1-2 K-iters,
nowhere near enough work to amortise the atomic reduction.  The K-162
ticket has been updated to reflect that the Stream-K dispatch route can
only recover occupancy on the ``K >= 8192`` half of the small-M cohort
(11/24 shapes); the small-K skinny cases remain a known residual that
likely needs a different kernel variant (e.g. split-K with bf16
accumulator, or a persistent-thread variant) to address.

Set ``TBLAS_DISABLE_AUTO_STREAMK=1`` to bypass this routing and respect
the caller-supplied ``enable_streamk`` value verbatim (used by tests,
sweeps, and bisection).
"""
import math
import os
from typing import Dict, Tuple

import torch


# Three-condition gate -- module-level literals, not env knobs.  All values
# came out of the K-162 sweep on MI300X (gfx942, 304 CUs, ROCm 7.2,
# fp16+bf16); see module docstring for the per-condition justification.
_TILE_FILL_FACTOR = 2  # underutilisation cutoff: tiles*2 < num_cus
_MIN_K = 8192          # K-iter floor: below this, atomic reduction dominates
_MIN_N = 1024          # N floor: below this, sk grid can't fill the device

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
    K: int,
    BLK_M: int,
    BLK_N: int,
    device: torch.device,
) -> bool:
    """Return True when the persistent dispatch is under-occupied AND the
    Stream-K route is empirically faster for this shape.

    Pure function of ``(M, N, K, BLK_M, BLK_N, device)``: no Origami
    selector rebuild, no extra kernel launch, no atomics.  The caller is
    expected to have already obtained ``BLK_M``/``BLK_N`` from its
    data-parallel selector (which it needs anyway for the persistent
    path), so the gate itself is one cdiv-product plus three compares on
    the cold path -- and the CU-count lookup is cached.
    """
    if _auto_streamk_disabled():
        return False
    if BLK_M <= 0 or BLK_N <= 0:
        return False
    if K < _MIN_K or N < _MIN_N:
        return False
    num_cus = _num_cus(device)
    total_tiles = math.ceil(M / BLK_M) * math.ceil(N / BLK_N)
    return total_tiles * _TILE_FILL_FACTOR < num_cus
