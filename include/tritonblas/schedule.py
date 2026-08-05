# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
The order a matmul will visit its output tiles in, exposed to the caller.

A GEMM's tile order is chosen for L2 reuse and then kept inside the kernel. Anything that has to
act on it — a triggered collective arming descriptors, a fused-op scheduler, a cost model — must
re-derive it, and nothing enforces that the re-derivations agree. They do not fail loudly when
they disagree: the work still completes and the result is merely slow.

``schedule()`` publishes it instead. It builds the selector by the same path ``matmul()`` does and
mirrors :class:`~tritonblas.kernels.stages.schedule.ScheduleContext`, so what it reports is the
order the kernel will actually run rather than a reconstruction of it.

    >>> sched = tritonblas.schedule(a, b)
    >>> sched.tile_of_wg[0]            # linear tile id computed by program 0
    >>> sched.first_ready_fraction(num_cus=304)

Returns ``None`` when there is no static order to publish, which is a normal answer:

``work_stealing``
    tiles are claimed from counters at runtime; the assignment differs per launch.
``enable_streamk``
    a tile is not owned by one workgroup, so there is no permutation over tiles at all.

Callers must handle ``None`` rather than substitute a guess. For a triggered collective the
correct response is to not attempt gated overlap.
"""

from dataclasses import dataclass, field
from typing import List, Optional

__all__ = ["TileSchedule", "schedule", "schedule_for_problem"]


def _chiplet_transform_chunked(pid, num_workgroups, num_xcds, chunk_size):
    """Host mirror of ``kernels.stages.indexing.pid_transforms.chiplet_transform_chunked``."""
    if num_xcds <= 1:
        return pid
    if pid > (num_workgroups // (num_xcds * chunk_size)) * (num_xcds * chunk_size):
        return pid
    local_pid = pid // num_xcds
    chunk_idx = local_pid // chunk_size
    pos_in_chunk = local_pid % chunk_size
    xcd = pid % num_xcds
    return chunk_idx * num_xcds * chunk_size + xcd * chunk_size + pos_in_chunk


def _group_m_swizzle(tile_id, num_pid_m, num_pid_n, group_size_m):
    """Host mirror of ``ScheduleContext.get_tile_from_idx``."""
    num_pid_in_group = group_size_m * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * group_size_m
    span = min(num_pid_m - first_pid_m, group_size_m)
    pid_m = first_pid_m + ((tile_id % num_pid_in_group) % span)
    pid_n = (tile_id % num_pid_in_group) // span
    return pid_m, pid_n


@dataclass(frozen=True)
class TileSchedule:
    """The tile order of one matmul, plus the parameters that determine it."""

    tiles_m: int
    tiles_n: int
    block_m: int
    block_n: int
    block_k: int
    group_m: int
    num_xcds: int
    chunk_size: int
    #: Dispatch position -> linear tile id ``pid_m * tiles_n + pid_n``.
    tile_of_wg: List[int] = field(repr=False)

    @property
    def num_tiles(self):
        return self.tiles_m * self.tiles_n

    def wg_of_tile(self):
        """The inverse: linear tile id -> dispatch position."""
        out = [0] * self.num_tiles
        for position, tile in enumerate(self.tile_of_wg):
            out[tile] = position
        return out

    def coords(self):
        """Dispatch position -> ``(pid_m, pid_n)``."""
        return [(t // self.tiles_n, t % self.tiles_n) for t in self.tile_of_wg]

    def finish_wave(self, num_cus, position):
        """
        Which wave the workgroup at ``position`` retires in, given ``num_cus`` compute units.

        The hardware places launch index ``k`` on XCD ``k % num_xcds`` and then on the
        earliest-free CU of that die, so under uniform tile cost the ``j``-th workgroup on a die
        runs in wave ``j // cus_per_xcd``. Dispatch position alone is only a proxy for time when
        tiles greatly outnumber CUs; a grid that fits in one wave has no temporal spread at all.
        """
        cus_per_xcd = max(1, num_cus // self.num_xcds)
        return (position // self.num_xcds) // cus_per_xcd + 1

    def first_ready_fraction(self, num_cus):
        """
        How far into the GEMM the earliest output tile is finished, in ``[0, 1]``.

        ``1 - this`` is the overlap window: an upper bound on how much of a dependent collective
        could be hidden behind this GEMM. **The number to check before building gated overlap at
        all** — a grid that fits in one wave returns 1.0, meaning nothing can be hidden however
        the collective is scheduled.

        ``num_cus`` is required rather than optional. Measured in dispatch positions the answer
        is always 0 (some tile is always first), which says nothing; the question is temporal
        and needs to know how many tiles run at once.
        """
        n = self.num_tiles
        if not n:
            return 0.0
        waves = [self.finish_wave(num_cus, p) for p in range(n)]
        span = max(waves)
        return min(waves) / span if span else 0.0

    def ready_wave_of_tile(self, num_cus):
        """Linear tile id -> the wave it is finished in. The input a signal layout needs."""
        out = [0] * self.num_tiles
        for position, tile in enumerate(self.tile_of_wg):
            out[tile] = self.finish_wave(num_cus, position)
        return out


def schedule_for_problem(M, N, K, a_dtype, b_dtype, out_dtype, device, *,
                         enable_streamk=False, work_stealing=False, num_stages=2):
    """``schedule()`` for a problem shape rather than tensors. See that function."""
    if work_stealing or enable_streamk:
        return None

    from .origami import OrigamiMatmulSelector

    selector = OrigamiMatmulSelector(
        M, N, K, a_dtype, b_dtype, out_dtype, device,
        streamk=enable_streamk, num_stages=num_stages,
    )
    block_m, block_n = selector.block_m, selector.block_n
    tiles_m = (M + block_m - 1) // block_m
    tiles_n = (N + block_n - 1) // block_n
    total = tiles_m * tiles_n

    # Exactly `persistent_matmul_lt`'s derivation: `group_m` is the effective mapping (Origami's
    # selection is overwritten by `_select_ws_params`), `num_sms` carries wgmxcc, and the chunk
    # is capped to one XCD's share of the grid.
    group_m = selector.group_m
    num_xcds = selector.num_sms
    chunk_size = group_m * group_m
    if num_xcds and num_xcds > 0:
        chunk_size = min(chunk_size, max(1, total // num_xcds))
    else:
        num_xcds = 1

    tile_of_wg = []
    for pid in range(total):
        tile_id = _chiplet_transform_chunked(pid, total, num_xcds, chunk_size)
        pid_m, pid_n = _group_m_swizzle(tile_id, tiles_m, tiles_n, group_m)
        tile_of_wg.append(pid_m * tiles_n + pid_n)

    return TileSchedule(
        tiles_m=tiles_m, tiles_n=tiles_n,
        block_m=block_m, block_n=block_n, block_k=selector.block_k,
        group_m=group_m, num_xcds=num_xcds, chunk_size=chunk_size,
        tile_of_wg=tile_of_wg,
    )


def schedule(a, b, *, enable_streamk=False, work_stealing=False) -> Optional[TileSchedule]:
    """
    The tile order ``matmul(a, b, ...)`` will run in, or ``None`` if there is not a static one.

    Pass the same flags you would pass to :func:`tritonblas.matmul`; the schedule depends on
    them, and on the config Origami selects for this problem.
    """
    M, K = a.shape
    _, N = b.shape
    return schedule_for_problem(
        M, N, K, a.dtype, b.dtype, a.dtype, a.device,
        enable_streamk=enable_streamk, work_stealing=work_stealing,
    )
