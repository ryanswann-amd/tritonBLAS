"""steal-k: a work-stealing split-K GEMM kernel, with in-kernel event tracing.

Design (deadlock-free, two phase):
  Phase 1 (steal): every workgroup repeatedly grabs the next (tile, k_group)
    work item via a single global atomic counter (`work_ctr`), computes that
    tile's partial sum over its K sub-range, writes it to `Ppartial[wid]`, and
    sets `done[wid]=1`.  Work items are stolen dynamically, so a fast WG does
    more items than a slow one.
  Phase 2 (reduce): each WG statically owns tiles `tile % GRID == wg`.  For each
    owned tile it waits for all SPLITK peers' `done` flags, sums the SPLITK
    partials, and writes C.  Deadlock-free because phase 1 fully drains the work
    counter (every item is owned by some WG that sets `done` before its next
    steal) before any reduction can block.

Tracing: each WG appends events to `events[wg, slot, :]` (int64):
  [wg, tile, k_group, phase, t_start, t_end, k_start_it, k_end_it]
  phase: 0=compute, 1=reduce (sk-reduction), 2=steal (atomic queue-index pull)
Timers use s_memrealtime (100 MHz).  The `steal` timer brackets ONLY the
`tl.atomic_add(work_ctr, ...)` — reliable because the atomic is a side-effecting
op whose result is used (the two clock reads can't be reordered/merged across
it), unlike empty-region sub-timers.  So we separately measure the atomic
queue-index pull vs the sk-reduction vs the compute.
"""
import triton
import triton.language as tl

from trace_helpers import read_realtime

EVENT_FIELDS = 8
# field offsets
E_WG, E_TILE, E_KG, E_PHASE, E_T0, E_T1, E_KS, E_KE = range(8)


@triton.jit()
def steal_k_matmul(
    A, B, C,
    Ppartial, done, work_ctr, events,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLITK: tl.constexpr, NUM_TILES: tl.constexpr,
    ITERS_PER_TILE: tl.constexpr, ITERS_PER_GROUP: tl.constexpr,
    GRID: tl.constexpr, MAX_SLOTS: tl.constexpr, EVENT_FIELDS: tl.constexpr,
    TRACE: tl.constexpr = True,
):
    wg = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    total_items = NUM_TILES * SPLITK
    rk = tl.arange(0, BLOCK_K)
    slot = 0

    # ---------------- Phase 1: work-stealing split-K compute ----------------
    # steal (atomic queue-index pull) — timed separately from compute/reduction.
    # Timer reads are compiled out when TRACE=False (fair perf comparison).
    if TRACE:
        s0 = read_realtime()
    wid = tl.atomic_add(work_ctr, 1, scope="gpu")
    if TRACE:
        s1 = read_realtime()
        if slot < MAX_SLOTS:
            e = events + (wg * MAX_SLOTS + slot) * EVENT_FIELDS
            tl.store(e + 0, wg.to(tl.int64))
            tl.store(e + 1, tl.cast(-1, tl.int64))
            tl.store(e + 2, tl.cast(-1, tl.int64))
            tl.store(e + 3, tl.cast(2, tl.int64))
            tl.store(e + 4, s0)
            tl.store(e + 5, s1)
            tl.store(e + 6, tl.cast(0, tl.int64))
            tl.store(e + 7, tl.cast(0, tl.int64))
            slot += 1
    while wid < total_items:
        tile = wid // SPLITK
        kg = wid % SPLITK
        pid_m = tile // num_pid_n
        pid_n = tile % num_pid_n
        k_start = kg * ITERS_PER_GROUP
        k_end = min((kg + 1) * ITERS_PER_GROUP, ITERS_PER_TILE)

        t0 = read_realtime()
        rm_raw = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn_raw = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = rm_raw < M
        mask_n = rn_raw < N
        rm = rm_raw % M
        rn = rn_raw % N
        A_BASE = A + rm[:, None] * stride_am + (k_start * BLOCK_K + rk)[None, :] * stride_ak
        B_BASE = B + (k_start * BLOCK_K + rk)[:, None] * stride_bk + rn[None, :] * stride_bn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for it in range(k_start, k_end):
            kmask = (it * BLOCK_K + rk) < K
            a = tl.load(A_BASE, mask=mask_m[:, None] & kmask[None, :], other=0.0)
            b = tl.load(B_BASE, mask=kmask[:, None] & mask_n[None, :], other=0.0)
            acc += tl.dot(a, b, allow_tf32=False)
            A_BASE += BLOCK_K * stride_ak
            B_BASE += BLOCK_K * stride_bk

        P_ = Ppartial + wid * BLOCK_M * BLOCK_N + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        tl.store(P_, acc, cache_modifier=".wt")
        tl.debug_barrier()
        tl.store(done + wid, 1, cache_modifier=".wt")
        t1 = read_realtime()

        if TRACE:
            if slot < MAX_SLOTS:
                e = events + (wg * MAX_SLOTS + slot) * EVENT_FIELDS
                tl.store(e + 0, wg.to(tl.int64))
                tl.store(e + 1, tile.to(tl.int64))
                tl.store(e + 2, kg.to(tl.int64))
                tl.store(e + 3, tl.cast(0, tl.int64))
                tl.store(e + 4, t0)
                tl.store(e + 5, t1)
                tl.store(e + 6, k_start.to(tl.int64))
                tl.store(e + 7, k_end.to(tl.int64))
                slot += 1

        # next steal (atomic queue-index pull) — timed separately
        if TRACE:
            s0 = read_realtime()
        wid = tl.atomic_add(work_ctr, 1, scope="gpu")
        if TRACE:
            s1 = read_realtime()
            if slot < MAX_SLOTS:
                e = events + (wg * MAX_SLOTS + slot) * EVENT_FIELDS
                tl.store(e + 0, wg.to(tl.int64))
                tl.store(e + 1, tl.cast(-1, tl.int64))
                tl.store(e + 2, tl.cast(-1, tl.int64))
                tl.store(e + 3, tl.cast(2, tl.int64))
                tl.store(e + 4, s0)
                tl.store(e + 5, s1)
                tl.store(e + 6, tl.cast(0, tl.int64))
                tl.store(e + 7, tl.cast(0, tl.int64))
                slot += 1

    # ---------------- Phase 2: static reduction of owned tiles ----------------
    for tile in range(wg, NUM_TILES, GRID):
        tr0 = read_realtime()
        pid_m = tile // num_pid_n
        pid_n = tile % num_pid_n
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for kg in range(0, SPLITK):
            wid = tile * SPLITK + kg
            while tl.load(done + wid, cache_modifier=".cv", volatile=True) != 1:
                pass
            P_ = Ppartial + wid * BLOCK_M * BLOCK_N + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
            acc += tl.load(P_, cache_modifier=".cv")

        rm_raw = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn_raw = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (rm_raw < M)[:, None] & (rn_raw < N)[None, :]
        C_ = C + rm_raw[:, None] * stride_cm + rn_raw[None, :] * stride_cn
        tl.store(C_, acc.to(C.type.element_ty), mask=mask)
        tr1 = read_realtime()

        if TRACE:
            if slot < MAX_SLOTS:
                e = events + (wg * MAX_SLOTS + slot) * EVENT_FIELDS
                tl.store(e + 0, wg.to(tl.int64))
                tl.store(e + 1, tile.to(tl.int64))
                tl.store(e + 2, tl.cast(-1, tl.int64))
                tl.store(e + 3, tl.cast(1, tl.int64))
                tl.store(e + 4, tr0)
                tl.store(e + 5, tr1)
                tl.store(e + 6, tl.cast(0, tl.int64))
                tl.store(e + 7, tl.cast(ITERS_PER_TILE, tl.int64))
                slot += 1


@triton.jit()
def steal_k_adaptive(
    A, B, C, P, locks,
    work_ctr, events,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_TILES: tl.constexpr, ITERS_PER_TILE: tl.constexpr, NUM_SMS: tl.constexpr,
    MAX_SLOTS: tl.constexpr, EVENT_FIELDS: tl.constexpr,
    STREAMK: tl.constexpr = False, TRACE: tl.constexpr = True, PER_ITER: tl.constexpr = False,
    EVEN_K: tl.constexpr = True, GROUP_M: tl.constexpr = 8,
    TREE: tl.constexpr = False, TREE_ROUNDS: tl.constexpr = 10,
):
    """Adaptive steal-k.

    STREAMK == False (num_tiles >= NUM_SMS: enough parallelism): work-stealing
    whole tiles.  Each WG grabs the next tile from `work_ctr`, computes the full
    K in registers and writes C directly — no global traffic, no reduction.

    STREAMK == True (underfilled: fewer tiles than CUs): **persistent Stream-K**,
    grid == NUM_SMS (== CU count, so all WGs are resident: occupancy 1/CU, which
    makes the owner/contributor fixup deadlock-free by construction). The flat
    tile×k iteration space (NUM_TILES * ITERS_PER_TILE) is split contiguously
    across the NUM_SMS persistent WGs; each pid walks its [start_iter, last_iter)
    in per-tile chunks:
      - a chunk that does NOT start on a tile boundary is a *contributor*: it
        writes its partial to the per-pid slot `P[pid]` (`.wt`) and sets
        `locks[pid]` (`.wt`).  Each pid contributes at most one such partial.
      - a chunk that starts on a tile boundary is the tile *owner*: it folds the
        partials of the following pids (`.cv` loads, gated by their `locks`) into
        its own registers and writes C once.
    Consumers only ever wait on *higher* pids, which are co-resident and running —
    no global fp atomic-add (slow on MI300/MI350); coherence via `.wt`/`.cv`.
    """
    wg = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    slot = 0

    if not STREAMK:
        # ================= work-stealing whole tiles (register reduce) =========
        wid = tl.atomic_add(work_ctr, 1, scope="gpu")
        while wid < NUM_TILES:
            tile = wid
            pid_m = tile // num_pid_n
            pid_n = tile % num_pid_n
            # Timers gated on TRACE: read_realtime() is s_memrealtime inline asm
            # that the compiler can't DCE, so leaving it in the perf path adds
            # live regs + scheduling barriers that spill/fault a 256x256 acc.
            tp0 = read_realtime() if TRACE else tl.cast(0, tl.int64)
            rm_raw = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            rn_raw = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_m = rm_raw < M
            mask_n = rn_raw < N
            rm = rm_raw % M
            rn = rn_raw % N
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            tc0 = read_realtime() if TRACE else tl.cast(0, tl.int64)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for it in range(0, ITERS_PER_TILE):
                if TRACE and PER_ITER:
                    ti0 = read_realtime()
                # EVEN_K: load UNMASKED (rm%M/rn%N wrapping keeps addresses valid;
                # garbage rows/cols are masked at the store). A masked load inside
                # a software-pipelined (num_stages>=2) loop miscompiles on this
                # Triton 3.6/gfx942, so the unmasked path is what lets us pipeline.
                if EVEN_K:
                    a = tl.load(A_BASE)
                    b = tl.load(B_BASE)
                else:
                    kmask = (it * BLOCK_K + rk) < K
                    a = tl.load(A_BASE, mask=mask_m[:, None] & kmask[None, :], other=0.0)
                    b = tl.load(B_BASE, mask=kmask[:, None] & mask_n[None, :], other=0.0)
                acc += tl.dot(a, b, allow_tf32=False)
                A_BASE += BLOCK_K * stride_ak
                B_BASE += BLOCK_K * stride_bk
                if TRACE and PER_ITER:
                    if slot < MAX_SLOTS:
                        e = events + (wg * MAX_SLOTS + slot) * EVENT_FIELDS
                        tl.store(e + 0, wg.to(tl.int64)); tl.store(e + 1, tile.to(tl.int64))
                        tl.store(e + 2, tl.cast(0, tl.int64)); tl.store(e + 3, tl.cast(11, tl.int64))
                        tl.store(e + 4, ti0); tl.store(e + 5, read_realtime())
                        tl.store(e + 6, it.to(tl.int64)); tl.store(e + 7, (it + 1).to(tl.int64))
                        slot += 1
            tsr0 = read_realtime() if TRACE else tl.cast(0, tl.int64)
            mask = mask_m[:, None] & mask_n[None, :]
            C_ = C + rm_raw[:, None] * stride_cm + rn_raw[None, :] * stride_cn
            tl.store(C_, acc.to(C.type.element_ty), mask=mask)
            tend = read_realtime() if TRACE else tl.cast(0, tl.int64)
            if TRACE:
                if slot < MAX_SLOTS:   # prologue (10)
                    e = events + (wg * MAX_SLOTS + slot) * EVENT_FIELDS
                    tl.store(e + 0, wg.to(tl.int64)); tl.store(e + 1, tile.to(tl.int64))
                    tl.store(e + 2, tl.cast(0, tl.int64)); tl.store(e + 3, tl.cast(10, tl.int64))
                    tl.store(e + 4, tp0); tl.store(e + 5, tc0)
                    tl.store(e + 6, tl.cast(0, tl.int64)); tl.store(e + 7, tl.cast(ITERS_PER_TILE, tl.int64))
                    slot += 1
                if not PER_ITER:       # compute (11)
                    if slot < MAX_SLOTS:
                        e = events + (wg * MAX_SLOTS + slot) * EVENT_FIELDS
                        tl.store(e + 0, wg.to(tl.int64)); tl.store(e + 1, tile.to(tl.int64))
                        tl.store(e + 2, tl.cast(0, tl.int64)); tl.store(e + 3, tl.cast(11, tl.int64))
                        tl.store(e + 4, tc0); tl.store(e + 5, tsr0)
                        tl.store(e + 6, tl.cast(0, tl.int64)); tl.store(e + 7, tl.cast(ITERS_PER_TILE, tl.int64))
                        slot += 1
                if slot < MAX_SLOTS:   # store (12)
                    e = events + (wg * MAX_SLOTS + slot) * EVENT_FIELDS
                    tl.store(e + 0, wg.to(tl.int64)); tl.store(e + 1, tile.to(tl.int64))
                    tl.store(e + 2, tl.cast(0, tl.int64)); tl.store(e + 3, tl.cast(12, tl.int64))
                    tl.store(e + 4, tsr0); tl.store(e + 5, tend)
                    tl.store(e + 6, tl.cast(0, tl.int64)); tl.store(e + 7, tl.cast(ITERS_PER_TILE, tl.int64))
                    slot += 1
            wid = tl.atomic_add(work_ctr, 1, scope="gpu")
    else:
        # ================= persistent Stream-K (grid == NUM_SMS == CU) =========
        pid = wg
        total_sk_iters = NUM_TILES * ITERS_PER_TILE
        iters_pcu = total_sk_iters // NUM_SMS
        rem_iters = total_sk_iters % NUM_SMS
        start_iter = pid * iters_pcu + tl.minimum(pid, rem_iters)
        last_iter = (pid + 1) * iters_pcu + tl.minimum(pid + 1, rem_iters)

        while start_iter < last_iter:
            remainder = start_iter % ITERS_PER_TILE
            end_iter = tl.minimum(start_iter + (ITERS_PER_TILE - remainder), last_iter)
            tile = start_iter // ITERS_PER_TILE
            tile_iter = tile * ITERS_PER_TILE
            pid_m = tile // num_pid_n
            pid_n = tile % num_pid_n

            tp0 = read_realtime() if TRACE else tl.cast(0, tl.int64)
            rm_raw = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            rn_raw = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_m = rm_raw < M
            mask_n = rn_raw < N
            rm = rm_raw % M
            rn = rn_raw % N
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak + BLOCK_K * stride_ak * remainder
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn + BLOCK_K * stride_bk * remainder
            tc0 = read_realtime() if TRACE else tl.cast(0, tl.int64)

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for current_iter in range(start_iter, end_iter):
                if TRACE and PER_ITER:
                    ti0 = read_realtime()
                if EVEN_K:
                    a = tl.load(A_BASE)
                    b = tl.load(B_BASE)
                else:
                    gko = (current_iter - tile_iter) * BLOCK_K
                    kmask = gko + rk < K
                    a = tl.load(A_BASE, mask=mask_m[:, None] & kmask[None, :], other=0.0)
                    b = tl.load(B_BASE, mask=kmask[:, None] & mask_n[None, :], other=0.0)
                acc += tl.dot(a, b, allow_tf32=False)
                A_BASE += BLOCK_K * stride_ak
                B_BASE += BLOCK_K * stride_bk
                if TRACE and PER_ITER:
                    if slot < MAX_SLOTS:
                        e = events + (wg * MAX_SLOTS + slot) * EVENT_FIELDS
                        tl.store(e + 0, wg.to(tl.int64)); tl.store(e + 1, tile.to(tl.int64))
                        tl.store(e + 2, (current_iter - tile_iter).to(tl.int64)); tl.store(e + 3, tl.cast(11, tl.int64))
                        tl.store(e + 4, ti0); tl.store(e + 5, read_realtime())
                        tl.store(e + 6, (current_iter - tile_iter).to(tl.int64)); tl.store(e + 7, (current_iter - tile_iter + 1).to(tl.int64))
                        slot += 1
            tsr0 = read_realtime() if TRACE else tl.cast(0, tl.int64)

            if TRACE:
                if slot < MAX_SLOTS:   # prologue (10)
                    e = events + (wg * MAX_SLOTS + slot) * EVENT_FIELDS
                    tl.store(e + 0, wg.to(tl.int64)); tl.store(e + 1, tile.to(tl.int64))
                    tl.store(e + 2, (start_iter - tile_iter).to(tl.int64)); tl.store(e + 3, tl.cast(10, tl.int64))
                    tl.store(e + 4, tp0); tl.store(e + 5, tc0)
                    tl.store(e + 6, (start_iter - tile_iter).to(tl.int64)); tl.store(e + 7, (end_iter - tile_iter).to(tl.int64))
                    slot += 1
                if not PER_ITER:       # compute (11)
                    if slot < MAX_SLOTS:
                        e = events + (wg * MAX_SLOTS + slot) * EVENT_FIELDS
                        tl.store(e + 0, wg.to(tl.int64)); tl.store(e + 1, tile.to(tl.int64))
                        tl.store(e + 2, (start_iter - tile_iter).to(tl.int64)); tl.store(e + 3, tl.cast(11, tl.int64))
                        tl.store(e + 4, tc0); tl.store(e + 5, tsr0)
                        tl.store(e + 6, (start_iter - tile_iter).to(tl.int64)); tl.store(e + 7, (end_iter - tile_iter).to(tl.int64))
                        slot += 1

            if not TREE:
                if start_iter != tile_iter:
                    # contributor: publish partial to per-pid slot, then signal lock
                    P_ = P + pid * BLOCK_M * BLOCK_N + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
                    tl.store(P_, acc, cache_modifier=".wt")
                    tl.debug_barrier()
                    tl.store(locks + pid, 1, cache_modifier=".wt")
                    tend = read_realtime() if TRACE else tl.cast(0, tl.int64)
                    if TRACE:
                        if slot < MAX_SLOTS:   # store (12)
                            e = events + (wg * MAX_SLOTS + slot) * EVENT_FIELDS
                            tl.store(e + 0, wg.to(tl.int64)); tl.store(e + 1, tile.to(tl.int64))
                            tl.store(e + 2, (start_iter - tile_iter).to(tl.int64)); tl.store(e + 3, tl.cast(12, tl.int64))
                            tl.store(e + 4, tsr0); tl.store(e + 5, tend)
                            tl.store(e + 6, (start_iter - tile_iter).to(tl.int64)); tl.store(e + 7, (end_iter - tile_iter).to(tl.int64))
                            slot += 1
                else:
                    # owner: fold following pids' partials into registers, write C once
                    tr0 = read_realtime() if TRACE else tl.cast(0, tl.int64)
                    next_pid = pid + 1
                    tile_iter_end = tile_iter + ITERS_PER_TILE
                    end = end_iter
                    while (end < tile_iter_end) and (next_pid < NUM_SMS):
                        while tl.load(locks + next_pid, cache_modifier=".cv", volatile=True) != 1:
                            pass
                        Pf = P + next_pid * BLOCK_M * BLOCK_N + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
                        acc += tl.load(Pf, cache_modifier=".cv")
                        end += iters_pcu + tl.where(next_pid < rem_iters, 1, 0)
                        next_pid += 1
                    mask = mask_m[:, None] & mask_n[None, :]
                    C_ = C + rm_raw[:, None] * stride_cm + rn_raw[None, :] * stride_cn
                    tl.store(C_, acc.to(C.type.element_ty), mask=mask)
                    tr1 = read_realtime() if TRACE else tl.cast(0, tl.int64)
                    if TRACE:
                        if slot < MAX_SLOTS:   # reduce (13)
                            e = events + (wg * MAX_SLOTS + slot) * EVENT_FIELDS
                            tl.store(e + 0, wg.to(tl.int64)); tl.store(e + 1, tile.to(tl.int64))
                            tl.store(e + 2, tl.cast(-1, tl.int64)); tl.store(e + 3, tl.cast(13, tl.int64))
                            tl.store(e + 4, tr0); tl.store(e + 5, tr1)
                            tl.store(e + 6, tl.cast(0, tl.int64)); tl.store(e + 7, tl.cast(ITERS_PER_TILE, tl.int64))
                            slot += 1
            else:
                # ---------------- TREE reduction over contributors ----------------
                # The tile's contributors are the contiguous pid run that covers
                # [tile_iter, tile_iter+ITERS_PER_TILE). Compute base(owner)/s/local
                # by inverting the linear iter->pid map, then do a log-depth binary
                # tree: receiver 'local' pulls partner 'local+2^r' (a sender whose
                # trailing-zero level == r) each round; owner (local 0) writes C.
                q = iters_pcu
                rr = rem_iters
                thr = rr * (q + 1)
                ti = tile_iter
                tj = tile_iter + ITERS_PER_TILE - 1
                base = tl.where(ti < thr, ti // (q + 1), rr + (ti - thr) // tl.maximum(q, 1))
                last = tl.where(tj < thr, tj // (q + 1), rr + (tj - thr) // tl.maximum(q, 1))
                s = last - base + 1
                local = pid - base
                tr0 = read_realtime() if TRACE else tl.cast(0, tl.int64)
                done_tree = local < 0   # tl.int1 scalar, False; True once WG sends
                for r in range(0, TREE_ROUNDS):
                    stride = 1 << r
                    if ((stride < s) & (~done_tree)):
                        if (local % (stride * 2)) == 0:
                            partner = local + stride
                            if partner < s:
                                ppid = base + partner
                                while tl.atomic_add(locks + ppid, 0, sem="acquire", scope="sys") < (r + 1):
                                    pass
                                Pf = P + ppid * BLOCK_M * BLOCK_N + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
                                acc += tl.load(Pf, cache_modifier=".cv")
                        else:
                            Ps = P + pid * BLOCK_M * BLOCK_N + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
                            tl.store(Ps, acc, cache_modifier=".wt")
                            tl.debug_barrier()
                            tl.atomic_xchg(locks + pid, r + 1, sem="release", scope="sys")
                            done_tree = local >= 0   # tl.int1 scalar, True
                if local == 0:
                    mask = mask_m[:, None] & mask_n[None, :]
                    C_ = C + rm_raw[:, None] * stride_cm + rn_raw[None, :] * stride_cn
                    tl.store(C_, acc.to(C.type.element_ty), mask=mask)
                tr1 = read_realtime() if TRACE else tl.cast(0, tl.int64)
                if TRACE:
                    if slot < MAX_SLOTS:   # owner -> reduce(13), sender -> store(12)
                        e = events + (wg * MAX_SLOTS + slot) * EVENT_FIELDS
                        ph = tl.where(local == 0, tl.cast(13, tl.int64), tl.cast(12, tl.int64))
                        tl.store(e + 0, wg.to(tl.int64)); tl.store(e + 1, tile.to(tl.int64))
                        tl.store(e + 2, local.to(tl.int64)); tl.store(e + 3, ph)
                        tl.store(e + 4, tr0); tl.store(e + 5, tr1)
                        tl.store(e + 6, tl.cast(0, tl.int64)); tl.store(e + 7, s.to(tl.int64))
                        slot += 1

            start_iter = end_iter


@triton.jit()
def sk_tree(
    A, B, C, P, locks,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    NUM_TILES, ITERS_PER_TILE, NUM_SMS,            # RUNTIME (not constexpr) -> one compile
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    TREE_ROUNDS: tl.constexpr = 12, EVEN_K: tl.constexpr = True,
):
    """Lean persistent Stream-K + tree reduction, no tracing, runtime tile counts.

    Same algorithm as steal_k_adaptive's STREAMK+TREE path but with NUM_TILES /
    ITERS_PER_TILE / NUM_SMS passed at runtime, so a single compiled kernel serves
    any shape (needed for large shape sweeps). Requires K % BLOCK_K == 0 (EVEN_K).
    grid must be launched with NUM_SMS programs.
    """
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    total = NUM_TILES * ITERS_PER_TILE
    q = total // NUM_SMS
    rem = total % NUM_SMS
    start_iter = pid * q + tl.minimum(pid, rem)
    last_iter = (pid + 1) * q + tl.minimum(pid + 1, rem)

    while start_iter < last_iter:
        remainder = start_iter % ITERS_PER_TILE
        end_iter = tl.minimum(start_iter + (ITERS_PER_TILE - remainder), last_iter)
        tile = start_iter // ITERS_PER_TILE
        tile_iter = tile * ITERS_PER_TILE
        pid_m = tile // num_pid_n
        pid_n = tile % num_pid_n
        rm_raw = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn_raw = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = rm_raw < M
        mask_n = rn_raw < N
        rm = rm_raw % M
        rn = rn_raw % N
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak + BLOCK_K * stride_ak * remainder
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn + BLOCK_K * stride_bk * remainder
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for current_iter in range(start_iter, end_iter):
            if EVEN_K:
                a = tl.load(A_BASE)
                b = tl.load(B_BASE)
            else:
                gko = (current_iter - tile_iter) * BLOCK_K
                kmask = gko + rk < K
                a = tl.load(A_BASE, mask=mask_m[:, None] & kmask[None, :], other=0.0)
                b = tl.load(B_BASE, mask=kmask[:, None] & mask_n[None, :], other=0.0)
            acc += tl.dot(a, b, allow_tf32=False)
            A_BASE += BLOCK_K * stride_ak
            B_BASE += BLOCK_K * stride_bk

        # ---- tree reduction over the tile's contiguous contributor pid-run ----
        thr = rem * (q + 1)
        ti = tile_iter
        tj = tile_iter + ITERS_PER_TILE - 1
        base = tl.where(ti < thr, ti // (q + 1), rem + (ti - thr) // tl.maximum(q, 1))
        last = tl.where(tj < thr, tj // (q + 1), rem + (tj - thr) // tl.maximum(q, 1))
        s = last - base + 1
        local = pid - base
        done_tree = local < 0
        for r in range(0, TREE_ROUNDS):
            stride = 1 << r
            if ((stride < s) & (~done_tree)):
                if (local % (stride * 2)) == 0:
                    partner = local + stride
                    if partner < s:
                        ppid = base + partner
                        # ACQUIRE (system scope) so the partner's partial store is
                        # visible before we read P. Multi-hop tree tightens the
                        # timing vs single-hop fold, so gpu-scope alone leaked a
                        # rare stale read; sys scope forces the cross-XCD drain.
                        while tl.atomic_add(locks + ppid, 0, sem="acquire", scope="sys") < (r + 1):
                            pass
                        Pf = P + ppid * BLOCK_M * BLOCK_N + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
                        acc += tl.load(Pf, cache_modifier=".cv")
                else:
                    Ps = P + pid * BLOCK_M * BLOCK_N + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
                    tl.store(Ps, acc, cache_modifier=".wt")
                    tl.debug_barrier()   # drain the workgroup's P stores...
                    # ...then RELEASE (system scope) publishes P before the lock flips.
                    tl.atomic_xchg(locks + pid, r + 1, sem="release", scope="sys")
                    done_tree = local >= 0
        if local == 0:
            mask = mask_m[:, None] & mask_n[None, :]
            C_ = C + rm_raw[:, None] * stride_cm + rn_raw[None, :] * stride_cn
            tl.store(C_, acc.to(C.type.element_ty), mask=mask)

        start_iter = end_iter
