# steal-k — a work-stealing / Stream-K GEMM kernel (research)

`steal-k` is an experimental Triton GEMM kernel for AMD CDNA (MI300X, gfx942)
exploring **work-stealing** and **Stream-K** scheduling with in-kernel
`s_memrealtime` tracing. It is a research/experiment kernel, not part of the
tritonBLAS public API.

Benchmarked on MI300X (gfx942), fp16.

## Kernel (`steal_k.py`)

- `steal_k_adaptive` — two scheduling paths selected by a `STREAMK` constexpr:
  - **register / work-stealing whole tiles** (`STREAMK=False`): each workgroup
    pulls the next tile from a global atomic counter, computes full-K in
    registers, writes C directly (no reduction). Data-parallel regime (`nt>=CU`).
  - **persistent Stream-K + fixup** (`STREAMK=True`): grid == CU (all WGs
    resident); the flat `tiles x k` iteration space is split contiguously;
    contributors publish partials to `P[pid]`/`locks[pid]`, the tile owner folds
    them and writes C. Deadlock-free because consumers only wait on higher,
    co-resident pids. Underfilled regime (`nt<CU`).
  - Optional **tree reduction** over a tile's contributors (`TREE=True`):
    log-depth binary fold, ~2x on heavily-split shapes.
- `sk_tree` — a lean, single-compile Stream-K + tree kernel with the tile-count
  scalars passed at **runtime** (only the tile is constexpr), so large shape
  sweeps need only a handful of compiles.

### Key implementation notes / gotchas (gfx942, Triton 3.6)
- **Unmasked `EVEN_K` loads** are required to let the software pipeliner
  (`num_stages>=2`) work; a masked load inside the pipelined K-loop miscompiles.
- **Gate the `s_memrealtime` timers behind `TRACE`** — the inline asm doesn't
  dead-code-eliminate and, left in, spills the 256x256 accumulator.
- Reduction uses **lock + `.wt`/`.cv`**, never fp32 global atomic-add.
- The **multi-hop tree** needs **system-scope release/acquire** atomics on the
  lock (`sem="release"/"acquire", scope="sys"`) + a `debug_barrier`; the
  single-hop owner-fold is timing-safe without them.

## Drivers / harnesses
- `bench_variants.py` — steal-k vs library data-parallel & Stream-K, per shape.
- `bench_tree.py`, `stress_tree.py`, `stress_sktree.py` — tree reduction perf +
  correctness stress.
- `sweep_tune.py` / `aggregate_sweep.py` — tile/path/grid/num_stages sweep.
- `pt_sweep.py` / `pt_aggregate.py` / `gen_shapes.py` — vs `torch.matmul` at scale.
- `tune_vs_torch.py`, `equal_opp.py`, `generalize_test.py`, `origami_stealk.py`,
  `lib_vs_torch.py` — tuning / fairness / Origami-selector studies.
- `trace_stealk_adaptive.py`, `trace_helpers.py` — emit Chrome/Perfetto traces.
- `diag_*.py`, `probe_large.py` — isolation harnesses.

## Findings (summary)
- steal-k **beats the tritonBLAS library** (persistent + Stream-K) by ~1.2-2.5x
  on skinny/underfilled big-K shapes, even when both are tuned the same way, and
  matches/edges it on large shapes.
- steal-k **loses to rocBLAS/PyTorch** on skinny GEMM (~0.55-0.62x geomean);
  the gap is kernel/launch efficiency, not tile config.
- The **Origami Stream-K selector generalizes** well as steal-k's tile selector
  (~+35% over a fixed tile across a diverse shape mix; ~82% of a full tune).
- Peak ~525 TF/s (large squares, 256x256 register) ≈ 40% of MI300X fp16 peak.

Requires Triton + the `origami`/`tritonblas` package for the comparison drivers;
the core `steal_k.py` kernel only needs Triton + torch.
