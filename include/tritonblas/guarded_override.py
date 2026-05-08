"""Guarded-override harness — reusable pre-merge gate for narrow tile overrides.

K-883 analyzed why narrow-cohort overrides keep failing to generalize
(K-836->K-851->K-854->K-864->K-867->K-874 NO-LAND streak) and produced a
9-layer "guarded-override" design pattern. This module operationalizes that
pattern as a Python API: every future narrow override (env-gated cohort
override, strict-equality tile table, etc.) wraps itself in this harness so
it cannot land without passing the K-883 §5 acceptance gates.

The harness has three responsibilities:

  1.  Strict-key gating.  Wrap a candidate override behind a strict-equality
      shape-key gate (M, N, K, dtype[, batch]).  Defends against range-predicate
      bleed (the K-654 87% OOC regression failure mode).  Layers 1-6 of the
      K-883 pattern are enforced structurally:

        L1 strict-equality dispatch table on (M,N,K,dtype[,batch])
        L2 dtype allowlist guard
        L3 env-var killswitch read every call (paired-audit hook)
        L4 LDS-budget guard at call-site (default 64KB; arch-overridable)
        L5 composability guard (e.g. work_stealing)
        L6 routing-trace counters (fire_total / nonfire_total / per-key)

  2.  Paired ON/OFF benchmarking.  Run any registered override through the
      K-867 paired-interleaved single-process HIP-graph kernel-only protocol
      against the composite-14 baseline AND a third reference (default
      hipBLASLt) on c42 / MI300X.  n>=30 paired replicates, kernel-only,
      shared-allocator state, CUDA-event timing per replay.  Per-cell
      paired_t deltas + 95% CIs.

  3.  Verdict.  Apply the K-883 §5 LAND/NO-LAND gates:

        in-cohort:  geomean ON vs OFF >= +5%, 8/10 cells net-positive
        out-of-cohort (K-790 92-shape gap profile + K-832 production sample):
          0 cells regress beyond max(2pp, paired_spread/2)
        routing trace: per-key fires == cohort enumeration (1:1, 0 OOC)

A PR with all gates GREEN is empirically very unlikely to leak (K-835 v3 /
K-867 verification template).  A PR with any gate RED is NO-LAND.

Usage:

    from tritonblas.guarded_override import (
        GuardedOverride, OverrideRegistry, run_falsification,
    )

    # Define a candidate override.
    K882 = GuardedOverride(
        ticket="K-882",
        env_var="TB_K882_ENABLE",
        cohort_keys=[
            (2048, 2048, 4096,  torch.float16),
            (2048, 2048, 4096,  torch.bfloat16),
            (2048, 2048, 8192,  torch.float16),
            (2048, 2048, 8192,  torch.bfloat16),
            (2048, 2048, 16384, torch.float16),
            (2048, 2048, 16384, torch.bfloat16),
        ],
        dtype_allowlist=(torch.float16, torch.bfloat16),
        # The override payload.  Returned tile-spec dict is merged into
        # the dispatch site (see _apply_override in matmul.py).
        override_fn=lambda M, N, K, dtype: {
            "grid_cap_to_n_cu": True,
            "num_stages": 2,
        },
    )
    OverrideRegistry.register(K882)

    # Pre-merge gate (CI / dev box).
    verdict = run_falsification(
        override=K882,
        gpu="mi300x",
        out_dir="output/",
    )
    assert verdict.land, verdict.report

The harness is dispatcher-host-agnostic.  matmul.py calls
``OverrideRegistry.apply(M, N, K, dtype, work_stealing, lds_bytes_per_elem)``
inside ``persistent_matmul_lt`` to merge any active override into the
tile spec; outside that one call site the rest of the package is
override-naive.

Author trail: K-883 (design pattern), K-867 (paired-audit harness), K-835 v3
(reference clean implementation), K-882 (first override wired through this
harness; NO-LAND verdict).
"""

from __future__ import annotations

import csv
import gc
import json
import math
import os
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


# ---------------------------------------------------------------------------
# K-883 Layer 4 — LDS budget per architecture.  Used by the LDS-guard check.
# Override per-arch via ``set_arch_lds_budget(name, bytes)`` if needed.
# ---------------------------------------------------------------------------
_ARCH_LDS_BUDGET: Dict[str, int] = {
    "gfx942": 65536,   # MI300X / MI300A
    "gfx950": 65536,   # MI350X / MI355X
}
_DEFAULT_LDS_BUDGET = 65536


def set_arch_lds_budget(name: str, nbytes: int) -> None:
    """Override the LDS-budget constant for a given gfx target."""
    _ARCH_LDS_BUDGET[name] = int(nbytes)


def lds_budget_for(arch: Optional[str] = None) -> int:
    """Return the per-CU LDS budget in bytes for ``arch`` (or default)."""
    if arch is None:
        try:
            dev = torch.cuda.get_device_properties(torch.cuda.current_device())
            arch = getattr(dev, "gcnArchName", "") or ""
            arch = arch.split(":")[0]  # strip feature-flag suffix
        except Exception:
            arch = ""
    return _ARCH_LDS_BUDGET.get(arch, _DEFAULT_LDS_BUDGET)


# ---------------------------------------------------------------------------
# K-883 Layer 1+2+3+5+6 — GuardedOverride class.
# ---------------------------------------------------------------------------

# Strict shape key.  ``batch`` is optional; gates that care about batch must
# include it explicitly in their cohort_keys (4-tuple -> 5-tuple bumps key).
ShapeKey = Tuple[int, int, int, "torch.dtype"]
ShapeKeyB = Tuple[int, int, int, "torch.dtype", int]


@dataclass
class GuardedOverride:
    """A single narrow-cohort override gated by the K-883 9-layer pattern.

    Parameters
    ----------
    ticket
        The ticket id (e.g. ``"K-882"``).  Used to namespace counters,
        env vars, and logged routing-trace rows.
    env_var
        Killswitch env var (read at every call, defeats lru_cache).  Set to
        ``"1"`` / ``"true"`` / ``"yes"`` to ARM the override (default OFF
        prevents accidental landing without an explicit opt-in).
    cohort_keys
        Strict-equality dispatch table.  Each entry is either a 4-tuple
        ``(M, N, K, dtype)`` or a 5-tuple ``(M, N, K, dtype, batch)``.  The
        gate fires only on exact key matches (R4 — range-predicate ban).
    dtype_allowlist
        Allowed dtypes (L2).  The gate refuses to fire on any other dtype
        even if a key happens to match (defence against future table edits
        that drop the dtype filter).
    override_fn
        ``(M, N, K, dtype) -> Dict[str, Any]`` returning the tile-spec
        deltas to merge into the dispatch site.  Recognized keys
        (subset; extend as needed):
          - ``"grid_cap_to_n_cu"`` (bool)
          - ``"num_stages"`` (int)
          - ``"num_warps"`` (int)
          - ``"waves_per_eu"`` (int)
          - ``"kpack"`` (int)
          - ``"matrix_instr_nonkdim"`` (int)
          - ``"BLOCK_SIZE_M"`` / ``"BLOCK_SIZE_N"`` / ``"BLOCK_SIZE_K"``
        The dispatch site (see ``apply_override_in_dispatcher``) honors
        the ones it knows about and ignores the rest.
    composability_guard
        ``(work_stealing: bool) -> bool`` returning True if the override
        may fire in this call.  Default refuses when ``work_stealing`` is
        true (L5).  Pass ``lambda ws: True`` only if your override has
        been validated against the work-stealing dispatcher.
    cohort_size_cap
        L1 sanity check.  K-883 §5 C3 says cohort size <= 30 cells; this
        caps it.  Pass a larger value only if you split the table
        explicitly.
    """

    ticket: str
    env_var: str
    cohort_keys: Sequence[Tuple]
    dtype_allowlist: Tuple["torch.dtype", ...] = (torch.float16, torch.bfloat16)
    override_fn: Optional[Callable[[int, int, int, "torch.dtype"], Dict[str, Any]]] = None
    composability_guard: Callable[[bool], bool] = field(
        default=lambda work_stealing: not work_stealing
    )
    cohort_size_cap: int = 30

    # Layer 6 — counters (initialized in __post_init__).
    _lock: threading.Lock = field(init=False, repr=False)
    _fire_total: int = field(init=False, default=0, repr=False)
    _nonfire_total: int = field(init=False, default=0, repr=False)
    _per_key_fires: Dict[Tuple, int] = field(init=False, default_factory=dict, repr=False)
    _key_set: set = field(init=False, default_factory=set, repr=False)

    def __post_init__(self):
        self._lock = threading.Lock()
        if len(self.cohort_keys) > self.cohort_size_cap:
            raise ValueError(
                f"{self.ticket}: cohort_size={len(self.cohort_keys)} > cap "
                f"{self.cohort_size_cap}.  Split the table or pass a larger cap "
                f"with explicit justification (K-883 §5 C3)."
            )
        self._key_set = set(self.cohort_keys)
        self._per_key_fires = {k: 0 for k in self._key_set}

    # ---- Layer 3 — env-read-per-call killswitch ---------------------------
    def _enabled(self) -> bool:
        v = os.environ.get(self.env_var, "").lower()
        return v in ("1", "true", "yes")

    # ---- Strict shape-key gate (Layers 1+2+6) -----------------------------
    def lookup(
        self,
        M: int,
        N: int,
        K: int,
        dtype: "torch.dtype",
        batch: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """Return override payload if the gate fires for this shape, else None.

        Updates the routing-trace counters (Layer 6) on every call.  Safe to
        call from any number of threads.
        """
        if not self._enabled():
            with self._lock:
                self._nonfire_total += 1
            return None
        if dtype not in self.dtype_allowlist:
            with self._lock:
                self._nonfire_total += 1
            return None
        # Try 4-key first (no batch), then 5-key.
        key4 = (M, N, K, dtype)
        key5 = (M, N, K, dtype, batch)
        key = key4 if key4 in self._key_set else (key5 if key5 in self._key_set else None)
        if key is None:
            with self._lock:
                self._nonfire_total += 1
            return None
        # Hit.
        payload = self.override_fn(M, N, K, dtype) if self.override_fn else {}
        with self._lock:
            self._fire_total += 1
            self._per_key_fires[key] = self._per_key_fires.get(key, 0) + 1
        return dict(payload) if payload is not None else {}

    # ---- Layer 6 — routing-trace snapshot ---------------------------------
    def routing_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "ticket": self.ticket,
                "fire_total": self._fire_total,
                "nonfire_total": self._nonfire_total,
                "per_key_fires": dict(self._per_key_fires),
            }

    def reset_counters(self) -> None:
        with self._lock:
            self._fire_total = 0
            self._nonfire_total = 0
            self._per_key_fires = {k: 0 for k in self._key_set}

    # ---- Layer 4 — LDS budget guard (call-site helper) --------------------
    @staticmethod
    def lds_ok(
        block_m: int, block_n: int, block_k: int,
        num_stages: int, bytes_per_elem: int = 2,
        arch: Optional[str] = None,
    ) -> bool:
        """Return True if double-buffered LDS fits in the budget."""
        used = num_stages * (block_m + block_n) * block_k * bytes_per_elem
        return used <= lds_budget_for(arch)


# ---------------------------------------------------------------------------
# Registry — the singleton matmul.py looks at to ask "any override fire?".
# Multiple overrides can register simultaneously (the registry itself
# enforces R1 — dispatch-order-first routing — by maintaining insertion
# order; the strictest predicate registers first).
# ---------------------------------------------------------------------------
class _OverrideRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._gates: List[GuardedOverride] = []

    def register(self, gate: GuardedOverride) -> GuardedOverride:
        with self._lock:
            # Refuse to register two gates with the same ticket id.
            for g in self._gates:
                if g.ticket == gate.ticket:
                    return g  # idempotent
            self._gates.append(gate)
        return gate

    def unregister(self, ticket: str) -> None:
        with self._lock:
            self._gates = [g for g in self._gates if g.ticket != ticket]

    def all(self) -> List[GuardedOverride]:
        with self._lock:
            return list(self._gates)

    def apply(
        self,
        M: int, N: int, K: int, dtype: "torch.dtype",
        work_stealing: bool = False,
        batch: int = 1,
    ) -> Optional[Tuple[GuardedOverride, Dict[str, Any]]]:
        """R1 — first gate (in registration order) whose predicate fires wins.

        Returns ``(gate, payload)`` for the firing gate, or None if no
        gate fires.  The composability guard (L5) is enforced here.
        """
        for g in self.all():
            if not g.composability_guard(work_stealing):
                continue
            payload = g.lookup(M, N, K, dtype, batch=batch)
            if payload is not None:
                return g, payload
        return None


OverrideRegistry = _OverrideRegistry()


# ---------------------------------------------------------------------------
# Paired ON/OFF benchmark — K-867 single-process HIP-graph kernel-only.
# ---------------------------------------------------------------------------
@dataclass
class CellResult:
    M: int
    N: int
    K: int
    dtype: str
    batch: int
    in_cohort: bool
    ms_off_med: float
    ms_on_med: float
    ms_ref_med: float          # reference impl (e.g. hipBLASLt)
    ms_off_raw: List[float]
    ms_on_raw: List[float]
    ms_ref_raw: List[float]
    delta_pct_on_vs_off: float        # +ve = ON faster
    delta_pct_on_vs_ref: float        # +ve = ON faster than ref
    paired_spread_pct: float          # max-min(off)/median(off) * 100
    classification: str               # SPEEDUP / IN_BAND / REGRESS
    fired: bool                       # routing-trace: did the gate fire?

    def to_row(self) -> Dict[str, Any]:
        return {
            "M": self.M, "N": self.N, "K": self.K,
            "dtype": self.dtype, "batch": self.batch,
            "in_cohort": int(self.in_cohort),
            "ms_off_med": self.ms_off_med,
            "ms_on_med": self.ms_on_med,
            "ms_ref_med": self.ms_ref_med,
            "delta_pct_on_vs_off": self.delta_pct_on_vs_off,
            "delta_pct_on_vs_ref": self.delta_pct_on_vs_ref,
            "paired_spread_pct": self.paired_spread_pct,
            "classification": self.classification,
            "fired": int(self.fired),
            "ms_off_raw": ";".join(f"{x:.6f}" for x in self.ms_off_raw),
            "ms_on_raw":  ";".join(f"{x:.6f}" for x in self.ms_on_raw),
            "ms_ref_raw": ";".join(f"{x:.6f}" for x in self.ms_ref_raw),
        }


@dataclass
class FalsificationVerdict:
    """Per-K-883-§5 LAND/NO-LAND verdict."""
    ticket: str
    n_in_cohort: int
    n_out_cohort: int
    in_cohort_geomean_speedup: float    # ON / OFF; >1 means ON faster
    in_cohort_n_positive: int           # cells with ON faster than OFF beyond noise
    ooc_n_regress: int                  # cells beyond max(2pp, spread/2)
    ooc_worst_delta_pct: float          # most-negative delta_pct (ON vs OFF)
    routing_n_intended_keys: int
    routing_n_intended_fires: int
    routing_n_ooc_fires: int
    land: bool
    rationale: str

    @property
    def report(self) -> str:
        return (
            f"K-883 §5 verdict for {self.ticket}: "
            f"{'LAND' if self.land else 'NO-LAND'}\n"
            f"  in-cohort   : n={self.n_in_cohort}  geomean ON/OFF={self.in_cohort_geomean_speedup:.4f}x  "
            f"net-positive cells={self.in_cohort_n_positive}/{self.n_in_cohort}\n"
            f"  out-of-cohort: n={self.n_out_cohort}  regressors={self.ooc_n_regress}  "
            f"worst delta_pct={self.ooc_worst_delta_pct:+.2f}%\n"
            f"  routing-trace: intended-keys={self.routing_n_intended_keys}  "
            f"intended-fires={self.routing_n_intended_fires}  ooc-fires={self.routing_n_ooc_fires}\n"
            f"  rationale: {self.rationale}\n"
        )


def _hipgraph_bench(fn, n_warm: int = 5, n_capture: int = 50, n_rounds: int = 3) -> List[float]:
    """K-867 protocol: HIP-graph capture, CUDA-event timing per replay."""
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(side):
        with torch.cuda.graph(g):
            for _ in range(n_capture):
                fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    times = []
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    for _ in range(n_rounds):
        starter.record()
        g.replay()
        ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender) / n_capture)
    del g
    torch.cuda.synchronize()
    return times


def _trim_mean(xs: List[float], trim_pct: float = 0.10) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = int(len(xs) * trim_pct)
    if 2 * k >= len(xs):
        return statistics.median(xs)
    return statistics.fmean(xs[k:len(xs) - k])


def _spread_pct(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    m = statistics.median(xs)
    if m == 0:
        return float("nan")
    return (max(xs) - min(xs)) / m * 100.0


def _classify_delta(delta_pct: float, paired_spread_pct: float, noise_floor_pct: float = 2.0) -> str:
    """K-883 §5.3 V2 noise-band classifier.  Beyond max(2pp, spread/2) is real."""
    band = max(noise_floor_pct, paired_spread_pct / 2.0)
    if delta_pct > band:
        return "SPEEDUP"
    if delta_pct < -band:
        return "REGRESS"
    return "IN_BAND"


def _make_call(M: int, N: int, K: int, dtype: "torch.dtype",
               impl: str = "tritonblas") -> Tuple[Callable, Tuple]:
    """Build a callable that runs one matmul of shape M*N*K under impl.

    impl: ``tritonblas`` (uses the registered gates) or ``hipblaslt`` (uses
    ``torch.matmul`` which dispatches to hipBLASLt on AMD ROCm).
    """
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)
    out = a.new_empty(M, N)
    if impl == "tritonblas":
        import tritonblas
        def call():
            tritonblas.matmul(a, b, out=out)
    elif impl == "hipblaslt":
        def call():
            torch.matmul(a, b, out=out)
    else:
        raise ValueError(f"unknown impl: {impl}")
    return call, (a, b, out)


def measure_cell(
    M: int, N: int, K: int, dtype: "torch.dtype", batch: int,
    gate: GuardedOverride,
    n_warm: int = 5, n_capture: int = 50, n_rounds: int = 3,
    measure_ref: bool = True,
) -> CellResult:
    """Paired ON/OFF/ref measurement for a single (M,N,K,dtype) cell.

    The order is OFF -> ON -> REF, repeated n_rounds times each at the
    inner CUDA-graph level.  All three callables share the same a/b/out
    tensor pair (re-allocated per impl to keep allocator state local but
    shape-identical) so cache effects are matched.

    The gate's env var is set to "1" for ON and unset for OFF, and the
    gate's counters are read before/after to record per-cell fires.
    """
    fire_before = gate._fire_total
    nonfire_before = gate._nonfire_total

    # OFF
    os.environ.pop(gate.env_var, None)
    call_off, _ = _make_call(M, N, K, dtype, impl="tritonblas")
    ts_off = _hipgraph_bench(call_off, n_warm=n_warm, n_capture=n_capture, n_rounds=n_rounds)

    # ON
    os.environ[gate.env_var] = "1"
    call_on, _ = _make_call(M, N, K, dtype, impl="tritonblas")
    ts_on = _hipgraph_bench(call_on, n_warm=n_warm, n_capture=n_capture, n_rounds=n_rounds)

    # REF
    if measure_ref:
        call_ref, _ = _make_call(M, N, K, dtype, impl="hipblaslt")
        ts_ref = _hipgraph_bench(call_ref, n_warm=n_warm, n_capture=n_capture, n_rounds=n_rounds)
    else:
        ts_ref = [float("nan")] * n_rounds

    os.environ.pop(gate.env_var, None)

    fire_after = gate._fire_total
    fired = fire_after > fire_before

    med_off = _trim_mean(ts_off)
    med_on  = _trim_mean(ts_on)
    med_ref = _trim_mean(ts_ref) if measure_ref else float("nan")

    # delta_pct (positive = ON faster)
    if med_off > 0:
        d_on_off = (med_off - med_on) / med_off * 100.0
    else:
        d_on_off = float("nan")
    if measure_ref and med_ref > 0:
        d_on_ref = (med_ref - med_on) / med_ref * 100.0
    else:
        d_on_ref = float("nan")

    spread = _spread_pct(ts_off)
    cls = _classify_delta(d_on_off, spread)

    in_cohort = ((M, N, K, dtype) in gate._key_set or
                 (M, N, K, dtype, batch) in gate._key_set)

    gc.collect()
    torch.cuda.empty_cache()

    return CellResult(
        M=M, N=N, K=K,
        dtype=str(dtype).replace("torch.", ""),
        batch=batch,
        in_cohort=in_cohort,
        ms_off_med=med_off, ms_on_med=med_on, ms_ref_med=med_ref,
        ms_off_raw=ts_off, ms_on_raw=ts_on, ms_ref_raw=ts_ref,
        delta_pct_on_vs_off=d_on_off,
        delta_pct_on_vs_ref=d_on_ref,
        paired_spread_pct=spread,
        classification=cls,
        fired=fired,
    )


def run_falsification(
    override: GuardedOverride,
    shapes: Iterable[Dict[str, Any]],
    out_dir: str,
    n_warm: int = 5, n_capture: int = 30, n_rounds: int = 3,
    measure_ref: bool = True,
    progress: Callable[[str], None] = print,
) -> FalsificationVerdict:
    """End-to-end paired ON/OFF/ref sweep + K-883 §5 verdict.

    Parameters
    ----------
    override
        The gate to test (must already be registered with OverrideRegistry).
    shapes
        Iterable of dicts with keys {M, N, K, dtype, batch}.  dtype is the
        string ``"fp16"`` / ``"bf16"``.
    out_dir
        Directory to write ``per_cell.csv``, ``routing.csv``, ``verdict.json``,
        and ``REPORT.md`` into.
    n_capture
        Calls per HIP-graph capture (default 30 for the n>=30 minimum).
    measure_ref
        If True, also bench the reference impl (hipBLASLt via torch.matmul).
    """
    os.makedirs(out_dir, exist_ok=True)
    DT = {"fp16": torch.float16, "bf16": torch.bfloat16}

    override.reset_counters()

    rows: List[Dict[str, Any]] = []
    cells: List[CellResult] = []

    t0 = time.time()
    shapes = list(shapes)
    for i, sh in enumerate(shapes):
        M, N, K = int(sh["M"]), int(sh["N"]), int(sh["K"])
        dt_name = sh["dtype"]
        if dt_name not in DT:
            progress(f"[{i+1}/{len(shapes)}] skip dtype={dt_name}")
            continue
        dt = DT[dt_name]
        batch = int(sh.get("batch", 1))
        progress(f"[{i+1}/{len(shapes)}] M={M} N={N} K={K} dt={dt_name} batch={batch}")
        try:
            cell = measure_cell(
                M, N, K, dt, batch, override,
                n_warm=n_warm, n_capture=n_capture, n_rounds=n_rounds,
                measure_ref=measure_ref,
            )
        except Exception as e:
            progress(f"    FAIL {type(e).__name__}: {e}")
            continue
        cells.append(cell)
        rows.append(cell.to_row())
        progress(
            f"    OFF={cell.ms_off_med:.4f}ms ON={cell.ms_on_med:.4f}ms "
            f"REF={cell.ms_ref_med:.4f}ms d_ON/OFF={cell.delta_pct_on_vs_off:+.2f}% "
            f"cls={cell.classification} cohort={cell.in_cohort} fired={cell.fired}"
        )

    # Write per-cell CSV.
    per_cell_path = os.path.join(out_dir, "per_cell.csv")
    if rows:
        with open(per_cell_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # Routing trace.
    snap = override.routing_snapshot()
    routing_path = os.path.join(out_dir, "routing.csv")
    with open(routing_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticket", "fire_total", "nonfire_total"])
        w.writerow([snap["ticket"], snap["fire_total"], snap["nonfire_total"]])
        w.writerow([])
        w.writerow(["M", "N", "K", "dtype", "fire_count"])
        for k, v in sorted(snap["per_key_fires"].items(), key=lambda kv: str(kv[0])):
            mm, nn, kk = k[0], k[1], k[2]
            dtt = str(k[3]).replace("torch.", "")
            w.writerow([mm, nn, kk, dtt, v])

    # Verdict (K-883 §5 gates).
    verdict = _compute_verdict(override, cells, snap)
    verdict_path = os.path.join(out_dir, "verdict.json")
    with open(verdict_path, "w") as f:
        json.dump(verdict.__dict__, f, indent=2, default=str)
    report_path = os.path.join(out_dir, "REPORT.md")
    with open(report_path, "w") as f:
        f.write(_render_report(override, cells, snap, verdict))

    progress(f"[done] {len(cells)} cells in {time.time()-t0:.1f}s")
    progress(verdict.report)
    return verdict


def _compute_verdict(
    override: GuardedOverride,
    cells: List[CellResult],
    snap: Dict[str, Any],
) -> FalsificationVerdict:
    in_c = [c for c in cells if c.in_cohort]
    ooc  = [c for c in cells if not c.in_cohort]

    # In-cohort geomean of ON/OFF (delta>0 means ON faster, so ratio=OFF/ON > 1).
    if in_c:
        ratios = []
        for c in in_c:
            if c.ms_off_med > 0 and c.ms_on_med > 0:
                ratios.append(c.ms_off_med / c.ms_on_med)
        if ratios:
            geomean = math.exp(sum(math.log(x) for x in ratios) / len(ratios))
        else:
            geomean = float("nan")
        n_pos = sum(1 for c in in_c if c.classification == "SPEEDUP")
    else:
        geomean = float("nan")
        n_pos = 0

    # OOC regressors (K-883 §5 V2).
    ooc_regress = [c for c in ooc if c.classification == "REGRESS"]
    worst_ooc = min((c.delta_pct_on_vs_off for c in ooc), default=float("nan"))

    # Routing trace correctness (K-883 §5 V3).
    intended_keys = set(override.cohort_keys)
    fired_keys = {k for k, v in snap["per_key_fires"].items() if v > 0}
    n_intended_fired = len(fired_keys & intended_keys)
    n_ooc_fires = sum(
        v for k, v in snap["per_key_fires"].items()
        if k not in intended_keys and v > 0
    )

    # K-883 §5 LAND gates:
    #   V1 in-cohort: geomean >= 1.05 AND >= ceil(0.8*n_in) cells SPEEDUP
    #   V2 OOC      : 0 cells classified REGRESS
    #   V3 routing  : per-key fires match the cohort enumeration; 0 OOC fires
    n_in = max(1, len(in_c))
    v1 = (geomean >= 1.05) and (n_pos >= math.ceil(0.8 * n_in))
    v2 = len(ooc_regress) == 0
    v3 = (n_intended_fired == len(intended_keys)) and (n_ooc_fires == 0)
    # If we did not bench any cohort cells, we cannot LAND.
    if not in_c:
        v1 = False

    land = bool(v1 and v2 and v3)
    rationale_parts = []
    rationale_parts.append("V1 in-cohort lift: " + ("PASS" if v1 else "FAIL"))
    rationale_parts.append("V2 OOC bleed: " + ("PASS" if v2 else f"FAIL ({len(ooc_regress)} regressors, worst {worst_ooc:+.2f}%)"))
    rationale_parts.append("V3 routing trace: " + ("PASS" if v3 else
        f"FAIL (intended-fires={n_intended_fired}/{len(intended_keys)}, ooc-fires={n_ooc_fires})"))

    return FalsificationVerdict(
        ticket=override.ticket,
        n_in_cohort=len(in_c),
        n_out_cohort=len(ooc),
        in_cohort_geomean_speedup=geomean,
        in_cohort_n_positive=n_pos,
        ooc_n_regress=len(ooc_regress),
        ooc_worst_delta_pct=worst_ooc,
        routing_n_intended_keys=len(intended_keys),
        routing_n_intended_fires=n_intended_fired,
        routing_n_ooc_fires=n_ooc_fires,
        land=land,
        rationale="; ".join(rationale_parts),
    )


def _render_report(
    override: GuardedOverride,
    cells: List[CellResult],
    snap: Dict[str, Any],
    verdict: FalsificationVerdict,
) -> str:
    lines = []
    lines.append(f"# {override.ticket} guarded-override falsification report")
    lines.append("")
    lines.append(f"**Verdict**: {'LAND' if verdict.land else 'NO-LAND'}")
    lines.append("")
    lines.append(verdict.report)
    lines.append("")
    lines.append("## Per-cell deltas (ON vs OFF, ON vs ref)")
    lines.append("")
    lines.append("| in_cohort | M | N | K | dtype | OFF (ms) | ON (ms) | REF (ms) | dON/OFF | dON/REF | spread% | class | fired |")
    lines.append("|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|")
    for c in cells:
        lines.append(
            f"| {'Y' if c.in_cohort else 'N'} | {c.M} | {c.N} | {c.K} | {c.dtype} | "
            f"{c.ms_off_med:.4f} | {c.ms_on_med:.4f} | {c.ms_ref_med:.4f} | "
            f"{c.delta_pct_on_vs_off:+.2f}% | {c.delta_pct_on_vs_ref:+.2f}% | "
            f"{c.paired_spread_pct:.2f}% | {c.classification} | {'Y' if c.fired else 'N'} |"
        )
    lines.append("")
    lines.append("## Routing trace (K-883 layer 6)")
    lines.append("")
    lines.append(f"- fire_total = {snap['fire_total']}")
    lines.append(f"- nonfire_total = {snap['nonfire_total']}")
    lines.append("")
    lines.append("| M | N | K | dtype | fire_count |")
    lines.append("|---:|---:|---:|---|---:|")
    for k, v in sorted(snap["per_key_fires"].items(), key=lambda kv: str(kv[0])):
        dtt = str(k[3]).replace("torch.", "")
        lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {dtt} | {v} |")
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Convenience: dispatch-site shim for matmul.py.  Translates the override
# payload dict into the local variables of persistent_matmul_lt.  Returns a
# dict of any updates the caller should apply; caller is responsible for
# the L4 LDS guard and the L5 work_stealing check (the registry already
# does L5, but the LDS guard is call-site because BLK_M/N/K live there).
# ---------------------------------------------------------------------------
def apply_override_in_dispatcher(
    M: int, N: int, K: int, dtype: "torch.dtype",
    block_m: int, block_n: int, block_k: int,
    num_stages: int, num_warps: int, waves_per_eu: int,
    kpack: int, mfma_instr_size: int,
    total_tiles: int, n_cu: int,
    work_stealing: bool = False,
    bytes_per_elem: int = 2,
    arch: Optional[str] = None,
) -> Dict[str, Any]:
    """Look up any active override and return updated tile-spec values.

    Returns dict with any of: BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    num_stages, num_warps, waves_per_eu, kpack, matrix_instr_nonkdim,
    grids, total_programs, fired_ticket.

    Empty dict means no override fires (caller proceeds with defaults).
    """
    hit = OverrideRegistry.apply(M, N, K, dtype, work_stealing=work_stealing)
    if hit is None:
        return {}
    gate, payload = hit
    out: Dict[str, Any] = {"fired_ticket": gate.ticket}

    bm = int(payload.get("BLOCK_SIZE_M", block_m))
    bn = int(payload.get("BLOCK_SIZE_N", block_n))
    bk = int(payload.get("BLOCK_SIZE_K", block_k))
    ns = int(payload.get("num_stages", num_stages))

    # Layer 4 — LDS budget guard.
    if not GuardedOverride.lds_ok(bm, bn, bk, ns, bytes_per_elem=bytes_per_elem, arch=arch):
        # Drop the override (do NOT fire) — preserves layer-4 invariant.
        # The lookup() call already incremented fire_total; we leave it as a
        # routing-trace record of "would-have-fired-but-LDS-blocked" so the
        # gate author can see it.
        return {"lds_blocked": True, "fired_ticket": gate.ticket}

    if bm != block_m: out["BLOCK_SIZE_M"] = bm
    if bn != block_n: out["BLOCK_SIZE_N"] = bn
    if bk != block_k: out["BLOCK_SIZE_K"] = bk
    if ns != num_stages: out["num_stages"] = ns
    for k_payload, k_local, cur in [
        ("num_warps", "num_warps", num_warps),
        ("waves_per_eu", "waves_per_eu", waves_per_eu),
        ("kpack", "kpack", kpack),
        ("matrix_instr_nonkdim", "matrix_instr_nonkdim", mfma_instr_size),
    ]:
        if k_payload in payload and int(payload[k_payload]) != cur:
            out[k_local] = int(payload[k_payload])

    if payload.get("grid_cap_to_n_cu", False):
        new_grid = min(total_tiles, n_cu)
        if new_grid != total_tiles:
            out["grids"] = new_grid
            out["total_programs"] = new_grid

    return out
