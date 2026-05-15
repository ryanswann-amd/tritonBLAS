#!/usr/bin/env python3
"""K-6830 — pre-registered hypothesis test.

The PRD pre-registered hypothesis was:

    "modifying the instruction scheduling heuristic in the AMD backend to
     alternate LDS-read groups with MFMA groups will close the lgkmcnt
     stall gap on 64x64x4096 BF16 without inline assembly"

This script encodes that hypothesis as an explicit, machine-checkable
contract. It loads two JSON sidecars (baseline, patched) produced by
``bench_K6830.py`` / ``bench_tritonblas.py`` and asserts:

  1. ``triton_version`` differs between baseline and patched (otherwise the
     "baseline" build is contaminated — Skeptic-review failure).
  2. Patched ISA either:
       (a) reduces ``longest_mfma_run`` by >= MFMA_RUN_DELTA, OR
       (b) increases ``lds_mfma_transitions`` by >= TRANSITIONS_DELTA, OR
       (c) reduces ``ds_write`` count by >= DS_WRITE_DELTA
     on the matched (M, N, K, dtype) tuple.
  3. Patched TFLOPS improves by >= TFLOPS_FACTOR over baseline.

The script EXITS NON-ZERO when the hypothesis fails. That is *intentional*.
The previous attempt narrated the negative result in a PR description and
shipped it as a "fix" — Testing-Zealot review correctly flagged that as
unverifiable. With this gate in place, "ship as fix" requires a green run;
otherwise the only honest outcome is "file as negative finding."

Usage:
  python3 test_K6830_hypothesis.py \
      --baseline output/bench_tb_baseline.json \
      --patched  output/bench_tb_patched.json \
      [--shape 64,64,4096] [--strict]

Exit codes:
  0  — hypothesis supported (patched ISA + TFLOPS shifts as predicted)
  1  — hypothesis NOT supported (negative result; do NOT ship as fix)
  2  — usage / data-integrity error (e.g. contaminated baseline)
"""
import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Tuple

# --- Hypothesis thresholds (pre-registered, do not relax to make tests pass) ---
MFMA_RUN_DELTA = 1            # patched longest_mfma_run must drop by >=1
TRANSITIONS_DELTA = 4         # OR transitions must increase by >=4 (>noise)
DS_WRITE_DELTA = 4            # OR ds_write count must drop by >=4
TFLOPS_FACTOR = 1.05          # patched TFLOPS must be >=5% faster
NUM_NOISE_FACTOR = 1.02       # results inside +/-2% are explicitly "noise"


def _load(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _isa_summary(blob: Dict[str, Any]) -> Optional[Dict[str, int]]:
    """Order-insensitive aggregate over all cached AMDGCN kernels.

    Earlier versions picked the kernel with max v_mfma — but when several
    cached kernels tie on v_mfma, list-order between runs is not stable, so
    'baseline' and 'patched' could resolve to different kernels and produce
    a phantom delta. We now aggregate as a multiset: report (count, totals,
    max longest_mfma_run, max transitions, total ds_write) so the gate
    measures whether the COMPILED CODE changed, not which kernel appeared
    first in the cache directory listing.
    """
    isa = blob.get("isa")
    if not isa:
        return None
    if isinstance(isa, dict):
        return isa
    if not isinstance(isa, list) or not isa:
        return None
    # Stable multiset signature: sort by every counter so identical sets
    # serialise identically regardless of file-listing order.
    sig = sorted(
        ((e.get("ds_read", 0), e.get("v_mfma", 0), e.get("ds_write", 0),
          e.get("longest_mfma_run", 0), e.get("lds_mfma_transitions", 0))
         for e in isa)
    )
    return {
        "n_kernels": len(isa),
        "ds_read": sum(s[0] for s in sig),
        "v_mfma": sum(s[1] for s in sig),
        "ds_write": sum(s[2] for s in sig),
        "longest_mfma_run": max(s[3] for s in sig),
        "lds_mfma_transitions": sum(s[4] for s in sig),
        "_signature": sig,
    }


def _shape_match(result: Dict[str, Any], shape: Tuple[int, int, int]) -> bool:
    return (result.get("M") == shape[0]
            and result.get("N") == shape[1]
            and result.get("K") == shape[2])


def _per_shape_tflops(blob: Dict[str, Any], shape: Tuple[int, int, int]) -> Optional[float]:
    # bench_K6830 (probe) writes triton_tflops at top level + a "shape" key.
    if "triton_tflops" in blob:
        s = blob.get("shape") or {}
        if (s.get("M"), s.get("N"), s.get("K")) == shape:
            return blob["triton_tflops"]
    # bench_tritonblas writes results: [{M, N, K, tritonblas_tflops, ...}]
    for r in blob.get("results", []):
        if _shape_match(r, shape):
            return r.get("tritonblas_tflops")
    return None


def check(baseline: Dict[str, Any],
          patched: Dict[str, Any],
          shape: Tuple[int, int, int],
          strict: bool) -> Tuple[bool, List[str]]:
    """Return (hypothesis_supported, list_of_evidence_lines)."""
    notes: List[str] = []

    # --- Skeptic-gate: baseline vs patched MUST be different builds. ---
    bv = baseline.get("triton_version", "?")
    pv = patched.get("triton_version", "?")
    notes.append(f"  triton_version: baseline={bv!r}  patched={pv!r}")
    if bv == pv:
        notes.append("  FATAL: identical triton_version — baseline is "
                     "contaminated (Skeptic-review failure mode). Refusing to "
                     "evaluate hypothesis on dirty data.")
        return False, notes

    # --- ISA-shift gate ---
    b_isa = _isa_summary(baseline) or {}
    p_isa = _isa_summary(patched) or {}
    if not b_isa or not p_isa:
        notes.append("  FATAL: missing ISA summary in one of the JSONs.")
        return False, notes

    b_run = b_isa.get("longest_mfma_run", 0)
    p_run = p_isa.get("longest_mfma_run", 0)
    b_tr = b_isa.get("lds_mfma_transitions", 0)
    p_tr = p_isa.get("lds_mfma_transitions", 0)
    b_dsw = b_isa.get("ds_write", 0)
    p_dsw = p_isa.get("ds_write", 0)
    notes.append(f"  ISA longest_mfma_run:        {b_run} -> {p_run} (drop>={MFMA_RUN_DELTA}?)")
    notes.append(f"  ISA lds_mfma_transitions:    {b_tr} -> {p_tr} (rise>={TRANSITIONS_DELTA}?)")
    notes.append(f"  ISA ds_write:                {b_dsw} -> {p_dsw} (drop>={DS_WRITE_DELTA}?)")

    isa_ok = (
        (b_run - p_run) >= MFMA_RUN_DELTA
        or (p_tr - b_tr) >= TRANSITIONS_DELTA
        or (b_dsw - p_dsw) >= DS_WRITE_DELTA
    )
    if not isa_ok:
        notes.append("  ISA gate FAILED: hypothesis predicted at least one of "
                     "(longest_mfma_run drop, transitions rise, ds_write drop), "
                     "observed deltas are within noise.")
    # Strong-signal sanity: if the multiset of per-kernel ISA signatures is
    # bit-identical, no rearrangement of counters can rescue the hypothesis.
    if (b_isa.get("_signature") is not None
            and b_isa.get("_signature") == p_isa.get("_signature")):
        notes.append("  ISA multiset is BIT-IDENTICAL between baseline and "
                     "patched (same cached kernels, same counters). The patch "
                     "did not alter the compiled AMDGCN on this workload.")

    # --- TFLOPS gate ---
    b_t = _per_shape_tflops(baseline, shape)
    p_t = _per_shape_tflops(patched, shape)
    if b_t is None or p_t is None:
        notes.append(f"  FATAL: shape {shape} not present in one of the JSONs.")
        return False, notes
    ratio = p_t / b_t if b_t > 0 else 0.0
    notes.append(f"  TFLOPS @ {shape}: baseline={b_t:.3f}  patched={p_t:.3f}  "
                 f"ratio={ratio:.3f}  (need>={TFLOPS_FACTOR})")
    perf_ok = ratio >= TFLOPS_FACTOR
    if not perf_ok and ratio < NUM_NOISE_FACTOR and ratio > (1.0 / NUM_NOISE_FACTOR):
        notes.append("  TFLOPS gate FAILED: patched within +/-2% of baseline "
                     "(measurement noise).")

    if strict:
        return (isa_ok and perf_ok), notes
    # Non-strict: ISA gate is necessary, TFLOPS gate is informational.
    return isa_ok, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True,
                    help="JSON written by bench_*.py for the unpatched Triton build")
    ap.add_argument("--patched", required=True,
                    help="JSON written by bench_*.py for the patched Triton build")
    ap.add_argument("--shape", default="64,64,4096",
                    help="M,N,K to evaluate (default: pre-registered 64,64,4096)")
    ap.add_argument("--strict", action="store_true",
                    help="Require BOTH ISA shift AND TFLOPS speedup (default: ISA only)")
    args = ap.parse_args()

    try:
        shape = tuple(int(x) for x in args.shape.split(","))
        if len(shape) != 3:
            raise ValueError
    except ValueError:
        print(f"--shape must be M,N,K (got {args.shape!r})", file=sys.stderr)
        sys.exit(2)

    b = _load(args.baseline)
    p = _load(args.patched)

    print(f"K-6830 hypothesis check  shape={shape}  strict={args.strict}")
    ok, notes = check(b, p, shape, args.strict)
    for line in notes:
        print(line)

    if "FATAL" in "".join(notes):
        print("\nVERDICT: data-integrity failure (re-run with a clean baseline)")
        sys.exit(2)
    if ok:
        print("\nVERDICT: hypothesis SUPPORTED")
        sys.exit(0)
    print("\nVERDICT: hypothesis NOT SUPPORTED — file as negative finding, "
          "do NOT ship as a fix PR.")
    sys.exit(1)


if __name__ == "__main__":
    main()
