"""K-949 paired-falsification driver --- cohort A LDS-pressure mitigation.

Reproduces the K-949 documented NO-LAND finding (see PR description /
reports/prs/K-949-pr.md). Constructs a GuardedOverride for cohort A
(M=N=1024, K in {16384, 32768}, fp16/bf16 / kpack=2 + num_stages=2 +
num_warps=8) ENTIRELY INSIDE THIS SCRIPT and runs the K-901 9-layer
guarded-override harness on it as a pre-merge falsification check.
The override is NEVER registered in the production package
(``tritonblas.overrides`` does not import this module) --- importing
``tritonblas`` from production code does not change behaviour for any
cohort A cell.

Why this script exists
----------------------
K-905 mechanistically root-caused cohort A as LDS-bound:
``SQ_LDS_BANK_CONFLICT`` 8.39 M cyc/disp vs hipBLASLt 0,
``SQ_WAIT_INST_LDS`` 41x hipBLASLt, 8x ``ds_read`` pressure per K-step.
The kpack=2 lever halves the LDS-issue-slot pressure on rocprofv2
PMC counters (verified K-923 anchor cell: 8.39M -> 4.19M cyc/disp,
0.50x), but the wall-clock delta is essentially zero because
Triton-AMD does NOT actually emit ``ds_read_b128`` even when kpack=2
widens the logical fragment (codegen-gap class K-878 sec.3 / K-905 R5 /
K-877 R1 / K-854 / K-864 / K-935 / **K-949**, 5th confirmation).

Verdict (n=30 paired, MI300X / gfx942):

  V1 in-cohort lift   geomean ON/OFF = 0.9960x  (FAIL >=1.05x gate)
  V2 OOC bleed        2 nominal regressors, both fired=0 (no causation)
  V3 routing trace    4 intended fires, 0 OOC fires (PASS)
  Brief OOT geomean   1.0010x in [0.98, 1.02]  (PASS)

Per the four reviewer votes on the previous attempt (Testing Zealot,
Pragmatist, Minimalist, Skeptic), the override is NOT shipped as a
production registration. This driver is the reproducer for the
negative result --- run it any time Triton-AMD ships a new release to
re-bench the cohort and verify whether the codegen gap has closed.

Usage
-----
  python scripts/run_falsification_k949.py [--quick] [--out-dir DIR]

Outputs to output/k949/:
  per_cell.csv, routing.csv, verdict.json, REPORT.md
"""

import argparse
import os
import sys

import torch

# Ensure local checkout wins over any installed package.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "include"))

import tritonblas  # noqa: F401
from tritonblas.guarded_override import (
    GuardedOverride,
    OverrideRegistry,
    run_falsification,
)


# ---------------------------------------------------------------------------
# K-949 cohort A LDS-pressure override --- DEFINED LOCALLY HERE.
# Not imported from tritonblas.overrides on purpose: the production
# package must NOT carry this gate (NO-LAND verdict; see module docstring).
# ---------------------------------------------------------------------------

K949_COHORT_KEYS = [
    (1024, 1024, 16384, torch.float16),
    (1024, 1024, 16384, torch.bfloat16),
    (1024, 1024, 32768, torch.float16),
    (1024, 1024, 32768, torch.bfloat16),
]


def _k949_payload(M, N, K, dtype):
    # kpack=2  --> would lower ds_read_b64 -> ds_read_b128 IF Triton-AMD
    #              actually emitted the wider load (it does not, K-905 R5).
    # num_stages=2 --> defensive pin against future Origami drift to NS=3
    #                  which would 2x the LDS footprint and revive
    #                  bank-conflict pressure.
    # num_warps=8 --> matches the K-935 prototype config.
    return {"kpack": 2, "num_stages": 2, "num_warps": 8}


def _build_k949_gate() -> GuardedOverride:
    return GuardedOverride(
        ticket="K-949",
        env_var="TB_K949_ENABLE",
        cohort_keys=list(K949_COHORT_KEYS),
        dtype_allowlist=(torch.float16, torch.bfloat16),
        override_fn=_k949_payload,
    )


# ---------------------------------------------------------------------------
# Falsification panel
# ---------------------------------------------------------------------------

# In-cohort: 4 keys.
COHORT = [
    {"M": 1024, "N": 1024, "K": 16384, "dtype": "fp16"},
    {"M": 1024, "N": 1024, "K": 16384, "dtype": "bf16"},
    {"M": 1024, "N": 1024, "K": 32768, "dtype": "fp16"},
    {"M": 1024, "N": 1024, "K": 32768, "dtype": "bf16"},
]

# 50-cell OOT leakage suite --- bounding-box neighbours of cohort A
# (varying +/-M, +/-N, +/-K, +/-dtype) plus K-832 top production shapes
# that K-919 flagged as structurally similar to cohort A, plus K-790
# 92-shape gap-profile small-K square neighbours.
OOT = [
    # +/-K axis at M=N=1024
    {"M": 1024, "N": 1024, "K":  8192, "dtype": "fp16"},
    {"M": 1024, "N": 1024, "K":  8192, "dtype": "bf16"},
    {"M": 1024, "N": 1024, "K": 12288, "dtype": "fp16"},
    {"M": 1024, "N": 1024, "K": 12288, "dtype": "bf16"},
    {"M": 1024, "N": 1024, "K": 24576, "dtype": "fp16"},
    {"M": 1024, "N": 1024, "K": 24576, "dtype": "bf16"},
    {"M": 1024, "N": 1024, "K": 49152, "dtype": "fp16"},
    {"M": 1024, "N": 1024, "K": 49152, "dtype": "bf16"},
    # +/-M axis
    {"M":  512, "N": 1024, "K": 16384, "dtype": "fp16"},
    {"M":  512, "N": 1024, "K": 16384, "dtype": "bf16"},
    {"M":  768, "N": 1024, "K": 16384, "dtype": "fp16"},
    {"M":  768, "N": 1024, "K": 16384, "dtype": "bf16"},
    {"M": 2048, "N": 1024, "K": 16384, "dtype": "fp16"},
    {"M": 2048, "N": 1024, "K": 16384, "dtype": "bf16"},
    {"M":  512, "N": 1024, "K": 32768, "dtype": "fp16"},
    {"M":  512, "N": 1024, "K": 32768, "dtype": "bf16"},
    {"M": 2048, "N": 1024, "K": 32768, "dtype": "fp16"},
    # +/-N axis
    {"M": 1024, "N":  512, "K": 16384, "dtype": "fp16"},
    {"M": 1024, "N":  512, "K": 16384, "dtype": "bf16"},
    {"M": 1024, "N":  768, "K": 16384, "dtype": "fp16"},
    {"M": 1024, "N": 2048, "K": 16384, "dtype": "fp16"},
    {"M": 1024, "N": 2048, "K": 16384, "dtype": "bf16"},
    {"M": 1024, "N":  512, "K": 32768, "dtype": "fp16"},
    {"M": 1024, "N": 2048, "K": 32768, "dtype": "fp16"},
    # Square M=N=512 (K-866 LANDed sister-cohort)
    {"M":  512, "N":  512, "K": 16384, "dtype": "fp16"},
    {"M":  512, "N":  512, "K": 16384, "dtype": "bf16"},
    {"M":  512, "N":  512, "K": 32768, "dtype": "fp16"},
    {"M":  512, "N":  512, "K": 32768, "dtype": "bf16"},
    # Square M=N=2048 (K-882 sister-cohort)
    {"M": 2048, "N": 2048, "K": 16384, "dtype": "fp16"},
    {"M": 2048, "N": 2048, "K": 16384, "dtype": "bf16"},
    # K-832 top production shapes flagged by K-919 bounding-box
    {"M":  256, "N": 1024, "K": 16384, "dtype": "fp16"},
    {"M":  256, "N": 1024, "K": 16384, "dtype": "bf16"},
    {"M": 4096, "N": 1024, "K": 16384, "dtype": "fp16"},
    {"M": 1024, "N": 4096, "K": 16384, "dtype": "fp16"},
    {"M": 1536, "N": 1024, "K": 16384, "dtype": "fp16"},
    {"M": 1024, "N": 1536, "K": 16384, "dtype": "fp16"},
    {"M": 1280, "N": 1024, "K": 16384, "dtype": "fp16"},
    {"M": 1024, "N": 1280, "K": 16384, "dtype": "fp16"},
    {"M":  640, "N": 1024, "K": 16384, "dtype": "fp16"},
    {"M": 1024, "N":  640, "K": 16384, "dtype": "fp16"},
    {"M":  896, "N": 1024, "K": 16384, "dtype": "fp16"},
    {"M": 1024, "N":  896, "K": 16384, "dtype": "fp16"},
    # K-790 92-shape gap profile sample (small-K square neighbours)
    {"M": 1024, "N": 1024, "K":  4096, "dtype": "fp16"},
    {"M": 1024, "N": 1024, "K":  4096, "dtype": "bf16"},
    {"M": 1024, "N": 1024, "K":  6144, "dtype": "fp16"},
    {"M": 1024, "N": 1024, "K":  2048, "dtype": "fp16"},
    {"M": 1024, "N": 1024, "K":  1024, "dtype": "fp16"},
    {"M": 1024, "N": 1024, "K":   512, "dtype": "fp16"},
    {"M": 1024, "N": 1024, "K":   256, "dtype": "fp16"},
    # Mid-rect (K-849 cohort)
    {"M": 1536, "N": 2048, "K": 16384, "dtype": "fp16"},
    {"M": 2048, "N": 1536, "K": 16384, "dtype": "fp16"},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="output/k949")
    ap.add_argument("--n-warm", type=int, default=5)
    ap.add_argument("--n-capture", type=int, default=30)
    ap.add_argument("--n-rounds", type=int, default=3)
    ap.add_argument("--quick", action="store_true",
                    help="Smoke-test mode: 1 round, n_capture=10")
    ap.add_argument("--no-ref", action="store_true",
                    help="Skip hipBLASLt reference measurement")
    ap.add_argument("--cohort-only", action="store_true",
                    help="Skip OOT panel (faster smoke test)")
    args = ap.parse_args()

    if args.quick:
        args.n_capture = 10
        args.n_rounds = 1

    if not torch.cuda.is_available():
        print("CUDA/ROCm not available --- aborting")
        sys.exit(2)

    # Build the gate locally and register it with the in-process registry
    # for the duration of this driver. The production tritonblas package
    # NEVER imports this module, so this registration is invisible to any
    # other consumer of the library. Arm the env-var killswitch in-process
    # so layer L3 lets the gate fire during the bench.
    gate = _build_k949_gate()
    OverrideRegistry.register(gate)
    os.environ["TB_K949_ENABLE"] = "1"

    # Dedupe: cohort cells stay first; OOT must not contain any cohort key.
    cohort_set = {(s["M"], s["N"], s["K"], s["dtype"]) for s in COHORT}
    oot_seen = set()
    oot_clean = []
    for s in OOT:
        key = (s["M"], s["N"], s["K"], s["dtype"])
        if key in cohort_set or key in oot_seen:
            continue
        oot_seen.add(key)
        oot_clean.append(s)

    shapes = list(COHORT) + (oot_clean if not args.cohort_only else [])
    print(f"K-949 falsification: {len(COHORT)} in-cohort + "
          f"{len(shapes) - len(COHORT)} OOT cells")
    print(f"  n_warm={args.n_warm} n_capture={args.n_capture} "
          f"n_rounds={args.n_rounds}")

    verdict = run_falsification(
        override=gate,
        shapes=shapes,
        out_dir=args.out_dir,
        n_warm=args.n_warm,
        n_capture=args.n_capture,
        n_rounds=args.n_rounds,
        measure_ref=not args.no_ref,
    )
    print(verdict.report)
    # Exit non-zero on NO-LAND so CI can gate on it.
    sys.exit(0 if verdict.land else 1)


if __name__ == "__main__":
    main()
