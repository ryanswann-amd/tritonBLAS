from .matmul import matmul, matmul_a8w8
from .matmul import matmul_lt, matmul_a8w8_lt
from .matmul import matmul_fp4
from .matmul import addmm
from .config import MatmulConfig, matmul_preamble
from .bench import do_bench
from .origami import OrigamiMatmulSelector
from .guarded_override import (
    GuardedOverride,
    OverrideRegistry,
    FalsificationVerdict,
    run_falsification,
    measure_cell,
    apply_override_in_dispatcher,
    set_arch_lds_budget,
    lds_budget_for,
)
# NOTE: no GuardedOverride is registered in the production package by default.
# The harness is the deliverable; concrete overrides must register themselves
# explicitly only after passing the K-883 §5 LAND verdict via run_falsification.
# Two NO-LAND prototypes (K-882 grid-cap, K-935 cohort-A LDS-mitigation) live
# only in tests/ and scripts/ as regression fixtures and never ship here, per
# K-901 M2 (fail-closed semantics require package-deletion, not disable-flags).
