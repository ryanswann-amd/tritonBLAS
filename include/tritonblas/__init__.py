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
# (K-882's prototype lives in tests/ as a regression fixture for the harness;
# it carries a NO-LAND verdict and therefore intentionally does not ship here.)
