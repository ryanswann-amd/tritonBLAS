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
# NOTE: GuardedOverrides are registered through the ``overrides`` subpackage
# below.  Each override is OFF by default (env-var killswitch must be flipped
# to fire) so this is fail-closed at import time — the registry singleton
# carries the gates but ``OverrideRegistry.apply(...)`` returns None until the
# operator explicitly arms the corresponding env var.  Concrete overrides
# must have a LAND verdict from ``run_falsification`` before being added.
# K-882's NO-LAND prototype intentionally lives in tests/ and never ships here.
from . import overrides as _overrides  # noqa: F401  (registers K-935 gate)
