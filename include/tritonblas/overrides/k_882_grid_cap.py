"""K-882 — env-gated narrow override for M=N=2048 K in {4096,8192,16384} fp16/bf16.

Hypothesis (K-882): on the M=N=2048 small-K residual cohort, capping the
persistent-wave grid to N_CU and forcing num_stages=2 reduces per-dispatch
launch-overhead.

K-882's standalone prototype reached NO-LAND verdict because the override is
structurally a no-op: Origami's selector already picks NS=2 and total_tiles=256
< N_CU=304, so min(total_tiles, N_CU) does not reduce the grid.  K-901 re-wires
the same prototype through the K-883/K-901 guarded-override harness, where the
NO-LAND verdict can be re-validated mechanically across the K-790 92-shape gap
profile and K-832 production sample.

The gate is OFF by default (env var TB_K882_ENABLE not set) so this code is
dead in production until a CI / dev box explicitly enables it for a paired
audit.  The guarded-override harness's L3 (env-read-per-call) makes that
toggle visible to in-process paired sweeps.
"""

import torch

from ..guarded_override import GuardedOverride, OverrideRegistry


def _k882_payload(M, N, K, dtype):
    return {
        "grid_cap_to_n_cu": True,
        "num_stages": 2,
    }


GUARD = GuardedOverride(
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
    override_fn=_k882_payload,
)
OverrideRegistry.register(GUARD)
