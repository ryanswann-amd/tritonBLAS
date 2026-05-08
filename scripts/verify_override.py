"""Sanity-check that the K-967 guarded override fires correctly."""
import os
import torch

# This must be set BEFORE import to be tested for env-killswitch behavior
os.environ["TB_K967_PROTO"] = "1"
import tritonblas
from tritonblas.matmul import _tb_k967_overrides

cases = [
    ("PROTO=1", 14208, 2048, 1024, torch.bfloat16, 1),    # in-cohort
    ("PROTO=1", 14208, 2048, 1024, torch.float16, None),  # wrong dtype
    ("PROTO=1",  6016, 2048, 1024, torch.bfloat16, None), # OOC neighbor
    ("PROTO=1", 14207, 2048, 1024, torch.bfloat16, None), # off-by-1
]
for tag, M, N, K, dt, expected in cases:
    got = _tb_k967_overrides(M, N, K, dt)
    ok = "OK" if got == expected else "FAIL"
    print(f"{ok} {tag} ({M},{N},{K}) {dt} -> {got} (expected {expected})")

# Killswitch test
os.environ["TB_K967_PROTO"] = "0"
got = _tb_k967_overrides(14208, 2048, 1024, torch.bfloat16)
print(f"{'OK' if got is None else 'FAIL'} PROTO=0 (14208,2048,1024) bf16 -> {got} (expected None)")
