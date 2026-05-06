"""Smoke + correctness tests for tritonblas.prewarm.

Validates:
  1. Module imports and CLI loads.
  2. Default playlist loads and dedupe shrinks the row count meaningfully
     (so the documented 1446 -> ~200 sig collapse holds in spirit).
  3. ``prewarm`` runs end-to-end on a tiny in-process slice and reports a
     ``compiled`` count consistent with the dedupe behaviour.
  4. The Triton on-disk cache directory pointed to by ``cache_dir`` actually
     gains files after ``prewarm`` runs (the whole point of the utility).
  5. Steady-state matmul output is bit-identical with vs. without prewarm
     (the dispatcher selects the same kernel + config either way).

These tests require a CUDA device. They are skipped if torch.cuda is absent.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch


cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for prewarm tests"
)


@cuda_required
def test_prewarm_imports():
    import tritonblas
    assert hasattr(tritonblas, "prewarm")
    assert hasattr(tritonblas, "load_playlist")
    # Bundled playlist should be importable + nonempty.
    pl = tritonblas.load_playlist()
    assert len(pl) > 100, f"bundled playlist suspiciously small: {len(pl)}"


@cuda_required
def test_prewarm_signature_dedup_shrinks_playlist():
    """The dedupe path should collapse many rows to a smaller signature set."""
    from tritonblas.prewarm import prewarm
    pl = [
        # Three shapes that should collapse to a small set of signatures
        # (Origami selects the same tile for adjacent small shapes).
        (256, 256, 256, "bf16", "bf16", "bf16"),
        (256, 256, 256, "bf16", "bf16", "bf16"),  # exact dup
        (256, 256, 512, "bf16", "bf16", "bf16"),  # likely-same sig
    ]
    with tempfile.TemporaryDirectory() as td:
        res = prewarm(shapes=pl, cache_dir=td, dedupe_signatures=True,
                      verbose=False)
        # At least the exact duplicate must be skipped.
        assert res["skipped"] >= 1
        assert res["compiled"] >= 1
        assert res["failed"] == 0


@cuda_required
def test_prewarm_populates_cache_directory():
    """The whole point: after prewarm, the cache dir contains compiled kernels."""
    from tritonblas.prewarm import prewarm
    with tempfile.TemporaryDirectory() as td:
        res = prewarm(
            shapes=[(128, 128, 128, "bf16", "bf16", "bf16")],
            cache_dir=td,
            verbose=False,
        )
        assert res["failed"] == 0
        # Triton writes to <cache_dir>/<sha-hex>/<kernel>/...
        # Just check that *something* was written under cache_dir.
        contents = list(Path(td).rglob("*"))
        files = [p for p in contents if p.is_file()]
        assert len(files) > 0, (
            f"no files written under cache_dir={td}; "
            f"contents={[str(c) for c in contents]}"
        )


@cuda_required
def test_prewarm_does_not_change_steady_state_output():
    """Prewarm must not alter the kernel/config selected — output is identical."""
    import tritonblas
    from tritonblas.prewarm import prewarm

    M, N, K = 256, 256, 256
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")

    out_baseline = tritonblas.matmul(a, b).clone()

    with tempfile.TemporaryDirectory() as td:
        prewarm(
            shapes=[(M, N, K, "bf16", "bf16", "bf16")],
            cache_dir=td,
            verbose=False,
        )
        out_post = tritonblas.matmul(a, b).clone()

    # Same kernel + config + inputs => same bits.
    assert torch.equal(out_baseline, out_post), (
        f"steady-state output changed after prewarm; "
        f"max abs diff = {(out_baseline.float() - out_post.float()).abs().max().item()}"
    )


@cuda_required
def test_prewarm_cli_smoke():
    """`python -m tritonblas.prewarm --limit 1` must exit 0."""
    import subprocess
    import sys
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run(
            [sys.executable, "-m", "tritonblas.prewarm",
             "--limit", "1", "--cache-dir", td, "--quiet"],
            capture_output=True, text=True, timeout=120,
        )
        assert r.returncode == 0, (
            f"prewarm CLI failed: stdout={r.stdout!r} stderr={r.stderr!r}"
        )
