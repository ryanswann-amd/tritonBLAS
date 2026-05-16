"""Unit tests for the per-shape dispatch table (K-4108).

The dispatch table is a small ``(M, N, K, dtype) -> ShapeConfig`` lookup
that the Origami selector consults before computing the StreamK grid /
workgroup mapping. These tests cover the three critical paths claimed in
the PR description:

1. Exact-match hit returns the recorded ``ShapeConfig`` unchanged.
2. Any miss (out-of-table shape, unsupported dtype) returns ``None`` so
   the Origami heuristic remains in charge — i.e. landing the table is
   guaranteed not to change behaviour for shapes it does not know about.
3. An LDS-budget-violating override is rejected by the selector wiring
   (``OrigamiMatmulSelector._apply_per_shape_override``) without crashing
   and without replacing the Origami pick.

The tests deliberately avoid GPU execution where possible: paths (1) and
(2) are pure-Python lookups; path (3) needs an OrigamiMatmulSelector
instance, which in turn needs an origami ``hardware_t`` — we obtain that
from ``torch.cuda`` when a GPU is present, otherwise the LDS test skips.
"""

from __future__ import annotations

import logging

import pytest
import torch

from tritonblas import dispatch_table as dt
from tritonblas.dispatch_table import (
    SHAPE_DISPATCH_TABLE,
    ShapeConfig,
    covered_shapes,
    lookup,
    lookup_torch,
)


# ---------------------------------------------------------------------------
# Path 1: exact-match hit
# ---------------------------------------------------------------------------


def test_lookup_exact_match_returns_recorded_shape_config():
    """Every key in SHAPE_DISPATCH_TABLE must be retrievable by lookup()."""
    assert SHAPE_DISPATCH_TABLE, "dispatch table must not be empty"
    for (m, n, k, dtype_str), expected in SHAPE_DISPATCH_TABLE.items():
        got = lookup(m, n, k, dtype_str)
        assert got is expected, (
            f"lookup({m},{n},{k},{dtype_str!r}) returned {got!r}, "
            f"expected {expected!r}"
        )
        assert isinstance(got, ShapeConfig)


def test_lookup_torch_resolves_dtype_for_known_shape():
    """The torch.dtype convenience wrapper must hit on a known FP16 shape."""
    # 3072x3072x3072 is a stable K-4437 entry.
    cfg = lookup_torch(3072, 3072, 3072, torch.float16)
    assert isinstance(cfg, ShapeConfig)
    assert cfg == lookup(3072, 3072, 3072, "f16")


def test_table_is_fp16_only_pending_bf16_followup():
    """BF16 deferral (PR scope): no bf16 rows ship in this PR.

    K-4539 measured only 7/10 cells agree between FP16 and BF16, so the
    BF16 mirror was dropped pending its own benchmark. This test will
    have to be updated when BF16 entries land.
    """
    dtypes_in_table = {dtype_str for *_, dtype_str in SHAPE_DISPATCH_TABLE}
    assert dtypes_in_table == {"f16"}, (
        f"unexpected dtypes in dispatch table: {dtypes_in_table}"
    )


# ---------------------------------------------------------------------------
# Path 2: miss returns None (out-of-table dispatch is byte-identical)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "m,n,k,dtype_str",
    [
        # Out-of-table FP16 shapes: must miss so Origami stays in charge.
        (4096, 11008, 4096, "f16"),   # K-4437 explicit "LOSS — keep current"
        (1024, 1024, 1024, "f16"),
        (768,  1024, 512,  "f16"),
        (33,   17,   513,  "f16"),
        # K-3019 small-shape winners — dropped from the table because their
        # grid-best tiles exceed MI300X 64 KB LDS cap. Must miss so the
        # selector keeps Origami's pick.
        (64,   64,   64,   "f16"),
        (128,  128,  128,  "f16"),
        (256,  256,  256,  "f16"),
        (512,  512,  512,  "f16"),
        # In-table M/N/K but unsupported dtype string: must miss (BF16/FP8).
        (3072, 3072, 3072, "bf16"),
        (4096, 256,  4096, "bf16"),
        (2560, 2560, 2560, "f8"),
    ],
)
def test_lookup_miss_returns_none(m, n, k, dtype_str):
    """Any (M, N, K, dtype) not in the table must return None."""
    assert lookup(m, n, k, dtype_str) is None


def test_lookup_torch_miss_returns_none_for_unmapped_torch_dtype():
    """A torch.dtype that is in dtype_to_str but not in the table must miss."""
    # 3072³ exists as FP16 but not BF16 (BF16 deferred to a follow-up PR).
    assert lookup_torch(3072, 3072, 3072, torch.bfloat16) is None


def test_lookup_torch_miss_for_unrecognized_dtype():
    """A torch.dtype not in OrigamiMatmulSelector.dtype_to_str must miss."""
    assert lookup_torch(64, 64, 64, torch.uint8) is None


def test_covered_shapes_matches_dispatch_table_keys():
    """covered_shapes() is the canonical key listing for downstream tools."""
    assert set(covered_shapes()) == set(SHAPE_DISPATCH_TABLE.keys())


# ---------------------------------------------------------------------------
# Path 3: LDS-budget-violating override is rejected with a warning.
# ---------------------------------------------------------------------------


# This needs a real OrigamiMatmulSelector, which requires a GPU device for
# origami.get_hardware_for_device(...). Skip cleanly when no GPU is present.
pytestmark_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="OrigamiMatmulSelector requires a CUDA/HIP device",
)


@pytestmark_gpu
def test_lds_violating_override_is_rejected_and_warned(monkeypatch, caplog):
    """A pathological dispatch entry must NOT replace the Origami pick.

    We monkey-patch dispatch_table.lookup so that a known shape (which
    Origami can handle) requests a tile that obviously overflows LDS at
    the requested num_stages. The selector should:
      - log a warning at WARNING level mentioning "rejected"
      - leave _dispatch_table_hit == False
      - leave _result.config.mt unchanged from Origami's pick

    This is also the historical fate of every K-3019 small-shape winner
    on MI300X (their grid-best tiles exceed the 64 KB LDS cap), so the
    behaviour exercised here is load-bearing for the current table.
    """
    from tritonblas.origami import OrigamiMatmulSelector

    m, n, k = 256, 256, 256
    a_dtype = torch.float16

    # First, run the selector with the table disabled to capture Origami's
    # untouched tile for this shape — that is our reference.
    monkeypatch.setattr(dt, "lookup", lambda *a, **kw: None)
    base = OrigamiMatmulSelector(
        m, n, k, a_dtype, a_dtype, a_dtype,
        torch.device("cuda", torch.cuda.current_device()),
    )
    baseline_tile = (base.block_m, base.block_n, base.block_k)
    baseline_ns = base.num_stages

    # Now request a pathological override: BM=BN=BK=512 at ns=4 will request
    # ~3 * (512*512*2 + 512*512*2) = ~3 MB of LDS on an FP16 problem, far
    # above any MI300X / gfx942 LDS capacity (~64-160 KB).
    pathological = ShapeConfig(
        block_m=512, block_n=512, block_k=512,
        num_stages=4, num_warps=8,
    )
    monkeypatch.setattr(dt, "lookup", lambda *a, **kw: pathological)

    with caplog.at_level(logging.WARNING, logger="tritonblas.origami"):
        sel = OrigamiMatmulSelector(
            m, n, k, a_dtype, a_dtype, a_dtype,
            torch.device("cuda", torch.cuda.current_device()),
        )

    # The override must have been rejected, NOT applied.
    assert sel._dispatch_table_hit is False
    assert (sel.block_m, sel.block_n, sel.block_k) == baseline_tile
    assert sel.num_stages == baseline_ns
    assert not hasattr(sel, "_override_num_warps"), (
        "override fields must not be set when the override is rejected"
    )

    # And a WARNING must have been emitted naming the shape + reason.
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("dispatch_table override rejected" in m for m in msgs), (
        f"expected a WARNING about override rejection, got: {msgs!r}"
    )


@pytestmark_gpu
def test_valid_override_is_applied():
    """A known-good in-table FP16 shape must produce _dispatch_table_hit=True."""
    from tritonblas.origami import OrigamiMatmulSelector

    # 3072³ FP16 is in the table with (256,128,64, ns=2, nw=8) — well within
    # LDS capacity on every gfx942/gfx950 part we ship for.
    expected = lookup(3072, 3072, 3072, "f16")
    assert expected is not None

    sel = OrigamiMatmulSelector(
        3072, 3072, 3072,
        torch.float16, torch.float16, torch.float16,
        torch.device("cuda", torch.cuda.current_device()),
    )

    assert sel._dispatch_table_hit is True
    assert (sel.block_m, sel.block_n, sel.block_k) == (
        expected.block_m, expected.block_n, expected.block_k,
    )
    assert sel.num_stages == expected.num_stages
    assert sel._override_num_warps == expected.num_warps
    assert sel._override_kpack == expected.kpack
