"""K-1793 regression tests for the N=192 K-COMPLEMENT hipBLASLt route-OUT gate.

The gate routes 18 verified-winning cells to torch.matmul (hipBLASLt):
    M in {2048, 4096, 8192} x N==192 x K in {4096, 8192, 16384} x dtype in {fp16, bf16}

These tests cover:
  - admit-set membership (all 18 cells fire; out-of-band cells stay unrouted)
  - the documented `enable_streamk=True` carve-out (gate is BYPASSED — caller
    explicitly requested stream-K and must not be silently re-routed)
  - disjointness from prior K-COMPLEMENT alias-stack slots

The streamk carve-out test exercises `_matmul` / `_matmul_out` end-to-end and
asserts the gate's route-OUT target (`torch.matmul`) is NOT called when the
caller passes `enable_streamk=True`. A GPU correctness/perf check is performed
by the validation harness in mc2-workspaces/K-1793/scripts/k1793_paired_n30.py.
"""
import itertools
from unittest import mock

import pytest
import torch

from tritonblas.matmul import _K1793_GATE_CELLS


# ------ admit-set cardinality / structure ------

def test_k1793_admit_set_cardinality():
    """The K-1793 admit set covers exactly 18 cells (3 M x 1 N x 3 K x 2 dtype)."""
    assert len(_K1793_GATE_CELLS) == 18


def test_k1793_admit_set_axes():
    """Pin the M / N / K / dtype projections against accidental drift."""
    assert {c[0] for c in _K1793_GATE_CELLS} == {2048, 4096, 8192}
    assert {c[1] for c in _K1793_GATE_CELLS} == {192}
    assert {c[2] for c in _K1793_GATE_CELLS} == {4096, 8192, 16384}
    assert {c[3] for c in _K1793_GATE_CELLS} == {torch.float16, torch.bfloat16}


# ------ exclusion: adjacent N values stay unrouted ------

@pytest.mark.parametrize("N", [16, 32, 48, 64, 80, 96, 128, 160, 224, 256, 512, 1024])
def test_k1793_adjacent_N_not_routed(N):
    """Adjacent N values (especially N=128 / N=256) must NOT trigger K-1793.

    Those bands are owned by upstream alias-stack slots (K-1685 P28 N=128,
    K-1538 P22 N=256). K-1793's gate is N==192 ONLY — widening would shadow.
    """
    for M, K, dt in itertools.product(
        (2048, 4096, 8192), (4096, 8192, 16384), (torch.float16, torch.bfloat16)
    ):
        assert (M, N, K, dt) not in _K1793_GATE_CELLS


# ------ exclusion: out-of-band M / K / dtype stay unrouted ------

@pytest.mark.parametrize("M", [512, 1024, 1536, 3072, 6144, 16384])
def test_k1793_out_of_band_M_not_routed(M):
    for K in (4096, 8192, 16384):
        for dt in (torch.float16, torch.bfloat16):
            assert (M, 192, K, dt) not in _K1793_GATE_CELLS


@pytest.mark.parametrize("K", [256, 512, 1024, 2048, 3072, 12288, 32768, 65536])
def test_k1793_out_of_band_K_not_routed(K):
    for M in (2048, 4096, 8192):
        for dt in (torch.float16, torch.bfloat16):
            assert (M, 192, K, dt) not in _K1793_GATE_CELLS


@pytest.mark.parametrize("dt", [torch.float32, torch.float64, torch.int8])
def test_k1793_out_of_band_dtype_not_routed(dt):
    for M in (2048, 4096, 8192):
        for K in (4096, 8192, 16384):
            assert (M, 192, K, dt) not in _K1793_GATE_CELLS


# ------ membership matrix: every admitted cell flips True ------

@pytest.mark.parametrize("M", [2048, 4096, 8192])
@pytest.mark.parametrize("K", [4096, 8192, 16384])
@pytest.mark.parametrize("dt", [torch.float16, torch.bfloat16])
def test_k1793_admitted_cells_route(M, K, dt):
    assert (M, 192, K, dt) in _K1793_GATE_CELLS


# ------ disjointness from K-1764 (M=N=4096 mid-K square cohort) ------

def test_k1793_disjoint_from_K1764_admit_set():
    """K-1764 fires only at N==4096; K-1793 only at N==192. Trivially disjoint."""
    assert {c[1] for c in _K1793_GATE_CELLS} == {192}


# ------ enable_streamk=True CARVE-OUT (documented exclusion clause) ------
#
# The gate's exclusion clause says: when the caller passes enable_streamk=True
# they have explicitly requested the stream-K kernel and the route-OUT MUST
# be bypassed. This is the critical branch the Testing Zealot review flagged
# as un-pinned. We exercise the actual `_matmul` / `_matmul_out` entry points
# and assert the route-OUT target (torch.matmul) is NOT called.

def _routed_cell_args():
    """A (M, N, K, dt) cell that IS in the K-1793 admit set, on CPU.

    CPU tensors are sufficient: the gate predicate only inspects `.shape` and
    `.dtype`. If the gate fires it returns immediately via `torch.matmul`. If
    it does NOT fire, execution proceeds into the selector / triton kernel
    path which has no CPU implementation and raises — and that downstream
    raise is the *proof* the gate was bypassed.

    Using `torch.empty` (un-initialised storage) keeps allocation cheap.
    """
    M, N, K, dt = 4096, 192, 8192, torch.float16
    a = torch.empty((M, K), dtype=dt)
    b = torch.empty((K, N), dtype=dt)
    return a, b


def test_k1793_streamk_carveout_bypasses_gate_in_matmul():
    """enable_streamk=True must bypass the route-OUT in `_matmul`.

    If the gate erroneously fires under streamk, `torch.matmul` (the HBL
    route-OUT target) is the FIRST thing called — we monkey-patch it to a
    sentinel and assert it is NOT touched. Any downstream exception is fine
    because it proves we proceeded past the gate into the streamk path.
    """
    from tritonblas.matmul import _matmul

    a, b = _routed_cell_args()
    sentinel = mock.Mock(side_effect=RuntimeError("K-1793 gate FIRED under streamk"))
    with mock.patch("tritonblas.matmul.torch.matmul", new=sentinel):
        with pytest.raises(Exception) as ei:
            _matmul(a, b, enable_streamk=True)
    assert not sentinel.called, (
        "K-1793 gate fired under enable_streamk=True (carve-out broken)"
    )
    # The exception must be a downstream failure (not the sentinel),
    # proving execution proceeded past the gate.
    assert "K-1793 gate FIRED" not in str(ei.value)


def test_k1793_streamk_carveout_bypasses_gate_in_matmul_out():
    """Same carve-out, mirrored on the `_matmul_out` (out= variant) entry."""
    from tritonblas.matmul import _matmul_out

    a, b = _routed_cell_args()
    out = torch.empty((4096, 192), dtype=torch.float16)
    sentinel = mock.Mock(side_effect=RuntimeError("K-1793 gate FIRED under streamk"))
    with mock.patch("tritonblas.matmul.torch.matmul", new=sentinel):
        with pytest.raises(Exception) as ei:
            _matmul_out(a, b, out, enable_streamk=True)
    assert not sentinel.called, (
        "K-1793 gate fired under enable_streamk=True (out= variant carve-out broken)"
    )
    assert "K-1793 gate FIRED" not in str(ei.value)


def test_k1793_default_path_DOES_fire_gate_in_matmul():
    """Positive control: with default enable_streamk=False on a routed cell,
    the gate MUST fire — `torch.matmul` IS called and its return propagated.

    This bookends the carve-out tests by proving we are exercising the right
    branch: same call, only `enable_streamk` differs, and behaviour flips.
    The torch.matmul stub returns a real Tensor of the expected output shape
    so the @triton_op dispatcher can re-cast the result.
    """
    from tritonblas.matmul import _matmul

    a, b = _routed_cell_args()
    sentinel_out = torch.full((a.shape[0], b.shape[1]), 1.234, dtype=a.dtype)
    fake = mock.Mock(return_value=sentinel_out)
    with mock.patch("tritonblas.matmul.torch.matmul", new=fake):
        result = _matmul(a, b, enable_streamk=False)
    assert fake.called, "K-1793 gate did NOT fire on a routed cell with streamk=False"
    # Tensor identity is not preserved across the dispatcher boundary, so we
    # check value equality on the sentinel marker instead.
    assert result.shape == sentinel_out.shape
    assert torch.equal(result, sentinel_out), (
        "Gate fired but did not propagate torch.matmul's output value"
    )


def test_k1793_default_path_DOES_fire_gate_in_matmul_out():
    """Same positive control on the out= variant."""
    from tritonblas.matmul import _matmul_out

    a, b = _routed_cell_args()
    out = torch.empty((4096, 192), dtype=torch.float16)

    def _fake_mm(x, y, out=None):
        if out is not None:
            out.fill_(2.345)
        return out

    fake = mock.Mock(side_effect=_fake_mm)
    with mock.patch("tritonblas.matmul.torch.matmul", new=fake):
        ret = _matmul_out(a, b, out, enable_streamk=False)
    assert fake.called, "K-1793 gate did NOT fire (out=) on a routed cell with streamk=False"
    assert ret is None
    _, kwargs = fake.call_args
    assert kwargs.get("out") is out
    # Confirm the gate actually wrote into `out` via torch.matmul(out=out).
    assert torch.all(out == 2.345)
