"""K-4449 guard-predicate unit tests.

The K-4449 fix-up in tritonblas/matmul.py overrides Origami's BLK_N=256
selection back down to BLK_N=128 (+ kpack=2 + waves_per_eu=1) for a narrow
band of cube-ish shapes that under-utilize the MI300X's compute units.

These tests pin the predicate behavior so a future refactor of the
trigger condition (BLK_M == BLK_N == 256, N % 128 == 0, doubled grid fits
one wave, TB_K4449_DISABLE not set) cannot silently regress without a
test failure.

The predicate is exposed as a pure function `_k4449_should_override_blkn`
in tritonblas.matmul so it can be unit-tested without invoking the GEMM
kernel.
"""

import os
import pytest

from tritonblas.matmul import _k4449_should_override_blkn


# MAX_SMS pinned to the MI300X compute-unit count.  Passed explicitly so
# the test is portable (passes on any GPU and on CPU-only environments
# where the module can be imported).
MI300X_MAX_SMS = 0x130


# ---------------------------------------------------------------------------
# (a) TB_K4449_DISABLE escape hatch restores baseline tile selection
# ---------------------------------------------------------------------------


def _clear_disable_env(monkeypatch):
    monkeypatch.delenv("TB_K4449_DISABLE", raising=False)


def test_disable_env_var_returns_false_on_triggering_shape(monkeypatch):
    """With TB_K4449_DISABLE=1, the predicate must NOT fire even when all
    other guard conditions are met. This is the documented rollback /
    A-B escape hatch."""
    monkeypatch.setenv("TB_K4449_DISABLE", "1")
    # 2304^3 with the 256x256 macro is the canonical winner from the A-B
    # sweep; without the disable env it returns True.
    assert _k4449_should_override_blkn(
        BLK_M=256, BLK_N=256, M=2304, N=2304, max_sms=MI300X_MAX_SMS
    ) is False


def test_disable_env_var_unset_allows_override(monkeypatch):
    """Sanity counterpart: with the env var explicitly unset, the same
    shape DOES trigger.  This pins the disable behaviour as the only
    reason the prior test returned False."""
    _clear_disable_env(monkeypatch)
    assert _k4449_should_override_blkn(
        BLK_M=256, BLK_N=256, M=2304, N=2304, max_sms=MI300X_MAX_SMS
    ) is True


# ---------------------------------------------------------------------------
# (b) Guard predicate fires for exactly the documented band
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [2304, 2560, 2816, 3072])
def test_triggers_on_documented_winners(dim, monkeypatch):
    """The K-4396 / K-4449 sweep showed >5% wins at 2304, 2560, 2816, 3072
    cube shapes.  All four must fire the override."""
    _clear_disable_env(monkeypatch)
    assert _k4449_should_override_blkn(
        BLK_M=256, BLK_N=256, M=dim, N=dim, max_sms=MI300X_MAX_SMS
    ) is True


@pytest.mark.parametrize("dim", [3584, 4096, 8192])
def test_skips_when_doubled_grid_overflows_one_wave(dim, monkeypatch):
    """For dims >= 3328 the doubled grid (cdiv(M,256)*cdiv(N,128)) exceeds
    MAX_SMS=MI300X_MAX_SMS and spills to a 2nd wave; the predicate MUST skip to
    avoid the regression observed without this guard (-34% at 3584^3)."""
    _clear_disable_env(monkeypatch)
    assert _k4449_should_override_blkn(
        BLK_M=256, BLK_N=256, M=dim, N=dim, max_sms=MI300X_MAX_SMS
    ) is False


def test_skips_when_blk_m_not_256(monkeypatch):
    """If Origami didn't pick the 256x256 macro-tile, no override."""
    _clear_disable_env(monkeypatch)
    assert _k4449_should_override_blkn(
        BLK_M=128, BLK_N=256, M=2304, N=2304, max_sms=MI300X_MAX_SMS
    ) is False


def test_skips_when_blk_n_not_256(monkeypatch):
    """Symmetric: if Origami already picked BLK_N=128, nothing to do."""
    _clear_disable_env(monkeypatch)
    assert _k4449_should_override_blkn(
        BLK_M=256, BLK_N=128, M=2304, N=2304, max_sms=MI300X_MAX_SMS
    ) is False


def test_skips_when_n_not_divisible_by_128(monkeypatch):
    """N % 128 != 0 is unsafe for the smaller N tile (alignment)."""
    _clear_disable_env(monkeypatch)
    assert _k4449_should_override_blkn(
        BLK_M=256, BLK_N=256, M=2304, N=2304 + 64, max_sms=MI300X_MAX_SMS
    ) is False


def test_boundary_doubled_grid_exactly_max_sms(monkeypatch):
    """The guard uses `<= MAX_SMS`, so the boundary case (doubled grid
    exactly fills one wave) MUST be a fire.  This pins the boundary."""
    _clear_disable_env(monkeypatch)
    # Pick (M, N) so that cdiv(M,256)*cdiv(N,128) == MAX_SMS exactly.
    # cdiv(2304,256)=9, cdiv(4352,128)=34, 9*34=306 -> too big.
    # cdiv(2048,256)=8, cdiv(4864,128)=38, 8*38=MI300X_MAX_SMS exactly.
    assert _k4449_should_override_blkn(
        BLK_M=256, BLK_N=256, M=2048, N=4864, max_sms=MI300X_MAX_SMS
    ) is True


def test_boundary_doubled_grid_one_over_max_sms(monkeypatch):
    """One tile over the wave boundary MUST skip."""
    _clear_disable_env(monkeypatch)
    # cdiv(2304,256)=9, cdiv(4352,128)=34, 9*34=306 > MI300X_MAX_SMS.
    assert _k4449_should_override_blkn(
        BLK_M=256, BLK_N=256, M=2304, N=4352, max_sms=MI300X_MAX_SMS
    ) is False
