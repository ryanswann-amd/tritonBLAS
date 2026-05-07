"""Unit tests for the dual-MFMA-shape selection knob (16x16 vs 32x32).

Tests are pure-Python (no GPU required); they verify the predicate
in `tritonblas.origami.select_mfma_instr_size` behaves correctly under
all override / arch / dtype / tile combinations.
"""
import os
import pytest

from tritonblas.origami import (
    select_mfma_instr_size,
    _parse_mfma_override,
)


class TestParseOverride:
    def test_none_returns_none(self):
        assert _parse_mfma_override(None) is None

    def test_empty_string_returns_none(self):
        assert _parse_mfma_override("") is None

    def test_int_16(self):
        assert _parse_mfma_override(16) == 16

    def test_int_32(self):
        assert _parse_mfma_override(32) == 32

    def test_string_16(self):
        assert _parse_mfma_override("16") == 16

    def test_string_32(self):
        assert _parse_mfma_override("32") == 32

    def test_invalid_int_returns_none(self):
        assert _parse_mfma_override(64) is None
        assert _parse_mfma_override(8) is None

    def test_garbage_string_returns_none(self):
        assert _parse_mfma_override("hello") is None
        assert _parse_mfma_override("16x16") is None


class TestPredicateDefault:
    """Without an override, the predicate must always return 16
    (legacy behavior).  Auto-promotion is intentionally disabled."""

    @pytest.mark.parametrize(
        "m,n,k",
        [
            (1024, 1024, 1024),
            (16384, 1024, 8192),     # large skinny
            (1024, 16384, 8192),
            (4096, 4096, 8192),      # square + big K
            (8192, 8192, 8192),
        ],
    )
    def test_no_override_always_16(self, m, n, k):
        got = select_mfma_instr_size(
            m=m, n=n, k=k,
            block_m=128, block_n=128, block_k=64,
            largest_input_bitsize=16, arch_name="gfx942",
        )
        assert got == 16


class TestPredicateOverride:
    def test_override_16_returns_16(self):
        assert select_mfma_instr_size(
            m=4096, n=4096, k=4096,
            block_m=128, block_n=128, block_k=64,
            largest_input_bitsize=16, arch_name="gfx942",
            override=16,
        ) == 16

    def test_override_32_valid_returns_32(self):
        assert select_mfma_instr_size(
            m=4096, n=4096, k=4096,
            block_m=128, block_n=128, block_k=64,
            largest_input_bitsize=16, arch_name="gfx942",
            override=32,
        ) == 32

    def test_override_32_fp8_returns_32(self):
        assert select_mfma_instr_size(
            m=4096, n=4096, k=4096,
            block_m=128, block_n=128, block_k=64,
            largest_input_bitsize=8, arch_name="gfx942",
            override=32,
        ) == 32

    def test_override_32_fp32_falls_back(self):
        # FP32 (32 bit) does not have 32x32x* MFMA in Triton's path.
        assert select_mfma_instr_size(
            m=4096, n=4096, k=4096,
            block_m=128, block_n=128, block_k=64,
            largest_input_bitsize=32, arch_name="gfx942",
            override=32,
        ) == 16

    @pytest.mark.parametrize("arch", ["gfx90a", "gfx1100", "unknown"])
    def test_override_32_unsupported_arch_falls_back(self, arch):
        assert select_mfma_instr_size(
            m=4096, n=4096, k=4096,
            block_m=128, block_n=128, block_k=64,
            largest_input_bitsize=16, arch_name=arch,
            override=32,
        ) == 16

    @pytest.mark.parametrize("bm,bn", [(16, 16), (16, 128), (128, 16)])
    def test_override_32_tile_too_small_falls_back(self, bm, bn):
        assert select_mfma_instr_size(
            m=4096, n=4096, k=4096,
            block_m=bm, block_n=bn, block_k=64,
            largest_input_bitsize=16, arch_name="gfx942",
            override=32,
        ) == 16

    @pytest.mark.parametrize("bm,bn", [(48, 64), (64, 48), (80, 80), (48, 32)])
    def test_override_32_tile_not_aligned_falls_back(self, bm, bn):
        assert select_mfma_instr_size(
            m=4096, n=4096, k=4096,
            block_m=bm, block_n=bn, block_k=64,
            largest_input_bitsize=16, arch_name="gfx942",
            override=32,
        ) == 16

    def test_override_32_problem_too_small_falls_back(self):
        assert select_mfma_instr_size(
            m=16, n=16, k=4096,
            block_m=128, block_n=128, block_k=64,
            largest_input_bitsize=16, arch_name="gfx942",
            override=32,
        ) == 16

    def test_invalid_override_value_returns_16(self):
        # Anything other than 16/32 -> safe default 16.
        assert select_mfma_instr_size(
            m=4096, n=4096, k=4096,
            block_m=128, block_n=128, block_k=64,
            largest_input_bitsize=16, arch_name="gfx942",
            override=64,
        ) == 16


class TestEnvVarOverride:
    """The env var TRITONBLAS_MFMA_INSTR_SIZE plumbs through the
    selector constructor.  We can't construct the full selector here
    without origami / GPU, so just verify the parser path."""

    def test_env_value_16(self, monkeypatch):
        monkeypatch.setenv("TRITONBLAS_MFMA_INSTR_SIZE", "16")
        v = _parse_mfma_override(os.environ["TRITONBLAS_MFMA_INSTR_SIZE"])
        assert v == 16

    def test_env_value_32(self, monkeypatch):
        monkeypatch.setenv("TRITONBLAS_MFMA_INSTR_SIZE", "32")
        v = _parse_mfma_override(os.environ["TRITONBLAS_MFMA_INSTR_SIZE"])
        assert v == 32

    def test_env_value_garbage(self, monkeypatch):
        monkeypatch.setenv("TRITONBLAS_MFMA_INSTR_SIZE", "auto")
        v = _parse_mfma_override(os.environ["TRITONBLAS_MFMA_INSTR_SIZE"])
        assert v is None
