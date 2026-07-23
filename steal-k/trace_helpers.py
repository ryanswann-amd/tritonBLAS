import triton
import triton.language as tl

# The HIP real-time counter is a constant-frequency 100 MHz clock on CDNA
# (unaffected by power states / clock gating).  1 tick = 10 ns.
REALTIME_HZ = 100_000_000

# memrealtime is required; smid is optional (absent in some Triton builds).
try:
    from triton.language.extra.hip import memrealtime as _memrealtime
    _HAS_REALTIME = True
except ImportError:
    _HAS_REALTIME = False

try:
    from triton.language.extra.hip import smid as _smid
    _HAS_SMID = True
except ImportError:
    _HAS_SMID = False


if _HAS_REALTIME:

    @triton.jit
    def read_realtime():
        """Read the GPU's constant-frequency (100 MHz) real-time counter.

        Delegates to ``tl.extra.hip.memrealtime()`` (``s_memrealtime`` on CDNA3,
        ``s_sendmsg_rtn_b64`` on gfx11/12).  Returns int64 ticks (100 MHz).
        """
        return _memrealtime()
else:

    @triton.jit
    def read_realtime():
        tl.static_assert(False, "memrealtime is unavailable in this Triton build")
        return tl.cast(0, tl.int64)


if _HAS_SMID:

    @triton.jit
    def get_cu_id():
        """CU / workgroup-processor id for the current wave."""
        return _smid()
else:

    @triton.jit
    def get_cu_id():
        """smid unavailable in this Triton build -> sentinel; use pid instead."""
        return tl.cast(-1, tl.int32)
