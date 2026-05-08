"""K-935 — narrow LDS-pressure mitigation overrides.

Per K-901 design, this package contains only narrow guarded overrides that
have a LAND verdict from the K-883 §5 falsification harness.  Each override
registers itself with ``OverrideRegistry`` on import and is OFF by default
(env-var killswitch must be flipped to fire).  See ``k935_lds_pressure.py``
for the canonical pattern.
"""

from .k935_lds_pressure import K935_GATE  # noqa: F401
