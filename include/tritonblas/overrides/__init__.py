"""Registered guarded-overrides.

Each module under this package registers one ``GuardedOverride`` with the
``OverrideRegistry`` singleton on import.  Importing the package
(``import tritonblas.overrides``) registers all bundled overrides.

Adding a new override:

  1. Create ``tritonblas/overrides/k_<NNN>_<short>.py``.
  2. Define a ``GUARD = GuardedOverride(...)`` and call
     ``OverrideRegistry.register(GUARD)`` at module import.
  3. Add an ``import .k_<NNN>_<short>`` line below.
  4. Run ``run_falsification(GUARD, shapes=...)`` in CI; the override does
     NOT land unless the verdict is LAND.

R1 — dispatch-order-first routing — is encoded by the order of the
imports below.  The strictest predicate (smallest cohort) imports first.
"""

from . import k_882_grid_cap  # noqa: F401
