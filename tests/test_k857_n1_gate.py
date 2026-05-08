"""K-857 [S-002] N1 cohort narrow-gate — production no-leak guard.

The K-857 paired bench (n=50 hot HIP-graph on c42/MI300X b07u13) returned
**VERDICT: NO-GO** — geomean ON/OFF on the 4 N1-envelope cells = 0.815x
(i.e. the K-850 override is ~18.5% slower than `main`). Mechanism: Origami
on `main @ 95e2c47` already picks BM=BN=256 (4 MB tile-area) for these
cells; the K-850 recommendation of 256x128 (2 MB) is a *tile shrink* on
this baseline, dispatching 2x more waves and triggering exactly the
wave-overdispatch pathology K-850 was meant to fix. Full mechanism +
per-cell numbers in ``output/REPORT.md``.

Because the override was **NOT** landed in production
``include/tritonblas/`` (NO-GO), the only invariant worth regression-
protecting is the *no-leak* guard: a future cherry-pick or revert of the
experimental branch must NOT silently re-enable the override. This file
ships exactly one test that pins that decision by greppping every Python
source file under ``include/tritonblas/`` for the K-857-internal token
fingerprint.

The predicate, tile picker, and disjointness near-miss truth table that
were prototyped during the experiment are NOT shipped here — there is no
production code to regress them against, so they would be dead pure-Python
helpers. If composite-14 lands and shifts Origami's tile picks back toward
128x128, the next attempt should re-derive the predicate fresh against
the new baseline.

Test runs CPU-only (no Triton compile, no GPU touch) so it works in the
standard ``python -m pytest tests/`` lane.
"""

from __future__ import annotations

import pathlib


# K-857-internal symbol fingerprint. If any of these tokens appear in any
# .py file under ``include/tritonblas/``, the experimental override has
# been re-introduced into the production path. The bench result was
# NO-GO; re-run the K-857 paired bench before re-landing.
#
# Tokens are intentionally specific to the K-857 experiment (every name
# here was prefixed with ``k857``/``K857`` or contains ``K_857``) so this
# guard cannot collide with unrelated production code that happens to
# mention "kpack" or "narrow-gate".
_K857_FORBIDDEN_TOKENS = (
    "is_K857_N1_cohort",
    "_maybe_apply_k857_n1",
    "_K857_N1_OverrideSelector",
    "TRITONBLAS_ENABLE_K857_N1",
    "_k857_kpack",
    "k857_n1_tile",
    "K857_N1",
)


def _tritonblas_package_root() -> pathlib.Path:
    """Return the ``include/tritonblas/`` package root regardless of CWD."""
    # tests/ sits next to include/ in the repo layout.
    here = pathlib.Path(__file__).resolve().parent
    repo = here.parent
    pkg = repo / "include" / "tritonblas"
    assert pkg.is_dir(), (
        f"Expected tritonblas package at {pkg} (repo layout changed?)"
    )
    return pkg


def test_k857_override_not_wired_into_production_package():
    """K-857 N1 narrow-gate override must NOT appear anywhere under
    ``include/tritonblas/``.

    The K-857 paired bench was NO-GO (geomean 0.815x on the 4 N1-envelope
    cells; see output/REPORT.md). The override was deliberately kept out
    of the production path. This guard scans **every** .py file under
    ``include/tritonblas/`` (not just matmul.py) so a cherry-pick into
    any sibling module — config.py, origami.py, kernels/* — also trips
    the assertion.

    If this test starts failing, someone re-landed an experiment that
    empirically regressed its own target cohort. Re-run the K-857 paired
    bench (scripts/k857_run_bench.sh) and re-confirm the verdict before
    shipping.
    """
    pkg = _tritonblas_package_root()
    py_files = sorted(pkg.rglob("*.py"))
    assert py_files, f"No .py files found under {pkg} — repo layout changed?"

    leaks: list[str] = []
    for path in py_files:
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:  # pragma: no cover - defensive
            raise AssertionError(f"Could not read {path}: {exc}") from exc
        hits = [tok for tok in _K857_FORBIDDEN_TOKENS if tok in src]
        if hits:
            rel = path.relative_to(pkg.parent.parent)
            leaks.append(f"{rel}: {hits}")

    assert not leaks, (
        "K-857 N1 override hooks leaked into the production tritonblas "
        "package:\n  "
        + "\n  ".join(leaks)
        + "\n\nThe K-857 experiment was NO-GO (geomean 0.815x on N1; see "
        "output/REPORT.md). The override must stay out of the production "
        "path. Re-run the paired bench (scripts/k857_run_bench.sh) before "
        "re-landing."
    )
