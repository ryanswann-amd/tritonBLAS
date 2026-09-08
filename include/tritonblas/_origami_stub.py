"""Import-only stand-in for the compiled `origami` extension.

Why this exists
---------------
`tritonblas/origami.py` does a bare `import origami` at module scope, and
`tritonblas/__init__.py` wraps every top-level import in a single
`try: ... except ImportError: pass`. So on a machine without origami:

    import tritonblas          # SUCCEEDS
    tritonblas.matmul          # AttributeError
    tritonblas.schedule        # AttributeError

One missing dependency silently removes *everything* the package exposes, and
`import tritonblas` succeeding proves nothing. Checking
`hasattr(tritonblas, "schedule")` is the only reliable test.

The real origami is a compiled CPython extension that ships inside a private
sglang image. Two ways to get it properly, both preferable to this file:

    pip install rocm-origami          # public, "Analytical GEMM Solution Selection"
    # or run inside the image that ships it

What this deliberately does NOT do
----------------------------------
It does not emulate the selector. Every attribute access returns something that
raises on call, so a code path that actually asks origami to *choose* a
configuration fails loudly instead of silently substituting a different kernel
than the one being measured. A stub that guessed would corrupt a benchmark
quietly, which is worse than not running at all.

Callers that pass an explicit tile configuration never reach the selector, so
for those the import succeeding is all that is required.
"""

_INSTALL_HINT = (
    "the real origami extension is not present. Install it with "
    "`pip install rocm-origami`, or run inside the image that ships it. "
    "Refusing to substitute a different kernel selection."
)


class _Unavailable:
    """Raises on call or on any attribute chain below it, never on access."""

    def __init__(self, name):
        self._name = name

    def __call__(self, *a, **k):
        raise RuntimeError(f"origami.{self._name}() called, but {_INSTALL_HINT}")

    def __getattr__(self, item):
        return _Unavailable(f"{self._name}.{item}")

    def __repr__(self):
        return f"<origami stub: {self._name} unavailable>"


def __getattr__(name):
    # Module-level __getattr__ (PEP 562): any `origami.<anything>` resolves to a
    # loud placeholder rather than an AttributeError, so the failure names the
    # missing dependency instead of looking like a typo.
    return _Unavailable(name)
