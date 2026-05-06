#!/usr/bin/env python3
"""Pre-warm the Triton on-disk JIT cache for tritonblas.

Thin shim over ``python -m tritonblas.prewarm`` for operators who prefer a
``tools/``-resident script. See module docstring in
``include/tritonblas/prewarm.py`` for full strategy / background.

Usage examples:

    # Default playlist (K-169-derived), default cache dir (~/.triton/cache):
    python tools/prewarm_triton_cache.py

    # Custom playlist, custom cache dir, quiet output:
    python tools/prewarm_triton_cache.py \\
        --playlist deployment_shapes.csv \\
        --cache-dir /opt/triton-cache \\
        --quiet

    # Smoke test on first 5 shapes only:
    python tools/prewarm_triton_cache.py --limit 5
"""

import sys

if __name__ == "__main__":
    from tritonblas.prewarm import main
    sys.exit(main())
