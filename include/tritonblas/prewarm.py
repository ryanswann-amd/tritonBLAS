"""Triton on-disk JIT cache pre-warm utility for tritonblas.

Production deployments pay a 0.5-3s per-kernel JIT compile penalty on the
first ``tritonblas.matmul()`` invocation against any new
``(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, dtype, EVEN_K)``
signature.  This utility eliminates that penalty by walking a representative
playlist of GEMM shapes through the production dispatch path once, populating
``$TRITON_CACHE_DIR`` (default ``~/.triton/cache``) with compiled artifacts.

Subsequent processes that import tritonblas and call ``matmul()`` on a shape
whose Origami-selected signature matches a pre-warmed entry skip the JIT phase
entirely; first-call latency collapses to the analytical-selector cost
(~155 µs) plus a small kernel-handle load.

Typical use::

    # one-time at install/deploy:
    python -m tritonblas.prewarm --cache-dir /opt/triton-cache
    # then in production: TRITON_CACHE_DIR=/opt/triton-cache python serve.py

A bundled default playlist (``include/tritonblas/data/prewarm_playlist.csv``)
covers the K-169 regression-suite shapes; users may pass ``--playlist`` to
target their own deployment-specific shape distribution.

Background: the strategy is option (b-ii) from S-002 K-500 (cold-start
mitigation strategy doc); zero source change to the dispatcher, zero risk
(unused signatures simply waste JIT cycles), and steady-state perf is
provably unchanged because the same kernel + config is selected with or
without the pre-warm.
"""

from __future__ import annotations

import csv
import gc
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import torch


# ---------------------------------------------------------------------------
# Dtype string normalization (matches OrigamiMatmulSelector.dtype_to_str).
# ---------------------------------------------------------------------------

def _build_dtype_table() -> dict:
    table = {
        "bf16": torch.bfloat16,
        "f16": torch.float16,
        "fp16": torch.float16,
        "f32": torch.float32,
        "fp32": torch.float32,
        "i8": torch.int8,
        "int8": torch.int8,
    }
    # FP8: prefer FNUZ on gfx9 / non-gfx950, fall back to OCP on gfx950.
    if hasattr(torch, "float8_e4m3fnuz"):
        table["f8"] = torch.float8_e4m3fnuz
        table["f8_e4m3"] = torch.float8_e4m3fnuz
    elif hasattr(torch, "float8_e4m3fn"):
        table["f8"] = torch.float8_e4m3fn
        table["f8_e4m3"] = torch.float8_e4m3fn
    if hasattr(torch, "float8_e5m2fnuz"):
        table["f8_e5m2"] = torch.float8_e5m2fnuz
    elif hasattr(torch, "float8_e5m2"):
        table["f8_e5m2"] = torch.float8_e5m2
    return table


_DTYPE_FROM_STR = _build_dtype_table()


def _resolve_dtype(spec: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(spec, torch.dtype):
        return spec
    s = spec.strip().lower()
    if s not in _DTYPE_FROM_STR:
        raise ValueError(f"unknown dtype string: {spec!r}")
    return _DTYPE_FROM_STR[s]


def _is_fp8(dtype: torch.dtype) -> bool:
    return "float8" in str(dtype)


# ---------------------------------------------------------------------------
# Playlist loading.
# ---------------------------------------------------------------------------

def _default_playlist_path() -> Path:
    return Path(__file__).parent / "data" / "prewarm_playlist.csv"


def load_playlist(
    path: Optional[Union[str, Path]] = None,
) -> list[Tuple[int, int, int, str, str, str]]:
    """Load a prewarm playlist CSV.

    Accepted column layouts (per row):
        - ``m, n, k, a_dtype, b_dtype, c_dtype``
        - ``m, n, k, in_dtype, out_dtype``  (a_dtype = b_dtype = in_dtype)

    Returns a list of ``(m, n, k, a_dtype_str, b_dtype_str, c_dtype_str)``
    tuples with dtype strings normalized to lower-case. Extra columns are
    ignored.
    """
    p = Path(path) if path is not None else _default_playlist_path()
    if not p.exists():
        raise FileNotFoundError(f"prewarm playlist not found: {p}")
    out: list[Tuple[int, int, int, str, str, str]] = []
    with open(p, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = int(row["m"]); n = int(row["n"]); k = int(row["k"])
            a = (row.get("a_dtype") or row.get("in_dtype") or "bf16").strip().lower()
            b = (row.get("b_dtype") or row.get("in_dtype") or a).strip().lower()
            c = (row.get("c_dtype") or row.get("out_dtype") or a).strip().lower()
            out.append((m, n, k, a, b, c))
    return out


# ---------------------------------------------------------------------------
# Pre-warm core.
# ---------------------------------------------------------------------------

def _drive_one_shape(
    matmul_fn,
    matmul_a8w8_fn,
    m: int,
    n: int,
    k: int,
    a_dtype: torch.dtype,
    b_dtype: torch.dtype,
    c_dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Run a single shape through the production dispatcher.

    FP8 inputs route to ``matmul_a8w8`` (per-tensor scales) since that is
    the production entry-point for quantized GEMM; everything else routes
    to ``matmul``.
    """
    if _is_fp8(a_dtype) or _is_fp8(b_dtype):
        a = torch.empty((m, k), dtype=a_dtype, device=device)
        b = torch.empty((k, n), dtype=b_dtype, device=device)
        c = torch.empty((m, n), dtype=c_dtype, device=device)
        a_scale = torch.ones((1,), dtype=torch.float32, device=device)
        b_scale = torch.ones((1,), dtype=torch.float32, device=device)
        matmul_a8w8_fn(a, b, a_scale, b_scale, c)
    else:
        a = torch.empty((m, k), dtype=a_dtype, device=device)
        b = torch.empty((k, n), dtype=b_dtype, device=device)
        out = torch.empty((m, n), dtype=c_dtype, device=device)
        matmul_fn(a, b, out=out)


def prewarm(
    shapes: Optional[Iterable[Tuple[int, int, int, Union[str, torch.dtype],
                                    Union[str, torch.dtype],
                                    Union[str, torch.dtype]]]] = None,
    *,
    cache_dir: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
    verbose: bool = False,
    skip_on_error: bool = True,
    dedupe_signatures: bool = True,
) -> dict:
    """Pre-compile Triton kernels for a list of GEMM shapes.

    Args:
        shapes: iterable of ``(m, n, k, a_dtype, b_dtype, c_dtype)`` tuples.
            Dtypes may be ``torch.dtype`` or strings ("bf16", "f16", "f8",
            "i8", "f32"). Defaults to the bundled K-169-derived playlist.
        cache_dir: if given, sets ``TRITON_CACHE_DIR`` for this process so the
            JIT writes artifacts to that directory. Production processes that
            export the same ``TRITON_CACHE_DIR`` will then hit cache.
        device: torch device. Defaults to current CUDA device.
        verbose: per-shape progress printing.
        skip_on_error: if False, the first compile error is raised. Default
            True: failures (e.g. unsupported dtype combos for a kernel) are
            tallied and reported at the end.
        dedupe_signatures: if True (default), instantiate ``OrigamiMatmulSelector``
            on CPU first and skip shapes whose
            ``(block_m, block_n, block_k, num_stages, dtypes, even_k)`` signature
            is already represented. Cuts wall time on shape-dense playlists by
            avoiding redundant identical compiles.

    Returns:
        Dict with ``compiled``, ``skipped`` (dedup or unknown dtype),
        ``failed`` (compile/runtime error), ``wall_seconds``, ``cache_dir``.
    """
    if cache_dir is not None:
        os.environ["TRITON_CACHE_DIR"] = str(Path(cache_dir).expanduser())

    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("prewarm requires a CUDA device")
        device = torch.device("cuda", torch.cuda.current_device())

    if shapes is None:
        shapes = load_playlist()
    shapes = list(shapes)

    # Lazy import to avoid circulars at package import time.
    from .matmul import matmul, matmul_a8w8
    selector_cls = None
    if dedupe_signatures:
        try:
            from .origami import OrigamiMatmulSelector
            selector_cls = OrigamiMatmulSelector
        except Exception as e:
            if verbose:
                print(f"[prewarm] dedup disabled (selector import failed: {e})",
                      file=sys.stderr)

    seen: set[tuple] = set()
    compiled = skipped = failed = 0
    t0 = time.perf_counter()

    for i, row in enumerate(shapes):
        m, n, k, a_s, b_s, c_s = row
        # Resolve dtypes
        try:
            a_dtype = _resolve_dtype(a_s)
            b_dtype = _resolve_dtype(b_s)
            c_dtype = _resolve_dtype(c_s)
        except Exception as e:
            skipped += 1
            if verbose:
                print(f"[{i+1}/{len(shapes)}] skip-dtype {row}: {e}",
                      file=sys.stderr)
            continue

        # Signature dedup (CPU only — selector init is ~155 µs).
        if selector_cls is not None:
            try:
                sel = selector_cls(m, n, k, a_dtype, b_dtype, c_dtype, device)
                ns = getattr(sel, "_num_stages", getattr(sel, "num_stages", 2))
                even_k = (k % sel.block_k) == 0
                sig = (sel.block_m, sel.block_n, sel.block_k, ns,
                       str(a_dtype), str(b_dtype), str(c_dtype), even_k)
                if sig in seen:
                    skipped += 1
                    if verbose:
                        print(f"[{i+1}/{len(shapes)}] dedup {m}x{n}x{k} sig={sig}")
                    continue
                seen.add(sig)
            except Exception as e:
                # Fall through to a real compile so we still try.
                if verbose:
                    print(f"[{i+1}/{len(shapes)}] selector-warn {m}x{n}x{k}: {e}",
                          file=sys.stderr)

        # Drive the dispatcher.
        try:
            t1 = time.perf_counter()
            _drive_one_shape(matmul, matmul_a8w8, m, n, k,
                             a_dtype, b_dtype, c_dtype, device)
            torch.cuda.synchronize(device)
            dt = time.perf_counter() - t1
            compiled += 1
            if verbose:
                print(f"[{i+1}/{len(shapes)}] {m}x{n}x{k} "
                      f"{a_s}->{c_s}  {dt:.2f}s")
        except Exception as e:
            failed += 1
            if verbose:
                print(f"[{i+1}/{len(shapes)}] FAIL {m}x{n}x{k} "
                      f"{a_s}->{c_s}: {e}", file=sys.stderr)
            if not skip_on_error:
                raise
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return {
        "compiled": compiled,
        "skipped": skipped,
        "failed": failed,
        "wall_seconds": time.perf_counter() - t0,
        "cache_dir": os.environ.get("TRITON_CACHE_DIR", "<default>"),
    }


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        prog="python -m tritonblas.prewarm",
        description=("Pre-warm the Triton on-disk JIT cache for tritonblas. "
                     "Drives a representative GEMM shape playlist through the "
                     "production dispatcher so subsequent processes skip "
                     "compile latency on first call."),
    )
    ap.add_argument("--playlist", default=None,
                    help="CSV with columns m,n,k[,a_dtype,b_dtype,c_dtype | "
                         "in_dtype,out_dtype]. Defaults to bundled K-169 playlist.")
    ap.add_argument("--cache-dir", default=None,
                    help="Override TRITON_CACHE_DIR for this run.")
    ap.add_argument("--no-dedupe", action="store_true",
                    help="Disable selector-signature dedup (compile every row).")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress per-shape progress.")
    ap.add_argument("--strict", action="store_true",
                    help="Stop on first failure instead of skipping.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N rows (smoke testing).")
    args = ap.parse_args(argv)

    shapes = load_playlist(args.playlist) if args.playlist else load_playlist()
    if args.limit is not None:
        shapes = shapes[: args.limit]
    res = prewarm(shapes=shapes,
                  cache_dir=args.cache_dir,
                  verbose=not args.quiet,
                  skip_on_error=not args.strict,
                  dedupe_signatures=not args.no_dedupe)
    print(
        f"prewarm: compiled={res['compiled']} dedup_or_skip={res['skipped']} "
        f"failed={res['failed']} wall={res['wall_seconds']:.1f}s "
        f"cache_dir={res['cache_dir']}"
    )
    return 0 if res["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
