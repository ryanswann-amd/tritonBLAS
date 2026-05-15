#!/usr/bin/env bash
# Build the K-6808 hand-written interleaved MFMA kernel into a shared lib.
#
# The .so is written next to this script as `libmfma_gemm.so` and is loaded
# at import time via ctypes by `dispatch.py`.
#
# Requires: hipcc (ROCm 6+), gfx942 device offload support.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${DIR}/mfma_gemm.hip"
OUT="${DIR}/libmfma_gemm.so"

if ! command -v hipcc >/dev/null 2>&1; then
  echo "[hip_kernels/build.sh] hipcc not found — skipping build (HIP fallback disabled)" >&2
  exit 0
fi

echo "[hip_kernels/build.sh] hipcc -O3 --offload-arch=gfx942 -fPIC -shared -o ${OUT} ${SRC}"
hipcc -O3 --offload-arch=gfx942 -fPIC -shared -o "${OUT}" "${SRC}"
echo "[hip_kernels/build.sh] built ${OUT}"
