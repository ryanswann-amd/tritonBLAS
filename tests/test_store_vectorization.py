#!/usr/bin/env python3
"""Test which store patterns produce vectorized vs scalar ISA on gfx942.

Approaches:
1. Indirect pointer + fp16 store (mirrors grouped GEMM production path)
2. Indirect pointer + fp32 store (skip fp16 conversion)
3. Direct pointer + fp16 store (baseline — should vectorize)
4. Direct pointer + fp32 store (baseline — should vectorize)
5. Direct base + runtime offset + fp16 store (proposed fix)
6. Indirect pointer + fp16 store with contiguity hints
7. Indirect pointer + fp32 store with contiguity hints
8. Inline asm: v_pack_b32_f16 + store as int32 via tl.store (1D)
"""
import triton
import triton.language as tl
import torch
import glob, os, re, shutil


# ---------------------------------------------------------------------------
# Kernel 1: Indirect pointer + fp16 store (matches grouped kernel)
# ---------------------------------------------------------------------------
@triton.jit
def store_indirect_fp16(c_ptrs, N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    c_ptr = tl.load(c_ptrs).to(tl.pointer_type(tl.float16))
    rm = tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32) + 1.0
    C_ = c_ptr + rm[:, None] * N + rn[None, :]
    tl.store(C_, acc.to(tl.float16))


# ---------------------------------------------------------------------------
# Kernel 2: Indirect pointer + fp32 store
# ---------------------------------------------------------------------------
@triton.jit
def store_indirect_fp32(c_ptrs, N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    c_ptr = tl.load(c_ptrs).to(tl.pointer_type(tl.float32))
    rm = tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32) + 1.0
    C_ = c_ptr + rm[:, None] * N + rn[None, :]
    tl.store(C_, acc)


# ---------------------------------------------------------------------------
# Kernel 3: Direct pointer + fp16 store (baseline — should be vectorized)
# ---------------------------------------------------------------------------
@triton.jit
def store_direct_fp16(c_ptr, N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rm = tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32) + 1.0
    C_ = c_ptr + rm[:, None] * N + rn[None, :]
    tl.store(C_, acc.to(tl.float16))


# ---------------------------------------------------------------------------
# Kernel 4: Direct pointer + fp32 store
# ---------------------------------------------------------------------------
@triton.jit
def store_direct_fp32(c_ptr, N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rm = tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32) + 1.0
    C_ = c_ptr + rm[:, None] * N + rn[None, :]
    tl.store(C_, acc)


# ---------------------------------------------------------------------------
# Kernel 5: Direct base pointer + runtime offset (proposed fix)
# ---------------------------------------------------------------------------
@triton.jit
def store_base_plus_offset(
    c_base,
    group_offsets,
    N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    offset = tl.load(group_offsets)
    c_ptr = c_base + offset
    rm = tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32) + 1.0
    C_ = c_ptr + rm[:, None] * N + rn[None, :]
    tl.store(C_, acc.to(tl.float16))


# ---------------------------------------------------------------------------
# Kernel 6: Indirect pointer + fp16 store WITH contiguity hints
# ---------------------------------------------------------------------------
@triton.jit
def store_indirect_fp16_hints(c_ptrs, N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    c_ptr = tl.load(c_ptrs).to(tl.pointer_type(tl.float16))
    rm = tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32) + 1.0
    C_ = c_ptr + rm[:, None] * N + rn[None, :]
    tl.store(C_, acc.to(tl.float16))


# ---------------------------------------------------------------------------
# Kernel 7: Indirect pointer + fp32 store WITH contiguity hints
# ---------------------------------------------------------------------------
@triton.jit
def store_indirect_fp32_hints(c_ptrs, N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    c_ptr = tl.load(c_ptrs).to(tl.pointer_type(tl.float32))
    rm = tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32) + 1.0
    C_ = c_ptr + rm[:, None] * N + rn[None, :]
    tl.store(C_, acc)


# ---------------------------------------------------------------------------
# Kernel 8: Inline asm - pack two fp16 into dword, store as int32
# 1D layout for simplicity. Processes pairs along the vector.
# ---------------------------------------------------------------------------
@triton.jit
def store_packed_asm_1d(c_ptrs, N: tl.constexpr):
    """1D kernel: packs pairs of fp16 values into int32, stores via tl.store."""
    HALF_N: tl.constexpr = N // 2
    c_ptr = tl.load(c_ptrs).to(tl.pointer_type(tl.float16))

    # Simulate two sets of fp16 values (even/odd elements)
    val_lo = tl.full((HALF_N,), 1.0, dtype=tl.float16)
    val_hi = tl.full((HALF_N,), 2.0, dtype=tl.float16)

    # Pack two fp16 into one int32: lo in bits[15:0], hi in bits[31:16]
    packed = tl.inline_asm_elementwise(
        "v_pack_b32_f16 $0, $1, $2",
        "=v, v, v",
        [val_lo, val_hi],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )

    # Store as dword — each int32 covers two fp16 elements
    c_i32 = c_ptr.to(tl.pointer_type(tl.int32))
    offs = tl.arange(0, HALF_N)
    tl.store(c_i32 + offs, packed)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def find_amdgcn(kernel_name, cache_dir):
    """Find the .amdgcn file for a kernel in the triton cache."""
    pattern = os.path.join(cache_dir, "**", f"{kernel_name}.amdgcn")
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def run_test():
    cache_dir = os.path.expanduser("~/.triton/cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    M, N, BLOCK_M, BLOCK_N = 128, 128, 128, 128

    print("=" * 100)
    print("Store Vectorization Test — MI300X (gfx942)")
    print("=" * 100)

    # ---- Run all 2D kernels ----

    # 1. Indirect fp16
    C1 = torch.zeros(M, N, dtype=torch.float16, device='cuda')
    ptrs1 = torch.tensor([C1.data_ptr()], dtype=torch.int64, device='cuda')
    store_indirect_fp16[(1,)](ptrs1, N, BLOCK_M, BLOCK_N, num_warps=8)
    torch.cuda.synchronize()

    # 2. Indirect fp32
    C2 = torch.zeros(M, N, dtype=torch.float32, device='cuda')
    ptrs2 = torch.tensor([C2.data_ptr()], dtype=torch.int64, device='cuda')
    store_indirect_fp32[(1,)](ptrs2, N, BLOCK_M, BLOCK_N, num_warps=8)
    torch.cuda.synchronize()

    # 3. Direct fp16
    C3 = torch.zeros(M, N, dtype=torch.float16, device='cuda')
    store_direct_fp16[(1,)](C3, N, BLOCK_M, BLOCK_N, num_warps=8)
    torch.cuda.synchronize()

    # 4. Direct fp32
    C4 = torch.zeros(M, N, dtype=torch.float32, device='cuda')
    store_direct_fp32[(1,)](C4, N, BLOCK_M, BLOCK_N, num_warps=8)
    torch.cuda.synchronize()

    # 5. Direct base + runtime offset
    C5 = torch.zeros(M, N, dtype=torch.float16, device='cuda')
    offsets5 = torch.tensor([0], dtype=torch.int64, device='cuda')
    store_base_plus_offset[(1,)](C5, offsets5, N, BLOCK_M, BLOCK_N, num_warps=8)
    torch.cuda.synchronize()

    # 6. Indirect fp16 with hints
    C6 = torch.zeros(M, N, dtype=torch.float16, device='cuda')
    ptrs6 = torch.tensor([C6.data_ptr()], dtype=torch.int64, device='cuda')
    store_indirect_fp16_hints[(1,)](ptrs6, N, BLOCK_M, BLOCK_N, num_warps=8)
    torch.cuda.synchronize()

    # 7. Indirect fp32 with hints
    C7 = torch.zeros(M, N, dtype=torch.float32, device='cuda')
    ptrs7 = torch.tensor([C7.data_ptr()], dtype=torch.int64, device='cuda')
    store_indirect_fp32_hints[(1,)](ptrs7, N, BLOCK_M, BLOCK_N, num_warps=8)
    torch.cuda.synchronize()

    # 8. Inline asm packed 1D
    C8 = torch.zeros(N, dtype=torch.float16, device='cuda')
    ptrs8 = torch.tensor([C8.data_ptr()], dtype=torch.int64, device='cuda')
    asm_error = None
    try:
        store_packed_asm_1d[(1,)](ptrs8, N, num_warps=1)
        torch.cuda.synchronize()
        expected = torch.tensor([1.0, 2.0] * (N // 2), dtype=torch.float16, device='cuda')
        if torch.allclose(C8, expected):
            print("\n[8] Inline asm packed store: correctness PASS")
        else:
            print(f"\n[8] Inline asm packed store: correctness FAIL")
            print(f"    Got:      {C8[:8].tolist()}")
            print(f"    Expected: {expected[:8].tolist()}")
    except Exception as e:
        asm_error = str(e)
        print(f"\n[8] Inline asm packed store: ERROR — {e}")

    # ---- Analyze ISA ----

    kernel_names = [
        "store_indirect_fp16",
        "store_indirect_fp32",
        "store_direct_fp16",
        "store_direct_fp32",
        "store_base_plus_offset",
        "store_indirect_fp16_hints",
        "store_indirect_fp32_hints",
        "store_packed_asm_1d",
    ]

    W = 28  # kernel name column width

    header = (
        f"{'Kernel':<{W}}"
        f"{'store_short':>14}"
        f"{'store_dword':>14}"
        f"{'store_dwordx2':>16}"
        f"{'store_dwordx4':>16}"
        f"{'buffer_store':>14}"
    )
    print(f"\n{header}")
    print("=" * len(header))

    for name in kernel_names:
        amdgcn_path = find_amdgcn(name, cache_dir)
        if amdgcn_path is None:
            if name == "store_packed_asm_1d" and asm_error:
                print(f"{name:<{W}} (compilation failed)")
            else:
                print(f"{name:<{W}} (no .amdgcn found)")
            continue

        with open(amdgcn_path) as f:
            asm = f.read()

        short = asm.count("global_store_short")
        dword = len(re.findall(r'global_store_dword(?!x)', asm))
        dwordx2 = asm.count("global_store_dwordx2")
        dwordx4 = asm.count("global_store_dwordx4")
        buffer = len(re.findall(r'buffer_store_\w+', asm))

        print(
            f"{name:<{W}}"
            f"{short:>14}"
            f"{dword:>14}"
            f"{dwordx2:>16}"
            f"{dwordx4:>16}"
            f"{buffer:>14}"
        )

    print()
    print("KEY:")
    print("  global_store_short   = 16-bit scalar store (2 bytes)  <-- SLOW, current problem")
    print("  global_store_dword   = 32-bit store (4 bytes)")
    print("  global_store_dwordx2 = 64-bit store (8 bytes)")
    print("  global_store_dwordx4 = 128-bit store (16 bytes)       <-- IDEAL target")
    print("  buffer_store         = uses buffer descriptors (SGPR)")
    print()


if __name__ == "__main__":
    run_test()
