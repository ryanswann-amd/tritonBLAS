// mfma_gemm.hip — Hand-scheduled MFMA inner loops for MI300X (gfx942)
//
// Two kernel variants for the same GEMM compute, differing ONLY in the
// inner-loop schedule of LDS reads vs MFMA ops:
//
//   Variant A (CLUSTERED):  all LDS reads first, then all MFMAs.
//                           Mirrors what tritonblas currently emits.
//   Variant B (INTERLEAVED): LDS reads and MFMAs alternated 1:1.
//                            Mirrors the hipBLASLt schedule (K-6753).
//
// Hypothesis (K-6808): variant B closes ≥90% of the gap to torch.matmul
// (hipBLASLt) on a 64×64×4096 BF16 GEMM.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

using v4bf16 = __attribute__((__vector_size__(8))) short;   // 4 bf16 per lane
using fv4    = __attribute__((__vector_size__(16))) float;  // 4 fp32 per lane

// MI300X intrinsic: 16×16×16 bf16, A/B are <4 × bf16> per lane, C is <4 × f32>.
#define MFMA_16x16x16_BF16(C, A, B) \
  C = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(A, B, C, 0, 0, 0)

// Tile config: each WG = 1 wave (64 lanes), produces 16×16 fp32 output via
// MFMAs accumulated across the K dim.  Inner K-loop unrolled by K_UNROLL=4
// (4 MFMA-K=16 chunks), giving 4 pairs of LDS reads and 4 MFMAs per iter.
constexpr int BLOCK_M = 16;
constexpr int BLOCK_N = 16;
constexpr int BLOCK_K = 16;
constexpr int K_UNROLL = 4;     // 4 MFMAs per inner-loop iteration

// Cooperative load: 64 lanes × 4 bf16 = 256 = 16×16 bf16 panel.
__device__ __forceinline__
void load_panel(const __hip_bfloat16* g, int ld,
                __hip_bfloat16* s, int rows, int cols) {
  int tid = threadIdx.x;
  int row = tid / 4;       // 0..15 (assuming rows=cols=16)
  int col = (tid % 4) * 4; // 0,4,8,12
  if (row < rows && col < cols) {
    *((uint64_t*)&s[row * cols + col]) =
        *((const uint64_t*)&g[row * ld + col]);
  }
}

// Read one A-frag (16×16 bf16) from LDS into a v4bf16 per lane.
// Lane (id) → (row, kpack) with row = id % 16, kpack = id / 16.
// Per-lane data: 4 contiguous bf16 = sA[row * 16 + kpack*4 .. +3].
__device__ __forceinline__
v4bf16 load_frag_a(const __hip_bfloat16* sA) {
  int tid = threadIdx.x;
  int row = tid % 16;
  int kpack = tid / 16;
  const uint64_t* p = (const uint64_t*)&sA[row * BLOCK_K + kpack * 4];
  uint64_t v = *p;
  return *(v4bf16*)&v;
}
__device__ __forceinline__
v4bf16 load_frag_b(const __hip_bfloat16* sB) {
  int tid = threadIdx.x;
  int col = tid % 16;
  int kpack = tid / 16;
  const uint64_t* p = (const uint64_t*)&sB[(kpack * 4) * BLOCK_N + col * 4 / 4];
  // sB layout: K-major, BLOCK_K rows × BLOCK_N cols.
  // Per-lane data: 4 bf16 from sB[kpack*4 + i][col]  (i=0..3).
  // Since cols are not contiguous in memory for column-strided loads,
  // we instead store sB transposed (col-major within each K=16 tile)
  // so that the 4 bf16 for one (col, kpack) lane are contiguous.
  // To keep load_panel symmetric we use the same packed layout for B
  // by addressing as: sB[col*16 + kpack*4 + i] (B has been pre-transposed).
  const uint64_t* pp = (const uint64_t*)&sB[col * BLOCK_K + kpack * 4];
  uint64_t v2 = *pp;
  return *(v4bf16*)&v2;
}

// Pre-transpose store helper: write a 16×16 BF16 panel of B into LDS in
// transposed (col-major within tile) form so that load_frag_b can do
// a single ds_read_b64 per lane.
__device__ __forceinline__
void load_b_panel_transposed(const __hip_bfloat16* g, int ld,
                             __hip_bfloat16* sB) {
  // Each lane writes 4 bf16 (one row of B, contiguous in K).
  // After write, sB[col*16 + k] = B[k][col]  (col-major within tile).
  int tid = threadIdx.x;
  int row_k = tid / 4;        // K index, 0..15
  int col_n = (tid % 4) * 4;  // N start, 0,4,8,12
  uint64_t v = *((const uint64_t*)&g[row_k * ld + col_n]);
  __hip_bfloat16 tmp[4];
  *((uint64_t*)tmp) = v;
  for (int i = 0; i < 4; ++i) {
    sB[(col_n + i) * BLOCK_K + row_k] = tmp[i];
  }
}

// ---------------------------------------------------------------------------
// Kernel A — CLUSTERED.  All ds_reads first, then all MFMAs.
// ---------------------------------------------------------------------------
extern "C" __global__ void mfma_clustered(
    const __hip_bfloat16* __restrict__ A,
    const __hip_bfloat16* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {

  const int row_block = blockIdx.y * BLOCK_M;
  const int col_block = blockIdx.x * BLOCK_N;

  __shared__ __hip_bfloat16 sA[BLOCK_M * BLOCK_K * K_UNROLL];
  __shared__ __hip_bfloat16 sB[BLOCK_N * BLOCK_K * K_UNROLL];

  fv4 c = {0.f, 0.f, 0.f, 0.f};

  for (int k0 = 0; k0 < K; k0 += BLOCK_K * K_UNROLL) {
    // Stage A and B for K_UNROLL chunks.
    for (int u = 0; u < K_UNROLL; ++u) {
      load_panel(&A[row_block * K + k0 + u * BLOCK_K], K,
                 &sA[u * BLOCK_M * BLOCK_K], BLOCK_M, BLOCK_K);
      load_b_panel_transposed(&B[(k0 + u * BLOCK_K) * N + col_block], N,
                              &sB[u * BLOCK_N * BLOCK_K]);
    }
    __syncthreads();

    v4bf16 a0, a1, a2, a3;
    v4bf16 b0, b1, b2, b3;

    // ===== CLUSTERED: 8 ds_read* THEN 4 MFMA =====
    a0 = load_frag_a(&sA[0 * BLOCK_M * BLOCK_K]);
    b0 = load_frag_b(&sB[0 * BLOCK_N * BLOCK_K]);
    a1 = load_frag_a(&sA[1 * BLOCK_M * BLOCK_K]);
    b1 = load_frag_b(&sB[1 * BLOCK_N * BLOCK_K]);
    a2 = load_frag_a(&sA[2 * BLOCK_M * BLOCK_K]);
    b2 = load_frag_b(&sB[2 * BLOCK_N * BLOCK_K]);
    a3 = load_frag_a(&sA[3 * BLOCK_M * BLOCK_K]);
    b3 = load_frag_b(&sB[3 * BLOCK_N * BLOCK_K]);

    __builtin_amdgcn_sched_barrier(0);

    MFMA_16x16x16_BF16(c, a0, b0);
    MFMA_16x16x16_BF16(c, a1, b1);
    MFMA_16x16x16_BF16(c, a2, b2);
    MFMA_16x16x16_BF16(c, a3, b3);

    __builtin_amdgcn_sched_barrier(0);
    __syncthreads();
  }

  // CDNA3 v_mfma_f32_16x16x16 output ABI: lane i holds 4 fp32 at rows
  //   ((i/16)*4 + 0..3), col (i%16).  So store one element per row.
  int tid = threadIdx.x;
  int col = tid % 16;
  int row_start = (tid / 16) * 4;
  int co = col_block + col;
  if (co < N) {
    for (int j = 0; j < 4; ++j) {
      int r = row_block + row_start + j;
      if (r < M) C[r * N + co] = c[j];
    }
  }
}

// ---------------------------------------------------------------------------
// Kernel B — INTERLEAVED.  Pair-of-loads → MFMA → pair-of-loads → MFMA …
// ---------------------------------------------------------------------------
extern "C" __global__ void mfma_interleaved(
    const __hip_bfloat16* __restrict__ A,
    const __hip_bfloat16* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {

  const int row_block = blockIdx.y * BLOCK_M;
  const int col_block = blockIdx.x * BLOCK_N;

  __shared__ __hip_bfloat16 sA[BLOCK_M * BLOCK_K * K_UNROLL];
  __shared__ __hip_bfloat16 sB[BLOCK_N * BLOCK_K * K_UNROLL];

  fv4 c = {0.f, 0.f, 0.f, 0.f};

  for (int k0 = 0; k0 < K; k0 += BLOCK_K * K_UNROLL) {
    for (int u = 0; u < K_UNROLL; ++u) {
      load_panel(&A[row_block * K + k0 + u * BLOCK_K], K,
                 &sA[u * BLOCK_M * BLOCK_K], BLOCK_M, BLOCK_K);
      load_b_panel_transposed(&B[(k0 + u * BLOCK_K) * N + col_block], N,
                              &sB[u * BLOCK_N * BLOCK_K]);
    }
    __syncthreads();

    // ===== INTERLEAVED: source-order pair-load → MFMA → pair-load → MFMA =====
    // We rely on source-order + named locals to communicate the schedule.
    // sched_barrier with mask 0xFFB allows re-ordering of all but mfma class,
    // which preserves the structural pattern without breaking accumulator
    // liveness (mask 0 was found to silently break correctness here).
    v4bf16 a0 = load_frag_a(&sA[0 * BLOCK_M * BLOCK_K]);
    v4bf16 b0 = load_frag_b(&sB[0 * BLOCK_N * BLOCK_K]);
    MFMA_16x16x16_BF16(c, a0, b0);
    v4bf16 a1 = load_frag_a(&sA[1 * BLOCK_M * BLOCK_K]);
    v4bf16 b1 = load_frag_b(&sB[1 * BLOCK_N * BLOCK_K]);
    MFMA_16x16x16_BF16(c, a1, b1);
    v4bf16 a2 = load_frag_a(&sA[2 * BLOCK_M * BLOCK_K]);
    v4bf16 b2 = load_frag_b(&sB[2 * BLOCK_N * BLOCK_K]);
    MFMA_16x16x16_BF16(c, a2, b2);
    v4bf16 a3 = load_frag_a(&sA[3 * BLOCK_M * BLOCK_K]);
    v4bf16 b3 = load_frag_b(&sB[3 * BLOCK_N * BLOCK_K]);
    MFMA_16x16x16_BF16(c, a3, b3);

    __syncthreads();
  }

  // CDNA3 v_mfma_f32_16x16x16 OUTPUT ABI (NOT the same as the LDS-input lane
  // mapping used by load_frag_a/b above): lane i holds 4 fp32 values laid out
  // at rows ((i/16)*4 + j) for j=0..3, all sharing column (i%16).
  // Earlier revisions of this kernel reused the input ABI here, which caused
  // max_abs_err ≈ |ref| (essentially garbage output). Fixed in this revision.
  int tid = threadIdx.x;
  int col = tid % 16;
  int row_start = (tid / 16) * 4;
  int co = col_block + col;
  if (co < N) {
    for (int j = 0; j < 4; ++j) {
      int r = row_block + row_start + j;
      if (r < M) C[r * N + co] = c[j];
    }
  }
}

// ---------------------------------------------------------------------------
// Host launch wrappers (extern "C" for ctypes).
// ---------------------------------------------------------------------------
extern "C" {

void launch_clustered(const void* A, const void* B, void* C,
                      int M, int N, int K, hipStream_t stream) {
  dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M, 1);
  dim3 block(64, 1, 1);
  mfma_clustered<<<grid, block, 0, stream>>>(
      (const __hip_bfloat16*)A, (const __hip_bfloat16*)B,
      (float*)C, M, N, K);
}

void launch_interleaved(const void* A, const void* B, void* C,
                        int M, int N, int K, hipStream_t stream) {
  dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M, 1);
  dim3 block(64, 1, 1);
  mfma_interleaved<<<grid, block, 0, stream>>>(
      (const __hip_bfloat16*)A, (const __hip_bfloat16*)B,
      (float*)C, M, N, K);
}

}  // extern "C"
