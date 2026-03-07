/*
 * DSQG V5 — Backward CUDA Kernels
 *
 * Two kernels:
 *   dsqg_bwd_dkdv_kernel: compute dK, dV, d_phase_base, d_phase_gain, dz_pre
 *   dsqg_bwd_dq_kernel:   compute dQ, d_pos_bias, d_scale_embed, dy_pre
 *
 * === dKdV kernel design ===
 * Grid: (B*H, ceil(N/BLOCK_N))  — one block per KEY tile
 *
 * Key optimization vs Triton: Q and dout windows for all 33 dense offsets
 * are loaded ONCE into shared memory. For a key tile at [t_start, t_end]:
 *   - Dense offset delta=0..32: queries at t+delta lie in [t_start, t_end+32]
 *   - Load Q and dout for positions [t_start, t_start+BLOCK_N+32] into smem
 *   - All 33 dense offsets serviced from smem — 33 global reads → 1 larger read
 *   - Sparse offsets (delta=48..1536): load from global (no overlap)
 *
 * dK/dV accumulation: each thread owns one key position, accumulates in registers.
 * Writes are contiguous with no atomics.
 *
 * Saved theta (COSSIN [B,H,N,11,2]): loaded for sparse offsets → __sincosf.
 * ~2x faster than recomputing theta from y/z/phase params.
 *
 * === dQ kernel design ===
 * Grid: (B*H, ceil(N/BLOCK_N))  — one block per QUERY tile
 * Each thread (query n) iterates over 44 offsets, loading K[n-delta[j]] per offset.
 * dQ accumulation in registers: no atomics, each thread owns its gradient.
 * d_pos_bias/d_scale_embed: atomic adds (same pattern as Triton, unavoidable).
 */

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>

// ── Constants (duplicated from dsqg_fwd.cu — keep in sync) ──────────────────
#define MAX_OFFSET       1536
#define N_OFFSETS        44
#define N_DENSE          33
#define N_SPARSE         11
#define MAX_DENSE_DELTA  32  // largest dense offset

__constant__ int ALL_OFFSETS_BWD[44] = {
     0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    30, 31, 32,
    48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536
};

// ── Helper: bf16 dot for HD-dim vectors ─────────────────────────────────────
__device__ __forceinline__ float bf16_dot_bwd(
        const __nv_bfloat16* a,
        const __nv_bfloat16* b,
        int HD)
{
    float acc = 0.f;
    for (int d = 0; d + 1 < HD; d += 2) {
        __nv_bfloat162 a2 = *reinterpret_cast<const __nv_bfloat162*>(a + d);
        __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(b + d);
        float2 af = __bfloat1622float2(a2);
        float2 bf2 = __bfloat1622float2(b2);
        acc += af.x * bf2.x + af.y * bf2.y;
    }
    if (HD & 1) {
        acc += __bfloat162float(a[HD-1]) * __bfloat162float(b[HD-1]);
    }
    return acc;
}

// ── dKdV backward kernel ─────────────────────────────────────────────────────
template<int BLOCK_N, int HD>
__global__ void dsqg_bwd_dkdv_kernel(
    // Forward inputs
    const __nv_bfloat16* __restrict__ Q,      // [B,H,N,HD]
    const __nv_bfloat16* __restrict__ K,      // [B,H,N,HD]
    const __nv_bfloat16* __restrict__ V,      // [B,H,N,HD]
    const float*         __restrict__ PB,     // [44,H]
    const float*         __restrict__ SE,     // [44,HD]
    const float*         __restrict__ PHASE_BASE, // [11,H,2]
    const float*         __restrict__ PHASE_GAIN, // [11,H,2]
    const float*         __restrict__ Y_PRE,      // [B,H,N,2]
    const float*         __restrict__ Z_PRE,      // [B,H,N,2]
    const float*         __restrict__ LSE,         // [B,H,N]
    const float*         __restrict__ COSSIN,      // [B,H,N,11,2] saved theta
    // Forward output gradient
    const __nv_bfloat16* __restrict__ DOUT,   // [B,H,N,HD]
    // D = sum_j a_j * dout · v_j  (precomputed softmax auxiliary)
    const float*         __restrict__ D,      // [B,H,N]
    // Outputs
    float*               __restrict__ DK,     // [B,H,N,HD]  fp32
    float*               __restrict__ DV,     // [B,H,N,HD]  fp32
    // Phase gradients — buffer [B*H, N_tiles, 11*2]  no atomics
    float*               __restrict__ DPHASE_BASE_BUF, // [B*H, ceil(N/BN), 11*2]
    float*               __restrict__ DPHASE_GAIN_BUF, // same shape
    float*               __restrict__ DZ_PRE,          // [B,H,N,2]  atomic
    int B, int H, int N,
    float scale)
{
    const int bh       = blockIdx.x;
    const int tile_idx = blockIdx.y;
    const int tid      = threadIdx.x;

    const int b = bh / H;
    const int h = bh % H;
    const int t_start = tile_idx * BLOCK_N;
    const int t       = t_start + tid;   // key position this thread owns

    // ── Base pointers ─────────────────────────────────────────────────────────
    const long long bh_stride = (long long)N * HD;
    const __nv_bfloat16* Q_bh    = Q    + (long long)bh * bh_stride;
    const __nv_bfloat16* K_bh    = K    + (long long)bh * bh_stride;
    const __nv_bfloat16* V_bh    = V    + (long long)bh * bh_stride;
    const __nv_bfloat16* DOUT_bh = DOUT + (long long)bh * bh_stride;
          float*         DK_bh   = DK   + (long long)bh * bh_stride;
          float*         DV_bh   = DV   + (long long)bh * bh_stride;
    const float*         LSE_bh  = LSE  + (long long)bh * N;
    const float*         D_bh    = D    + (long long)bh * N;
    const float*         Y_bh    = Y_PRE + (long long)bh * N * 2;
    const float*         Z_bh    = Z_PRE + (long long)bh * N * 2;
    const float*         CS_bh   = COSSIN + (long long)bh * N * N_SPARSE * 2;
          float*         DZ_bh   = DZ_PRE + (long long)bh * N * 2;

    // ── Shared memory layout ─────────────────────────────────────────────────
    // Q window:    [(BLOCK_N + MAX_DENSE_DELTA), HD+1] bf16  = [96, 65] = 12.5 KB
    // DOUT window: [(BLOCK_N + MAX_DENSE_DELTA), HD+1] bf16  = 12.5 KB
    // K tile:      [BLOCK_N, HD+1] bf16                      = 8.3 KB
    // V tile:      [BLOCK_N, HD+1] bf16                      = 8.3 KB
    // Total:  41.6 KB
    const int Q_WIN    = BLOCK_N + MAX_DENSE_DELTA;   // 96
    const int smem_row = HD + 2;  // +2: even → 4-byte aligned for bf16x2 loads

    extern __shared__ char smem_raw[];
    __nv_bfloat16* Q_smem    = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* DOUT_smem = Q_smem    + Q_WIN   * smem_row;
    __nv_bfloat16* K_smem    = DOUT_smem + Q_WIN   * smem_row;
    __nv_bfloat16* V_smem    = K_smem    + BLOCK_N * smem_row;

    // ── Load K and V tiles into shared memory (once, reused for all offsets) ─
    if (t < N) {
        const __nv_bfloat16* k_src = K_bh + (long long)t * HD;
        const __nv_bfloat16* v_src = V_bh + (long long)t * HD;
        __nv_bfloat16* k_dst = K_smem + tid * smem_row;
        __nv_bfloat16* v_dst = V_smem + tid * smem_row;
        for (int d = 0; d < HD; d++) { k_dst[d] = k_src[d]; v_dst[d] = v_src[d]; }
    } else {
        __nv_bfloat16* k_dst = K_smem + tid * smem_row;
        __nv_bfloat16* v_dst = V_smem + tid * smem_row;
        for (int d = 0; d < HD; d++) { k_dst[d] = __float2bfloat16(0.f); v_dst[d] = __float2bfloat16(0.f); }
    }

    // ── Load Q and dout window [t_start, t_start + Q_WIN) ───────────────────
    // Each thread loads one row; for Q_WIN=96 > BLOCK_N=64 threads,
    // thread tid also loads row tid+BLOCK_N when tid < MAX_DENSE_DELTA (32 threads).
    {
        // Primary load: row tid
        int pos = t_start + tid;
        __nv_bfloat16* q_dst    = Q_smem    + tid * smem_row;
        __nv_bfloat16* dout_dst = DOUT_smem + tid * smem_row;
        if (pos < N) {
            const __nv_bfloat16* q_src    = Q_bh    + (long long)pos * HD;
            const __nv_bfloat16* dout_src = DOUT_bh + (long long)pos * HD;
            for (int d = 0; d < HD; d++) { q_dst[d] = q_src[d]; dout_dst[d] = dout_src[d]; }
        } else {
            for (int d = 0; d < HD; d++) { q_dst[d] = __float2bfloat16(0.f); dout_dst[d] = __float2bfloat16(0.f); }
        }
        // Secondary load: row tid+BLOCK_N (only first MAX_DENSE_DELTA threads do this)
        if (tid < MAX_DENSE_DELTA) {
            int pos2 = t_start + BLOCK_N + tid;
            __nv_bfloat16* q_dst2    = Q_smem    + (BLOCK_N + tid) * smem_row;
            __nv_bfloat16* dout_dst2 = DOUT_smem + (BLOCK_N + tid) * smem_row;
            if (pos2 < N) {
                const __nv_bfloat16* q_src2    = Q_bh    + (long long)pos2 * HD;
                const __nv_bfloat16* dout_src2 = DOUT_bh + (long long)pos2 * HD;
                for (int d = 0; d < HD; d++) { q_dst2[d] = q_src2[d]; dout_dst2[d] = dout_src2[d]; }
            } else {
                for (int d = 0; d < HD; d++) { q_dst2[d] = __float2bfloat16(0.f); dout_dst2[d] = __float2bfloat16(0.f); }
            }
        }
    }
    __syncthreads();

    // ── Per-thread accumulators (in registers) ────────────────────────────────
    float dk_acc[HD], dv_acc[HD];
    #pragma unroll
    for (int i = 0; i < HD; i++) { dk_acc[i] = 0.f; dv_acc[i] = 0.f; }

    // Phase gradient buffer accumulators for this block (sparse offsets only)
    float dpb_acc[N_SPARSE * 2], dpg_acc[N_SPARSE * 2];
    #pragma unroll
    for (int i = 0; i < N_SPARSE * 2; i++) { dpb_acc[i] = 0.f; dpg_acc[i] = 0.f; }
    float dz0_acc = 0.f, dz1_acc = 0.f;

    // ── Main loop over all 44 offsets ─────────────────────────────────────────
    #pragma unroll 1
    for (int j = 0; j < N_OFFSETS; j++) {
        const int delta = ALL_OFFSETS_BWD[j];
        const int n     = t + delta;    // query position for this (t, offset)
        const bool valid = (t < N) && (n < N);

        // Load query and dout: from smem for dense offsets, global for sparse
        __nv_bfloat16 q_local[HD], dout_local[HD];

        if (j < N_DENSE) {
            // Dense: query at index tid+delta in the smem window
            const __nv_bfloat16* q_ptr    = Q_smem    + (tid + delta) * smem_row;
            const __nv_bfloat16* dout_ptr = DOUT_smem + (tid + delta) * smem_row;
            #pragma unroll
            for (int d = 0; d < HD; d++) { q_local[d] = q_ptr[d]; dout_local[d] = dout_ptr[d]; }
        } else {
            // Sparse: load from global memory
            if (valid) {
                const __nv_bfloat16* q_ptr    = Q_bh    + (long long)n * HD;
                const __nv_bfloat16* dout_ptr = DOUT_bh + (long long)n * HD;
                #pragma unroll
                for (int d = 0; d < HD; d++) { q_local[d] = q_ptr[d]; dout_local[d] = dout_ptr[d]; }
            } else {
                #pragma unroll
                for (int d = 0; d < HD; d++) { q_local[d] = __float2bfloat16(0.f); dout_local[d] = __float2bfloat16(0.f); }
            }
        }

        if (!valid) continue;

        // ── Compute score ─────────────────────────────────────────────────────
        float lse_n = LSE_bh[n];
        float D_n   = D_bh[n];

        const __nv_bfloat16* k_smem_row = K_smem + tid * smem_row;
        float score = bf16_dot_bwd(q_local, k_smem_row, HD) * scale;
        score += PB[j * H + h];
        // scale_embed
        const float* se_row = SE + j * HD;
        float se_dot = 0.f;
        #pragma unroll
        for (int d = 0; d < HD; d++) {
            se_dot += __bfloat162float(q_local[d]) * se_row[d];
        }
        score += se_dot * scale;

        float alpha = expf(score - lse_n);

        // ── Sparse offsets: MOVT rotation ──────────────────────────────────────
        float cos0 = 1.f, sin0 = 0.f, cos1 = 1.f, sin1 = 0.f;
        int si = -1;
        if (j >= N_DENSE) {
            si = j - N_DENSE;
            float theta0 = CS_bh[(n * N_SPARSE + si) * 2 + 0];
            float theta1 = CS_bh[(n * N_SPARSE + si) * 2 + 1];
            __sincosf(theta0, &sin0, &cos0);
            __sincosf(theta1, &sin1, &cos1);
        }

        // ── dot(dout, rotated_V) ──────────────────────────────────────────────
        const __nv_bfloat16* v_smem_row = V_smem + tid * smem_row;
        float dot_rv;
        if (j >= N_DENSE) {
            float v0 = __bfloat162float(v_smem_row[0]);
            float v1 = __bfloat162float(v_smem_row[1]);
            float v2 = __bfloat162float(v_smem_row[2]);
            float v3 = __bfloat162float(v_smem_row[3]);
            float rv0 = cos0 * v0 - sin0 * v1;
            float rv1 = sin0 * v0 + cos0 * v1;
            float rv2 = cos1 * v2 - sin1 * v3;
            float rv3 = sin1 * v2 + cos1 * v3;
            dot_rv = __bfloat162float(dout_local[0]) * rv0
                   + __bfloat162float(dout_local[1]) * rv1
                   + __bfloat162float(dout_local[2]) * rv2
                   + __bfloat162float(dout_local[3]) * rv3;
            for (int d = 4; d < HD; d++) {
                dot_rv += __bfloat162float(dout_local[d]) * __bfloat162float(v_smem_row[d]);
            }
        } else {
            dot_rv = bf16_dot_bwd(dout_local, v_smem_row, HD);
        }
        float ds_v = alpha * (dot_rv - D_n);

        // ── dK accumulation ───────────────────────────────────────────────────
        float sc = scale;
        #pragma unroll
        for (int d = 0; d < HD; d++) {
            dk_acc[d] += ds_v * __bfloat162float(q_local[d]) * sc;
        }

        // ── dV accumulation (inverse rotation for sparse) ─────────────────────
        // d_local[d] = alpha * dout[d] (dense) or inverse-rotated (sparse)
        if (j >= N_DENSE) {
            float do0 = __bfloat162float(dout_local[0]);
            float do1 = __bfloat162float(dout_local[1]);
            float do2 = __bfloat162float(dout_local[2]);
            float do3 = __bfloat162float(dout_local[3]);
            // Inverse rotation: R(-theta) = [cos, sin; -sin, cos]
            dv_acc[0] += alpha * ( cos0 * do0 + sin0 * do1);
            dv_acc[1] += alpha * (-sin0 * do0 + cos0 * do1);
            dv_acc[2] += alpha * ( cos1 * do2 + sin1 * do3);
            dv_acc[3] += alpha * (-sin1 * do2 + cos1 * do3);
            for (int d = 4; d < HD; d++) {
                dv_acc[d] += alpha * __bfloat162float(dout_local[d]);
            }

            // ── d_theta (→ d_phase_base, d_phase_gain, dz_pre) ───────────────
            float v0 = __bfloat162float(v_smem_row[0]);
            float v1 = __bfloat162float(v_smem_row[1]);
            float v2 = __bfloat162float(v_smem_row[2]);
            float v3 = __bfloat162float(v_smem_row[3]);
            // d_theta_0: dout · rotate_perp_0(v)  where rotate_perp_0 = d/dθ rotate(v,θ) at plane (0,1)
            float dth0 = alpha * (do0 * (-v0 * sin0 - v1 * cos0) +
                                  do1 * ( v0 * cos0 - v1 * sin0));
            float dth1 = alpha * (do2 * (-v2 * sin1 - v3 * cos1) +
                                  do3 * ( v2 * cos1 - v3 * sin1));

            // Accumulate phase gradients (blocked per key tile, reduced after kernel)
            dpb_acc[si * 2 + 0] += dth0;
            dpb_acc[si * 2 + 1] += dth1;

            // d_phase_gain: also needs y_pre at query position n
            float y0 = Y_bh[n * 2 + 0];
            float y1 = Y_bh[n * 2 + 1];
            float z0 = Z_bh[t * 2 + 0];
            float z1 = Z_bh[t * 2 + 1];
            float pg0 = PHASE_GAIN[(si * H + h) * 2 + 0];
            float pg1 = PHASE_GAIN[(si * H + h) * 2 + 1];
            dpg_acc[si * 2 + 0] += dth0 * y0 * z0;
            dpg_acc[si * 2 + 1] += dth1 * y1 * z1;

            // dz_pre: atomic (multiple n positions write to same t position via different offsets)
            dz0_acc += dth0 * pg0 * y0;
            dz1_acc += dth1 * pg1 * y1;
        } else {
            // Dense: plain accumulation
            #pragma unroll
            for (int d = 0; d < HD; d++) {
                dv_acc[d] += alpha * __bfloat162float(dout_local[d]);
            }
        }
    }

    // Barrier: ensure all threads have exited the main loop before reusing V_smem
    // as the phase gradient reduction buffer.
    __syncthreads();

    // ── Write dK, dV (no atomics — each thread owns its key position t) ───────
    if (t < N) {
        float* dk_row = DK_bh + (long long)t * HD;
        float* dv_row = DV_bh + (long long)t * HD;
        #pragma unroll
        for (int d = 0; d < HD; d++) { dk_row[d] = dk_acc[d]; dv_row[d] = dv_acc[d]; }
    }

    // ── Write phase gradient buffer (one slot per [block, sparse_offset]) ─────
    // Layout: DPHASE_BASE_BUF[bh, tile_idx, si*2+m]
    const int n_tiles = (N + BLOCK_N - 1) / BLOCK_N;
    float* dpb_buf = DPHASE_BASE_BUF + (long long)(bh * n_tiles + tile_idx) * (N_SPARSE * 2);
    float* dpg_buf = DPHASE_GAIN_BUF + (long long)(bh * n_tiles + tile_idx) * (N_SPARSE * 2);
    // Thread 0 writes the accumulated phase gradients for this tile
    // (accumulated across all BLOCK_N key positions in this block)
    // We need ALL threads to contribute to a single tile-level sum.
    // Use shared memory reduction for the phase grad across threads in the block.
    //
    // Since dpb_acc[si,m] is per-thread (each thread accumulated over its 44 offsets),
    // we need to sum across threads: for each (si,m), sum over tid.
    //
    // Reuse V_smem (no longer needed) as reduction buffer: [BLOCK_N, N_SPARSE*2] fp32
    float* phase_reduce = reinterpret_cast<float*>(V_smem);  // BLOCK_N * N_SPARSE*2 floats

    // Each thread writes its phase accumulators into the reduction buffer
    for (int i = 0; i < N_SPARSE * 2; i++) {
        phase_reduce[tid * (N_SPARSE * 2) + i] = dpb_acc[i];
    }
    __syncthreads();

    // Thread 0..N_SPARSE*2-1 each reduce one element across all BLOCK_N threads
    if (tid < N_SPARSE * 2) {
        float sum = 0.f;
        for (int thr = 0; thr < BLOCK_N; thr++) {
            sum += phase_reduce[thr * (N_SPARSE * 2) + tid];
        }
        dpb_buf[tid] = sum;
    }
    __syncthreads();

    for (int i = 0; i < N_SPARSE * 2; i++) {
        phase_reduce[tid * (N_SPARSE * 2) + i] = dpg_acc[i];
    }
    __syncthreads();
    if (tid < N_SPARSE * 2) {
        float sum = 0.f;
        for (int thr = 0; thr < BLOCK_N; thr++) {
            sum += phase_reduce[thr * (N_SPARSE * 2) + tid];
        }
        dpg_buf[tid] = sum;
    }

    // ── dz_pre: atomic add (multiple key tiles write to same t position) ──────
    if (t < N) {
        atomicAdd(DZ_bh + t * 2 + 0, dz0_acc);
        atomicAdd(DZ_bh + t * 2 + 1, dz1_acc);
    }
}

// ── dQ backward kernel ────────────────────────────────────────────────────────
template<int BLOCK_N, int HD>
__global__ void dsqg_bwd_dq_kernel(
    const __nv_bfloat16* __restrict__ Q,          // [B,H,N,HD]
    const __nv_bfloat16* __restrict__ K,          // [B,H,N,HD]
    const __nv_bfloat16* __restrict__ V,          // [B,H,N,HD]
    const float*         __restrict__ PB,         // [44,H]
    const float*         __restrict__ SE,         // [44,HD]
    const float*         __restrict__ PHASE_BASE, // [11,H,2]
    const float*         __restrict__ PHASE_GAIN, // [11,H,2]
    const float*         __restrict__ Y_PRE,      // [B,H,N,2]
    const float*         __restrict__ Z_PRE,      // [B,H,N,2]
    const float*         __restrict__ LSE,        // [B,H,N]
    const float*         __restrict__ COSSIN,     // [B,H,N,11,2]
    const __nv_bfloat16* __restrict__ DOUT,       // [B,H,N,HD]
    const float*         __restrict__ D,          // [B,H,N]
    // Outputs
    float*               __restrict__ DQ,         // [B,H,N,HD]
    float*               __restrict__ DPB,        // [44,H]   atomic
    float*               __restrict__ DSE,        // [44,HD]  atomic
    float*               __restrict__ DY_PRE,     // [B,H,N,2] atomic (no contention within block)
    int B, int H, int N,
    float scale)
{
    const int bh       = blockIdx.x;
    const int tile_idx = blockIdx.y;
    const int tid      = threadIdx.x;

    const int b = bh / H;
    const int h = bh % H;
    const int n_start = tile_idx * BLOCK_N;
    const int n       = n_start + tid;   // query position this thread owns

    const long long bh_stride = (long long)N * HD;
    const __nv_bfloat16* Q_bh    = Q    + (long long)bh * bh_stride;
    const __nv_bfloat16* K_bh    = K    + (long long)bh * bh_stride;
    const __nv_bfloat16* V_bh    = V    + (long long)bh * bh_stride;
    const __nv_bfloat16* DOUT_bh = DOUT + (long long)bh * bh_stride;
          float*         DQ_bh   = DQ   + (long long)bh * bh_stride;
    const float*         LSE_bh  = LSE  + (long long)bh * N;
    const float*         D_bh    = D    + (long long)bh * N;
    const float*         Y_bh    = Y_PRE + (long long)bh * N * 2;
    const float*         Z_bh    = Z_PRE + (long long)bh * N * 2;
    const float*         CS_bh   = COSSIN + (long long)bh * N * N_SPARSE * 2;
          float*         DY_bh   = DY_PRE + (long long)bh * N * 2;

    // Shared memory: KV tile for one offset — [BLOCK_N, HD+1] bf16 (reused)
    const int smem_row = HD + 2;  // +2: even → 4-byte aligned for bf16x2 loads
    extern __shared__ char smem_raw[];
    __nv_bfloat16* KV_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);

    // Per-thread state
    float dq_acc[HD];
    #pragma unroll
    for (int i = 0; i < HD; i++) dq_acc[i] = 0.f;

    float lse_n = (n < N) ? LSE_bh[n] : 0.f;
    float D_n   = (n < N) ? D_bh[n]   : 0.f;
    float y0    = (n < N) ? Y_bh[n * 2 + 0] : 0.f;
    float y1    = (n < N) ? Y_bh[n * 2 + 1] : 0.f;
    float dy0_acc = 0.f, dy1_acc = 0.f;

    // Load query for this thread
    __nv_bfloat16 q_local[HD];
    if (n < N) {
        const __nv_bfloat16* q_src = Q_bh + (long long)n * HD;
        #pragma unroll
        for (int d = 0; d < HD; d++) q_local[d] = q_src[d];
    }

    // Load dout for this thread
    __nv_bfloat16 dout_local[HD];
    if (n < N) {
        const __nv_bfloat16* do_src = DOUT_bh + (long long)n * HD;
        #pragma unroll
        for (int d = 0; d < HD; d++) dout_local[d] = do_src[d];
    }

    #pragma unroll 1
    for (int j = 0; j < N_OFFSETS; j++) {
        const int delta = ALL_OFFSETS_BWD[j];
        const int t     = n - delta;
        const bool valid = (n < N) && (t >= 0);

        // Load K[t] into KV_smem (coalesced: threads access K[n_start-delta..n_end-delta])
        __nv_bfloat16* kv_row = KV_smem + tid * smem_row;
        if (valid) {
            const __nv_bfloat16* k_src = K_bh + (long long)t * HD;
            #pragma unroll
            for (int d = 0; d < HD; d++) kv_row[d] = k_src[d];
        } else {
            #pragma unroll
            for (int d = 0; d < HD; d++) kv_row[d] = __float2bfloat16(0.f);
        }
        // dQ kernel: each thread uses its OWN kv_row (tid * smem_row) — no cross-thread
        // sharing, so no __syncthreads__ needed anywhere inside this loop.
        if (!valid) continue;

        // Compute score and alpha
        float score = bf16_dot_bwd(q_local, kv_row, HD) * scale;
        score += PB[j * H + h];
        float se_dot = 0.f;
        const float* se_row = SE + j * HD;
        #pragma unroll
        for (int d = 0; d < HD; d++) {
            se_dot += __bfloat162float(q_local[d]) * se_row[d];
        }
        score += se_dot * scale;
        float alpha = expf(score - lse_n);

        // Load V[t] into KV_smem (reuse slot)
        {
            const __nv_bfloat16* v_src = V_bh + (long long)t * HD;
            #pragma unroll
            for (int d = 0; d < HD; d++) kv_row[d] = v_src[d];
        }

        // Sparse: get rotation angles from COSSIN
        float cos0 = 1.f, sin0 = 0.f, cos1 = 1.f, sin1 = 0.f;
        int si = -1;
        if (j >= N_DENSE) {
            si = j - N_DENSE;
            float theta0 = CS_bh[(n * N_SPARSE + si) * 2 + 0];
            float theta1 = CS_bh[(n * N_SPARSE + si) * 2 + 1];
            __sincosf(theta0, &sin0, &cos0);
            __sincosf(theta1, &sin1, &cos1);
        }

        // dot(dout, rotated_V)
        float dot_rv;
        if (j >= N_DENSE) {
            float v0 = __bfloat162float(kv_row[0]);
            float v1 = __bfloat162float(kv_row[1]);
            float v2 = __bfloat162float(kv_row[2]);
            float v3 = __bfloat162float(kv_row[3]);
            float rv0 = cos0 * v0 - sin0 * v1;
            float rv1 = sin0 * v0 + cos0 * v1;
            float rv2 = cos1 * v2 - sin1 * v3;
            float rv3 = sin1 * v2 + cos1 * v3;
            float do0 = __bfloat162float(dout_local[0]);
            float do1 = __bfloat162float(dout_local[1]);
            float do2 = __bfloat162float(dout_local[2]);
            float do3 = __bfloat162float(dout_local[3]);
            dot_rv = do0*rv0 + do1*rv1 + do2*rv2 + do3*rv3;
            for (int d = 4; d < HD; d++) {
                dot_rv += __bfloat162float(dout_local[d]) * __bfloat162float(kv_row[d]);
            }
        } else {
            dot_rv = bf16_dot_bwd(dout_local, kv_row, HD);
        }

        float ds_v = alpha * (dot_rv - D_n);

        // dQ: ds_v * K * scale
        #pragma unroll
        for (int d = 0; d < HD; d++) {
            dq_acc[d] += ds_v * __bfloat162float(kv_row[d]) * scale;
        }

        // d_pos_bias: atomic add to DPB[j, h]
        atomicAdd(DPB + j * H + h, ds_v);

        // d_scale_embed: ds_v * q[d] * scale for each d
        // (atomic to DSE[j, d])
        float dse_scale = ds_v * scale;
        for (int d = 0; d < HD; d++) {
            atomicAdd(DSE + j * HD + d, dse_scale * __bfloat162float(q_local[d]));
        }

        // For sparse: d_y_pre contributions
        if (j >= N_DENSE) {
            float pg0 = PHASE_GAIN[(si * H + h) * 2 + 0];
            float pg1 = PHASE_GAIN[(si * H + h) * 2 + 1];
            float z0  = Z_bh[t * 2 + 0];
            float z1  = Z_bh[t * 2 + 1];
            float v0 = __bfloat162float(kv_row[0]);
            float v1 = __bfloat162float(kv_row[1]);
            float v2 = __bfloat162float(kv_row[2]);
            float v3 = __bfloat162float(kv_row[3]);
            float do0 = __bfloat162float(dout_local[0]);
            float do1 = __bfloat162float(dout_local[1]);
            float do2 = __bfloat162float(dout_local[2]);
            float do3 = __bfloat162float(dout_local[3]);
            // dth0 = alpha * (do0 * (-v0*sin0 - v1*cos0) + do1*(v0*cos0 - v1*sin0))
            float dth0 = alpha * (do0*(-v0*sin0 - v1*cos0) + do1*(v0*cos0 - v1*sin0));
            float dth1 = alpha * (do2*(-v2*sin1 - v3*cos1) + do3*(v2*cos1 - v3*sin1));
            // dy_pre[n,0] += dth0 * pg0 * z0
            dy0_acc += dth0 * pg0 * z0;
            dy1_acc += dth1 * pg1 * z1;
        }
    }  // end offset loop

    // Write dQ (no atomics, each thread owns its query position)
    if (n < N) {
        float* dq_row = DQ_bh + (long long)n * HD;
        #pragma unroll
        for (int d = 0; d < HD; d++) dq_row[d] = dq_acc[d];

        // Write dy_pre (no contention within block since each thread owns its n)
        atomicAdd(DY_bh + n * 2 + 0, dy0_acc);
        atomicAdd(DY_bh + n * 2 + 1, dy1_acc);
    }
}

// ── Launchers ────────────────────────────────────────────────────────────────
void dsqg_bwd_dkdv_launch(
    const void* Q, const void* K, const void* V,
    const float* PB, const float* SE,
    const float* PHASE_BASE, const float* PHASE_GAIN,
    const float* Y_PRE, const float* Z_PRE,
    const float* LSE, const float* COSSIN,
    const void* DOUT, const float* D,
    float* DK, float* DV,
    float* DPHASE_BASE_BUF, float* DPHASE_GAIN_BUF, float* DZ_PRE,
    int B, int H, int N, int HD, float scale, cudaStream_t stream)
{
    const int BLOCK_N = 64;
    dim3 grid(B * H, (N + BLOCK_N - 1) / BLOCK_N);
    dim3 block(BLOCK_N);

    // Shared memory:
    //   Q_smem    : [(BLOCK_N + MAX_DENSE_DELTA), HD+1] bf16 = [96, 65] * 2
    //   DOUT_smem : same
    //   K_smem    : [BLOCK_N, HD+1] bf16
    //   V_smem    : [BLOCK_N, HD+1] bf16 (also reused for phase reduce as fp32)
    // phase_reduce needs: BLOCK_N * N_SPARSE * 2 * 4 bytes = 64 * 22 * 4 = 5.6 KB
    // V_smem as fp32: [BLOCK_N, HD+1] = 64*65*4 = 16.6 KB -- big enough for reduce
    const int Q_WIN = BLOCK_N + MAX_DENSE_DELTA;
    size_t smem = (size_t)(2 * Q_WIN + 2 * BLOCK_N) * (HD + 1) * sizeof(__nv_bfloat16);
    // Ensure enough for phase_reduce (reuses V_smem slot, which must be >= BLOCK_N*N_SPARSE*2 fp32)
    size_t phase_smem_needed = BLOCK_N * N_SPARSE * 2 * sizeof(float);
    size_t v_smem_bytes = (size_t)BLOCK_N * (HD + 1) * sizeof(__nv_bfloat16);
    if (phase_smem_needed > v_smem_bytes) {
        // Bump up smem allocation; happens only for very small HD
        smem += phase_smem_needed - v_smem_bytes;
    }

#define LAUNCH_DKDV(BN, hd) \
    dsqg_bwd_dkdv_kernel<BN, hd><<<grid, block, smem, stream>>>( \
        (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V, \
        PB, SE, PHASE_BASE, PHASE_GAIN, Y_PRE, Z_PRE, LSE, COSSIN, \
        (const __nv_bfloat16*)DOUT, D, DK, DV, \
        DPHASE_BASE_BUF, DPHASE_GAIN_BUF, DZ_PRE, B, H, N, scale)

    if      (HD == 32)  { LAUNCH_DKDV(64, 32);  }
    else if (HD == 64)  { LAUNCH_DKDV(64, 64);  }
    else if (HD == 96)  { LAUNCH_DKDV(64, 96);  }
    else if (HD == 128) { LAUNCH_DKDV(64, 128); }
#undef LAUNCH_DKDV
}

void dsqg_bwd_dq_launch(
    const void* Q, const void* K, const void* V,
    const float* PB, const float* SE,
    const float* PHASE_BASE, const float* PHASE_GAIN,
    const float* Y_PRE, const float* Z_PRE,
    const float* LSE, const float* COSSIN,
    const void* DOUT, const float* D,
    float* DQ, float* DPB, float* DSE, float* DY_PRE,
    int B, int H, int N, int HD, float scale, cudaStream_t stream)
{
    const int BLOCK_N = 64;
    dim3 grid(B * H, (N + BLOCK_N - 1) / BLOCK_N);
    dim3 block(BLOCK_N);
    // Shared: KV_smem [BLOCK_N, HD+1] bf16 (reused for K then V per offset)
    size_t smem = (size_t)BLOCK_N * (HD + 1) * sizeof(__nv_bfloat16);

#define LAUNCH_DQ(BN, hd) \
    dsqg_bwd_dq_kernel<BN, hd><<<grid, block, smem, stream>>>( \
        (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V, \
        PB, SE, PHASE_BASE, PHASE_GAIN, Y_PRE, Z_PRE, LSE, COSSIN, \
        (const __nv_bfloat16*)DOUT, D, DQ, DPB, DSE, DY_PRE, B, H, N, scale)

    if      (HD == 32)  { LAUNCH_DQ(64, 32);  }
    else if (HD == 64)  { LAUNCH_DQ(64, 64);  }
    else if (HD == 96)  { LAUNCH_DQ(64, 96);  }
    else if (HD == 128) { LAUNCH_DQ(64, 128); }
#undef LAUNCH_DQ
}
