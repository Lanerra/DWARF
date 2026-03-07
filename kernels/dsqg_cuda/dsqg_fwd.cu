/*
 * DSQG V5 — Forward CUDA Kernel
 * Sparse-only MOVT (offsets 33–43) + QK-OVT + dense local (offsets 0–32)
 *
 * Thread layout: one thread per query position within a BLOCK_N tile.
 * Grid:          (B*H, ceil(N, BLOCK_N))
 *
 * Each block:
 *   1. Loads Q[n_start:n_start+BLOCK_N, :] into shared memory (once)
 *   2. Online softmax over all 44 offsets (maintain running m, d per thread)
 *   3. For each offset j:
 *        - Load K tile [n-delta_j] → compute score → update online softmax
 *        - Load V tile [n-delta_j] → (rotate if sparse) → accumulate
 *   4. Write output + LSE
 *   5. Write saved cos/sin for sparse offsets [B,H,N,11,2] (backward reuse)
 *
 * Memory:  bf16 Q/K/V/O,  fp32 accumulators, pos_bias, scale_embed, phases
 * Padding: HD+1 in shared memory to avoid bank conflicts
 */

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>

// ── Constants ────────────────────────────────────────────────────────────────
#define MAX_OFFSET 1536
#define N_OFFSETS  44
#define N_DENSE    33
#define N_SPARSE   11

// All 44 causal offsets: 0..32 (dense) + 48,64,96,128,192,256,384,512,768,1024,1536 (sparse)
__constant__ int ALL_OFFSETS[44] = {
     0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    30, 31, 32,
    48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536
};

// ── Helper: bf16 dot product for HD elements ─────────────────────────────────
// q_row, k_row: pointers to HD bf16 elements; returns fp32 accumulation
__device__ __forceinline__ float bf16_dot(
        const __nv_bfloat16* q_row,
        const __nv_bfloat16* k_row,
        int HD)
{
    float acc = 0.f;
    // Process 2 elements at a time using bf162
    int d = 0;
    for (; d + 1 < HD; d += 2) {
        __nv_bfloat162 q2 = *reinterpret_cast<const __nv_bfloat162*>(q_row + d);
        __nv_bfloat162 k2 = *reinterpret_cast<const __nv_bfloat162*>(k_row + d);
        float2 qf = __bfloat1622float2(q2);
        float2 kf = __bfloat1622float2(k2);
        acc += qf.x * kf.x + qf.y * kf.y;
    }
    if (d < HD) {
        acc += __bfloat162float(q_row[d]) * __bfloat162float(k_row[d]);
    }
    return acc;
}

// ── Forward kernel ────────────────────────────────────────────────────────────
template<int BLOCK_N, int HD>
__global__ void dsqg_fwd_kernel(
    // Inputs
    const __nv_bfloat16* __restrict__ Q,    // [B,H,N,HD]
    const __nv_bfloat16* __restrict__ K,    // [B,H,N,HD]
    const __nv_bfloat16* __restrict__ V,    // [B,H,N,HD]
    const float*         __restrict__ PB,   // [44,H]   pos_bias
    const float*         __restrict__ SE,   // [44,HD]  scale_embed
    const float*         __restrict__ PHASE_BASE, // [11,H,2]
    const float*         __restrict__ PHASE_GAIN, // [11,H,2]
    const float*         __restrict__ Y_PRE,      // [B,H,N,2]
    const float*         __restrict__ Z_PRE,      // [B,H,N,2]
    // Outputs
    __nv_bfloat16*       __restrict__ OUT,    // [B,H,N,HD]
    float*               __restrict__ LSE,    // [B,H,N]
    float*               __restrict__ COSSIN, // [B,H,N,11,2]  saved cos/sin
    // Dims
    int B, int H, int N,
    float scale)             // 1/sqrt(HD)
{
    // ── Block / thread IDs ───────────────────────────────────────────────────
    const int bh       = blockIdx.x;               // combined (b,h) index
    const int tile_idx = blockIdx.y;
    const int tid      = threadIdx.x;              // 0..BLOCK_N-1

    const int b = bh / H;
    const int h = bh % H;
    const int n_start = tile_idx * BLOCK_N;
    const int n       = n_start + tid;             // query position for this thread

    // ── Base pointers for this (b,h) ────────────────────────────────────────
    const long long bh_stride = (long long)N * HD;
    const __nv_bfloat16* Q_bh = Q + (long long)bh * bh_stride;
    const __nv_bfloat16* K_bh = K + (long long)bh * bh_stride;
    const __nv_bfloat16* V_bh = V + (long long)bh * bh_stride;
          __nv_bfloat16* O_bh = OUT + (long long)bh * bh_stride;
          float*         L_bh = LSE + (long long)bh * N;
    const float*         Y_bh = Y_PRE + (long long)bh * N * 2;
    const float*         Z_bh = Z_PRE + (long long)bh * N * 2;
          float*         CS_bh = COSSIN + (long long)bh * N * N_SPARSE * 2;

    // ── Shared memory layout ─────────────────────────────────────────────────
    // Q tile:  [BLOCK_N, HD+1] bf16
    // KV tile: [BLOCK_N, HD+1] bf16  (reused for K, then V)
    extern __shared__ char smem_raw[];
    const int smem_row = HD + 2;  // +2: even → 4-byte aligned for bf16x2 loads  // pad to avoid bank conflicts

    __nv_bfloat16* Q_smem  = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* KV_smem = Q_smem + BLOCK_N * smem_row;
    // fp32 row for scale_embed computation: [HD] per block (shared across threads)
    float* SE_smem = reinterpret_cast<float*>(KV_smem + BLOCK_N * smem_row);

    // ── Load Q tile into shared memory ──────────────────────────────────────
    if (n < N) {
        const __nv_bfloat16* q_src = Q_bh + (long long)n * HD;
        __nv_bfloat16* q_dst = Q_smem + tid * smem_row;
        for (int d = 0; d < HD; d++) q_dst[d] = q_src[d];
    } else {
        // Pad with zeros for out-of-bounds threads
        __nv_bfloat16* q_dst = Q_smem + tid * smem_row;
        for (int d = 0; d < HD; d++) q_dst[d] = __float2bfloat16(0.f);
    }
    __syncthreads();

    // ── Online softmax state ─────────────────────────────────────────────────
    float m = -1e38f;   // running max
    float d_sum = 0.f;  // running denominator
    float out_acc[HD];  // output accumulator
    #pragma unroll
    for (int i = 0; i < HD; i++) out_acc[i] = 0.f;

    const __nv_bfloat16* q_row = Q_smem + tid * smem_row;  // this thread's Q

    // ── Load y_pre for this query position (QK-OVT) ─────────────────────────
    float y0 = 0.f, y1 = 0.f;
    if (n < N) {
        y0 = Y_bh[n * 2 + 0];
        y1 = Y_bh[n * 2 + 1];
    }

    // ── Iterate over all 44 offsets ──────────────────────────────────────────
    #pragma unroll 1
    for (int j = 0; j < N_OFFSETS; j++) {
        const int delta = ALL_OFFSETS[j];
        const int t     = n - delta;           // key position

        // Causal mask: skip if t < 0
        const bool valid = (n < N) && (t >= 0);

        // ── Load K tile ──────────────────────────────────────────────────────
        // Each thread loads its own K row (coalesced: thread tid accesses
        // K[n_start+tid - delta], which is contiguous across threads)
        __nv_bfloat16* kv_row = KV_smem + tid * smem_row;
        if (valid) {
            const __nv_bfloat16* k_src = K_bh + (long long)t * HD;
            for (int d = 0; d < HD; d++) kv_row[d] = k_src[d];
        } else {
            for (int d = 0; d < HD; d++) kv_row[d] = __float2bfloat16(0.f);
        }
        __syncthreads();

        // ── Compute score ────────────────────────────────────────────────────
        float score = 0.f;
        if (valid) {
            // Q·K dot product
            score = bf16_dot(q_row, kv_row, HD) * scale;

            // pos_bias[j, h]
            score += PB[j * H + h];

            // scale_embed: (Q · se[j]) * scale
            // Load SE row for this offset into SE_smem (thread 0 loads for all)
            // Actually each thread computes its own SE dot product independently
            float se_dot = 0.f;
            const float* se_row = SE + j * HD;
            for (int d = 0; d < HD; d++) {
                se_dot += __bfloat162float(q_row[d]) * se_row[d];
            }
            score += se_dot * scale;
        }

        // ── Handle sparse offsets: MOVT rotation on V ────────────────────────
        float cos0 = 1.f, sin0 = 0.f, cos1 = 1.f, sin1 = 0.f;
        if (j >= N_DENSE && valid) {
            const int si = j - N_DENSE;  // sparse index 0..10
            // Load z_pre at key position t
            float z0 = Z_bh[t * 2 + 0];
            float z1 = Z_bh[t * 2 + 1];
            // theta = phase_base[si,h,m] + phase_gain[si,h,m]*y[n,m]*z[t,m]
            float pb0 = PHASE_BASE[(si * H + h) * 2 + 0];
            float pb1 = PHASE_BASE[(si * H + h) * 2 + 1];
            float pg0 = PHASE_GAIN[(si * H + h) * 2 + 0];
            float pg1 = PHASE_GAIN[(si * H + h) * 2 + 1];
            float theta0 = pb0 + pg0 * y0 * z0;
            float theta1 = pb1 + pg1 * y1 * z1;
            __sincosf(theta0, &sin0, &cos0);
            __sincosf(theta1, &sin1, &cos1);
            // Save for backward
            if (n < N) {
                CS_bh[(n * N_SPARSE + si) * 2 + 0] = cos0;
                CS_bh[(n * N_SPARSE + si) * 2 + 1] = cos1;
                // Store sin in second half — pack as [cos0, cos1, sin0, sin1]
                // Actually store separately: COSSIN[b,h,n,si,0]=theta0, [si,1]=theta1
                // Better: save cos and sin both planes in [B,H,N,11,4]
                // For now: [B,H,N,11,2] stores theta0 and theta1 and recomputes
                // cos/sin in backward (cheaper than saving 4 values per position)
                CS_bh[(n * N_SPARSE + si) * 2 + 0] = theta0;
                CS_bh[(n * N_SPARSE + si) * 2 + 1] = theta1;
            }
        }

        // ── Load V tile and apply rotation ───────────────────────────────────
        if (valid) {
            const __nv_bfloat16* v_src = V_bh + (long long)t * HD;
            for (int d = 0; d < HD; d++) kv_row[d] = v_src[d];
        }
        __syncthreads();

        // ── Online softmax update ─────────────────────────────────────────────
        float s = valid ? score : -1e38f;
        float m_new = fmaxf(m, s);
        float alpha  = valid ? expf(s - m_new) : 0.f;
        float scale_old = expf(m - m_new);  // rescale previous accumulator

        d_sum = scale_old * d_sum + alpha;

        // Rescale accumulator
        #pragma unroll
        for (int i = 0; i < HD; i++) out_acc[i] *= scale_old;

        // Accumulate: alpha * (rotated) V
        if (valid) {
            // For sparse offsets: apply Givens rotation in-register
            // Planes: (0,1) and (2,3)
            // For dense offsets: direct accumulation
            if (j >= N_DENSE) {
                // Sparse: rotate channels 0,1 and 2,3
                float v0 = __bfloat162float(kv_row[0]);
                float v1 = __bfloat162float(kv_row[1]);
                float v2 = __bfloat162float(kv_row[2]);
                float v3 = __bfloat162float(kv_row[3]);
                // Plane (0,1): [cos0, -sin0; sin0, cos0]
                out_acc[0] += alpha * (cos0 * v0 - sin0 * v1);
                out_acc[1] += alpha * (sin0 * v0 + cos0 * v1);
                // Plane (2,3): [cos1, -sin1; sin1, cos1]
                out_acc[2] += alpha * (cos1 * v2 - sin1 * v3);
                out_acc[3] += alpha * (sin1 * v2 + cos1 * v3);
                // Remaining channels: direct
                for (int i = 4; i < HD; i++) {
                    out_acc[i] += alpha * __bfloat162float(kv_row[i]);
                }
            } else {
                // Dense: direct accumulation
                for (int i = 0; i < HD; i++) {
                    out_acc[i] += alpha * __bfloat162float(kv_row[i]);
                }
            }
        }

        m = m_new;
        __syncthreads();  // ensure KV_smem reuse is safe before next iteration
    }

    // ── Write output ─────────────────────────────────────────────────────────
    if (n < N) {
        float lse_val = (d_sum > 0.f) ? (m + logf(d_sum)) : m;
        L_bh[n] = lse_val;

        float inv_d = (d_sum > 0.f) ? (1.f / d_sum) : 0.f;
        __nv_bfloat16* out_row = O_bh + (long long)n * HD;
        for (int i = 0; i < HD; i++) {
            out_row[i] = __float2bfloat16(out_acc[i] * inv_d);
        }
    }
}

// ── Launcher ─────────────────────────────────────────────────────────────────
// Called from dsqg_cuda_ext.cpp

void dsqg_fwd_launch(
    const void* Q, const void* K, const void* V,
    const float* PB, const float* SE,
    const float* PHASE_BASE, const float* PHASE_GAIN,
    const float* Y_PRE, const float* Z_PRE,
    void* OUT, float* LSE, float* COSSIN,
    int B, int H, int N, int HD,
    float scale, cudaStream_t stream)
{
    const int BLOCK_N = 64;
    dim3 grid(B * H, (N + BLOCK_N - 1) / BLOCK_N);
    dim3 block(BLOCK_N);

    // Shared memory: Q_smem + KV_smem (both [BLOCK_N, HD+1] bf16) + SE_smem [HD] fp32
    size_t smem = 2 * BLOCK_N * (HD + 1) * sizeof(__nv_bfloat16)
                + HD * sizeof(float);

    // Dispatch on HD (compile-time template parameter for #pragma unroll)
    if (HD == 32) {
        dsqg_fwd_kernel<64, 32><<<grid, block, smem, stream>>>(
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
            PB, SE, PHASE_BASE, PHASE_GAIN, Y_PRE, Z_PRE,
            (__nv_bfloat16*)OUT, LSE, COSSIN, B, H, N, scale);
    } else if (HD == 64) {
        dsqg_fwd_kernel<64, 64><<<grid, block, smem, stream>>>(
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
            PB, SE, PHASE_BASE, PHASE_GAIN, Y_PRE, Z_PRE,
            (__nv_bfloat16*)OUT, LSE, COSSIN, B, H, N, scale);
    } else if (HD == 96) {
        dsqg_fwd_kernel<64, 96><<<grid, block, smem, stream>>>(
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
            PB, SE, PHASE_BASE, PHASE_GAIN, Y_PRE, Z_PRE,
            (__nv_bfloat16*)OUT, LSE, COSSIN, B, H, N, scale);
    } else if (HD == 128) {
        dsqg_fwd_kernel<64, 128><<<grid, block, smem, stream>>>(
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
            PB, SE, PHASE_BASE, PHASE_GAIN, Y_PRE, Z_PRE,
            (__nv_bfloat16*)OUT, LSE, COSSIN, B, H, N, scale);
    }
}
