/*
 * DSQG Attention V3 — CUDA Kernel (d50 Pure Geometry, J=44)
 *
 * Offset set: δ ∈ [0..40, 128, 384, 1536]  (41 dense + 3 sparse = 44 total)
 *
 * Architecture: Warp-per-position, SE in shared memory, K/V from global (L2).
 *   Dense loop fully unrolled (#pragma unroll) to enable load pipelining.
 *   Dot products via warp-shuffle butterfly all-reduce.
 *
 * Score: (Q·K + Q·SE) / √HD + pos_bias.  Causal: gpos < 0 → -inf.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <vector>

static constexpr int J_TOTAL = 44;
static constexpr int DENSE_COUNT = 41;
static constexpr int SPARSE_COUNT = 3;
static constexpr int WARP_SZ = 32;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    return val;
}


// ─────────────────────────────────────────────────────────────────────────────
// Forward
// ─────────────────────────────────────────────────────────────────────────────
//
// Grid: (ceil(N/WPB), B*H)   Block: WPB*32
// Dense loop fully unrolled: K/V from global, SE from shared memory.

template<int WPB, int HD_T>
__global__ __launch_bounds__(WPB * WARP_SZ)
void dsqg_fwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const float* __restrict__ pos_bias,
    const float* __restrict__ scale_embed,
    __nv_bfloat16* __restrict__ Out,
    float* __restrict__ LSE,
    int H, int N
) {
    static constexpr int ELEMS = (HD_T + WARP_SZ - 1) / WARP_SZ;
    static constexpr int SE_PAD = HD_T + 1;

    const int bh = blockIdx.y;
    const int warp_id = threadIdx.x / WARP_SZ;
    const int lane = threadIdx.x % WARP_SZ;
    const int n = blockIdx.x * WPB + warp_id;
    const int h = bh % H;
    const float sc = rsqrtf(float(HD_T));
    const int base = bh * N * HD_T;

    __shared__ float SE_s[J_TOTAL][SE_PAD];
    for (int i = threadIdx.x; i < J_TOTAL * HD_T; i += WPB * WARP_SZ)
        SE_s[i / HD_T][i % HD_T] = scale_embed[i];
    __syncthreads();

    if (n >= N) return;

    float q_r[ELEMS];
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) {
        const int hd = lane + e * WARP_SZ;
        q_r[e] = (hd < HD_T) ? __bfloat162float(Q[base + n * HD_T + hd]) : 0.f;
    }

    float row_max = -FLT_MAX;
    float row_sum = 0.f;
    float acc[ELEMS];
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) acc[e] = 0.f;

    #pragma unroll
    for (int j = 0; j < DENSE_COUNT; j++) {
        const int gpos = n - j;
        const int safe_gpos = max(gpos, 0);

        float dot = 0.f;
        #pragma unroll
        for (int e = 0; e < ELEMS; e++) {
            const int hd = lane + e * WARP_SZ;
            const float k_val = (hd < HD_T) ? __bfloat162float(K[base + safe_gpos * HD_T + hd]) : 0.f;
            const float se_val = (hd < HD_T) ? SE_s[j][hd] : 0.f;
            dot += q_r[e] * (k_val + se_val);
        }
        dot = warp_reduce_sum(dot);
        float score = dot * sc + pos_bias[j * H + h];
        if (gpos < 0) score = -FLT_MAX;

        const float old_max = row_max;
        row_max = fmaxf(row_max, score);
        const float rescale = expf(old_max - row_max);
        const float exp_s = expf(score - row_max);
        row_sum = row_sum * rescale + exp_s;

        #pragma unroll
        for (int e = 0; e < ELEMS; e++) {
            acc[e] *= rescale;
            const int hd = lane + e * WARP_SZ;
            const float v_val = (hd < HD_T) ? __bfloat162float(V[base + safe_gpos * HD_T + hd]) : 0.f;
            acc[e] += exp_s * v_val;
        }
    }

    #pragma unroll
    for (int s = 0; s < SPARSE_COUNT; s++) {
        const int off = (s == 0) ? 128 : (s == 1) ? 384 : 1536;
        const int j = DENSE_COUNT + s;
        const int gpos = n - off;
        const bool valid = (gpos >= 0);
        const int safe = max(gpos, 0);

        float dot = 0.f;
        #pragma unroll
        for (int e = 0; e < ELEMS; e++) {
            const int hd = lane + e * WARP_SZ;
            const float k_val = (hd < HD_T) ? __bfloat162float(K[base + safe * HD_T + hd]) : 0.f;
            const float se_val = (hd < HD_T) ? SE_s[j][hd] : 0.f;
            dot += q_r[e] * (k_val + se_val);
        }
        dot = warp_reduce_sum(dot);
        float score = dot * sc + pos_bias[j * H + h];
        if (!valid) score = -FLT_MAX;

        const float old_max = row_max;
        row_max = fmaxf(row_max, score);
        const float rescale = expf(old_max - row_max);
        const float exp_s = expf(score - row_max);
        row_sum = row_sum * rescale + exp_s;

        #pragma unroll
        for (int e = 0; e < ELEMS; e++) {
            acc[e] *= rescale;
            const int hd = lane + e * WARP_SZ;
            const float v_val = (hd < HD_T) ? __bfloat162float(V[base + safe * HD_T + hd]) : 0.f;
            acc[e] += exp_s * v_val;
        }
    }

    const float inv_sum = (row_sum > 0.f) ? (1.f / row_sum) : 1.f;
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) {
        const int hd = lane + e * WARP_SZ;
        if (hd < HD_T)
            Out[base + n * HD_T + hd] = __float2bfloat16_rn(acc[e] * inv_sum);
    }
    if (lane == 0)
        LSE[bh * N + n] = row_max + logf(row_sum);
}


// ─────────────────────────────────────────────────────────────────────────────
// Compute D[n] = dot(dout[n], out[n])
// ─────────────────────────────────────────────────────────────────────────────

template<int BLOCK_N, int HD_T>
__global__ void dsqg_compute_D_kernel(
    const __nv_bfloat16* __restrict__ DO,
    const __nv_bfloat16* __restrict__ O,
    float* __restrict__ D,
    int H, int N
) {
    const int bh = blockIdx.y;
    const int n = blockIdx.x * BLOCK_N + threadIdx.x;
    if (n >= N) return;
    const int idx = (bh * N + n) * HD_T;

    float dot = 0.f;
    #pragma unroll
    for (int d = 0; d < HD_T; d++)
        dot += __bfloat162float(DO[idx + d]) * __bfloat162float(O[idx + d]);
    D[bh * N + n] = dot;
}


// ─────────────────────────────────────────────────────────────────────────────
// Backward: dQ + dPOS_BIAS + dSCALE_EMBED
// ─────────────────────────────────────────────────────────────────────────────
//
// Dense loop fully unrolled. K/V from global, SE from shared memory.

template<int WPB, int HD_T>
__global__ __launch_bounds__(WPB * WARP_SZ)
void dsqg_bwd_dq_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const float* __restrict__ pos_bias,
    const float* __restrict__ scale_embed,
    const __nv_bfloat16* __restrict__ DO,
    const float* __restrict__ LSE,
    const float* __restrict__ Dval,
    __nv_bfloat16* __restrict__ DQ,
    float* __restrict__ DPB,
    float* __restrict__ DSE,
    int H, int N
) {
    static constexpr int ELEMS = (HD_T + WARP_SZ - 1) / WARP_SZ;
    static constexpr int SE_PAD = HD_T + 1;

    const int bh = blockIdx.y;
    const int warp_id = threadIdx.x / WARP_SZ;
    const int lane = threadIdx.x % WARP_SZ;
    const int n = blockIdx.x * WPB + warp_id;
    const int h = bh % H;
    const float sc = rsqrtf(float(HD_T));
    const int base = bh * N * HD_T;

    __shared__ float SE_s[J_TOTAL][SE_PAD];
    for (int i = threadIdx.x; i < J_TOTAL * HD_T; i += WPB * WARP_SZ)
        SE_s[i / HD_T][i % HD_T] = scale_embed[i];
    __syncthreads();

    if (n >= N) return;

    float q_r[ELEMS], do_r[ELEMS];
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) {
        const int hd = lane + e * WARP_SZ;
        q_r[e]  = (hd < HD_T) ? __bfloat162float(Q[base + n * HD_T + hd]) : 0.f;
        do_r[e] = (hd < HD_T) ? __bfloat162float(DO[base + n * HD_T + hd]) : 0.f;
    }

    const float lse_n = LSE[bh * N + n];
    const float D_n = Dval[bh * N + n];

    float dq_acc[ELEMS];
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) dq_acc[e] = 0.f;

    #pragma unroll
    for (int j = 0; j < DENSE_COUNT; j++) {
        const int gpos = n - j;
        if (gpos < 0) continue;

        float k_plus_se[ELEMS];
        float dot = 0.f, do_v = 0.f;
        #pragma unroll
        for (int e = 0; e < ELEMS; e++) {
            const int hd = lane + e * WARP_SZ;
            const float k_val = (hd < HD_T) ? __bfloat162float(K[base + gpos * HD_T + hd]) : 0.f;
            const float se = (hd < HD_T) ? SE_s[j][hd] : 0.f;
            k_plus_se[e] = k_val + se;
            dot += q_r[e] * k_plus_se[e];
            const float v_val = (hd < HD_T) ? __bfloat162float(V[base + gpos * HD_T + hd]) : 0.f;
            do_v += do_r[e] * v_val;
        }
        dot = warp_reduce_sum(dot);
        do_v = warp_reduce_sum(do_v);

        const float weight = expf(dot * sc + pos_bias[j * H + h] - lse_n);
        const float d_score = weight * (do_v - D_n);
        const float ds_sc = d_score * sc;

        #pragma unroll
        for (int e = 0; e < ELEMS; e++)
            dq_acc[e] += ds_sc * k_plus_se[e];

        if (lane == 0) atomicAdd(&DPB[j * H + h], d_score);
        #pragma unroll
        for (int e = 0; e < ELEMS; e++) {
            const int hd = lane + e * WARP_SZ;
            if (hd < HD_T)
                atomicAdd(&DSE[j * HD_T + hd], ds_sc * q_r[e]);
        }
    }

    #pragma unroll
    for (int s = 0; s < SPARSE_COUNT; s++) {
        const int off = (s == 0) ? 128 : (s == 1) ? 384 : 1536;
        const int j = DENSE_COUNT + s;
        const int gpos = n - off;
        if (gpos < 0) continue;

        float k_plus_se[ELEMS];
        float dot = 0.f, do_v = 0.f;
        #pragma unroll
        for (int e = 0; e < ELEMS; e++) {
            const int hd = lane + e * WARP_SZ;
            const float k_val = (hd < HD_T) ? __bfloat162float(K[base + gpos * HD_T + hd]) : 0.f;
            const float se = (hd < HD_T) ? SE_s[j][hd] : 0.f;
            k_plus_se[e] = k_val + se;
            dot += q_r[e] * k_plus_se[e];
            const float v_val = (hd < HD_T) ? __bfloat162float(V[base + gpos * HD_T + hd]) : 0.f;
            do_v += do_r[e] * v_val;
        }
        dot = warp_reduce_sum(dot);
        do_v = warp_reduce_sum(do_v);

        const float weight = expf(dot * sc + pos_bias[j * H + h] - lse_n);
        const float d_score = weight * (do_v - D_n);
        const float ds_sc = d_score * sc;

        #pragma unroll
        for (int e = 0; e < ELEMS; e++)
            dq_acc[e] += ds_sc * k_plus_se[e];

        if (lane == 0) atomicAdd(&DPB[j * H + h], d_score);
        #pragma unroll
        for (int e = 0; e < ELEMS; e++) {
            const int hd = lane + e * WARP_SZ;
            if (hd < HD_T)
                atomicAdd(&DSE[j * HD_T + hd], ds_sc * q_r[e]);
        }
    }

    #pragma unroll
    for (int e = 0; e < ELEMS; e++) {
        const int hd = lane + e * WARP_SZ;
        if (hd < HD_T)
            DQ[base + n * HD_T + hd] = __float2bfloat16_rn(dq_acc[e]);
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// Backward: dK / dV
// ─────────────────────────────────────────────────────────────────────────────
//
// Dense loop fully unrolled. Q/DO from global, SE from shared memory.

template<int WPB, int HD_T>
__global__ __launch_bounds__(WPB * WARP_SZ)
void dsqg_bwd_dkdv_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const float* __restrict__ pos_bias,
    const float* __restrict__ scale_embed,
    const __nv_bfloat16* __restrict__ DO,
    const float* __restrict__ LSE,
    const float* __restrict__ Dval,
    __nv_bfloat16* __restrict__ DK,
    __nv_bfloat16* __restrict__ DV,
    int H, int N
) {
    static constexpr int ELEMS = (HD_T + WARP_SZ - 1) / WARP_SZ;
    static constexpr int SE_PAD = HD_T + 1;

    const int bh = blockIdx.y;
    const int warp_id = threadIdx.x / WARP_SZ;
    const int lane = threadIdx.x % WARP_SZ;
    const int m = blockIdx.x * WPB + warp_id;
    const int h = bh % H;
    const float sc = rsqrtf(float(HD_T));
    const int base = bh * N * HD_T;
    const int bhN = bh * N;

    __shared__ float SE_s[J_TOTAL][SE_PAD];
    for (int i = threadIdx.x; i < J_TOTAL * HD_T; i += WPB * WARP_SZ)
        SE_s[i / HD_T][i % HD_T] = scale_embed[i];
    __syncthreads();

    if (m >= N) return;

    float k_r[ELEMS], v_r[ELEMS];
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) {
        const int hd = lane + e * WARP_SZ;
        k_r[e] = (hd < HD_T) ? __bfloat162float(K[base + m * HD_T + hd]) : 0.f;
        v_r[e] = (hd < HD_T) ? __bfloat162float(V[base + m * HD_T + hd]) : 0.f;
    }

    float dk_acc[ELEMS], dv_acc[ELEMS];
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) { dk_acc[e] = 0.f; dv_acc[e] = 0.f; }

    #pragma unroll
    for (int j = 0; j < DENSE_COUNT; j++) {
        const int target_n = m + j;
        if (target_n >= N) continue;

        float q_val[ELEMS], do_val[ELEMS];
        float dot = 0.f, do_v = 0.f;
        #pragma unroll
        for (int e = 0; e < ELEMS; e++) {
            const int hd = lane + e * WARP_SZ;
            q_val[e]  = (hd < HD_T) ? __bfloat162float(Q[base + target_n * HD_T + hd]) : 0.f;
            do_val[e] = (hd < HD_T) ? __bfloat162float(DO[base + target_n * HD_T + hd]) : 0.f;
            const float se = (hd < HD_T) ? SE_s[j][hd] : 0.f;
            dot += q_val[e] * (k_r[e] + se);
            do_v += do_val[e] * v_r[e];
        }
        dot = warp_reduce_sum(dot);
        do_v = warp_reduce_sum(do_v);

        const float lse_n = LSE[bhN + target_n];
        const float D_n = Dval[bhN + target_n];
        const float weight = expf(dot * sc + pos_bias[j * H + h] - lse_n);
        const float ds_sc = weight * (do_v - D_n) * sc;

        #pragma unroll
        for (int e = 0; e < ELEMS; e++) {
            dk_acc[e] += ds_sc * q_val[e];
            dv_acc[e] += weight * do_val[e];
        }
    }

    #pragma unroll
    for (int s = 0; s < SPARSE_COUNT; s++) {
        const int off = (s == 0) ? 128 : (s == 1) ? 384 : 1536;
        const int j = DENSE_COUNT + s;
        const int target_n = m + off;
        if (target_n >= N) continue;

        float q_val[ELEMS], do_val[ELEMS];
        #pragma unroll
        for (int e = 0; e < ELEMS; e++) {
            const int hd = lane + e * WARP_SZ;
            q_val[e]  = (hd < HD_T) ? __bfloat162float(Q[base + target_n * HD_T + hd]) : 0.f;
            do_val[e] = (hd < HD_T) ? __bfloat162float(DO[base + target_n * HD_T + hd]) : 0.f;
        }

        const float lse_n = LSE[bhN + target_n];
        const float D_n = Dval[bhN + target_n];

        float dot = 0.f, do_v = 0.f;
        #pragma unroll
        for (int e = 0; e < ELEMS; e++) {
            const int hd = lane + e * WARP_SZ;
            const float se = (hd < HD_T) ? SE_s[j][hd] : 0.f;
            dot += q_val[e] * (k_r[e] + se);
            do_v += do_val[e] * v_r[e];
        }
        dot = warp_reduce_sum(dot);
        do_v = warp_reduce_sum(do_v);

        const float weight = expf(dot * sc + pos_bias[j * H + h] - lse_n);
        const float ds_sc = weight * (do_v - D_n) * sc;

        #pragma unroll
        for (int e = 0; e < ELEMS; e++) {
            dk_acc[e] += ds_sc * q_val[e];
            dv_acc[e] += weight * do_val[e];
        }
    }

    #pragma unroll
    for (int e = 0; e < ELEMS; e++) {
        const int hd = lane + e * WARP_SZ;
        if (hd < HD_T) {
            DK[base + m * HD_T + hd] = __float2bfloat16_rn(dk_acc[e]);
            DV[base + m * HD_T + hd] = __float2bfloat16_rn(dv_acc[e]);
        }
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// C++ entry points
// ─────────────────────────────────────────────────────────────────────────────

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be on CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_BF16(x) TORCH_CHECK(x.dtype() == torch::kBFloat16, #x " must be bfloat16")
#define CHECK_F32(x) TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must be float32")

static constexpr int FWD_WPB = 16;
static constexpr int BWD_DQ_WPB = 16;
static constexpr int BWD_DKDV_WPB = 16;
static constexpr int D_BLOCK = 256;

template<int HD_T>
static void launch_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor pos_bias, torch::Tensor scale_embed,
    torch::Tensor out, torch::Tensor lse,
    int B, int H, int N
) {
    const int BH = B * H;
    dim3 grid((N + FWD_WPB - 1) / FWD_WPB, BH);
    dim3 block(FWD_WPB * WARP_SZ);

    dsqg_fwd_kernel<FWD_WPB, HD_T><<<grid, block>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
        pos_bias.data_ptr<float>(),
        scale_embed.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        lse.data_ptr<float>(),
        H, N
    );
}

std::vector<torch::Tensor> dsqg_forward_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor pos_bias, torch::Tensor scale_embed
) {
    CHECK_CUDA(q); CHECK_CUDA(k); CHECK_CUDA(v);
    CHECK_CUDA(pos_bias); CHECK_CUDA(scale_embed);
    CHECK_BF16(q); CHECK_BF16(k); CHECK_BF16(v);
    CHECK_F32(pos_bias); CHECK_F32(scale_embed);
    CHECK_CONTIGUOUS(q); CHECK_CONTIGUOUS(k); CHECK_CONTIGUOUS(v);
    CHECK_CONTIGUOUS(pos_bias); CHECK_CONTIGUOUS(scale_embed);

    const int B = q.size(0), H = q.size(1), N = q.size(2), HD = q.size(3);
    TORCH_CHECK(pos_bias.size(0) == J_TOTAL && pos_bias.size(1) == H);
    TORCH_CHECK(scale_embed.size(0) == J_TOTAL && scale_embed.size(1) == HD);
    TORCH_CHECK(HD == 32 || HD == 48, "HD must be 32 or 48");

    auto out = torch::empty_like(q);
    auto lse = torch::empty({B, H, N}, q.options().dtype(torch::kFloat32));

    if (HD == 32) launch_forward<32>(q, k, v, pos_bias, scale_embed, out, lse, B, H, N);
    else          launch_forward<48>(q, k, v, pos_bias, scale_embed, out, lse, B, H, N);

    return {out, lse};
}

template<int HD_T>
static void launch_backward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor pos_bias, torch::Tensor scale_embed,
    torch::Tensor out, torch::Tensor lse, torch::Tensor dout,
    torch::Tensor dq, torch::Tensor dk, torch::Tensor dv,
    torch::Tensor dpb, torch::Tensor dse,
    int B, int H, int N
) {
    const int BH = B * H;

    auto D = torch::empty({B, H, N}, q.options().dtype(torch::kFloat32));
    {
        dim3 grid((N + D_BLOCK - 1) / D_BLOCK, BH);
        dim3 block(D_BLOCK);
        dsqg_compute_D_kernel<D_BLOCK, HD_T><<<grid, block>>>(
            reinterpret_cast<const __nv_bfloat16*>(dout.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(out.data_ptr()),
            D.data_ptr<float>(),
            H, N
        );
    }

    {
        dim3 grid((N + BWD_DQ_WPB - 1) / BWD_DQ_WPB, BH);
        dim3 block(BWD_DQ_WPB * WARP_SZ);
        dsqg_bwd_dq_kernel<BWD_DQ_WPB, HD_T><<<grid, block>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
            pos_bias.data_ptr<float>(),
            scale_embed.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16*>(dout.data_ptr()),
            lse.data_ptr<float>(),
            D.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(dq.data_ptr()),
            dpb.data_ptr<float>(),
            dse.data_ptr<float>(),
            H, N
        );
    }

    {
        dim3 grid((N + BWD_DKDV_WPB - 1) / BWD_DKDV_WPB, BH);
        dim3 block(BWD_DKDV_WPB * WARP_SZ);
        dsqg_bwd_dkdv_kernel<BWD_DKDV_WPB, HD_T><<<grid, block>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
            pos_bias.data_ptr<float>(),
            scale_embed.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16*>(dout.data_ptr()),
            lse.data_ptr<float>(),
            D.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(dk.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(dv.data_ptr()),
            H, N
        );
    }
}

std::vector<torch::Tensor> dsqg_backward_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor pos_bias, torch::Tensor scale_embed,
    torch::Tensor out, torch::Tensor lse, torch::Tensor dout
) {
    CHECK_CUDA(q); CHECK_CUDA(k); CHECK_CUDA(v);
    CHECK_CUDA(pos_bias); CHECK_CUDA(scale_embed);
    CHECK_CUDA(out); CHECK_CUDA(lse); CHECK_CUDA(dout);
    CHECK_BF16(q); CHECK_BF16(k); CHECK_BF16(v);
    CHECK_BF16(out); CHECK_BF16(dout);
    CHECK_F32(pos_bias); CHECK_F32(scale_embed); CHECK_F32(lse);
    CHECK_CONTIGUOUS(q); CHECK_CONTIGUOUS(k); CHECK_CONTIGUOUS(v);
    CHECK_CONTIGUOUS(dout); CHECK_CONTIGUOUS(out);
    CHECK_CONTIGUOUS(pos_bias); CHECK_CONTIGUOUS(scale_embed);

    const int B = q.size(0), H = q.size(1), N = q.size(2), HD = q.size(3);

    auto dq  = torch::empty_like(q);
    auto dk  = torch::empty_like(k);
    auto dv  = torch::empty_like(v);
    auto dpb = torch::zeros_like(pos_bias);
    auto dse = torch::zeros_like(scale_embed);

    if (HD == 32) launch_backward<32>(q,k,v,pos_bias,scale_embed,out,lse,dout,dq,dk,dv,dpb,dse,B,H,N);
    else          launch_backward<48>(q,k,v,pos_bias,scale_embed,out,lse,dout,dq,dk,dv,dpb,dse,B,H,N);

    return {dq, dk, dv, dpb, dse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &dsqg_forward_cuda, "DSQG V3 forward (CUDA)");
    m.def("backward", &dsqg_backward_cuda, "DSQG V3 backward (CUDA)");
}
