/*
 * DSQG V5 CUDA Extension — pybind11 bindings
 *
 * Exposes:
 *   dsqg_fwd(Q, K, V, PB, SE, PHASE_BASE, PHASE_GAIN, Y_PRE, Z_PRE, scale)
 *     → (OUT, LSE, COSSIN)
 *
 *   dsqg_bwd(Q, K, V, PB, SE, PHASE_BASE, PHASE_GAIN, Y_PRE, Z_PRE,
 *            LSE, COSSIN, DOUT, D, N_TILES, scale)
 *     → (DQ, DK, DV, DPHASE_BASE, DPHASE_GAIN, DY_PRE, DZ_PRE, DPBIAS, DSCALE)
 *
 * Tensors:
 *   Q, K, V, DOUT : bf16  [B, H, N, HD]
 *   OUT           : bf16  [B, H, N, HD]
 *   PB            : fp32  [44, H]
 *   SE            : fp32  [44, HD]
 *   PHASE_BASE    : fp32  [11, H, 2]
 *   PHASE_GAIN    : fp32  [11, H, 2]
 *   Y_PRE, Z_PRE  : fp32  [B, H, N, 2]
 *   LSE           : fp32  [B, H, N]
 *   COSSIN        : fp32  [B, H, N, 11, 2]  (saved theta per query, sparse offsets)
 *   D             : fp32  [B, H, N]  (precomputed softmax auxiliary = sum_j a_j * dout·v_j)
 *   DQ, DK, DV    : fp32  [B, H, N, HD]
 *   DPHASE_BASE   : fp32  [11, H, 2]
 *   DPHASE_GAIN   : fp32  [11, H, 2]
 *   DY_PRE        : fp32  [B, H, N, 2]
 *   DZ_PRE        : fp32  [B, H, N, 2]
 *   DPBIAS        : fp32  [44, H]
 *   DSCALE        : fp32  [44, HD]
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// Forward declarations of launchers defined in dsqg_fwd.cu and dsqg_bwd.cu
void dsqg_fwd_launch(
    const void* Q, const void* K, const void* V,
    const float* PB, const float* SE,
    const float* PHASE_BASE, const float* PHASE_GAIN,
    const float* Y_PRE, const float* Z_PRE,
    void* OUT, float* LSE, float* COSSIN,
    int B, int H, int N, int HD,
    float scale, cudaStream_t stream);

void dsqg_bwd_dkdv_launch(
    const void* Q, const void* K, const void* V,
    const float* PB, const float* SE,
    const float* PHASE_BASE, const float* PHASE_GAIN,
    const float* Y_PRE, const float* Z_PRE,
    const float* LSE, const float* COSSIN,
    const void* DOUT, const float* D,
    float* DK, float* DV,
    float* DPHASE_BASE_BUF, float* DPHASE_GAIN_BUF, float* DZ_PRE,
    int B, int H, int N, int HD, float scale, cudaStream_t stream);

void dsqg_bwd_dq_launch(
    const void* Q, const void* K, const void* V,
    const float* PB, const float* SE,
    const float* PHASE_BASE, const float* PHASE_GAIN,
    const float* Y_PRE, const float* Z_PRE,
    const float* LSE, const float* COSSIN,
    const void* DOUT, const float* D,
    float* DQ, float* DPB, float* DSE, float* DY_PRE,
    int B, int H, int N, int HD, float scale, cudaStream_t stream);

// ── Helpers ──────────────────────────────────────────────────────────────────
#define CHECK_CUDA(x)     TORCH_CHECK(x.is_cuda(),    #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_BF16(x)     TORCH_CHECK(x.dtype() == torch::kBFloat16, #x " must be bf16")
#define CHECK_F32(x)      TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must be fp32")

// ── Forward ──────────────────────────────────────────────────────────────────
std::vector<torch::Tensor> dsqg_fwd(
    torch::Tensor Q,           // [B,H,N,HD] bf16
    torch::Tensor K,           // [B,H,N,HD] bf16
    torch::Tensor V,           // [B,H,N,HD] bf16
    torch::Tensor PB,          // [44,H]     fp32
    torch::Tensor SE,          // [44,HD]    fp32
    torch::Tensor PHASE_BASE,  // [11,H,2]   fp32
    torch::Tensor PHASE_GAIN,  // [11,H,2]   fp32
    torch::Tensor Y_PRE,       // [B,H,N,2]  fp32
    torch::Tensor Z_PRE,       // [B,H,N,2]  fp32
    float scale)
{
    CHECK_CUDA(Q); CHECK_CONTIGUOUS(Q); CHECK_BF16(Q);
    CHECK_CUDA(K); CHECK_CONTIGUOUS(K); CHECK_BF16(K);
    CHECK_CUDA(V); CHECK_CONTIGUOUS(V); CHECK_BF16(V);

    int B  = Q.size(0), H = Q.size(1), N = Q.size(2), HD = Q.size(3);

    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(Q.device());
    auto opts_f32  = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());

    auto OUT    = torch::empty({B, H, N, HD},       opts_bf16);
    auto LSE    = torch::empty({B, H, N},            opts_f32);
    auto COSSIN = torch::empty({B, H, N, 11, 2},     opts_f32);

    auto stream = at::cuda::getCurrentCUDAStream();
    dsqg_fwd_launch(
        Q.data_ptr(), K.data_ptr(), V.data_ptr(),
        PB.data_ptr<float>(), SE.data_ptr<float>(),
        PHASE_BASE.data_ptr<float>(), PHASE_GAIN.data_ptr<float>(),
        Y_PRE.data_ptr<float>(), Z_PRE.data_ptr<float>(),
        OUT.data_ptr(), LSE.data_ptr<float>(), COSSIN.data_ptr<float>(),
        B, H, N, HD, scale, stream);

    return {OUT, LSE, COSSIN};
}

// ── Backward ─────────────────────────────────────────────────────────────────
std::vector<torch::Tensor> dsqg_bwd(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor PB,
    torch::Tensor SE,
    torch::Tensor PHASE_BASE,
    torch::Tensor PHASE_GAIN,
    torch::Tensor Y_PRE,
    torch::Tensor Z_PRE,
    torch::Tensor LSE,
    torch::Tensor COSSIN,
    torch::Tensor DOUT,
    torch::Tensor D,          // [B,H,N] precomputed by Python before calling
    float scale)
{
    int B  = Q.size(0), H = Q.size(1), N = Q.size(2), HD = Q.size(3);
    int n_tiles = (N + 63) / 64;   // BLOCK_N = 64

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());

    // Gradient tensors (initialized to zero)
    auto DQ = torch::zeros({B, H, N, HD},    opts);
    auto DK = torch::zeros({B, H, N, HD},    opts);
    auto DV = torch::zeros({B, H, N, HD},    opts);
    auto DPHASE_BASE_BUF = torch::zeros({B * H, n_tiles, 11 * 2}, opts);
    auto DPHASE_GAIN_BUF = torch::zeros({B * H, n_tiles, 11 * 2}, opts);
    auto DZ_PRE  = torch::zeros({B, H, N, 2},    opts);
    auto DY_PRE  = torch::zeros({B, H, N, 2},    opts);
    auto DPBIAS  = torch::zeros({44, H},           opts);
    auto DSCALE  = torch::zeros({44, HD},          opts);

    auto stream = at::cuda::getCurrentCUDAStream();

    // dKdV kernel
    dsqg_bwd_dkdv_launch(
        Q.data_ptr(), K.data_ptr(), V.data_ptr(),
        PB.data_ptr<float>(), SE.data_ptr<float>(),
        PHASE_BASE.data_ptr<float>(), PHASE_GAIN.data_ptr<float>(),
        Y_PRE.data_ptr<float>(), Z_PRE.data_ptr<float>(),
        LSE.data_ptr<float>(), COSSIN.data_ptr<float>(),
        DOUT.data_ptr(), D.data_ptr<float>(),
        DK.data_ptr<float>(), DV.data_ptr<float>(),
        DPHASE_BASE_BUF.data_ptr<float>(), DPHASE_GAIN_BUF.data_ptr<float>(),
        DZ_PRE.data_ptr<float>(),
        B, H, N, HD, scale, stream);

    // dQ kernel
    dsqg_bwd_dq_launch(
        Q.data_ptr(), K.data_ptr(), V.data_ptr(),
        PB.data_ptr<float>(), SE.data_ptr<float>(),
        PHASE_BASE.data_ptr<float>(), PHASE_GAIN.data_ptr<float>(),
        Y_PRE.data_ptr<float>(), Z_PRE.data_ptr<float>(),
        LSE.data_ptr<float>(), COSSIN.data_ptr<float>(),
        DOUT.data_ptr(), D.data_ptr<float>(),
        DQ.data_ptr<float>(), DPBIAS.data_ptr<float>(), DSCALE.data_ptr<float>(),
        DY_PRE.data_ptr<float>(),
        B, H, N, HD, scale, stream);

    // Reduce phase buffers: [B*H, n_tiles, 11*2] → [11, H, 2]
    // This is the Python-side reduction (cheap, O(BH * n_tiles * 22))
    // Returned as raw buffers; Python wrapper does .view(B,H,n_tiles,11,2).sum(dim=(0,2)).permute(...)
    // (same pattern as Triton kernel)

    return {DQ, DK, DV, DPHASE_BASE_BUF, DPHASE_GAIN_BUF, DY_PRE, DZ_PRE, DPBIAS, DSCALE};
}

// ── Module registration ───────────────────────────────────────────────────────
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DSQG V5 CUDA extension — sparse-only MOVT + QK-OVT";
    m.def("fwd", &dsqg_fwd,
          "DSQG forward pass",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("PB"), py::arg("SE"),
          py::arg("PHASE_BASE"), py::arg("PHASE_GAIN"),
          py::arg("Y_PRE"), py::arg("Z_PRE"),
          py::arg("scale"));
    m.def("bwd", &dsqg_bwd,
          "DSQG backward pass",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("PB"), py::arg("SE"),
          py::arg("PHASE_BASE"), py::arg("PHASE_GAIN"),
          py::arg("Y_PRE"), py::arg("Z_PRE"),
          py::arg("LSE"), py::arg("COSSIN"),
          py::arg("DOUT"), py::arg("D"),
          py::arg("scale"));
}
