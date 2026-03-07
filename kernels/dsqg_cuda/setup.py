"""
Build script for DSQG V5 CUDA extension.

Usage:
    cd kernels/dsqg_cuda
    python setup.py install
  or:
    python setup.py build_ext --inplace

Requires: PyTorch with CUDA, nvcc >= 11.0
Target GPU: RTX 4090 (Ada Lovelace, sm_89)
            RTX 3090 (Ampere, sm_86)
"""

import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# Detect compute capability from current GPU
def get_compute_capability():
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability()
        return f"{cc[0]}{cc[1]}"
    return "89"  # default: Ada Lovelace (4090)

cc = get_compute_capability()
arch_flag = f"-gencode=arch=compute_{cc},code=sm_{cc}"

# Also compile for sm_86 (3090) if primary is sm_89 (4090)
extra_arches = []
if cc == "89":
    extra_arches = ["-gencode=arch=compute_86,code=sm_86"]

nvcc_flags = [
    arch_flag,
    *extra_arches,
    "-O3",
    "--use_fast_math",              # enables __sincosf and fast exp
    "-DNDEBUG",
    "--ptxas-options=-v",           # verbose register/smem usage
    "-maxrregcount=128",            # cap registers to improve occupancy
    "--extra-device-vectorization", # extra vectorization passes
]

# For debugging: add -G -lineinfo and remove -O3
# nvcc_flags += ["-G", "-lineinfo"]

cxx_flags = ["-O3", "-std=c++17"]

setup(
    name="dsqg_cuda",
    ext_modules=[
        CUDAExtension(
            name="dsqg_cuda",
            sources=[
                "dsqg_cuda_ext.cpp",
                "dsqg_fwd.cu",
                "dsqg_bwd.cu",
            ],
            extra_compile_args={
                "cxx":  cxx_flags,
                "nvcc": nvcc_flags,
            },
            include_dirs=[],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
