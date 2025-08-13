# Minimal setup.py for GTX 1080 Ti + PyTorch 2.8 (cu126)
# Forces local build, only compiles SM 6.1

import sys, os
from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

this_dir = os.path.dirname(os.path.abspath(__file__))
PACKAGE_NAME = "causal_conv1d"

# Always build locally
os.environ["CAUSAL_CONV1D_FORCE_BUILD"] = "TRUE"

# Hardcode GPU arch for GTX 1080 Ti (Pascal, SM 6.1)
cc_flag = [
    "-gencode", "arch=compute_61,code=sm_61",
    "-gencode", "arch=compute_61,code=compute_61"  # PTX fallback
]

extra_compile_args = {
    "cxx": ["-O3"],
    "nvcc": [
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "--ptxas-options=-v",
        "-lineinfo",
    ] + cc_flag
}

ext_modules = [
    CUDAExtension(
        name="causal_conv1d_cuda",
        sources=[
            "csrc/causal_conv1d.cpp",
            "csrc/causal_conv1d_fwd.cu",
            "csrc/causal_conv1d_bwd.cu",
            "csrc/causal_conv1d_update.cu",
        ],
        extra_compile_args=extra_compile_args,
        include_dirs=[Path(this_dir) / "csrc"],
    )
]

setup(
    name=PACKAGE_NAME,
    version="1.5.2", #todo manually update on merge
    packages=find_packages(exclude=("build", "tests", "docs")),
    author="Tri Dao",
    author_email="tri@tridao.me",
    description="Causal depthwise conv1d in CUDA, with a PyTorch interface",
    long_description="Local build for GTX 1080 Ti (SM 6.1)",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "packaging",
        "ninja",
    ],
)
