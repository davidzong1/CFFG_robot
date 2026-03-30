# biubushy | 2026-01

import os
import warnings
from pathlib import Path

from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME, HIP_HOME

this_dir = os.path.dirname(os.path.abspath(__file__))

FORCE_CXX11_ABI = os.getenv("MAMBA_FORCE_CXX11_ABI", "FALSE") == "TRUE"

HIP_BUILD = bool(torch.version.hip)


def get_cuda_bare_metal_version(cuda_dir):
    import subprocess

    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = output[release_idx].split(",")[0]
    from packaging.version import parse

    return raw_output, parse(bare_metal_version)


def get_hip_version(rocm_dir):
    import subprocess
    from packaging.version import parse

    hipcc_bin = "hipcc" if rocm_dir is None else os.path.join(rocm_dir, "bin", "hipcc")
    try:
        raw_output = subprocess.check_output([hipcc_bin, "--version"], universal_newlines=True)
    except Exception:
        return None, None
    for line in raw_output.split("\n"):
        if "HIP version" in line:
            rocm_version = parse(line.split()[-1].rstrip("-").replace("-", "+"))
            return line, rocm_version
    return None, None


def append_nvcc_threads(nvcc_extra_args):
    return nvcc_extra_args + ["--threads", "4"]


cc_flag = []

if HIP_BUILD:
    if HIP_HOME is None:
        warnings.warn("HIP_HOME is not set. hipcc may not be available.")

    rocm_home = os.getenv("ROCM_PATH")
    _, hip_version = get_hip_version(rocm_home)

    cc_flag.append("-DBUILD_PYTHON_PACKAGE")

    extra_compile_args = {
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": [
            "-O3",
            "-std=c++17",
            f"--offload-arch={os.getenv('HIP_ARCHITECTURES', 'native')}",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-fgpu-flush-denormals-to-zero",
        ]
        + cc_flag,
    }
else:
    if CUDA_HOME is None:
        warnings.warn("CUDA_HOME is not set. nvcc may not be available.")

    if CUDA_HOME is not None:
        _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
        from packaging.version import Version

        if bare_metal_version < Version("11.6"):
            raise RuntimeError("mamba_selective_scan requires CUDA 11.6 and above.")

        if bare_metal_version <= Version("12.9"):
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_53,code=sm_53")
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_62,code=sm_62")
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_70,code=sm_70")
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_72,code=sm_72")
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_75,code=sm_75")
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_80,code=sm_80")
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_87,code=sm_87")
        if bare_metal_version >= Version("11.8"):
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_90,code=sm_90")
        if bare_metal_version >= Version("12.8"):
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_100,code=sm_100")
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_120,code=sm_120")
        if bare_metal_version >= Version("13.0"):
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_103,code=sm_103")
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_110,code=sm_110")
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_121,code=sm_121")

    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True

    extra_compile_args = {
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": append_nvcc_threads(
            [
                "-O3",
                "-std=c++17",
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
            ]
            + cc_flag
        ),
    }

ext_modules = [
    CUDAExtension(
        name="selective_scan_cuda",
        sources=[
            "csrc/selective_scan.cpp",
            "csrc/selective_scan_fwd_fp32.cu",
            "csrc/selective_scan_fwd_fp16.cu",
            "csrc/selective_scan_fwd_bf16.cu",
            "csrc/selective_scan_bwd_fp32_real.cu",
            "csrc/selective_scan_bwd_fp32_complex.cu",
            "csrc/selective_scan_bwd_fp16_real.cu",
            "csrc/selective_scan_bwd_fp16_complex.cu",
            "csrc/selective_scan_bwd_bf16_real.cu",
            "csrc/selective_scan_bwd_bf16_complex.cu",
        ],
        extra_compile_args=extra_compile_args,
        include_dirs=[Path(this_dir) / "csrc"],
    )
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
