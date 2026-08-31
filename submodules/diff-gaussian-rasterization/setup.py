#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

# Prefer a CUDA-supported host compiler for nvcc (CUDA 11.8 supports GCC <= 11).
# In many conda envs CC/CXX are set to a newer toolchain that nvcc rejects.
os.environ["CC"] = os.environ.get("CC", "/usr/bin/gcc")
os.environ["CXX"] = os.environ.get("CXX", "/usr/bin/g++")
if "conda" in os.environ["CC"] or "x86_64-conda-linux-gnu" in os.environ["CC"]:
    os.environ["CC"] = "/usr/bin/gcc"
if "conda" in os.environ["CXX"] or "x86_64-conda-linux-gnu" in os.environ["CXX"]:
    os.environ["CXX"] = "/usr/bin/g++"

this_dir = os.path.dirname(os.path.abspath(__file__))
glm_include = "-I" + os.path.join(this_dir, "third_party/glm/")
compat_include = "-I" + os.path.join(this_dir, "third_party/compat/")
nvcc_ccbin = "-ccbin=" + os.environ["CC"]

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={
                "cxx": [compat_include],
                "nvcc": [glm_include, compat_include, nvcc_ccbin],
            })
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
