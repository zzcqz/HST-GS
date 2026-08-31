from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

# Golden fixed-tile rasterizer (fork of HST-GS submodule).
os.environ["CC"] = os.environ.get("CC", "/usr/bin/gcc")
os.environ["CXX"] = os.environ.get("CXX", "/usr/bin/g++")
if "conda" in os.environ["CC"] or "x86_64-conda-linux-gnu" in os.environ["CC"]:
    os.environ["CC"] = "/usr/bin/gcc"
if "conda" in os.environ["CXX"] or "x86_64-conda-linux-gnu" in os.environ["CXX"]:
    os.environ["CXX"] = "/usr/bin/g++"

this_dir = os.path.dirname(os.path.abspath(__file__))
# third_party is one level up (TFR/cuda/third_party)
root = os.path.dirname(this_dir)
glm_include = "-I" + os.path.join(root, "third_party/glm/")
compat_include = "-I" + os.path.join(root, "third_party/compat/")
nvcc_ccbin = "-ccbin=" + os.environ["CC"]

setup(
    name="tfr_golden_rasterization",
    packages=["diff_gaussian_rasterization"],
    package_dir={"diff_gaussian_rasterization": "diff_gaussian_rasterization"},
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "rasterize_points.cu",
                "ext.cpp",
            ],
            extra_compile_args={
                "cxx": [compat_include],
                "nvcc": [glm_include, compat_include, nvcc_ccbin],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
