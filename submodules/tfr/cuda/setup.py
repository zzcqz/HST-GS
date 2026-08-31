from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

os.environ["CC"] = os.environ.get("CC", "/usr/bin/gcc")
os.environ["CXX"] = os.environ.get("CXX", "/usr/bin/g++")
if "conda" in os.environ["CC"] or "x86_64-conda-linux-gnu" in os.environ["CC"]:
    os.environ["CC"] = "/usr/bin/gcc"
if "conda" in os.environ["CXX"] or "x86_64-conda-linux-gnu" in os.environ["CXX"]:
    os.environ["CXX"] = "/usr/bin/g++"

this_dir = os.path.dirname(os.path.abspath(__file__))
nvcc_ccbin = "-ccbin=" + os.environ["CC"]
glm_include = "-I" + os.path.join(this_dir, "third_party/glm/")

setup(
    name="tfr_cuda",
    packages=["tfr_cuda"],
    ext_modules=[
        CUDAExtension(
            name="tfr_cuda._C",
            sources=[
                "csrc/tfr_ext.cu",
                "csrc/tfr_render.cu",
                "csrc/tfr_preprocess.cu",
                "csrc/tfr_binning.cu",
                "csrc/tfr_backward.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", nvcc_ccbin, "--use_fast_math", glm_include],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
