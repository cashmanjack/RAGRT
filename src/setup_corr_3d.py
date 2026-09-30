import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

optix_path = os.environ.get("OPTIX_PATH", "/home/min/a/cashman3/optix_sdk")

setup(
    name="rtrag_corr_3d",
    ext_modules=[
        CUDAExtension(
            "rtrag_corr_3d",
            [
                "optix_corr_torch_3d.cpp",
                "ragrt_fused_kernel.cu",
                "tile_maxsim_kernel.cu",
                "tile_maxsim_fused_decomp.cu",
            ],
            include_dirs=[
                os.path.join(optix_path, "include"),
                "/usr/local/cuda/include",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-std=c++17",
                    "-D__CUDA_NO_HALF_OPERATORS__",
                    "-D__CUDA_NO_HALF_CONVERSIONS__",
                    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-D__CUDA_NO_HALF2_OPERATORS__",
                ],
            },
            libraries=["cuda", "cudart"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
