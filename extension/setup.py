from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='bfloat16_compression',
    ext_modules=[
        CUDAExtension('bfloat16_compression', [
            'bfloat16_compression.cu',
        ], 
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': ['-O3', '-arch=sm_90']  # Adjust sm_XX based on your GPU architecture
        })
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
