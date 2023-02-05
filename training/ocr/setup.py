# from setuptools import setup, Extension
# from torch.utils import cpp_extension

# setup(name='custom_ctc_cpp',
#       ext_modules=[cpp_extension.CppExtension('custom_ctc_cpp', ['custom_ctc.cc'])],
#       cmdclass={'build_ext': cpp_extension.BuildExtension})

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_ctc_cu',
    ext_modules=[
        CUDAExtension('custom_ctc_cu', [
            'custom_ctc_cuda_driver.cc',
            'custom_ctc_kernel.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
