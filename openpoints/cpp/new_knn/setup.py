# -*- coding: utf-8 -*-
# @Author: Weiwei Jin
# @Date:   2024-01-18 20:17:19
# @Email:  weiweijin1109@gmail.com

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='new_knn',
      version='1.0.0',
      ext_modules=[
          CUDAExtension('new_knn', [
              'new_knn.cpp',
              'new_knn_kernel.cu',
          ],
          extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']}),
      ],
      cmdclass={'build_ext': BuildExtension})
