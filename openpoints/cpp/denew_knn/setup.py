# -*- coding: utf-8 -*-
# @Author: Weiwei Jin
# @Date:   2024-01-18 20:17:19
# @Email:  weiweijin1109@gmail.com

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='denew_knn',
      version='1.0.0',
      ext_modules=[
          CUDAExtension('denew_knn', [
              'denew_knn.cpp',
              'denew_knn_kernel.cu',
          ]),
      ],
      cmdclass={'build_ext': BuildExtension})
