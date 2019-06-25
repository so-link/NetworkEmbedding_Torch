# coding:utf-8

from distutils.core import setup, Extension
from Cython.Build import cythonize

import numpy

setup(
    ext_modules=cythonize(Extension(
        'necython',
        sources=[
            'necython/extension.pyx',
            'necython/cpp/aco.cpp',
            'necython/cpp/common.cpp',
            'necython/cpp/sampling.cpp',
            'necython/cpp/walker.cpp',
        ],
        extra_compile_args = ['-std=c++11'],
        language='c++',
    )),
    include_dirs = [numpy.get_include()]
)