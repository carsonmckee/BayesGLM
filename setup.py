"""
setup file
"""

from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        'py_model',
        ["bayesglm/c_model/py_model.pyx"],
        extra_compile_args=["-std=c++17"])
]

setup(ext_package='bayesglm', 
      ext_modules=cythonize(extensions),
      
      )