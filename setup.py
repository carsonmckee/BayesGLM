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

setup(name='bayesglm',
      version='1.0',
      description='Bayesian Generalized Linear Models in Python',
      author='Carson McKee',
      author_email='carsonmckee82@gmail.com',
      ext_package='bayesglm', 
      ext_modules=cythonize(extensions),
      packages=['bayesglm'],
      )