import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

ext_modules = [
    Extension(
        "tranmf._nmf",
        ["src/tranmf/_nmf.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3", "-Ofast"],
        extra_link_args=["-O3", "-Ofast"],
    )
]


setup(ext_modules=cythonize(ext_modules))
