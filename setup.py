import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

ext_modules = [
    Extension(
        "tranmf._nmf",
        ["src/tranmf/_nmf.py"],
        include_dirs=[numpy.get_include()],
        # extra_compile_args=["-fopenmp", "-Ofast"],
        # extra_link_args=["-fopenmp", "-Ofast"],
    )
]


setup(ext_modules=cythonize(ext_modules))
