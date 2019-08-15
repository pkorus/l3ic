from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

pyfse = Extension(
    name="pyfse",
    sources=["pyfse.pyx"],
    libraries=["fse"],
    extra_compile_args=["-fPIC"],
    extra_link_args=["-fPIC"],
    library_dirs=["FiniteStateEntropy/lib"],
    include_dirs=["FiniteStateEntropy/lib"]
)
setup(
    name="pyfse",
    ext_modules=cythonize([pyfse])
)
