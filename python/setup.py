from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ga_install = "/home/d3n000/ga/ga-dev/bld_openmpi_shared"
numpy_include = "/home/d3n000/ga/local/lib/python2.6/site-packages/numpy/core/include"

ext_modules = [
    Extension(
        name="ga",
        sources=["ga.pyx"],
        include_dirs=[ga_install+"/include",numpy_include],
        library_dirs=[ga_install+"/lib"],
        libraries=["ga"],
    )
]

setup(
        name = "Global Arrays",
        cmdclass = {"build_ext": build_ext},
        ext_modules = ext_modules
)
