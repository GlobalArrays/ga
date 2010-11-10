from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os
import sys

try:
    import numpy
except ImportError:
    print "numpy is required"
    raise
numpy_include = numpy.get_include()

linalg_include = []
linalg_library = []
linalg_lib = []

if sys.platform.startswith('darwin'):
    path = "/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Versions/A"
    linalg_include = []
    if os.path.exists(path):
        linalg_library = [path]
        linalg_lib = ["LAPACK","BLAS"]

include_dirs = [numpy_include]
library_dirs = []
libraries = ["ga"]

for dir in linalg_include:
    include_dirs.append(dir)
for dir in linalg_library:
    library_dirs.append(dir)
for lib in linalg_lib:
    libraries.append(lib)

ext_modules = [
    Extension(
        name="ga",
        sources=["ga.pyx"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries
    )
]

setup(
        name = "Global Arrays",
        cmdclass = {"build_ext": build_ext},
        ext_modules = ext_modules
)
