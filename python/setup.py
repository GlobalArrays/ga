import os
from subprocess import Popen, PIPE
import sys

from distutils.core import setup
from distutils.extension import Extension
from distutils.spawn import find_executable

# numpy is required -- attempt import
try:
    import numpy
except ImportError:
    print "numpy is required"
    raise

# cython is optional -- attempt import
use_cython = False
try:
    from Cython.Build import cythonize
    use_cython = True
except:
    pass

# need to find 'ga-config' to gather how GA was configured
ga_config = find_executable("ga-config", None)
if not ga_config:
    raise ValueError, "ga-config not found in path -- required"
p = Popen("%s --cppflags" % ga_config, shell=True, stdout=PIPE, stderr=PIPE,
        close_fds=True)
ga_cppflags,ignore = p.communicate()
p = Popen("%s --ldflags" % ga_config, shell=True, stdout=PIPE, stderr=PIPE,
        close_fds=True)
ga_ldflags,ignore = p.communicate()
p = Popen("%s --libs" % ga_config, shell=True, stdout=PIPE, stderr=PIPE,
        close_fds=True)
ga_clibs,ignore = p.communicate()

# On osx, '-framework Accelerate' doesn't link the actual LAPACK and BLAS
# libraries. Locate them manually if GA was configured to use them.
linalg_include = []
linalg_library = []
linalg_lib = []
if 'Accelerate' in ga_clibs or 'vecLib' in ga_clibs:
    path = "/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Versions/A"
    linalg_include = []
    if os.path.exists(path):
        linalg_library = [path]
        linalg_lib = ["LAPACK","BLAS"]
    # remove '-framework Accelerate' from flags
    ga_clibs = ga_clibs.replace("-framework","")
    ga_clibs = ga_clibs.replace("Accelerate","")
    ga_clibs = ga_clibs.replace("vecLib","")

include_dirs = [numpy.get_include()]
library_dirs = []
libraries = []

# add the GA stuff
for dir in ga_cppflags.split():
    dir = dir.strip()
    include_dirs.append(dir.replace("-I",""))
for dir in ga_ldflags.split():
    dir = dir.strip()
    library_dirs.append(dir.replace("-L",""))
for part in ga_clibs.split():
    part = part.strip()
    if '-L' in part:
        library_dirs.append(part.replace("-L",""))
    elif '-l' in part:
        libraries.append(part.replace("-l",""))

include_dirs.extend(linalg_include)
library_dirs.extend(linalg_library)
libraries.extend(linalg_lib)

ga_ga_sources = ["ga/ga.c"]
ga_gain_core_sources = ["ga/gain/core.c"]
ga_gain_everything_sources = ["ga/gain/everything.c"]
ga_gain_util_sources = ["ga/gain/util.c"]
if use_cython:
    ga_ga_sources = ["ga/ga.pyx"]
    ga_gain_core_sources = ["ga/gain/core.pyx"]
    ga_gain_everything_sources = ["ga/gain/everything.pyx"]
    ga_gain_util_sources = ["ga/gain/util.pyx"]

include_dirs.append(".")

ext_modules = [
    Extension(
        name="ga.ga",
        sources=ga_ga_sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries
    ),
    Extension(
        name="ga.gain.core",
        sources=ga_gain_core_sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries
    ),
    Extension(
        name="ga.gain.everything",
        sources=ga_gain_everything_sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries
    ),
    Extension(
        name="ga.gain.util",
        sources=ga_gain_util_sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries
    ),
]

if use_cython:
    ext_modules = cythonize(ext_modules)

setup(
    name = "Global Arrays",
    packages = ["ga","ga.gain"],
    ext_modules = ext_modules,
)
