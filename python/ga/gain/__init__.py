import mpi4py.MPI
from core import *
from misc import *
import random

if __name__ != '__main__':
    import inspect as _inspect
    import sys
    import numpy
    # imports from 'numpy' module every missing attribute into 'gain' module
    self_module = sys.modules[__name__]
    # import all classes not already overridden
    for name in dir(numpy):
        if not hasattr(self_module, name):
            attr = getattr(numpy, name)
            if _inspect.isclass(attr):
                setattr(self_module, name, attr)
    # import some other numpy functions directly
    from numpy import alen
