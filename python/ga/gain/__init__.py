import mpi4py.MPI
from core import *
import random

if __name__ != '__main__':
    import inspect as _inspect
    # imports from 'numpy' module every missing attribute into 'gain' module
    self_module = sys.modules[__name__]
    for name in dir(np):
        if not hasattr(self_module, name):
            attr = getattr(np, name)
            # only import classes e.g. the data types
            if _inspect.isclass(attr):
                setattr(self_module, name, attr)
