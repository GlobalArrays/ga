import mpi4py.MPI
from core import *
import random

if __name__ != '__main__':
    # reports how much of numpy as been overridden
    if False:
        import inspect
        np_function_count = 0
        ov_function_count = 0
        np_class_count = 0
        ov_class_count = 0
        np_ufunc_count = 0
        ov_ufunc_count = 0
        self_module = sys.modules[__name__]
        for attr in dir(np):
            np_obj = getattr(np, attr)
            override = False
            if hasattr(self_module, attr):
                #if not me: print "gain override exists for: %s" % attr
                override = True
            else:
                setattr(self_module, attr, getattr(np, attr))
            if inspect.isfunction(np_obj):
                np_function_count += 1
                if override:
                    ov_function_count += 1
            elif type(np_obj) is type(np.add):
                np_ufunc_count += 1
                if override:
                    ov_ufunc_count += 1
            elif inspect.isclass(np_obj):
                np_class_count += 1
                if override:
                    ov_class_count += 1
        print "%d/%d numpy functions overridden by gain" % (
                ov_function_count,np_function_count)
        print "%d/%d numpy classes overridden by gain" % (
                ov_class_count,np_class_count)
        print "%d/%d numpy ufuncs overridden by gain" % (
                ov_ufunc_count,np_ufunc_count)
    # imports from 'numpy' module every missing attribute into 'gain' module
    self_module = sys.modules[__name__]
    for attr in dir(np):
        if not hasattr(self_module, attr):
            setattr(self_module, attr, getattr(np, attr))
