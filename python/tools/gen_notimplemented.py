import inspect
import sys

import numpy as np
from ga import gain

empty = "empty"

print '''
import sys

import numpy as np
cimport numpy as np
'''

for name in dir(np):
    np_attr = getattr(np,name)
    gain_attr = getattr(gain,name,None)
    if inspect.isfunction(np_attr) and gain_attr is None:
        args,varargs,keywords,defaults = inspect.getargspec(np_attr)
        newdefaults = [empty]*len(args)
        if defaults is not None:
            newdefaults[(len(args)-len(defaults)):] = defaults
        sig = "%s(" % name
        stuff = zip(args,newdefaults)
        def stringify(arg,default):
            if arg == "char": # 'char' is a cython keyword
                arg = "char_"
            if default is empty:
                return "%s" % arg
            elif default is None:
                return "%s=None" % (arg)
            elif default is sys.stdout:
                return "%s=sys.stdout" % (arg)
            elif default is int:
                return "%s=int" % (arg)
            elif default is float:
                return "%s=float" % (arg)
            elif default is np.float64:
                return "%s=np.float64" % (arg)
            elif isinstance(default,basestring):
                if '\n' in default:
                    return "%s='\\n'" % (arg)
                else:
                    return "%s='%s'" % (arg,default)
            elif inspect.isbuiltin(default):
                return "%s=%s" % (arg,default.__name__)
            else:
                return "%s=%s" % (arg,default)
        if len(stuff) > 0:
            for arg,default in stuff[:-1]:
                sig += stringify(arg,default)
                sig += ", "
            arg,default = stuff[-1]
            sig += stringify(arg,default)
            sig += ")"
        else:
            sig += ")"

        doc = ""
        if np_attr.__doc__:
            doc = np_attr.__doc__
        print '''
def %s:
    """%s
    
    """
    raise NotImplementedError
''' % (sig, doc.strip().replace('"""',"'''"))
