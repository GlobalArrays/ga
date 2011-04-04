from mpi4py import MPI
from ga import ga
from ga import gain
import numpy as np

me = gain.me

if not me: print "test gain"

#buf = np.ones((10,20))
#a = gain.ndarray((10,20), buffer=buf)
#b = gain.ones((20,30))
#c = a[5:,10:]
#a = gain.sin(a,a)
#if not me:
#    print "a.handle", a.handle
#    if a.base is  None:
#        print "a.base is None"
#    else:
#        print "a.base.handle", a.base.handle
#    print "b.handle", b.handle
#    if b.base is None:
#        print "b.base is None"
#    else:
#        print "b.base.handle", b.base.handle
#    print "c.handle", c.handle
#    if c.base is None:
#        print "c.base is None"
#    else:
#        print "c.base.handle", c.base.handle
#
#x = gain.ones((20,30))
#x_part = x[10:,15:]
#y = gain.ones((20,30))
#y_part = y[10:,15:]
#print x_part
#print y_part
#print gain.exp(x_part,y_part)
#print x
#print y

results = {}
results[np] = []
results[gain] = []

def foo(module):
    if not me: print "using module %s" % module
    d = module.arange(100, dtype=module.float32)
    e = module.arange(100, dtype=module.float32)
    #d_lower = d[:50:2]
    d_lower = d[49::-2]
    #d_upper = d[99:49:-2]
    d_upper = d[50:100:2]
    e_quarter = e[::4]
    if not me: print d_lower
    if not me: print d_upper
    e = module.sin(d_lower,d_upper)
    if not me: print e
    results[module].append(e)
    a = module.ones(10, dtype=module.int16)
    #b = module.ones(10, dtype=module.int16)
    #print module.sin(a,b)
    print module.sin(a)
    results[module].append(module.add(d_lower,d_upper))
    results[module].append(module.add(d_lower,d_upper,e_quarter))

if __name__ == '__main__':
    ga.sync()
    foo(np)
    ga.sync()
    foo(gain)
    ga.sync()
    if not me: print len(results)
    if not me:
        for result_np,result_gain in zip(results[np],results[gain]):
            print "RESULT---------------------------------"
            print type(result_np)
            print result_np
            print "RESULT---------------------------------"
            print type(result_gain)
            print result_gain
            print "difference ---------------------------------"
            print result_np-result_gain.get()
