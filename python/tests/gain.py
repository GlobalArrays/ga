from mpi4py import MPI
from ga import ga
from ga import gain
from ga import util
import numpy as np

me = gain.me
util.DEBUG = False
gain.DEBUG = False
gain.DEBUG_SYNC = True

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

# quick test of ga.gemm
g_a = ga.create(ga.C_DBL, (10,20))
g_b = ga.create(ga.C_DBL, (20,30))
g_c = ga.create(ga.C_DBL, (10,30))
ga.gemm(False, False, 10, 30, 20, 1, g_a, g_b, 1, g_c)

def foo(module):
    if not me: print "using module %s" % module
    results[module].append(module.arange(100, dtype=module.float32))
    results[module].append(module.arange(100, dtype=module.float32))
    #d = module.arange(100, dtype=module.float32)
    #e = module.arange(100, dtype=module.float32)
    ##d_lower = d[:50:2]
    #d_lower = d[49::-2]
    ##d_upper = d[99:49:-2]
    #d_upper = d[50:100:2]
    #e_quarter = e[::4]
    #if not me: print d_lower
    #if not me: print d_upper
    #e = module.sin(d_lower,d_upper)
    #if not me: print e
    #results[module].append(e)
    #results[module].append(module.sin(1))
    #results[module].append(module.sin([1,2,3]))
    #z = module.asarray([0,0,0], dtype=module.float32)
    #results[module].append(module.sin([1,2,3], z))
    #a = module.ones(10, dtype=module.int16)
    ##b = module.ones(10, dtype=module.int16)
    ##print module.sin(a,b)
    #print module.sin(a)
    #results[module].append(module.add(d_lower,d_upper))
    #results[module].append(module.add(d_lower,d_upper,e_quarter))
    #results[module].append(module.add(d_upper,5))
    #results[module].append(module.add(5,d_lower))
    #s = module.ones((3,4,5), dtype=module.float32)
    #t = module.ones((3,4,5), dtype=module.float32)
    #results[module].append(module.sin(s,t))
    #x = module.ones((2,3,4))
    #y = module.ones((3,4))
    #results[module].append(module.add(x,y))
    results[module].append(module.linspace(2.0,3.0,num=5))
    results[module].append(module.linspace(2.0,3.0,num=5,endpoint=False))
    results[module].append(module.linspace(2.0,3.0,num=5,retstep=True))
    results[module].append(module.logspace(2.0,3.0,num=4))
    results[module].append(module.logspace(2.0,3.0,num=4,endpoint=False))
    results[module].append(module.logspace(2.0,3.0,num=4,base=2.0))
    a = module.ones((10,20), dtype=float)
    b = module.ones((20,30), dtype=float)
    results[module].append(module.dot(a,b))
    d = module.arange(100, dtype=module.float32)
    results[module].append(module.dot(d,d))

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
            if hasattr(result_np, "dtype"): print result_np.dtype
            print result_np
            print "RESULT---------------------------------"
            print type(result_gain)
            if hasattr(result_gain, "dtype"): print result_gain.dtype
            print result_gain
            print "difference ---------------------------------"
            diff = None
            diff_warn = "---------------------------------WARNING DIFF FAILED"
            if isinstance(result_gain, gain.ndarray):
                diff = result_np-result_gain.get()
                print diff
                if not np.all(diff == 0): print diff_warn
                if not result_np.dtype == result_gain.dtype: print diff_warn
            elif isinstance(result_gain, tuple):
                pass
            else:
                diff = result_np-result_gain
                print diff
                if not np.all(diff == 0): print diff_warn
                if not result_np.dtype == result_gain.dtype: print diff_warn
