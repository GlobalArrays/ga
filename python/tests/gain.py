from mpi4py import MPI
from ga import ga
from ga import gain
from ga import util
import numpy as np
import inspect

me = gain.me
util.DEBUG = False
gain.DEBUG = False
gain.DEBUG_SYNC = False

results = {}
results[np] = []
results[gain] = []
current_module = None

# quick test of ga.gemm
g_a = ga.create(ga.C_DBL, (10,20))
g_b = ga.create(ga.C_DBL, (20,30))
g_c = ga.create(ga.C_DBL, (10,30))
ga.gemm(False, False, 10, 30, 20, 1, g_a, g_b, 1, g_c)

def check(what):
    global results
    results[current_module].append(what)

def test(module):
    global current_module
    current_module = module
    check(module.arange(100, dtype=module.float32))
    a = module.arange(100, dtype=module.float32)
    check(module.sin(a[49::-2],a[50::2]))
    check(module.sin(1))
    check(module.sin([1,2,3]))
    b = module.asarray([0,0,0], dtype=module.float32)
    check(module.sin([1,2,3], b))
    c = module.ones(10, dtype=module.int16)
    d = module.ones(10, dtype=module.int16)
    check(module.sin(c,d))
    check(module.sin(c))
    e = module.arange(100, dtype=module.float32)
    e_lower = e[:50:2]
    e_upper = e[50::2]
    e_quarter = e[::4]
    check(module.add(e_lower,e_upper))
    # commented out because numpy produces inconsistent result
    #check(module.add(e_lower,e_upper,e_quarter))
    check(module.add(e_upper,5))
    check(module.add(5,e_lower))
    s = module.ones((3,4,5), dtype=module.float32)
    t = module.ones((3,4,5), dtype=module.float32)
    check(s)
    check(module.sin(s,t))
    x = module.ones((2,3,4))
    y = module.ones((3,4))
    check(module.add(x,y))
    check(module.linspace(2.0,3.0,num=5))
    check(module.linspace(2.0,3.0,num=5,endpoint=False))
    check(module.linspace(2.0,3.0,num=5,retstep=True))
    check(module.logspace(2.0,3.0,num=4))
    check(module.logspace(2.0,3.0,num=4,endpoint=False))
    check(module.logspace(2.0,3.0,num=4,base=2.0))
    f = module.ones((100,200), dtype=float)
    g = module.ones((200,300), dtype=float)
    check(module.dot(f,g))
    h = module.arange(100, dtype=module.float32)
    check(module.dot(h,h))
    check(module.eye(24,25))
    check(module.eye(24,25,4))
    check(module.eye(24,25,-8))
    check(module.identity(11))
    check(module.add.reduce([1,2,3,4]))
    check(module.add.reduce(module.arange(100)))
    check(module.add.reduce(module.ones((100,200))))

if __name__ == '__main__':
    ga.sync()
    test(np)
    ga.sync()
    test(gain)
    ga.sync()
    if not me:
        def print_result(result_np,result_gain,diff):
            print "RESULT---------------------------------"
            print type(result_np)
            if hasattr(result_np, "dtype"): print result_np.dtype
            print result_np
            print "RESULT---------------------------------"
            print type(result_gain)
            if hasattr(result_gain, "dtype"): print result_gain.dtype
            print result_gain
            print "difference ---------------------------------"
            print diff
            print "---------------------------------WARNING DIFF FAILED"
        for i,result in enumerate(zip(results[np],results[gain])):
            result_np,result_gain = result
            diff = None
            if isinstance(result_gain, gain.ndarray):
                diff = result_np-result_gain.get()
            elif isinstance(result_gain, tuple):
                result_np = result_np[0]
                result_gain = result_gain[0]
                diff = result_np-result_gain.get()
            else:
                diff = result_np-result_gain
            err = False
            if not np.all(diff == 0):
                print_result(result_np,result_gain,diff)
                err = True
            if not result_np.dtype == result_gain.dtype:
                print "different types np=%s gain=%s" % (
                        result_np.dtype, result_gain.dtype)
                err = True
            if err:
                raise ValueError, "something bad at %s" % i
