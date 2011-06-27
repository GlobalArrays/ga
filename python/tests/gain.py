from mpi4py import MPI
import ga
from ga import gain
from ga.gain import util
import numpy as np
from getopt import getopt
import sys

me = gain.me
count = 0

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
    global count
    global results
    count += 1
    if not me:
        print "test %s" % count
    results[current_module].append(what)

def test(module):
    global current_module
    current_module = module
    check(module.arange(100, dtype=module.float32))
    a = module.arange(100, dtype=module.float32)
    check(a.copy())
    check(a[9:17].copy())
    check(module.sin(a[49::-2],a[50::2]))
    a[5:15] = module.arange(10, dtype=module.float32)
    check(a)
    a[5:15] = np.arange(10, dtype=np.float32)
    check(a)
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
    check(module.add(x,x))
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
    check(module.dot(6,10))
    check(module.dot(6,h))
    check(module.eye(24,25))
    check(module.eye(24,25,4))
    check(module.eye(24,25,-8))
    check(module.identity(11))
    check(module.add.reduce([1,2,3,4]))
    check(module.add.reduce(module.arange(100)))
    check(module.add.reduce(module.ones((100,200))))
    check(module.add.accumulate(module.ones(7)))
    check(module.add.accumulate(module.ones((7,7))))
    check(module.add.accumulate(module.ones((7,7,7)), axis=0))
    check(module.add.accumulate(module.ones((7,7,7)), axis=1))
    check(module.add.accumulate(module.ones((7,7,7)), axis=2))
    check(module.add.accumulate(module.ones((7,7,7)), axis=0))
    check(module.add.accumulate(module.ones((7,7,7)), axis=1))
    check(module.add.accumulate(module.ones((7,7,7)), axis=2))
    check(module.alen((1,2,3)))
    check(module.alen(module.zeros((4,5,6))))
    foo = np.arange(4*25).reshape(4,25)
    i = module.zeros((4,25))
    i[:] = foo
    check(i)
    j = i.flat
    check(j[2])
    check(j[2:19])
    j[:] = 6
    check(j)
    j[2:19] = 7
    check(j)
    foo = module.zeros((3,4,5))
    bar = module.arange(3*4*5)
    foo.flat = bar
    check(foo)
    foo.flat = 6
    check(foo)
    # works for GAiN but not NumPy
    # raises TypeError return arrays must be ArrayType
    #check(module.add(foo,bar,i.flat))
    check(module.add(s,t,np.ones((3,4,5), dtype=np.float32)))
    check(module.clip(module.arange(10), 1, 8))
    check(module.clip(module.arange(100), 10, 80))
    check(module.clip(module.arange(10), [3,4,1,1,1,4,4,4,4,4], 8))
    check(i.transpose())
    foo = np.arange(4*5*77).reshape(4,5,77)
    k = module.zeros((4,5,77))
    k[:] = foo
    check(k.transpose())
    check(k.transpose(1,2,0).shape)
    check(k.transpose(1,2,0))

def main():
    global count
    if not me:
        print "=========== TESTING numpy =============================="
    ga.sync()
    count = 0
    test(np)
    if not me:
        print "=========== TESTING gain ==============================="
    ga.sync()
    count = 0
    test(gain)
    if not me:
        print "=========== COMPARING RESULTS =========================="
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
        for num,result in enumerate(zip(results[np],results[gain])):
            print "result %s" % (num+1)
            err = False
            try:
                result_np,result_gain = result
                diff = None
                if isinstance(result_gain, gain.ndarray):
                    diff = result_np-result_gain.get()
                elif isinstance(result_gain, gain.flatiter):
                    diff = result_np-result_gain.get()
                elif isinstance(result_gain, tuple):
                    if (len(result_gain) > 1
                        and type(result_gain[0]) is type(result_gain[1])):
                        diff = np.asarray(result_np)-np.asarray(result_gain)
                    else:
                        result_np = result_np[0]
                        result_gain = result_gain[0]
                        diff = result_np-result_gain.get()
                else:
                    diff = result_np-result_gain
                if not np.all(diff == 0):
                    print_result(result_np,result_gain,diff)
                    err = True
                if hasattr(result_np,'dtype'):
                    if not result_np.dtype == result_gain.dtype:
                        print "different types np=%s gain=%s" % (
                                result_np.dtype, result_gain.dtype)
                        err = True
            except Exception,e:
                print "caught exception:", e
                err = True
            if err:
                raise ValueError, "something bad at %s" % num

if __name__ == '__main__':
    profile = False
    (optsvals,args) = getopt(sys.argv[1:],'p')
    for (opt,val) in optsvals:
        if opt == '-p':
            profile = True
    if profile:
        import cProfile
        if not me:
            print "Profiling enabled"
        cProfile.run("main()", "gaintest.%s.prof" % str(me))
    else:
        main()
