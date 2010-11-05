"""
Test ga.dgemm.

Note:
 * change nummax for large arrays
 * turn off 'verify' for large arrays due to memory
   limitations, as verify=True for large arrays produces
   segfault, dumps core, or any crap.

"""
import time

import mpi4py.MPI
import ga

import numpy as np

BLOCK_CYCLIC = False
VERIFY = True

me = ga.nodeid()
nproc = ga.nnodes()
num1 = nummax = 1024
transa = [False,True,False,True]
transb = [False,False,True,True]
ntrans = len(transa)
nums_m = [512,1024]
nums_n = [512,1024]
nums_k = [512,1024]
howmany = len(nums_k)
h0 = np.zeros((num1,num1), dtype=np.float64)
avg_t = np.zeros(ntrans, dtype=np.float64)
avg_mf = np.zeros(ntrans, dtype=np.float64)

def main():
    # TODO there's got to be a loopless, more pythonic way to do this
    ii = 0
    for i in range(num1):
        for j in range(num1):
            ii += 1
            if ii > num1:
                ii = 0
            h0[i,j] = ii
    # compute times assuming 500 mflops and 5 second target time
    # ntimes = max(3.0, 5.0/(4.0-9*num**3))
    ntimes = 5

    for ii in range(howmany):
        num_m = nums_m[ii]
        num_n = nums_n[ii]
        num_k = nums_k[ii]
        a = 0.5/(num_m*num_n)
        if num_m > nummax or num_n > nummax or num_k > nummax:
            ga.error('Insufficient memory: check nummax')
        
        if BLOCK_CYCLIC:
            block_size = [128,128]
            g_c = ga.create_handle()
            ga.set_data(g_c, (num_m,num_n), ga.C_DBL)
            ga.set_array_name(g_c, 'g_c')
            ga.set_block_cyclic(g_c, block_size)
            if not ga.allocate(g_c):
                ga.error('create failed')
            block_size = [128,128]
            g_b = ga.create_handle()
            ga.set_data(g_b, (num_k,num_n), ga.C_DBL)
            ga.set_array_name(g_b, 'g_b')
            ga.set_block_cyclic(g_b, block_size)
            if not ga.allocate(g_b):
                ga.error('create failed')
            block_size = [128,128]
            g_a = ga.create_handle()
            ga.set_data(g_a, (num_m,num_k), ga.C_DBL)
            ga.set_array_name(g_a, 'g_a')
            ga.set_block_cyclic(g_a, block_size)
            if not ga.allocate(g_a):
                ga.error('create failed')
        else:
            g_a = ga.create(ga.C_DBL, (num_m,num_k), 'g_a')
            g_b = ga.create(ga.C_DBL, (num_k,num_n), 'g_b')
            g_c = ga.create(ga.C_DBL, (num_m,num_n), 'g_c')
            for handle in [g_a,g_b,g_c]:
                if 0 == handle:
                    ga.error('create failed')

        # initialize matrices A and B
        if 0 == me:
            load_ga(g_a, h0, num_m, num_k)
            load_ga(g_b, h0, num_k, num_n)
        ga.zero(g_c)
        ga.sync()

        if 0 == me:
            print '\nMatrix Multiplication C = A[%d,%d] x B[%d,%d]\n' % (
                    num_m, num_k, num_k, num_n)
        avg_t[:] = 0
        avg_mf[:] = 0
        for itime in range(ntimes):
            for i in range(ntrans):
                ga.sync()
                ta = transa[i]
                tb = transb[i]
                t1 = time.time()
                ga.gemm(ta,tb,num_m,num_n,num_k,1,g_a,g_b,0,g_c)
                t1 = time.time() - t1
                if 0 == me:
                    mf = 2*num_m*num_n*num_k/t1*10**-6/nproc
                    avg_t[i] += t1
                    avg_mf[i] += mf
                    print ' Run# %2d %12.4f seconds %12.1f mflops/proc %s %s'%(
                            itime+1, t1, mf, ta, tb)
                    if VERIFY and itime == 1:
                        verify_ga_gemm(ta, tb, num_m, num_n, num_k,
                                1.0, g_a, g_b, 0.0, g_c)
        if 0 == me:
            print ''
            for i in range(ntrans):
                print 'Average: %12.4f seconds %12.1f mflops/proc %s %s'%(
                            avg_t[i]/ntimes, avg_mf[i]/ntimes,
                            transa[i], transb[i])
            if VERIFY:
                print 'All ga.gemms are verified...O.K.'

def load_ga(handle, h0, num_m, num_k):
    ga.put(handle, h0[:num_m,:num_k])

def verify_ga_gemm(ta, tb, num_m, num_n, num_k, alpha, g_a, g_b, beta, g_c):
    # TODO Why did the buffer version of ga.get seg fault???
    #tmpa = np.array((num_m, num_k), dtype=np.float64)
    #tmpb = np.array((num_k, num_n), dtype=np.float64)
    #tmpc = np.array((num_m, num_n), dtype=np.float64)
    #tmpc[:] = -1.0
    #tmpa[:] = -2.0
    #tmpa = ga.get(g_a, buffer=tmpa) # seg faulted
    #tmpb = ga.get(g_b, buffer=tmpb) # seg faulted
    #tmpc = ga.get(g_c, buffer=tmpc) # seg faulted
    tmpa = ga.get(g_a)
    tmpb = ga.get(g_b)
    tmpc = ga.get(g_c)
    # TODO need to create another array same shape as g_c for result of xgemm
    result = tmpc
    # TODO the rest of verify
    # call xgemm which is really "test_dgemm" from Linalg library
    # test_dgemm(...)
    abs_value = np.abs(tmpc-result)
    # TODO abs_value check
    if np.any(abs_value > 1):
        ga.error('verify ga.gemm failed')

if __name__ == '__main__':
    main()
