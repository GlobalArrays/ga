#!/usr/bin/env python

import random
import sys

from mpi4py import MPI
import ga
import numpy as np

nproc = ga.nnodes()
me = ga.nodeid()
inode = ga.cluster_nodeid()
lprocs = ga.cluster_nprocs(inode)
nnodes = ga.cluster_nnodes()

MEM_INC = 1000
MIRROR = False
USE_RESTRICTED = False
NEW_API = False
NGA_GATSCAT = False
BLOCK_CYCLIC = False
USE_SCALAPACK_DISTR = False

# This was used to debug as it makes printed numpy arrays a bit easier to read
np.set_printoptions(precision=6, suppress=True, edgeitems=4)

def mismatch(x,y):
    return abs(x-y)/max(1.0,abs(x)) > 1e-12

def main():
    if 0 == me:
        if MIRROR:
            print ' Performing tests on Mirrored Arrays'
        print ' GA initialized'

    # note that MA is not used, so no need to initialize it
    # "import ga" registers malloc/free as memory allocators internally

    #if nproc-1 == me:
    if 0 == me:
        print 'using %d process(es) %d custer nodes' % (
                nproc, ga.cluster_nnodes())
        print 'process %d is on node %d with %d processes' % (
                me, ga.cluster_nodeid(), ga.cluster_nprocs(-1))
    ga.sync()

    # create array to force staggering of memory and uneven distribution
    # of pointers
    dim1 = MEM_INC
    mapc = [0]*nproc
    for i in range(nproc):
        mapc[i] = MEM_INC*i
        dim1 += MEM_INC*i
    g_s = ga.create_handle()
    ga.set_data(g_s, [dim1], ga.C_INT)
    ga.set_array_name(g_s, 's')
    ga.set_irreg_distr(g_s, mapc, [nproc])

    if MIRROR:
        if 0 == me:
            print ''
            print '  TESTING MIRRORED ARRAYS  '
            print ''
    ga.sync()

    # check support for double precision arrays
    if 0 == me:
        print ''
        print ' CHECKING DOUBLE PRECISION '
        print ''
    check(ga.C_DBL, np.float64)

    # check support for single precision complex arrays
    if 0 == me:
        print ''
        print ' CHECKING SINGLE COMPLEX '
        print ''
        print ' !!!SKIPPED'
    #check(ga.C_SCPL, np.complex64)

    # check support for double precision complex arrays
    if 0 == me:
        print ''
        print ' CHECKING DOUBLE COMPLEX '
        print ''
        print ' !!!SKIPPED'
    #check(ga.C_DCPL, np.complex128)

    # check support for integer arrays
    if 0 == me:
        print ''
        print ' CHECKING INT'
        print ''
    check(ga.C_INT, np.int32)

    # check support for long integer arrays
    if 0 == me:
        print ''
        print ' CHECKING LONG INT'
        print ''
    check(ga.C_LONG, np.int64)

    # check support for single precision arrays
    if 0 == me:
        print ''
        print ' CHECKING SINGLE PRECISION '
        print ''
    check(ga.C_FLOAT, np.float32)

    if 0 == me:
        print ''
        print ' CHECKING Wrappers to Message Passing Collective ops '
        print ''
    check_wrappers()

    # check if memory limits are enforced
    #check_mem(ma_heap*ga.nnodes())

    if 0 == me: ga.print_stats()
    if 0 == me: print ' '
    if 0 == me: print 'All tests successful'

    # tidy up the ga package
    # NO NEED -- atexit registered ga.terminate()

    # tidy up after message-passing library
    # NO NEED -- atexti registered MPI.Finalize()

    # Note: so long as mpi4py is imported before ga, cleanup is automatic

def check(gatype, nptype):
    n = 256
    m = 2*n
    a = np.zeros((n,n), dtype=nptype)
    b = np.zeros((n,n), dtype=nptype)
    maxloop = 100
    maxproc = 4096
    num_restricted = 0
    restricted_list = 0
    iproc = me % lprocs
    nloop = min(maxloop,n)
    if USE_RESTRICTED:
        num_restricted = nproc/2
        restricted_list = [0]*num_restricted
        if (num_restricted == 0):
            num_restricted = 1
        for i in range(num_restricted):
            restricted_list[i] = (num_restricted/2) + i
    if BLOCK_CYCLIC:
        block_size = [32,32]
        if USE_SCALAPACK_DISTR:
            if nproc % 2 == 0:
                ga.error('Available procs must be divisible by 2',nproc)
            proc_grid = [2,nproc/2]
    # a[] is a local copy of what the global array should start as
    if MIRROR:
        a[:] = np.fromfunction(lambda i,j: inode+i+j*n, (n,n), dtype=nptype)
    else:
        a[:] = np.fromfunction(lambda i,j: i+j*n, (n,n), dtype=nptype)
    b[:] = -1
    # create a global array
    if NEW_API:
        g_a = ga.create_handle()
        ga.set_data(g_a, [n,n], gatype)
        ga.set_array_name(g_a, 'a')
        if USE_RESTRICTED:
            ga.set_restricted(g_a, restricted_list)
        if BLOCK_CYCLIC:
            if USE_SCALAPACK_DISTR:
                ga.set_block_cyclic_proc_grid(g_a, block_size, proc_grid)
            else:
                ga.set_block_cyclic(g_a, block_size)
        if MIRROR:
            p_mirror = ga.pgroup_get_mirror()
            ga.set_pgroup(g_a, p_mirror)
        ga.allocate(g_a)
    else:
        if MIRROR:
            p_mirror = ga.pgroup_get_mirror()
            ga.create_config(gatype, (n,n), 'a', None, p_mirror)
        else:
            g_a = ga.create(gatype, (n,n), 'a')
    if 0 == g_a:
        ga.error('ga.create failed')
    if MIRROR:
        lproc = me - ga.cluster_procid(inode, 0)
        lo,hi = ga.distribution(g_a, lproc)
    else:
        lo,hi = ga.distribution(g_a, me)
    ga.sync()
    # zero the array
    if 0 == me:
        print '> Checking zero ...'
    ga.zero(g_a)
    # check that it is indeed zero
    b = ga.get(g_a, buffer=b) # gets the result into supplied buffer b
    ga.sync()
    if not np.all(b == 0):
        ga.error('ga.zero failed')
    if 0 == me:
        print ''
        print ' ga.zero is OK'
        print ''
    ga.sync()
    # each node fills in disjoint sections of the array
    if 0 == me:
        print '> Checking disjoint put ... '
    ga.sync()
    inc = (n-1)/20 + 1
    ij = 0
    for i in range(0,n,inc):
        for j in range(0,n,inc):
            check = False
            if MIRROR:
                check = ij % lprocs == iproc
            else:
                check = ij % nproc == me
            if check:
                lo = [i,j]
                hi = [min(i+inc,n), min(j+inc,n)]
                #piece = a[lo[0]:hi[0], lo[1]:hi[1]]
                piece = a[ga.zip(lo,hi)]
                ga.put(g_a, piece, lo, hi)
                # the following check is not part of the original test.F
                result = ga.get(g_a, lo, hi) # ndarray created inside get
                if not np.all(result == piece):
                    ga.error("put followed by get failed", 1)
            ga.sync()
            ij += 1
    ga.sync()
    # all nodes check all of a
    b[:] = 0
    b = ga.get(g_a, buffer=b)
    if not np.all(a == b):
        ga.error('put failed, exiting')
    if 0 == me:
        print ''
        print ' ga.put is OK'
        print ''
    # now check nloop random gets from each node
    if 0 == me:
        print '> Checking random get (%d calls)...' % nloop
    ga.sync()
    nwords = 0
    random.seed(ga.nodeid()*51+1) # different seed for each proc
    for loop in range(nloop):
        ilo,ihi = random.randint(0, nloop-1),random.randint(0, nloop-1)
        if ihi < ilo: ilo,ihi = ihi,ilo
        jlo,jhi = random.randint(0, nloop-1),random.randint(0, nloop-1)
        if jhi < jlo: jlo,jhi = jhi,jlo
        nwords += (ihi-ilo+1)*(jhi-jlo+1)
        ihi += 1
        jhi += 1
        result = ga.get(g_a, (ilo,jlo), (ihi,jhi))
        if not np.all(result == a[ilo:ihi,jlo:jhi]):
            ga.error('random get failed')
        if 0 == me and loop % max(1,nloop/20) == 0:
            print ' call %d node %d checking get((%d,%d),(%d,%d)) total %f' % (
                    loop, me, ilo, ihi, jlo, jhi, nwords)
    if 0 == me:
        print ''
        print ' ga_get is OK'
        print ''
    # each node accumulates into disjoint sections of the array
    if 0 == me:
        print '> Checking accumulate ... '
    ga.sync()
    random.seed(12345) # same seed for each process
    b[:] = np.fromfunction(lambda i,j: i+j+2, (n,n), dtype=nptype)
    inc = (n-1)/20 + 1
    ij = 0
    for i in range(0,n,inc):
        for j in range(0,n,inc):
            x = 10.0
            lo = [i,j]
            hi = [min(i+inc,n), min(j+inc,n)]
            piece = b[lo[0]:hi[0], lo[1]:hi[1]]
            check = False
            if MIRROR:
                check = ij % lprocs == iproc
            else:
                check = ij % nproc == me
            if check:
                ga.acc(g_a, lo, hi, piece, x)
            ga.sync()
            ij += 1
            # each process applies all updates to its local copy
            a[lo[0]:hi[0], lo[1]:hi[1]] += x * piece
    ga.sync()
    # all nodes check all of a
    if not np.all(ga.get(g_a) == a):
        ga.error('acc failed')
    if 0 == me:
        print ''
        print ' disjoint ga.acc is OK'
        print ''
    # overlapping accumulate
    ga.sync()
    if NEW_API:
        g_b = ga.create_handle()
        ga.set_data(g_b, (n,n), gatype)
        ga.set_array_name(g_b, 'b')
        if BLOCK_CYCLIC:
            if USE_SCALAPACK_DISTR:
                ga.set_block_cyclic_proc_grid(g_b, block_size, proc_grid)
            else:
                ga.set_block_cyclic(g_b, block_size)
        if MIRROR:
            ga.set_pgroup(g_b, p_mirror)
        if not ga.allocate(g_b):
            ga.error('ga.create failed for second array')
    else:
        if MIRROR:
            g_b = ga.create_config(gatype, (n,n), 'b', chunk, p_mirror)
        else:
            g_b = ga.create(gatype, (n,n), 'b')
        if 0 == g_b:
            ga.error('ga.create failed for second array')
    ga.zero(g_b)
    ga.acc(g_b, (n/2,n/2), (n/2+1,n/2+1), [1], 1)
    ga.sync()
    x = None
    if MIRROR:
        if 0 == iproc:
            x = abs(ga.get(g_b, (n/2,n/2), (n/2+1,n/2+1))[0,0] - lprocs)
            if not 0 == x:
                ga.error('overlapping accumulate failed -- expected %s got %s'%(
                        x, lprocs))
    else:
        if 0 == me:
            x = abs(ga.get(g_b, (n/2,n/2), (n/2+1,n/2+1))[0,0] - nproc)
            if not 0 == x:
                ga.error('overlapping accumulate failed -- expected %s got %s'%(
                        x, nproc))
    if 0 == me:
        print ''
        print ' overlapping ga.acc is OK'
        print ''
    # check the ga.add function
    if 0 == me:
        print '> Checking add ...'
    random.seed(12345) # everyone has same seed
    for i in range(n):
        for j in range(n):
            b[i,j] = random.random()
            a[i,j] = 0.1*a[i,j] + 0.9*b[i,j]
    if MIRROR:
        if 0 == iproc:
            ga.put(g_b, b)
    else:
        if 0 == me:
            ga.put(g_b, b)
    ga.add(g_a, g_b, g_b, 0.1, 0.9)
    if not np.all(ga.get(g_b) == a):
        ga.error('add failed')
    if 0 == me:
        print ''
        print ' add is OK '
        print ''
    # check the dot function
    if 0 == me:
        print '> Checking dot ...'
    random.seed(12345) # everyone has same seed
    sum1 = 0.0
    for i in range(n):
        for j in range(n):
            b[i,j] = random.random()
            sum1 += a[i,j]*b[i,j]
    if MIRROR:
        if 0 == iproc:
            pass
    else:
        if 0 == me:
            ga.put(g_b, b)
            ga.put(g_a, a)
    ga.sync()
    sum2 = ga.dot(g_a, g_b)
    if mismatch(sum1, sum2):
        ga.error('dot wrong %s != %s' % (sum1, sum2))
    if 0 == me:
        print ''
        print ' dot is OK '
        print ''
    # check the ga.scale function
    if 0 == me:
        print '> Checking scale ...'
    ga.scale(g_a, 0.123)
    result = ga.get(g_a)
    if not np.all(a*0.123 == ga.get(g_a)):
        ga.error('scale failed')
    if 0 == me:
        print ''
        print ' scale is OK '
        print ''
    # check the ga.copy function
    if 0 == me:
        print ''
        print '> Checking copy'
        print ''
    if 0 == me:
        ga.put(g_a, a)
    ga.copy(g_a, g_b)
    if not np.all(a == ga.get(g_b)):
        ga.error('copy failed')
    if 0 == me:
        print ''
        print ' copy is OK '
        print ''
    ga.sync()
    if 0 == me:
        print '> Checking scatter/gather (might be slow)...'
    ga.sync()
    ijv = np.zeros((m,2), dtype=np.int64)
    random.seed(ga.nodeid()*51 + 1) # different seed for each proc
    for j in range(10):
        itmp = None
        if MIRROR:
            itmp = random.randint(0,lprocs-1)
        else:
            itmp = random.randint(0,nproc-1)
        if itmp == me:
            for loop in range(m):
                ijv[loop,:] = (random.randint(0,n-1),random.randint(0,n-1))
                #if ijv[loop,0] > ijv[loop,1]:
                #    ijv[loop,:] = ijv[loop,::-1] # reverse
            result = ga.gather(g_a, ijv)
            for loop in range(m):
                value = ga.get(g_a, ijv[loop], ijv[loop]+1).flatten()
                if not result[loop] == value:
                    ga.error('gather failed')
    if 0 == me:
        print ''
        print ' gather is OK'
        print ''

def check_complex_float():
    pass

def check_complex():
    pass

def check_int():
    pass

def check_flt():
    pass

def check_wrappers():
    pass

if __name__ == '__main__':
    main()
