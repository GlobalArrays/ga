#!/usr/bin/env python

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
    #sys.stdout.flush()
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
        print ' CHECKING DOUBLES '
        print ''
    check_dbl()

    # check support for single precision complex arrays
    if 0 == me:
        print ''
        print ' CHECKING SINGLE COMPLEX '
        print ''
    check_complex_float()

    # check support for double precision complex arrays
    if 0 == me:
        print ''
        print ' CHECKING DOUBLE COMPLEX '
        print ''
    check_complex()

    # check support for integer arrays
    if 0 == me:
        print ''
        print ' CHECKING INTEGERS '
        print ''
    check_int()

    # check support for single precision arrays
    if 0 == me:
        print ''
        print ' CHECKING SINGLE PRECISION '
        print ''
    check_flt()

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

def check_dbl():
    n = 256
    m = 2*n
    a = np.zeros((n,n), dtype=np.float64)
    b = np.zeros((n,n), dtype=np.float64)
    v = np.zeros((m,), dtype=np.float64)
    w = np.zeros((m,), dtype=np.float64)
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
    for i in range(n):
        for j in range(n):
            if MIRROR:
                a[i,j] = inode + i + j*n
            else:
                a[i,j] = i + j*n
            b[i,j] = -1
    # create a global array
    if NEW_API:
        g_a = ga.create_handle()
        ga.set_data(g_a, [n,n], ga.C_DBL)
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
            ga.create_config(ga.C_DBL, (n,n), 'a', None, p_mirror)
        else:
            g_a = ga.create(ga.C_DBL, (n,n), 'a')
    if 0 == g_a:
        print ' ga.create failed'
        ga.error('... exiting ', 0)
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
    b = ga.get(g_a, buffer=b)
    ga.sync()
    for i in range(n):
        for j in range(n):
            if b[i,j] != 0.0:
                print '%d zero %d %d %s' % (me, i, j, b[i,j])
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
                hi = [min(i+inc,n)-1, min(j+inc,n)-1]
                piece = a[lo[0]:(hi[0]+1), lo[1]:(hi[1]+1)]
                ga.put(g_a, lo, hi, piece)
                result = ga.get(g_a, lo, hi)
                if not np.all(result == piece):
                    print piece
                    print result
                    ga.error("put failed", 1)
            ga.sync()
            ij += 1
    ga.sync()
    # all nodes check all of a
    b[:] = 0
    b = ga.get(g_a, buffer=b)
    for i in range(n):
        for j in range(n):
            if b[i,j] != a[i,j]:
                print ' put %d %d %d %s %s' % (me, i, j, a[i,j], b[i,j])
                ga.error('... exiting ', 0)
    if 0 == me:
        print ''
        print ' ga.put is OK'
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
