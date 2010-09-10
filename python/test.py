#!/usr/bin/env python

from mpi4py import MPI
import ga

nproc = ga.nnodes()
me = ga.nodeid()

def main():
    if 0 == me:
        print 'GA initialized'

    # note that MA is not used, so no need to initialize it
    # "ga4py" registers malloc/free as memory allocators internally

    if nproc-1 == me:
        print 'using %d process(es) %d custer nodes' % (
                nproc, ga.cluster_nnodes())
        print 'process %d is on node %d with %d processes' % (
                me, ga.cluster_nodeid(), ga.cluster_nprocs(-1))

    # create array to force staggering of memory and uneven distribution
    # of pointers
    MEM_INC = 1000
    dim1 = MEM_INC
    mapc = [0]*nproc
    for i in range(1,nproc):
        map[i] = MEM_INC*i
        dim1 += MEM_INC*(i+1)
    g_s = ga.create_handle()
    ndim = 1
    ga.set_data(g_s, ndim, dim1, ga.C_INT)
    ga.set_array_name(g_s, 's')
    ga.set_irreg_distr(g_s, map, nproc)
    ga.sync()

    if 0 == me:
        print ''
        print 'CHECKING DOUBLES'
        print ''

    check_dbl()

def check_dbl():
    pass

if __name__ == '__main__':
    main()
