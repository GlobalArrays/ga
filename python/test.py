#!/usr/bin/env python

from mpi4py import MPI
import numpy as np
import ga

gtype = ga.C_DBL
#ga.initialize()
me = ga.nodeid()
g_a = ga.create(gtype, [5,20], "g_a")
ga.fill(g_a, -100.0)
g_b = ga.create(gtype, [5,20])
ga.fill(g_b, -200.0)
g_c = ga.create(gtype, (5,20))
ga.fill(g_c, 2)
#ga.randomize(g_a, 100.0)
ga.abs_value(g_a, (2,9), np.asarray((2,11)))
ga.acc(g_a, (2,9), (2,11), (1,2,3))
ga.acc(g_a, (2,9), (2,11), (1,2,3), 2.2)
ga.print_stdout(g_a)
if me == 0:
    buffer = ga.get(g_a)
    buffer[:] = 0
    print buffer
    buffer = ga.get(g_a, buffer=buffer)
    print buffer
dst_lo,dst_hi = ga.distribution(g_a)
if me == 0:
    tmp = ga.access(g_a)
    #print tmp.shape
    #print tmp
    ga.release(g_a)
    tmp = ga.access(g_a, dst_lo+1, dst_hi-1)
    #tmp = ga.access(g_a, dst_lo-1, dst_hi)
    #tmp = ga.access(g_a, dst_lo, dst_hi+1)
    #tmp = ga.access(g_a, dst_hi, dst_lo)
    #print tmp.shape
    #print tmp
    ga.release(g_a, dst_lo+1, dst_hi-1)
ga.add(g_a, g_b, g_a, 3, 2, alo=(4,19), blo=(4,19), clo=(4,19))
ga.print_stdout(g_a)
print me, ga.gop((1,2,3), "+")
print me, ga.gop_multiply((1,2,3))
print me, ga.gop_multiply((1.0,2,3))
print me, ga.gather(g_a, ((1,2),(2,10)))
print me, ga.gather(g_a, (1,2,2,10))
ga.elem_multiply(g_c,g_c,g_c)
ga.print_stdout(g_c)
ga.elem_maximum(g_a,g_c,g_c)
ga.print_stdout(g_c)
if me == 0:
    print ga.locate(g_a, (3,12))
    print ga.locate_nnodes(g_a, (2,9), (4,12))
    print ga.locate_region(g_a, (2,9), (4,12))

