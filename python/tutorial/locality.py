import mpi4py.MPI # initialize Message Passing Interface
import ga # initialize Global Arrays

me = ga.nodeid()

def print_distribution(g_a):
    for i in range(ga.nnodes()):
        print "Printing g_a info for processor", i
        lo,hi = ga.distribution(g_a, i)
        print "%s lo=%s hi=%s" % (i,lo,hi)

# create some arrays
g_a = ga.create(ga.C_DBL, (10,20,30))
g_b = ga.create(ga.C_INT, (2,3,4,5,6))

if not me:
    print_distribution(g_a)
    print_distribution(g_b)
