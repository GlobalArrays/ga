##############################################################################
# tcgmsg
#
if MSG_COMMS_TCGMSG
libarmci_la_SOURCES += tcgmsg/capi.c

include_HEADERS += tcgmsg/msgtypesc.h
include_HEADERS += tcgmsg/msgtypesf.h
include_HEADERS += tcgmsg/tcgmsg.fh
include_HEADERS += tcgmsg/tcgmsg.h

AM_CPPFLAGS += -I$(top_srcdir)/tcgmsg

TCGMSG_SERIAL_TESTS =
TCGMSG_SERIAL_TESTS_XFAIL =
TCGMSG_PARALLEL_TESTS =
TCGMSG_PARALLEL_TESTS_XFAIL =
TCGMSG_TESTS = $(TCGMSG_SERIAL_TESTS) $(TCGMSG_PARALLEL_TESTS)
TCGMSG_TESTS_XFAIL = $(TCGMSG_SERIAL_TESTS_XFAIL) $(TCGMSG_PARALLEL_TESTS_XFAIL)

endif # MSG_COMMS_TCGMSG

EXTRA_DIST += tcgmsg/README

##############################################################################
# tcgmsg/examples
#
# NOTE: While we don't build the tcgmsg examples, we include everything in the
# distribution anyway.
#
EXTRA_DIST += tcgmsg/examples/blkdat120lin.f
EXTRA_DIST += tcgmsg/examples/blkdat15.f
EXTRA_DIST += tcgmsg/examples/blkdat240lin.f
EXTRA_DIST += tcgmsg/examples/blkdat30.f
EXTRA_DIST += tcgmsg/examples/blkdat60.f
EXTRA_DIST += tcgmsg/examples/blkdat60lin.f
EXTRA_DIST += tcgmsg/examples/cscf120lin.h
EXTRA_DIST += tcgmsg/examples/cscf15.h
EXTRA_DIST += tcgmsg/examples/cscf240lin.h
EXTRA_DIST += tcgmsg/examples/cscf30.h
EXTRA_DIST += tcgmsg/examples/cscf60.h
EXTRA_DIST += tcgmsg/examples/cscf60lin.h
EXTRA_DIST += tcgmsg/examples/cscf.h
EXTRA_DIST += tcgmsg/examples/daxpy1.s
EXTRA_DIST += tcgmsg/examples/daxpy.f
EXTRA_DIST += tcgmsg/examples/ddot.f
EXTRA_DIST += tcgmsg/examples/demo.proto
EXTRA_DIST += tcgmsg/examples/diagon.f
EXTRA_DIST += tcgmsg/examples/dscal.f
EXTRA_DIST += tcgmsg/examples/fexit.f.proto
EXTRA_DIST += tcgmsg/examples/getmem.c
EXTRA_DIST += tcgmsg/examples/grid.c
EXTRA_DIST += tcgmsg/examples/ieeetrap.c
EXTRA_DIST += tcgmsg/examples/integ.f
EXTRA_DIST += tcgmsg/examples/jacobi.f
EXTRA_DIST += tcgmsg/examples/Makefile.proto
EXTRA_DIST += tcgmsg/examples/mc.f
EXTRA_DIST += tcgmsg/examples/mc.input
EXTRA_DIST += tcgmsg/examples/md.f
EXTRA_DIST += tcgmsg/examples/mxv_daxpy1.f
EXTRA_DIST += tcgmsg/examples/mxv_dgemv.f
EXTRA_DIST += tcgmsg/examples/mxv_fortran.f
EXTRA_DIST += tcgmsg/examples/output.f
EXTRA_DIST += tcgmsg/examples/prtri.f
EXTRA_DIST += tcgmsg/examples/random.c
EXTRA_DIST += tcgmsg/examples/README
EXTRA_DIST += tcgmsg/examples/runit
EXTRA_DIST += tcgmsg/examples/runit.grid
EXTRA_DIST += tcgmsg/examples/scfblas.f
EXTRA_DIST += tcgmsg/examples/scf.f
EXTRA_DIST += tcgmsg/examples/timer.f
EXTRA_DIST += tcgmsg/examples/trace.out
EXTRA_DIST += tcgmsg/examples/xpix.shar

##############################################################################
# tcgmsg/ipcv4.0
#
if MSG_COMMS_TCGMSG
if MSG_COMMS_TCGMSG4
libarmci_la_SOURCES += tcgmsg/ipcv4.0/brdcst.c
libarmci_la_SOURCES += tcgmsg/ipcv4.0/checkbyte.c
libarmci_la_SOURCES += tcgmsg/ipcv4.0/cluster.c
libarmci_la_SOURCES += tcgmsg/ipcv4.0/cluster.h
libarmci_la_SOURCES += tcgmsg/ipcv4.0/defglobals.h
libarmci_la_SOURCES += tcgmsg/ipcv4.0/drand48.c
libarmci_la_SOURCES += tcgmsg/ipcv4.0/error.c
libarmci_la_SOURCES += tcgmsg/ipcv4.0/evlog.c
libarmci_la_SOURCES += tcgmsg/ipcv4.0/evlog.h
libarmci_la_SOURCES += tcgmsg/ipcv4.0/evon.c
libarmci_la_SOURCES += tcgmsg/ipcv4.0/globalop.c
libarmci_la_SOURCES += tcgmsg/ipcv4.0/llog.c
libarmci_la_SOURCES += tcgmsg/ipcv4.0/mdtob.c
libarmci_la_SOURCES += tcgmsg/ipcv4.0/mdtoi.c
libarmci_la_SOURCES += tcgmsg/ipcv4.0/mitob.c
libarmci_la_SOURCES += tcgmsg/ipcv4.0/mitod.c
libarmci_la_SOURCES += tcgmsg/ipcv4.0/mtime.c
libarmci_la_SOURCES += tcgmsg/ipcv4.0/niceftn.c
libarmci_la_SOURCES += tcgmsg/ipcv4.0/nnodes.c
libarmci_la_SOURCES += tcgmsg/ipcv4.0/nodeid.c
libarmci_la_SOURCES += tcgmsg/ipcv4.0/nxtval.c
libarmci_la_SOURCES += tcgmsg/ipcv4.0/pbegin.c
libarmci_la_SOURCES += tcgmsg/ipcv4.0/pbeginf.c
libarmci_la_SOURCES += tcgmsg/ipcv4.0/pfilecopy.c
libarmci_la_SOURCES += tcgmsg/ipcv4.0/sema.c
libarmci_la_SOURCES += tcgmsg/ipcv4.0/sema.h
libarmci_la_SOURCES += tcgmsg/ipcv4.0/setdbg.c
libarmci_la_SOURCES += tcgmsg/ipcv4.0/shmem.c
libarmci_la_SOURCES += tcgmsg/ipcv4.0/signals.c
libarmci_la_SOURCES += tcgmsg/ipcv4.0/signals.h
libarmci_la_SOURCES += tcgmsg/ipcv4.0/snd.c
libarmci_la_SOURCES += tcgmsg/ipcv4.0/sndrcv.h
libarmci_la_SOURCES += tcgmsg/ipcv4.0/sndrcvP.h
libarmci_la_SOURCES += tcgmsg/ipcv4.0/sockets.c
libarmci_la_SOURCES += tcgmsg/ipcv4.0/srftoc.h
libarmci_la_SOURCES += tcgmsg/ipcv4.0/srmover.c
libarmci_la_SOURCES += tcgmsg/ipcv4.0/stats.c
libarmci_la_SOURCES += tcgmsg/ipcv4.0/synch.c
libarmci_la_SOURCES += tcgmsg/ipcv4.0/usleep.c
libarmci_la_SOURCES += tcgmsg/ipcv4.0/waitall.c
libarmci_la_SOURCES += tcgmsg/ipcv4.0/waitcom.c
libarmci_la_SOURCES += tcgmsg/ipcv4.0/xdrstuff.c
libarmci_la_SOURCES += tcgmsg/ipcv4.0/xdrstuff.h

AM_CPPFLAGS += -I$(top_srcdir)/tcgmsg/ipcv4.0

check_PROGRAMS += tcgmsg/ipcv4.0/hello
check_PROGRAMS += tcgmsg/ipcv4.0/test
check_PROGRAMS += tcgmsg/ipcv4.0/testmap
check_PROGRAMS += tcgmsg/ipcv4.0/toplot
check_PROGRAMS += tcgmsg/ipcv4.0/xdrtest
if ENABLE_F77
check_PROGRAMS += tcgmsg/ipcv4.0/testf
check_PROGRAMS += tcgmsg/ipcv4.0/testpf
endif

TCGMSG_PARALLEL_TESTS += tcgmsg/ipcv4.0/hello$(EXEEXT)
#TCGMSG_PARALLEL_TESTS += tcgmsg/ipcv4.0/test$(EXEEXT) # requires user input
#TCGMSG_PARALLEL_TESTS += tcgmsg/ipcv4.0/testmap$(EXEEXT) # requires user input
#TCGMSG_PARALLEL_TESTS += tcgmsg/ipcv4.0/toplot$(EXEEXT) # not a test
#TCGMSG_PARALLEL_TESTS += tcgmsg/ipcv4.0/xdrtest$(EXEEXT) # requires user input
if ENABLE_F77
TCGMSG_PARALLEL_TESTS += tcgmsg/ipcv4.0/testf$(EXEEXT)
#TCGMSG_PARALLEL_TESTS += tcgmsg/ipcv4.0/testpf$(EXEEXT) # requires input files
endif

tcgmsg_ipcv4_0_hello_SOURCES   = tcgmsg/ipcv4.0/hello.c
tcgmsg_ipcv4_0_test_SOURCES    = tcgmsg/ipcv4.0/test.c
tcgmsg_ipcv4_0_testf_SOURCES   = tcgmsg/ipcv4.0/testf.F
tcgmsg_ipcv4_0_testmap_SOURCES = tcgmsg/ipcv4.0/testmap.c
tcgmsg_ipcv4_0_testpf_SOURCES  = tcgmsg/ipcv4.0/testpf.f
tcgmsg_ipcv4_0_toplot_SOURCES  = tcgmsg/ipcv4.0/toplot.c
tcgmsg_ipcv4_0_xdrtest_SOURCES = tcgmsg/ipcv4.0/xdrtest.c

tcgmsg_ipcv4_0_testf_LDFLAGS   = $(FLDFLAGS)
tcgmsg_ipcv4_0_testpf_LDFLAGS  = $(FLDFLAGS)

nodist_EXTRA_tcgmsg_ipcv4_0_hello_SOURCES   = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_tcgmsg_ipcv4_0_test_SOURCES    = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_tcgmsg_ipcv4_0_testmap_SOURCES = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_tcgmsg_ipcv4_0_toplot_SOURCES  = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_tcgmsg_ipcv4_0_xdrtest_SOURCES = $(MAYBE_DUMMY_F_LINK)

bin_PROGRAMS += parallel

parallel_SOURCES = tcgmsg/ipcv4.0/parallel.c
nodist_EXTRA_parallel_SOURCES = $(MAYBE_DUMMY_F_LINK)

endif # MSG_COMMS_TCGMSG4
endif # MSG_COMMS_TCGMSG

EXTRA_DIST += tcgmsg/ipcv4.0/README

##############################################################################
# tcgmsg/ipcv5.0
#
if MSG_COMMS_TCGMSG
if MSG_COMMS_TCGMSG5
libarmci_la_SOURCES += tcgmsg/ipcv5.0/drand48.c
libarmci_la_SOURCES += tcgmsg/ipcv5.0/evlog.h
libarmci_la_SOURCES += tcgmsg/ipcv5.0/globalop.c
libarmci_la_SOURCES += tcgmsg/ipcv5.0/llog.c
libarmci_la_SOURCES += tcgmsg/ipcv5.0/mdtob.c
libarmci_la_SOURCES += tcgmsg/ipcv5.0/mdtoi.c
libarmci_la_SOURCES += tcgmsg/ipcv5.0/misc.c
libarmci_la_SOURCES += tcgmsg/ipcv5.0/mitob.c
libarmci_la_SOURCES += tcgmsg/ipcv5.0/mitod.c
libarmci_la_SOURCES += tcgmsg/ipcv5.0/mtime.c
libarmci_la_SOURCES += tcgmsg/ipcv5.0/niceftn.c
libarmci_la_SOURCES += tcgmsg/ipcv5.0/nnodes.c
libarmci_la_SOURCES += tcgmsg/ipcv5.0/nodeid.c
libarmci_la_SOURCES += tcgmsg/ipcv5.0/pbegin.c
libarmci_la_SOURCES += tcgmsg/ipcv5.0/pfilecopy.c
libarmci_la_SOURCES += tcgmsg/ipcv5.0/queues.c
libarmci_la_SOURCES += tcgmsg/ipcv5.0/signals.c
libarmci_la_SOURCES += tcgmsg/ipcv5.0/sndrcv.h
libarmci_la_SOURCES += tcgmsg/ipcv5.0/snd_rcv_probe.c
libarmci_la_SOURCES += tcgmsg/ipcv5.0/srftoc.h
libarmci_la_SOURCES += tcgmsg/ipcv5.0/tcgmsgP.h
libarmci_la_SOURCES += tcgmsg/ipcv5.0/usleep.c
libarmci_la_SOURCES += tcgmsg/ipcv5.0/waitall.c
if ARMCI_NETWORK_LAPI
libarmci_la_SOURCES += tcgmsg/ipcv5.0/async_send_lapi.c
libarmci_la_SOURCES += tcgmsg/ipcv5.0/lapi_putget.c
else
libarmci_la_SOURCES += tcgmsg/ipcv5.0/async_send.c
libarmci_la_SOURCES += tcgmsg/ipcv5.0/busy.c
libarmci_la_SOURCES += tcgmsg/ipcv5.0/error.c
libarmci_la_SOURCES += tcgmsg/ipcv5.0/nxtval.shm.c
libarmci_la_SOURCES += tcgmsg/ipcv5.0/pbeginf.c
libarmci_la_SOURCES += tcgmsg/ipcv5.0/shmem.c
libarmci_la_SOURCES += tcgmsg/ipcv5.0/synch.c
endif # ARMCI_NETWORK_LAPI

AM_CPPFLAGS += -I$(top_srcdir)/tcgmsg/ipcv5.0

check_PROGRAMS += tcgmsg/ipcv5.0/hello
check_PROGRAMS += tcgmsg/ipcv5.0/test
if MSG_COMMS_PVM
check_PROGRAMS += tcgmsg/ipcv5.0/pvm.ring
endif
if ENABLE_F77
check_PROGRAMS += tcgmsg/ipcv5.0/jacobi
check_PROGRAMS += tcgmsg/ipcv5.0/testf
endif

TCGMSG_PARALLEL_TESTS += tcgmsg/ipcv5.0/hello$(EXEEXT)
#TCGMSG_PARALLEL_TESTS += tcgmsg/ipcv5.0/test$(EXEEXT) # requires user input
if MSG_COMMS_PVM
TCGMSG_PARALLEL_TESTS += tcgmsg/ipcv5.0/pvm.ring$(EXEEXT)
endif
if ENABLE_F77
#TCGMSG_PARALLEL_TESTS += tcgmsg/ipcv5.0/jacobi$(EXEEXT) # requires user input
#TCGMSG_PARALLEL_TESTS += tcgmsg/ipcv5.0/testf$(EXEEXT) # requires user input
endif

tcgmsg_ipcv5_0_hello_SOURCES    = tcgmsg/ipcv5.0/hello.c
tcgmsg_ipcv5_0_jacobi_SOURCES   = tcgmsg/ipcv5.0/fexit.f       \
                                  tcgmsg/ipcv5.0/getmem.c      \
                                  tcgmsg/ipcv5.0/jacobi.F      \
                                  tcgmsg/ipcv5.0/mxv_fortran.f \
                                  tcgmsg/ipcv5.0/timer.f
tcgmsg_ipcv5_0_pvm_ring_SOURCES = tcgmsg/ipcv5.0/checkbyte.c   \
                                  tcgmsg/ipcv5.0/pvm.c
tcgmsg_ipcv5_0_testf_SOURCES    = tcgmsg/ipcv5.0/testf.F
tcgmsg_ipcv5_0_test_SOURCES     = tcgmsg/ipcv5.0/checkbyte.c   \
                                  tcgmsg/ipcv5.0/test.c

tcgmsg_ipcv5_0_jacobi_LDFLAGS   = $(FLDFLAGS)
tcgmsg_ipcv5_0_testf_LDFLAGS    = $(FLDFLAGS)

nodist_EXTRA_tcgmsg_ipcv5_0_hello_SOURCES    = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_tcgmsg_ipcv5_0_pvm_ring_SOURCES = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_tcgmsg_ipcv5_0_test_SOURCES     = $(MAYBE_DUMMY_F_LINK)

endif # MSG_COMMS_TCGMSG5
endif # MSG_COMMS_TCGMSG

##############################################################################
# tcgmsg/tcgmsg-mpi
#
if MSG_COMMS_TCGMSG
if MSG_COMMS_TCGMSGMPI
libarmci_la_SOURCES += tcgmsg/tcgmsg-mpi/checkbyte.c
libarmci_la_SOURCES += tcgmsg/tcgmsg-mpi/clustercheck.c
libarmci_la_SOURCES += tcgmsg/tcgmsg-mpi/collect.c
libarmci_la_SOURCES += tcgmsg/tcgmsg-mpi/drand48.c
libarmci_la_SOURCES += tcgmsg/tcgmsg-mpi/evlog.c
libarmci_la_SOURCES += tcgmsg/tcgmsg-mpi/evlog.h
libarmci_la_SOURCES += tcgmsg/tcgmsg-mpi/evon.c
libarmci_la_SOURCES += tcgmsg/tcgmsg-mpi/llog.c
libarmci_la_SOURCES += tcgmsg/tcgmsg-mpi/misc.c
libarmci_la_SOURCES += tcgmsg/tcgmsg-mpi/niceftn.c
libarmci_la_SOURCES += tcgmsg/tcgmsg-mpi/nxtval-armci.c
libarmci_la_SOURCES += tcgmsg/tcgmsg-mpi/p2p.c
libarmci_la_SOURCES += tcgmsg/tcgmsg-mpi/pbeginf.c
libarmci_la_SOURCES += tcgmsg/tcgmsg-mpi/pfilecopy.c
libarmci_la_SOURCES += tcgmsg/tcgmsg-mpi/sizeof.c
libarmci_la_SOURCES += tcgmsg/tcgmsg-mpi/sndrcv.h
libarmci_la_SOURCES += tcgmsg/tcgmsg-mpi/srftoc.h
libarmci_la_SOURCES += tcgmsg/tcgmsg-mpi/tcgmsgP.h

AM_CPPFLAGS += -I$(top_srcdir)/tcgmsg/tcgmsg-mpi

check_PROGRAMS += tcgmsg/tcgmsg-mpi/test_asyn
check_PROGRAMS += tcgmsg/tcgmsg-mpi/test
if ENABLE_F77
check_PROGRAMS += tcgmsg/tcgmsg-mpi/testf
endif

#TCGMSG_PARALLEL_TESTS += tcgmsg/tcgmsg-mpi/test_asyn$(EXEEXT) # useless test
#TCGMSG_PARALLEL_TESTS += tcgmsg/tcgmsg-mpi/test$(EXEEXT) # requires user input
if ENABLE_F77
TCGMSG_PARALLEL_TESTS += tcgmsg/tcgmsg-mpi/testf$(EXEEXT)
endif

tcgmsg_tcgmsg_mpi_test_asyn_SOURCES = tcgmsg/tcgmsg-mpi/test_asyn.c
tcgmsg_tcgmsg_mpi_testf_SOURCES     = tcgmsg/tcgmsg-mpi/testf.F
tcgmsg_tcgmsg_mpi_test_SOURCES      = tcgmsg/tcgmsg-mpi/test.c

tcgmsg_tcgmsg_mpi_testf_LDFLAGS     = $(FLDFLAGS)

nodist_EXTRA_tcgmsg_tcgmsg_mpi_test_asyn_SOURCES = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_tcgmsg_tcgmsg_mpi_test_SOURCES      = $(MAYBE_DUMMY_F_LINK)

endif # MSG_COMMS_TCGMSGMPI
endif # MSG_COMMS_TCGMSG

EXTRA_DIST += tcgmsg/tcgmsg-mpi/README
