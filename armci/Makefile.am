##############################################################################
# Makefile.am for ARMCI.
#
# Rationale:
# This Makefile.am follows many of the suggestions outlined in the paper
# "Recursive Make Considered Harmful".  We do not use Automake's
# 'include' feature (instead preferring a single, large Makefile.am).
#
# Additional targets:
# Besides the traditional make targets supplied by Automake, we have added the
# "examples" and "checkprogs" targets to build example programs and test
# programs, respectively.
#
# Notes:
# In general, each subdirectory has a corresponding section of this
# Makefile.am with the notable exception being the many examples getting
# rolled up into the examples section.
#

# The usual aclocal nonsense to get include paths right.
ACLOCAL_AMFLAGS = -I m4 -I ../m4

# All public headers, installed programs, test programs, and example programs
# are listed in these variables.  Appended to throughout.  These are the
# automake variables used.
include_HEADERS =
bin_PROGRAMS =
check_PROGRAMS =
check_LTLIBRARIES =
lib_LTLIBRARIES = libarmci.la
noinst_LTLIBRARIES = libarmcii.la
libarmci_la_SOURCES =
libarmcii_la_SOURCES = $(libarmci_la_SOURCES)
EXTRA_DIST =
BUILT_SOURCES =
AM_FFLAGS =
AM_CFLAGS =
AM_CXXFLAGS =
AM_CPPFLAGS =
AM_LDFLAGS =
LDADD =

AM_FFLAGS += $(ARMCI_FOPT)
AM_FFLAGS += $(FFLAG_INT_4)
AM_CFLAGS += $(ARMCI_COPT)
AM_CXXFLAGS += $(ARMCI_CXXOPT)
AM_CPPFLAGS += $(GA_MP_CPPFLAGS)
AM_CPPFLAGS += $(ARMCI_NETWORK_CPPFLAGS)
if MSG_COMMS_TCGMSG
AM_CPPFLAGS += -I$(top_srcdir)/../tcgmsg
endif
AM_LDFLAGS += $(GA_MP_LDFLAGS)
AM_LDFLAGS += $(ARMCI_NETWORK_LDFLAGS)
if NOFORT
else
AM_LDFLAGS += $(FLD_NOMAIN)
endif
if MSG_COMMS_TCGMSG
LDADD += $(top_bld_prefix)../libga.la
else
LDADD += libarmci.la
endif
LDADD += $(LTLIBOBJS)
LDADD += $(GA_MP_LIBS)
LDADD += $(ARMCI_NETWORK_LIBS)

# Certain trickery when turning Fortran support on or off.
if NOFORT
else
MAYBE_DUMMY_F_LINK = dummy.f
MAYBE_FLIBS = $(FLIBS)
endif

##############################################################################
# config
#
# The config.h file is automatically generated from config.h.in at configure
# time.  However, it contains symbols which certain Fortran compilers do not
# understand.  Therefore, we create a config.fh file which is Fortran-specific
# and works with all Fortran compilers.
#
BUILT_SOURCES += config.fh

config.fh: config.h
	-rm -f config.fh && $(SED) "/^#/!d" config.h > config.fh

clean-config:
	-rm -f config.fh

##############################################################################
# src-portals
#
if ARMCI_NETWORK_PORTALS
libarmci_la_SOURCES += src-portals/acc.c
libarmci_la_SOURCES += src-portals/acc.h
libarmci_la_SOURCES += src-portals/aggregate.c
libarmci_la_SOURCES += src-portals/armci.c
libarmci_la_SOURCES += src-portals/armcip.h
libarmci_la_SOURCES += src-portals/armci_portals.c
libarmci_la_SOURCES += src-portals/armci_portals.h
libarmci_la_SOURCES += src-portals/armci_profile.h
libarmci_la_SOURCES += src-portals/atomics-i386.h
libarmci_la_SOURCES += src-portals/buffers.c
libarmci_la_SOURCES += src-portals/capi.c
libarmci_la_SOURCES += src-portals/clusterinfo.c
libarmci_la_SOURCES += src-portals/copy.h
libarmci_la_SOURCES += src-portals/ds-shared.c
libarmci_la_SOURCES += src-portals/fence.c
libarmci_la_SOURCES += src-portals/gpc.h
libarmci_la_SOURCES += src-portals/kr_malloc.c
libarmci_la_SOURCES += src-portals/kr_malloc.h
libarmci_la_SOURCES += src-portals/locks.c
libarmci_la_SOURCES += src-portals/locks.h
libarmci_la_SOURCES += src-portals/memlock.c
libarmci_la_SOURCES += src-portals/memlock.h
libarmci_la_SOURCES += src-portals/memory.c
libarmci_la_SOURCES += src-portals/message.c
libarmci_la_SOURCES += src-portals/mp3.h
libarmci_la_SOURCES += src-portals/mutex.c
libarmci_la_SOURCES += src-portals/pack.c
libarmci_la_SOURCES += src-portals/pendbufs.h
libarmci_la_SOURCES += src-portals/portals_cp.c
libarmci_la_SOURCES += src-portals/portals_ds.c
libarmci_la_SOURCES += src-portals/portals.c
libarmci_la_SOURCES += src-portals/request.c
libarmci_la_SOURCES += src-portals/request.h
libarmci_la_SOURCES += src-portals/rmw.c
libarmci_la_SOURCES += src-portals/rtinfo.c
libarmci_la_SOURCES += src-portals/semaphores.c
libarmci_la_SOURCES += src-portals/semaphores.h
libarmci_la_SOURCES += src-portals/shmalloc.h
libarmci_la_SOURCES += src-portals/shmem.c
libarmci_la_SOURCES += src-portals/shmem.h
libarmci_la_SOURCES += src-portals/shmlimit.c
libarmci_la_SOURCES += src-portals/shmlimit.h
libarmci_la_SOURCES += src-portals/signaltrap.c
libarmci_la_SOURCES += src-portals/signaltrap.h
libarmci_la_SOURCES += src-portals/sockets.h
libarmci_la_SOURCES += src-portals/spawn.c
libarmci_la_SOURCES += src-portals/spinlock.h
libarmci_la_SOURCES += src-portals/strided.c
libarmci_la_SOURCES += src-portals/tas-alpha.h
libarmci_la_SOURCES += src-portals/tas-i386.h
libarmci_la_SOURCES += src-portals/tas-ia64.h
libarmci_la_SOURCES += src-portals/tas-ppc.h
libarmci_la_SOURCES += src-portals/utils.h
libarmci_la_SOURCES += src-portals/vector.c
if PORTALS_ENABLE_NEW_MALLOC
libarmci_la_SOURCES += src-portals/new_memory.c
endif
if NOFORT
else
libarmci_la_SOURCES += src-portals/faccumulate.f
libarmci_la_SOURCES += src-portals/fcopy.f
endif
if MSG_COMMS_MPI
libarmci_la_SOURCES += src-portals/groups.c
endif
if ARMCI_ENABLE_GPC_CALLS
libarmci_la_SOURCES += src-portals/gpc.c
endif
if ENABLE_PROFILING
libarmci_la_SOURCES += src-portals/armci_profile.c
endif
if THREAD_SAFE
libarmci_la_SOURCES += src-portals/threads.c
libarmci_la_SOURCES += src-portals/utils.c
endif
if INLINE_TAS_OKAY
libarmci_la_SOURCES += src-portals/tas.c
else
if INLINE_TAS_WITH
BUILT_SOURCES       += src-portals/tas_as.s
libarmci_la_SOURCES += src-portals/tas_as.s
endif
endif

include_HEADERS += src-portals/armci.h
include_HEADERS += src-portals/message.h
include_HEADERS += src-portals/parmci.h

AM_CPPFLAGS += -I$(top_srcdir)/src-portals

if ENABLE_SHARED
MAYBE_PIC = -fPIC -DPIC
endif

src-portals/tas_as.s: src-portals/tas.c
	$(ARMCI_TAS_AS) $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) \
	    $(AM_CPPFLAGS) $(CPPFLAGS) -S $(MAYBE_PIC) -o $@ $<

clean-asm:
	rm -f src-portals/tas_as.s

check_PROGRAMS += testing-portals/clone
check_PROGRAMS += testing-portals/fork
check_PROGRAMS += testing-portals/msgcheck
check_PROGRAMS += testing-portals/origptl
check_PROGRAMS += testing-portals/perf_aggr
check_PROGRAMS += testing-portals/perf_nb
check_PROGRAMS += testing-portals/perf
check_PROGRAMS += testing-portals/ptltest
check_PROGRAMS += testing-portals/shmclean
check_PROGRAMS += testing-portals/shmtest
check_PROGRAMS += testing-portals/simplelock
check_PROGRAMS += testing-portals/simple
check_PROGRAMS += testing-portals/test2
check_PROGRAMS += testing-portals/testnotify
check_PROGRAMS += testing-portals/test
check_PROGRAMS += testing-portals/te
if ARMCI_ENABLE_GPC_CALLS
check_PROGRAMS += testing-portals/gpctest
endif
if MSG_COMMS_MPI
check_PROGRAMS += testing-portals/ipctest
check_PROGRAMS += testing-portals/test_groups
endif
if THREAD_SAFE
check_PROGRAMS += testing-portals/test_mt
endif

testing_portals_clone_SOURCES       = testing-portals/clone.c
testing_portals_fork_SOURCES        = testing-portals/fork.c
testing_portals_gpctest_SOURCES     = testing-portals/gpctest.c
testing_portals_ipctest_SOURCES     = testing-portals/ipctest.c
testing_portals_msgcheck_SOURCES    = testing-portals/msgcheck.c
testing_portals_origptl_SOURCES     = testing-portals/origptl.c
testing_portals_perf_aggr_SOURCES   = testing-portals/perf_aggr.c
testing_portals_perf_nb_SOURCES     = testing-portals/perf_nb.c
testing_portals_perf_SOURCES        = testing-portals/perf.c
testing_portals_ptltest_SOURCES     = testing-portals/ptltest.c
testing_portals_shmclean_SOURCES    = testing-portals/shmclean.c
testing_portals_shmtest_SOURCES     = testing-portals/shmtest.c
testing_portals_simple_SOURCES      = testing-portals/simple.c
testing_portals_simplelock_SOURCES  = testing-portals/simplelock.c
testing_portals_te_SOURCES          = testing-portals/te.c
testing_portals_test2_SOURCES       = testing-portals/test2.c
testing_portals_test_groups_SOURCES = testing-portals/test_groups.c
testing_portals_test_mt_SOURCES     = testing-portals/test_mt.c
testing_portals_testnotify_SOURCES  = testing-portals/testnotify.c
testing_portals_test_SOURCES        = testing-portals/test.c

nodist_EXTRA_testing_portals_clone_SOURCES       = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_testing_portals_fork_SOURCES        = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_testing_portals_gpctest_SOURCES     = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_testing_portals_ipctest_SOURCES     = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_testing_portals_msgcheck_SOURCES    = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_testing_portals_origptl_SOURCES     = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_testing_portals_perf_aggr_SOURCES   = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_testing_portals_perf_nb_SOURCES     = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_testing_portals_perf_SOURCES        = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_testing_portals_ptltest_SOURCES     = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_testing_portals_shmclean_SOURCES    = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_testing_portals_shmtest_SOURCES     = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_testing_portals_simple_SOURCES      = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_testing_portals_simplelock_SOURCES  = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_testing_portals_te_SOURCES          = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_testing_portals_test2_SOURCES       = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_testing_portals_test_groups_SOURCES = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_testing_portals_test_mt_SOURCES     = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_testing_portals_testnotify_SOURCES  = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_testing_portals_test_SOURCES        = $(MAYBE_DUMMY_F_LINK)

##############################################################################
# src
#
else # !ARMCI_NETWORK_PORTALS
libarmci_la_SOURCES += src/acc.h
libarmci_la_SOURCES += src/aggregate.c
libarmci_la_SOURCES += src/armci.c
libarmci_la_SOURCES += src/armci_chkpt.h
libarmci_la_SOURCES += src/armcip.h
libarmci_la_SOURCES += src/armci_portals.h
libarmci_la_SOURCES += src/armci_profile.h
libarmci_la_SOURCES += src/armci_storage.h
libarmci_la_SOURCES += src/armci_vampir.h
libarmci_la_SOURCES += src/armci-vapi.h
libarmci_la_SOURCES += src/atomics-i386.h
libarmci_la_SOURCES += src/bgmlacc.h
libarmci_la_SOURCES += src/bgmldefs.h
libarmci_la_SOURCES += src/bgml.h
libarmci_la_SOURCES += src/bgmltas-ppc.h
libarmci_la_SOURCES += src/caccumulate.c
libarmci_la_SOURCES += src/capi.c
libarmci_la_SOURCES += src/cbuf.h
libarmci_la_SOURCES += src/ccopy.c
libarmci_la_SOURCES += src/clusterinfo.c
libarmci_la_SOURCES += src/copy.h
libarmci_la_SOURCES += src/elandefs.h
libarmci_la_SOURCES += src/fence.c
libarmci_la_SOURCES += src/fujitsu-vpp.h
libarmci_la_SOURCES += src/gpc.h
libarmci_la_SOURCES += src/iterator.c
libarmci_la_SOURCES += src/iterator.h
libarmci_la_SOURCES += src/kr_malloc.c
libarmci_la_SOURCES += src/kr_malloc.h
libarmci_la_SOURCES += src/lapidefs.h
libarmci_la_SOURCES += src/locks.c
libarmci_la_SOURCES += src/locks.h
libarmci_la_SOURCES += src/memlock.c
libarmci_la_SOURCES += src/memlock.h
libarmci_la_SOURCES += src/memory.c
libarmci_la_SOURCES += src/message.c
libarmci_la_SOURCES += src/mp3.h
libarmci_la_SOURCES += src/mpi2.h
libarmci_la_SOURCES += src/mutex.c
libarmci_la_SOURCES += src/pack.c
libarmci_la_SOURCES += src/parmci.h
libarmci_la_SOURCES += src/pendbufs.h
libarmci_la_SOURCES += src/pgs_sys.h
libarmci_la_SOURCES += src/pgs_thread.h
libarmci_la_SOURCES += src/portals.h
libarmci_la_SOURCES += src/request.h
libarmci_la_SOURCES += src/rmw.c
libarmci_la_SOURCES += src/rtinfo.c
libarmci_la_SOURCES += src/semaphores.c
libarmci_la_SOURCES += src/semaphores.h
libarmci_la_SOURCES += src/shmalloc.h
libarmci_la_SOURCES += src/shmem.h
libarmci_la_SOURCES += src/shmlimit.h
libarmci_la_SOURCES += src/signaltrap.h
libarmci_la_SOURCES += src/sockets.h
libarmci_la_SOURCES += src/spinlock.h
libarmci_la_SOURCES += src/strided.c
libarmci_la_SOURCES += src/tas-alpha.h
libarmci_la_SOURCES += src/tas-i386.h
libarmci_la_SOURCES += src/tas-ia64.h
libarmci_la_SOURCES += src/tas-ppc.h
libarmci_la_SOURCES += src/utils.h
libarmci_la_SOURCES += src/vector.c
if NOFORT
else # !NOFORT
libarmci_la_SOURCES += src/faccumulate.f
libarmci_la_SOURCES += src/fcopy.f
endif # NOFORT
if MSG_COMMS_MPI
libarmci_la_SOURCES += src/groups.c
endif
if ARMCI_NETWORK_BGML
libarmci_la_SOURCES += src/bgml.c
libarmci_la_SOURCES += src/groups.c
libarmci_la_SOURCES += src/message.c
endif
if ARMCI_NETWORK_CRAY_SHMEM
libarmci_la_SOURCES += src/shmalloc.c
endif
if ARMCI_NETWORK_DCMF
libarmci_la_SOURCES += src/groups.c
libarmci_la_SOURCES += src/message.c
libarmci_la_SOURCES += src/request.c
if DCMF_VER_2
libarmci_la_SOURCES += src/armcix/dcmf2/armcix_acc.c
libarmci_la_SOURCES += src/armcix/dcmf2/armcix_fence.c
libarmci_la_SOURCES += src/armcix/dcmf2/armcix_get.c
libarmci_la_SOURCES += src/armcix/dcmf2/armcix_impl.c
libarmci_la_SOURCES += src/armcix/dcmf2/armcix_lock.c
libarmci_la_SOURCES += src/armcix/dcmf2/armcix_put.c
libarmci_la_SOURCES += src/armcix/dcmf2/armcix_rmw.c
libarmci_la_SOURCES += src/armcix/dcmf2/armcix_wait.c
endif
if DCMF_VER_0_2
libarmci_la_SOURCES += src/armcix/dcmf-0.2.0/armcix_acc.c
libarmci_la_SOURCES += src/armcix/dcmf-0.2.0/armcix_fence.c
libarmci_la_SOURCES += src/armcix/dcmf-0.2.0/armcix_get.c
libarmci_la_SOURCES += src/armcix/dcmf-0.2.0/armcix_impl.c
libarmci_la_SOURCES += src/armcix/dcmf-0.2.0/armcix_lock.c
libarmci_la_SOURCES += src/armcix/dcmf-0.2.0/armcix_put.c
libarmci_la_SOURCES += src/armcix/dcmf-0.2.0/armcix_rmw.c
libarmci_la_SOURCES += src/armcix/dcmf-0.2.0/armcix_wait.c
endif
if DCMF_VER_0_3
libarmci_la_SOURCES += src/armcix/dcmf-0.3.0/armcix_acc.c
libarmci_la_SOURCES += src/armcix/dcmf-0.3.0/armcix_fence.c
libarmci_la_SOURCES += src/armcix/dcmf-0.3.0/armcix_get.c
libarmci_la_SOURCES += src/armcix/dcmf-0.3.0/armcix_impl.c
libarmci_la_SOURCES += src/armcix/dcmf-0.3.0/armcix_lock.c
libarmci_la_SOURCES += src/armcix/dcmf-0.3.0/armcix_put.c
libarmci_la_SOURCES += src/armcix/dcmf-0.3.0/armcix_rmw.c
libarmci_la_SOURCES += src/armcix/dcmf-0.3.0/armcix_wait.c
endif
endif # ARMCI_NETWORK_DCMF
if ARMCI_NETWORK_LAPI
libarmci_la_SOURCES += src/async.c
libarmci_la_SOURCES += src/buffers.c
libarmci_la_SOURCES += src/lapi.c
libarmci_la_SOURCES += src/request.c
endif
if ARMCI_NETWORK_MPI_SPAWN
libarmci_la_SOURCES += src/buffers.c
libarmci_la_SOURCES += src/ds-shared.c
libarmci_la_SOURCES += src/mpi2_client.c
libarmci_la_SOURCES += src/mpi2_server.c
libarmci_la_SOURCES += src/request.c
endif
if ARMCI_NETWORK_OPENIB
libarmci_la_SOURCES += src/async.c
libarmci_la_SOURCES += src/buffers.c
libarmci_la_SOURCES += src/cbuf.c
libarmci_la_SOURCES += src/ds-shared.c
libarmci_la_SOURCES += src/gpc.c
libarmci_la_SOURCES += src/openib.c
libarmci_la_SOURCES += src/pendbufs.c
libarmci_la_SOURCES += src/regions.c
libarmci_la_SOURCES += src/request.c
libarmci_la_SOURCES += src/spawn.c
endif
if ARMCI_NETWORK_SOCKETS
libarmci_la_SOURCES += src/async.c
libarmci_la_SOURCES += src/buffers.c
libarmci_la_SOURCES += src/dataserv.c
libarmci_la_SOURCES += src/ds-shared.c
libarmci_la_SOURCES += src/request.c
libarmci_la_SOURCES += src/sockets.c
libarmci_la_SOURCES += src/spawn.c
endif
#if ARMCI_NETWORK_VIA
#libarmci_la_SOURCES += src/async.c
#libarmci_la_SOURCES += src/buffers.c
#libarmci_la_SOURCES += src/ds-shared.c
#libarmci_la_SOURCES += src/request.c
#libarmci_la_SOURCES += src/spawn.c
#endif
if ENABLE_CHECKPOINT
libarmci_la_SOURCES += src/armci_chkpt.c
libarmci_la_SOURCES += src/armci_storage.c
endif
if ARMCI_ENABLE_GPC_CALLS
libarmci_la_SOURCES += src/gpc.c
endif
if MACX
libarmci_la_SOURCES += src/shmalloc.c
libarmci_la_SOURCES += src/signaltrap.c
libarmci_la_SOURCES += src/winshmem.c
AM_CPPFLAGS += -DSHMEM -DMMAP
else
if SYSV
libarmci_la_SOURCES += src/shmem.c
libarmci_la_SOURCES += src/shmlimit.c
libarmci_la_SOURCES += src/signaltrap.c
endif
endif
if NB_NONCONT
libarmci_la_SOURCES += src/noncont.c
endif
if USE_VAMPIR
libarmci_la_SOURCES += src/armci_vampir.c
endif
if ENABLE_PROFILING
libarmci_la_SOURCES += src/armci_profile.c
endif
if THREAD_SAFE
libarmci_la_SOURCES += src/threads.c
libarmci_la_SOURCES += src/utils.c
endif
if INLINE_TAS_OKAY
libarmci_la_SOURCES += src/tas.c
else
if INLINE_TAS_WITH
BUILT_SOURCES       += src/tas_as.s
libarmci_la_SOURCES += src/tas_as.s
endif
endif
if INLINE_X86COPY_NEEDED
if INLINE_X86COPY_OKAY
libarmci_la_SOURCES += src/x86copy.c
else
if INLINE_X86COPY_WITH
BUILT_SOURCES       += src/x86copy_as.s
libarmci_la_SOURCES += src/x86copy_as.s
endif
endif
endif

include_HEADERS += src/armci.h
include_HEADERS += src/message.h

AM_CPPFLAGS += -I$(top_srcdir)/src

if ENABLE_SHARED
MAYBE_PIC = -fPIC -DPIC
endif

src/tas_as.s: src/tas.c
	$(ARMCI_TAS_AS) $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) \
	    $(AM_CPPFLAGS) $(CPPFLAGS) -S $(MAYBE_PIC) -o $@ $<

src/x86copy_as.s: src/x86copy.c
	$(ARMCI_X86COPY_AS) $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) \
	    $(AM_CPPFLAGS) $(CPPFLAGS) -S $(MAYBE_PIC) -o $@ $<

clean-asm:
	rm -f src/tas_as.s src/x86copy_as.s

check_PROGRAMS += testing/msgcheck
check_PROGRAMS += testing/perf_aggr
check_PROGRAMS += testing/perf_nb
check_PROGRAMS += testing/perf
check_PROGRAMS += testing/shmclean
check_PROGRAMS += testing/shmtest
check_PROGRAMS += testing/test2
check_PROGRAMS += testing/testitr
check_PROGRAMS += testing/testnotify
check_PROGRAMS += testing/test
if ENABLE_CHECKPOINT
check_PROGRAMS += testing/fttest
endif
if ARMCI_ENABLE_GPC_CALLS
check_PROGRAMS += testing/gpctest
endif
if MSG_COMMS_MPI
check_PROGRAMS += testing/test_groups
if SYSV
check_PROGRAMS += testing/ipctest
endif
endif
if THREAD_SAFE
check_PROGRAMS += testing/test_mt
endif
if NOFORT
else
check_PROGRAMS += testing/acc_test
check_PROGRAMS += testing/copy_test
endif

testing_acc_test_SOURCES    = testing/acc_test.c
testing_copy_test_SOURCES   = testing/copy_test.c
testing_fttest_SOURCES      = testing/fttest.c
testing_gpctest_SOURCES     = testing/gpctest.c
testing_ipctest_SOURCES     = testing/ipctest.c
testing_msgcheck_SOURCES    = testing/msgcheck.c
testing_perf_aggr_SOURCES   = testing/perf_aggr.c
testing_perf_nb_SOURCES     = testing/perf_nb.c
testing_perf_SOURCES        = testing/perf.c
testing_shmclean_SOURCES    = testing/shmclean.c
testing_shmtest_SOURCES     = testing/shmtest.c
testing_test2_SOURCES       = testing/test2.c
testing_test_groups_SOURCES = testing/test_groups.c
testing_testitr_SOURCES     = testing/testitr.c
testing_test_mt_SOURCES     = testing/test_mt.c
testing_testnotify_SOURCES  = testing/testnotify.c
testing_test_SOURCES        = testing/test.c

nodist_EXTRA_testing_acc_test_SOURCES    = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_testing_copy_test_SOURCES   = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_testing_fttest_SOURCES      = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_testing_gpctest_SOURCES     = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_testing_ipctest_SOURCES     = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_testing_msgcheck_SOURCES    = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_testing_perf_aggr_SOURCES   = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_testing_perf_nb_SOURCES     = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_testing_perf_SOURCES        = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_testing_shmclean_SOURCES    = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_testing_shmtest_SOURCES     = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_testing_test2_SOURCES       = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_testing_test_groups_SOURCES = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_testing_testitr_SOURCES     = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_testing_test_mt_SOURCES     = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_testing_testnotify_SOURCES  = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_testing_test_SOURCES        = $(MAYBE_DUMMY_F_LINK)

endif # ARMCI_NETWORK_PORTALS

##############################################################################
# examples
#
check_PROGRAMS += examples/benchmarks/cg/armci_sharedmemory/cg
check_PROGRAMS += examples/benchmarks/lu/armci_blocking/lu-b-bc
check_PROGRAMS += examples/benchmarks/lu/armci_blocking/lu-block
check_PROGRAMS += examples/benchmarks/lu/armci_blocking/lu
check_PROGRAMS += examples/benchmarks/lu/armci_nonblocking/lu_nb_get
check_PROGRAMS += examples/benchmarks/lu/armci_nonblocking/lu_nb_put
check_PROGRAMS += examples/benchmarks/lu/pthreads/lu-thread
check_PROGRAMS += examples/features/aggregation/simple/simple
check_PROGRAMS += examples/features/aggregation/sparse_matvecmul/sparse_matvecmul
check_PROGRAMS += examples/features/concurrency/simple/comdegree
check_PROGRAMS += examples/features/non-blocking/simple/simple
check_PROGRAMS += examples/features/notification/simple/testnotify
check_PROGRAMS += examples/features/symmetric_memory/simple/simple
check_PROGRAMS += examples/simpleApps/transp1D-c
if ARMCI_ENABLE_GPC_CALLS
check_PROGRAMS += examples/features/gpc/hashtable/test_hashtable
check_PROGRAMS += examples/features/gpc/simple/gpctest
check_PROGRAMS += examples/features/proc_affinity/computation_impact/computation_impact
check_PROGRAMS += examples/features/proc_affinity/simple/simple
endif
if MSG_COMMS_MPI
check_PROGRAMS += examples/features/concurrency/multidma/multidma
check_PROGRAMS += examples/features/non-blocking/overlap/overlap
endif
if RANDOM_ACCESS
check_PROGRAMS += examples/benchmarks/RandomAccess/simple/simple
check_PROGRAMS += examples/benchmarks/RandomAccess/smp_bucket/smp_bucket
endif
if THREAD_SAFE
check_PROGRAMS += examples/benchmarks/lu/armci_multithreaded/lu-block-th
endif

examples_benchmarks_cg_armci_sharedmemory_cg_SOURCES = \
examples/benchmarks/cg/armci_sharedmemory/cg.c         \
examples/benchmarks/cg/armci_sharedmemory/compute.c    \
examples/benchmarks/cg/armci_sharedmemory/read_input.c \
examples/benchmarks/cg/armci_sharedmemory/timing.c

examples_benchmarks_lu_armci_blocking_lu_SOURCES = \
examples/benchmarks/lu/armci_blocking/lu.c         \
examples/benchmarks/lu/armci_blocking/timing.c

examples_benchmarks_lu_armci_blocking_lu_block_SOURCES = \
examples/benchmarks/lu/armci_blocking/lu-block.c         \
examples/benchmarks/lu/armci_blocking/timing.c

examples_benchmarks_lu_armci_blocking_lu_b_bc_SOURCES =  \
examples/benchmarks/lu/armci_blocking/lu-b-bc.c          \
examples/benchmarks/lu/armci_blocking/timing.c

examples_benchmarks_lu_armci_multithreaded_lu_block_th_SOURCES = \
examples/benchmarks/lu/armci_multithreaded/lu-block-th.c         \
examples/benchmarks/lu/armci_multithreaded/timing.c

examples_benchmarks_lu_armci_nonblocking_lu_nb_put_SOURCES = \
examples/benchmarks/lu/armci_nonblocking/lu_nb_put.c         \
examples/benchmarks/lu/armci_nonblocking/timing.c

examples_benchmarks_lu_armci_nonblocking_lu_nb_get_SOURCES = \
examples/benchmarks/lu/armci_nonblocking/lu_nb_get.c         \
examples/benchmarks/lu/armci_nonblocking/timing.c

examples_benchmarks_lu_pthreads_lu_thread_SOURCES = \
examples/benchmarks/lu/pthreads/barrier.c           \
examples/benchmarks/lu/pthreads/barrier.h           \
examples/benchmarks/lu/pthreads/errors.h            \
examples/benchmarks/lu/pthreads/lu-thread.c         \
examples/benchmarks/lu/pthreads/timing.c
examples_benchmarks_lu_pthreads_lu_thread_LDADD = -lpthread

examples_benchmarks_RandomAccess_simple_simple_SOURCES = \
examples/benchmarks/RandomAccess/simple/simple.c         \
examples/benchmarks/RandomAccess/timing.c

examples_benchmarks_RandomAccess_smp_bucket_smp_bucket_SOURCES = \
examples/benchmarks/RandomAccess/smp_bucket/smp_bucket.c         \
examples/benchmarks/RandomAccess/timing.c

examples_features_aggregation_simple_simple_SOURCES = \
examples/features/aggregation/simple/simple.c

examples_features_aggregation_sparse_matvecmul_sparse_matvecmul_SOURCES = \
examples/features/aggregation/sparse_matvecmul/sparse_matvecmul.c

examples_features_concurrency_multidma_multidma_SOURCES = \
examples/features/concurrency/multidma/multidma.c

examples_features_concurrency_simple_comdegree_SOURCES = \
examples/features/concurrency/simple/comdegree.c

examples_features_gpc_hashtable_test_hashtable_SOURCES = \
examples/features/gpc/hashtable/DistHashmap.cc           \
examples/features/gpc/hashtable/DistHashmap.h            \
examples/features/gpc/hashtable/GPCHashmap.cc            \
examples/features/gpc/hashtable/GPCHashmap.h             \
examples/features/gpc/hashtable/GPCHashmapHandler.cc     \
examples/features/gpc/hashtable/Hash_common.h            \
examples/features/gpc/hashtable/HashFunctions.cc         \
examples/features/gpc/hashtable/hash_map.h               \
examples/features/gpc/hashtable/HashUtil.cc              \
examples/features/gpc/hashtable/Util.h

examples_features_gpc_simple_gpctest_SOURCES = \
examples/features/gpc/simple/gpctest.c

examples_features_non_blocking_overlap_overlap_SOURCES = \
examples/features/non-blocking/overlap/overlap.c

examples_features_non_blocking_simple_simple_SOURCES = \
examples/features/non-blocking/simple/simple.c

examples_features_notification_simple_testnotify_SOURCES = \
examples/features/notification/simple/testnotify.c

examples_features_proc_affinity_computation_impact_computation_impact_SOURCES = \
examples/features/proc_affinity/computation_impact/computation_impact.c

examples_features_proc_affinity_simple_simple_SOURCES = \
examples/features/proc_affinity/simple/simple.c

examples_features_symmetric_memory_simple_simple_SOURCES = \
examples/features/symmetric_memory/simple/simple.c

examples_simpleApps_transp1D_c_SOURCES = \
examples/simpleApps/transp1D-c.c

nodist_EXTRA_examples_benchmarks_cg_armci_sharedmemory_cg_SOURCES                          = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_examples_benchmarks_lu_armci_blocking_lu_SOURCES                              = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_examples_benchmarks_lu_armci_blocking_lu_block_SOURCES                        = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_examples_benchmarks_lu_armci_blocking_lu_b_bc_SOURCES                         = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_examples_benchmarks_lu_armci_multithreaded_lu_block_th_SOURCES                = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_examples_benchmarks_lu_armci_nonblocking_lu_nb_put_SOURCES                    = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_examples_benchmarks_lu_armci_nonblocking_lu_nb_get_SOURCES                    = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_examples_benchmarks_lu_pthreads_lu_thread_SOURCES                             = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_examples_benchmarks_RandomAccess_simple_simple_SOURCES                        = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_examples_benchmarks_RandomAccess_smp_bucket_smp_bucket_SOURCES                = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_examples_features_aggregation_simple_simple_SOURCES                           = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_examples_features_aggregation_sparse_matvecmul_sparse_matvecmul_SOURCES       = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_examples_features_concurrency_multidma_multidma_SOURCES                       = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_examples_features_concurrency_simple_comdegree_SOURCES                        = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_examples_features_gpc_hashtable_test_hashtable_SOURCES                        = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_examples_features_gpc_simple_gpctest_SOURCES                                  = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_examples_features_non_blocking_overlap_overlap_SOURCES                        = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_examples_features_non_blocking_simple_simple_SOURCES                          = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_examples_features_notification_simple_testnotify_SOURCES                      = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_examples_features_proc_affinity_computation_impact_computation_impact_SOURCES = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_examples_features_proc_affinity_simple_simple_SOURCES                         = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_examples_features_symmetric_memory_simple_simple_SOURCES                      = $(MAYBE_DUMMY_F_LINK)
nodist_EXTRA_examples_simpleApps_transp1D_c_SOURCES                                        = $(MAYBE_DUMMY_F_LINK)

##############################################################################
# the end
#
clean-local: clean-config clean-asm

.PHONY: checkprogs
checkprogs: $(check_PROGRAMS)

# automake does not append $(EXEEXT) to user-defined make variables
# so we must duplicate all examples here to get this custom target... yuck
.PHONY: examples
examples: $(EXAMPLES)
EXAMPLES =
EXAMPLES += examples/benchmarks/cg/armci_sharedmemory/cg$(EXEEXT)
EXAMPLES += examples/benchmarks/lu/armci_blocking/lu-b-bc$(EXEEXT)
EXAMPLES += examples/benchmarks/lu/armci_blocking/lu-block$(EXEEXT)
EXAMPLES += examples/benchmarks/lu/armci_blocking/lu$(EXEEXT)
EXAMPLES += examples/benchmarks/lu/armci_nonblocking/lu_nb_get$(EXEEXT)
EXAMPLES += examples/benchmarks/lu/armci_nonblocking/lu_nb_put$(EXEEXT)
EXAMPLES += examples/benchmarks/lu/pthreads/lu-thread$(EXEEXT)
EXAMPLES += examples/features/aggregation/simple/simple$(EXEEXT)
EXAMPLES += examples/features/aggregation/sparse_matvecmul/sparse_matvecmul$(EXEEXT)
EXAMPLES += examples/features/concurrency/simple/comdegree$(EXEEXT)
EXAMPLES += examples/features/non-blocking/simple/simple$(EXEEXT)
EXAMPLES += examples/features/notification/simple/testnotify$(EXEEXT)
EXAMPLES += examples/features/symmetric_memory/simple/simple$(EXEEXT)
EXAMPLES += examples/simpleApps/transp1D-c$(EXEEXT)
if ARMCI_ENABLE_GPC_CALLS
EXAMPLES += examples/features/gpc/hashtable/test_hashtable$(EXEEXT)
EXAMPLES += examples/features/gpc/simple/gpctest$(EXEEXT)
EXAMPLES += examples/features/proc_affinity/computation_impact/computation_impact$(EXEEXT)
EXAMPLES += examples/features/proc_affinity/simple/simple$(EXEEXT)
endif
if MSG_COMMS_MPI
EXAMPLES += examples/features/concurrency/multidma/multidma$(EXEEXT)
EXAMPLES += examples/features/non-blocking/overlap/overlap$(EXEEXT)
endif
if RANDOM_ACCESS
EXAMPLES += examples/benchmarks/RandomAccess/simple/simple$(EXEEXT)
EXAMPLES += examples/benchmarks/RandomAccess/smp_bucket/smp_bucket$(EXEEXT)
endif
if THREAD_SAFE
EXAMPLES += examples/benchmarks/lu/armci_multithreaded/lu-block-th$(EXEEXT)
endif

##############################################################################
# test suite
#
# some tests were commented out either because they required an input file
# (some examples do) or they were failing for unknown reasons and we didn't
# want to further debug
#
TESTS = $(SERIAL_TESTS) $(PARALLEL_TESTS)
XFAIL_TESTS = $(SERIAL_TESTS_XFAIL) $(PARALLEL_TESTS_XFAIL)
TESTS_ENVIRONMENT = \
maybe_mpi=`if echo "$(SERIAL_TESTS)" | $(GREP) "$$p" > /dev/null; then echo ""; else echo "$(MPIEXEC)"; fi`; $$maybe_mpi
SERIAL_TESTS =
SERIAL_TESTS_XFAIL =
PARALLEL_TESTS =
PARALLEL_TESTS_XFAIL =
if ARMCI_NETWORK_PORTALS
PARALLEL_TESTS += testing-portals/clone$(EXEEXT)
PARALLEL_TESTS += testing-portals/fork$(EXEEXT)
PARALLEL_TESTS += testing-portals/msgcheck$(EXEEXT)
PARALLEL_TESTS += testing-portals/origptl$(EXEEXT)
PARALLEL_TESTS += testing-portals/perf_aggr$(EXEEXT)
#PARALLEL_TESTS += testing-portals/perf_nb$(EXEEXT) # needs bug fixes
PARALLEL_TESTS += testing-portals/perf$(EXEEXT)
PARALLEL_TESTS += testing-portals/ptltest$(EXEEXT)
#SERIAL_TESTS += testing-portals/shmclean$(EXEEXT) # not sure we need this
#SERIAL_TESTS += testing-portals/shmtest$(EXEEXT) # not sure we need this
PARALLEL_TESTS += testing-portals/simplelock$(EXEEXT)
PARALLEL_TESTS += testing-portals/simple$(EXEEXT)
PARALLEL_TESTS += testing-portals/test2$(EXEEXT)
PARALLEL_TESTS += testing-portals/testnotify$(EXEEXT)
PARALLEL_TESTS += testing-portals/test$(EXEEXT)
PARALLEL_TESTS += testing-portals/te$(EXEEXT)
if ARMCI_ENABLE_GPC_CALLS
PARALLEL_TESTS += testing-portals/gpctest$(EXEEXT)
endif
if MSG_COMMS_MPI
#PARALLEL_TESTS += testing-portals/ipctest$(EXEEXT) # needs bug fixes
PARALLEL_TESTS += testing-portals/test_groups$(EXEEXT)
endif
if THREAD_SAFE
PARALLEL_TESTS += testing-portals/test_mt$(EXEEXT)
endif
else # !ARMCI_NETWORK_PORTALS
PARALLEL_TESTS += testing/msgcheck$(EXEEXT)
PARALLEL_TESTS += testing/perf_aggr$(EXEEXT)
#PARALLEL_TESTS += testing/perf_nb$(EXEEXT) # needs bug fixes
PARALLEL_TESTS += testing/perf$(EXEEXT)
#SERIAL_TESTS += testing/shmclean$(EXEEXT) # not sure we need this
#SERIAL_TESTS += testing/shmtest$(EXEEXT) # not sure we need this
PARALLEL_TESTS += testing/test2$(EXEEXT)
PARALLEL_TESTS += testing/testitr$(EXEEXT)
PARALLEL_TESTS += testing/testnotify$(EXEEXT)
PARALLEL_TESTS += testing/test$(EXEEXT)
if ENABLE_CHECKPOINT
PARALLEL_TESTS += testing/fttest$(EXEEXT)
endif
if ARMCI_ENABLE_GPC_CALLS
PARALLEL_TESTS += testing/gpctest$(EXEEXT)
endif
if MSG_COMMS_MPI
PARALLEL_TESTS += testing/test_groups$(EXEEXT)
if SYSV
#PARALLEL_TESTS += testing/ipctest$(EXEEXT) # needs bug fixes
endif
endif
if THREAD_SAFE
PARALLEL_TESTS += testing/test_mt$(EXEEXT)
endif
if NOFORT
else
SERIAL_TESTS += testing/acc_test$(EXEEXT)
SERIAL_TESTS += testing/copy_test$(EXEEXT)
endif
endif # ARMCI_NETWORK_PORTALS
#PARALLEL_TESTS += examples/benchmarks/cg/armci_sharedmemory/cg$(EXEEXT) # needs input file
PARALLEL_TESTS += examples/benchmarks/lu/armci_blocking/lu-b-bc$(EXEEXT)
PARALLEL_TESTS += examples/benchmarks/lu/armci_blocking/lu-block$(EXEEXT)
PARALLEL_TESTS += examples/benchmarks/lu/armci_blocking/lu$(EXEEXT)
PARALLEL_TESTS += examples/benchmarks/lu/armci_nonblocking/lu_nb_get$(EXEEXT)
PARALLEL_TESTS += examples/benchmarks/lu/armci_nonblocking/lu_nb_put$(EXEEXT)
PARALLEL_TESTS += examples/benchmarks/lu/pthreads/lu-thread$(EXEEXT)
PARALLEL_TESTS += examples/features/aggregation/simple/simple$(EXEEXT)
#PARALLEL_TESTS += examples/features/aggregation/sparse_matvecmul/sparse_matvecmul$(EXEEXT) # needs input file
PARALLEL_TESTS += examples/features/concurrency/simple/comdegree$(EXEEXT)
PARALLEL_TESTS += examples/features/non-blocking/simple/simple$(EXEEXT)
PARALLEL_TESTS += examples/features/notification/simple/testnotify$(EXEEXT)
PARALLEL_TESTS += examples/features/symmetric_memory/simple/simple$(EXEEXT)
PARALLEL_TESTS += examples/simpleApps/transp1D-c$(EXEEXT)
if ARMCI_ENABLE_GPC_CALLS
PARALLEL_TESTS += examples/features/gpc/hashtable/test_hashtable$(EXEEXT)
PARALLEL_TESTS += examples/features/gpc/simple/gpctest$(EXEEXT)
PARALLEL_TESTS += examples/features/proc_affinity/computation_impact/computation_impact$(EXEEXT)
PARALLEL_TESTS += examples/features/proc_affinity/simple/simple$(EXEEXT)
endif
if MSG_COMMS_MPI
#PARALLEL_TESTS += examples/features/concurrency/multidma/multidma$(EXEEXT) # needs bug fixes
#PARALLEL_TESTS += examples/features/non-blocking/overlap/overlap$(EXEEXT) # needs bug fixes
endif
if RANDOM_ACCESS
PARALLEL_TESTS += examples/benchmarks/RandomAccess/simple/simple$(EXEEXT)
PARALLEL_TESTS += examples/benchmarks/RandomAccess/smp_bucket/smp_bucket$(EXEEXT)
endif
if THREAD_SAFE
PARALLEL_TESTS += examples/benchmarks/lu/armci_multithreaded/lu-block-th$(EXEEXT)
endif
