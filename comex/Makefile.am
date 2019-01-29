##############################################################################
# Makefile.am for COMEX.
#
# Rationale:
# This Makefile.am follows many of the suggestions outlined in the paper
# "Recursive Make Considered Harmful".
#
# Additional targets:
# Besides the traditional make targets supplied by Automake, we have added the
# "checkprogs" target to build test programs.
#

# The usual aclocal nonsense to get include paths right.
ACLOCAL_AMFLAGS = -I m4

# All public headers, installed programs, test programs, and example programs
# are listed in these variables.  Appended to throughout.  These are the
# automake variables used.
include_HEADERS =
bin_PROGRAMS =
bin_SCRIPTS =
check_PROGRAMS =
check_LTLIBRARIES =
lib_LTLIBRARIES =
noinst_LTLIBRARIES =
EXTRA_DIST = README
BUILT_SOURCES =
MOSTLYCLEANFILES =
CLEANFILES =
DISTCLEANFILES =
MAINTAINERCLEANFILES =
AM_CFLAGS =
AM_CPPFLAGS =
AM_LDFLAGS =
LDADD =

lib_LTLIBRARIES += libcomex.la
# later Makefile fragments append to this
libcomex_la_SOURCES =
libcomex_la_LIBADD =
libcomex_la_LIBADD += $(MPI_LIBS)
libcomex_la_LIBADD += $(COMEX_NETWORK_LIBS)
libcomex_la_LIBADD += $(BLAS_LIBS)
libcomex_la_LIBADD += $(LIBS)
if COMEX_NETWORK_OFI
libcomex_la_LIBADD += $(LIBADD_DLOPEN)
endif
include_HEADERS += $(top_srcdir)/src-common/comex.h

AM_CPPFLAGS += $(MPI_CPPFLAGS)
AM_CPPFLAGS += $(COMEX_NETWORK_CPPFLAGS)
AM_CPPFLAGS += $(BLAS_CPPFLAGS)
AM_CPPFLAGS += -I$(top_srcdir)/src-common

AM_LDFLAGS += $(MPI_LDFLAGS)
AM_LDFLAGS += $(COMEX_NETWORK_LDFLAGS)
AM_LDFLAGS += $(BLAS_LDFLAGS)

LDADD += libcomex.la

# internal libcomex for possibly rolling up into other libraries
noinst_LTLIBRARIES += libcomexi.la
libcomexi_la_SOURCES = $(libcomex_la_SOURCES)
libcomexi_la_LIBADD =
libcomexi_la_LIBADD += $(MPI_LIBS)
libcomexi_la_LIBADD += $(COMEX_NETWORK_LIBS)
libcomexi_la_LIBADD += $(BLAS_LIBS)
libcomexi_la_LIBADD += $(LIBS)
if COMEX_NETWORK_OFI
libcomexi_la_LIBADD += $(LIBADD_DLOPEN)
endif

##############################################################################
# compiler and linker flags
#
# Important for external tools wanting to know how to link to COMEX.
SED_NORMALIZE_WHITESPACE = $(SED) 's/ [ ]*/ /g;s/" /"/g;s/ "/"/g'
.PHONY: flags
flags:
	@echo ' =========================================================================== '
	@echo ' Suggested compiler/linker options are as follows.'
	@echo ' COMEX libraries are installed in $(libdir)' 
	@echo ' COMEX headers are installed in $(includedir)' 
	@echo ' '
	@echo ' CPPFLAGS="$(MPI_CPPFLAGS) $(COMEX_NETWORK_CPPFLAGS) $(BLAS_CPPFLAGS) -I$(includedir)"' | $(SED_NORMALIZE_WHITESPACE)
	@echo ' '
	@echo ' LDFLAGS="$(MPI_LDFLAGS) $(COMEX_NETWORK_LDFLAGS) $(BLAS_LDFLAGS) -L$(libdir)"' | $(SED_NORMALIZE_WHITESPACE)
	@echo ' '
	@echo ' For C/C++ Programs: '
	@echo ' LIBS="-lcomex $(MPI_LIBS) $(COMEX_NETWORK_LIBS) $(BLAS_LIBS) $(LIBS)"' | $(SED_NORMALIZE_WHITESPACE)
	@echo ' =========================================================================== '

bin_SCRIPTS += tools/armci-config
bin_SCRIPTS += tools/comex-config
CLEANFILES += $(bin_SCRIPTS)

##############################################################################
# src-portals4
#
if COMEX_NETWORK_PORTALS4
include $(top_srcdir)/src-portals4/Makefile.inc
endif

##############################################################################
# src-dmapp
#
if COMEX_NETWORK_DMAPP
include $(top_srcdir)/src-dmapp/Makefile.inc
endif

##############################################################################
# src-ofa
#
if COMEX_NETWORK_OFA
include $(top_srcdir)/src-ofa/Makefile.inc
endif

##############################################################################
# src-mpi
#
if COMEX_NETWORK_MPI_TS
include $(top_srcdir)/src-mpi/Makefile.inc
endif

##############################################################################
# src-mpi-mt
#
if COMEX_NETWORK_MPI_MT
include $(top_srcdir)/src-mpi-mt/Makefile.inc
endif

##############################################################################
# src-mpi-pt
#
if COMEX_NETWORK_MPI_PT
include $(top_srcdir)/src-mpi-pt/Makefile.inc
endif

##############################################################################
# src-mpi-pr
#
if COMEX_NETWORK_MPI_PR
include $(top_srcdir)/src-mpi-pr/Makefile.inc
endif

##############################################################################
# src-mpi3
#
if COMEX_NETWORK_MPI3
include $(top_srcdir)/src-mpi3/Makefile.inc
endif

##############################################################################
# src-ofi
#
if COMEX_NETWORK_OFI
include $(top_srcdir)/src-ofi/Makefile.inc
endif

##############################################################################
# src-armci
#
include $(top_srcdir)/src-armci/Makefile.inc

##############################################################################
# profiling
#
if ENABLE_PROFILING_ARMCI
AM_CPPFLAGS += -I$(top_srcdir)/src-armci
if HAVE_SYS_WEAK_ALIAS_PRAGMA
lib_LTLIBRARIES += libarmci_prof.la
libarmci_prof_la_SOURCES =
libarmci_prof_la_SOURCES += tools/armci_prof.c
libarmci_la_SOURCES += src-armci/capi.c
else # HAVE_SYS_WEAK_ALIAS_PRAGMA
libarmci_la_SOURCES += tools/armci_prof.c
endif # HAVE_SYS_WEAK_ALIAS_PRAGMA
else # ENABLE_PROFILING_ARMCI
libarmci_la_SOURCES += src-armci/capi.c
include_HEADERS += src-armci/parmci.h
endif

##############################################################################
# testing
#
check_PROGRAMS += testing/perf
check_PROGRAMS += testing/perf_amo
check_PROGRAMS += testing/perf_contig
check_PROGRAMS += testing/perf_strided
check_PROGRAMS += testing/shift
check_PROGRAMS += testing/test

COMEX_SERIAL_TESTS =
COMEX_SERIAL_TESTS_XFAIL =
COMEX_DUAL_TESTS =
COMEX_DUAL_TESTS_XFAIL =
COMEX_PARALLEL_TESTS =
COMEX_PARALLEL_TESTS_XFAIL =
COMEX_TESTS = $(COMEX_SERIAL_TESTS) $(COMEX_DUAL_TESTS) $(COMEX_PARALLEL_TESTS)
COMEX_TESTS_XFAIL = $(COMEX_SERIAL_TESTS_XFAIL) $(COMEX_DUAL_TESTS_XFAIL) $(COMEX_PARALLEL_TESTS_XFAIL)

COMEX_DUAL_TESTS += testing/perf$(EXEEXT)
COMEX_DUAL_TESTS += testing/perf_contig$(EXEEXT)
COMEX_DUAL_TESTS += testing/perf_strided$(EXEEXT)
COMEX_PARALLEL_TESTS += testing/perf_amo$(EXEEXT)
COMEX_PARALLEL_TESTS += testing/shift$(EXEEXT)
COMEX_PARALLEL_TESTS += testing/test$(EXEEXT)

testing_perf_SOURCES         = testing/perf.c
testing_perf_amo_SOURCES     = testing/perf_amo.c
testing_perf_contig_SOURCES  = testing/perf_contig.c
testing_perf_strided_SOURCES = testing/perf_strided.c
testing_shift_SOURCES        = testing/shift.c
testing_test_SOURCES         = testing/test.c

##############################################################################
# the end
#
.PHONY: checkprogs
checkprogs: $(check_PROGRAMS)

# support verbose/silent make rules for additional programs
# sed
SED_V = $(SED__v_$(V))
SED__v_ = $(SED__v_$(AM_DEFAULT_VERBOSITY))
SED__v_0 = @echo "  SED   " $@;

##############################################################################
# test suite
#
# Some tests were commented out either because they required an input file or
# they were failing for unknown reasons and we didn't want to further debug.
#
SERIAL_TESTS =
SERIAL_TESTS += $(COMEX_SERIAL_TESTS)

SERIAL_TESTS_XFAIL =
SERIAL_TESTS_XFAIL += $(COMEX_SERIAL_TESTS_XFAIL)

DUAL_TESTS =
DUAL_TESTS += $(COMEX_DUAL_TESTS)

DUAL_TESTS_XFAIL =
DUAL_TESTS_XFAIL += $(COMEX_DUAL_TESTS_XFAIL)

PARALLEL_TESTS =
PARALLEL_TESTS += $(COMEX_PARALLEL_TESTS)

PARALLEL_TESTS_XFAIL =
PARALLEL_TESTS_XFAIL += $(COMEX_PARALLEL_TESTS_XFAIL)

TESTS =
TESTS += $(SERIAL_TESTS)
TESTS += $(DUAL_TESTS)
TESTS += $(PARALLEL_TESTS)

XFAIL_TESTS =
XFAIL_TESTS += $(SERIAL_TESTS_XFAIL)
XFAIL_TESTS += $(DUAL_TESTS_XFAIL)
XFAIL_TESTS += $(PARALLEL_TESTS_XFAIL)

if CROSS_COMPILING
maybe_mpiexec=`if echo "$(SERIAL_TESTS)" | $(GREP) "$$p" > /dev/null; then echo "$(MPIEXEC)" | $(SED) 's/%NP%/1/'; elif echo "$(DUAL_TESTS)" | $(GREP) "$$p" > /dev/null; then echo "$(MPIEXEC)" | $(SED) 's/%NP%/2/'; else echo "$(MPIEXEC)" | $(SED) 's/%NP%/$(NPROCS)/'; fi`; eval $$maybe_mpiexec
else
if COMEX_NETWORK_MPI_PR
NPROCS1=$(shell expr $(NPROCS) + 1)
LOG_COMPILER = \
maybe_mpiexec=`if echo "$(SERIAL_TESTS)" | $(GREP) "$$p" > /dev/null; then echo ""; elif echo "$(DUAL_TESTS)" | $(GREP) "$$p" > /dev/null; then echo "$(MPIEXEC)" | $(SED) 's/%NP%/3/'; elif echo "$(MPIEXEC)" | $(GREP) "%NP%" > /dev/null; then echo "$(MPIEXEC)" | $(SED) 's/%NP%/$(NPROCS1)/'; else echo "$(MPIEXEC)"; fi`; eval $$maybe_mpiexec
else
LOG_COMPILER = \
maybe_mpiexec=`if echo "$(SERIAL_TESTS)" | $(GREP) "$$p" > /dev/null; then echo ""; elif echo "$(DUAL_TESTS)" | $(GREP) "$$p" > /dev/null; then echo "$(MPIEXEC)" | $(SED) 's/%NP%/2/'; elif echo "$(MPIEXEC)" | $(GREP) "%NP%" > /dev/null; then echo "$(MPIEXEC)" | $(SED) 's/%NP%/$(NPROCS)/'; else echo "$(MPIEXEC)"; fi`; eval $$maybe_mpiexec
endif # COMEX_NETWORK_MPI_PR
endif # CROSS_COMPILING
