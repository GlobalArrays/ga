libcmx_la_SOURCES += src-mpi-pr/cmx.c
libcmx_la_SOURCES += src-mpi-pr/cmx_impl.h
libcmx_la_SOURCES += src-mpi-pr/groups.c
libcmx_la_SOURCES += src-mpi-pr/groups.h
libcmx_la_SOURCES += src-mpi-pr/reg_cache.c
libcmx_la_SOURCES += src-mpi-pr/reg_cache.h

AM_CPPFLAGS += -I$(top_srcdir)/src-mpi-pr

check_PROGRAMS += src-mpi-pr/hello
src_mpi_pr_hello_SOURCES = src-mpi-pr/hello.c
