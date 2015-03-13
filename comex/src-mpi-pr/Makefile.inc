libcomex_la_SOURCES += src-mpi-pr/comex.c
libcomex_la_SOURCES += src-mpi-pr/comex_impl.h
libcomex_la_SOURCES += src-mpi-pr/groups.c
libcomex_la_SOURCES += src-mpi-pr/groups.h
libcomex_la_SOURCES += src-mpi-pr/reg_cache.c
libcomex_la_SOURCES += src-mpi-pr/reg_cache.h

AM_CPPFLAGS += -I$(top_srcdir)/src-mpi-pr

check_PROGRAMS += src-mpi-pr/hello
src_mpi_pr_hello_SOURCES = src-mpi-pr/hello.c
