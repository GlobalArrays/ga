libcomex_la_SOURCES += src-mpi-prwin/comex.c
libcomex_la_SOURCES += src-mpi-prwin/comex_impl.h
libcomex_la_SOURCES += src-mpi-prwin/groups.c
libcomex_la_SOURCES += src-mpi-prwin/groups.h
libcomex_la_SOURCES += src-mpi-prwin/reg_cache.c
libcomex_la_SOURCES += src-mpi-prwin/reg_cache.h

AM_CPPFLAGS += -I$(top_srcdir)/src-mpi-prwin

check_PROGRAMS += src-mpi-prwin/hello
src_mpi_pr_hello_SOURCES = src-mpi-prwin/hello.c
