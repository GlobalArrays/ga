libcomex_la_SOURCES += src-mpi-pt/comex.c
libcomex_la_SOURCES += src-mpi-pt/comex_impl.h
libcomex_la_SOURCES += src-mpi-pt/groups.c
libcomex_la_SOURCES += src-mpi-pt/groups.h
libcomex_la_SOURCES += src-mpi-pt/reg_cache.c
libcomex_la_SOURCES += src-mpi-pt/reg_cache.h

AM_CPPFLAGS += -I$(top_srcdir)/src-mpi-pt

check_PROGRAMS += src-mpi-pt/hello
src_mpi_pt_hello_SOURCES = src-mpi-pt/hello.c
