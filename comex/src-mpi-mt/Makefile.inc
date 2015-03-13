libcomex_la_SOURCES += src-mpi-mt/comex.c
libcomex_la_SOURCES += src-mpi-mt/comex_impl.h
libcomex_la_SOURCES += src-mpi-mt/groups.c
libcomex_la_SOURCES += src-mpi-mt/groups.h

AM_CPPFLAGS += -I$(top_srcdir)/src-mpi-mt
