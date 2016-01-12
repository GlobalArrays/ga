libcomex_la_SOURCES += src-mpi3/comex.c
libcomex_la_SOURCES += src-mpi3/comex_impl.h
libcomex_la_SOURCES += src-mpi3/groups.c
libcomex_la_SOURCES += src-mpi3/groups.h
libcomex_la_SOURCES += src-mpi3/reg_win.c
libcomex_la_SOURCES += src-mpi3/reg_win.h

AM_CPPFLAGS += -I$(top_srcdir)/src-mpi3
