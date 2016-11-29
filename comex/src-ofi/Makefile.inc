libcomex_la_SOURCES += src-ofi/comex.c
libcomex_la_SOURCES += src-ofi/comex_impl.h
libcomex_la_SOURCES += src-ofi/datatype.h
libcomex_la_SOURCES += src-ofi/groups.c
libcomex_la_SOURCES += src-ofi/groups.h
libcomex_la_SOURCES += src-ofi/mutex.h
libcomex_la_SOURCES += src-ofi/ofi.h
libcomex_la_SOURCES += src-ofi/request.h

AM_CPPFLAGS += -I$(top_srcdir)/src-ofi
