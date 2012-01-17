libarmci_la_SOURCES += src-gemini/acc.h
libarmci_la_SOURCES += src-gemini/aggregate.c
libarmci_la_SOURCES += src-gemini/armci.c
libarmci_la_SOURCES += src-gemini/armci-onesided.c
libarmci_la_SOURCES += src-gemini/armci-onesided.h
libarmci_la_SOURCES += src-gemini/armcip.h
libarmci_la_SOURCES += src-gemini/atomics-i386.h
libarmci_la_SOURCES += src-gemini/buffers.c
libarmci_la_SOURCES += src-gemini/caccumulate.c
libarmci_la_SOURCES += src-gemini/ccopy.c
libarmci_la_SOURCES += src-gemini/clusterinfo.c
libarmci_la_SOURCES += src-gemini/copy.h
libarmci_la_SOURCES += src-gemini/ds-shared.c
libarmci_la_SOURCES += src-gemini/fence.c
libarmci_la_SOURCES += src-gemini/groups.c
libarmci_la_SOURCES += src-gemini/kr_malloc.c
libarmci_la_SOURCES += src-gemini/kr_malloc.h
libarmci_la_SOURCES += src-gemini/locks.c
libarmci_la_SOURCES += src-gemini/locks.h
libarmci_la_SOURCES += src-gemini/memlock.c
libarmci_la_SOURCES += src-gemini/memlock.h
libarmci_la_SOURCES += src-gemini/memory.c
libarmci_la_SOURCES += src-gemini/message.c
libarmci_la_SOURCES += src-gemini/mutex.c
libarmci_la_SOURCES += src-gemini/pack.c
libarmci_la_SOURCES += src-gemini/pendbufs.h
libarmci_la_SOURCES += src-gemini/request.c
libarmci_la_SOURCES += src-gemini/request.h
libarmci_la_SOURCES += src-gemini/rmw.c
libarmci_la_SOURCES += src-gemini/rtinfo.c
libarmci_la_SOURCES += src-gemini/semaphores.c
libarmci_la_SOURCES += src-gemini/semaphores.h
libarmci_la_SOURCES += src-gemini/shmalloc.h
libarmci_la_SOURCES += src-gemini/shmem.c
libarmci_la_SOURCES += src-gemini/shmem.h
libarmci_la_SOURCES += src-gemini/shmlimit.c
libarmci_la_SOURCES += src-gemini/shmlimit.h
libarmci_la_SOURCES += src-gemini/signaltrap.c
libarmci_la_SOURCES += src-gemini/signaltrap.h
libarmci_la_SOURCES += src-gemini/sockets.h
libarmci_la_SOURCES += src-gemini/spawn.c
libarmci_la_SOURCES += src-gemini/spinlock.h
libarmci_la_SOURCES += src-gemini/strided.c
libarmci_la_SOURCES += src-gemini/tas-alpha.h
libarmci_la_SOURCES += src-gemini/tas-i386.h
libarmci_la_SOURCES += src-gemini/tas-ia64.h
libarmci_la_SOURCES += src-gemini/tas-ppc.h
libarmci_la_SOURCES += src-gemini/utils.h
libarmci_la_SOURCES += src-gemini/vector.c
if ENABLE_F77
noinst_LTLIBRARIES += libfcopyfacc.la
libfcopyfacc_la_SOURCES =
libfcopyfacc_la_SOURCES += src/common/fcopy.f
libfcopyfacc_la_SOURCES += src/xfer/faccumulate.f
libfcopyfacc_la_FFLAGS =
libfcopyfacc_la_FFLAGS += $(ARMCI_FOPT)
libfcopyfacc_la_FFLAGS += $(FFLAG_INT_4)
libarmci_la_LIBADD += libfcopyfacc.la
libarmcii_la_LIBADD += libfcopyfacc.la
endif # ENABLE_F77
if ARMCI_ENABLE_GPC_CALLS
libarmci_la_SOURCES += src-gemini/gpc.c
endif
if THREAD_SAFE
libarmci_la_SOURCES += src-gemini/threads.c
libarmci_la_SOURCES += src-gemini/utils.c
endif
if INLINE_TAS_OKAY
libarmci_la_SOURCES += src-gemini/tas.c
else
if INLINE_TAS_WITH
nodist_libarmci_la_SOURCES += src-gemini/tas_as.s
endif
endif

include_HEADERS += src-gemini/armci.h
include_HEADERS += src-gemini/gpc.h
include_HEADERS += src-gemini/message.h

AM_CPPFLAGS += -I$(top_srcdir)/src-gemini
AM_CPPFLAGS += -I$(top_srcdir)/src/include
LDADD += -lnumatoolkit
LDADD += -lonesided

if ENABLE_SHARED
MAYBE_PIC = -fPIC -DPIC
endif

src-gemini/tas_as.s: src-gemini/tas.c
	$(ARMCI_TAS_AS_V) \
	$(ARMCI_TAS_AS) $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) \
	$(AM_CPPFLAGS) $(CPPFLAGS) -S $(MAYBE_PIC) -o $@ \
	$(top_srcdir)/src-gemini/tas.c

CLEANFILES += src-gemini/tas_as.s
