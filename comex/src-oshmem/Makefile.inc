# Makefile.inc for comex src-oshmem backend
# This file is intended to be included by the top-level build system when enabling the oshmem backend.

libcomex_srcs = \
    comex.c \
    groups.c \
    nb.c \
    locks.c

AM_CPPFLAGS += -I$(top_srcdir)/comex/src-oshmem

COMEX_OSHMEM_LIBS = -lshmem

# Note: Integrator must add this backend to the build (Makefile.am/CMakeLists.txt) and add linking to the SHMEM library.
