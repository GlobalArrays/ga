# $Id: GNUmakefile,v 1.5 2004-06-03 16:53:50 vinod Exp $
#
# Makefile for a portable dynamic memory allocator.

      LIBRARY = libma.a
  LIB_DISTRIB = ../lib
          OBJ = error.o f2c.o ma.o maf.o table.o
      HEADERS = macommon.h matypes.h macdecls.h \
                scope.h error.h memcpy.h \
                string-util.h table.h mafdecls.fh\
                typesf2c.h winutil.h winf2c.h

  LIB_TARGETS = *.x *.o
  LIB_DEFINES = -DVERIFY -DSTATS
  INCDIR      = ../include

include ../config/makefile.h
include ../config/makelib.h

ifdef ARMCI_NETWORK
  CFLAGS += -DENABLE_USE_ARMCI_MEM_OPTION
endif

testf.x:  testf.o $(FULL_LIBRARY_PATH)
	$(FC) -o $@ $(FFLAGS) $^

testc.x: testc.o string-util.o $(FULL_LIBRARY_PATH)
	$(CC) $(CFLAGS) -o $@ $^

test-coalesce.x: test-coalesce.o $(FULL_LIBRARY_PATH)
	$(CC) $(CFLAGS) -o $@ $^

test-inquire.x: test-inquire.o $(FULL_LIBRARY_PATH)
	$(CC) $(CFLAGS) -o $@ $^
