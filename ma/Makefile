# $Id: Makefile,v 1.3 1996-10-23 05:48:50 d3g681 Exp $
#
# Makefile for a portable dynamic memory allocator.

      LIBRARY = libma.a
  LIB_DISTRIB = ../lib
 OBJ_OPTIMIZE = error.o f2c.o ma.o maf.o table.o
      HEADERS = macommon.h matypes.h macdecls.h \
                scope.h error.h memory.h \
                string-util.h table.h mafdecls.fh

  LIB_TARGETS = testf.o testf.x testc.o testc.x
  LIB_DEFINES = -DVERIFY -DSTATS

include ../config/makefile.h
include ../config/makelib.h

testf.x:  testf.o $(LIBRARY_PATH)
	$(FC) -o $@ $(FFLAGS) $^

testc.x:  testc.o $(LIBRARY_PATH)
	$(CC) $(CFLAGS) -o $@ $^

