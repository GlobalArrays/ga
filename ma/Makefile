# $Id: Makefile,v 1.5 1998-01-30 18:49:29 d3h325 Exp $
#
# Makefile for a portable dynamic memory allocator.

      LIBRARY = libma.a
  LIB_DISTRIB = ../lib
 OBJ_OPTIMIZE = error.o f2c.o ma.o maf.o table.o
      HEADERS = macommon.h matypes.h macdecls.h \
                scope.h error.h memcpy.h \
                string-util.h table.h mafdecls.fh

  LIB_TARGETS = testf.o testf.x testc.o testc.x string-util.o
  LIB_DEFINES = -DVERIFY -DSTATS

include ../config/makefile.h
include ../config/makelib.h

testf.x:  testf.o $(LIBRARY_PATH)
	$(FC) -o $@ $(FFLAGS) $^

testc.x: testc.o string-util.o $(LIBRARY_PATH)
	$(CC) $(CFLAGS) -o $@ $^

test-coalesce.x: test-coalesce.o $(LIBRARY_PATH)
	$(CC) $(CFLAGS) -o $@ $^
