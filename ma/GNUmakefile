# $Id: GNUmakefile,v 1.3 2002-09-14 05:40:29 d3g001 Exp $
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

testf.x:  testf.o $(LIBRARY_PATH)
	$(FC) -o $@ $(FFLAGS) $^

testc.x: testc.o string-util.o $(LIBRARY_PATH)
	$(CC) $(CFLAGS) -o $@ $^

test-coalesce.x: test-coalesce.o $(LIBRARY_PATH)
	$(CC) $(CFLAGS) -o $@ $^

test-inquire.x: test-inquire.o $(LIBRARY_PATH)
	$(CC) $(CFLAGS) -o $@ $^
