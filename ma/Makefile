# $Id: Makefile,v 1.2 1996-10-09 18:00:20 d3h325 Exp $
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

# DO NOT EDIT BENEATH THIS LINE ... GENERATED AUTOMATICALLY
$(LIBRARY_PATH)(testc.o):	./string-util.h 
$(LIBRARY_PATH)(table.o):	./table.h ./scope.h ./memory.h ./error.h 
$(LIBRARY_PATH)(string-util.o):	./string-util.h 
$(LIBRARY_PATH)(maf.o):	./mafdecls.fh 
$(LIBRARY_PATH)(ma.o):	./table.h ./scope.h ./memory.h ./ma.h 
$(LIBRARY_PATH)(f2c.o):	./scope.h 
$(LIBRARY_PATH)(error.o):	./scope.h ./error.h 
