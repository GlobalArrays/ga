#
# Makefile for a portable dynamic memory allocator.

      LIBRARY = libma.a
  LIB_DISTRIB = ../lib

 OBJ_OPTIMIZE = f2c.o ma.o maf.o table.o
          OBJ = string-util.o error.o
      HEADERS = macommon.h mafdecls.h matypes.h macdecls.h \
                scope.h error.h memory.h \
                string-util.h table.h mafdecls.fh

  LIB_TARGETS = testf.o testc.x
  LIB_DEFINES = -DVERIFY -DSTATS

ifeq ($(TARGET),SGITFP)
    LIB_CDEFS = -DLongInteger
endif

ifeq ($(TARGET),KSR)
    LIB_CDEFS = -DLongInteger
endif

ifeq ($(TARGET),DECOSF)
    LIB_CDEFS = -DLongInteger
endif



include ../config/makefile.h
include ../config/makelib.h

testf.x:  testf.o $(LIBRARY_PATH)
	$(FC) $(FFLAGS) -o $@ $^

testc.x:  testc.o $(LIBRARY_PATH)
	$(CC) $(CFLAGS) -o $@ $^

$(LIBRARY_PATH)(error.o):        ./scope.h ./error.h
$(LIBRARY_PATH)(f2c.o):          ./scope.h ./ma.h
$(LIBRARY_PATH)(ma.o):           ./table.h ./scope.h ./memory.h ./ma.h ./error.h
$(LIBRARY_PATH)(maf.o):          ./mafdecls.fh ./maf2c.h
$(LIBRARY_PATH)(string-util.o):  ./string-util.h
$(LIBRARY_PATH)(table.o):        ./table.h ./scope.h ./memory.h ./error.h
$(LIBRARY_PATH)(testc.o):        ./string-util.h ./macdecls.h
$(LIBRARY_PATH)(testf.o):        ./mafdecls.fh

