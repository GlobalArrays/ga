  LIB_DISTRIB = ../../lib
     LIBRARY = libchemio.a
         OBJ = disk.arrays.o disk.param.o patch.util.o fortran.o
     HEADERS = dra.fh
LIB_INCLUDES = -I../elio


# Uncomment the line below to enable the implementation based on a
# collection of independent files. Also set HPIODIR=/tmp on the SP.
#LIB_DEFINES  += -DINDEPFILES
ifeq ($(TARGET),CRAY-T3E)
  LOC_DEFINES  = -DINDEPFILES
endif

include ../makefile.h
include ../../config/makefile.h
include ../../config/makelib.h

$(patsubst %,$(LIBRARY_PATH)(%),disk.arrays.o): drap.h dra.h
$(patsubst %,$(LIBRARY_PATH)(%),disk.param.o) : drap.h
