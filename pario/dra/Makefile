  LIB_DISTRIB = ../../lib
     LIBRARY = libchemio.a
         OBJ = disk.arrays.o disk.param.o patch.util.o fortran.o env.o
     HEADERS = dra.fh
LIB_INCLUDES = -I../elio

include ../makefile.h
include ../../config/makefile.h
include ../../config/makelib.h

$(patsubst %,$(LIBRARY_PATH)(%),disk.arrays.o): drap.h dra.h
$(patsubst %,$(LIBRARY_PATH)(%),disk.param.o) : drap.h
