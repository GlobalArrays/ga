OBJ =  usc.o ga_trace.o
LIB_DISTRIB = ../../lib
LIBRARY = libtrace.a
LIB_TARGETS = test adjust collisions
LIBM = -L$(LIB_DISTRIB)/$(TARGET) -lma

# TARGET to DEFINE translation
#
ifeq ($(TARGET),IPSC)
     LIB_DEFINES = IPSC860_NODE
endif
LIB_INCLUDES = -I../../ma

include ../../config/makefile.h
include ../../config/makelib.h

test.x:   test.o $(LIBRARY_PATH)
	$(FC) -o  $@ $(FFLAGS) $^ $(LIBM)
