include ../Makefile.h

LIBM = -lma
INCLUDE = ../../ma/src
OBJ =  usc.o ga_trace.o

# TARGET to DEFINE translation 
#
DEFINE = $(TARGET)
ifeq ($(TARGET),IPSC)
     DEFINE = IPSC860_NODE
endif
ifeq ($(TARGET),IBM)
     DEFINE = IBM_RS6000
endif
ifeq ($(TARGET),SP1)
     DEFINE = IBM_RS6000
endif

LIB_DISTRIB = ../../lib
LIBRARY = libtrace.a
LIB_TARGETS = test adjust collisions

include ../../config/makelib.h

.c.o:   $*.c
	$(CC) $(COPT) -I$(INCLUDE) -D$(DEFINE) -c $*.c

.f.o:   $*.c
	$(F77) $(FOPT)  -c  $*.f

test:   $(OBJ)
	$(F77) -o test $(OBJ) $(LIB) -L$(LIBDIR) $(LIBM)

