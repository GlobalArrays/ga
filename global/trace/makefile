include ../Makefile.h

LIBM = ../../ma/libma.a
INCLUDE = ../../ma
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

LIB = libtrace.a

$(LIB):  $(OBJ)
	$(RM) -f $@
	$(AR) $(ARFLAGS) $@ $(OBJ)
	$(RANLIB) $@

.c.o:   $*.c
	$(CC) $(COPT) -I$(INCLUDE) -D$(DEFINE) -c $*.c

.f.o:   $*.c
	$(F77) $(FOPT)  -c  $*.f

test:   $(OBJ)
	$(F77) -o test $(OBJ) $(LIB) $(LIBM)
clean:
	$(RM) -rf *.o  *.a test adjust collisions 

