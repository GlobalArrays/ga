###############################################################################
#                    Disk Array library GNU makefile 
#
# You need GA and associated libs already built and TARGET defined.
# To generate library, type "make".
# To build your test program foo.x type "make foo.x" where foo.[c,f,F] is source
# To run program on (network of) workstations using more than one process,
# type "parallel foo.x". On other platforms, use whatever is required to
# run a program.
#
# HPIODIR should define path to high-performance storage system, for example
#         setenv HPIODIR /pfs 
#         make test.x
#
###############################################################################

ifndef TARGET
error:
	@echo "TARGET machine not defined "
	@echo "-- one of [SUN,SGI,SGITFP,IBM,DECOSF,SP1,SP,IPSC,DELTA,PARAGON,KSR,CRAY-T3D,CONVEX-SPP,HPUX]"
	exit
endif


LOC_INCLUDES = -I../../global/src

# Uncomment line below to enable implementation based on
# collection of independent files. Please set HPIODIR=/tmp on the SP.
#LOC_DEFINES  = -DINDEPFILES


include ../../global/Makefile.h
include ../../global/Makelib.h
INCLUDES = -I../elio -I../../ma -I../../global/src

COPT    = -g
FOPT    = -g  

ifdef HPIODIR
      FDEFS += -DHPIODIR=\'$(HPIODIR)/\'
endif

ifeq ($(TARGET),DECOSF)
   DA_EXTRA_LIBS = -laio -lpthreads
endif
 

# object files for the library and test program(s)
#
OBJ_LIB = disk.arrays.o disk.param.o patch.util.o fortran.o
OBJ_TST = global.unsup.o
OBJ     = $(OBJ_LIB) $(OBJ_TST)

DA_LIBS = -L$(LIBDIR) -ldra -lchemio $(CORE_LIBS) $(LINALG) $(BLAS) $(LIBCOM) $(DA_EXTRA_LIBS)


LIB_DISTRIB = ../../lib
LIB_TARGETS = *.x *.p
LIBRARY = libdra.a

include ../../config/makelib.h

#
# this "thing" below builds test programs and generates .p files if needed:
# a test program, say, foo.x corresponds to foo.f or foo.F or foo.c;
# the name of executable must have extension .x
#
#.PRECIOUS: %.o
ifeq ($(P_FILE),YES)
    ifeq ($(EXPLICITF),FALSE)
        .SUFFIXES : 
        .SUFFIXES : .F .o .x .c

        %.o : %.F
	   $(FC) $(FFLAGS) -c $*.F 
        %.o : %.c
	    $(CC) $(CFLAGS) -c $*.c 
    endif

    %.x : %.o $(LIBDIR)/$(LIBRARY) $(EXTRA)
	if [ -f $(basename $@).c ]; then\
		$(CC) $(CLDOPT) -o $@ $< $(DA_LIBS) $(CLIBS);\
	else\
		$(FC) $(FLDOPT) -o $@ $< $(DA_LIBS);\
	fi
	echo "`whoami` `hostname` 4 `pwd`/$@ ." > $@.p
else
    %.x : %.o $(LIBDIR)/$(LIBRARY)  $(EXTRA)
	if [ -f $(basename $@).c ]; then\
		$(CC) $(CLDOPT) -o $@ $< $(DA_LIBS) $(CLIBS);\
	else\
		$(FC) $(FLDOPT) -o $@ $< $(DA_LIBS);\
	fi
endif


$(patsubst %,$(LIBRARY_PATH)(%),disk.arrays.o): drap.h dra.h 
$(patsubst %,$(LIBRARY_PATH)(%),disk.param.o) : drap.h
