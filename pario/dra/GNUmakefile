###############################################################################
#                    Disk Resident Array library GNU makefile
#
# You need GA and associated libs already built and TARGET defined.
# To generate library, type "make".
# To build your test program foo.x type "make foo.x" where foo.[c,f,F] is source
# program.
#
# For tcgmsg platforms that require "parallel" to run a program you need
# to create explicitly .p file, for example "make test.x.p". Then
# run the program as "parallel test.x"
#
# HPIODIR should define path to high-performance storage system, for example
#         setenv HPIODIR /pfs
#         make test.x
#
###############################################################################

 LIB_DISTRIB = ../../lib
     LIBRARY = libchemio.a
         OBJ = disk.arrays.o disk.param.o patch.util.o fortran.o env.o
 LIB_TARGETS = *.x *.p
     HEADERS = drap.h dra.h dra.fh
      INCDIR = ../../include
LIB_INCLUDES = -I$(INCDIR)

include ../makefile.h
include ../../config/makefile.h
include ../../config/makelib.h

COPT    = -g
FOPT    = -g  

# a rule to build excutables: the filetype of an executable must be .x
# for example, foo.x corresponds to either foo.f or foo.F or foo.c
#
%.x : %.o $(LIBRARY_PATH)  $(EXTRA)
	if [ -f $(basename $@).c ]; then\
		$(LINK.c) $(CLDOPT) -o $@ $< $(LIBS) $(CLIBS);\
	else\
		$(LINK.f) $(FOPT_REN) $(FLDOPT) -o $@ $< $(LIBS);\
	fi

# *.p file is required for tcgmsg `parallel' command to run your program
%.p:
	echo "`whoami` `hostname` 4 `pwd`/$* ." > $@

$(patsubst %,$(LIBRARY_PATH)(%),disk.arrays.o): drap.h dra.h 
$(patsubst %,$(LIBRARY_PATH)(%),disk.param.o) : drap.h
