# make file for Shared Files library
#
# Jarek Nieplocha, Aug  1 1996
#

 LIB_DISTRIB = ../../lib
     LIBRARY = libchemio.a
 LIB_TARGETS = *.x *.p
         OBJ = sf_fortran.o shared.files.o
     HEADERS = sf.h sf.fh
LIB_INCLUDES = -I../elio

# select the communication library for Shared Files
     COM_LIB = -ltcgmsg
ifeq ($(SFCOM), GA)
 LIB_DEFINES = -DGA
     COM_LIB = -lglobal -ltcgmsg
endif
ifeq ($(SFCOM), MPI)
 LIB_DEFINES = -DMPI
     COM_LIB = -lmpi
endif

ifdef HPIODIR
 LIB_DEFINES += -DHPIODIR=\'$(HPIODIR)/\'
endif


include ../../config/makefile.h
include ../../config/makelib.h


 COPT    = -g
 FOPT    = -g
LOC_LIBS = -L$(LIBDIR) -lchemio -lglobal -lma $(COM_LIB) $(EXTRA_COMM_LIBS)

ifeq ($(TARGET), DECOSF)
  LOC_LIBS += -laio -lpthreads 
endif

# a rule to build excutables: the filetype of an executable must be .x
# for example, foo.x corresponds to foo.f or foo.F or foo.c
#
%.x : %.o $(LIBRARY_PATH)  $(EXTRA)
	if [ -f $(basename $@).c ]; then\
		$(CLD) $(CLDOPT) -o $@ $< $(LOC_LIBS) $(CLIBS);\
	else\
		$(FLD) $(FOPT_REN) $(FLDOPT) -o $@ $< $(LOC_LIBS);\
	fi

# *.p file is required for tcgmsg `parallel' command to run your program
%.p:
	echo "`whoami` `hostname` 4 `pwd`/$* ." > $@

$(patsubst %,$(LIBRARY_PATH)(%),shared.files.o): sf.h coms.h
$(patsubst %,$(LIBRARY_PATH)(%),fortran.o):      sf.h
