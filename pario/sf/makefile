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


include ../makefile.h
include ../../config/makefile.h
include ../../config/makelib.h

COPT=-g
FOPT=-g
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

$(patsubst %,$(LIBRARY_PATH)(%),shared.files.o): sf.h coms.h
$(patsubst %,$(LIBRARY_PATH)(%),fortran.o):      sf.h
