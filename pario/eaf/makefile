#$Id: makefile,v 1.4 1996-09-17 22:12:21 d3h325 Exp $
#
# Nakefile for EAF of ChemIO 
#
#   Written by Jace A Mogill
#              12 June 1996
#	       Argonne National Laboratory
#
#
# . .  . .  . .  . .  . .  . .  . .  . .  . .  . .  . . 
# Fortran Affixation
#    Note: Use of both or neither is legal
#       AIX     post
#       T3D     none
#       PARAGON post
#       SGITFP  post
F_AFFIX :=
#F_AFFIX += -DEAF_FC_PREFIX=_
F_AFFIX += -DEAF_FC_POSTFIX=_

# . .  . .  . .  . .  . .  . .  . .  . .  . .  . .  . . 
# Fortran Case Translation
#    Use null F_CASE for mixed case
#
#       AIX     lower
#       T3D     upper
#       PARAGON lower
#       SGITFP  lower
F_CASE = -DEAF_FC_LOWERCASE
#F_CASE = -DEAF_FC_UPPERCASE
#F_CASE =

#points to location where library should be built
LIB_DISTRIB = ../../lib

OBJ = eaf.o eaf_c2f.o
LIBRARY = libeaf.a
HEADERS = eaf.h eaf.fh eaf_c2f.h
LIB_TARGETS = *.x

LIB_DEFINES = $(F_AFFIX) $(F_CASE)
LIB_INCLUDES= -I../elio

include ../../config/makefile.h
include ../../config/makelib.h

#aftest_eaf.F   atest_eaf.c    ftest_eaf.F    test_eaf.c

LIBS = $(LIB_DISTRIB)/$(TARGET)/libeaf.a $(LIB_DISTRIB)/$(TARGET)/libelio.a

%.x : %.o $(LIBS)
	$(FC) $(FFLAGS) -o $@ $^

ftest : ftest_eaf.o $(LIBS)

$(patsubst %,$(LIBRARY_PATH)(%),eaf.o): eaf.h
$(patsubst %,$(LIBRARY_PATH)(%),eaf_c2f.o): eaf.h eaf_c2f.h
