#$Id: makefile,v 1.2 1996-07-17 15:58:12 mogill Exp $
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


OBJ = eaf.o eaf_c2f.o
LIBRARY = libeaf.a
HEADERS = eaf.h eaf.fh eaf_c2f.h
LIB_DEFINES = $(F_AFFIX) $(F_CASE)


include ../../config/makefile.h
include ../../config/makelib.h


$(OBJ) : $(HEADERS)

